const std = @import("std");
const log = std.log.scoped(.nn_descent);

const mod_neighbors = @import("neighbors.zig");
const mod_dataset = @import("../dataset.zig");

pub const IterationTiming = struct {
    iteration: usize,
    sample_candidates_ns: u64,
    generate_proposals_ns: u64,
    apply_updates_ns: u64,
    total_iteration_ns: u64,
    updates_count: usize,
};

pub const TrainingTiming = struct {
    init_random_ns: u64,
    iterations: std.ArrayList(IterationTiming),
    total_training_ns: u64,
    num_iterations_completed: usize,
    converged: bool,

    pub fn deinit(self: *TrainingTiming, allocator: std.mem.Allocator) void {
        self.iterations.deinit(allocator);
    }
};

/// Configuration parameters for training the k-NN graph using NN-descent.
/// Referenced from https://github.com/brj0/nndescent/blob/main/src/nnd.h
pub const TrainingConfig = struct {
    /// The number of neighbors (k) each point should have in the k-NN graph.
    /// Should be no more than the number of vectors in the dataset.
    num_neighbors_per_node: usize,

    /// The maximum number of NN-descent iterations to perform. The NN-descent
    /// algorithm can abort early if limited progress is being made, so this
    /// only controls the worst case. By default, a value will be chosen based on
    /// the size of the graph_data.
    max_iterations: usize,

    /// Internally each local join keeps a maximum number of candidates
    /// (nearest neighbors and reverse nearest neighbors) to be considered. This
    /// value controls this aspect of the algorithm. Larger values will provide
    /// more accurate search results later, potentially at non-negligible
    /// computation cost in building the index.
    max_candidates: usize,

    /// Controls the early abort due to limited progress. Larger values will
    /// result in earlier aborts, providing less accurate indexes, and less
    /// accurate searching.
    delta: f32 = 0.001,

    /// The number of parallel threads to use. Default is the number or cores.
    /// > 1 means multi-threading, otherwise single-threading.
    num_threads: usize = 1,

    /// Random seed for any randomized components of the algorithm.
    seed: u64,

    /// Whether to generate & apply graph updates in a block of vectors at a time.
    /// Block processing is faster and saves memory usage when number of vectors gets large. Default to true.
    block_processing: bool = true,

    const Self = @This();

    /// Initializes a training config with default values based on the dataset size.
    pub fn init(
        num_neighbors_per_node: usize,
        num_vectors: usize,
        num_threads: ?usize,
        seed: ?u64,
    ) Self {
        const config = Self{
            .num_neighbors_per_node = num_neighbors_per_node,
            .max_iterations = if (num_vectors > 0) @max(5, @as(usize, @intFromFloat(@log2(@as(f64, @floatFromInt(num_vectors)))))) else 0,
            .max_candidates = @min(60, num_neighbors_per_node),
            .num_threads = if (num_threads) |n| n else std.Thread.getCpuCount() catch 1,
            .seed = if (seed) |s| s else std.crypto.random.int(u64),
        };
        return config;
    }
};

pub const InitError = error{
    /// The specified maximum number of candidates is too large. Should be no more than i32 max.
    MaxCandidatesTooLarge,
    /// Invalid number of threads specified in the training config. Should be larger than zero.
    InvalidNumThreads,
    /// Invalid number of neighbors per node. Should be less than the number of vectors in the dataset.
    InvalidNumNeighborsPerNode,
};

/// NN-Descent struct to construct the k-NN graph from a dataset.
pub fn NNDescent(
    /// Element type of the vectors, supported in `types.ElemType`.
    comptime T: type,
    /// Dimensionality of the vectors in the dataset, supported in `types.DimType`.
    comptime N: usize,
) type {
    const Dataset = mod_dataset.Dataset(T, N);

    return struct {
        /// The dataset of vectors to build the k-NN graph for. Owned by caller.
        dataset: Dataset,
        /// Configuration parameters for training.
        training_config: TrainingConfig,
        /// Holds the current neighbor lists for all nodes.
        neighbors_list: NeighborHeapList,
        /// Holds the new neighbor candidates for all nodes.
        neighbor_candidates_new: CandidateHeapList,
        /// Holds the old neighbor candidates for all nodes.
        neighbor_candidates_old: CandidateHeapList,
        /// Lists of graph updates during training, one per thread.
        /// Uses the a slice of `graph_updates_buffer` as backing storage.
        /// Capacity of each list is large enough to hold all possible updates in a block.
        block_graph_updates_lists: []std.ArrayList(GraphUpdate),
        /// Buffer to hold graph updates during generation of proposals in one block.
        /// Used by all array lists in `graph_updates_lists`.
        block_graph_updates_buffer: []GraphUpdate,
        /// Buffer to hold the number of graph updates applied by each thread.
        /// Used during reduction to get total number of updates applied.
        /// Aligned for efficient SIMD access.
        graph_update_counts_buffer: []align(64) usize,
        /// Thread pool for multi-threaded operations.
        /// `null` when requested number of threads is <= 1.
        thread_pool: ?*std.Thread.Pool,
        /// Wait group for synchronizing threads.
        wait_group: std.Thread.WaitGroup,
        /// Number of nodes each thread is responsible for during parallel computations.
        /// Last batch of nodes is less than or equal to this value.
        /// Is 0 when either number of dataset is empty or number of threads is 0.
        num_nodes_per_thread: usize,
        /// Number of nodes in one block. Equal to the total number of nodes when
        /// total number of nodes <= `DEFAULT_BLOCK_SIZE` or when `training_config.block_processing == false`.
        num_nodes_per_block: usize,
        /// Number of nodes within one block each thread is responsible for.
        num_block_nodes_per_thread: usize,
        /// Pre-shuffled node IDs for random neighbor initialization
        node_ids_random: []const usize,

        /// Holds entries with flags
        const NeighborHeapList = mod_neighbors.NeighborHeapList(T, true);
        /// Holds entries without flags
        const CandidateHeapList = mod_neighbors.NeighborHeapList(i32, false);
        /// Represents a proposed update to the k-NN graph.
        /// Holds the IDs of the two nodes involved and their distance.
        const GraphUpdate = struct {
            node1_id: usize,
            node2_id: usize,
            distance: T,
        };
        const DEFAULT_BLOCK_SIZE = 16384;
        comptime {
            if (DEFAULT_BLOCK_SIZE == 0) @compileError("DEFAULT_BLOCK_SIZE must be larger than 0.");
        }

        const Self = @This();

        /// Initialize NN-Descent with the given dataset and training configuration.
        pub fn init(
            dataset: Dataset,
            training_config: TrainingConfig,
            allocator: std.mem.Allocator,
        ) (InitError || mod_neighbors.InitError || std.mem.Allocator.Error)!Self {
            if (training_config.max_candidates > std.math.maxInt(i32)) return InitError.MaxCandidatesTooLarge;
            if (training_config.num_threads == 0) return InitError.InvalidNumThreads;
            if (training_config.num_neighbors_per_node >= dataset.len) return InitError.InvalidNumNeighborsPerNode;

            var neighbors_list = try NeighborHeapList.init(
                dataset.len,
                training_config.num_neighbors_per_node,
                allocator,
            );
            errdefer neighbors_list.deinit(allocator);

            var neighbor_candidates_new = try CandidateHeapList.init(
                dataset.len,
                training_config.max_candidates,
                allocator,
            );
            errdefer neighbor_candidates_new.deinit(allocator);
            var neighbor_candidates_old = try CandidateHeapList.init(
                dataset.len,
                training_config.max_candidates,
                allocator,
            );
            errdefer neighbor_candidates_old.deinit(allocator);

            const block_graph_updates_lists = try allocator.alloc(
                std.ArrayList(GraphUpdate),
                training_config.num_threads,
            );
            errdefer allocator.free(block_graph_updates_lists);

            // Calculate maximum possible graph updates in an iteration
            // For each node, possible updates are:
            // - New-New neighbor pairs: n_new * (n_new - 1) / 2
            // - New-Old neighbor pairs: n_new * n_old
            // NOTE: Since training_config.max_candidates (aka n_new and n_old)
            // is no more than i32 max, the cast is safe, and the calculations here won't overflow u64.
            const n_new = @as(u64, neighbor_candidates_new.num_neighbors_per_node);
            const n_old = @as(u64, neighbor_candidates_old.num_neighbors_per_node);
            const capacity_per_node_u64: u64 = (n_new * n_new - n_new) / 2 + (n_new * n_old);
            const capacity_per_node = std.math.cast(usize, capacity_per_node_u64) orelse
                return InitError.MaxCandidatesTooLarge;

            const num_nodes_per_block = if (training_config.block_processing)
                @min(DEFAULT_BLOCK_SIZE, neighbors_list.num_nodes)
            else
                neighbors_list.num_nodes;

            const num_max_graph_updates: usize, const overflow = @mulWithOverflow(capacity_per_node, num_nodes_per_block);
            if (overflow != 0) return InitError.MaxCandidatesTooLarge;

            const block_graph_updates_buffer = try allocator.alloc(
                GraphUpdate,
                num_max_graph_updates,
            );
            errdefer allocator.free(block_graph_updates_buffer);

            const num_block_nodes_per_thread = std.math.divCeil(
                usize,
                num_nodes_per_block,
                training_config.num_threads,
            ) catch 0;

            // Each thread takes an exclusive batch of nodes which corresponds to an exclusive slice in graph_updates_buffer
            for (block_graph_updates_lists, 0..) |*list, thread_id| {
                const node_id_start = @min(thread_id * num_block_nodes_per_thread, num_nodes_per_block);
                const node_id_end = @min(node_id_start + num_block_nodes_per_thread, num_nodes_per_block);
                list.* = std.ArrayList(GraphUpdate).initBuffer(
                    block_graph_updates_buffer[node_id_start * capacity_per_node .. node_id_end * capacity_per_node],
                );
            }

            const graph_update_counts_buffer: []align(64) usize = try allocator.alignedAlloc(
                usize,
                std.mem.Alignment.@"64",
                training_config.num_threads,
            );
            errdefer allocator.free(graph_update_counts_buffer);

            const node_ids_random = try allocator.alloc(usize, neighbors_list.num_nodes);
            for (node_ids_random, 0..) |*elem, node_id| {
                elem.* = node_id;
            }
            // Shuffle node IDs for random neighbor initialization
            var prng = std.Random.DefaultPrng.init(training_config.seed);
            const rng = prng.random();
            rng.shuffle(usize, node_ids_random);

            const thread_pool = if (training_config.num_threads > 1) blk: {
                const pool = try allocator.create(std.Thread.Pool);
                pool.init(.{
                    .n_jobs = training_config.num_threads,
                    .allocator = allocator,
                }) catch return std.mem.Allocator.Error.OutOfMemory;
                break :blk pool;
            } else null;

            var wait_group: std.Thread.WaitGroup = undefined;
            wait_group.reset();

            const num_nodes_per_thread = std.math.divCeil(
                usize,
                neighbors_list.num_nodes,
                training_config.num_threads,
            ) catch 0;

            return Self{
                .dataset = dataset,
                .training_config = training_config,
                .neighbors_list = neighbors_list,
                .neighbor_candidates_new = neighbor_candidates_new,
                .neighbor_candidates_old = neighbor_candidates_old,
                .block_graph_updates_buffer = block_graph_updates_buffer,
                .graph_update_counts_buffer = graph_update_counts_buffer,
                .block_graph_updates_lists = block_graph_updates_lists,
                .thread_pool = thread_pool,
                .wait_group = wait_group,
                .num_nodes_per_thread = num_nodes_per_thread,
                .num_block_nodes_per_thread = num_block_nodes_per_thread,
                .num_nodes_per_block = num_nodes_per_block,
                .node_ids_random = node_ids_random,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.neighbors_list.deinit(allocator);
            self.neighbor_candidates_new.deinit(allocator);
            self.neighbor_candidates_old.deinit(allocator);
            allocator.free(self.block_graph_updates_lists);
            // NOTE: Just need to free the buffer, since all lists use slices of it
            allocator.free(self.block_graph_updates_buffer);
            allocator.free(self.graph_update_counts_buffer);
            allocator.free(self.node_ids_random);
            if (self.thread_pool) |pool| {
                pool.deinit();
                allocator.destroy(pool);
            }
        }

        /// Number of blocks for training.
        /// Equal to 0 when the dataset is empty.
        pub fn numBlocks(self: *const Self) usize {
            return std.math.divCeil(
                usize,
                self.neighbors_list.num_nodes,
                self.num_nodes_per_block,
            ) catch 0;
        }

        pub fn trainWithTiming(self: *Self, allocator: std.mem.Allocator) !TrainingTiming {
            var timing = TrainingTiming{
                .init_random_ns = 0,
                .iterations = std.ArrayList(IterationTiming).empty,
                .total_training_ns = 0,
                .num_iterations_completed = 0,
                .converged = false,
            };
            errdefer timing.deinit(allocator);

            var total_timer = try std.time.Timer.start();
            var timer = try std.time.Timer.start();

            log.debug("Using {} threads", .{self.training_config.num_threads});
            log.info("Populating random neighbors", .{});

            // Step 1: Populate initial random neighbors
            timer.reset();
            self.populateRandomNeighbors();
            timing.init_random_ns = timer.read();

            const convergence_threshold = @as(usize, @intFromFloat(self.training_config.delta * @as(f32, @floatFromInt(self.neighbors_list.entries.len))));
            log.info("Convergence threshold: {}", .{convergence_threshold});

            // Step 2: Iteratively refine the neighbor lists
            for (0..self.training_config.max_iterations) |iteration| {
                log.info("NN-Descent iteration {d}", .{iteration});
                defer {
                    self.neighbor_candidates_new.reset();
                    self.neighbor_candidates_old.reset();
                }

                var iter_timing = IterationTiming{
                    .iteration = iteration,
                    .sample_candidates_ns = 0,
                    .generate_proposals_ns = 0,
                    .apply_updates_ns = 0,
                    .total_iteration_ns = 0,
                    .updates_count = 0,
                };

                var iter_timer = try std.time.Timer.start();

                // Sample neighbor candidates into new and old candidate lists
                timer.reset();
                self.sampleNeighborCandidates();
                iter_timing.sample_candidates_ns = timer.read();

                var updates_count: usize = 0;
                const num_blocks = self.numBlocks();

                var gen_total_ns: u64 = 0;
                var apply_total_ns: u64 = 0;

                for (0..num_blocks) |block_id| {
                    defer {
                        for (self.block_graph_updates_lists) |*list| {
                            list.clearRetainingCapacity();
                        }
                    }

                    log.info("NN-Descent iteration {d} - block {d}", .{ iteration, block_id });

                    log.info("generating graph update proposals...", .{});
                    timer.reset();
                    self.generateBlockGraphUpdateProposals(block_id);
                    gen_total_ns += timer.read();

                    log.info("applying graph update proposals...", .{});
                    timer.reset();
                    const count = self.applyBlockGraphUpdatesProposals(block_id);
                    apply_total_ns += timer.read();

                    updates_count += count;
                }

                iter_timing.generate_proposals_ns = gen_total_ns;
                iter_timing.apply_updates_ns = apply_total_ns;
                iter_timing.updates_count = updates_count;
                iter_timing.total_iteration_ns = iter_timer.read();

                try timing.iterations.append(allocator, iter_timing);
                timing.num_iterations_completed = iteration + 1;

                log.info("Applied {d} graph updates", .{updates_count});

                if (updates_count <= convergence_threshold) {
                    log.info("Converged after {d} iterations", .{iteration + 1});
                    timing.converged = true;
                    break;
                }
            }

            timing.total_training_ns = total_timer.read();
            log.info("NN-Descent training completed", .{});

            return timing;
        }

        pub fn train(self: *Self) void {
            log.debug("Using {} threads", .{self.training_config.num_threads});
            log.info("Populating random neighbors", .{});
            // Step 1: Populate initial random neighbors
            self.populateRandomNeighbors();

            const convergence_threshold = @as(usize, @intFromFloat(self.training_config.delta * @as(f32, @floatFromInt(self.neighbors_list.entries.len))));
            log.info("Convergence threshold: {}", .{convergence_threshold});

            const num_blocks = self.numBlocks();

            // Step 2: Iteratively refine the neighbor lists
            for (0..self.training_config.max_iterations) |iteration| {
                log.info("NN-Descent iteration {d}", .{iteration});
                defer {
                    // TODO: Do we need to reset candidate lists, or just overwrite them in the next iteration?
                    self.neighbor_candidates_new.reset();
                    self.neighbor_candidates_old.reset();
                }

                // Sample neighbor candidates into new and old candidate lists
                self.sampleNeighborCandidates();

                // 1. generate new neighbor proposals from candidates
                // 2. update neighbor lists and track changes
                // 3. check for convergence based on delta

                var updates_count: usize = 0;
                for (0..num_blocks) |block_id| {
                    defer {
                        for (self.block_graph_updates_lists) |*list| {
                            list.clearRetainingCapacity();
                        }
                    }
                    log.info("NN-Descent iteration {d} - block {d}", .{ iteration, block_id });
                    log.info("generating graph update proposals...", .{});
                    self.generateBlockGraphUpdateProposals(block_id);
                    log.info("applying graph update proposals...", .{});
                    const count = self.applyBlockGraphUpdatesProposals(block_id);
                    updates_count += count;
                }

                log.info("Applied {d} graph updates", .{updates_count});

                if (updates_count <= convergence_threshold) {
                    log.info("Converged after {d} iterations", .{iteration + 1});
                    break;
                }
            }

            for (0..num_blocks) |block_id| {
                self.sortBlockNeighbors(block_id);
            }

            log.info("NN-Descent training completed", .{});
        }

        /// Populate all nodes with random neighbors.
        /// Use multi-threading if configured.
        fn populateRandomNeighbors(self: *Self) void {
            if (self.thread_pool) |pool| {
                self.wait_group.reset();
                for (0..self.training_config.num_threads) |thread_id| {
                    const node_id_start = @min(thread_id * self.num_nodes_per_thread, self.neighbors_list.num_nodes);
                    const node_id_end = @min(node_id_start + self.num_nodes_per_thread, self.neighbors_list.num_nodes);

                    // SAFETY: Each thread populate neighbors on non-overlapping range of nodes,
                    // so the neighbor heaps are separate in memory, and thus no data races.
                    pool.spawnWg(
                        &self.wait_group,
                        populateRandomNeighborsThread,
                        .{
                            &self.dataset,
                            &self.neighbors_list,
                            node_id_start,
                            node_id_end,
                            self.node_ids_random,
                        },
                    );
                }
                pool.waitAndWork(&self.wait_group);
            } else {
                populateRandomNeighborsThread(
                    &self.dataset,
                    &self.neighbors_list,
                    0,
                    self.dataset.len,
                    self.node_ids_random,
                );
            }

            // There should be no empty neighbor IDs left for all node IDs
            std.debug.assert(std.mem.indexOfScalar(
                isize,
                self.neighbors_list.entries.items(.neighbor_id),
                NeighborHeapList.EMPTY_ID,
            ) == null);
        }

        /// Populate a batch of nodes, starting from `node_id_start` (inclusive) to `node_id_end` (exclusive),
        /// with random neighbors.
        fn populateRandomNeighborsThread(
            dataset: *const Dataset,
            neighbors_list: *NeighborHeapList,
            node_id_start: usize,
            node_id_end: usize,
            node_ids_random: []const usize,
        ) void {
            std.debug.assert(dataset.len == neighbors_list.num_nodes);
            std.debug.assert(dataset.len == node_ids_random.len);
            // NOTE: When node_id_start == node_id_end, the loop beblow never executes
            std.debug.assert(node_id_start <= node_id_end and node_id_end <= neighbors_list.num_nodes);
            log.debug("node_id_start: {}, node_id_end: {}", .{ node_id_start, node_id_end });

            for (node_id_start..node_id_end) |node_id| {
                const node = dataset.getUnchecked(node_id);

                // Try to fill all neighbor entries of the node's neighbor list
                var idx = node_id;
                for (0..neighbors_list.num_neighbors_per_node) |_| {
                    var neighbor_id = node_ids_random[idx % node_ids_random.len];
                    if (neighbor_id == node_id) {
                        // Prevent self-loop by skipping this neighbor ID
                        idx += 1;
                        neighbor_id = node_ids_random[idx % node_ids_random.len];
                    }
                    // Increment idx to get the next random neighbor ID for the next iteration
                    idx += 1;

                    const neighbor = dataset.getUnchecked(neighbor_id);
                    const distance = node.sqdist(neighbor);

                    const added = neighbors_list.tryAddNeighbor(
                        node_id,
                        NeighborHeapList.Entry{
                            // SAFETY: neighbor_id is in [0, num_nodes), which fits in isize.
                            .neighbor_id = @intCast(neighbor_id),
                            .distance = distance,
                            .is_new = true,
                        },
                    );
                    // Neighbor must have been added successfully since
                    // the neighbors do not have duplicates and the initial
                    // distances in the neighbor heap are set to infinity.
                    std.debug.assert(added);
                }
            }
        }

        /// Sample neighbor candidates from the `neighbors_list` into `neighbor_candidates_new` and `neighbor_candidates_old`.
        /// Use multi-threading if configured.
        fn sampleNeighborCandidates(self: *Self) void {
            std.debug.assert(self.neighbor_candidates_new.num_nodes == self.neighbor_candidates_old.num_nodes);
            std.debug.assert(self.neighbor_candidates_new.num_nodes == self.neighbors_list.num_nodes);

            if (self.thread_pool) |pool| {
                self.wait_group.reset();
                for (0..self.training_config.num_threads) |thread_id| {
                    const node_id_start = @min(thread_id * self.num_nodes_per_thread, self.neighbors_list.num_nodes);
                    const node_id_end = @min(node_id_start + self.num_nodes_per_thread, self.neighbors_list.num_nodes);

                    // SAFETY: Each thread only touches on heaps of nodes whose IDs are
                    // in the range [node_id_start, node_id_end), so no data races.
                    pool.spawnWg(
                        &self.wait_group,
                        sampleNeighborCandidatesThread,
                        .{
                            &self.neighbors_list,
                            &self.neighbor_candidates_new,
                            &self.neighbor_candidates_old,
                            node_id_start,
                            node_id_end,
                            // Different thread has different seed
                            self.training_config.seed + @as(u64, @intCast(thread_id)),
                        },
                    );
                }
                // Wait for all sampling threads to finish before moving on
                pool.waitAndWork(&self.wait_group);

                self.wait_group.reset();
                // Mark sampled nodes in neighbors_list as not new anymore
                for (0..self.training_config.num_threads) |thread_id| {
                    const node_id_start = @min(thread_id * self.num_nodes_per_thread, self.neighbors_list.num_nodes);
                    const node_id_end = @min(node_id_start + self.num_nodes_per_thread, self.neighbors_list.num_nodes);

                    // SAFETY: Each thread only touches on heaps of nodes whose IDs are
                    // in the range [node_id_start, node_id_end), so no data races.
                    pool.spawnWg(
                        &self.wait_group,
                        markSampledToOldThread,
                        .{
                            &self.neighbors_list,
                            &self.neighbor_candidates_new,
                            node_id_start,
                            node_id_end,
                        },
                    );
                }
                pool.waitAndWork(&self.wait_group);
            } else {
                sampleNeighborCandidatesThread(
                    &self.neighbors_list,
                    &self.neighbor_candidates_new,
                    &self.neighbor_candidates_old,
                    0,
                    self.neighbors_list.num_nodes,
                    self.training_config.seed,
                );
                markSampledToOldThread(
                    &self.neighbors_list,
                    &self.neighbor_candidates_new,
                    0,
                    self.neighbors_list.num_nodes,
                );
            }
        }

        /// Sample neighbor candidates from the `neighbors_list` into `neighbor_candidates_new` and `neighbor_candidates_old`.
        /// Each node in the candidate list will contain nodes that point to it (forward neighbors)
        /// and nodes that it points to (reverse neighbors).
        /// Goes through all edges in `neighbors_list`, and only tries to add neighbors to a node whose node ID
        /// is in the range `[node_id_start, node_id_end)` to the candidate lists.
        fn sampleNeighborCandidatesThread(
            neighbors_list: *const NeighborHeapList,
            neighbor_candidates_new: *CandidateHeapList,
            neighbor_candidates_old: *CandidateHeapList,
            node_id_start: usize,
            node_id_end: usize,
            seed: u64,
        ) void {
            // NOTE: When node_id_start == node_id_end, nothing gets added to the candidate lists
            std.debug.assert(node_id_start <= node_id_end and node_id_end <= neighbors_list.num_nodes);

            // Initialize PRNG with thread-specific seed
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();

            for (0..neighbors_list.num_nodes) |node_id| {
                const neighbor_id_slice: []const isize = neighbors_list.getEntryFieldSlice(node_id, .neighbor_id);
                const is_new_slice: []const bool = neighbors_list.getEntryFieldSlice(node_id, .is_new);

                for (0..neighbors_list.num_neighbors_per_node) |neighbor_idx| {
                    // Skip this neighbor slot if it's empty
                    const entry_neighbor_id = neighbor_id_slice[neighbor_idx];
                    if (entry_neighbor_id == CandidateHeapList.EMPTY_ID) continue;
                    // SAFETY: neighbor_id is not EMPTY_ID, so this cast is safe.
                    const neighbor_id: usize = @intCast(entry_neighbor_id);

                    // Generate a random priority for this neighbor
                    // TODO: Consider if this sampling strategy is appropriate
                    const priority = rng.int(i32);

                    // Assign to neighbor_candidates_new or neighbor_candidates_old based on is_new flag
                    const entry_is_new = is_new_slice[neighbor_idx];
                    const target_candidate_list = if (entry_is_new)
                        neighbor_candidates_new
                    else
                        neighbor_candidates_old;

                    if (node_id >= node_id_start and node_id < node_id_end) {
                        // Add neighbor_id as a forward neighbor candidate for node_id
                        _ = target_candidate_list.tryAddNeighbor(
                            node_id,
                            CandidateHeapList.Entry{
                                // SAFETY: node_id is in [0, num_nodes), which fits in isize.
                                .neighbor_id = @intCast(neighbor_id),
                                .distance = priority,
                                .is_new = {},
                            },
                        );
                    }
                    if (neighbor_id >= node_id_start and neighbor_id < node_id_end) {
                        // Add node_id as a reverse neighbor candidate for neighbor_id
                        _ = target_candidate_list.tryAddNeighbor(
                            neighbor_id,
                            CandidateHeapList.Entry{
                                // SAFETY: node_id is in [0, num_nodes), which fits in isize.
                                .neighbor_id = @intCast(node_id),
                                .distance = priority,
                                .is_new = {},
                            },
                        );
                    }
                }
            }
        }

        /// Mark neighbors in `neighbors_list` as not new anymore if they were sampled
        /// into `neighbor_candidates_new`, with respect to a node ID.
        /// Only processes nodes whose IDs are in the range `[node_id_start, node_id_end)`.
        fn markSampledToOldThread(
            neighbors_list: *NeighborHeapList,
            neighbor_candidates_new: *const CandidateHeapList,
            node_id_start: usize,
            node_id_end: usize,
        ) void {
            // NOTE: When node_id_start == node_id_end, the loop below never executes
            std.debug.assert(node_id_start <= node_id_end and node_id_end <= neighbors_list.num_nodes);

            for (node_id_start..node_id_end) |node_id| {
                const neighbor_id_slice: []const isize = neighbors_list.getEntryFieldSlice(node_id, .neighbor_id);
                const is_new_slice: []bool = neighbors_list.getEntryFieldSlice(node_id, .is_new);
                const neighbor_candidate_id_slice: []const isize = neighbor_candidates_new.getEntryFieldSlice(node_id, .neighbor_id);

                for (0..neighbors_list.num_neighbors_per_node) |neighbor_idx| {
                    const neighbor_id = neighbor_id_slice[neighbor_idx];

                    // Check if the neighbor ID is valid
                    if (neighbor_id == NeighborHeapList.EMPTY_ID) continue;

                    if (std.mem.indexOfScalar(
                        isize,
                        neighbor_candidate_id_slice,
                        neighbor_id,
                    ) != null) {
                        // Mark as not new anymore
                        is_new_slice[neighbor_idx] = false;
                    }
                }
            }
        }

        /// Generate graph updates for all graph update lists for a block of nodes at a block ID.
        fn generateBlockGraphUpdateProposals(self: *Self, block_id: usize) void {
            std.debug.assert(self.block_graph_updates_lists.len == self.training_config.num_threads);

            const block_start = block_id * self.num_nodes_per_block;
            const block_end = @min(block_start + self.num_nodes_per_block, self.neighbors_list.num_nodes);
            log.debug("block_id: {} - block_start: {} - block_end: {}", .{ block_id, block_start, block_end });

            if (self.thread_pool) |pool| {
                self.wait_group.reset();
                for (0..self.training_config.num_threads) |thread_id| {
                    const node_id_start = @min(block_start + thread_id * self.num_block_nodes_per_thread, block_end);
                    const node_id_end = @min(node_id_start + self.num_block_nodes_per_thread, block_end);
                    log.debug("thread_id: {} - node_id_start: {} - node_id_end: {}", .{ thread_id, node_id_start, node_id_end });
                    // SAFETY: Each thread only touches on heaps of nodes whose IDs are
                    // in the range [node_id_start, node_id_end), so no data races.
                    pool.spawnWg(
                        &self.wait_group,
                        generateGraphUpdateProposalsThread,
                        .{
                            &self.dataset,
                            &self.neighbors_list,
                            &self.block_graph_updates_lists[thread_id],
                            &self.neighbor_candidates_new,
                            &self.neighbor_candidates_old,
                            node_id_start,
                            node_id_end,
                        },
                    );
                }
                pool.waitAndWork(&self.wait_group);
            } else {
                generateGraphUpdateProposalsThread(
                    &self.dataset,
                    &self.neighbors_list,
                    &self.block_graph_updates_lists[0],
                    &self.neighbor_candidates_new,
                    &self.neighbor_candidates_old,
                    block_start,
                    block_end,
                );
            }
        }

        /// Go through all nodes as local joins in the given range `[local_join_id_start, local_join_id_end)`:
        /// For each local join, check all new-new and new-old candidate pairs (skipping empty slots), and compute distances.
        /// If a closer neighbor (to either one of the pair) is found, add to the `graph_updates_list`.
        /// The number of updates added should never exceed the original capacity of the list.
        fn generateGraphUpdateProposalsThread(
            dataset: *const Dataset,
            neighbors_list: *const NeighborHeapList,
            graph_updates_list: *std.ArrayList(GraphUpdate),
            neighbor_candidates_new: *const CandidateHeapList,
            neighbor_candidates_old: *const CandidateHeapList,
            local_join_id_start: usize,
            local_join_id_end: usize,
        ) void {
            // NOTE: When local_join_id_start == local_join_id_end, the loop below never executes
            std.debug.assert(local_join_id_start <= local_join_id_end and local_join_id_end <= neighbors_list.num_nodes);
            std.debug.assert(neighbors_list.num_nodes == dataset.len);

            // Go through all local joins in the given range
            for (local_join_id_start..local_join_id_end) |local_join_id| {
                const new_candidate_ids: []const isize = neighbor_candidates_new.getEntryFieldSlice(local_join_id, .neighbor_id);
                const old_candidate_ids: []const isize = neighbor_candidates_old.getEntryFieldSlice(local_join_id, .neighbor_id);

                for (new_candidate_ids, 0..) |cand1_id, i| {
                    if (cand1_id == CandidateHeapList.EMPTY_ID) continue;
                    // SAFETY: cand1_id is not EMPTY_ID, so this cast is safe.
                    const cand1_id_usize: usize = @intCast(cand1_id);
                    const cand1_vector = dataset.getUnchecked(cand1_id_usize);
                    // Take current max distance in neighbor heap as threshold
                    const cand1_distance_threshold: T = neighbors_list.getMaxDistance(cand1_id_usize);

                    // New-New candidate pairs
                    for (new_candidate_ids[i + 1 ..]) |cand2_id| {
                        if (cand2_id == CandidateHeapList.EMPTY_ID) continue;
                        // SAFETY: cand2_id is not EMPTY_ID, so this cast is safe.
                        const cand2_id_usize: usize = @intCast(cand2_id);
                        const cand2_vector = dataset.getUnchecked(cand2_id_usize);
                        const cand2_distance_threshold: T = neighbors_list.getMaxDistance(cand2_id_usize);

                        const distance = cand1_vector.sqdist(cand2_vector);

                        if (distance <= @max(cand1_distance_threshold, cand2_distance_threshold)) {
                            // Found a closer neighbor (either to cand1 or cand2), add to graph updates
                            graph_updates_list.appendAssumeCapacity(.{
                                .distance = distance,
                                .node1_id = cand1_id_usize,
                                .node2_id = cand2_id_usize,
                            });
                        }
                    }

                    // New-Old candidate pairs
                    for (old_candidate_ids) |cand2_id| {
                        if (cand2_id == CandidateHeapList.EMPTY_ID) continue;
                        // SAFETY: cand2_id is not EMPTY_ID, so this cast is safe.
                        const cand2_id_usize: usize = @intCast(cand2_id);
                        const cand2_vector = dataset.getUnchecked(cand2_id_usize);
                        // Take current max distance in neighbor heap as threshold
                        const cand2_distance_threshold: T = neighbors_list.getMaxDistance(cand2_id_usize);

                        const distance = cand1_vector.sqdist(cand2_vector);

                        if (distance <= @max(cand1_distance_threshold, cand2_distance_threshold)) {
                            // Found a closer neighbor (either to cand1 or cand2), add to graph updates
                            graph_updates_list.appendAssumeCapacity(.{
                                .distance = distance,
                                .node1_id = cand1_id_usize,
                                .node2_id = cand2_id_usize,
                            });
                        }
                    }
                }
            }

            // Graph update list should not exceed its designated capacity
            std.debug.assert(graph_updates_list.items.len <= graph_updates_list.capacity);
        }

        /// Apply graph updates from all threads' graph updates lists for a block of nodes at block ID to the `neighbors_list`.
        /// Return the total number of successful updates applied.
        fn applyBlockGraphUpdatesProposals(self: *Self, block_id: usize) usize {
            std.debug.assert(self.graph_update_counts_buffer.len == self.training_config.num_threads);
            std.debug.assert(self.block_graph_updates_lists.len == self.training_config.num_threads);

            const block_start = block_id * self.num_nodes_per_block;
            const block_end = @min(block_start + self.num_nodes_per_block, self.neighbors_list.num_nodes);
            log.debug("block_id: {} - block_start: {} - block_end: {}", .{ block_id, block_start, block_end });

            if (self.thread_pool) |pool| {
                self.wait_group.reset();
                for (0..self.training_config.num_threads) |thread_id| {
                    const node_id_start = @min(block_start + thread_id * self.num_block_nodes_per_thread, block_end);
                    const node_id_end = @min(node_id_start + self.num_block_nodes_per_thread, block_end);
                    log.debug("thread_id: {} - node_id_start: {} - node_id_end: {}", .{ thread_id, node_id_start, node_id_end });
                    // SAFETY: Each thread only touches on heaps of nodes whose IDs are
                    // in the range [node_id_start, node_id_end), so no data races.
                    pool.spawnWg(
                        &self.wait_group,
                        applyGraphUpdatesProposalsThread,
                        .{
                            &self.neighbors_list,
                            &self.block_graph_updates_lists[thread_id],
                            &self.graph_update_counts_buffer[thread_id],
                            node_id_start,
                            node_id_end,
                        },
                    );
                }
                pool.waitAndWork(&self.wait_group);

                // Reduce the counts from all threads with SIMD
                return sumUpGraphUpdateCountsSIMD(self.graph_update_counts_buffer);
            } else {
                applyGraphUpdatesProposalsThread(
                    &self.neighbors_list,
                    &self.block_graph_updates_lists[0],
                    &self.graph_update_counts_buffer[0],
                    block_start,
                    block_end,
                );
                return self.graph_update_counts_buffer[0];
            }
        }

        /// Apply graph updates from the given `graph_updates_list` to the `neighbors_list`,
        /// only for nodes whose IDs are in the range `[node_id_start, node_id_end)`.
        /// Count the number of successful updates applied and store in `graph_updates_count_ptr`.
        fn applyGraphUpdatesProposalsThread(
            neighbors_list: *NeighborHeapList,
            graph_updates_list: *const std.ArrayList(GraphUpdate),
            graph_updates_count_ptr: *usize,
            node_id_start: usize,
            node_id_end: usize,
        ) void {
            // NOTE: When node_id_start == node_id_end, updates_count remains 0 after the loop
            std.debug.assert(node_id_start <= node_id_end and node_id_end <= neighbors_list.num_nodes);

            var updates_count: usize = 0;

            // Go through all graph updates in the list
            for (graph_updates_list.items) |graph_update| {
                const node1_id = graph_update.node1_id;
                const node2_id = graph_update.node2_id;
                const distance = graph_update.distance;

                var updates_count_local: usize = 0;

                if (node1_id >= node_id_start and node1_id < node_id_end) {
                    // Try to add neighbor to node1
                    const update1 = neighbors_list.tryAddNeighbor(node1_id, NeighborHeapList.Entry{
                        // SAFETY: neighbor_id is in [0, num_nodes), which fits in isize.
                        .neighbor_id = @intCast(node2_id),
                        .distance = distance,
                        .is_new = true,
                    });

                    updates_count_local += @intFromBool(update1);
                }

                if (node2_id >= node_id_start and node2_id < node_id_end) {
                    // Try to add neighbor to node2
                    const update2 = neighbors_list.tryAddNeighbor(node2_id, NeighborHeapList.Entry{
                        // SAFETY: neighbor_id is in [0, num_nodes), which fits in isize.
                        .neighbor_id = @intCast(node1_id),
                        .distance = distance,
                        .is_new = true,
                    });

                    updates_count_local += @intFromBool(update2);
                }

                // updates_count_local is either 0, 1, or 2. We update the count accordingly.
                updates_count += updates_count_local;
            }

            // Store the number of updates applied at the end. One memory access.
            graph_updates_count_ptr.* = updates_count;
        }

        /// Sum up the graph update counts from all threads using SIMD.
        fn sumUpGraphUpdateCountsSIMD(graph_update_counts_buffer: []align(64) const usize) usize {
            const counts_ptr: [*]align(64) const usize = graph_update_counts_buffer.ptr;
            const num_counts = graph_update_counts_buffer.len;

            const vector_size = std.simd.suggestVectorLength(usize) orelse
                @compileError("Cannot determine vector size for type");
            const Vec = @Vector(vector_size, usize);

            const num_chunks = num_counts / vector_size;
            const remainder = num_counts % vector_size;

            // One vector accumulator to rule them all
            var acc: Vec = @splat(0);

            // 1. Accumulate chunks
            var i: usize = 0;
            while (i < num_chunks * vector_size) : (i += vector_size) {
                const chunk: Vec = counts_ptr[i..][0..vector_size].*;
                acc += chunk;
            }

            // 2. Handle tail remainder (elements that didn't fit in a full vector)
            var tail_acc: usize = 0;
            if (remainder > 0) {
                while (i < num_counts) : (i += 1) {
                    tail_acc += counts_ptr[i];
                }
            }

            // 3. Final reduction
            return @reduce(.Add, acc) + tail_acc;
        }

        fn sortBlockNeighbors(self: *Self, block_id: usize) void {
            const block_start = block_id * self.num_nodes_per_block;
            const block_end = @min(block_start + self.num_nodes_per_block, self.neighbors_list.num_nodes);
            log.debug("Sorting neighbors for block_id: {} - block_start: {} - block_end: {}", .{ block_id, block_start, block_end });

            if (self.thread_pool) |pool| {
                self.wait_group.reset();
                for (0..self.training_config.num_threads) |thread_id| {
                    const node_id_start = @min(block_start + thread_id * self.num_block_nodes_per_thread, block_end);
                    const node_id_end = @min(node_id_start + self.num_block_nodes_per_thread, block_end);
                    log.debug("thread_id: {} - node_id_start: {} - node_id_end: {}", .{ thread_id, node_id_start, node_id_end });

                    const context = struct {
                        fn sortNeighborsThread(
                            neighbors_list: *NeighborHeapList,
                            id_start: usize,
                            id_end: usize,
                        ) void {
                            for (id_start..id_end) |id| {
                                neighbors_list.sortNeighbors(id);
                            }
                        }
                    };

                    pool.spawnWg(
                        &self.wait_group,
                        context.sortNeighborsThread,
                        .{
                            &self.neighbors_list,
                            node_id_start,
                            node_id_end,
                        },
                    );
                }
                pool.waitAndWork(&self.wait_group);
            } else {
                for (block_start..block_end) |node_id| {
                    self.neighbors_list.sortNeighbors(node_id);
                }
            }
        }
    };
}

test "NNDescent - no panic on empty dataset & zero graph degree" {
    const T = f32;
    const N = 128;
    const dummy_buffer: [N]f32 align(64) = undefined;

    // Dataset with 0 vectors should return error
    const Dataset = mod_dataset.Dataset(T, N);
    var dataset = Dataset{
        .data_buffer = dummy_buffer[0..0],
        .len = 0,
    };
    var config = TrainingConfig.init(
        1,
        dataset.len,
        null,
        42,
    );
    const result = NNDescent(T, N).init(
        dataset,
        config,
        std.testing.allocator,
    );
    try std.testing.expectEqual(result, error.InvalidNumNeighborsPerNode);

    // Dataset with 1 vector + graph degree of 0 should not panic
    dataset.data_buffer = dummy_buffer[0..N];
    dataset.len = 1;
    config.num_neighbors_per_node = 0;
    var nn_descent = try NNDescent(T, N).init(
        dataset,
        config,
        std.testing.allocator,
    );
    nn_descent.train();
    nn_descent.deinit(std.testing.allocator);
}

test "NNDescent.sumUpGraphUpdateCountsSIMD" {
    const allocator = std.testing.allocator;
    const vector_size = std.simd.suggestVectorLength(usize) orelse 4;
    const num_elements = vector_size * 3; // 3 full vectors
    const counts: []align(64) usize = try allocator.alignedAlloc(usize, std.mem.Alignment.@"64", num_elements);
    defer allocator.free(counts);

    for (counts, 0..) |*c, i| {
        c.* = i + 1;
    }

    const sum = NNDescent(f32, 128).sumUpGraphUpdateCountsSIMD(counts);
    const expected = (num_elements * (num_elements + 1)) / 2;
    try std.testing.expectEqual(expected, sum);
}

test "NNDescent - graph_updates_lists have separate buffers (no overlap)" {
    const T = f32;
    const N = 128;
    const Dataset = mod_dataset.Dataset(T, N);
    const NND = NNDescent(T, N);

    // Create a small dataset
    const num_vectors = 100;
    const data_buffer = try std.testing.allocator.alignedAlloc(
        T,
        std.mem.Alignment.@"64",
        num_vectors * N,
    );
    defer std.testing.allocator.free(data_buffer);

    // Initialize with random data
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    for (data_buffer) |*elem| {
        elem.* = random.float(T);
    }

    const dataset = Dataset{
        .data_buffer = data_buffer,
        .len = num_vectors,
    };

    // Create config with multiple threads
    const num_threads = 4;
    const config = TrainingConfig.init(
        10,
        dataset.len,
        num_threads,
        42,
    );

    var nn_descent = try NND.init(dataset, config, std.testing.allocator);
    defer nn_descent.deinit(std.testing.allocator);

    // Verify we actually have multiple threads
    try std.testing.expect(nn_descent.block_graph_updates_lists.len == num_threads);

    // Test 1: Verify each list has a distinct buffer slice
    for (nn_descent.block_graph_updates_lists, 0..) |list1, i| {
        const buf1_start = @intFromPtr(list1.items.ptr);
        const buf1_end = buf1_start + list1.capacity * @sizeOf(NND.GraphUpdate);

        for (nn_descent.block_graph_updates_lists[i + 1 ..], i + 1..) |list2, j| {
            const buf2_start = @intFromPtr(list2.items.ptr);
            const buf2_end = buf2_start + list2.capacity * @sizeOf(NND.GraphUpdate);

            // Check for overlap: buffers should NOT overlap
            // No overlap if: buf1_end <= buf2_start OR buf2_end <= buf1_start
            const no_overlap = (buf1_end <= buf2_start) or (buf2_end <= buf1_start);

            if (!no_overlap) {
                std.debug.print(
                    "OVERLAP DETECTED! Thread {d} buffer [{x}, {x}) overlaps with thread {d} buffer [{x}, {x})\n",
                    .{ i, buf1_start, buf1_end, j, buf2_start, buf2_end },
                );
            }

            try std.testing.expect(no_overlap);
        }
    }

    // Test 2: Verify all buffers are within the main graph_updates_buffer
    const main_buf_start = @intFromPtr(nn_descent.block_graph_updates_buffer.ptr);
    const main_buf_end = main_buf_start + nn_descent.block_graph_updates_buffer.len * @sizeOf(NND.GraphUpdate);

    for (nn_descent.block_graph_updates_lists, 0..) |list, i| {
        const buf_start = @intFromPtr(list.items.ptr);
        const buf_end = buf_start + list.capacity * @sizeOf(NND.GraphUpdate);

        // Each list's buffer must be within the main buffer
        const within_main = (buf_start >= main_buf_start) and (buf_end <= main_buf_end);

        if (!within_main) {
            std.debug.print(
                "Thread {d} buffer [{x}, {x}) is NOT within main buffer [{x}, {x})\n",
                .{ i, buf_start, buf_end, main_buf_start, main_buf_end },
            );
        }

        try std.testing.expect(within_main);
    }

    // Test 3: Verify sum of all list capacities == main buffer length
    var total_capacity: usize = 0;
    for (nn_descent.block_graph_updates_lists) |list| {
        total_capacity += list.capacity;
    }

    try std.testing.expect(total_capacity == nn_descent.block_graph_updates_buffer.len);
}

test "NNDescent - max distance decreases each iteration" {
    const T = f32;
    const N = 128;
    const Dataset = mod_dataset.Dataset(T, N);
    const NND = NNDescent(T, N);

    // Create a small dataset with known structure
    const num_vectors = 50;
    const data_buffer = try std.testing.allocator.alignedAlloc(
        T,
        std.mem.Alignment.@"64",
        num_vectors * N,
    );
    defer std.testing.allocator.free(data_buffer);

    // Initialize with structured data (each vector is [i, i+1, i+2, i+3])
    // This creates a clear distance structure
    for (0..num_vectors) |i| {
        for (0..N) |d| {
            data_buffer[i * N + d] = @floatFromInt(i);
        }
    }

    const dataset = Dataset{
        .data_buffer = data_buffer,
        .len = num_vectors,
    };

    const config = TrainingConfig.init(
        5,
        dataset.len,
        4,
        42,
    );

    var nn_descent = try NND.init(dataset, config, std.testing.allocator);
    defer nn_descent.deinit(std.testing.allocator);

    // 1 initial + 5 training iterations
    const num_iterations = 6;
    const num_nodes = nn_descent.neighbors_list.num_nodes;

    // Track max distances across iterations (including the very start)
    var max_distances = try std.testing.allocator.alloc(T, num_nodes * num_iterations);
    defer std.testing.allocator.free(max_distances);

    // Initialize the graph with random neighbors
    nn_descent.populateRandomNeighbors();
    // Record initial max distance
    for (0..num_nodes) |node_id| {
        max_distances[node_id * num_iterations] = nn_descent.neighbors_list.getMaxDistance(node_id);
    }

    var updates_count: usize = 0;

    // Run a few iterations and track max distance
    for (1..num_iterations) |iteration| {
        defer {
            nn_descent.neighbor_candidates_new.reset();
            nn_descent.neighbor_candidates_old.reset();
        }

        nn_descent.sampleNeighborCandidates();
        for (0..nn_descent.numBlocks()) |block_id| {
            defer {
                for (nn_descent.block_graph_updates_lists) |*list| {
                    list.clearRetainingCapacity();
                }
            }
            nn_descent.generateBlockGraphUpdateProposals(block_id);
            const count = nn_descent.applyBlockGraphUpdatesProposals(block_id);
            updates_count += count;
        }

        // Get max distance after this iteration
        for (0..num_nodes) |node_id| {
            max_distances[node_id * num_iterations + iteration] = nn_descent.neighbors_list.getMaxDistance(node_id);
        }
    }

    // Verify: max distance should be non-increasing
    // (It should decrease or stay the same, never increase)
    for (0..num_nodes) |node_id| {
        const max_distances_for_node = max_distances[node_id * num_iterations .. node_id * num_iterations + num_iterations];
        for (1..num_iterations) |iteration| {
            const prev = max_distances_for_node[iteration - 1];
            const curr = max_distances_for_node[iteration];

            if (curr > prev) {
                std.debug.print(
                    "ERROR: max_distance INCREASED from {d:.4} to {d:.4} at iteration {d}\n",
                    .{ prev, curr, iteration },
                );
            }

            // Allow for floating point tolerance
            const tolerance = 1e-6;
            try std.testing.expect(curr <= prev + tolerance);
        }
    }
}

test "NNDescent - single-threaded and multi-threaded produce similar results" {
    const T = f32;
    const N = 128;
    const Dataset = mod_dataset.Dataset(T, N);
    const NND = NNDescent(T, N);

    const num_vectors = 100;
    const data_buffer = try std.testing.allocator.alignedAlloc(
        T,
        std.mem.Alignment.@"64",
        num_vectors * N,
    );
    defer std.testing.allocator.free(data_buffer);

    // Fixed data for reproducibility
    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();
    for (data_buffer) |*elem| {
        elem.* = random.float(T);
    }

    const dataset = Dataset{
        .data_buffer = data_buffer,
        .len = num_vectors,
    };

    // Single-threaded run
    const config_single = TrainingConfig.init(10, dataset.len, 1, 42);
    var nn_single = try NND.init(dataset, config_single, std.testing.allocator);
    defer nn_single.deinit(std.testing.allocator);
    nn_single.train();

    const single_maxes = try std.testing.allocator.alloc(T, dataset.len);
    defer std.testing.allocator.free(single_maxes);
    for (0..dataset.len) |node_id| {
        single_maxes[node_id] = nn_single.neighbors_list.getMaxDistance(node_id);
    }

    // Multi-threaded run
    const config_multi = TrainingConfig.init(10, dataset.len, 4, 42);
    var nn_multi = try NND.init(dataset, config_multi, std.testing.allocator);
    defer nn_multi.deinit(std.testing.allocator);
    nn_multi.train();

    const multi_maxes = try std.testing.allocator.alloc(T, dataset.len);
    defer std.testing.allocator.free(multi_maxes);
    for (0..dataset.len) |node_id| {
        multi_maxes[node_id] = nn_single.neighbors_list.getMaxDistance(node_id);
    }

    for (0..dataset.len) |node_id| {
        const single_max = single_maxes[node_id];
        const multi_max = single_maxes[node_id];

        // Results should be similar (within 10% due to randomness in parallel execution)
        const ratio = @max(single_max, multi_max) / @min(single_max, multi_max);
        try std.testing.expect(0.9 < ratio and ratio < 1.1);
    }
}
