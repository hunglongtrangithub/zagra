const std = @import("std");
const log = std.log.scoped(.nn_descent);

const mod_neighbors = @import("neighbors.zig");
const mod_dataset = @import("../dataset.zig");

/// Configuration parameters for training the k-NN graph using NN-descent.
/// Referenced from https://github.com/brj0/nndescent/blob/main/src/nnd.h
pub const TrainingConfig = struct {
    /// The number of neighbors (k) each point should have in the k-NN graph.
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
    num_threads: usize = std.Thread.getCpuCount() catch 1,

    /// Random seed for any randomized components of the algorithm.
    seed: u64 = std.crypto.random.int(u64),

    const Self = @This();

    /// Initializes a training config with default values based on the dataset size.
    pub fn init(num_neighbors_per_node: usize, num_vectors: usize, seed: ?u64) Self {
        const config = Self{
            .num_neighbors_per_node = num_neighbors_per_node,
            .max_iterations = @max(5, @log2(num_vectors)),
            .max_candidates = @min(60, num_neighbors_per_node),
        };
        if (seed) |s| config.seed = s;
        return config;
    }
};

/// NN-Descent struct to construct the k-NN graph from a dataset.
/// Generics:
/// - T: Element type of the vectors, supported in `types.ElemType`.
/// - N: Dimensionality of the vectors in the dataset, supported in `types.DimType`.
pub fn NNDescent(comptime T: type, comptime N: usize) type {
    const Dataset = mod_dataset.Dataset(T, N);

    return struct {
        /// The dataset of vectors to build the k-NN graph for.
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
        /// Capacity of each list is large enough to hold all possible updates in an iteration.
        graph_updates_lists: []std.ArrayList(GraphUpdate),
        /// Buffer to hold graph updates during generation of proposals.
        /// Used by all arrays in `graph_updates_lists`.
        graph_updates_buffer: []GraphUpdate,
        graph_update_counts_buffer: []align(64) usize,
        /// Thread pool for multi-threaded operations.
        pool: std.Thread.Pool,
        /// Wait group for synchronizing threads.
        wait_group: std.Thread.WaitGroup,

        /// Holds EntryWithFlag entries.
        const NeighborHeapList = mod_neighbors.NeighborHeapList(T, true);
        /// Holds EntryWithoutFlag entries.
        const CandidateHeapList = mod_neighbors.NeighborHeapList(i32, false);
        /// Represents a proposed update to the k-NN graph.
        /// Holds the IDs of the two nodes involved and their distance.
        const GraphUpdate = struct {
            node1_id: usize,
            node2_id: usize,
            distance: T,
        };
        const Self = @This();

        /// Initialize NN-Descent with the given dataset and training configuration.
        pub fn init(
            dataset: Dataset,
            training_config: TrainingConfig,
            allocator: std.mem.Allocator,
        ) (mod_neighbors.InitError || std.mem.Allocator.Error)!Self {
            const neighbors_list = try NeighborHeapList.init(
                dataset.len,
                training_config.num_neighbors_per_node,
                allocator,
            );

            const neighbor_candidates_new = try CandidateHeapList.init(
                dataset.len,
                training_config.max_candidates,
            );
            const neighbor_candidates_old = try CandidateHeapList.init(
                dataset.len,
                training_config.max_candidates,
            );

            const graph_updates_lists = try allocator.alloc(
                std.ArrayList(GraphUpdate),
                training_config.num_threads,
            );

            // Calculate maximum possible graph updates in an iteration
            // For each node, possible updates are:
            // - New-New neighbor pairs: n_new * (n_new - 1) / 2
            // - New-Old neighbor pairs: n_new * n_old
            const capacity_per_thread = (neighbor_candidates_new.num_nodes * (neighbor_candidates_new.num_nodes - 1)) / 2 +
                (neighbor_candidates_new.num_nodes * neighbor_candidates_old.num_nodes);
            const num_max_graph_updates = capacity_per_thread * training_config.num_threads;
            const graph_updates_buffer = try allocator.alloc(
                GraphUpdate,
                num_max_graph_updates,
            );

            // Initialize each graph updates list using a slice of the buffer
            for (graph_updates_lists, 0..) |*list, i| {
                const start = i * capacity_per_thread;
                const end = start + capacity_per_thread;
                list.* = std.ArrayList(GraphUpdate).initBuffer(
                    graph_updates_buffer[start..end],
                );
            }

            const graph_update_counts_buffer: []align(64) usize = try allocator.alignedAlloc(
                usize,
                std.mem.Alignment.@"64",
                training_config.num_threads,
            );

            var pool: std.Thread.Pool = undefined;
            try pool.init(.{
                .n_jobs = training_config.num_threads,
                .allocator = allocator,
            });

            var wait_group: std.Thread.WaitGroup = undefined;
            wait_group.reset();

            return Self{
                .dataset = dataset,
                .training_config = training_config,
                .neighbors_list = neighbors_list,
                .neighbor_candidates_new = neighbor_candidates_new,
                .neighbor_candidates_old = neighbor_candidates_old,
                .graph_updates_buffer = graph_updates_buffer,
                .graph_update_counts_buffer = graph_update_counts_buffer,
                .graph_updates_lists = graph_updates_lists,
                .pool = pool,
                .wait_group = wait_group,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.neighbors_list.deinit(allocator);
            self.dataset.deinit(allocator);
            self.neighbor_candidates_new.deinit(allocator);
            self.neighbor_candidates_old.deinit(allocator);
            allocator.free(self.graph_updates_lists);
            // NOTE: Just need to free the buffer, since all lists use slices of it
            allocator.free(self.graph_updates_buffer);
            allocator.free(self.graph_update_counts_buffer);
            self.pool.deinit();
        }

        pub fn train(self: *Self) void {
            // Step 1: Populate initial random neighbors
            self.populateRandomNeighbors();

            // Step 2: Iteratively refine the neighbor lists
            for (0..self.training_config.max_iterations) |iteration| {
                log.info("NN-Descent iteration {d}", .{iteration});
                defer {
                    // TODO: Do we need to reset candidate lists, or just overwrite them in the next iteration?
                    self.neighbor_candidates_new.reset();
                    self.neighbor_candidates_old.reset();
                    for (self.graph_updates_lists) |*list| {
                        list.clearRetainingCapacity();
                    }
                }

                // Sample neighbor candidates into new and old candidate lists
                try self.sampleNeighborCandidates();

                // TODO: Implement the rest:
                // - [x] generate new neighbor proposals from candidates
                // - [ ] update neighbor lists and track changes
                // - [ ] check for convergence based on delta
                self.generateGraphUpdateProposals();
                const updates_count = self.applyGraphUpdatesProposals();

                log.info("Applied {d} graph updates", .{updates_count});

                if (updates_count <=
                    @as(usize, @intFromFloat(self.training_config.delta)) * self.neighbors_list.num_nodes * self.neighbors_list.num_neighbors_per_node)
                {
                    log.info("Converged after {d} iterations", .{iteration + 1});
                    break;
                }
            }

            log.info("NN-Descent training completed", .{});
        }

        /// Populate all nodes with random neighbors.
        /// Use multi-threading if configured.
        fn populateRandomNeighbors(self: *Self) void {
            if (self.training_config.num_threads <= 1) {
                populateRandomNeighborsThread(
                    &self.dataset,
                    &self.neighbors_list,
                    0,
                    self.dataset.len,
                    self.training_config.seed,
                );
                return;
            }

            const num_threads = self.training_config.num_threads;

            const num_nodes = self.neighbors_list.num_nodes;
            const batch_size = (num_nodes + num_threads - 1) / num_threads;

            for (0..num_threads) |thread_id| {
                const node_id_start = thread_id * batch_size;
                const node_id_end = @min(node_id_start + batch_size, num_nodes);

                // SAFETY: Each thread populate neighbors on non-overlapping range of nodes,
                // so the neighbor heaps are separate in memory, and thus no data races.
                self.pool.spawnWg(
                    &self.wait_group,
                    populateRandomNeighborsThread,
                    .{
                        .dataset = &self.dataset,
                        .neighbors_list = &self.neighbors_list,
                        .node_id_start = node_id_start,
                        .node_id_end = node_id_end,
                        // Different thread has different seed
                        .seed = self.training_config.seed + @as(u64, @intCast(thread_id)),
                    },
                );
            }
            self.pool.waitAndWork(&self.wait_group);
        }

        /// Populate a batch of nodes, starting from `node_id_start` (inclusive) to `node_id_end` (exclusive),
        /// with random neighbors.
        fn populateRandomNeighborsThread(
            dataset: *const Dataset,
            neighbors_list: *NeighborHeapList,
            node_id_start: usize,
            node_id_end: usize,
            seed: u64,
        ) void {
            const num_nodes = neighbors_list.num_nodes;
            std.debug.assert(node_id_start < node_id_end and node_id_end <= num_nodes);

            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();

            for (node_id_start..node_id_end) |node_id| {
                const node = dataset.getUnchecked(node_id);
                for (0..neighbors_list.num_neighbors_per_node) |_| {
                    // We accept the possibility of a node pointing to itself here.
                    // TODO: Consider avoiding self-loops?
                    const neighbor_id = rng.intRangeAtMost(usize, 0, num_nodes - 1);

                    const neighbor = dataset.getUnchecked(neighbor_id);
                    const distance = node.sqdist(neighbor);

                    const neighbor_entry = NeighborHeapList.Entry{
                        // SAFETY: neighbor_id is in [0, num_nodes), which fits in isize.
                        .neighbor_id = @intCast(neighbor_id),
                        .distance = distance,
                        .is_new = true,
                    };

                    _ = neighbors_list.tryAddNeighbor(node_id, neighbor_entry);
                }
            }
        }

        /// Sample neighbor candidates from the `neighbors_list` into `neighbor_candidates_new` and `neighbor_candidates_old`.
        /// Use multi-threading if configured.
        fn sampleNeighborCandidates(self: *Self) void {
            std.debug.assert(self.neighbor_candidates_new.num_nodes == self.neighbor_candidates_old.num_nodes);
            std.debug.assert(self.neighbor_candidates_new.num_nodes == self.neighbors_list.num_nodes);

            if (self.training_config.num_threads <= 1) {
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
                return;
            }

            const num_threads = self.training_config.num_threads;

            const num_nodes = self.neighbors_list.num_nodes;
            const batch_size = (num_nodes + num_threads - 1) / num_threads;

            for (0..num_threads) |thread_id| {
                const node_id_start = thread_id * batch_size;
                const node_id_end = @min(node_id_start + batch_size, num_nodes);

                // SAFETY: Each thread only touches on heaps of nodes whose IDs are
                // in the range [node_id_start, node_id_end), so no data races.
                self.pool.spawnWg(
                    &self.wait_group,
                    sampleNeighborCandidatesThread,
                    .{
                        .neighbors_list = &self.neighbors_list,
                        .neighbor_candidates_new = &self.neighbor_candidates_new,
                        .neighbor_candidates_old = &self.neighbor_candidates_old,
                        .node_id_start = node_id_start,
                        .node_id_end = node_id_end,
                        // Different thread has different seed
                        .seed = self.training_config.seed + @as(u64, @intCast(thread_id)),
                    },
                );
            }
            // Wait for all sampling threads to finish before moving on
            self.pool.waitAndWork(&self.wait_group);

            // Mark sampled nodes in neighbors_list as not new anymore
            for (0..num_threads) |thread_id| {
                const node_id_start = thread_id * batch_size;
                const node_id_end = @min(node_id_start + batch_size, num_nodes);

                // SAFETY: Each thread only touches on heaps of nodes whose IDs are
                // in the range [node_id_start, node_id_end), so no data races.
                self.pool.spawnWg(
                    &self.wait_group,
                    markSampledToOldThread,
                    .{
                        .neighbors_list = &self.neighbors_list,
                        .neighbor_candidates_new = &self.neighbor_candidates_new,
                        .node_id_start = node_id_start,
                        .node_id_end = node_id_end,
                    },
                );
            }
            self.pool.waitAndWork(&self.wait_group);
        }

        /// Sample neighbor candidates from the `neighbors_list` into `neighbor_candidates_new` and `neighbor_candidates_old`.
        /// Each node in the candidate list will contain nodes that point to it (forward neighbors)
        /// and nodes that it points to (reverse neighbors).
        /// Goes through all edges in `neighbors_list`, and only tries to add neighbors to a node whose node ID
        /// is in the range `[node_id_start, node_id_end)` to the candidate lists.
        fn sampleNeighborCandidatesThread(
            neighbors_list: *NeighborHeapList,
            neighbor_candidates_new: *CandidateHeapList,
            neighbor_candidates_old: *CandidateHeapList,
            node_id_start: usize,
            node_id_end: usize,
            seed: u64,
        ) void {
            std.debug.assert(node_id_start < node_id_end and node_id_end <= neighbors_list.num_nodes);

            // Initialize PRNG with thread-specific seed
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();

            for (0..neighbors_list.num_nodes) |node_id| {
                const neighbor_id_slice: []isize = neighbors_list.getEntryFieldSlice(node_id, .neighbor_id);
                const is_new_slice: []bool = neighbors_list.getEntryFieldSlice(node_id, .is_new);

                for (0..neighbors_list.num_neighbors_per_node) |neighbor_idx| {
                    const entry_neighbor_id = neighbor_id_slice[neighbor_idx];
                    const entry_is_new = is_new_slice[neighbor_idx];

                    // Skip this neighbor slot if it's empty
                    if (entry_neighbor_id == CandidateHeapList.EMPTY_ID) continue;
                    // SAFETY: neighbor_id is not EMPTY_ID, so this cast is safe.
                    const neighbor_id = @as(usize, entry_neighbor_id);

                    // Generate a random priority for this neighbor
                    // TODO: Consider if this sampling strategy is appropriate
                    const priority = rng.int(i32);

                    // Assign to neighbor_candidates_new or neighbor_candidates_old based on is_new flag
                    const target_candidate_list = if (entry_is_new)
                        neighbor_candidates_new
                    else
                        neighbor_candidates_old;

                    if (node_id >= node_id_start and node_id < node_id_end) {
                        // Add neighbor_id as a forward neighbor candidate for node_id
                        _ = target_candidate_list.tryAddNeighbor(
                            node_id,
                            CandidateHeapList.Entry{
                                .neighbor_id = neighbor_id,
                                .distance = priority,
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
            neighbor_candidates_new: *CandidateHeapList,
            node_id_start: usize,
            node_id_end: usize,
        ) void {
            std.debug.assert(node_id_start < node_id_end and node_id_end <= neighbors_list.num_nodes);

            for (node_id_start..node_id_end) |node_id| {
                // Get the slice of neighbor IDs in the new candidate list for this node
                const neighbor_candidate_ids: []isize = neighbor_candidates_new.getEntryFieldSlice(
                    node_id,
                    .neighbor_id,
                );
                for (0..neighbors_list.num_neighbors_per_node) |neighbor_idx| {
                    const neighbor_id: isize = neighbors_list.getEntryFieldPtr(
                        node_id,
                        neighbor_idx,
                        .neighbor_id,
                    ).*;

                    // Check if the neighbor ID is valid
                    if (neighbor_id == NeighborHeapList.EMPTY_ID) continue;

                    if (std.mem.indexOfScalar(
                        isize,
                        neighbor_candidate_ids,
                        neighbor_id,
                    ) != null) {
                        // Mark as not new anymore
                        neighbors_list.getEntryFieldPtr(
                            node_id,
                            neighbor_idx,
                            .is_new,
                        ).* = false;
                    }
                }
            }
        }

        fn generateGraphUpdateProposals(self: *Self) void {
            if (self.training_config.num_threads <= 1) {
                generateGraphUpdateProposalsThread(
                    &self.dataset,
                    &self.neighbors_list,
                    &self.graph_updates_lists[0],
                    &self.neighbor_candidates_new,
                    &self.neighbor_candidates_old,
                    0,
                    self.neighbors_list.num_nodes,
                );
                return;
            }
            const num_threads = self.training_config.num_threads;

            const num_nodes = self.neighbors_list.num_nodes;
            const batch_size = (num_nodes + num_threads - 1) / num_threads;

            for (0..num_threads) |thread_id| {
                const node_id_start = thread_id * batch_size;
                const node_id_end = @min(node_id_start + batch_size, num_nodes);
                self.pool.spawnWg(
                    &self.wait_group,
                    generateGraphUpdateProposalsThread,
                    .{
                        .dataset = &self.dataset,
                        .neighbors_list = &self.neighbors_list,
                        .graph_updates_list = &self.graph_updates_lists[thread_id],
                        .neighbor_candidates_new = &self.neighbor_candidates_new,
                        .neighbor_candidates_old = &self.neighbor_candidates_old,
                        .local_join_id_start = node_id_start,
                        .local_join_id_end = node_id_end,
                    },
                );
            }
            self.pool.waitAndWork(&self.wait_group);
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
            std.debug.assert(local_join_id_start < local_join_id_end and local_join_id_end <= neighbors_list.num_nodes);
            std.debug.assert(neighbors_list.num_nodes == dataset.len);

            // Go through all local joins in the given range
            for (local_join_id_start..local_join_id_end) |local_join_id| {
                const new_candidate_ids: []isize = neighbor_candidates_new.getEntryFieldSlice(local_join_id, .neighbor_id);
                const old_candidate_ids: []isize = neighbor_candidates_old.getEntryFieldSlice(local_join_id, .neighbor_id);

                for (new_candidate_ids, 0..) |cand1_id, i| {
                    if (cand1_id == CandidateHeapList.EMPTY_ID) continue;
                    const cand1_vector = dataset.getUnchecked(@as(usize, cand1_id));
                    // Take current max distance in neighbor heap as threshold
                    const cand1_distance_threshold: T = neighbors_list.getEntryFieldSlice(cand1_id, .distance)[0];

                    // New-New candidate pairs
                    for (new_candidate_ids[i + 1 ..]) |cand2_id| {
                        if (cand2_id == CandidateHeapList.EMPTY_ID) continue;
                        const cand2_vector = dataset.getUnchecked(@as(usize, cand2_id));
                        const cand2_distance_threshold: T = neighbors_list.getEntryFieldSlice(cand2_id, .distance)[0];

                        const distance = cand1_vector.sqdist(cand2_vector);

                        if (distance <= @max(cand1_distance_threshold, cand2_distance_threshold)) {
                            // Found a closer neighbor (either to cand1 or cand2), add to graph updates
                            graph_updates_list.appendAssumeCapacity(.{
                                .distance = distance,
                                .node1_id = cand1_id,
                                .node2_id = cand2_id,
                            });
                        }
                    }

                    // New-Old candidate pairs
                    for (old_candidate_ids) |cand2_id| {
                        if (cand2_id == CandidateHeapList.EMPTY_ID) continue;
                        const cand2_vector = dataset.getUnchecked(@as(usize, cand2_id));
                        // Take current max distance in neighbor heap as threshold
                        const cand2_distance_threshold: []T = neighbors_list.getEntryFieldSlice(local_join_id, .distance)[0];

                        const distance = cand1_vector.sqdist(cand2_vector);

                        if (distance <= @max(cand1_distance_threshold, cand2_distance_threshold)) {
                            // Found a closer neighbor (either to cand1 or cand2), add to graph updates
                            graph_updates_list.appendAssumeCapacity(.{
                                .distance = distance,
                                .node1_id = cand1_id,
                                .node2_id = cand2_id,
                            });
                        }
                    }
                }
            }

            // Graph update list should not exceed its designated capacity
            std.debug.assert(graph_updates_list.items.len <= graph_updates_list.capacity);
        }

        /// Apply graph updates from all threads' `graph_updates_lists` to the `neighbors_list`.
        /// Return the total number of successful updates applied.
        fn applyGraphUpdatesProposals(self: *Self) usize {
            const num_threads = self.training_config.num_threads;

            if (num_threads <= 1) {
                applyGraphUpdatesProposalsThread(
                    &self.neighbors_list,
                    &self.graph_updates_lists[0],
                    &self.graph_update_counts_buffer[0],
                    0,
                    self.neighbors_list.num_nodes,
                );
                return self.graph_update_counts_buffer[0];
            }

            const num_nodes = self.neighbors_list.num_nodes;
            const batch_size = (num_nodes + num_threads - 1) / num_threads;

            for (0..num_threads) |thread_id| {
                const node_id_start = thread_id * batch_size;
                const node_id_end = @min(node_id_start + batch_size, num_nodes);
                // SAFETY: Each thread only touches on heaps of nodes whose IDs are
                // in the range [node_id_start, node_id_end), so no data races.
                self.pool.spawnWg(
                    &self.wait_group,
                    applyGraphUpdatesProposalsThread,
                    .{
                        .neighbors_list = &self.neighbors_list,
                        .graph_updates_list = &self.graph_updates_lists[thread_id],
                        .graph_updates_count_ptr = &self.graph_update_counts_buffer[thread_id],
                        .local_join_id_start = node_id_start,
                        .local_join_id_end = node_id_end,
                    },
                );
            }
            self.pool.waitAndWork(&self.wait_group);

            // Reduce the counts from all threads with SIMD
            return self.sumUpGraphUpdateCountsSIMD();
        }

        /// Apply graph updates from the given `graph_updates_list` to the `neighbors_list`,
        /// only for nodes whose IDs are in the range `[node_id_start, node_id_end)`.
        /// Count the number of successful updates applied and store in `graph_updates_count_ptr`.
        fn applyGraphUpdatesProposalsThread(
            neighbors_list: *NeighborHeapList,
            graph_updates_list: *std.ArrayList(GraphUpdate),
            graph_updates_count_ptr: *usize,
            node_id_start: usize,
            node_id_end: usize,
        ) void {
            std.debug.assert(node_id_start < node_id_end and node_id_end <= neighbors_list.num_nodes);

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
        fn sumUpGraphUpdateCountsSIMD(self: *Self) usize {
            const counts_ptr: [*]align(64) usize = self.graph_update_counts_buffer.ptr;
            const num_counts = self.graph_update_counts_buffer.len;

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
    };
}
