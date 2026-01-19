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
        /// Thread pool for multi-threaded operations.
        pool: std.Thread.Pool,
        /// Wait group for synchronizing threads.
        wait_group: std.Thread.WaitGroup,

        /// Holds EntryWithFlag entries.
        const NeighborHeapList = mod_neighbors.NeighborHeapList(T, true);
        /// Holds EntryWithoutFlag entries.
        const CandidateHeapList = mod_neighbors.NeighborHeapList(i32, false);
        const GraphUpdate = struct {
            node_id: usize,
            neighbor_id: isize,
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
                    self.graph_updates_list.clearRetainingCapacity();
                }

                // Sample neighbor candidates into new and old candidate lists
                try self.sampleNeighborCandidates();

                // TODO: Implement the rest:
                // - generate new neighbor proposals from candidates
                // - update neighbor lists and track changes
                // - check for convergence based on delta
                self.generateGraphUpdateProposals();
            }
        }

        /// Populate all nodes with random neighbors.
        /// Use multi-threading if configured.
        fn populateRandomNeighbors(self: *Self) void {
            if (self.training_config.num_threads <= 1) {
                populateRandomNeighborsThread(
                    &self.dataset,
                    &self.neighbors_list,
                    self.training_config.num_neighbors_per_node,
                    0,
                    self.dataset.len,
                    self.training_config.seed,
                );
            } else {
                const num_threads = self.training_config.num_threads;

                const num_nodes = self.neighbors_list.num_nodes;
                const batch_size = (num_nodes + num_threads - 1) / num_threads;

                var node_id_start: usize = 0;
                while (node_id_start < num_nodes) : (node_id_start += batch_size) {
                    const node_id_end = @min(node_id_start + batch_size, num_nodes);

                    // SAFETY: Each thread populate neighbors on non-overlapping range of nodes,
                    // so the neighbor heaps are separate in memory, and thus no data races.
                    self.pool.spawnWg(
                        &self.wait_group,
                        populateRandomNeighborsThread,
                        .{
                            .dataset = &self.dataset,
                            .neighbors_list = &self.neighbors_list,
                            .num_neighbors = self.training_config.num_neighbors_per_node,
                            .node_id_start = node_id_start,
                            .node_id_end = node_id_end,
                            // Different thread has different seed
                            .seed = self.training_config.seed + @as(u64, @intCast(node_id_start)),
                        },
                    );
                }
                self.pool.waitAndWork(&self.wait_group);
            }
        }

        /// Populate a batch of nodes, starting from `node_id_start` (inclusive) to `node_id_end` (exclusive),
        /// with random neighbors.
        fn populateRandomNeighborsThread(
            dataset: *const Dataset,
            neighbors_list: *NeighborHeapList,
            num_neighbors_per_node: usize,
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
                for (0..num_neighbors_per_node) |_| {
                    // We accept the possibility of a node pointing to itself here.
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
                markSampledOldThread(
                    &self.neighbors_list,
                    &self.neighbor_candidates_new,
                    0,
                    self.neighbors_list.num_nodes,
                );
            } else {
                const num_threads = self.training_config.num_threads;

                const num_nodes = self.dataset.len;
                const batch_size = (num_nodes + num_threads - 1) / num_threads;

                var node_id_start: usize = 0;
                while (node_id_start < num_nodes) : (node_id_start += batch_size) {
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
                            .seed = self.training_config.seed + @as(u64, @intCast(node_id_start)),
                        },
                    );
                }
                // Wait for all sampling threads to finish before moving on
                self.pool.waitAndWork(&self.wait_group);

                // Mark sampled nodes in neighbors_list as not new anymore
                node_id_start = 0;
                while (node_id_start < num_nodes) : (node_id_start += batch_size) {
                    const node_id_end = @min(node_id_start + batch_size, num_nodes);
                    self.pool.spawnWg(
                        &self.wait_group,
                        markSampledOldThread,
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
                for (0..neighbors_list.num_neighbors_per_node) |neighbor_idx| {
                    const neighbor_entry = neighbors_list.getEntry(node_id, neighbor_idx);

                    // Skip this neighbor slot if it's empty
                    if (neighbor_entry.neighbor_id == CandidateHeapList.EMPTY_ID) continue;
                    // SAFETY: neighbor_id is not EMPTY_ID, so this cast is safe.
                    const neighbor_id = @as(usize, neighbor_entry.neighbor_id);

                    // Generate a random priority for this neighbor
                    // TODO: Consider if this sampling strategy is appropriate
                    const priority = rng.int(i32);

                    // Assign to neighbor_candidates_new or neighbor_candidates_old based on is_new flag
                    const target_candidate_list = if (neighbor_entry.is_new)
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
        fn markSampledOldThread(
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

        fn generateGraphUpdateProposals(_: *Self) void {
            // TODO: Go through all nodes:
            // For each node, check all new-new and new-old candidate pairs (skipping empty slots),
            // compute distances, and if a closer neighbor is found,
            // add to the graph_updates_lists.
            // The number of updates added should never exceed the original capacity of each list.
        }
    };
}
