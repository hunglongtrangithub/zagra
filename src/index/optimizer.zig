const std = @import("std");
const log = std.log.scoped(.optimizer);

const mod_types = @import("../types.zig");
const mod_soa_slice = @import("soa_slice.zig");
const mod_nn_descent = @import("nn_descent.zig");

pub const Optimizer = struct {
    pub const Error = error{
        /// The number of edges is too large to fit in memory.
        NumberOfEdgesTooLarge,
        /// The desired graph degree is too large that causes overflow.
        NumNeighborsPerNodeTooLarge,
        /// The graph degree argument is larger the input graph degree.
        InvalidGraphDegree,
    };

    pub const OptimizationTiming = struct {
        count_detours_ns: u64,
        prune_ns: u64,
        build_reverse_graph_ns: u64,
        combine_ns: u64,
        total_optimization_ns: u64,
    };

    /// Generic neighbors list container.
    /// - `store_detour_count`:
    ///   if `true`, detour_count field is a `usize` (for input graph)
    ///   if `false`, detour_count field is a `void` (for output graph)
    pub fn NeighborsList(comptime store_detour_count: bool) type {
        return struct {
            pub const Entry = struct {
                neighbor_id: usize,
                detour_count: if (store_detour_count) usize else void,
            };

            /// Total number of points (n).
            num_nodes: usize,
            /// Number of neighbors per point (k).
            num_neighbors_per_node: usize,
            /// Row-major storage of all entries.
            /// Indexing: i * num_neighbors_per_node + j
            entries: mod_soa_slice.SoaSlice(Entry),

            pub fn init(
                num_nodes: usize,
                num_neighbors_per_node: usize,
                allocator: std.mem.Allocator,
            ) (error{NumberOfEdgesTooLarge} || std.mem.Allocator.Error)!@This() {
                const total_edges = std.math.mul(usize, num_nodes, num_neighbors_per_node) catch return error.NumberOfEdgesTooLarge;
                const total_size = std.math.mul(usize, total_edges, @sizeOf(Entry)) catch return error.NumberOfEdgesTooLarge;
                if (total_size > std.math.maxInt(isize)) return error.NumberOfEdgesTooLarge;
                return .{
                    .num_nodes = num_nodes,
                    .num_neighbors_per_node = num_neighbors_per_node,
                    .entries = try mod_soa_slice.SoaSlice(Entry).init(total_edges, allocator),
                };
            }

            pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
                self.entries.deinit(allocator);
            }

            pub inline fn getEntryFieldSlice(
                self: *const @This(),
                node_id: usize,
                comptime field: std.meta.FieldEnum(Entry),
            ) []std.meta.fieldInfo(Entry, field).type {
                std.debug.assert(node_id < self.num_nodes);
                const start = node_id * self.num_neighbors_per_node;
                return self.entries.items(field)[start .. start + self.num_neighbors_per_node];
            }
        };
    }

    /// Entries for all neighbors of all nodes, stored in row-major order.
    /// Neighbor slots for every node are all valid (non-empty) and there are no self-loops (a node cannot be its own neighbor).
    /// Neighbors for a node are arranged in descending order of distance, thus descending order of rank (this is guaranteed by the caller).
    neighbors_list: NeighborsList(true),
    /// Number of nodes in one block for block-wise computation. Capped by the total number of nodes.
    num_nodes_per_block: usize,

    /// Thread pool for multi-threaded operations.
    /// If null, the optimizer will run in single-threaded mode.
    /// If the thread pool has 0 threads, the optimizer will do nothing.
    thread_pool: ?*std.Thread.Pool,
    /// Wait group for synchronizing threads.
    wait_group: std.Thread.WaitGroup,

    const Self = @This();

    /// Initializes the optimizer with the borrowed neighbors list, thread pool, and number of nodes per block.
    pub fn init(
        neighbors_list: NeighborsList(true),
        thread_pool: ?*std.Thread.Pool,
        num_nodes_per_block: usize,
    ) Self {
        var optimizer: Self = undefined;
        optimizer.neighbors_list = neighbors_list;
        optimizer.num_nodes_per_block = @min(num_nodes_per_block, neighbors_list.num_nodes);
        optimizer.thread_pool = thread_pool;
        optimizer.wait_group.reset();

        return optimizer;
    }

    /// Optimizes the graph by removing redundant edges based on the number of detourable routes.
    /// Final graph degree is less than or equal to the initial number of neighbors per node.
    pub fn optimize(
        self: *Self,
        graph_degree: usize,
        allocator: std.mem.Allocator,
    ) (Error || std.mem.Allocator.Error)!NeighborsList(false) {
        const num_nodes = self.neighbors_list.num_nodes;
        const input_degree = self.neighbors_list.num_neighbors_per_node;
        const output_degree = if (graph_degree <= input_degree) graph_degree else return Error.InvalidGraphDegree;

        // Output graph that will store the optimized neighbors.
        // Does not store detour counts, only neighbor IDs.
        var output_graph = try NeighborsList(false).init(
            num_nodes,
            output_degree,
            allocator,
        );

        // Number of 2-hop neighbors to store for one node.
        // If input degree is 0, the number will just be 0
        const num_two_hop_neighbors_per_node =
            std.math.mul(
                usize,
                input_degree -| 1,
                input_degree -| 1,
            ) catch return Error.NumNeighborsPerNodeTooLarge;

        // Buffer for storing 2-hop neighbors during detour counting for each thread.
        const two_hop_neighbors_buffer_size =
            std.math.mul(
                usize,
                self.numThreads(),
                num_two_hop_neighbors_per_node,
            ) catch return Error.NumNeighborsPerNodeTooLarge;
        const two_hop_neighbors_buffer = try allocator.alloc(usize, two_hop_neighbors_buffer_size);
        defer allocator.free(two_hop_neighbors_buffer);

        // Buffer for storing reverse neighbor counts per node. The value of ech element
        // in this buffer may be larger than the output_degree (which is invalid),
        // but we will cap the number of reverse neighbors to output_degree in the combine step.
        const reverse_neighbor_counts = try allocator.alloc(usize, num_nodes);
        defer allocator.free(reverse_neighbor_counts);

        // Buffer for storing reverse neighbors.
        // Each node can have at most `output_degree` reverse neighbors.
        // The multiplication cannot overflow because of the check when initializing output graph.
        const reverse_neighbor_ids = try allocator.alloc(usize, num_nodes * output_degree);
        defer allocator.free(reverse_neighbor_ids);

        log.info("Couting detours in the input graph...", .{});
        self.countDetours(
            two_hop_neighbors_buffer,
            num_two_hop_neighbors_per_node,
        );
        log.info("Pruning edges from the graph...", .{});
        self.prune(&output_graph);
        log.info("Building reverse graph from pruned graph...", .{});
        self.buildReverseGraph(
            &output_graph,
            reverse_neighbor_counts,
            reverse_neighbor_ids,
        );
        log.info("Combining edges from pruned graph and reverse graph...", .{});
        self.combine(
            &output_graph,
            reverse_neighbor_counts,
            reverse_neighbor_ids,
        );

        return output_graph;
    }

    /// Optimizes the graph with timing information for each phase.
    /// Returns both the optimized graph and timing information.
    pub fn optimizeWithTiming(
        self: *Self,
        graph_degree: usize,
        allocator: std.mem.Allocator,
    ) (Error || std.mem.Allocator.Error || std.time.Timer.Error)!struct {
        graph: NeighborsList(false),
        timing: OptimizationTiming,
    } {
        var total_timer = try std.time.Timer.start();
        var timer = try std.time.Timer.start();

        const num_nodes = self.neighbors_list.num_nodes;
        const input_degree = self.neighbors_list.num_neighbors_per_node;
        const output_degree = @min(graph_degree, input_degree);

        var output_graph = try NeighborsList(false).init(
            num_nodes,
            output_degree,
            allocator,
        );

        const num_two_hop_neighbors_per_node =
            std.math.mul(
                usize,
                input_degree -| 1,
                input_degree -| 1,
            ) catch return Error.NumNeighborsPerNodeTooLarge;

        const two_hop_neighbors_buffer_size =
            std.math.mul(
                usize,
                self.numThreads(),
                num_two_hop_neighbors_per_node,
            ) catch return Error.NumNeighborsPerNodeTooLarge;
        const two_hop_neighbors_buffer = try allocator.alloc(usize, two_hop_neighbors_buffer_size);
        defer allocator.free(two_hop_neighbors_buffer);

        const reverse_neighbor_counts = try allocator.alloc(usize, num_nodes);
        defer allocator.free(reverse_neighbor_counts);

        const reverse_neighbor_ids = try allocator.alloc(usize, num_nodes * output_degree);
        defer allocator.free(reverse_neighbor_ids);

        timer.reset();
        self.countDetours(
            two_hop_neighbors_buffer,
            num_two_hop_neighbors_per_node,
        );
        const count_detours_ns = timer.read();

        timer.reset();
        self.prune(&output_graph);
        const prune_ns = timer.read();

        timer.reset();
        self.buildReverseGraph(
            &output_graph,
            reverse_neighbor_counts,
            reverse_neighbor_ids,
        );
        const build_reverse_graph_ns = timer.read();

        timer.reset();
        self.combine(
            &output_graph,
            reverse_neighbor_counts,
            reverse_neighbor_ids,
        );
        const combine_ns = timer.read();

        const total_optimization_ns = total_timer.read();

        return .{
            .graph = output_graph,
            .timing = .{
                .count_detours_ns = count_detours_ns,
                .prune_ns = prune_ns,
                .build_reverse_graph_ns = build_reverse_graph_ns,
                .combine_ns = combine_ns,
                .total_optimization_ns = total_optimization_ns,
            },
        };
    }

    /// Number of threads to use for optimization.
    /// Returns 1 if thread_pool is null, otherwise returns the number of threads in the pool.
    inline fn numThreads(self: *const Self) usize {
        return if (self.thread_pool) |pool| pool.threads.len else 1;
    }

    /// Number of blocks for training.
    /// Equal to 0 when `self.num_nodes_per_block` or number of nodes is 0.
    fn numBlocks(self: *const Self) usize {
        return std.math.divCeil(
            usize,
            self.neighbors_list.num_nodes,
            self.num_nodes_per_block,
        ) catch 0;
    }

    /// Number of nodes each thread should process for one block.
    /// Zero when `self.num_nodes_per_block` is 0 or when the thread pool has 0 threads.
    fn numBlockNodesPerThread(self: *const Self) usize {
        return std.math.divCeil(
            usize,
            self.num_nodes_per_block,
            self.numThreads(),
        ) catch 0;
    }

    /// Counts the number of detourable routes for each edge in the graph.
    /// A detourable route exists when there's a path of length 2 between two nodes
    /// that are already neighbors. The count represents how many such paths exist.
    /// For each edge (u, v), counts how many middle nodes w exist such that:
    ///   - w is to the right of v in u's neighbor list (w has lower rank than v)
    ///   - u is to the right of v in w's neighbor list (u has lower rank than v)
    pub fn countDetours(
        self: *Self,
        two_hop_neighbors_buffer: []usize,
        num_two_hop_neighbors_per_node: usize,
    ) void {
        // Reset detour counts to 0 before counting
        const detour_counts: []usize = self.neighbors_list.entries.items(.detour_count);
        @memset(detour_counts, 0);

        const num_blocks = self.numBlocks();

        for (0..num_blocks) |block_id| {
            self.countDetoursBlock(
                block_id,
                two_hop_neighbors_buffer,
                num_two_hop_neighbors_per_node,
            );
        }
    }

    /// Processes a block of nodes for detour counting.
    /// Splits the block across threads if a thread pool is available.
    fn countDetoursBlock(
        self: *Self,
        block_id: usize,
        two_hop_neighbors_buffer: []usize,
        num_two_hop_neighbors_per_node: usize,
    ) void {
        std.debug.assert(two_hop_neighbors_buffer.len == self.numThreads() * num_two_hop_neighbors_per_node);
        const block_start = @min(block_id * self.num_nodes_per_block, self.neighbors_list.num_nodes);
        const block_end = @min(block_start + self.num_nodes_per_block, self.neighbors_list.num_nodes);

        if (self.thread_pool) |pool| {
            self.wait_group.reset();
            const num_block_nodes_per_thread = self.numBlockNodesPerThread();
            for (0..pool.threads.len) |thread_id| {
                const node_id_start = @min(block_start + thread_id * num_block_nodes_per_thread, block_end);
                const node_id_end = @min(node_id_start + num_block_nodes_per_thread, block_end);
                const two_hop_neighbors_buffer_start = thread_id * num_two_hop_neighbors_per_node;
                const two_hop_neighbors_buffer_end = two_hop_neighbors_buffer_start + num_two_hop_neighbors_per_node;
                pool.spawnWg(
                    &self.wait_group,
                    countDetoursThread,
                    .{
                        &self.neighbors_list,
                        node_id_start,
                        node_id_end,
                        two_hop_neighbors_buffer[two_hop_neighbors_buffer_start..two_hop_neighbors_buffer_end],
                    },
                );
            }
            pool.waitAndWork(&self.wait_group);
        } else {
            countDetoursThread(
                &self.neighbors_list,
                block_start,
                block_end,
                two_hop_neighbors_buffer,
            );
        }
    }

    /// Counts detours for a range of nodes in the range [node_id_start, node_id_end).
    fn countDetoursThread(
        neighbors_list: *NeighborsList(true),
        node_id_start: usize,
        node_id_end: usize,
        two_hop_neighbors_buffer: []usize,
    ) void {
        const num_nodes = neighbors_list.num_nodes;
        const degree = neighbors_list.num_neighbors_per_node;
        std.debug.assert(node_id_start <= node_id_end and node_id_end <= neighbors_list.num_nodes);
        std.debug.assert(two_hop_neighbors_buffer.len == (degree -| 1) * (degree -| 1));

        for (node_id_start..node_id_end) |node_id| {
            const neighbor_ids: []const usize = neighbors_list.getEntryFieldSlice(node_id, .neighbor_id);
            const detour_counts: []usize = neighbors_list.getEntryFieldSlice(node_id, .detour_count);

            // For one node, look at all nodes the node can reach in 2 hops by looking at all neighbors of its neighbors.
            // Store these 2-hop neighbors in a buffer for quick lookup later. The buffer will be fully filled.
            // We ignore the farthest neighbor at both hops since it cannot be part of any detour starting from the node.
            for (0..degree -| 1) |neighbor_rank_hop1| {
                const neighbor_idx_hop1 = degree - 1 - neighbor_rank_hop1;
                const neighbor_id_hop1: usize = neighbor_ids[neighbor_idx_hop1];
                const neighbor_ids_hop2: []const usize = neighbors_list.getEntryFieldSlice(neighbor_id_hop1, .neighbor_id);

                for (0..degree -| 1) |neighbor_rank_hop2| {
                    const neighbor_idx_hop2 = degree - 1 - neighbor_rank_hop2;
                    var neighbor_id_hop2: usize = neighbor_ids_hop2[neighbor_idx_hop2];

                    if (neighbor_id_hop2 == node_id) {
                        // Detected bidirectional edge: node_id -> neighbor_id_hop1 -> node_id.
                        // Mark with an invalid neighbor ID to avoid counting it as a detour later.
                        log.debug(
                            "Detected bidirectional edge between node {} and neighbor {}. Marking as invalid in 2-hop neighbors buffer.",
                            .{ node_id, neighbor_id_hop1 },
                        );
                        neighbor_id_hop2 = num_nodes;
                    }

                    // Convert the pair of neighbor ranks (neighbor_rank_hop1, neighbor_rank_hop2) to an index in the buffer.
                    const idx = if (neighbor_rank_hop1 < neighbor_rank_hop2)
                        neighbor_rank_hop2 * neighbor_rank_hop2 + neighbor_rank_hop1
                    else
                        neighbor_rank_hop1 * (neighbor_rank_hop1 + 1) + neighbor_rank_hop2;
                    log.debug(
                        "Storing 2-hop neighbor {} for node {} at buffer index {} (neighbor ranks: {}, {})",
                        .{ neighbor_id_hop2, node_id, idx, neighbor_rank_hop1, neighbor_rank_hop2 },
                    );
                    two_hop_neighbors_buffer[idx] = neighbor_id_hop2;
                }
            }

            for (0..degree) |neighbor_rank| {
                const neighbor_idx = degree - 1 - neighbor_rank;
                const neighbor_id: usize = neighbor_ids[neighbor_idx];
                std.debug.assert(neighbor_id < num_nodes);

                var detour_count: usize = 0;
                // We only count detours with neighbor_rank_hop1 and neighbor_rank_hop2 of at most neighbor_rank - 1,
                // so the largest possible idx in the two-hop neighbors buffer is:
                // (neighbor_rank - 1) * neighbor_rank + (neighbor_rank - 1) = neighbor_rank * neighbor_rank - 1,
                // which is less than neighbor_rank * neighbor_rank.
                for (0..neighbor_rank * neighbor_rank) |idx| {
                    // Invalid neighbor IDs from the buffer will not match any valid neighbor_id,
                    // so they will be effectively ignored in detour counting.
                    if (two_hop_neighbors_buffer[idx] == neighbor_id) {
                        // Found a detour: node_id -> some_neighbor -> neighbor_id.
                        // Increment detour count for the edge between node_id and neighbor_id.
                        detour_count += 1;
                    }
                }
                detour_counts[neighbor_idx] = detour_count;
            }
        }
    }

    /// Select neighbors with the smallest detour counts from the struct's neighbors list (input graph)
    /// and fill the pruned graph with the selected neighbors. The pruned graph
    /// must have the same number of nodes as the input graph,
    /// and cannot have higher graph degree than the input graph.
    fn prune(self: *Self, pruned_graph: *NeighborsList(false)) void {
        const num_blocks = self.numBlocks();
        for (0..num_blocks) |block_id| {
            self.pruneBlock(block_id, pruned_graph);
        }
    }

    /// Processes a block of nodes for pruning.
    /// Splits the block across threads if a thread pool is available.
    fn pruneBlock(
        self: *Self,
        block_id: usize,
        pruned_graph: *NeighborsList(false),
    ) void {
        const block_start = @min(block_id * self.num_nodes_per_block, self.neighbors_list.num_nodes);
        const block_end = @min(block_start + self.num_nodes_per_block, self.neighbors_list.num_nodes);

        if (self.thread_pool) |pool| {
            self.wait_group.reset();
            const num_nodes_per_thread = self.numBlockNodesPerThread();
            for (0..pool.threads.len) |thread_id| {
                const node_id_start = @min(block_start + thread_id * num_nodes_per_thread, block_end);
                const node_id_end = @min(node_id_start + num_nodes_per_thread, block_end);
                // SAFETY: Each thread only touches neighbor data for nodes in the range [node_id_start, node_id_end), so no data races.
                pool.spawnWg(
                    &self.wait_group,
                    pruneThread,
                    .{
                        &self.neighbors_list,
                        pruned_graph,
                        node_id_start,
                        node_id_end,
                    },
                );
            }
            pool.waitAndWork(&self.wait_group);
        } else {
            pruneThread(
                &self.neighbors_list,
                pruned_graph,
                block_start,
                block_end,
            );
        }
    }

    /// Prunes neighbors based on detour counts for a range of nodes using counting-based selection.
    /// Fills output graph with the selected neighbors. Neighbors with smaller detour counts are prioritized for selection.
    /// Output graph must have the same number of nodes as the input graph,
    /// but cannot have higher graph degree than the input graph.
    fn pruneThread(
        input_graph: *NeighborsList(true),
        output_graph: *NeighborsList(false),
        node_id_start: usize,
        node_id_end: usize,
    ) void {
        std.debug.assert(input_graph.num_nodes == output_graph.num_nodes);
        std.debug.assert(node_id_start <= node_id_end and node_id_end <= input_graph.num_nodes);

        const num_nodes = input_graph.num_nodes;
        const input_degree = input_graph.num_neighbors_per_node;
        const output_degree = output_graph.num_neighbors_per_node;
        std.debug.assert(output_degree <= input_degree);

        for (node_id_start..node_id_end) |node_id| {
            const input_neighbor_ids: []const usize = input_graph.getEntryFieldSlice(node_id, .neighbor_id);
            const input_detour_counts: []const usize = input_graph.getEntryFieldSlice(node_id, .detour_count);
            const output_neighbor_ids: []usize = output_graph.getEntryFieldSlice(node_id, .neighbor_id);

            // Number of neighbors already selected for the output graph.
            var num_neighbors_selected: usize = 0;
            // The detour count value we're currently collecting neighbors for.
            // We iterate from 0 upward to select neighbors with the smallest detour counts first.
            var current_detour_count: usize = 0;

            // Continue selecting neighbors until we have output_degree neighbors.
            outer: while (num_neighbors_selected < output_degree) {
                // The minimum detour count that is greater than current_detour_count.
                // Used to advance to the next detour count value after we finish scanning for the current value.
                var next_detour_count: usize = std.math.maxInt(usize);

                log.debug(
                    "[Node {}] Looking for neighbors with detour count {}. Currently selected {} neighbors",
                    .{ node_id, current_detour_count, num_neighbors_selected },
                );
                for (0..input_degree) |neighbor_idx| {
                    const candidate_detour_count = input_detour_counts[neighbor_idx];
                    // Update next_detour_count if we find a larger detour count
                    if (candidate_detour_count > current_detour_count) {
                        next_detour_count = @min(candidate_detour_count, next_detour_count);
                    }
                    // Only proceed with neighbors that have the current detour count.
                    if (candidate_detour_count != current_detour_count) continue;

                    const candidate_neighbor_id = input_neighbor_ids[neighbor_idx];
                    std.debug.assert(candidate_neighbor_id < num_nodes);

                    // Skip if the neighbor ID is a duplicate. Otherwise add this neighbor to the output.
                    if (std.mem.indexOfScalar(
                        usize,
                        output_neighbor_ids[0..num_neighbors_selected],
                        candidate_neighbor_id,
                    ) != null) continue;

                    output_neighbor_ids[num_neighbors_selected] = candidate_neighbor_id;
                    num_neighbors_selected += 1;

                    if (num_neighbors_selected >= output_degree) break :outer;
                }

                // Early exit: no valid edges remain in the input graph.
                // This can happen if the input graph has too many duplicates.
                if (next_detour_count == std.math.maxInt(usize)) break;

                // Move to the next detour count value.
                log.debug(
                    "[Node {}] Finished looking for neighbors with detour count {}. Found {} neighbors so far. Moving on to detour count {}.",
                    .{ node_id, current_detour_count, num_neighbors_selected, next_detour_count },
                );
                current_detour_count = next_detour_count;
            }
        }
    }

    /// Builds a reverse graph where each node stores which nodes have it as a neighbor.
    /// Reverse neighbor IDs are stored in row-major order, with dimensions `[num_nodes][degree]`.
    /// Since some nodes have less than `degree` reverse neighbors, the reverse neighbor counts buffer
    /// is used to keep track of how many valid reverse neighbors each node has, capped at `degree`.
    /// For each edge `(u, v)` in `pruned_graph`, adds `u` to `reverse_neighbors[v]`'s list.
    fn buildReverseGraph(
        self: *Self,
        pruned_graph: *NeighborsList(false),
        reverse_neighbor_counts: []usize,
        reverse_neighbor_ids: []usize,
    ) void {
        const num_blocks = self.numBlocks();

        // Reset reverse neighbor counts to 0 before building the reverse graph.
        @memset(reverse_neighbor_counts, 0);
        for (0..num_blocks) |block_id| {
            self.buildReverseGraphBlock(
                block_id,
                pruned_graph,
                reverse_neighbor_counts,
                reverse_neighbor_ids,
            );
        }
    }

    /// Processes a block of nodes for building the reverse graph.
    /// Splits the block across threads if a thread pool is available.
    fn buildReverseGraphBlock(
        self: *Self,
        block_id: usize,
        pruned_graph: *NeighborsList(false),
        reverse_neighbor_counts: []usize,
        reverse_neighbor_ids: []usize,
    ) void {
        const block_start = @min(block_id * self.num_nodes_per_block, self.neighbors_list.num_nodes);
        const block_end = @min(block_start + self.num_nodes_per_block, self.neighbors_list.num_nodes);

        if (self.thread_pool) |pool| {
            self.wait_group.reset();
            const num_nodes_per_thread = self.numBlockNodesPerThread();
            for (0..pool.threads.len) |thread_id| {
                const node_id_start = @min(block_start + thread_id * num_nodes_per_thread, block_end);
                const node_id_end = @min(node_id_start + num_nodes_per_thread, block_end);
                // SAFETY: Multiple threads can write to the same reverse count/buffer concurrently,
                // but atomicRmw ensures no data races
                pool.spawnWg(
                    &self.wait_group,
                    buildReverseGraphThread,
                    .{
                        pruned_graph,
                        reverse_neighbor_counts,
                        reverse_neighbor_ids,
                        node_id_start,
                        node_id_end,
                    },
                );
            }
            pool.waitAndWork(&self.wait_group);
        } else {
            buildReverseGraphThread(
                pruned_graph,
                reverse_neighbor_counts,
                reverse_neighbor_ids,
                block_start,
                block_end,
            );
        }
    }

    /// Builds reverse graph for a range of source nodes.
    /// For each edge `(src_node, dst_node)` in `pruned_graph` where `src_node` is in the range
    /// `[node_id_start, node_id_end)`, adds `src_node` to `reverse_neighbors[dst_node]`'s list.
    /// Reverse neighbor IDs are stored in row-major order, with dimensions `[num_nodes][degree]`.
    /// Reverse neighbor counts buffer should be initialized to 0 before calling this function.
    fn buildReverseGraphThread(
        pruned_graph: *NeighborsList(false),
        reverse_neighbor_counts: []usize,
        reverse_neighbor_ids: []usize,
        node_id_start: usize,
        node_id_end: usize,
    ) void {
        const num_nodes = pruned_graph.num_nodes;
        const degree = pruned_graph.num_neighbors_per_node;
        std.debug.assert(reverse_neighbor_counts.len == num_nodes);
        std.debug.assert(reverse_neighbor_ids.len == num_nodes * degree);
        std.debug.assert(node_id_start <= node_id_end and node_id_end <= num_nodes);

        // Iterate over source nodes in this thread's range
        for (node_id_start..node_id_end) |node_id_src| {
            const neighbor_ids: []const usize = pruned_graph.getEntryFieldSlice(node_id_src, .neighbor_id);
            for (neighbor_ids) |node_id_dst| {
                std.debug.assert(node_id_dst < num_nodes);
                // Atomically increment the counter and get the previous value (position to write)
                const slot = @atomicRmw(usize, &reverse_neighbor_counts[node_id_dst], .Add, 1, .monotonic);
                // Only write if there's room; silently drop if buffer is full
                if (slot < degree) {
                    log.debug("Adding reverse neighbor {d} for node {d} in slot {d}", .{ node_id_src, node_id_dst, slot });
                    reverse_neighbor_ids[node_id_dst * degree + slot] = node_id_src;
                } else {
                    log.debug("Reverse neighbor buffer full for node {d}, cannot add neighbor {d}", .{ node_id_dst, node_id_src });
                }
            }
        }
    }

    /// Combines the edges from pruned graph and reverse graph by mutating pruned graph in-place.
    /// Keep pruned neighbors first, then adds reverse neighbors (avoiding duplicates).
    fn combine(
        self: *Self,
        pruned_graph: *NeighborsList(false),
        reverse_neighbor_counts: []const usize,
        reverse_neighbor_ids: []const usize,
    ) void {
        const num_blocks = self.numBlocks();
        for (0..num_blocks) |block_id| {
            self.combineBlock(
                block_id,
                pruned_graph,
                reverse_neighbor_counts,
                reverse_neighbor_ids,
            );
        }
    }

    /// Processes a block of nodes for combining graphs.
    /// Splits the block across threads if a thread pool is available.
    fn combineBlock(
        self: *Self,
        block_id: usize,
        pruned_graph: *NeighborsList(false),
        reverse_neighbor_counts: []const usize,
        reverse_neighbor_ids: []const usize,
    ) void {
        const block_start = @min(block_id * self.num_nodes_per_block, self.neighbors_list.num_nodes);
        const block_end = @min(block_start + self.num_nodes_per_block, self.neighbors_list.num_nodes);

        if (self.thread_pool) |pool| {
            self.wait_group.reset();
            const num_nodes_per_thread = self.numBlockNodesPerThread();
            for (0..pool.threads.len) |thread_id| {
                const node_id_start = @min(block_start + thread_id * num_nodes_per_thread, block_end);
                const node_id_end = @min(node_id_start + num_nodes_per_thread, block_end);
                // SAFETY: Each thread only touches neighbor data for nodes in the range [node_id_start, node_id_end), so no data races.
                pool.spawnWg(
                    &self.wait_group,
                    combineThread,
                    .{
                        pruned_graph,
                        reverse_neighbor_counts,
                        reverse_neighbor_ids,
                        node_id_start,
                        node_id_end,
                    },
                );
            }
            pool.waitAndWork(&self.wait_group);
        } else {
            combineThread(
                pruned_graph,
                reverse_neighbor_counts,
                reverse_neighbor_ids,
                block_start,
                block_end,
            );
        }
    }

    /// Combines the edges from pruned and reverse graphs for a range of nodes, and store them on the pruned graph.
    /// Reverse neighbor IDs are stored in row-major order, with dimensions `[num_nodes][degree]`.
    /// Reverse neighbor counts store the number of reverse neighbors for each node, capped at pruned graph's degree.
    /// For each node:
    ///   1. Protect first half of pruned neighbors (lowest detour counts)
    ///   2. Fill remaining slots with reverse edges, avoiding duplicates
    ///   3. If there are still empty slots, the remaining pruned neighbors (higher detour counts) are kept.
    fn combineThread(
        pruned_graph: *NeighborsList(false),
        reverse_neighbor_counts: []const usize,
        reverse_neighbor_ids: []const usize,
        node_id_start: usize,
        node_id_end: usize,
    ) void {
        const num_nodes = pruned_graph.num_nodes;
        const degree = pruned_graph.num_neighbors_per_node;
        std.debug.assert(reverse_neighbor_counts.len == num_nodes);
        std.debug.assert(reverse_neighbor_ids.len == num_nodes * degree);
        std.debug.assert(node_id_start <= node_id_end and node_id_end <= num_nodes);

        for (node_id_start..node_id_end) |node_id| {
            const pruned_neighbor_ids: []usize = pruned_graph.getEntryFieldSlice(node_id, .neighbor_id);

            // The current index in the pruned graph's neighbor list where we will write the next neighbor ID.
            // No duplicate check needed: the pruned graph should not have duplicate edges as it's pruned from the neighbors list.
            var current_neighbor_idx: usize = degree / 2; // Protect first half
            log.debug(
                "[Node {}] {} protected neighbors before combine: {any}",
                .{ node_id, current_neighbor_idx, pruned_neighbor_ids[0..current_neighbor_idx] },
            );

            // Fill non-protected region with reverse edges.
            // Duplicate check needed: reverse edges can duplicate protected edges
            // when there are bidirectional connections.
            const num_reverse_neighbors = @min(reverse_neighbor_counts[node_id], degree);
            for (0..num_reverse_neighbors) |neighbor_idx| {
                if (current_neighbor_idx >= degree) break;
                const neighbor_id = reverse_neighbor_ids[node_id * degree + neighbor_idx];
                std.debug.assert(neighbor_id < num_nodes);

                // Only add if it's not a duplicate
                if (std.mem.indexOfScalar(
                    usize,
                    pruned_neighbor_ids,
                    neighbor_id,
                ) == null) {
                    pruned_neighbor_ids[current_neighbor_idx] = neighbor_id;
                    current_neighbor_idx += 1;
                    log.debug("Added reverse neighbor {d} for node {d}", .{ neighbor_id, node_id });
                } else {
                    log.debug("Skipping duplicate neighbor {d} for node {d}", .{ neighbor_id, node_id });
                }
            }

            log.debug(
                "[Node {}] {} reverse neighbors added: {any}",
                .{ node_id, current_neighbor_idx - degree / 2, pruned_neighbor_ids[degree / 2 .. current_neighbor_idx] },
            );
        }
    }
};

test "count detours" {
    // Adjacency list representation of the graph
    // Node 0: neighbors 1, 2, 3
    // Node 1: neighbors 0, 2, 3
    // Node 2: neighbors 0, 1, 3
    // Node 3: neighbors 0, 1, 2
    var neighbor_ids = [_]usize{
        1, 2, 3,
        0, 2, 3,
        0, 1, 3,
        0, 1, 2,
    };
    var detour_counts = [_]usize{undefined} ** 12;

    const neighbors_list = Optimizer.NeighborsList(true){
        .num_nodes = 4,
        .num_neighbors_per_node = 3,
        .entries = mod_soa_slice.SoaSlice(Optimizer.NeighborsList(true).Entry){
            .ptrs = [_][*]u8{
                @ptrCast(&neighbor_ids),
                @ptrCast(&detour_counts),
            },
            .len = 12,
        },
    };

    var optimizer = Optimizer.init(
        neighbors_list,
        null,
        2,
    );
    const num_two_hop_neighbors_per_node = (3 -| 1) * (3 -| 1);
    const two_hop_neighbors_buffer = try std.testing.allocator.alloc(usize, num_two_hop_neighbors_per_node * optimizer.numThreads());
    defer std.testing.allocator.free(two_hop_neighbors_buffer);
    optimizer.countDetours(
        two_hop_neighbors_buffer,
        num_two_hop_neighbors_per_node,
    );

    try std.testing.expectEqualSlices(
        usize,
        &[_]usize{
            // 0-1: 0-2-1, 0-3-1
            // 0-2: 0-3-2
            // 1-2: 1-3-2
            2, 1, 0,
            0, 1, 0,
            0, 0, 0,
            0, 0, 0,
        },
        &detour_counts,
    );
}

test "prune - output degree" {
    // Create a simple graph where we can verify prune behavior
    // Node 0: neighbors 1, 2, 3 with detour counts 5, 1, 3
    // Node 1: neighbors 0, 2, 3 with detour counts 2, 4, 1
    // Node 2: neighbors 0, 1, 3 with detour counts 1, 2, 5
    // Node 3: neighbors 0, 1, 2 with detour counts 3, 1, 4
    // The detour counts don't have to be valid for this test since we're just testing the pruning logic.
    var neighbor_ids = [_]usize{
        1, 2, 3,
        0, 2, 3,
        0, 1, 3,
        0, 1, 2,
    };
    var detour_counts = [_]usize{
        5, 1, 3,
        2, 4, 1,
        1, 2, 5,
        3, 1, 4,
    };

    const input_graph = Optimizer.NeighborsList(true){
        .num_nodes = 4,
        .num_neighbors_per_node = 3,
        .entries = mod_soa_slice.SoaSlice(Optimizer.NeighborsList(true).Entry){
            .ptrs = [_][*]u8{
                @ptrCast(&neighbor_ids),
                @ptrCast(&detour_counts),
            },
            .len = 12,
        },
    };

    var optimizer = Optimizer.init(input_graph, null, 2);

    // Prune to output degree 2
    var output_graph = try Optimizer.NeighborsList(false).init(4, 2, std.testing.allocator);
    defer output_graph.deinit(std.testing.allocator);

    optimizer.prune(&output_graph);

    // Verify output has correct degree
    try std.testing.expectEqual(@as(usize, 2), output_graph.num_neighbors_per_node);

    // Verify neighbors are sorted by detour count (ascending)
    // Node 0: neighbors 2 (dc=1), 3 (dc=3), 1 (dc=5) -> first 2 should be 2, 3
    // Node 1: neighbors 3 (dc=1), 0 (dc=2), 2 (dc=4) -> first 2 should be 3, 0
    // Node 2: neighbors 0 (dc=1), 1 (dc=2), 3 (dc=5) -> first 2 should be 0, 1
    // Node 3: neighbors 1 (dc=1), 0 (dc=3), 2 (dc=4) -> first 2 should be 1, 0
    const expected = [_]usize{
        2, 3,
        3, 0,
        0, 1,
        1, 0,
    };
    const actual: []const usize = output_graph.entries.items(.neighbor_id);
    try std.testing.expectEqualSlices(usize, &expected, actual);
}

test "combine - correct degree" {
    // Create a pruned graph with 4 neighbors per node for 6 nodes.
    var neighbor_ids = [_]usize{
        1, 2, 3, 4, // node 0
        0, 2, 3, 4, // node 1
        0, 1, 3, 4, // node 2
        0, 1, 2, 4, // node 3
        0, 1, 2, 3, // node 4
        0, 1, 2, 3, // node 5
    };
    var detour_counts_void = [_]void{undefined} ** 24;

    var pruned_graph = Optimizer.NeighborsList(false){
        .num_nodes = 6,
        .num_neighbors_per_node = 4,
        .entries = mod_soa_slice.SoaSlice(Optimizer.NeighborsList(false).Entry){
            .ptrs = [2][*]u8{
                @ptrCast(&neighbor_ids),
                @ptrCast(&detour_counts_void),
            },
            .len = 24,
        },
    };

    // The input graph for the Optimizer instance doesn't matter for combine(),
    // but an Optimizer value is required to call the method.
    const input_graph = Optimizer.NeighborsList(true){
        .num_nodes = 6,
        .num_neighbors_per_node = 4,
        .entries = undefined,
    };
    var optimizer = Optimizer.init(input_graph, null, 1);

    // Reverse neighbor IDs and counts
    const reverse_ids = [_]usize{
        // 4 left out (due to degree cap)
        1,         2,         3,         5,
        // 5 left out (due to degree cap)
        0,         2,         3,         4,
        // 5 left out (due to degree cap)
        0,         1,         3,         4,
        // 5 left out (due to degree cap)
        0,         1,         2,         4,
        // no neighbors left out
        0,         1,         2,         3,
        undefined, undefined, undefined, undefined,
    };
    const reverse_counts = [_]usize{ 4, 4, 4, 4, 4, 0 };

    // Call combine: it mutates `pruned_graph` in-place.
    optimizer.combine(&pruned_graph, &reverse_counts, &reverse_ids);

    const expected = [_]usize{
        // Only node 0 has revese neighbor id 5 sneaking in
        1, 2, 5, 4,
        // All other nodes are unaffected since their reverse neighbors
        // are duplicates of pruned neighbors, or they have no reverse neighbors.
        0, 2, 3, 4,
        0, 1, 3, 4,
        0, 1, 2, 4,
        0, 1, 2, 3,
        0, 1, 2, 3,
    };
    const actual: []const usize = pruned_graph.entries.items(.neighbor_id);
    try std.testing.expectEqualSlices(usize, &expected, actual);
}

test "edge case - single node graph" {
    const num_nodes: usize = 1;
    const input_degree: usize = 1;
    const output_degree: usize = 1;

    var neighbor_ids = [_]usize{0}; // Node 0's neighbor (itself)
    var detour_counts = [_]usize{0};

    const input_graph = Optimizer.NeighborsList(true){
        .num_nodes = num_nodes,
        .num_neighbors_per_node = input_degree,
        .entries = mod_soa_slice.SoaSlice(Optimizer.NeighborsList(true).Entry){
            .ptrs = [2][*]u8{
                @ptrCast(&neighbor_ids),
                @ptrCast(&detour_counts),
            },
            .len = num_nodes * input_degree,
        },
    };

    var optimizer = Optimizer.init(input_graph, null, 1);

    var output_graph = try optimizer.optimize(output_degree, std.testing.allocator);
    defer output_graph.deinit(std.testing.allocator);

    // Just verify it doesn't crash and has correct degree
    try std.testing.expectEqual(output_degree, output_graph.num_neighbors_per_node);
}
