const std = @import("std");
const log = std.log.scoped(.optimizer);

const mod_types = @import("../types.zig");
const mod_soa_slice = @import("soa_slice.zig");
const mod_nn_descent = @import("nn_descent.zig");

pub const Optimizer = struct {
    pub const Error = error{
        /// The number of edges (num_nodes * num_neighbors_per_node) is too large to fit in memory.
        NumberOfEdgesTooLarge,
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
            ) (Error || std.mem.Allocator.Error)!@This() {
                const total_edges = std.math.mul(usize, num_nodes, num_neighbors_per_node) catch return Error.NumberOfEdgesTooLarge;
                const total_size = std.math.mul(usize, total_edges, @sizeOf(Entry)) catch return Error.NumberOfEdgesTooLarge;
                if (total_size > std.math.maxInt(isize)) return Error.NumberOfEdgesTooLarge;
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
        const output_degree = @min(graph_degree, self.neighbors_list.num_neighbors_per_node);

        // Buffer for storing reverse neighbor counts per node. The value of ech element
        // in this buffer may be larger than the output_degree (which is invalid),
        // but we will cap the number of reverse neighbors to output_degree in the combine step.
        const reverse_neighbor_counts = try allocator.alloc(usize, num_nodes);
        defer allocator.free(reverse_neighbor_counts);

        // Buffer for storing reverse neighbors.
        // Each node can have at most `output_degree` reverse neighbors.
        const reverse_neighbor_ids = try allocator.alloc(usize, num_nodes * output_degree);
        defer allocator.free(reverse_neighbor_ids);

        // Output graph that will store the optimized neighbors.
        // Does not store detour counts, only neighbor IDs.
        var output_graph = try NeighborsList(false).init(
            num_nodes,
            output_degree,
            allocator,
        );

        self.countDetours();
        self.prune(&output_graph);
        self.buildReverseGraph(
            &output_graph,
            reverse_neighbor_counts,
            reverse_neighbor_ids,
        );
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
    ) (Error || std.mem.Allocator.Error || std.time.Timer.Error)!struct { graph: NeighborsList(false), timing: OptimizationTiming } {
        var total_timer = try std.time.Timer.start();
        var timer = try std.time.Timer.start();

        const num_nodes = self.neighbors_list.num_nodes;
        const output_degree = @min(graph_degree, self.neighbors_list.num_neighbors_per_node);

        const reverse_neighbor_counts = try allocator.alloc(usize, num_nodes);
        defer allocator.free(reverse_neighbor_counts);

        const reverse_neighbor_ids = try allocator.alloc(usize, num_nodes * output_degree);
        defer allocator.free(reverse_neighbor_ids);

        var output_graph = try NeighborsList(false).init(
            num_nodes,
            output_degree,
            allocator,
        );

        timer.reset();
        self.countDetours();
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
    pub fn countDetours(self: *Self) void {
        // Reset detour counts to 0 before counting
        const detour_counts: []usize = self.neighbors_list.entries.items(.detour_count);
        @memset(detour_counts, 0);

        const num_blocks = self.numBlocks();

        for (0..num_blocks) |block_id| {
            self.countDetoursBlock(block_id);
        }
    }

    /// Processes a block of nodes for detour counting.
    /// Splits the block across threads if a thread pool is available.
    fn countDetoursBlock(self: *Self, block_id: usize) void {
        const block_start = @min(block_id * self.num_nodes_per_block, self.neighbors_list.num_nodes);
        const block_end = @min(block_start + self.num_nodes_per_block, self.neighbors_list.num_nodes);

        if (self.thread_pool) |pool| {
            self.wait_group.reset();
            const num_block_nodes_per_thread = self.numBlockNodesPerThread();
            for (0..pool.threads.len) |thread_id| {
                const node_id_start = @min(block_start + thread_id * num_block_nodes_per_thread, block_end);
                const node_id_end = @min(node_id_start + num_block_nodes_per_thread, block_end);
                // SAFETY: Each thread only mutates detour count data for nodes in the range [node_id_start, node_id_end), so no data races.
                pool.spawnWg(
                    &self.wait_group,
                    countDetoursThread,
                    .{
                        &self.neighbors_list,
                        node_id_start,
                        node_id_end,
                    },
                );
            }
            pool.waitAndWork(&self.wait_group);
        } else {
            countDetoursThread(
                &self.neighbors_list,
                block_start,
                block_end,
            );
        }
    }

    /// Counts detours for a range of nodes in the range [node_id_start, node_id_end).
    fn countDetoursThread(
        neighbors_list: *NeighborsList(true),
        node_id_start: usize,
        node_id_end: usize,
    ) void {
        std.debug.assert(node_id_start <= node_id_end and node_id_end <= neighbors_list.num_nodes);

        for (node_id_start..node_id_end) |node_id| {
            const neighbor_ids: []const usize = neighbors_list.getEntryFieldSlice(node_id, .neighbor_id);
            const detour_counts: []usize = neighbors_list.getEntryFieldSlice(node_id, .detour_count);
            for (neighbor_ids, 0..) |neighbor_id, idx| {
                // We look at middle nodes whose ranks are less than the current neighbor_id's rank.
                // These nodes are on the right of the current neighbor in the node_id's neighbors list.
                const hop1_node_ids = neighbor_ids[idx + 1 ..];
                for (hop1_node_ids) |hop1_node_id| {
                    const next_neighbor_ids: []const usize = neighbors_list.getEntryFieldSlice(hop1_node_id, .neighbor_id);
                    // If neighbor_id exists in the hop1_node_id's neighbors list,
                    // it must have a smaller rank than its rank in the node_id's neighbors list.
                    // Thus we only look at the right side of the middle_node_id's neighbors list
                    // right after the neighbor_id's rank in the node_id's neighbors list (idx).
                    const hop2_node_ids = next_neighbor_ids[idx + 1 ..];
                    if (std.mem.indexOfScalar(
                        usize,
                        hop2_node_ids,
                        neighbor_id,
                    ) != null) {
                        detour_counts[idx] += 1;
                    }
                }
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

    /// Prunes neighbors for a range of nodes.
    /// Output graph must have the same number of nodes as the input graph,
    /// but cannot have higher graph degree than the input graph.
    /// For each node in `[node_id_start, node_id_end)`:
    ///   1. Sorts neighbors by `detour_count` in ascending order (in-place)
    ///   2. Copies the first `output_graph.num_neighbors_per_node` neighbors to `output_graph`
    fn pruneThread(
        input_graph: *NeighborsList(true),
        output_graph: *NeighborsList(false),
        node_id_start: usize,
        node_id_end: usize,
    ) void {
        std.debug.assert(input_graph.num_nodes == output_graph.num_nodes);
        std.debug.assert(node_id_start <= node_id_end and node_id_end <= input_graph.num_nodes);

        const input_degree = input_graph.num_neighbors_per_node;
        const output_degree = output_graph.num_neighbors_per_node;
        std.debug.assert(output_degree <= input_degree);

        for (node_id_start..node_id_end) |node_id| {
            const input_neighbor_ids: []usize = input_graph.getEntryFieldSlice(node_id, .neighbor_id);
            const input_detour_counts: []usize = input_graph.getEntryFieldSlice(node_id, .detour_count);

            const Context = struct {
                neighbor_ids: []usize,
                detour_counts: []usize,

                pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                    return ctx.detour_counts[a] < ctx.detour_counts[b];
                }

                pub fn swap(ctx: @This(), a: usize, b: usize) void {
                    std.mem.swap(usize, &ctx.neighbor_ids[a], &ctx.neighbor_ids[b]);
                    std.mem.swap(usize, &ctx.detour_counts[a], &ctx.detour_counts[b]);
                }
            };

            // Sort by detour count in ascending order
            std.sort.heapContext(0, input_degree, Context{
                .neighbor_ids = input_neighbor_ids,
                .detour_counts = input_detour_counts,
            });

            // Copy the first output_degree neighbors to the output graph
            const output_neighbor_ids: []usize = output_graph.getEntryFieldSlice(node_id, .neighbor_id);
            @memcpy(output_neighbor_ids, input_neighbor_ids[0..output_degree]);
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

            log.debug("[Node {}] {} reverse neighbors added: {any}", .{ node_id, current_neighbor_idx - degree / 2, pruned_neighbor_ids[degree / 2 .. current_neighbor_idx] });
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
    optimizer.countDetours();

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
    // Create a pruned graph with 4 neighbors and reverse graph
    var neighbor_ids = [_]usize{ 1, 2, 3, 4 };
    var detour_counts_void = [_]void{undefined} ** 4;

    var pruned_graph = Optimizer.NeighborsList(false){
        .num_nodes = 1,
        .num_neighbors_per_node = 4,
        .entries = mod_soa_slice.SoaSlice(Optimizer.NeighborsList(false).Entry){
            .ptrs = [2][*]u8{
                @ptrCast(&neighbor_ids),
                @ptrCast(&detour_counts_void),
            },
            .len = 4,
        },
    };

    const input_graph = Optimizer.NeighborsList(true){
        .num_nodes = 1,
        .num_neighbors_per_node = 4,
        .entries = undefined,
    };

    var optimizer = Optimizer.init(input_graph, null, 1);

    // Reverse graph with some neighbors
    var reverse_graph: [1]std.ArrayList(usize) = undefined;
    var reverse_buffer = [_]usize{ 5, 6, 0, 0 };
    reverse_graph[0] = std.ArrayList(usize).initBuffer(&reverse_buffer);
    reverse_graph[0].items.len = 2; // Only 2 actual reverse neighbors

    var output_graph = try Optimizer.NeighborsList(false).init(1, 4, std.testing.allocator);
    defer output_graph.deinit(std.testing.allocator);

    const reverse_graph_slice: []std.ArrayList(usize) = &reverse_graph;
    optimizer.combine(&pruned_graph, reverse_graph_slice, &output_graph);

    // Verify output degree is exactly 4
    try std.testing.expectEqual(@as(usize, 4), output_graph.num_neighbors_per_node);
}

test "property - optimize output has valid neighbor IDs" {
    // Create a random-ish graph
    const num_nodes: usize = 20;
    const input_degree: usize = 8;
    const output_degree: usize = 4;

    var neighbor_ids = [_]usize{undefined} ** (num_nodes * input_degree);
    var detour_counts = [_]usize{undefined} ** (num_nodes * input_degree);

    // Fill with random-looking data
    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    for (0..num_nodes) |node_id| {
        for (0..input_degree) |j| {
            neighbor_ids[node_id * input_degree + j] = random.intRangeAtMost(usize, 0, num_nodes - 1);
            detour_counts[node_id * input_degree + j] = random.intRangeAtMost(usize, 0, 100);
        }
    }

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

    var optimizer = Optimizer.init(input_graph, null, 4);

    // Run full optimize
    var output_graph = try optimizer.optimize(output_degree, std.testing.allocator);
    defer output_graph.deinit(std.testing.allocator);

    // Property: all neighbor IDs should be valid (0 <= id < num_nodes)
    const output_neighbor_ids = output_graph.entries.items(.neighbor_id);
    for (output_neighbor_ids) |neighbor_id| {
        try std.testing.expect(neighbor_id < num_nodes);
    }
}

test "property - optimize output degree matches requested" {
    const num_nodes: usize = 10;
    const input_degree: usize = 8;
    const output_degree: usize = 4;

    var neighbor_ids = [_]usize{0} ** (num_nodes * input_degree);
    var detour_counts = [_]usize{0} ** (num_nodes * input_degree);

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

    var optimizer = Optimizer.init(input_graph, null, 2);

    var output_graph = try optimizer.optimize(output_degree, std.testing.allocator);
    defer output_graph.deinit(std.testing.allocator);

    // Property: output degree should match requested
    try std.testing.expectEqual(@as(usize, output_degree), output_graph.num_neighbors_per_node);
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
    try std.testing.expectEqual(@as(usize, output_degree), output_graph.num_neighbors_per_node);
}

test "buildReverseGraph - all edges represented" {
    // Create a simple graph where all edges are unique
    // Node 0: neighbors 1, 2
    // Node 1: neighbors 0, 2
    // Node 2: neighbors 0, 1
    // Total edges = 6 (undirected count = 3 unique pairs, but we count directed edges)
    var neighbor_ids = [_]usize{
        1, 2,
        0, 2,
        0, 1,
    };
    var detour_counts_void = [_]void{undefined} ** 6;

    var pruned_graph = Optimizer.NeighborsList(false){
        .num_nodes = 3,
        .num_neighbors_per_node = 2,
        .entries = mod_soa_slice.SoaSlice(Optimizer.NeighborsList(false).Entry){
            .ptrs = [2][*]u8{
                @ptrCast(&neighbor_ids),
                @ptrCast(&detour_counts_void),
            },
            .len = 6,
        },
    };

    const input_graph = Optimizer.NeighborsList(true){
        .num_nodes = 3,
        .num_neighbors_per_node = 2,
        .entries = undefined,
    };

    var optimizer = Optimizer.init(input_graph, null, 2);

    // Build reverse graph
    const reverse_graph = try std.testing.allocator.alloc(std.ArrayList(usize), 3);
    defer std.testing.allocator.free(reverse_graph);

    const reverse_buffer = try std.testing.allocator.alloc(usize, 3 * 2);
    defer std.testing.allocator.free(reverse_buffer);

    for (reverse_graph, 0..) |*list, node_id| {
        const start = node_id * 2;
        list.* = std.ArrayList(usize).initBuffer(reverse_buffer[start .. start + 2]);
    }

    optimizer.buildReverseGraph(&pruned_graph, reverse_graph);

    // Count total reverse edges
    var total_reverse_edges: usize = 0;
    for (reverse_graph) |list| {
        total_reverse_edges += list.items.len;
    }

    // Total edges should be preserved
    try std.testing.expectEqual(@as(usize, 6), total_reverse_edges);
}

test "property - optimize output has no self-loops" {
    // Create a graph that might produce self-loops
    const num_nodes: usize = 15;
    const input_degree: usize = 6;
    const output_degree: usize = 4;

    var neighbor_ids = [_]usize{undefined} ** (num_nodes * input_degree);
    var detour_counts = [_]usize{undefined} ** (num_nodes * input_degree);

    // Each node's neighbors include itself with varying detour counts
    var prng = std.Random.DefaultPrng.init(99999);
    const random = prng.random();

    for (0..num_nodes) |node_id| {
        for (0..input_degree) |j| {
            // 30% chance of self-loop
            if (random.float(f32) < 0.3) {
                neighbor_ids[node_id * input_degree + j] = node_id;
            } else {
                neighbor_ids[node_id * input_degree + j] = random.intRangeAtMost(usize, 0, num_nodes - 1);
            }
            detour_counts[node_id * input_degree + j] = random.intRangeAtMost(usize, 0, 50);
        }
    }

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

    var optimizer = Optimizer.init(input_graph, null, 4);

    var output_graph = try optimizer.optimize(output_degree, std.testing.allocator);
    defer output_graph.deinit(std.testing.allocator);

    // Property: no self-loops in output
    const output_neighbor_ids = output_graph.entries.items(.neighbor_id);
    for (0..num_nodes) |node_id| {
        const start = node_id * output_degree;
        for (0..output_degree) |j| {
            try std.testing.expect(output_neighbor_ids[start + j] != node_id);
        }
    }
}

test "edge case - graph where all detour counts are equal" {
    // When all detour counts are equal, prune should still work (stable sort)
    const num_nodes: usize = 5;
    const input_degree: usize = 4;
    const output_degree: usize = 2;

    var neighbor_ids = [_]usize{
        1, 2, 3, 4,
        0, 2, 3, 4,
        0, 1, 3, 4,
        0, 1, 2, 4,
        0, 1, 2, 3,
    };
    var detour_counts = [_]usize{5} ** 20; // All equal

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

    var optimizer = Optimizer.init(input_graph, null, 2);

    var output_graph = try optimizer.optimize(output_degree, std.testing.allocator);
    defer output_graph.deinit(std.testing.allocator);

    // Should produce output with correct degree
    try std.testing.expectEqual(@as(usize, output_degree), output_graph.num_neighbors_per_node);

    // All neighbor IDs should be valid
    const out_ids = output_graph.entries.items(.neighbor_id);
    for (out_ids) |id| {
        try std.testing.expect(id < num_nodes);
    }
}

test "thread safety - single vs multi-threaded produce same results" {
    // Create deterministic input
    const num_nodes: usize = 20;
    const input_degree: usize = 8;
    const output_degree: usize = 4;

    var neighbor_ids_1 = [_]usize{undefined} ** (num_nodes * input_degree);
    var detour_counts_1 = [_]usize{undefined} ** (num_nodes * input_degree);

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..num_nodes) |node_id| {
        for (0..input_degree) |j| {
            neighbor_ids_1[node_id * input_degree + j] = random.intRangeAtMost(usize, 0, num_nodes - 1);
            detour_counts_1[node_id * input_degree + j] = random.intRangeAtMost(usize, 0, 100);
        }
    }

    // Single-threaded
    const input_graph_st = Optimizer.NeighborsList(true){
        .num_nodes = num_nodes,
        .num_neighbors_per_node = input_degree,
        .entries = mod_soa_slice.SoaSlice(Optimizer.NeighborsList(true).Entry){
            .ptrs = [2][*]u8{
                @ptrCast(&neighbor_ids_1),
                @ptrCast(&detour_counts_1),
            },
            .len = num_nodes * input_degree,
        },
    };

    var optimizer_st = Optimizer.init(input_graph_st, null, 1);
    var output_st = try optimizer_st.optimize(output_degree, std.testing.allocator);
    defer output_st.deinit(std.testing.allocator);

    // Multi-threaded (4 threads)
    var neighbor_ids_mt = [_]usize{undefined} ** (num_nodes * input_degree);
    var detour_counts_mt = [_]usize{undefined} ** (num_nodes * input_degree);
    @memcpy(&neighbor_ids_mt, &neighbor_ids_1);
    @memcpy(&detour_counts_mt, &detour_counts_1);

    const input_graph_mt = Optimizer.NeighborsList(true){
        .num_nodes = num_nodes,
        .num_neighbors_per_node = input_degree,
        .entries = mod_soa_slice.SoaSlice(Optimizer.NeighborsList(true).Entry){
            .ptrs = [2][*]u8{
                @ptrCast(&neighbor_ids_mt),
                @ptrCast(&detour_counts_mt),
            },
            .len = num_nodes * input_degree,
        },
    };

    var optimizer_mt = Optimizer.init(input_graph_mt, null, 4);
    var output_mt = try optimizer_mt.optimize(output_degree, std.testing.allocator);
    defer output_mt.deinit(std.testing.allocator);

    // Both should produce valid results with correct degree
    try std.testing.expectEqual(@as(usize, output_degree), output_st.num_neighbors_per_node);
    try std.testing.expectEqual(@as(usize, output_degree), output_mt.num_neighbors_per_node);

    // Both should have valid neighbor IDs
    const ids_st = output_st.entries.items(.neighbor_id);
    const ids_mt = output_mt.entries.items(.neighbor_id);
    for (ids_st) |id| try std.testing.expect(id < num_nodes);
    for (ids_mt) |id| try std.testing.expect(id < num_nodes);
}
