const std = @import("std");

const mod_types = @import("../types.zig");
const mod_soa_slice = @import("soa_slice.zig");
const mod_nn_descent = @import("nn_descent.zig");

pub const Optimizer = struct {
    pub const Entry = struct {
        /// Node ID of a neighbor. Inherited from `NeighborHeapList`.
        neighbor_id: usize,
        /// Number of detourable routes for the edge between the node and this neighbor.
        detour_count: usize,
    };

    pub const NeighborsList = struct {
        /// Total number of points (n).
        num_nodes: usize,
        /// Number of neighbors per point (k).
        num_neighbors_per_node: usize,
        /// Row-major storage of all entries.
        /// Indexing: i * num_neighbors_per_node + j
        entries: mod_soa_slice.SoaSlice(Entry),

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

    /// Entries for all neighbors of all nodes, stored in row-major order.
    /// Neighbor slots for every node are all valid (non-empty) and there are no self-loops (a node cannot be its own neighbor).
    /// Neighbors for a node are arranged in descending order of distance, thus descending order of rank (this is guaranteed by the caller).
    neighbors_list: NeighborsList,
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
        neighbors_list: NeighborsList,
        thread_pool: ?*std.Thread.Pool,
        num_nodes_per_block: usize,
    ) Self {
        var optimizer: Self = undefined;
        optimizer.neighbors_list = neighbors_list;
        optimizer.thread_pool = thread_pool;
        optimizer.num_nodes_per_block = @min(num_nodes_per_block, neighbors_list.num_nodes);

        return optimizer;
    }

    /// Optimizes the graph by removing redundant edges based on the number of detourable routes.
    /// Final graph degree should be less than or equal to the initial number of neighbors per node.
    pub fn optimize(self: *Self, graph_degree: usize) void {
        std.debug.assert(graph_degree <= self.neighbors_list.num_neighbors_per_node);

        self.countDetours();
        // TODO: Prune neighbors based on detour counts and graph_degree.
    }

    /// Number of threads to use for optimization. Returns 1 if thread_pool is null, otherwise returns the number of threads in the pool.
    inline fn numThreads(self: *const Self) usize {
        return if (self.thread_pool) |pool| pool.threads.len else 1;
    }

    /// Number of blocks for training.
    /// Equal to 0 when num_nodes_per_block or number of nodes is 0.
    fn numBlocks(self: *const Self) usize {
        return std.math.divCeil(
            usize,
            self.neighbors_list.num_nodes,
            self.num_nodes_per_block,
        ) catch 0;
    }

    /// Number of nodes each thread should process for one block.
    /// Zero when num_nodes_per_block is 0 or when the thread pool has 0 threads.
    fn numBlockNodesPerThread(self: *const Self) usize {
        return std.math.divCeil(
            usize,
            self.num_nodes_per_block,
            if (self.thread_pool) |pool| pool.threads.len else 1,
        ) catch 0;
    }

    fn countDetours(self: *Self) void {
        // Reset detour counts to 0 before counting
        const detour_counts: []usize = self.neighbors_list.entries.items(.detour_count);
        @memset(detour_counts, 0);

        const num_blocks = self.numBlocks();

        for (0..num_blocks) |block_id| {
            self.countDetoursBlock(block_id);
        }
    }

    fn countDetoursBlock(self: *Self, block_id: usize) void {
        const block_start = @min(block_id * self.num_nodes_per_block, self.neighbors_list.num_nodes);
        const block_end = @min(block_start + self.num_nodes_per_block, self.neighbors_list.num_nodes);

        if (self.thread_pool) |pool| {
            self.wait_group.reset();
            const num_block_nodes_per_thread = self.numBlockNodesPerThread();
            for (0..pool.threads.len) |thread_id| {
                const node_id_start = @min(block_start + thread_id * num_block_nodes_per_thread, block_end);
                const node_id_end = @min(node_id_start + num_block_nodes_per_thread, block_end);
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

    fn countDetoursThread(neighbors_list: *NeighborsList, node_id_start: usize, node_id_end: usize) void {
        std.debug.assert(node_id_start <= node_id_end and node_id_end <= neighbors_list.num_nodes);

        // TODO: make the detour counting faster by using data structures
        for (node_id_start..node_id_end) |node_id| {
            const neighbor_ids: []const usize = neighbors_list.getEntryFieldSlice(node_id, .neighbor_id);
            const detours_counts: []usize = neighbors_list.getEntryFieldSlice(node_id, .detour_count);
            for (neighbor_ids, 0..) |neighbor_id, idx| {
                // We look at middle nodes whose ranks are less than the current neighbor_id's rank.
                // These nodes are on the right of the current neighbor in the node_id's neighbors list.
                for (neighbor_ids[idx + 1 ..]) |middle_node_id| {
                    const node_ids: []const usize = neighbors_list.getEntryFieldSlice(middle_node_id, .neighbor_id);
                    if (std.mem.indexOfScalar(
                        usize,
                        // If neighbor_id exists in the middle_node_id's neighbors list,
                        // it must have a smaller rank than its rank in the node_id's neighbors list.
                        // Thus we only look at the right side of the middle_node_id's neighbors list
                        // right after the neighbor_id's rank in the node_id's neighbors list (idx).
                        node_ids[idx + 1 ..],
                        neighbor_id,
                    ) != null) {
                        detours_counts[idx] += 1;
                    }
                }
            }
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

    const neighbors_list = Optimizer.NeighborsList{
        .num_nodes = 4,
        .num_neighbors_per_node = 3,
        .entries = mod_soa_slice.SoaSlice(Optimizer.Entry){
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
