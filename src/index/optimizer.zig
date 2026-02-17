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

    /// Total number of points (n).
    num_nodes: usize,
    /// Number of neighbors per point (k).
    num_neighbors_per_node: usize,
    /// Row-major storage of all entries.
    /// Indexing: i * num_neighbors_per_node + j
    entries: mod_soa_slice.SoaSlice(Entry),

    /// Thread pool for multi-threaded operations.
    /// `null` when requested number of threads is <= 1.
    thread_pool: ?*std.Thread.Pool,
    /// Wait group for synchronizing threads.
    wait_group: std.Thread.WaitGroup,

    const Self = @This();

    /// Optimizes the graph by removing redundant edges based on the number of detourable routes.
    /// Final graph degree should be less than or equal to the initial number of neighbors per node.
    pub fn optimize(self: *Self, graph_degree: usize) void {
        std.debug.assert(graph_degree <= self.num_neighbors_per_node);
    }
};

test "optimizer" {
    _ = Optimizer;
}
