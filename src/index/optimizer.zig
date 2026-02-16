const std = @import("std");

const mod_types = @import("../types.zig");
const mod_nn_descent = @import("nn_descent.zig");

pub const Optimizer = struct {
    pub const Entry = struct {
        /// Node ID of a neighbor. Inherited from `NeighborHeapList`.
        neighbor_id: isize,
        /// Number of detourable routes for the edge between the node and this neighbor.
        detourable_count: usize,
    };

    /// Total number of points (n).
    num_nodes: usize,
    /// Number of neighbors per point (k).
    num_neighbors_per_node: usize,
    /// Row-major storage of all entries.
    /// Indexing: i * num_neighbors_per_node + j
    entries: std.MultiArrayList(Entry).Slice,

    /// Thread pool for multi-threaded operations.
    /// `null` when requested number of threads is <= 1.
    thread_pool: ?*std.Thread.Pool,
    /// Wait group for synchronizing threads.
    wait_group: std.Thread.WaitGroup,

    const Self = @This();
};

test "optimizer" {
    _ = Optimizer;
}
