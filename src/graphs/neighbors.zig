const std = @import("std");

const types = @import("../types.zig");

pub const InitError = error{
    /// The specified number of nodes is too large to fit in isize.
    NumberOfNodesTooLarge,
    /// The specified number of neighbors causes an overflow when multiplied by number of nodes.
    NumberOfNeighborsTooLarge,
};

/// A cache-friendly list of max heaps for storing k-nearest neighbors.
///
/// This structure stores multiple heaps in a contiguous row-major layout,
/// where each heap represents the k-nearest neighbors of a point in the dataset.
/// The row-major organization improves cache locality when iterating through
/// all points' neighbor lists sequentially.
///
/// Generic over the distance type T that is supported in `types.ElemType`,
/// and whether to store new/old flags for NN-Descent.
pub fn NeighborHeapList(comptime T: type, comptime store_flags: bool) type {
    const elem_type = types.ElemType.fromZigType(T) orelse
        @compileError("Unsupported element type: " ++ @typeName(T));

    return struct {
        /// One entry per heap slot. Entries are heapified by distance.
        /// Stored internally as structure-of-arrays by MultiArrayList.
        pub const Entry = struct {
            /// -1 represents an empty neighbor slot. All valid IDs are in [0, num_nodes).
            /// NOTE: using isize for point IDs. This is okay since the number of vectors
            /// in a dataset is no more than std.math.maxInt(isize).
            neighbor_id: isize,

            /// This is the key used for heap ordering (max heap: largest distance at root).
            distance: T,

            /// New/old flags for the NN-Descent algorithm.
            ///
            /// SEMANTICS: This flag represents an "exploration lease."
            /// - `true` (NEW): The neighbor is a fresh discovery. It acts as an active
            ///   "search agent." It must be introduced to all other neighbors in the
            ///   next local join to find even better connections.
            /// - `false` (OLD): The neighbor is a "passive contact." It has already
            ///   participated in a local join as a 'NEW' node. We have already explored
            ///   its immediate neighborhood, so we skip redundant comparisons.
            ///
            /// LIFECYCLE:
            /// 1. BORN: Set to `true` when first inserted into the NeighborList.
            /// 2. CONSUMED: Set to `false` immediately after being sampled into a
            ///    `new_candidates` pool for a local join.
            /// 3. RE-ARMED: If a local join replaces this slot with an even closer
            ///    point, the new point starts its life as `true`.
            is_new: if (store_flags) bool else void,
        };

        /// Row-major storage of all heap entries.
        /// Indexing: i * num_neighbors + j
        entries: std.MultiArrayList(Entry).Slice,

        /// Total number of points (number of heaps).
        num_nodes: usize,

        /// Number of neighbors per point (k).
        num_neighbors_per_node: usize,

        const Self = @This();
        pub const EMPTY_ID: isize = -1;

        /// Initializes a new instance with the specified number of nodes and neighbors per node.
        /// All neighbor IDs are set to -1 (empty), distances to max value, and is_new flags to true.
        /// `num_nodes * num_neighbors` must not overflow `usize`.
        pub fn init(
            num_nodes: usize,
            num_neighbors_per_node: usize,
            allocator: std.mem.Allocator,
        ) (InitError || std.mem.Allocator.Error)!Self {
            if (num_nodes > std.math.maxInt(isize)) return InitError.NumberOfNodesTooLarge;
            const total_size, const overflow = @mulWithOverflow(num_nodes, num_neighbors_per_node);
            if (overflow != 0) return InitError.NumberOfNeighborsTooLarge;

            // Allocate contiguous memory for all entries
            var entries = std.MultiArrayList(Entry).empty;
            try entries.setCapacity(allocator, total_size);

            const entries_slice = entries.slice();
            memsetBuffers(entries_slice);

            return Self{
                .entries = entries_slice,
                .num_nodes = num_nodes,
                .num_neighbors_per_node = num_neighbors_per_node,
            };
        }

        /// Resets all neighbor entries to their initial state:
        /// neighbor IDs to -1, distances to max value, and is_new flags to true.
        pub fn reset(self: *Self) void {
            memsetBuffers(self.entries);
        }

        fn memsetBuffers(entries: std.MultiArrayList(Entry).Slice) void {
            const max_dist = switch (elem_type) {
                .Int32 => std.math.maxInt(T),
                .Float, .Half => std.math.floatMax(T),
            };

            // Reset fields independently
            @memset(entries.items(.neighbor_id), EMPTY_ID);
            @memset(entries.items(.distance), max_dist);
            if (store_flags) @memset(entries.items(.is_new), true);
        }

        /// Deinitializes the HeapList, freeing its allocated memory.
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.entries.deinit(allocator);
        }

        /// Checks and updates the heap for the given node with a new neighbor entry.
        /// The neighbor is added only if its distance is smaller than the current maximum distance in the heap,
        /// and the neighbor ID is not already present in the heap.
        /// Returns `true` if the neighbor was added, `false` otherwise.
        pub fn tryAddNeighbor(
            self: *Self,
            node_id: usize,
            neighbor_entry: Entry,
        ) bool {
            std.debug.assert(node_id < self.num_nodes);

            // Calculate the start index for the heaps of the specified node
            const heap_start = node_id * self.num_neighbors_per_node;

            // If the new distance is not smaller than the max in the heap, reject it
            const max_distance = self.entries.items(.distance)[heap_start];
            if (neighbor_entry.distance >= max_distance) return false;

            // Check for duplicate neighbor IDs
            const neighbor_ids: []isize = self.entries.items(.neighbor_id)[heap_start .. heap_start + self.num_neighbors_per_node];
            if (std.mem.indexOfScalar(isize, neighbor_ids, neighbor_entry.neighbor_id) != null) {
                return false;
            }

            // Replace the maximum entry with the new neighbor entry
            self.replaceMaxEntry(node_id, neighbor_entry);
            return true;
        }

        /// Replaces the maximum entry in the heap for the specified node with a new entry,
        /// and restores the max-heap property for that heap.
        fn replaceMaxEntry(self: *Self, node_id: usize, new_entry: Entry) void {
            std.debug.assert(node_id < self.num_nodes);
            std.debug.assert(new_entry.neighbor_id < self.num_nodes);

            // Get the start index for the heap of the specified node
            const heap_start = node_id * self.num_neighbors_per_node;
            const distance_heap: []T = self.entries.items(.distance)[heap_start .. heap_start + self.num_neighbors_per_node];

            // Heapify down from the root to restore max-heap property
            // Set the new entry's initial index to 0 (root)
            var entry_idx: usize = 0;
            while (true) {
                const left_child_idx = 2 * entry_idx + 1;
                const right_child_idx = 2 * entry_idx + 2;

                // Find the largest among entry and its children
                var largest_idx = entry_idx;

                if (left_child_idx < self.num_neighbors_per_node and
                    distance_heap[left_child_idx] > distance_heap[largest_idx])
                {
                    largest_idx = left_child_idx;
                }

                if (right_child_idx < self.num_neighbors_per_node and
                    distance_heap[right_child_idx] > distance_heap[largest_idx])
                {
                    largest_idx = right_child_idx;
                }

                // If entry is largest, heap property is satisfied
                if (largest_idx == entry_idx) break;

                // Set entry to largest child
                self.entries.set(
                    heap_start + entry_idx,
                    self.entries.get(heap_start + largest_idx),
                );

                entry_idx = largest_idx;
            }

            // Place the new entry at the final position
            self.entries.set(heap_start + entry_idx, new_entry);
        }

        inline fn getEntryIndex(self: *const Self, node_id: usize, neighbor_idx: usize) usize {
            std.debug.assert(node_id < self.num_nodes);
            std.debug.assert(neighbor_idx < self.num_neighbors_per_node);
            return node_id * self.num_neighbors_per_node + neighbor_idx;
        }

        /// Retrieves a mutable pointer to the specified field of the neighbor entry
        /// for the given node and neighbor index.
        /// SAFETY:
        /// - The returned pointer is only valid as long as the underlying data exists
        /// - Do not store the pointer beyond the data's lifetime
        pub fn getEntryFieldPtr(
            self: *Self,
            node_id: usize,
            neighbor_idx: usize,
            comptime field: std.meta.FieldEnum(Entry),
        ) *std.meta.fieldInfo(Entry, field).type {
            const index = self.getEntryIndex(node_id, neighbor_idx);
            // items(field) returns the specific array buffer for that field
            return &self.entries.items(field)[index];
        }

        /// Retrieves a slice of the specified field for all neighbor entries of the given node.
        pub fn getEntryFieldSlice(
            self: *const Self,
            node_id: usize,
            comptime field: std.meta.FieldEnum(Entry),
        ) []std.meta.fieldInfo(Entry, field).type {
            std.debug.assert(node_id < self.num_nodes);
            const start = node_id * self.num_neighbors_per_node;
            return self.entries.items(field)[start .. start + self.num_neighbors_per_node];
        }

        /// Retrieves the neighbor entry for the specified node and neighbor index.
        pub fn getEntry(self: *const Self, node_id: usize, neighbor_idx: usize) Entry {
            const index = self.getEntryIndex(node_id, neighbor_idx);
            return self.entries.get(index);
        }
    };
}
