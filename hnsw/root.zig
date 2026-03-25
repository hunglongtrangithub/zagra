//! Zig bindings for hnswlib, a library for approximate nearest neighbor search.
//!
//! This module provides two index implementations:
//! - `Bruteforce`: Simple brute-force search for baseline comparisons
//! - `HierarchicalIndex`: Hierarchical Navigable Small World (HNSW) graph-based index
//!
//! # Example
//!
//! ```zig
//! const hnsw = @import("hnsw");
//! const allocator = std.heap.page_allocator;
//!
//! // Create a brute-force index
//! var index = try hnsw.Bruteforce.create(128, 10000);
//! defer index.deinit();
//!
//! // Add vectors
//! try index.addPoint(my_vector[0..], label);
//!
//! // Search
//! var result = try index.searchKnnAlloc(allocator, query[0..], 10);
//! defer result.deinit(allocator);
//!
//! for (result.items(.label), result.items(.distance)) |label, dist| {
//!     std.debug.print("label={}, dist={}\n", .{ label, dist });
//! }
//! ```

const std = @import("std");

/// Result codes returned by the C API.
pub const hnsw_res = enum(i32) {
    /// Operation succeeded
    HNSW_SUCCESS = 0,
    /// Invalid argument (null pointer, bad parameters)
    HNSW_EINVAL = -1,
    /// Allocation failure
    HNSW_ENOMEM = -2,
    /// Runtime error (capacity exceeded, file I/O error)
    HNSW_ERUNTIME = -3,
    /// Unknown/uncaught exception
    HNSW_EUNKNOWN = -4,
};

/// Base Zig error set mapping from `hnsw_res` codes.
pub const Error = error{
    /// Allocation failure (maps to HNSW_ENOMEM)
    OutOfMemory,
    /// Invalid argument (maps to HNSW_EINVAL)
    InvalidArgument,
    /// Runtime error (maps to HNSW_ERUNTIME)
    RuntimeError,
    /// Unknown error (maps to HNSW_EUNKNOWN)
    Unknown,
};

/// Maps an `hnsw_res` return code to a Zig error.
pub fn map_res(res: hnsw_res) Error!void {
    switch (res) {
        .HNSW_SUCCESS => return,
        .HNSW_ENOMEM => return Error.OutOfMemory,
        .HNSW_EINVAL => return Error.InvalidArgument,
        .HNSW_ERUNTIME => return Error.RuntimeError,
        .HNSW_EUNKNOWN => return Error.Unknown,
    }
}

/// A single search result item.
pub const SearchItem = struct {
    /// The label of the nearest neighbor
    label: usize,
    /// The distance to the nearest neighbor
    distance: f32,
};

/// Container for k-nearest neighbor search results using `std.MultiArrayList`.
///
/// # Example
///
/// ```zig
/// var result = try SearchResult.init(allocator, 10);
/// defer result.deinit(allocator);
///
/// // Access results
/// for (result.items(.label), result.items(.distance)) |label, dist| {
///     std.debug.print("label={}, dist={}\n", .{ label, dist });
/// }
/// ```
pub const SearchResult = std.MultiArrayList(SearchItem);

const bf = @import("bruteforce.zig");
const h = @import("hnsw.zig");

/// Label type used by the index (matches hnswlib::labeltype)
pub const hnsw_label_t = bf.hnsw_label_t;

/// Brute-force index implementation.
///
/// Provides exact nearest neighbor search with O(n) complexity per query.
/// Useful as a baseline for comparing approximate methods.
pub const Bruteforce = bf.Bruteforce;

/// Hierarchical Navigable Small World (HNSW) index.
///
/// Provides fast approximate nearest neighbor search with configurable
/// accuracy/speed tradeoffs.
pub const HierarchicalIndex = h.Index;
