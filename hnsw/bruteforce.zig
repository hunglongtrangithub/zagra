//! Zig FFI bindings for the bruteforce C API wrapper around hnswlib::BruteforceSearch<float>.
//!
//! Provides exact nearest neighbor search for baseline comparisons.

const std = @import("std");
const root = @import("root.zig");

const hnsw_label_t = usize;
const bruteforce_t = opaque {};

extern fn bruteforce_new_l2(dim: usize, max_elements: usize) ?*bruteforce_t;
extern fn bruteforce_load_l2(path: [*]const u8, dim: usize) ?*bruteforce_t;
extern fn bruteforce_free(idx: ?*bruteforce_t) void;
extern fn bruteforce_add_point(
    idx: *bruteforce_t,
    data: [*]const f32,
    label: hnsw_label_t,
) root.hnsw_res;
extern fn bruteforce_remove_point(idx: *bruteforce_t, label: hnsw_label_t) root.hnsw_res;
extern fn bruteforce_search_knn(
    idx: *bruteforce_t,
    query: [*]const f32,
    k: usize,
    out_labels: [*]hnsw_label_t,
    out_distances: [*]f32,
    out_result_count: *usize,
) root.hnsw_res;
extern fn bruteforce_save(idx: *bruteforce_t, path: [*]const u8) root.hnsw_res;

/// Brute-force nearest neighbor index.
///
/// Performs exact search with O(n) complexity per query. Use this as a
/// baseline to compare against approximate methods like HNSW.
///
/// # Example
///
/// ```zig
/// var index = try Bruteforce.create(128, 10000);
/// defer index.deinit();
///
/// // Add a vector
/// try index.addPoint(my_vector[0..], 42);
///
/// // Search
/// var result = try index.searchKnnAlloc(allocator, query[0..], 10);
/// defer result.deinit(allocator);
/// ```
pub const Bruteforce = struct {
    /// The underlying C handle
    handle: *bruteforce_t,
    /// Dimensionality of vectors in this index
    dim: usize,

    const Self = @This();

    /// Create a new L2 brute-force index.
    ///
    /// - `dim`: Vector dimensionality (number of floats per vector)
    /// - `max_elements`: Maximum number of vectors the index can hold
    ///
    /// Returns error.OutOfMemory if allocation fails.
    pub fn create(dim: usize, max_elements: usize) error{OutOfMemory}!Self {
        if (bruteforce_new_l2(dim, max_elements)) |h| {
            return Self{ .handle = h, .dim = dim };
        } else {
            return error.OutOfMemory;
        }
    }

    /// Load an index from disk.
    ///
    /// `path` must point to a NUL-terminated C string.
    pub fn load(path: [*]const u8, dim: usize) error{OutOfMemory}!Self {
        if (bruteforce_load_l2(path, dim)) |h| {
            return Self{ .handle = h, .dim = dim };
        } else {
            return error.OutOfMemory;
        }
    }

    /// Free resources held by the index.
    pub fn deinit(self: *Self) void {
        bruteforce_free(self.handle);
    }

    /// Add a vector to the index.
    ///
    /// `data` must be a slice of length exactly `dim`.
    /// `label` is an arbitrary identifier for this vector.
    pub fn addPoint(self: *Self, data: []const f32, label: usize) root.Error!void {
        if (data.len != self.dim) return root.Error.InvalidArgument;
        const res = bruteforce_add_point(self.handle, data.ptr, label);
        try root.map_res(res);
    }

    /// Remove a vector from the index by its label.
    ///
    /// Returns success even if the label doesn't exist.
    pub fn removePoint(self: *Self, label: usize) root.Error!void {
        const res = bruteforce_remove_point(self.handle, label);
        try root.map_res(res);
    }

    /// Search for k nearest neighbors.
    ///
    /// `query` must be a slice of length exactly `dim`.
    /// Caller must provide output buffers with length >= k.
    ///
    /// Returns the number of results written (0..k).
    pub fn searchKnn(
        self: *Self,
        query: []const f32,
        k: usize,
        out_labels: []usize,
        out_distances: []f32,
    ) root.Error!usize {
        if (query.len != self.dim) return root.Error.InvalidArgument;
        if (k > out_labels.len or k > out_distances.len) return root.Error.BufferTooSmall;

        var out_count: usize = 0;
        const res = bruteforce_search_knn(
            self.handle,
            query.ptr,
            k,
            out_labels.ptr,
            out_distances.ptr,
            &out_count,
        );
        try root.map_res(res);
        return out_count;
    }

    /// Search for k nearest neighbors, allocating output buffers.
    ///
    /// `query` must be a slice of length exactly `dim`.
    /// Returns a `SearchResult` with allocated buffers. Caller must call `deinit()`.
    ///
    /// # Example
    ///
    /// ```zig
    /// var result = try index.searchKnnAlloc(allocator, query[0..], 10);
    /// defer result.deinit(allocator);
    ///
    /// for (result.items(.label), result.items(.distance)) |label, dist| {
    ///     std.debug.print("label={}, dist={}\n", .{ label, dist });
    /// }
    /// ```
    pub fn searchKnnAlloc(
        self: *Self,
        allocator: std.mem.Allocator,
        query: []const f32,
        k: usize,
    ) root.Error!root.SearchResult {
        if (query.len != self.dim) return root.Error.InvalidArgument;

        var result = root.SearchResult{};
        try result.resize(allocator, k);
        errdefer result.deinit(allocator);

        const res = bruteforce_search_knn(
            self.handle,
            query.ptr,
            k,
            result.items(.label).ptr,
            result.items(.distance).ptr,
            &result.len,
        );
        try root.map_res(res);
        return result;
    }

    /// Save the index to disk.
    ///
    /// `path` must point to a NUL-terminated C string.
    pub fn save(self: *Self, path: [*]const u8) root.Error!void {
        const res = bruteforce_save(self.handle, path);
        try root.map_res(res);
    }
};
