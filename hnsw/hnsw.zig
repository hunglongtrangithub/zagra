//! Zig FFI bindings for the HNSW C API wrapper around hnswlib::HierarchicalNSW<float>.
//!
//! Provides fast approximate nearest neighbor search using a hierarchical
//! navigable small world graph structure.

const std = @import("std");
const root = @import("root.zig");

/// Label type used by the index.
const hnsw_label_t = usize;

/// Opaque handle type for HNSW index.
pub const hnsw_index_t = opaque {};

extern fn hnsw_new_l2(
    dim: usize,
    max_elements: usize,
    M: usize,
    ef_construction: usize,
    random_seed: usize,
    allow_replace_deleted: bool,
) ?*hnsw_index_t;
extern fn hnsw_load_l2(path: [*]const u8, dim: usize) ?*hnsw_index_t;
extern fn hnsw_free(idx: ?*hnsw_index_t) void;

extern fn hnsw_add_point(
    idx: *hnsw_index_t,
    data: [*]const f32,
    label: usize,
    replace_deleted: bool,
) root.hnsw_res;
extern fn hnsw_mark_delete(idx: *hnsw_index_t, label: usize) root.hnsw_res;
extern fn hnsw_unmark_delete(idx: *hnsw_index_t, label: usize) root.hnsw_res;
extern fn hnsw_resize_index(idx: *hnsw_index_t, new_max_elements: usize) root.hnsw_res;

extern fn hnsw_search_knn(
    idx: *hnsw_index_t,
    query: [*]const f32,
    k: usize,
    out_labels: [*]usize,
    out_distances: [*]f32,
    out_result_count: *usize,
) root.hnsw_res;
extern fn hnsw_search_knn_with_ef(
    idx: *hnsw_index_t,
    query: [*]const f32,
    k: usize,
    ef: usize,
    out_labels: [*]usize,
    out_distances: [*]f32,
    out_result_count: *usize,
) root.hnsw_res;

extern fn hnsw_set_ef(idx: *hnsw_index_t, ef: usize) root.hnsw_res;
extern fn hnsw_get_ef(idx: *hnsw_index_t, out_ef: *usize) root.hnsw_res;
extern fn hnsw_get_max_elements(idx: *hnsw_index_t, out_max_elements: *usize) root.hnsw_res;
extern fn hnsw_get_current_count(idx: *hnsw_index_t, out_current_count: *usize) root.hnsw_res;
extern fn hnsw_get_deleted_count(idx: *hnsw_index_t, out_deleted_count: *usize) root.hnsw_res;
extern fn hnsw_get_data_size(idx: *hnsw_index_t, out_data_size: *usize) root.hnsw_res;
extern fn hnsw_get_data_by_label(
    idx: *hnsw_index_t,
    label: usize,
    out_buffer: [*]u8,
    out_buffer_len: usize,
) root.hnsw_res;

extern fn hnsw_save(idx: *hnsw_index_t, path: [*]const u8) root.hnsw_res;
extern fn hnsw_set_num_threads(idx: *hnsw_index_t, num_threads: i32) root.hnsw_res;

fn map_res(res: root.hnsw_res) root.Error!void {
    return root.map_res(res);
}

/// Hierarchical Navigable Small World (HNSW) index.
///
/// Provides fast approximate nearest neighbor search with configurable
/// accuracy/speed tradeoffs.
///
/// # Example
///
/// ```zig
/// var index = try Index.create(
///     128,            // dim
///     100000,         // max_elements
///     16,             // M: connections per layer
///     200,            // ef_construction: build-time search depth
///     42,             // random_seed
///     false,          // allow_replace_deleted
/// );
/// defer index.deinit();
///
/// // Add vectors
/// try index.addPoint(my_vector[0..], 42, false);
///
/// // Search
/// var result = try index.searchKnnAlloc(allocator, query[0..], 10);
/// defer result.deinit(allocator);
/// ```
pub const Index = struct {
    /// The underlying C handle
    handle: *hnsw_index_t,
    /// Dimensionality of vectors in this index
    dim: usize,

    /// Create a new HNSW index for L2 distance search.
    ///
    /// - `dim`: Vector dimensionality (number of floats per vector)
    /// - `max_elements`: Maximum number of vectors the index can hold
    /// - `M`: Number of bi-directional links per node (higher = better accuracy, more memory)
    /// - `ef_construction`: Search depth during construction (higher = better accuracy, slower build)
    /// - `random_seed`: Seed for the random number generator
    /// - `allow_replace_deleted`: If true, deleted slots can be reused
    pub fn create(
        dim: usize,
        max_elements: usize,
        M: usize,
        ef_construction: usize,
        random_seed: usize,
        allow_replace_deleted: bool,
    ) error{OutOfMemory}!Index {
        if (hnsw_new_l2(
            dim,
            max_elements,
            M,
            ef_construction,
            random_seed,
            allow_replace_deleted,
        )) |h| {
            return Index{ .handle = h, .dim = dim };
        } else {
            return error.OutOfMemory;
        }
    }

    /// Load an index from disk.
    ///
    /// `path` must point to a NULL-terminated C string.
    pub fn load(path: [*]const u8, dim: usize) error{LoadFailed}!Index {
        if (hnsw_load_l2(path, dim)) |h| {
            return Index{ .handle = h, .dim = dim };
        } else {
            return error.LoadFailed;
        }
    }

    /// Free resources held by the index.
    pub fn deinit(self: *Index) void {
        hnsw_free(self.handle);
    }

    /// Add a vector to the index.
    ///
    /// `data` must be a slice of length exactly `dim`.
    /// `label` is an arbitrary identifier for this vector.
    /// `replace_deleted`: If true, reuse slots of previously deleted elements.
    pub fn addPoint(
        self: *Index,
        data: []const f32,
        label: usize,
        replace_deleted: bool,
    ) root.Error!void {
        if (data.len != self.dim) return root.Error.InvalidArgument;
        const res = hnsw_add_point(
            self.handle,
            data.ptr,
            label,
            replace_deleted,
        );
        try map_res(res);
    }

    /// Mark a vector as deleted (soft delete).
    ///
    /// The vector is excluded from search results but not removed from memory.
    pub fn markDelete(self: *Index, label: usize) root.Error!void {
        const res = hnsw_mark_delete(self.handle, label);
        try map_res(res);
    }

    /// Unmark a deleted vector (restore it).
    pub fn unmarkDelete(self: *Index, label: usize) root.Error!void {
        const res = hnsw_unmark_delete(self.handle, label);
        try map_res(res);
    }

    /// Resize the index capacity.
    ///
    /// `new_max_elements` must be >= current element count.
    pub fn resize(self: *Index, new_max_elements: usize) root.Error!void {
        const res = hnsw_resize_index(self.handle, new_max_elements);
        try map_res(res);
    }

    /// Search for k nearest neighbors using default ef.
    ///
    /// `query` must be a slice of length exactly `dim`.
    /// Caller must provide output buffers with length >= k.
    ///
    /// Returns the number of results written (0..k).
    pub fn searchKnn(
        self: *Index,
        query: []const f32,
        k: usize,
        out_labels: []usize,
        out_distances: []f32,
    ) (error{BufferTooSmall} || root.Error)!usize {
        if (query.len != self.dim) return root.Error.InvalidArgument;
        if (k > out_labels.len or k > out_distances.len) return error.BufferTooSmall;

        var out_count: usize = 0;
        const res = hnsw_search_knn(
            self.handle,
            query.ptr,
            k,
            out_labels.ptr,
            out_distances.ptr,
            &out_count,
        );
        try map_res(res);
        return out_count;
    }

    /// Search for k nearest neighbors with custom ef parameter.
    ///
    /// `ef` controls search breadth (higher = more accurate, slower).
    /// `query` must be a slice of length exactly `dim`.
    /// Caller must provide output buffers with length >= k.
    ///
    /// Returns the number of results written (0..k).
    pub fn searchKnnWithEf(
        self: *Index,
        query: []const f32,
        k: usize,
        ef: usize,
        out_labels: []usize,
        out_distances: []f32,
    ) (error{BufferTooSmall} || root.Error)!usize {
        if (query.len != self.dim) return root.Error.InvalidArgument;
        if (k > out_labels.len or k > out_distances.len) return error.BufferTooSmall;

        var out_count: usize = 0;
        const res = hnsw_search_knn_with_ef(
            self.handle,
            query.ptr,
            k,
            ef,
            out_labels.ptr,
            out_distances.ptr,
            &out_count,
        );
        try map_res(res);
        return out_count;
    }

    /// Search for k nearest neighbors, allocating output buffers.
    ///
    /// `query` must be a slice of length exactly `dim`.
    /// Returns a `SearchResult` with allocated buffers. Caller must call `deinit()`.
    pub fn searchKnnAlloc(
        self: *Index,
        allocator: std.mem.Allocator,
        query: []const f32,
        k: usize,
    ) root.Error!root.SearchResult {
        if (query.len != self.dim) return root.Error.InvalidArgument;

        var result = root.SearchResult.empty;
        try result.resize(allocator, k);
        errdefer result.deinit(allocator);

        const res = hnsw_search_knn(
            self.handle,
            query.ptr,
            k,
            result.items(.label).ptr,
            result.items(.distance).ptr,
            &result.len,
        );
        try map_res(res);
        return result;
    }

    /// Search for k nearest neighbors with custom ef, allocating output buffers.
    pub fn searchKnnWithEfAlloc(
        self: *Index,
        allocator: std.mem.Allocator,
        query: []const f32,
        k: usize,
        ef: usize,
    ) root.HnswError!root.SearchResult {
        if (self.handle == null) return root.HnswError.InvalidHandle;
        if (query.len != self.dim) return root.HnswError.InvalidArgument;

        var result = root.SearchResult.empty;
        try result.resize(allocator, k);
        errdefer result.deinit(allocator);

        const res = hnsw_search_knn_with_ef(
            self.handle,
            query.ptr,
            k,
            ef,
            result.items(.label).ptr,
            result.items(.distance).ptr,
            &result.len,
        );
        try map_res(res);
        return result;
    }

    /// Set the ef parameter for subsequent searches.
    pub fn setEf(self: *Index, ef: usize) root.Error!void {
        const res = hnsw_set_ef(self.handle, ef);
        try map_res(res);
    }

    /// Get the current ef parameter.
    pub fn getEf(self: *Index) root.Error!usize {
        var out_ef: usize = 0;
        const res = hnsw_get_ef(self.handle, &out_ef);
        try map_res(res);
        return out_ef;
    }

    /// Get the maximum number of elements the index can hold.
    pub fn getMaxElements(self: *Index) root.Error!usize {
        var v: usize = 0;
        const res = hnsw_get_max_elements(self.handle, &v);
        try map_res(res);
        return v;
    }

    /// Get the current number of elements in the index.
    pub fn getCurrentCount(self: *Index) root.Error!usize {
        var v: usize = 0;
        const res = hnsw_get_current_count(self.handle, &v);
        try map_res(res);
        return v;
    }

    /// Get the number of deleted elements in the index.
    pub fn getDeletedCount(self: *Index) root.Error!usize {
        var v: usize = 0;
        const res = hnsw_get_deleted_count(self.handle, &v);
        try map_res(res);
        return v;
    }

    /// Get the data size per vector in bytes.
    pub fn getDataSize(self: *Index) root.Error!usize {
        var v: usize = 0;
        const res = hnsw_get_data_size(self.handle, &v);
        try map_res(res);
        return v;
    }

    /// Retrieve the raw vector bytes for an element by label.
    ///
    /// `out_buffer` must have capacity >= `getDataSize()`.
    pub fn getDataByLabel(
        self: *Index,
        label: usize,
        out_buffer: []u8,
    ) (error{BufferTooSmall} || root.Error!usize) {
        const needed = try self.getDataSize();
        if (out_buffer.len < needed) return root.HnswError.BufferTooSmall;
        const res = hnsw_get_data_by_label(
            self.handle,
            label,
            out_buffer.ptr,
            out_buffer.len,
        );
        try map_res(res);
        return needed;
    }

    /// Save the index to disk.
    ///
    /// `path` must point to a NUL-terminated C string.
    pub fn save(self: *Index, path: [*]const u8) root.Error!void {
        const res = hnsw_save(self.handle, path);
        try map_res(res);
    }

    /// Set the number of threads for internal operations.
    ///
    /// Pass 0 to use the default.
    pub fn setNumThreads(self: *Index, n: i32) root.Error!void {
        const res = hnsw_set_num_threads(self.handle, n);
        try map_res(res);
    }
};
