#pragma once

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for a HierarchicalNSW index instance.
typedef struct hnsw_index_t hnsw_index_t;

/*
 * Creation / loading
 *
 * NOTE about spaces & vector layout:
 * - These convenience constructors build an index that uses L2 distance on
 * `float` vectors.
 * - `dim` is the number of coordinates per vector (so the byte-size per vector
 * is dim * sizeof(float)).
 *
 * `hnsw_new_l2`:
 *  - Create a new HierarchicalNSW index for L2 space.
 *  - Parameters:
 *      dim             : vector dimensionality (number of floats)
 *      max_elements    : capacity (maximum number of elements to allocate)
 *      M               : connectivity parameter (recommended default 16)
 *      ef_construction : construction-time ef parameter (recommended default
 * 200) random_seed    : RNG seed (implementation uses this to seed internal
 * RNGs) allow_replace_deleted : whether insertion may reuse slots of previously
 *        marked-deleted elements
 *  - Returns: pointer to `hnsw_index_t` on success, or NULL on failure
 *    (allocation/exception).
 */
hnsw_index_t *hnsw_new_l2(size_t dim, size_t max_elements, size_t M,
                          size_t ef_construction, size_t random_seed,
                          bool allow_replace_deleted);

/*
 * Load an index previously saved by `hnsw_save`.
 * - `path` is a NUL-terminated C string to the index file (binary format used
 * by hnswlib).
 * - `dim` must match the dimensionality used when the index was saved/created.
 * - Returns pointer to `hnsw_index_t` on success, or NULL on failure.
 */
hnsw_index_t *hnsw_load_l2(const char *path, size_t dim);

/*
 * Destroy index and free resources. Safe to pass NULL.
 */
void hnsw_free(hnsw_index_t *idx);

/*
 * Insert / remove / mark
 *
 * Add a point (float array of length `dim` as specified at creation/load time).
 * - `data` points to an array of `float` of length `dim`.
 * - If `replace_deleted` is true, the index may reuse slots of previously
 * marked-deleted elements.
 * - Returns HNSW_SUCCESS on success, or an error code on failure.
 */
hnsw_res hnsw_add_point(hnsw_index_t *idx, const float *data,
                        hnsw_label_t label, bool replace_deleted);

/*
 * Mark the element with the given label as deleted.
 * This mirrors HierarchicalNSW::markDelete: it marks the element deleted
 * without reshaping the graph. Returns HNSW_SUCCESS on success, or an error
 * code on failure.
 */
hnsw_res hnsw_mark_delete(hnsw_index_t *idx, hnsw_label_t label);

/*
 * Unmark a previously-marked-as-deleted element (if present).
 * Returns HNSW_SUCCESS on success, or an error code on failure.
 */
hnsw_res hnsw_unmark_delete(hnsw_index_t *idx, hnsw_label_t label);

/*
 * Resize index capacity.
 * - `new_max_elements` must be >= current element count.
 * - On success the index will be able to hold up to `new_max_elements` items.
 * - Returns HNSW_SUCCESS on success, or an error code on failure.
 */
hnsw_res hnsw_resize_index(hnsw_index_t *idx, size_t new_max_elements);

/*
 * Search
 *
 * Search for k nearest neighbors of `query` (array of `dim` floats).
 * - Caller must provide `out_labels` and `out_distances` buffers of length >=
 * k.
 * - On success returns HNSW_SUCCESS and stores the number of results written
 * into `out_result_count`. The results written will be in closest-first order
 * (closest at index 0).
 * - On error returns an HNSW_* error code and `out_result_count` is left
 * unchanged.
 */
hnsw_res hnsw_search_knn(hnsw_index_t *idx, const float *query, size_t k,
                         hnsw_label_t *out_labels, float *out_distances,
                         size_t *out_result_count);

/*
 * Convenience: same as hnsw_search_knn but returns results in the internal
 * (closer-first) order and supports an `ef` parameter to override the search ef
 * (if supported). If your wrapper/runtime does not implement ef override,
 * callers may pass 0 to use index default.
 */
hnsw_res hnsw_search_knn_with_ef(hnsw_index_t *idx, const float *query,
                                 size_t k, size_t ef, hnsw_label_t *out_labels,
                                 float *out_distances,
                                 size_t *out_result_count);

/*
 * Parameters, introspection
 *
 * Set the runtime `ef` parameter used by search. Returns HNSW_SUCCESS on
 * success.
 */
hnsw_res hnsw_set_ef(hnsw_index_t *idx, size_t ef);

/*
 * Get the current `ef` parameter. On success writes into `out_ef`.
 */
hnsw_res hnsw_get_ef(hnsw_index_t *idx, size_t *out_ef);

/*
 * Get capacity (max elements) and current element count.
 */
hnsw_res hnsw_get_max_elements(hnsw_index_t *idx, size_t *out_max_elements);
hnsw_res hnsw_get_current_count(hnsw_index_t *idx, size_t *out_current_count);
hnsw_res hnsw_get_deleted_count(hnsw_index_t *idx, size_t *out_deleted_count);

/*
 * Get the data byte-size per vector (dim * sizeof(float)) as reported by the
 * index. Useful for callers that want to allocate buffers for
 * `get_data_by_label`.
 */
hnsw_res hnsw_get_data_size(hnsw_index_t *idx, size_t *out_data_size);

/*
 * Retrieve the vector bytes for an element by its external label.
 * - `out_buffer` must point to a writable buffer at least `out_buffer_len`
 * bytes long.
 * - `out_buffer_len` should be >= the value returned by `hnsw_get_data_size`.
 * - On success copies the raw vector bytes into `out_buffer` and returns
 * HNSW_SUCCESS.
 * - On failure returns an error (e.g., label not found => HNSW_ERUNTIME).
 */
hnsw_res hnsw_get_data_by_label(hnsw_index_t *idx, hnsw_label_t label,
                                void *out_buffer, size_t out_buffer_len);

/*
 * Save / load
 *
 * Save the index to disk using the same binary format used by
 * hnswlib::HierarchicalNSW::saveIndex.
 * - `path` is a NUL-terminated C string path to write to.
 * - Returns HNSW_SUCCESS on success or an error code on failure.
 */
hnsw_res hnsw_save(hnsw_index_t *idx, const char *path);

/*
 * Low-level utility:
 * - Set the number of threads used by internal operations (if supported).
 * Passing 0 lets the implementation pick a default. This is a best-effort hint;
 * not all builds use this.
 */
hnsw_res hnsw_set_num_threads(hnsw_index_t *idx, int num_threads);

/*
 * Batch add points:
 * - `data` points to a flat array of `count * dim` floats (row-major).
 * - `labels` points to array of `count` labels.
 * - Uses ParallelFor internally for parallel insertion.
 * - Returns HNSW_SUCCESS on success, or an error code on failure.
 */
hnsw_res hnsw_add_points_batch(hnsw_index_t *idx, const float *data,
                                const hnsw_label_t *labels, size_t count,
                                bool replace_deleted);

/*
 * Batch search for k nearest neighbors:
 * - `queries` points to a flat array of `num_queries * dim` floats (row-major).
 * - `k` is number of results per query.
 * - Results written as flat array: query i's k results start at index i*k.
 * - `out_counts` returns actual result count per query (0 to k).
 * - Total allocated: `num_queries * k` slots per array.
 * - Returns HNSW_SUCCESS on success, or an error code on failure.
 */
hnsw_res hnsw_search_knn_batch(hnsw_index_t *idx, const float *queries,
                               size_t k, hnsw_label_t *out_labels,
                               float *out_distances, size_t *out_counts,
                               size_t num_queries);

/*
 * Notes
 * - All functions are C-callable; actual implementation will be in a C++
 * translation unit that constructs and manipulates
 * `hnswlib::HierarchicalNSW<float>` (or other concrete instantiation).
 * - The wrapper implementation MUST not let C++ exceptions propagate across the
 * C boundary; instead it should map exceptions to appropriate `hnsw_res`
 * values.
 * - Many operations (add/remove) are thread-safe in the C++ implementation;
 * search may rely on internal locking behavior. Consult the wrapper
 * documentation for concurrency semantics.
 */

#ifdef __cplusplus
}
#endif
