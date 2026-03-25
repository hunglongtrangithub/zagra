#pragma once

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle for the brute-force index */
typedef struct bruteforce_t bruteforce_t;

/*
 * Create a new L2 brute-force index for `dim`-dim vectors with capacity
 * `max_elements`. Returns a pointer to a `bruteforce_t` on success, or NULL on
 * allocation/fatal failure. The returned handle owns the underlying C++
 * objects; call `bruteforce_free` to destroy.
 */
bruteforce_t *bruteforce_new_l2(size_t dim, size_t max_elements);

/*
 * Load an index previously saved with bruteforce_save.
 * `dim` must match the dimension used when the index was created/saved.
 * Returns a pointer to a `bruteforce_t` on success, or NULL on failure.
 */
bruteforce_t *bruteforce_load_l2(const char *path, size_t dim);

/*
 * Destroy the index and free resources. Safe to pass NULL.
 */
void bruteforce_free(bruteforce_t *idx);

/*
 * Add a point to the index. `data` must point to `dim` floats (row-major).
 * The index copies the provided data into its internal storage.
 * Returns HNSW_SUCCESS on success or a negative error code on failure.
 */
hnsw_res bruteforce_add_point(bruteforce_t *idx, const float *data,
                              hnsw_label_t label);

/*
 * Remove a point by its label. Mirrors the C++ implementation behavior:
 * - If the label existed it will be removed and HNSW_SUCCESS is returned.
 * - If the label did not exist, HNSW_SUCCESS is also returned (no-op).
 */
hnsw_res bruteforce_remove_point(bruteforce_t *idx, hnsw_label_t label);

/*
 * Search for k nearest neighbors of `query` (array of `dim` floats).
 * Caller must provide `out_labels` and `out_distances` buffers of length >= k.
 * On success returns HNSW_SUCCESS and sets `out_result_count` to the number of
 * results written (0..k). On error returns a negative error code and
 * `out_result_count` is undefined.
 *
 * Note: The C++ implementation asserts k <= current_element_count. This wrapper
 * will return an error if arguments are invalid; callers should ensure k is
 * reasonable.
 */
hnsw_res bruteforce_search_knn(bruteforce_t *idx, const float *query, size_t k,
                               hnsw_label_t *out_labels, float *out_distances,
                               size_t *out_result_count);

/*
 * Save the index to disk (binary format used by
 * hnswlib::BruteforceSearch::saveIndex). Returns HNSW_SUCCESS on success or a
 * negative error code on failure.
 */
hnsw_res bruteforce_save(bruteforce_t *idx, const char *path);

#ifdef __cplusplus
}
#endif
