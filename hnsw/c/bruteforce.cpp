// C++ wrapper implementing a C ABI for hnswlib::BruteforceSearch<float> (L2
// space).
//
// Notes:
// - This wrapper exposes a small, opaque-handle C API declared in hnsw_c_api.h
// - It instantiates hnswlib::L2Space and hnswlib::BruteforceSearch<float>
// internally.
// - All exceptions are caught and mapped to negative error codes; no exceptions
// escape the C ABI.

#include "bruteforce.h"
#include "../hnswlib/hnswlib.h"
#include <functional>

using namespace hnswlib;

struct bruteforce_t {
  BruteforceSearch<float> *idx;
  L2Space *space;
  size_t dim;
};

static hnsw_res translate_exception(const std::function<hnsw_res()> &f) {
  try {
    return f();
  } catch (const std::bad_alloc &) {
    return HNSW_ENOMEM;
  } catch (const std::runtime_error &) {
    return HNSW_ERUNTIME;
  } catch (...) {
    return HNSW_EUNKNOWN;
  }
}

extern "C" {

// Create a new L2 brute-force index for `dim`-dim vectors with capacity
// `max_elements`. Returns nullptr on failure (allocation failure). Use
// `bruteforce_free` to destroy.
bruteforce_t *bruteforce_new_l2(size_t dim, size_t max_elements) {
  try {
    L2Space *space = new L2Space(dim);
    BruteforceSearch<float> *idx =
        new BruteforceSearch<float>(space, max_elements);
    bruteforce_t *h =
        static_cast<bruteforce_t *>(std::malloc(sizeof(bruteforce_t)));
    if (!h) {
      delete idx;
      delete space;
      return nullptr;
    }
    h->idx = idx;
    h->space = space;
    h->dim = dim;
    return h;
  } catch (const std::bad_alloc &) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

// Load an index previously saved by `bruteforce_save`. `dim` must match the
// saved index.
bruteforce_t *bruteforce_load_l2(const char *path, size_t dim) {
  if (!path)
    return nullptr;
  try {
    L2Space *space = new L2Space(dim);
    // BruteforceSearch constructor with (space, location) loads from disk.
    BruteforceSearch<float> *idx =
        new BruteforceSearch<float>(space, std::string(path));
    bruteforce_t *h =
        static_cast<bruteforce_t *>(std::malloc(sizeof(bruteforce_t)));
    if (!h) {
      delete idx;
      delete space;
      return nullptr;
    }
    h->idx = idx;
    h->space = space;
    h->dim = dim;
    return h;
  } catch (const std::bad_alloc &) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

// Destroy index and free resources. Safe to pass nullptr.
void bruteforce_free(bruteforce_t *h) {
  if (!h)
    return;
  delete h->idx;
  delete h->space;
  std::free(h);
}

// Add a point (array of `dim` floats). The index copies the data.
// Returns BF_SUCCESS on success or a negative error code.
hnsw_res bruteforce_add_point(bruteforce_t *h, const float *data,
                              hnsw_label_t label) {
  if (!h || !data)
    return HNSW_EINVAL;
  return translate_exception([&]() -> hnsw_res {
    h->idx->addPoint(reinterpret_cast<const void *>(data), label);
    return HNSW_SUCCESS;
  });
}

// Remove a point by label. Returns BF_SUCCESS even if label not found (mirrors
// C++ behavior).
hnsw_res bruteforce_remove_point(bruteforce_t *h, hnsw_label_t label) {
  if (!h)
    return HNSW_EINVAL;
  return translate_exception([&]() -> hnsw_res {
    h->idx->removePoint(label);
    return HNSW_SUCCESS;
  });
}

// Search for k nearest neighbors to `query` (array of `dim` floats).
// Caller must provide `out_labels` and `out_distances` buffers of length >= k.
// Stores the number of results written (0..k) in `out_result_count`.
hnsw_res bruteforce_search_knn(bruteforce_t *h, const float *query, size_t k,
                               hnsw_label_t *out_labels, float *out_distances,
                               size_t *out_result_count) {
  if (!h || !query || !out_result_count)
    return HNSW_EINVAL;
  if (k > 0 && (!out_labels || !out_distances))
    return HNSW_EINVAL;

  return translate_exception([&]() -> hnsw_res {
    auto heap =
        h->idx->searchKnn(reinterpret_cast<const void *>(query), k, nullptr);
    // heap: max-heap (furthest-first). Extract into vector then reverse to
    // closest-first.
    std::vector<std::pair<float, labeltype>> vec;
    vec.reserve(heap.size());
    while (!heap.empty()) {
      vec.emplace_back(heap.top());
      heap.pop();
    }
    std::reverse(vec.begin(), vec.end());
    size_t n = vec.size();
    size_t to_write = n;
    if (to_write > k)
      to_write = k;
    for (size_t i = 0; i < to_write; i++) {
      out_labels[i] = vec[i].second;
      out_distances[i] = vec[i].first;
    }
    *out_result_count = to_write;
    return HNSW_SUCCESS;
  });
}

// Save the index to disk (binary format used by BruteforceSearch::saveIndex).
// Returns BF_SUCCESS on success or negative error code.
hnsw_res bruteforce_save(bruteforce_t *h, const char *path) {
  if (!h || !path)
    return HNSW_EINVAL;
  return translate_exception([&]() -> hnsw_res {
    h->idx->saveIndex(std::string(path));
    return HNSW_SUCCESS;
  });
}

} // extern "C"
