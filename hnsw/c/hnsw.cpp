// C++ wrapper implementing a C ABI for hnswlib::HierarchicalNSW<float> (L2
// space).
//
// This file provides C-callable functions declared in hnsw_c_api.h and
// delegates to hnswlib C++ classes. All exceptions are caught and mapped
// to hnsw_res codes so that no C++ exceptions escape the C ABI.

#include "hnsw.h"
#include "../hnswlib/hnswlib.h"
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <cstring>
#include <exception>
#include <functional>

using namespace hnswlib;

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib 
 */
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if ((id >= end)) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

struct hnsw_index_t {
  HierarchicalNSW<float> *idx;
  L2Space *space;
  size_t dim;
  size_t num_threads;
};

// Helper: translate exceptions into hnsw_res
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

// Create a new HierarchicalNSW index for L2 space.
// Returns nullptr on allocation/fatal failure.
hnsw_index_t *hnsw_new_l2(size_t dim, size_t max_elements, size_t M,
                          size_t ef_construction, size_t random_seed,
                          bool allow_replace_deleted) {
  try {
    L2Space *space = new L2Space(dim);
    auto *idx =
        new HierarchicalNSW<float>(space, max_elements, M, ef_construction,
                                   random_seed, allow_replace_deleted);
    hnsw_index_t *h =
        static_cast<hnsw_index_t *>(std::malloc(sizeof(hnsw_index_t)));
    if (!h) {
      delete idx;
      delete space;
      return nullptr;
    }
    h->idx = idx;
    h->space = space;
    h->dim = dim;
    h->num_threads = std::thread::hardware_concurrency();
    return h;
  } catch (const std::bad_alloc &) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

// Load an index from disk. `dim` must match the saved index.
hnsw_index_t *hnsw_load_l2(const char *path, size_t dim) {
  if (!path)
    return nullptr;
  try {
    L2Space *space = new L2Space(dim);
    auto *idx =
        new HierarchicalNSW<float>(space, std::string(path), false, 0, false);
    hnsw_index_t *h =
        static_cast<hnsw_index_t *>(std::malloc(sizeof(hnsw_index_t)));
    if (!h) {
      delete idx;
      delete space;
      return nullptr;
    }
    h->idx = idx;
    h->space = space;
    h->dim = dim;
    h->num_threads = std::thread::hardware_concurrency();
    return h;
  } catch (const std::bad_alloc &) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

// Destroy index and free resources. Safe to pass NULL.
void hnsw_free(hnsw_index_t *h) {
  if (!h)
    return;
  delete h->idx;
  delete h->space;
  std::free(h);
}

// Add a point. data must point to `dim` floats. replace_deleted: treat as bool
// (true/false).
hnsw_res hnsw_add_point(hnsw_index_t *h, const float *data, hnsw_label_t label,
                        bool replace_deleted) {
  if (!h || !data)
    return HNSW_EINVAL;
  return translate_exception([&]() -> hnsw_res {
    h->idx->addPoint(reinterpret_cast<const void *>(data), label,
                     replace_deleted);
    return HNSW_SUCCESS;
  });
}

// Mark a label deleted.
hnsw_res hnsw_mark_delete(hnsw_index_t *h, hnsw_label_t label) {
  if (!h)
    return HNSW_EINVAL;
  return translate_exception([&]() -> hnsw_res {
    h->idx->markDelete(label);
    return HNSW_SUCCESS;
  });
}

// Unmark delete (restore).
hnsw_res hnsw_unmark_delete(hnsw_index_t *h, hnsw_label_t label) {
  if (!h)
    return HNSW_EINVAL;
  return translate_exception([&]() -> hnsw_res {
    h->idx->unmarkDelete(label);
    return HNSW_SUCCESS;
  });
}

// Resize index capacity.
hnsw_res hnsw_resize_index(hnsw_index_t *h, size_t new_max_elements) {
  if (!h)
    return HNSW_EINVAL;
  return translate_exception([&]() -> hnsw_res {
    h->idx->resizeIndex(new_max_elements);
    return HNSW_SUCCESS;
  });
}

// Search for k nearest neighbors. Writes up to k results into
// out_labels/out_distances. On success sets *out_result_count and returns
// HNSW_HNSW_SUCCESS.
hnsw_res hnsw_search_knn(hnsw_index_t *h, const float *query, size_t k,
                         hnsw_label_t *out_labels, float *out_distances,
                         size_t *out_result_count) {
  if (!h || !query || !out_result_count)
    return HNSW_EINVAL;
  if (k > 0 && (!out_labels || !out_distances))
    return HNSW_EINVAL;
  return translate_exception([&]() -> hnsw_res {
    // Use searchKnnCloserFirst to get results closest-first as a std::vector
    std::vector<std::pair<float, labeltype>> res = h->idx->searchKnnCloserFirst(
        reinterpret_cast<const void *>(query), k, nullptr);
    size_t n = res.size();
    size_t to_write = n;
    if (to_write > k)
      to_write = k;
    for (size_t i = 0; i < to_write; i++) {
      out_labels[i] = res[i].second;
      out_distances[i] = res[i].first;
    }
    *out_result_count = to_write;
    return HNSW_SUCCESS;
  });
}

// Search with ef override. Temporarily set ef, perform search, restore old ef.
hnsw_res hnsw_search_knn_with_ef(hnsw_index_t *h, const float *query, size_t k,
                                 size_t ef, hnsw_label_t *out_labels,
                                 float *out_distances,
                                 size_t *out_result_count) {
  if (!h || !query || !out_result_count)
    return HNSW_EINVAL;
  if (k > 0 && (!out_labels || !out_distances))
    return HNSW_EINVAL;
  return translate_exception([&]() -> hnsw_res {
    // Save old ef, set new, search, restore.
    size_t old_ef = h->idx->ef_;
    h->idx->setEf(ef == 0 ? old_ef : ef);
    std::vector<std::pair<float, labeltype>> res = h->idx->searchKnnCloserFirst(
        reinterpret_cast<const void *>(query), k, nullptr);
    h->idx->setEf(old_ef);
    size_t n = res.size();
    size_t to_write = n;
    if (to_write > k)
      to_write = k;
    for (size_t i = 0; i < to_write; i++) {
      out_labels[i] = res[i].second;
      out_distances[i] = res[i].first;
    }
    *out_result_count = to_write;
    return HNSW_SUCCESS;
  });
}

// Set runtime ef parameter.
hnsw_res hnsw_set_ef(hnsw_index_t *h, size_t ef) {
  if (!h)
    return HNSW_EINVAL;
  return translate_exception([&]() -> hnsw_res {
    h->idx->setEf(ef);
    return HNSW_SUCCESS;
  });
}

// Get current ef parameter (writes into out_ef).
hnsw_res hnsw_get_ef(hnsw_index_t *h, size_t *out_ef) {
  if (!h || !out_ef)
    return HNSW_EINVAL;
  return translate_exception([&]() -> hnsw_res {
    *out_ef = h->idx->ef_;
    return HNSW_SUCCESS;
  });
}

// Getters for sizes/counts.
hnsw_res hnsw_get_max_elements(hnsw_index_t *h, size_t *out_max_elements) {
  if (!h || !out_max_elements)
    return HNSW_EINVAL;
  *out_max_elements = h->idx->getMaxElements();
  return HNSW_SUCCESS;
}

hnsw_res hnsw_get_current_count(hnsw_index_t *h, size_t *out_current_count) {
  if (!h || !out_current_count)
    return HNSW_EINVAL;
  *out_current_count = h->idx->getCurrentElementCount();
  return HNSW_SUCCESS;
}

hnsw_res hnsw_get_deleted_count(hnsw_index_t *h, size_t *out_deleted_count) {
  if (!h || !out_deleted_count)
    return HNSW_EINVAL;
  *out_deleted_count = h->idx->getDeletedCount();
  return HNSW_SUCCESS;
}

// Get data size per vector in bytes.
hnsw_res hnsw_get_data_size(hnsw_index_t *h, size_t *out_data_size) {
  if (!h || !out_data_size)
    return HNSW_EINVAL;
  *out_data_size = h->idx->data_size_;
  return HNSW_SUCCESS;
}

// Retrieve raw vector bytes by external label into out_buffer.
// out_buffer_len must be >= data size reported by hnsw_get_data_size.
hnsw_res hnsw_get_data_by_label(hnsw_index_t *h, hnsw_label_t label,
                                void *out_buffer, size_t out_buffer_len) {
  if (!h || !out_buffer)
    return HNSW_EINVAL;
  return translate_exception([&]() -> hnsw_res {
    // HierarchicalNSW::getDataByLabel is a template; request the float
    // instantiation. The function returns a std::vector<float> containing the
    // vector coordinates.
    auto vec = h->idx->template getDataByLabel<float>(label);
    // Compute byte size: number of elements * sizeof(float)
    size_t bytes = vec.size() * sizeof(float);
    if (out_buffer_len < bytes) {
      return HNSW_EINVAL;
    }
    if (bytes > 0) {
      std::memcpy(out_buffer, vec.data(), bytes);
    }
    return HNSW_SUCCESS;
  });
}

// Save to disk.
hnsw_res hnsw_save(hnsw_index_t *h, const char *path) {
  if (!h || !path)
    return HNSW_EINVAL;
  return translate_exception([&]() -> hnsw_res {
    h->idx->saveIndex(std::string(path));
    return HNSW_SUCCESS;
  });
}

// Set number of threads for parallel operations.
hnsw_res hnsw_set_num_threads(hnsw_index_t *h, int num_threads) {
  if (!h)
    return HNSW_EINVAL;
  h->num_threads = num_threads > 0 ? num_threads : std::thread::hardware_concurrency();
  return HNSW_SUCCESS;
}

// Batch add points using ParallelFor.
hnsw_res hnsw_add_points_batch(hnsw_index_t *h, const float *data,
                                const hnsw_label_t *labels, size_t count,
                                bool replace_deleted) {
  if (!h || !data || !labels)
    return HNSW_EINVAL;
  return translate_exception([&]() -> hnsw_res {
    ParallelFor(0, count, h->num_threads, [&](size_t id, size_t threadId) {
      const float *vec = data + id * h->dim;
      h->idx->addPoint(reinterpret_cast<const void *>(vec), labels[id],
                       replace_deleted);
    });
    return HNSW_SUCCESS;
  });
}

// Batch search for k nearest neighbors.
// Results written as flat array: query i's k results start at index i*k.
// out_counts returns actual result count per query (0 to k).
// Total allocated: num_queries * k slots per array.
hnsw_res hnsw_search_knn_batch(hnsw_index_t *h, const float *queries,
                               size_t k, hnsw_label_t *out_labels,
                               float *out_distances, size_t *out_counts,
                               size_t num_queries) {
  if (!h || !queries || !out_labels || !out_distances || !out_counts)
    return HNSW_EINVAL;
  return translate_exception([&]() -> hnsw_res {
    ParallelFor(0, num_queries, h->num_threads, [&](size_t qid, size_t threadId) {
      const float *query = queries + qid * h->dim;
      hnsw_label_t *out_labels_q = out_labels + qid * k;
      float *out_distances_q = out_distances + qid * k;
      std::vector<std::pair<float, labeltype>> res =
          h->idx->searchKnnCloserFirst(reinterpret_cast<const void *>(query), k, nullptr);
      size_t n = res.size();
      out_counts[qid] = n;
      for (size_t i = 0; i < n; i++) {
        out_labels_q[i] = res[i].second;
        out_distances_q[i] = res[i].first;
      }
    });
    return HNSW_SUCCESS;
  });
}

} // extern "C"
