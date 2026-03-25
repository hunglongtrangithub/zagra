#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/// Label type used by the index (matches hnswlib::labeltype).
typedef size_t hnsw_label_t;

/// Result codes returned by the C API for HNSW and Bruteforce operations.
typedef enum hnsw_res {
  HNSW_SUCCESS = 0,   // operation succeeded
  HNSW_EINVAL = -1,   // invalid argument (null pointer, bad params)
  HNSW_ENOMEM = -2,   // allocation failure
  HNSW_ERUNTIME = -3, // runtime error (capacity exceeded, file IO error, etc.)
  HNSW_EUNKNOWN = -4  // unknown/uncaught exception
} hnsw_res;

#ifdef __cplusplus
}
#endif
