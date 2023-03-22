#ifndef AFFINITY_ALLOC_HH
#define AFFINITY_ALLOC_HH

#include "stdlib.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Alloc the block closer to a list of addresses.
 *
 * @param size Bytes to allocate.
 * @param ... A list of affinity addresses.
 * @return void*
 */
void *malloc_aff(size_t size, int n, const void **affinity_addrs);

/**
 * @brief Free the affinity allocated data. Currently do nothing.
 *
 * @param ptr
 */
void free_aff(void *ptr);

#ifdef __cplusplus
}
#endif

#endif