#include "affinity_alloc.h"

#include "affinity_allocator.h"

extern "C" void *malloc_aff(size_t size, int n, const void **affinity_addrs) {

  /**
   * First get the list of affinity addresses.
   */
  // affinity_alloc::AffinityAddressVecT affinityAddrs;

  // for (int i = 0; i < n; ++i) {
  //   affinityAddrs.push_back(reinterpret_cast<uintptr_t>(affinity_addrs[i]));
  // }

  affinity_alloc::AffinityAddressVecT affinityAddrs(
      n, reinterpret_cast<const affinity_alloc::Addr *>(affinity_addrs));

  return affinity_alloc::alloc(size, affinityAddrs);
}

extern "C" void free_aff(void *ptr) {
  // Currently we do nothing.
}

extern "C" void print_affinity_alloc_stats() {
  affinity_alloc::printAllocatorStats();
}

extern "C" void clear_affinity_alloc() { affinity_alloc::clearAllocator(); }