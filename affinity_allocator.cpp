#include "affinity_allocator.h"

#include <sys/types.h>
#include <unistd.h>

#include <cstdio>
#include <sstream>

namespace affinity_alloc {

AffinityAllocatorArgs AffinityAllocatorArgs::initialize() {
  AffinityAllocatorArgs args;

  if (const char *envVar = std::getenv("AFFINITY_ALLOCATOR_ALLOC_POLICY")) {
    std::string allocPolicy(envVar);
    DPRINTF("AllocPolicy = %s\n", envVar);
    if (allocPolicy == "RANDOM") {
      args.allocPolicy = AllocPolicy::RANDOM;
    } else if (allocPolicy == "MIN_HOPS") {
      args.allocPolicy = AllocPolicy::MIN_HOPS;
    } else if (allocPolicy == "MIN_LOAD") {
      args.allocPolicy = AllocPolicy::MIN_LOAD;
    } else if (allocPolicy == "HYBRID") {
      args.allocPolicy = AllocPolicy::HYBRID;
    } else if (allocPolicy == "DELTA") {
      args.allocPolicy = AllocPolicy::DELTA;
    }
  }
  if (const char *envLoadWeight =
          std::getenv("AFFINITY_ALLOCATOR_LOAD_WEIGHT")) {
    args.loadWeight = std::atoi(envLoadWeight);
    DPRINTF("LoadWeight  = %d\n", args.loadWeight);
  }
  if (const char *envLogLevel = std::getenv("AFFINITY_ALLOCATOR_LOG_LEVEL")) {
    args.logLevel = std::atoi(envLogLevel);
    DPRINTF("LogLevel = %d\n", args.logLevel);
  }

  return args;
}

// Define the fixed sizes.
constexpr size_t FixNodeSizes[] = {
    64, 128, 256, 512, 1024, 4096,
};

// Specialize the allocator for predefined sizes.
#ifndef AFFINITY_ALLOC_ARENA_SIZE
#define AFFINITY_ALLOC_ARENA_SIZE 8192
#endif
constexpr size_t FixArenaSize = AFFINITY_ALLOC_ARENA_SIZE;
MultiThreadAffinityAllocator<64, FixArenaSize> allocator64B;
// MultiThreadAffinityAllocator<128, FixArenaSize> allocator128B;
// MultiThreadAffinityAllocator<256, FixArenaSize> allocator256B;
// MultiThreadAffinityAllocator<512, FixArenaSize> allocator512B;
// MultiThreadAffinityAllocator<1024, FixArenaSize> allocator1024B;
// MultiThreadAffinityAllocator<4096, FixArenaSize> allocator4096B;

std::string printAddrs(const AffinityAddressVecT &addrs) {
  std::stringstream ss;
  for (const auto addr : addrs) {
    ss << std::hex << addr << ' ';
  }
  return ss.str();
}

void *alloc(size_t size, const AffinityAddressVecT &affinityAddrs) {
  // Round up the size.
  auto roundSize = 0;
  for (auto x : FixNodeSizes) {
    if (size <= x) {
      roundSize = x;
      break;
    }
  }
  assert(roundSize != 0 && "Illegal size.");

// Get the tid.
#ifdef AFFINITY_ALLOC_SINGLE_THREAD
  int tid = 0;
#else
  int tid = gettid();
#endif

  // printf("[AffAlloc Th=%d] Alloc Size %lu/%d AffAddrs %s.\n", tid, size,
  //        roundSize, printAddrs(affinityAddrs).c_str());

  // Dispatch to the correct allocator.
#define CASE(X)                                                                \
  case X:                                                                      \
    return allocator##X##B.alloc(tid, affinityAddrs);                          \
    break;
  switch (roundSize) {
    CASE(64);
    // CASE(128);
    // CASE(256);
    // CASE(512);
    // CASE(1024);
    // CASE(4096);
  default:
    assert(false && "Illegal size.");
    break;
  }
#undef CASE
}

void printAllocatorStats() {
  allocator64B.printStats();
  // allocator128B.printStats();
  // allocator256B.printStats();
  // allocator512B.printStats();
  // allocator1024B.printStats();
  // allocator4096B.printStats();
}

void clearAllocator() {
  allocator64B.clear();
  // allocator128B.clear();
  // allocator256B.clear();
  // allocator512B.clear();
  // allocator1024B.clear();
  // allocator4096B.clear();
}

} // namespace affinity_alloc
