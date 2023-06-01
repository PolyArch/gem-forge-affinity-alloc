#ifndef GEM_FORGE_OPS_H
#define GEM_FORGE_OPS_H

#include "gem5/m5ops.h"

#include <cstddef>
#include <map>
#include <string>

/**
 * Provide fake implementation for m5 operations.
 */

struct Region {
  std::string name;
  uint64_t vaddr;
  uint64_t elemSize;
  size_t numElem;
  size_t interleave = 64;
  Region(const char *_name, uint64_t _vaddr, uint64_t _elemSize,
         size_t _numElem)
      : name(_name), vaddr(_vaddr), elemSize(_elemSize), numElem(_numElem) {}
};

namespace {
std::map<uint64_t, Region> regionMap;

Region *getRegion(const void *ptr) {
  uint64_t vaddr = reinterpret_cast<uint64_t>(ptr);
  auto iter = regionMap.upper_bound(vaddr);
  if (iter == regionMap.begin()) {
    return nullptr;
  }
  --iter;
  auto &region = iter->second;
  if (region.vaddr + region.numElem * region.elemSize <= vaddr) {
    return nullptr;
  }
  return &region;
}

} // namespace

extern "C" void m5_stream_nuca_region(const char *regionName,
                                      const void *buffer, uint64_t elementSize,
                                      uint64_t dim1, uint64_t dim2,
                                      uint64_t dim3) {
  uint64_t numElem = dim1;
  if (dim2) {
    numElem *= dim2;
  }
  if (dim3) {
    numElem *= dim3;
  }
  uint64_t vaddr = reinterpret_cast<uint64_t>(buffer);
  regionMap.emplace(
      std::piecewise_construct, std::forward_as_tuple(vaddr),
      std::forward_as_tuple(regionName, vaddr, elementSize, numElem));
}

extern "C" void
m5_stream_nuca_set_property(const void *buffer,
                            enum StreamNUCARegionProperty property,
                            uint64_t value) {
  if (property == STREAM_NUCA_REGION_PROPERTY_INTERLEAVE) {
    if (auto region = getRegion(buffer)) {
      region->interleave = value;
    }
  }
}

extern "C" uint64_t
m5_stream_nuca_get_property(const void *buffer,
                            enum StreamNUCARegionProperty property) {

  if (property == STREAM_NUCA_REGION_PROPERTY_BANK_ROWS) {
    return 8;
  }
  if (property == STREAM_NUCA_REGION_PROPERTY_BANK_COLS) {
    return 8;
  }
  if (property == STREAM_NUCA_REGION_PROPERTY_INTERLEAVE) {
    if (auto region = getRegion(buffer)) {
      return region->interleave;
    }
    return 64;
  }
  if (property == STREAM_NUCA_REGION_PROPERTY_START_BANK) {
    return 0;
  }
  const size_t pageShift = 12;
  if (property == STREAM_NUCA_REGION_PROPERTY_START_VADDR) {
    if (auto region = getRegion(buffer)) {
      return region->vaddr;
    }
    return (reinterpret_cast<size_t>(buffer) >> pageShift) << pageShift;
  }
  if (property == STREAM_NUCA_REGION_PROPERTY_END_VADDR) {
    if (auto region = getRegion(buffer)) {
      return region->vaddr + region->elemSize * region->numElem;
    }
    return ((reinterpret_cast<size_t>(buffer) >> pageShift) + 1) << pageShift;
  }

  assert(false && "Illegal RegionPropoerty");
}

extern "C" void m5_stream_nuca_remap() {}

#endif