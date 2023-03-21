cmake_minimum_required(VERSION 3.4)
project("AffinityAlloc")

set(AffinityAllocSrc
    affinity_alloc.cpp
    affinity_allocator.cpp
    gem_forge_ops.cpp
)

set(AffinityAllocGemForgeSrc
    affinity_alloc.cpp
    affinity_allocator.cpp
)

if (NOT DEFINED ENV{GEM_FORGE_GEM5_PATH})
	message(FATAL_ERROR "Please specify gem-forge gem5 include path in GEM_FORGE_GEM5_PATH")
endif()
set(GEM_FORGE_GEM5_INC_PATH "$ENV{GEM_FORGE_GEM5_PATH}/include")

add_library(AffinityAlloc SHARED ${AffinityAllocSrc})
target_include_directories(AffinityAlloc PUBLIC 
    ${CMAKE_CURRENT_BINARY_DIR}
    ${GEM_FORGE_GEM5_INC_PATH}
)
set_target_properties(AffinityAlloc PROPERTIES
    COMPILE_FLAGS "-std=c++17 -O3 -mavx2 -g"
)

add_library(AffinityAllocStatic STATIC ${AffinityAllocSrc})
target_include_directories(AffinityAllocStatic PUBLIC 
    ${CMAKE_CURRENT_BINARY_DIR}
    ${GEM_FORGE_GEM5_INC_PATH}
)
set_target_properties(AffinityAllocStatic PROPERTIES
    COMPILE_FLAGS "-std=c++17 -O3 -mavx2 -g"
)

add_library(AffinityAllocGemForge SHARED ${AffinityAllocGemForgeSrc})
target_include_directories(AffinityAllocGemForge PUBLIC 
    ${CMAKE_CURRENT_BINARY_DIR}
    ${GEM_FORGE_GEM5_INC_PATH}
)
set_target_properties(AffinityAllocGemForge PROPERTIES
    COMPILE_FLAGS "-std=c++17 -O3 -DGEM_FORGE"
)

add_library(AffinityAllocGemForgeStatic STATIC ${AffinityAllocGemForgeSrc})
target_include_directories(AffinityAllocGemForgeStatic PUBLIC 
    ${CMAKE_CURRENT_BINARY_DIR}
    ${GEM_FORGE_GEM5_INC_PATH}
)
set_target_properties(AffinityAllocGemForgeStatic PROPERTIES
    COMPILE_FLAGS "-std=c++17 -O3 -DGEM_FORGE -DAFFINITY_ALLOC_DPRINTF"
)