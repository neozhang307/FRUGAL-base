# Memory Management Library
cmake_minimum_required(VERSION 3.18)

# Create memory management library
add_library(memoryManager STATIC
    memoryManager.cu
    memoryManager_v2.cu
)

# Set CUDA properties
set_target_properties(memoryManager PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(memoryManager PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(memoryManager
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
    PRIVATE
        cudaUtilities
)

# Compile features
target_compile_features(memoryManager PUBLIC cxx_std_17)