add_library(utilities STATIC utilities.cpp)

add_library(cudaUtilities STATIC cudaUtilities.cu)

add_library(cudaGraphUtilities STATIC cudaGraphUtilities.cu)
target_link_libraries(cudaGraphUtilities PRIVATE cudaUtilities CUDA::cuda_driver)

# Header-only library for CUDA graph construction abstraction
add_library(cudaGraphConstructor INTERFACE)
target_include_directories(cudaGraphConstructor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cudaGraphConstructor INTERFACE cudaUtilities CUDA::cuda_driver)