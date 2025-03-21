add_library(annotation STATIC annotation.cu)

add_library(cudaGraphExecutionTimelineProfiler STATIC cudaGraphExecutionTimelineProfiler.cu)
target_link_libraries(cudaGraphExecutionTimelineProfiler PRIVATE cudaUtilities CUDA::cupti)

add_library(peakMemoryUsageProfiler STATIC peakMemoryUsageProfiler.cu)
target_link_libraries(peakMemoryUsageProfiler PRIVATE cudaUtilities)

# Split memoryManager into two files for better organization
add_library(memoryManager STATIC memoryManager.cu memoryManager_v2.cu)
target_link_libraries(memoryManager PRIVATE cudaUtilities)

add_library(memred STATIC memred.cu)
target_link_libraries(memred PRIVATE cudaUtilities cudaGraphUtilities)
