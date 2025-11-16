add_library(annotation STATIC annotation.cu)

add_library(cudaGraphExecutionTimelineProfiler STATIC cudaGraphExecutionTimelineProfiler.cu)
target_link_libraries(cudaGraphExecutionTimelineProfiler PRIVATE cudaUtilities CUDA::cupti)

add_library(peakMemoryUsageProfiler STATIC peakMemoryUsageProfiler.cu)
target_link_libraries(peakMemoryUsageProfiler PRIVATE cudaUtilities)

# memoryManager has been moved to memory/ directory

add_library(memred STATIC memred.cu)
target_link_libraries(memred PRIVATE cudaUtilities cudaGraphUtilities)
