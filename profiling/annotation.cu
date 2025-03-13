#include <algorithm>
#include <cstring>
#include <functional>

#include "annotation.hpp"
#include "memoryManager.hpp"

namespace memopt {

/**
 * @brief Empty kernel used to carry annotation data in the CUDA execution stream
 * 
 * This kernel doesn't perform any computation. Its sole purpose is to be a marker
 * in the CUDA execution stream that carries metadata about computational tasks.
 * During profiling, these markers can be captured and analyzed to understand the
 * data access patterns of the computation.
 * 
 * @param taskAnnotation Structure containing task ID and input/output memory pointers
 * @param inferDataDependency Flag indicating whether to infer dependencies automatically
 */
__global__ void dummyKernelForAnnotation(TaskAnnotation taskAnnotation, bool inferDataDependency) {
  return; // Empty kernel - doesn't do any actual computation
}

/**
 * @brief Annotates the next computational task with its data dependencies
 * 
 * This function is called before executing a computational task to record:
 * 1. Which task ID is being executed
 * 2. Which memory arrays it will read from (inputs)
 * 3. Which memory arrays it will write to (outputs)
 * 
 * It filters out any non-managed memory addresses and then launches a dummy kernel
 * with the annotation data. This dummy kernel doesn't do any computation but serves
 * as a marker in the CUDA stream that 
 * 1. the profiling system can detect and analyze.
 * 2. mark as boundary of a task such that the scheduling system operate on
 *
 * The dummy kernel is scheduled in the same stream as the actual computation,
 * ensuring that it's executed in the correct sequence relative to the computation.
 * 
 * @param taskId Identifier for the computational task
 * @param inputs Vector of pointers to input arrays (read by the task)
 * @param outputs Vector of pointers to output arrays (written by the task)
 * @param stream CUDA stream where the annotation kernel and computation will run
 */
__host__ void annotateNextTask(
  TaskId taskId,
  std::vector<void *> inputs,
  std::vector<void *> outputs,
  cudaStream_t stream
) {
  // Create annotation structure to store task metadata
  TaskAnnotation taskAnnotation;
  taskAnnotation.taskId = taskId;

  // Define a lambda to check if a memory address is managed by our system
  auto isManagedArray = [&](void *arr) {
    return MemoryManager::managedMemoryAddressToIndexMap.count(arr) > 0; //return 0 (not exist) or 1 (exist)
  };

  // Filter out any memory addresses not tracked by our memory manager
  // Unregistered arrays would not be managed by our system, so remove them:
  // 1. std::remove_if moves unmanaged arrays to the end and returns iterator to new logical end
  // 2. std::not_fn(isManagedArray) returns true for addresses NOT in managedMemoryAddressToIndexMap
  // 3. The erase calls then actually remove these unmanaged arrays from the vectors
  auto inputsNewEnd = std::remove_if(inputs.begin(), inputs.end(), std::not_fn(isManagedArray));
  auto outputsNewEnd = std::remove_if(outputs.begin(), outputs.end(), std::not_fn(isManagedArray));
  inputs.erase(inputsNewEnd, inputs.end());
  outputs.erase(outputsNewEnd, outputs.end());

  // Initialize the annotation arrays with zeros to ensure clean state
  // (This is necessary because TaskAnnotation uses fixed-size arrays with MAX_NUM_PTR elements)
  memset(taskAnnotation.inputs, 0, TaskAnnotation::MAX_NUM_PTR * sizeof(void *));
  memset(taskAnnotation.outputs, 0, TaskAnnotation::MAX_NUM_PTR * sizeof(void *));
  
  // Copy the filtered input and output pointers from the dynamic vectors to the fixed-size arrays
  // in the TaskAnnotation structure that will be passed to the CUDA kernel
  memcpy(taskAnnotation.inputs, std::data(inputs), inputs.size() * sizeof(void *));
  memcpy(taskAnnotation.outputs, std::data(outputs), outputs.size() * sizeof(void *));

  // Launch the dummy kernel with the annotation data
  // Parameters: <<<blocks, threads, shared_memory, stream>>>
  // - 1 block, 1 thread (minimal execution footprint)
  // - 0 bytes of shared memory
  // - Same stream as the actual computation
  // - false: Don't try to infer dependencies automatically
  dummyKernelForAnnotation<<<1, 1, 0, stream>>>(taskAnnotation, false);
}

/**
 * @brief Simplified version of annotateNextTask that only records the task ID
 * 
 * This version is used when the system should automatically infer the data
 * dependencies rather than having them explicitly specified. It's a more
 * lightweight annotation when the full dependency tracking is not needed.
 * 
 * @param taskId Identifier for the computational task
 * @param stream CUDA stream where the annotation kernel and computation will run
 */
__host__ void annotateNextTask(
  TaskId taskId,
  cudaStream_t stream
) {
  TaskAnnotation taskAnnotation;
  taskAnnotation.taskId = taskId;
  
  // Launch dummy kernel with inferDataDependency=true to indicate automatic inference
  dummyKernelForAnnotation<<<1, 1, 0, stream>>>(taskAnnotation, true);
}

/**
 * @brief Empty kernel used as a stage separator in the execution timeline
 * 
 * This kernel marks the boundary between different stages of computation.
 * It helps the profiling system identify logical groups of operations.
 */
__global__ void dummyKernelForStageSeparator() {
  return; // Empty kernel - doesn't do any actual computation
}

/**
 * @brief Marks the end of a computational stage in the execution timeline
 * 
 * This function inserts a stage separator marker in the CUDA stream.
 * It helps divide the computation into logical phases for the profiler.
 * For example, it could separate the initialization phase from the main
 * computation phase, or mark iterations in an iterative algorithm.
 * 
 * @param stream CUDA stream where the separator marker will be inserted
 */
__host__ void endStage(
  cudaStream_t stream
) {
  dummyKernelForStageSeparator<<<1, 1, 0, stream>>>();
}

}  // namespace memopt
