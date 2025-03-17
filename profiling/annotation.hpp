#pragma once

#include <vector>

#include "../utilities/types.hpp"

namespace memopt {

/**
 * @brief Structure to hold metadata about computational tasks for profiling
 * 
 * This structure stores information about a computational task:
 * - The task's unique identifier
 * - Arrays that are read by the task (inputs)
 * - Arrays that are written by the task (outputs)
 * 
 * This metadata is used by the memory optimization system to analyze data
 * dependencies between tasks and make intelligent decisions about when to
 * move data between the GPU and storage (host or secondary GPU).
 */
struct TaskAnnotation {
  // Before CUDA 12.1, the size of the parameters passed to a kernel is
  // limited to 4096 bytes. CUDA 12.1 increases this limit to 32764 bytes.
  // We calculate max pointers based on this limit, accounting for TaskId
  static constexpr size_t MAX_NUM_PTR = (4096 - sizeof(TaskId)) / 8 / 2;

  TaskId taskId;           ///< Unique identifier for the computational task

  void *inputs[MAX_NUM_PTR];   ///< Arrays read by the task (inputs)
  void *outputs[MAX_NUM_PTR];  ///< Arrays written by the task (outputs)
};

/**
 * @brief Empty CUDA kernel that carries task annotation information
 * 
 * This kernel doesn't perform any actual computation. It's sole purpose
 * is to be a marker in the CUDA execution stream that carries metadata
 * about computational tasks for profiling.
 * 
 * @param taskAnnotation Structure containing task ID and memory access patterns
 * @param inferDataDependency Whether to infer dependencies automatically
 */
__global__ void dummyKernelForAnnotation(TaskAnnotation taskAnnotation, bool inferDataDependency);

/**
 * @brief Annotates a computational task with explicit data dependencies
 * 
 * This function is called before executing a computational task to record:
 * 1. The task's unique identifier
 * 2. Arrays that are read by the task (inputs)
 * 3. Arrays that are written by the task (outputs)
 * 
 * The function inserts a "dummy" kernel into the CUDA stream that carries
 * this metadata. During profiling, these annotations are captured and analyzed
 * to understand the data flow patterns of the computation.
 * 
 * @param taskId Unique identifier for the task
 * @param inputs Arrays that will be read by the task
 * @param outputs Arrays that will be written by the task
 * @param stream CUDA stream where the task will execute
 */
__host__ void annotateNextTask(
  TaskId taskId,
  std::vector<void *> inputs,
  std::vector<void *> outputs,
  cudaStream_t stream
);

/**
 * @brief Annotates a computational task with automatic dependency inference
 * 
 * Simplified version of annotateNextTask that only records the task ID
 * and relies on the system to automatically infer data dependencies.
 * Used when explicit dependency specification is not necessary.
 * 
 * @param taskId Unique identifier for the task
 * @param stream CUDA stream where the task will execute
 */
__host__ void annotateNextTask(
  TaskId taskId,
  cudaStream_t stream
);

/**
 * @brief Empty kernel used to mark the boundaries between computational stages
 * 
 * This kernel serves as a stage separator in the execution timeline,
 * helping the profiling system identify logical groups of operations.
 * 
 * Stages are a user-defined concept that divide a CUDA graph into separate
 * execution phases, allowing the memory optimizer to process each stage
 * independently. This separation is crucial for applications where:
 * 
 * 1. Different phases have different memory access patterns or requirements
 * 2. Memory can be completely cleared between phases
 * 3. The overall graph is too complex to optimize as a single unit
 * 4. Different optimization strategies are appropriate for different phases
 */
__global__ void dummyKernelForStageSeparator();

/**
 * @brief Marks the end of a computational stage in the execution timeline
 * 
 * Inserts a stage separator marker in the CUDA stream to divide the
 * computation into logical phases for the profiler. For example,
 * separating initialization from main computation, or marking iterations.
 * 
 * Stage separation is a powerful user-controlled optimization technique that:
 * - Allows the optimizer to find optimal memory management plans for each stage separately
 * - Reduces the complexity of the optimization problem by breaking it into manageable chunks
 * - Enables a complete memory reset between stages, potentially freeing all device memory
 * - Provides better optimization results for applications with distinct phases that have
 *   different memory access patterns (e.g., setup, computation, finalization)
 * 
 * Use stage separators strategically at natural boundaries in your application where
 * memory can be reorganized without affecting performance.
 * 
 * @param stream CUDA stream where the marker will be inserted
 */
__host__ void endStage(
  cudaStream_t stream
);

}  // namespace memopt
