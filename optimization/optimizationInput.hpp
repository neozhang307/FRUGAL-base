#pragma once

#include <map>
#include <set>
#include <vector>

#include "../utilities/types.hpp"

namespace memopt {

/**
 * @brief Input structure for memory optimization algorithms
 * 
 * OptimizationInput represents the computational structure and memory access patterns
 * of a CUDA application. It is constructed from profiling data and serves as the
 * foundation for memory optimization decisions. The structure captures task dependencies,
 * execution timing, and memory access patterns that allow the optimizer to intelligently
 * manage device memory and schedule data transfers.
 * 
 * This structure is the bridge between:
 * 1. The profiling phase that collects execution and memory usage data
 * 2. The optimization phase that generates a memory-efficient execution plan
 */
struct OptimizationInput {
  /**
   * @brief A group of related computational tasks
   * 
   * TaskGroups combine individual tasks that are related (execute concurrently
   * or share annotations) to reduce optimization complexity while preserving
   * the essential characteristics needed for memory optimization.
   */
  struct TaskGroup {
    /**
     * @brief Memory access patterns for a task group
     * 
     * Identifies the memory buffers read from (inputs) and written to (outputs)
     * by the tasks in this group. This information is crucial for determining
     * when memory buffers need to be present on the device and when they can
     * be moved to host memory.
     */
    struct DataDependency {
      std::set<void *> inputs;   ///< Memory buffers read by this task group
      std::set<void *> outputs;  ///< Memory buffers written by this task group
    };

    std::set<TaskId> nodes;  ///< Set of task IDs that belong to this task group
    std::map<TaskId, std::vector<TaskId>> edges;  ///< Dependencies between tasks within this task group
    float runningTime;  ///< Execution time of this task group in seconds
    DataDependency dataDependency;  ///< Memory buffers accessed by this task group
  };

  std::vector<TaskGroup> nodes;  ///< Collection of all task groups in the application
  std::map<TaskGroupId, std::vector<TaskGroupId>> edges;  ///< Dependencies between task groups

  /**
   * @brief Total execution time of the original CUDA graph before optimization
   * 
   * Used as a baseline for measuring the performance impact of optimization.
   */
  float originalTotalRunningTime;

  /**
   * @brief Flag indicating if all memory buffers should reside on host at start and end
   * 
   * When true, all memory must be on the host at the beginning and end of execution.
   * This is typically set to true for multi-stage applications where memory is completely
   * reset between stages, allowing for a fresh memory allocation strategy per stage.
   */
  bool forceAllArraysToResideOnHostInitiallyAndFinally;

  /**
   * @brief The execution stage this optimization input represents
   * 
   * For multi-stage applications (separated by stage separator nodes),
   * this identifies which stage this OptimizationInput structure corresponds to.
   * Each stage is optimized independently and later merged into a unified execution plan.
   */
  int stageIndex;
};

}  // namespace memopt
