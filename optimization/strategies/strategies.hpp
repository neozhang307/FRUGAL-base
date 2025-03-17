#pragma once

#include "../optimizer.hpp"

namespace memopt {

/**
 * @brief Baseline strategy that executes tasks without memory optimization
 * 
 * The NoOptimizationStrategy serves as a reference implementation that:
 * - Allocates all data on the GPU at the start of execution
 * - Does not generate any prefetch or offload operations
 * - Preserves the original execution order without attempting to optimize
 * 
 * This strategy is useful for:
 * - Providing a baseline to measure the effectiveness of optimization
 * - Handling cases where the entire workload fits in GPU memory
 * - Debugging the system without the complexity of memory optimization
 * - Isolating whether issues are related to memory management or other factors
 * 
 * Important: This strategy can only run applications where the total memory 
 * requirements fit within available GPU memory. Larger applications would 
 * result in out-of-memory errors.
 */
class NoOptimizationStrategy {
  public:
    /**
     * @brief Creates an execution plan without memory optimization
     * 
     * @param optimizationInput Structure containing task groups and dependencies
     * @return OptimizationOutput Execution plan with all data residing on device
     */
    OptimizationOutput run(OptimizationInput &optimizationInput);
};

/**
 * @brief Advanced memory optimization strategy that separates scheduling from memory management
 * 
 * The TwoStepOptimizationStrategy solves the complex problem of GPU memory optimization
 * by breaking it into two sequential, more manageable steps:
 * 
 * 1. First Step (Task Scheduling):
 *    - Determines the optimal execution order of task groups
 *    - Maximizes data reuse between consecutive tasks
 *    - Respects data dependencies between task groups
 *    - Uses a DFS-based algorithm to explore valid topological orderings
 * 
 * 2. Second Step (Memory Management):
 *    - Given the task execution order, optimizes data movement operations
 *    - Decides which arrays should be initially allocated on the device
 *    - Schedules prefetch operations (host-to-device transfers)
 *    - Schedules offload operations (device-to-host transfers)
 *    - Uses integer linear programming (ILP) to find optimal solutions
 * 
 * This two-step approach provides several advantages:
 * - Reduces problem complexity by tackling scheduling and memory management separately
 * - Uses specialized algorithms optimized for each subproblem
 * - Effectively balances memory usage against execution performance
 * - Enables applications to run even when their memory requirements exceed available GPU memory
 */
class TwoStepOptimizationStrategy {
  public:
    /**
     * @brief Runs the two-step optimization process on the given input
     * 
     * @param optimizationInput Structure containing task groups, dependencies, and memory access patterns
     * @return OptimizationOutput Optimized execution plan with data movement operations
     */
    OptimizationOutput run(OptimizationInput &optimizationInput);
};

}  // namespace memopt
