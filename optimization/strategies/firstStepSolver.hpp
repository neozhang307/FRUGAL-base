#pragma once

#include <vector>

#include "../../utilities/types.hpp"

namespace memopt {

/**
 * @class FirstStepSolver
 * @brief Solves the first step of the memory optimization problem by finding an optimal task execution order
 * 
 * This solver finds a topological ordering of task groups that maximizes data dependency overlap
 * between consecutive tasks. This helps reduce peak memory usage by encouraging memory reuse.
 */
class FirstStepSolver {
 public:
  /**
   * @struct Input
   * @brief Input parameters for the FirstStepSolver
   * 
   * @param n Number of task groups to schedule
   * @param edges Directed graph representation where edges[u][v] means task u must execute before task v
   * @param dataDependencyOverlapInBytes Matrix of memory overlap between each pair of tasks
   * @param stageIndex Identifier for the current optimization stage (used for debugging output)
   */
  struct Input {
    TaskGroupId n;                                      // Number of task groups
    std::vector<std::vector<TaskGroupId>> edges;        // Dependency graph (DAG)
    std::vector<std::vector<size_t>> dataDependencyOverlapInBytes; // Memory overlap between tasks
    int stageIndex;                                     // For debug output naming
  };

  /**
   * @struct Output
   * @brief Output of the FirstStepSolver
   * 
   * @param taskGroupExecutionOrder Ordered sequence of task group IDs representing execution order
   */
  struct Output {
    std::vector<TaskGroupId> taskGroupExecutionOrder;   // Optimized execution order
  };

  /**
   * @brief Constructor for FirstStepSolver
   * @param input Input parameters moved into solver using rvalue reference
   * 
   * This constructor takes an rvalue reference (&&) to enable move semantics.
   * The double ampersand (&&) indicates that this parameter is an rvalue reference,
   * which is a reference type that can bind to temporary objects about to be destroyed.
   * 
   * Inside the constructor, std::move is used to enable efficient transfer of resources
   * from the input parameter to the class member, avoiding expensive copying operations.
   * This is especially important when the Input structure contains large data structures
   * like vectors of vectors.
   */
  FirstStepSolver(Input &&input);
  
  /**
   * @brief Solves the optimization problem to find the best execution order
   * @return Output containing the optimized task execution order
   */
  Output solve();

 private:
  Input input;                                  // Solver input parameters
  Output output;                                // Solution being constructed
  size_t maxTotalOverlap;                       // Best overlap found so far
  std::vector<bool> visited;                    // Tracks visited nodes during DFS
  std::vector<int> inDegree;                    // In-degree count for each node (for topological ordering)
  std::vector<TaskGroupId> currentTopologicalSort; // Current task order being explored

  /**
   * @brief Recursive DFS to explore all valid topological orderings
   * @param currentTotalOverlap Running sum of overlap for the current ordering
   * 
   * Uses backtracking to find the ordering with maximum data dependency overlap
   * while respecting the dependency constraints in the directed acyclic graph
   */
  void dfs(size_t currentTotalOverlap);
  
  /**
   * @brief Outputs the solution to a debug file
   */
  void printSolution();
};

}  // namespace memopt
