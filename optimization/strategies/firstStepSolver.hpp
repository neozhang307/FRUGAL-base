#pragma once

#include <vector>
#include <stack>

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
  /**
   * @struct DFSState
   * @brief State information for iterative DFS traversal
   */
  struct DFSState {
    int nextTaskToTry;                      // Next task index to evaluate (for resuming)
    size_t currentOverlap;                  // Running total overlap at this state
    std::vector<TaskGroupId> currentPath;   // Current partial solution path
    std::vector<int> inDegree;              // In-degree state at this level
    std::vector<bool> visited;              // Visited state at this level
  };

  Input input;                                  // Solver input parameters
  Output output;                                // Solution being constructed
  size_t maxTotalOverlap;                       // Best overlap found so far
  std::vector<bool> visited;                    // Tracks visited nodes during DFS
  std::vector<int> inDegree;                    // In-degree count for each node (for topological ordering)
  std::vector<TaskGroupId> currentTopologicalSort; // Current task order being explored
  
  // Branch and bound statistics
  size_t totalStatesExplored = 0;               // Total states processed
  size_t totalStatesPruned = 0;                 // States pruned by branch and bound

  /**
   * @brief Recursive DFS to explore all valid topological orderings
   * @param currentTotalOverlap Running sum of overlap for the current ordering
   * 
   * Uses backtracking to find the ordering with maximum data dependency overlap
   * while respecting the dependency constraints in the directed acyclic graph
   */
  void dfs(size_t currentTotalOverlap);

  /**
   * @brief Iterative DFS to explore all valid topological orderings
   * 
   * Stack-based implementation that avoids recursion depth limits and provides
   * better control for future optimizations like pruning and early termination.
   */
  void dfsIterative();

  /**
   * @brief Estimates the maximum possible remaining overlap for branch and bound pruning
   * @param currentPath Current partial solution path
   * @param inDegree Current in-degree state for dependency checking
   * @param visited Current visited state for availability checking
   * @return Upper bound on remaining overlap that can be achieved
   */
  size_t estimateMaxRemainingOverlap(const std::vector<TaskGroupId>& currentPath, 
                                     const std::vector<int>& inDegree,
                                     const std::vector<bool>& visited);

  /**
   * @brief Brute force search - exhaustive search without pruning
   */
  void bruteForceSearch();

  /**
   * @brief Beam search with limited beam width
   */
  void beamSearch();

  /**
   * @brief Hybrid solver that selects the best approach based on problem size
   */
  void hybridSearch();
  
  /**
   * @brief Complete a partial solution with a random valid topological order
   * @param partialSolution The partial solution to complete
   * @param currentInDegree Current in-degree state
   * @param currentVisited Current visited state
   */
  void completePartialSolution(std::vector<TaskGroupId>& partialSolution,
                               std::vector<int>& currentInDegree,
                               std::vector<bool>& currentVisited);
  
  /**
   * @brief Calculate total overlap for a given task execution order
   * @param taskOrder The execution order to evaluate
   * @return Total memory overlap in bytes
   */
  size_t calculateTotalOverlap(const std::vector<TaskGroupId>& taskOrder);
  
  /**
   * @brief Outputs the solution to a debug file
   */
  void printSolution();
};

}  // namespace memopt
