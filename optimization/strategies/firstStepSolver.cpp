#include "firstStepSolver.hpp"

#include <fmt/core.h>

#include "../../utilities/configurationManager.hpp"
#include "../../utilities/logger.hpp"

namespace memopt {

/**
 * Constructor - takes ownership of input parameters
 * 
 * @param input An rvalue reference to the Input structure
 * 
 * The constructor uses std::move to efficiently transfer ownership of resources from 
 * the input parameter to the internal input member variable. This avoids copying the 
 * potentially large vectors contained in the Input struct (edges and dataDependencyOverlapInBytes).
 * 
 * Without std::move, we would create unnecessary copies of these vectors, which would:
 * 1. Consume additional memory (temporary copies)
 * 2. Waste CPU time copying data
 * 3. Potentially fragment memory
 * 
 * With std::move, we're essentially "stealing" the resources from the input parameter 
 * (which is about to be destroyed anyway since it's an rvalue reference) and giving 
 * ownership to our class member, avoiding all the overhead of copying.
 */
FirstStepSolver::FirstStepSolver(FirstStepSolver::Input &&input) {
  this->input = std::move(input);  // Transfer ownership of input's resources to member variable
}

/**
 * Main solver method that finds the optimal task execution order
 * to maximize memory overlap between consecutive tasks
 */
FirstStepSolver::Output FirstStepSolver::solve() {
  // Initialize tracking arrays
  this->visited.resize(this->input.n, false);  // Tracks visited nodes during traversal

  // Calculate initial in-degree for each node
  // In-degree represents how many tasks must complete before this task can execute
  this->inDegree.resize(this->input.n, 0);
  for (int u = 0; u < this->input.n; u++) {
    for (auto v : this->input.edges[u]) {
      this->inDegree[v]++;  // v has u as a prerequisite
    }
  }

  // Initialize solution parameters
  this->maxTotalOverlap = 0;  // Start with zero overlap
  this->currentTopologicalSort.clear();  // Clear the working solution

  // Start DFS exploration of all valid task orderings
  dfs(0);  // Start with 0 overlap

  // Output solution for debugging
  this->printSolution();

  return this->output;
}

/**
 * Recursive Depth-First Search function that explores all valid topological orderings
 * using backtracking to find the ordering with maximum memory overlap
 * 
 * @param currentTotalOverlap Running total of memory overlap for the current path
 */
void FirstStepSolver::dfs(size_t currentTotalOverlap) {
  // Base case: if we've scheduled all tasks, we have a complete solution
  if (this->currentTopologicalSort.size() == this->input.n) {
    // Update best solution if current solution has greater or equal overlap
    // Must accept solution update to handle the situation
    // when the best total overlap is zero.
    if (currentTotalOverlap >= this->maxTotalOverlap) {
      this->maxTotalOverlap = currentTotalOverlap;
      this->output.taskGroupExecutionOrder = this->currentTopologicalSort;
    }
    return;
  }

  // Try each task that has no remaining dependencies (in-degree = 0)
  for (int u = 0; u < this->input.n; u++) {
    // Skip if this task still has dependencies or has already been processed
    if (this->inDegree[u] != 0 || this->visited[u]) continue;

    // Calculate memory overlap with the previously scheduled task (if any)
    size_t totalOverlapIncrease = 0;
    if (this->currentTopologicalSort.size() > 0) {
      // Get the last task in our current ordering and calculate overlap with task u
      totalOverlapIncrease = this->input.dataDependencyOverlapInBytes[*(this->currentTopologicalSort.rbegin())][u];
    }

    // Temporarily update the in-degree of tasks that depend on u
    // (simulating u's completion for this branch of exploration)
    for (auto v : this->input.edges[u]) {
      this->inDegree[v]--;
    }

    // Add u to the current solution path
    this->visited[u] = true;
    this->currentTopologicalSort.push_back(u);
    
    // Recursively explore this solution path
    dfs(currentTotalOverlap + totalOverlapIncrease);
    
    // Backtrack: remove u from the solution path
    this->visited[u] = false;
    this->currentTopologicalSort.pop_back();

    // Restore the in-degree values (undo the temporary changes)
    for (auto v : this->input.edges[u]) {
      this->inDegree[v]++;
    }
  }
}

/**
 * Writes the solution to a debug file for analysis
 */
void FirstStepSolver::printSolution() {
  // Check if debug output is enabled in configuration
  if (!ConfigurationManager::getConfig().execution.enableDebugOutput) {
    return;
  }
  
  // Format the output filename using the stage index
  std::string outputFilePath = fmt::format("debug/{}.firstStepSolver.out", input.stageIndex);
  LOG_TRACE_WITH_INFO("Printing solution to %s", outputFilePath.c_str());

  // Open file and write solution details
  auto fp = fopen(outputFilePath.c_str(), "w");

  fmt::print(fp, "maxTotalOverlap = {}\n", this->maxTotalOverlap);

  for (int i = 0; i < this->input.n; i++) {
    fmt::print(fp, "taskGroupExecutionOrder[{}] = {}\n", i, this->output.taskGroupExecutionOrder[i]);
  }

  fclose(fp);
}

}  // namespace memopt
