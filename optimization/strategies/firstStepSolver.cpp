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

  // Use iterative DFS instead of recursive to avoid stack overflow
  // and enable easier optimization in the future
  dfsIterative();

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

    // Calculate memory overlap with previously scheduled tasks (gap-aware)
    size_t totalOverlapIncrease = 0;
    if (this->currentTopologicalSort.size() > 0) {
      // Overlap with immediately previous task (full weight)
      int lastTask = *(this->currentTopologicalSort.rbegin());
      totalOverlapIncrease = this->input.dataDependencyOverlapInBytes[lastTask][u];
      
      // Overlap with second-to-last task (gap overlap with 0.5 weight)
      if (this->currentTopologicalSort.size() >= 2) {
        int secondLastTask = this->currentTopologicalSort[this->currentTopologicalSort.size() - 2];
        size_t gapOverlap = this->input.dataDependencyOverlapInBytes[secondLastTask][u];
        totalOverlapIncrease += static_cast<size_t>(gapOverlap * 0.5);
      }
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
 * Iterative DFS implementation using explicit stack instead of recursion
 * This avoids stack overflow for large graphs and enables easier optimization
 */
void FirstStepSolver::dfsIterative() {
  // Create explicit stack for DFS
  std::stack<DFSState> stateStack;
  
  // Initialize and push the initial state
  DFSState initialState;
  initialState.nextTaskToTry = 0;
  initialState.currentOverlap = 0;
  initialState.currentPath.clear();
  initialState.inDegree = this->inDegree;  // Copy initial in-degree
  initialState.visited = this->visited;    // Copy initial visited (all false)
  
  stateStack.push(initialState);
  
  // Main iterative DFS loop
  while (!stateStack.empty()) {
    DFSState& currentState = stateStack.top();
    
    // Check if we have a complete solution
    if (currentState.currentPath.size() == this->input.n) {
      // Update best solution if current is better or equal
      // Must accept equal to handle case when best overlap is zero
      if (currentState.currentOverlap >= this->maxTotalOverlap) {
        this->maxTotalOverlap = currentState.currentOverlap;
        this->output.taskGroupExecutionOrder = currentState.currentPath;
      }
      stateStack.pop();
      continue;
    }
    
    // Find next unvisited task with in-degree 0
    bool foundNext = false;
    for (int u = currentState.nextTaskToTry; u < this->input.n; u++) {
      // Skip if task has dependencies or already visited
      if (currentState.inDegree[u] != 0 || currentState.visited[u]) continue;
      
      // Found a valid task to schedule next
      foundNext = true;
      
      // Calculate memory overlap with previous tasks (gap-aware)
      size_t overlapIncrease = 0;
      if (!currentState.currentPath.empty()) {
        // Overlap with immediately previous task (full weight)
        int lastTask = currentState.currentPath.back();
        overlapIncrease = this->input.dataDependencyOverlapInBytes[lastTask][u];
        
        // Overlap with second-to-last task (gap overlap with 0.5 weight)
        if (currentState.currentPath.size() >= 2) {
          int secondLastTask = currentState.currentPath[currentState.currentPath.size() - 2];
          size_t gapOverlap = this->input.dataDependencyOverlapInBytes[secondLastTask][u];
          overlapIncrease += static_cast<size_t>(gapOverlap * 0.5);
        }
      }
      
      // Create new state for exploring this branch
      DFSState newState;
      newState.nextTaskToTry = 0;  // Start from beginning for next level
      newState.currentOverlap = currentState.currentOverlap + overlapIncrease;
      newState.currentPath = currentState.currentPath;
      newState.currentPath.push_back(u);
      newState.inDegree = currentState.inDegree;
      newState.visited = currentState.visited;
      
      // Update in-degree for tasks dependent on u
      for (auto v : this->input.edges[u]) {
        newState.inDegree[v]--;
      }
      
      // Mark u as visited
      newState.visited[u] = true;
      
      // Update current state to resume from next task
      currentState.nextTaskToTry = u + 1;
      
      // Push new state to explore this branch
      stateStack.push(newState);
      break;
    }
    
    // If no valid next task found, backtrack
    if (!foundNext) {
      stateStack.pop();
    }
  }
}

/**
 * Writes the solution to a debug file for analysis
 */
void FirstStepSolver::printSolution() {
  // Skip debug output if disabled in configuration
  if (!ConfigurationManager::getConfig().execution.enableDebugOutput) {
    return;
  }
  
  // Format the output filename using the stage index
  std::string outputFilePath = fmt::format("debug/{}.firstStepSolver.out", input.stageIndex);
  LOG_TRACE_WITH_INFO("Printing solution to %s", outputFilePath.c_str());

  // Open file and write solution details
  auto fp = fopen(outputFilePath.c_str(), "w");
  if (!fp) {
    LOG_TRACE_WITH_INFO("Could not open debug output file %s", outputFilePath.c_str());
    return;
  }

  fmt::print(fp, "maxTotalOverlap = {}\n", this->maxTotalOverlap);

  for (int i = 0; i < this->input.n; i++) {
    fmt::print(fp, "taskGroupExecutionOrder[{}] = {}\n", i, this->output.taskGroupExecutionOrder[i]);
  }

  fclose(fp);
}

}  // namespace memopt
