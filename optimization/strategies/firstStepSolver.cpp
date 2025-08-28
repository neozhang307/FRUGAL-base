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

  // Select solver type based on configuration
  std::string solverType = ConfigurationManager::getConfig().optimization.firstStepSolverType;
  
  if (solverType == "BRUTE_FORCE") {
    bruteForceSearch();
  } else if (solverType == "BRANCH_AND_BOUND") {
    dfsIterative();  // Current iterative implementation with branch and bound
  } else if (solverType == "BEAM_SEARCH") {
    beamSearch();
  } else if (solverType == "HYBRID") {
    hybridSearch();
  } else {
    // Default to branch and bound
    LOG_TRACE_WITH_INFO("Unknown solver type %s, defaulting to BRANCH_AND_BOUND", solverType.c_str());
    dfsIterative();
  }

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

    // Calculate memory overlap with previously scheduled tasks (configurable gap-aware)
    size_t totalOverlapIncrease = 0;
    if (this->currentTopologicalSort.size() > 0) {
      // Overlap with immediately previous task (full weight)
      int lastTask = *(this->currentTopologicalSort.rbegin());
      totalOverlapIncrease = this->input.dataDependencyOverlapInBytes[lastTask][u];
      
      // Gap overlap with second-to-last task (if enabled)
      if (ConfigurationManager::getConfig().optimization.enableGapOverlap && 
          this->currentTopologicalSort.size() >= 2) {
        int secondLastTask = this->currentTopologicalSort[this->currentTopologicalSort.size() - 2];
        size_t gapOverlap = this->input.dataDependencyOverlapInBytes[secondLastTask][u];
        double decayFactor = ConfigurationManager::getConfig().optimization.gapOverlapDecayFactor;
        totalOverlapIncrease += static_cast<size_t>(gapOverlap * decayFactor);
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
  
  // Reset statistics (only if debug output is enabled)
  bool trackStats = ConfigurationManager::getConfig().execution.enableDebugOutput;
  if (trackStats) {
    this->totalStatesExplored = 0;
    this->totalStatesPruned = 0;
  }
  
  // Early termination configuration
  bool enableEarlyTermination = ConfigurationManager::getConfig().optimization.enableEarlyTermination;
  int maxIterations = ConfigurationManager::getConfig().optimization.maxSolverIterations;
  int iterations = 0;
  bool earlyTerminated = false;
  std::vector<TaskGroupId> bestPartialSolution;
  std::vector<int> bestPartialInDegree;
  std::vector<bool> bestPartialVisited;
  
  // Main iterative DFS loop
  while (!stateStack.empty()) {
    DFSState& currentState = stateStack.top();
    if (trackStats) this->totalStatesExplored++;
    iterations++;
    
    // Check for early termination
    if (enableEarlyTermination && iterations >= maxIterations) {
      // Save the best partial solution if we haven't found a complete one
      if (this->output.taskGroupExecutionOrder.empty() && 
          currentState.currentPath.size() > bestPartialSolution.size()) {
        bestPartialSolution = currentState.currentPath;
        bestPartialInDegree = currentState.inDegree;
        bestPartialVisited = currentState.visited;
      }
      earlyTerminated = true;
      break;
    }
    
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
    
    // Branch and bound pruning: check if this branch can possibly improve the best solution
    if (this->maxTotalOverlap > 0) {  // Only prune if we have a solution to compare against      
      size_t maxRemainingOverlap = estimateMaxRemainingOverlap(currentState.currentPath, 
                                                               currentState.inDegree, 
                                                               currentState.visited);
      
      // Prune if current + remaining can't beat the best solution
      if (currentState.currentOverlap + maxRemainingOverlap <= this->maxTotalOverlap) {
        if (trackStats) this->totalStatesPruned++;
        stateStack.pop();
        continue;
      }
    }
    
    // Find next unvisited task with in-degree 0
    bool foundNext = false;
    for (int u = currentState.nextTaskToTry; u < this->input.n; u++) {
      // Skip if task has dependencies or already visited
      if (currentState.inDegree[u] != 0 || currentState.visited[u]) continue;
      
      // Found a valid task to schedule next
      foundNext = true;
      
      // Calculate memory overlap with previous tasks (configurable gap-aware)
      size_t overlapIncrease = 0;
      if (!currentState.currentPath.empty()) {
        // Overlap with immediately previous task (full weight)
        int lastTask = currentState.currentPath.back();
        overlapIncrease = this->input.dataDependencyOverlapInBytes[lastTask][u];
        
        // Gap overlap with second-to-last task (if enabled)
        if (ConfigurationManager::getConfig().optimization.enableGapOverlap && 
            currentState.currentPath.size() >= 2) {
          int secondLastTask = currentState.currentPath[currentState.currentPath.size() - 2];
          size_t gapOverlap = this->input.dataDependencyOverlapInBytes[secondLastTask][u];
          double decayFactor = ConfigurationManager::getConfig().optimization.gapOverlapDecayFactor;
          overlapIncrease += static_cast<size_t>(gapOverlap * decayFactor);
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
  
  // Handle early termination
  if (earlyTerminated) {
    if (!this->output.taskGroupExecutionOrder.empty()) {
      // We found at least one complete solution
      fprintf(stderr, "[WARNING] Early termination triggered after %d iterations (limit: %d). Found complete solution but may not be optimal.\n", 
              iterations, maxIterations);
      LOG_TRACE_WITH_INFO("Early termination after %d iterations with complete solution", iterations);
    } else if (!bestPartialSolution.empty()) {
      // No complete solution found, use partial and complete it
      fprintf(stderr, "[ERROR] Early termination triggered after %d iterations (limit: %d). Only found partial solution with %zu/%d tasks. Completing with random topological order.\n",
              iterations, maxIterations, bestPartialSolution.size(), this->input.n);
      LOG_TRACE_WITH_INFO("Early termination after %d iterations. Found partial solution with %zu/%d tasks. Completing with random valid topological order.", 
                         iterations, bestPartialSolution.size(), this->input.n);
      completePartialSolution(bestPartialSolution, bestPartialInDegree, bestPartialVisited);
      this->output.taskGroupExecutionOrder = bestPartialSolution;
      this->maxTotalOverlap = 0; // Mark as non-optimal solution
    } else {
      // No solution at all, create a random valid topological order
      fprintf(stderr, "[ERROR] Early termination triggered after %d iterations (limit: %d). No solution found at all. Creating random valid topological order.\n",
              iterations, maxIterations);
      LOG_TRACE_WITH_INFO("Early termination after %d iterations with no solution. Creating random valid topological order.", iterations);
      std::vector<TaskGroupId> randomSolution;
      std::vector<int> tempInDegree = this->inDegree;
      std::vector<bool> tempVisited = this->visited;
      completePartialSolution(randomSolution, tempInDegree, tempVisited);
      this->output.taskGroupExecutionOrder = randomSolution;
      this->maxTotalOverlap = 0;
    }
  }
}

/**
 * Estimates the maximum possible remaining overlap for branch and bound pruning
 * 
 * This function calculates an upper bound on the total overlap that can be achieved
 * from the current partial solution. It considers both immediate overlap (with the
 * last task) and gap overlap (with the second-to-last task) when enabled.
 * 
 * Memory-optimized version that doesn't create temporary vectors.
 */
size_t FirstStepSolver::estimateMaxRemainingOverlap(const std::vector<TaskGroupId>& currentPath, 
                                                    const std::vector<int>& inDegree,
                                                    const std::vector<bool>& visited) {
  size_t maxPossibleOverlap = 0;
  
  // If no tasks scheduled yet, return 0 (no overlap possible)
  if (currentPath.empty()) {
    return 0;
  }
  
  // Get the last task in the current path
  int lastTask = currentPath.back();
  int secondLastTask = (currentPath.size() >= 2) ? currentPath[currentPath.size() - 2] : -1;
  
  // Cache configuration values to avoid repeated lookups
  bool gapOverlapEnabled = ConfigurationManager::getConfig().optimization.enableGapOverlap;
  double decayFactor = gapOverlapEnabled ? ConfigurationManager::getConfig().optimization.gapOverlapDecayFactor : 0.0;
  
  // For each remaining available task, calculate the maximum overlap it could contribute
  for (int u = 0; u < this->input.n; u++) {
    // Skip if task has dependencies or already visited
    if (inDegree[u] != 0 || visited[u]) continue;
    
    size_t taskOverlap = 0;
    
    // Immediate overlap with last task (full weight)
    taskOverlap = this->input.dataDependencyOverlapInBytes[lastTask][u];
    
    // Gap overlap with second-to-last task (if enabled and applicable)
    if (gapOverlapEnabled && secondLastTask >= 0) {
      size_t gapOverlap = this->input.dataDependencyOverlapInBytes[secondLastTask][u];
      taskOverlap += static_cast<size_t>(gapOverlap * decayFactor);
    }
    
    // Accumulate the maximum possible overlap
    maxPossibleOverlap += taskOverlap;
  }
  
  return maxPossibleOverlap;
}

/**
 * Brute force search - exhaustive search without any pruning optimizations
 * This serves as a baseline for performance comparison with other algorithms
 */
void FirstStepSolver::bruteForceSearch() {
  // Simply call the iterative DFS without any branch and bound pruning
  // We'll modify dfsIterative to have a parameter to disable pruning
  std::stack<DFSState> stateStack;
  
  // Initialize and push the initial state
  DFSState initialState;
  initialState.nextTaskToTry = 0;
  initialState.currentOverlap = 0;
  initialState.currentPath.clear();
  initialState.inDegree = this->inDegree;  // Copy initial in-degree
  initialState.visited = this->visited;    // Copy initial visited (all false)
  
  stateStack.push(initialState);
  
  // Reset statistics 
  bool trackStats = ConfigurationManager::getConfig().execution.enableDebugOutput;
  if (trackStats) {
    this->totalStatesExplored = 0;
    this->totalStatesPruned = 0;  // Will be 0 for brute force
  }
  
  // Early termination configuration
  bool enableEarlyTermination = ConfigurationManager::getConfig().optimization.enableEarlyTermination;
  int maxIterations = ConfigurationManager::getConfig().optimization.maxSolverIterations;
  int iterations = 0;
  bool earlyTerminated = false;
  std::vector<TaskGroupId> bestPartialSolution;
  std::vector<int> bestPartialInDegree;
  std::vector<bool> bestPartialVisited;
  
  // Main iterative DFS loop WITHOUT pruning
  while (!stateStack.empty()) {
    DFSState& currentState = stateStack.top();
    if (trackStats) this->totalStatesExplored++;
    iterations++;
    
    // Check for early termination
    if (enableEarlyTermination && iterations >= maxIterations) {
      // Save the best partial solution if we haven't found a complete one
      if (this->output.taskGroupExecutionOrder.empty() && 
          currentState.currentPath.size() > bestPartialSolution.size()) {
        bestPartialSolution = currentState.currentPath;
        bestPartialInDegree = currentState.inDegree;
        bestPartialVisited = currentState.visited;
      }
      earlyTerminated = true;
      break;
    }
    
    // Check if we have a complete solution
    if (currentState.currentPath.size() == this->input.n) {
      // Update best solution if current is better or equal
      if (currentState.currentOverlap >= this->maxTotalOverlap) {
        this->maxTotalOverlap = currentState.currentOverlap;
        this->output.taskGroupExecutionOrder = currentState.currentPath;
      }
      stateStack.pop();
      continue;
    }
    
    // NO BRANCH AND BOUND PRUNING - explore all paths
    
    // Find next unvisited task with in-degree 0
    bool foundNext = false;
    for (int u = currentState.nextTaskToTry; u < this->input.n; u++) {
      // Skip if task has dependencies or already visited
      if (currentState.inDegree[u] != 0 || currentState.visited[u]) continue;
      
      // Found a valid task to schedule next
      foundNext = true;
      
      // Calculate memory overlap with previous tasks (configurable gap-aware)
      size_t overlapIncrease = 0;
      if (!currentState.currentPath.empty()) {
        // Overlap with immediately previous task (full weight)
        int lastTask = currentState.currentPath.back();
        overlapIncrease = this->input.dataDependencyOverlapInBytes[lastTask][u];
        
        // Gap overlap with second-to-last task (if enabled)
        if (ConfigurationManager::getConfig().optimization.enableGapOverlap && 
            currentState.currentPath.size() >= 2) {
          int secondLastTask = currentState.currentPath[currentState.currentPath.size() - 2];
          size_t gapOverlap = this->input.dataDependencyOverlapInBytes[secondLastTask][u];
          double decayFactor = ConfigurationManager::getConfig().optimization.gapOverlapDecayFactor;
          overlapIncrease += static_cast<size_t>(gapOverlap * decayFactor);
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
  
  // Handle early termination
  if (earlyTerminated) {
    if (!this->output.taskGroupExecutionOrder.empty()) {
      // We found at least one complete solution
      fprintf(stderr, "[WARNING] Brute force early termination triggered after %d iterations (limit: %d). Found complete solution but may not be optimal.\n", 
              iterations, maxIterations);
      LOG_TRACE_WITH_INFO("Brute force early termination after %d iterations with complete solution", iterations);
    } else if (!bestPartialSolution.empty()) {
      // No complete solution found, use partial and complete it
      fprintf(stderr, "[ERROR] Brute force early termination triggered after %d iterations (limit: %d). Only found partial solution with %zu/%d tasks. Completing with random topological order.\n",
              iterations, maxIterations, bestPartialSolution.size(), this->input.n);
      LOG_TRACE_WITH_INFO("Brute force early termination after %d iterations. Found partial solution with %zu/%d tasks.", 
                         iterations, bestPartialSolution.size(), this->input.n);
      completePartialSolution(bestPartialSolution, bestPartialInDegree, bestPartialVisited);
      this->output.taskGroupExecutionOrder = bestPartialSolution;
      this->maxTotalOverlap = 0; // Mark as non-optimal solution
    } else {
      // No solution at all, create a random valid topological order
      fprintf(stderr, "[ERROR] Brute force early termination triggered after %d iterations (limit: %d). No solution found at all. Creating random valid topological order.\n",
              iterations, maxIterations);
      LOG_TRACE_WITH_INFO("Brute force early termination after %d iterations with no solution.", iterations);
      std::vector<TaskGroupId> randomSolution;
      std::vector<int> tempInDegree = this->inDegree;
      std::vector<bool> tempVisited = this->visited;
      completePartialSolution(randomSolution, tempInDegree, tempVisited);
      this->output.taskGroupExecutionOrder = randomSolution;
      this->maxTotalOverlap = 0;
    }
  }
}

/**
 * Beam search with limited beam width
 * TODO: Implement beam search algorithm
 */
void FirstStepSolver::beamSearch() {
  // For now, fall back to branch and bound
  // TODO: Implement actual beam search with configurable beam width
  dfsIterative();
}

/**
 * Hybrid solver that selects the best approach based on problem size
 * TODO: Implement hybrid selection logic
 */
void FirstStepSolver::hybridSearch() {
  // For now, use problem size to select algorithm
  // TODO: Implement more sophisticated selection heuristics
  if (this->input.n <= 10) {
    bruteForceSearch();  // Small problems - use brute force
  } else if (this->input.n <= 50) {
    dfsIterative();      // Medium problems - use branch and bound  
  } else {
    beamSearch();        // Large problems - use beam search (when implemented)
  }
}

/**
 * Complete a partial solution with a random valid topological order
 * This is used when early termination occurs and we need to fill in remaining tasks
 */
void FirstStepSolver::completePartialSolution(std::vector<TaskGroupId>& partialSolution,
                                              std::vector<int>& currentInDegree,
                                              std::vector<bool>& currentVisited) {
  // Keep adding tasks that have no dependencies until all are scheduled
  while (partialSolution.size() < this->input.n) {
    bool foundTask = false;
    
    // Find any task with in-degree 0 that hasn't been visited
    for (int u = 0; u < this->input.n; u++) {
      if (currentInDegree[u] == 0 && !currentVisited[u]) {
        // Add this task to the solution
        partialSolution.push_back(u);
        currentVisited[u] = true;
        
        // Update in-degree for dependent tasks
        for (auto v : this->input.edges[u]) {
          currentInDegree[v]--;
        }
        
        foundTask = true;
        break;  // Just take the first valid one (could randomize this later)
      }
    }
    
    if (!foundTask) {
      // This shouldn't happen if the graph is valid
      fprintf(stderr, "[ERROR] completePartialSolution: No valid task found, graph may have cycles!\n");
      break;
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
  
  // Print branch and bound statistics
  fmt::print(fp, "totalStatesExplored = {}\n", this->totalStatesExplored);
  fmt::print(fp, "totalStatesPruned = {}\n", this->totalStatesPruned);
  if (this->totalStatesExplored > 0) {
    double pruningRatio = (double)this->totalStatesPruned / (double)this->totalStatesExplored * 100.0;
    fmt::print(fp, "pruningRatio = {:.2f}%\n", pruningRatio);
  }

  for (int i = 0; i < this->input.n; i++) {
    fmt::print(fp, "taskGroupExecutionOrder[{}] = {}\n", i, this->output.taskGroupExecutionOrder[i]);
  }

  fclose(fp);
}

}  // namespace memopt
