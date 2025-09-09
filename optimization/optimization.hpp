#pragma once

#include "executor.hpp"
#include "taskManager.hpp"
#include "taskManager_v2.hpp"
#include "optimizationOutput.hpp"

namespace memopt {

OptimizationOutput profileAndOptimize(cudaGraph_t originalGraph);

/// @brief Profile and optimize tasks stage-by-stage to handle memory constraints
/// @param taskManager The task manager containing all registered tasks
/// @param stageTaskIds Vector of vectors where each inner vector contains TaskIds for one stage
/// @param stream Optional CUDA stream to use (will create one if not provided)
/// @return Optimized graph with all stages merged
/// @details This function constructs subgraphs for each stage by capturing task execution,
///          profiles each stage individually to avoid memory issues, optimizes all stages
///          in batch (enabling future CPU/GPU pipelining), and merges the results.
OptimizationOutput profileAndOptimizeStaged(
  TaskManager_v2& taskManager,
  const std::vector<std::vector<TaskId>>& stageTaskIds,
  cudaStream_t stream = nullptr
);

/// @brief Profile and optimize pre-constructed stage subgraphs
/// @param stageGraphs Vector of CUDA graphs, one per stage
/// @param enablePipelining Reserved for future use - enables GPU/CPU pipelining
/// @return Optimized graph with all stages merged
/// @details This overload accepts pre-constructed subgraphs for cases where the user
///          has already built the stage graphs. Each graph is profiled and optimized
///          separately, then merged into a single optimization output.
OptimizationOutput profileAndOptimizeStaged(
  const std::vector<cudaGraph_t>& stageGraphs,
  bool enablePipelining = false
);

/// @brief Merges multiple optimization outputs from different stages into one
/// @param optimizationOutputs Vector of optimization outputs from each stage
/// @return Merged optimization output with stages connected sequentially
OptimizationOutput mergeOptimizationOutputs(std::vector<OptimizationOutput> &optimizationOutputs);

/// @brief
/// @param optimizedGraph
/// @param executeRandomTask
/// @param runningTime
///   (Output) The running time
/// @param memManager
///   MemoryManager instance to use for memory management
void executeOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime,
  MemoryManager &memManager
);

/// @brief
/// @param optimizedGraph
/// @param executeRandomTask
/// @param runningTime
///   (Output) The running time
void executeOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime
);

/// @brief
/// @param optimizedGraph
/// @param executeRandomTask
/// @param runningTime
///   (Output) The running time
/// @param managedDeviceArrayToHostArrayMap
///   (Output) The mapping between old managed device array addresses and
///   new host array addresses where old arrays are moved to.
void executeOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime,
  std::map<void *, void *> &managedDeviceArrayToHostArrayMap
);

/// @brief
/// @param optimizedGraph
/// @param executeRandomTaskBase
/// @param runningTime
///   (Output) The running time
/// @param managedDeviceArrayToHostArrayMap
///   (Output) The mapping between old managed device array addresses and
///   new host array addresses where old arrays are moved to.
void executeOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTaskBase executeRandomTaskBase,
  float &runningTime,
  std::map<void *, void *> &managedDeviceArrayToHostArrayMap
);

/// @brief
/// @param optimizedGraph
/// @param executeRandomTaskBase
/// @param runningTime
///   (Output) The running time
/// @param memManager
///   MemoryManager instance to use for memory management
void executeOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTaskBase executeRandomTaskBase,
  float &runningTime,
  MemoryManager &memManager
);

/// @brief
/// @param optimizedGraph
/// @param executeRandomTask
/// @param shouldContinue
/// @param numIterations
///   (Output) The number of iterations executed
/// @param runningTime
///   (Output) The running time
/// @param memManager
///   MemoryManager instance to use for memory management
void executeOptimizedGraphRepeatedly(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  ShouldContinue shouldContinue,
  int &numIterations,
  float &runningTime,
  MemoryManager &memManager
);

/// @brief
/// @param optimizedGraph
/// @param executeRandomTask
/// @param shouldContinue
/// @param numIterations
///   (Output) The number of iterations executed
/// @param runningTime
///   (Output) The running time
void executeOptimizedGraphRepeatedly(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  ShouldContinue shouldContinue,
  int &numIterations,
  float &runningTime
);

/// @brief
/// @param optimizedGraph
/// @param executeRandomTask
/// @param shouldContinue
/// @param numIterations
///   (Output) The number of iterations executed
/// @param runningTime
///   (Output) The running time
/// @param managedDeviceArrayToHostArrayMap
///   (Output) The mapping between old managed device array addresses and
///   new host array addresses where old arrays are moved to.
void executeOptimizedGraphRepeatedly(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  ShouldContinue shouldContinue,
  int &numIterations,
  float &runningTime,
  std::map<void *, void *> &managedDeviceArrayToHostArrayMap
);

/// @brief Executes an optimized computation graph for evaluation purposes
/// @param optimizedGraph The optimized computation graph to execute
/// @param executeRandomTask Callback to execute specific computation tasks
/// @param runningTime Output parameter to store the execution time
/// @param memManager Reference to the MemoryManager instance to use
void executeOptimizedGraphForEvaluation(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime,
  MemoryManager &memManager
);

/// @brief Executes an optimized computation graph for evaluation purposes using singleton MemoryManager
/// @param optimizedGraph The optimized computation graph to execute
/// @param executeRandomTask Callback to execute specific computation tasks
/// @param runningTime Output parameter to store the execution time
void executeOptimizedGraphForEvaluation(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime
);

/// @brief Executes an optimized computation graph for evaluation with simplified callback
/// @param optimizedGraph The optimized computation graph to execute
/// @param executeRandomTaskBase Simplified callback function to execute specific tasks
/// @param runningTime Output parameter for recording execution time
/// @param memManager Reference to the MemoryManager instance to use
void executeOptimizedGraphForEvaluation(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTaskBase executeRandomTaskBase,
  float &runningTime,
  MemoryManager &memManager
);

/// @brief Executes an optimized computation graph for evaluation with singleton MemoryManager and simplified callback
/// @param optimizedGraph The optimized computation graph to execute
/// @param executeRandomTaskBase Simplified callback function to execute specific tasks
/// @param runningTime Output parameter for recording execution time
void executeOptimizedGraphForEvaluation(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTaskBase executeRandomTaskBase,
  float &runningTime
);

}  // namespace memopt
