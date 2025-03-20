#pragma once

#include "executor.hpp"
#include "taskManager.hpp"
#include "optimizationOutput.hpp"

namespace memopt {

OptimizationOutput profileAndOptimize(cudaGraph_t originalGraph);

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

}  // namespace memopt
