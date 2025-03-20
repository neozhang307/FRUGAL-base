#include <map>

#include "../utilities/logger.hpp"
#include "executor.hpp"
#include "optimization.hpp"
#include "optimizer.hpp"

namespace memopt {

OptimizationOutput profileAndOptimize(cudaGraph_t originalGraph) {
  LOG_TRACE();
  return Optimizer::getInstance()->profileAndOptimize(originalGraph);
}

void executeOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime,
  MemoryManager &memManager
) {
  LOG_TRACE();
  Executor::getInstance()->executeOptimizedGraph(
    optimizedGraph,
    executeRandomTask,
    runningTime,
    memManager
  );
}

void executeOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime
) {
  LOG_TRACE();
  Executor::getInstance()->executeOptimizedGraph(
    optimizedGraph,
    executeRandomTask,
    runningTime
  );
}

void executeOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime,
  std::map<void *, void *> &managedDeviceArrayToHostArrayMap
) {
  LOG_TRACE();
  Executor::getInstance()->executeOptimizedGraph(
    optimizedGraph,
    executeRandomTask,
    runningTime,
    managedDeviceArrayToHostArrayMap
  );
}

void executeOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTaskBase executeRandomTaskBase,
  float &runningTime,
  std::map<void *, void *> &managedDeviceArrayToHostArrayMap
) {
  LOG_TRACE();
  Executor::getInstance()->executeOptimizedGraph(
    optimizedGraph,
    executeRandomTaskBase,
    runningTime,
    managedDeviceArrayToHostArrayMap
  );
}

void executeOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTaskBase executeRandomTaskBase,
  float &runningTime,
  MemoryManager &memManager
) {
  LOG_TRACE();
  // Convert the simplified task execution to the full version with address mapping
  Executor::getInstance()->executeOptimizedGraph(
    optimizedGraph,
    [executeRandomTaskBase](int taskId, std::map<void *, void *> addressMapping, cudaStream_t stream) {
      // Ignore the address mapping and call the base task execution function
      executeRandomTaskBase(taskId, stream);
    },
    runningTime,
    memManager
  );
}

void executeOptimizedGraphRepeatedly(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  ShouldContinue shouldContinue,
  int &numIterations,
  float &runningTime,
  MemoryManager &memManager
) {
  LOG_TRACE();
  Executor::getInstance()->executeOptimizedGraphRepeatedly(
    optimizedGraph,
    executeRandomTask,
    shouldContinue,
    numIterations,
    runningTime,
    memManager
  );
}

void executeOptimizedGraphRepeatedly(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  ShouldContinue shouldContinue,
  int &numIterations,
  float &runningTime
) {
  LOG_TRACE();
  Executor::getInstance()->executeOptimizedGraphRepeatedly(
    optimizedGraph,
    executeRandomTask,
    shouldContinue,
    numIterations,
    runningTime
  );
}

void executeOptimizedGraphRepeatedly(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  ShouldContinue shouldContinue,
  int &numIterations,
  float &runningTime,
  std::map<void *, void *> &managedDeviceArrayToHostArrayMap
) {
  LOG_TRACE();
  Executor::getInstance()->executeOptimizedGraphRepeatedly(
    optimizedGraph,
    executeRandomTask,
    shouldContinue,
    numIterations,
    runningTime,
    managedDeviceArrayToHostArrayMap
  );
}

}  // namespace memopt
