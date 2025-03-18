#pragma once

#include <functional>
#include <map>

#include "optimizationOutput.hpp"

namespace memopt {

/**
 * @brief Callback function type for executing computational tasks
 * 
 * This function is provided by the application and called by the Executor
 * to perform specific computation tasks within an optimized graph.
 * 
 * @param taskId The identifier of the task to execute
 * @param addressMapping Mapping from original memory addresses to current locations (for data that has been moved)
 * @param stream CUDA stream on which to execute the task
 */
typedef std::function<void(int taskId, std::map<void *, void *> addressMapping, cudaStream_t stream)> ExecuteRandomTask;
typedef std::function<void(int taskId, cudaStream_t stream)> ExecuteRandomTaskBase;

/**
 * @brief Callback function type for determining when to stop iterative execution
 * 
 * This function is used with executeOptimizedGraphRepeatedly to control
 * when the execution loop should terminate.
 * 
 * @return true if execution should continue, false if it should stop
 */
typedef std::function<bool()> ShouldContinue;

/**
 * @brief Executes optimized CUDA graphs with memory management
 *
 * The Executor is responsible for running computation graphs that have been
 * optimized to reduce GPU memory usage. It handles:
 * 
 * 1. Data movement between GPU and storage (host memory or secondary GPU)
 * 2. Task scheduling according to the optimized execution plan
 * 3. Dynamic memory allocation and deallocation during execution
 * 4. Address translation between original and current memory locations
 * 
 * This class enables execution of computations that would otherwise exceed
 * available GPU memory by intelligently managing data movement operations.
 */
class Executor {
 public:
  /**
   * @brief Gets the singleton instance of the Executor
   * 
   * @return Pointer to the Executor instance
   */
  static Executor *getInstance();
  
  /**
   * @brief Copy constructor (deleted to enforce singleton pattern)
   */
  Executor(Executor &other) = delete;
  
  /**
   * @brief Assignment operator (deleted to enforce singleton pattern)
   */
  void operator=(const Executor &) = delete;

  /**
   * @brief Executes an optimized computation graph once
   * 
   * This method runs a graph that has been optimized to reduce memory usage
   * by dynamically managing data transfers between the main GPU and storage.
   * 
   * @param optimizedGraph The optimized computation graph to execute
   * @param executeRandomTask Callback function to execute specific tasks
   * @param runningTime Output parameter for recording execution time
   * @param managedDeviceArrayToHostArrayMap Mapping between device and storage memory addresses
   */
  void executeOptimizedGraph(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTask executeRandomTask,
    float &runningTime,
    std::map<void *, void *> &managedDeviceArrayToHostArrayMap
  );
  /**
   * @brief Executes an optimized computation graph once
   * 
   * This method runs a graph that has been optimized to reduce memory usage
   * by dynamically managing data transfers between the main GPU and storage.
   * 
   * @param optimizedGraph The optimized computation graph to execute
   * @param executeRandomTask Callback function to execute specific tasks
   * @param runningTime Output parameter for recording execution time
   * @param managedDeviceArrayToHostArrayMap Mapping between device and storage memory addresses
   */
  void executeOptimizedGraph(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTaskBase executeRandomTaskBase,
    float &runningTime,
    std::map<void *, void *> &managedDeviceArrayToHostArrayMap
  );
  /**
   * @brief Executes an optimized computation graph repeatedly
   * 
   * Similar to executeOptimizedGraph, but runs the graph multiple times
   * until the shouldContinue callback returns false. This is useful for:
   * 1. Iterative algorithms that converge over multiple passes
   * 2. Benchmarking with multiple runs
   * 3. Processing streaming data in batches
   * 
   * @param optimizedGraph The optimized computation graph to execute
   * @param executeRandomTask Callback function to execute specific tasks
   * @param shouldContinue Function that determines when to stop execution
   * @param numIterations Output parameter counting iterations performed
   * @param runningTime Output parameter for recording execution time
   * @param managedDeviceArrayToHostArrayMap Mapping between device and storage memory addresses
   */
  void executeOptimizedGraphRepeatedly(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTask executeRandomTask,
    ShouldContinue shouldContinue,
    int &numIterations,
    float &runningTime,
    std::map<void *, void *> &managedDeviceArrayToHostArrayMap
  );

 protected:
  /**
   * @brief Default constructor (protected to enforce singleton pattern)
   */
  Executor() = default;
  
  /**
   * @brief Singleton instance pointer
   */
  static Executor *instance;
};

}  // namespace memopt
