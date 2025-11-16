# CUDA Memory Optimization Programming Guide

This guide explains how to use the TaskManager_v2 and memory optimization components of the CUDA Memory Optimization framework.

## Table of Contents

1. [Introduction](#introduction)
2. [TaskManager_v2 Overview](#taskmanager_v2-overview)
3. [Using TaskManager_v2](#using-taskmanager_v2)
4. [Memory Registration](#memory-registration)
5. [Execution Modes](#execution-modes)
6. [Memory Optimization](#memory-optimization)
7. [Graph Profiling and Optimization](#graph-profiling-and-optimization)
8. [Advanced Usage Patterns](#advanced-usage-patterns)
9. [Offline Optimization Workflow](#offline-optimization-workflow)
10. [Configuration Options](#configuration-options)
11. [Best Practices](#best-practices)

## Introduction

The CUDA Memory Optimization framework provides tools for managing tasks and memory in CUDA applications, with a focus on optimizing memory usage for applications that may exceed GPU memory capacity. The two primary components are:

- **TaskManager_v2**: Manages the registration and execution of CUDA tasks with proper memory handling
- **Memory Optimization**: Provides tools for optimizing memory usage through prefetching and offloading

## TaskManager_v2 Overview

TaskManager_v2 is a template-based task management system that allows for type-safe registration and execution of CUDA tasks. Key features include:

- Type-safe task registration with automatic parameter handling
- Support for different execution modes (Basic and Production)
- Built-in task annotation for profiling and optimization
- Automatic parameter transformation via MemoryManager

## Using TaskManager_v2

### Creating a TaskManager

```cpp
// Create a TaskManager_v2 instance (optionally with debug mode)
memopt::TaskManager_v2 taskManager(true);  // true enables debug output
```

### Registering Tasks

```cpp
// Register a task with auto-generated ID
using namespace memopt;
TaskId taskId = taskManager.registerTask<std::function<void(cudaStream_t, double*)>, double*>(
    // Lambda function that takes a stream and parameters
    [](cudaStream_t stream, double* data) {
        // Task implementation
        // Example: Call a CUDA kernel or library function
        someFunction<<<blocks, threads, 0, stream>>>(data);
    },
    {inputPointer},         // Input pointers 
    {outputPointer},        // Output pointers
    TaskManager_v2::makeArgs(data),  // Default arguments
    "MyTask"                // Task name (optional)
);
```

### Executing Tasks

```cpp
// Execute a task with default parameters
cudaStream_t stream;
cudaStreamCreate(&stream);
taskManager.execute(taskId, stream);

// Execute a task with custom parameters
taskManager.executeWithParams(taskId, stream, 
    TaskManager_v2::makeArgs(customData));
```

## Memory Registration

Before using the memory optimization features, you need to register memory addresses with the MemoryManager:

```cpp
// Register memory with the memory manager
size_t size = N * sizeof(double);
double* devicePtr;
cudaMalloc(&devicePtr, size);
memopt::registerManagedMemoryAddress(devicePtr, size);
```

## Execution Modes

TaskManager_v2 supports two execution modes:

### Basic Mode

In Basic mode, pointers are used as-is without transformation.

```cpp
// Set execution mode to Basic (default)
taskManager.setExecutionMode(TaskManager_v2::ExecutionMode::Basic);
```

### Production Mode

In Production mode, all pointers are automatically processed through MemoryManager::getAddress to handle memory optimization:

```cpp
// Set execution mode to Production
taskManager.setExecutionMode(TaskManager_v2::ExecutionMode::Production);
```

## Memory Optimization

### Creating an Optimized Graph

```cpp
// Create a CUDA graph with your operations
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);
// ... add operations to graph ...

// Optimize the graph for memory usage
auto optimizedGraph = profileAndOptimize(graph);
```

### Executing an Optimized Graph

```cpp
// Execute the optimized graph
float runningTime;
auto& memManager = memopt::MemoryManager::getInstance();

executeOptimizedGraph(
    optimizedGraph,
    // Lambda function that uses TaskManager_v2 to execute tasks
    [&taskManager](int taskId, cudaStream_t stream) {
        // Execute the task using TaskManager_v2
        taskManager.execute(taskId, stream);
        return true; // Indicates successful execution
    },
    runningTime,
    memManager
);
```

## Graph Profiling and Optimization

### Understanding Dummy Kernels and ProfilingContext

When profiling CUDA graphs, the framework uses special "dummy kernels" as markers to identify task boundaries and stage separators. These are managed through the ProfilingContext RAII class.

#### What are Dummy Kernels?

Dummy kernels are special marker nodes inserted into CUDA graphs:
- **Annotation Kernels**: Mark task boundaries and carry metadata (inputs, outputs, task IDs)
- **Stage Separator Kernels**: Mark boundaries between optimization stages

```
Regular Graph:           [Kernel A] → [Kernel B] → [Kernel C]
Annotated Graph:        [Annotation] → [Kernel A] → [Kernel B] → [Stage Sep] → [Annotation] → [Kernel C]
```

#### ProfilingContext Usage

The ProfilingContext class manages the lifecycle of dummy kernel handles using RAII:

```cpp
#include "optimization/profilingContext.hpp"

// ProfilingContext automatically registers dummy kernels on creation
// and cleans them up on destruction
{
    memopt::ProfilingContext ctx;  // Registers dummy kernel handles

    // Now you can profile graphs - the handles exist
    auto optimizer = memopt::Optimizer::getInstance();
    auto input = optimizer->profileGraph(graph);
    auto output = optimizer->optimizeGraph(input);

}  // Automatically cleans up handles when ctx goes out of scope
```

**Important**: Only one ProfilingContext should exist at a time to avoid double registration errors.

### Separating Profiling and Optimization

For advanced use cases (like offline optimization), you can separate the profiling and optimization phases:

```cpp
// Step 1: Profile with CUDA (requires GPU)
{
    memopt::ProfilingContext ctx;  // Create context for profiling
    auto optimizer = memopt::Optimizer::getInstance();
    auto input = optimizer->profileGraph(graph);

    // Save profiling data for offline optimization
    saveOptimizationInput(input, "profile.json");
}

// Step 2: Optimize offline (no GPU needed!)
{
    // No ProfilingContext needed for optimization
    auto input = loadOptimizationInput("profile.json");
    auto optimizer = memopt::Optimizer::getInstance();
    auto output = optimizer->optimizeGraph(input);

    // Save the optimized plan
    writeOptimizationOutputToFile(output, "plan.json");
}
```

### Why ProfilingContext is Necessary

The `profileGraph()` function needs to identify special nodes in the graph:

```cpp
// Inside profileGraph, it needs to compare nodes against dummy kernel handles
if (compareKernelNodeFunctionHandle(node, dummyKernelForAnnotationHandle)) {
    // This is an annotation node - extract metadata
}

if (compareKernelNodeFunctionHandle(node, dummyKernelForStageSeparatorHandle)) {
    // This is a stage boundary
}
```

Without ProfilingContext:
- Dummy kernel handles would be NULL or invalid
- Cannot identify annotation nodes → cannot extract task metadata
- Cannot detect multi-stage graphs
- Profiling would fail or produce incorrect results

### Best Practices for ProfilingContext

1. **Create at the highest scope needed** - Don't create multiple contexts in nested functions
2. **Let RAII handle cleanup** - Don't manually manage registration/cleanup
3. **One context at a time** - Never create overlapping ProfilingContext instances
4. **Not needed for optimization** - Only profiling requires the context

## Advanced Usage Patterns

### Custom Parameter Type Handling

TaskManager_v2 automatically handles both pointer and non-pointer types properly. The `TaskFunctionArgs` template provides built-in parameter processing:

```cpp
// Create custom arguments with mixed pointer and non-pointer types
auto args = TaskManager_v2::makeArgs(
    dataPointer,   // Pointer (will be processed with getAddress in Production mode)
    123,           // Non-pointer (will be passed through unchanged)
    anotherPointer // Pointer (will be processed with getAddress in Production mode)
);
```

### Memory Prefetching

For better performance, you can use explicit prefetching:

```cpp
// Prefetch all data to the device
MemoryManager::getInstance().prefetchAllDataToDevice();
```

## Offline Optimization Workflow

The framework supports offline optimization where profiling and optimization can be performed separately. This is useful for:
- Running optimization on different machines (e.g., profile on GPU machine, optimize on CPU-only machine)
- Reusing profiling data with different optimization parameters
- Implementing ablation studies by separating optimization steps

### Three Execution Modes

1. **Profile-Only Mode**: Collect profiling data and save to JSON
2. **Offline Optimization**: Load profiling data and generate optimized plan without GPU
3. **Run-Plan Mode**: Execute a pre-optimized plan without re-profiling or re-optimizing

### Using tiledCholeskyAblation

```cpp
// 1. Profile only - saves profiling data to JSON
./tiledCholeskyAblation --N=2048 --T=8 --profile-only --save-profile=profile.json

// 2. Run pre-optimized plan
./tiledCholeskyAblation --N=2048 --T=8 --run-plan --load-plan=optimized.json
```

### Using standaloneOptimizer

The standalone optimizer allows CPU-only optimization of profiling data:

```cpp
// Basic optimization
./standaloneOptimizer profile.json optimized.json

// With custom parameters
./standaloneOptimizer profile.json optimized.json --memory-bound=1000 --solver=GREEDY

// Save first step (task scheduling) for reuse
./standaloneOptimizer profile.json optimized.json --save-first-step=step1.json

// Load first step and run only memory optimization
./standaloneOptimizer profile.json optimized.json --load-first-step=step1.json
```

### Serialization Functions

The framework provides serialization for all optimization data structures:

```cpp
#include "optimization/optimizationSerializer.hpp"

// Save/load profiling data (OptimizationInput)
saveOptimizationInput(input, "profile.json");
auto input = loadOptimizationInput("profile.json");

// Save/load optimized plans (OptimizationOutput)
saveOptimizationOutput(output, "plan.json");
auto output = loadOptimizationOutput("plan.json");

// Save/load first step results (for ablation studies)
saveFirstStepOutput(firstStep, "step1.json");
auto firstStep = loadFirstStepOutput("step1.json");
```

### Ablation Study Support

The framework supports breaking down the two-step optimization for independent analysis:

1. **First Step Only**: Run task scheduling optimization, save results
2. **Second Step Only**: Load task scheduling, run memory optimization with different parameters
3. **Skip First Step**: Use original task order, run only memory optimization
4. **Skip Second Step**: Run task scheduling, use greedy memory management

This enables detailed analysis of how each optimization step contributes to overall performance.

## Configuration Options

Key configuration parameters for optimization:

- `maxPeakMemoryUsageInMiB`: Memory constraint for optimization (0 = unlimited)
- `firstStepSolverType`: Algorithm for task scheduling (BEAM_SEARCH recommended)
- `secondStepSolverType`: Algorithm for memory management (MIP, GREEDY, GREEDY_WARMSTART)
- `beamWidth`: Beam width for first step solver (default: 100)
- `gurobiTimeLimitSeconds`: Timeout for MIP solver (default: 60)
- `weightOfPeakMemoryUsage`: Weight for memory in optimization (0 = optimize runtime only)
- `weightOfTotalRunningTime`: Weight for runtime in optimization
- `weightOfNumberOfMigrations`: Weight for migration count

## Best Practices

1. **Always register memory addresses** before using them with TaskManager_v2 or memory optimization.

2. **Use type-safe task registration** to ensure parameters are properly handled:
   ```cpp
   // Good: Type-safe registration
   TaskId taskId = taskManager.registerTask<std::function<void(cudaStream_t, float*)>, float*>(...);
   ```

3. **Set the appropriate execution mode** based on your needs:
   - Use Basic mode for development and debugging
   - Use Production mode when using memory optimization features

4. **Provide meaningful task names** to make debugging and profiling easier.

5. **Use cudaStreams consistently** to ensure proper task synchronization:
   ```cpp
   taskManager.execute(taskId, stream);
   ```

6. **Leverage automatic parameter processing** in Production mode rather than manually calling getAddress.

7. **Handle memory cleanup** properly when your application finishes:
   ```cpp
   MemoryManager::getInstance().freeManagedMemory(pointer);
   ```

8. **Ensure proper synchronization** when executing CUDA graphs:
   ```cpp
   checkCudaErrors(cudaDeviceSynchronize());
   ```

## Configuration Options

The framework provides several configuration options through the JSON configuration system:

### Debug and Verbose Logging

Two options control the level of debug output:

```json
"execution": {
  "enableDebugOutput": false,  // Controls output of graph DOT files and optimization details
  "enableVerboseOutput": false // Controls detailed operation logs for memory/task operations
}
```

To enable verbose output, run with a configuration file that has `enableVerboseOutput` set to `true`:

```bash
# Use the provided verbose configuration
make run-verbose

# Or create your own configuration file
./build/userApplications/tiledCholesky --configFile=myVerboseConfig.json
```

When verbose output is enabled:
- Memory operations will print detailed logs about each prefetch/offload
- Task execution details will be displayed
- Execution progress will be more thoroughly reported

This is useful for debugging memory optimization issues and understanding the execution flow.