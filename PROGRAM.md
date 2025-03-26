# CUDA Memory Optimization Programming Guide

This guide explains how to use the TaskManager_v2 and memory optimization components of the CUDA Memory Optimization framework.

## Table of Contents

1. [Introduction](#introduction)
2. [TaskManager_v2 Overview](#taskmanager_v2-overview)
3. [Using TaskManager_v2](#using-taskmanager_v2)
4. [Memory Registration](#memory-registration)
5. [Execution Modes](#execution-modes)
6. [Memory Optimization](#memory-optimization)
7. [Advanced Usage Patterns](#advanced-usage-patterns)
8. [Configuration Options](#configuration-options)
9. [Best Practices](#best-practices)

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