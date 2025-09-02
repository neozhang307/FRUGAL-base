## File structure
- `include/`: Third party libraries.
- `utilities/`: Common utilities.
- `optimization/`: Sources related to optimizing the user application's memory allocation by scheduling prefetchings and offloadings, and executing the optimized user application.
- `profiling/`: Sources related to profiling the user application to collect information like data dependencies of each kernel.
- `userApplications/`: The user applications that this work applys to.
- `experiments/`: Sources for benchmarks and other experiments.
- `data/`: The script for running experiments and the results.
- `playgrounds/`: Small programs for testing APIs and verifying ideas.

## Configuration System

The project uses a flexible JSON-based configuration system that allows controlling various aspects of application behavior without code modifications. The configuration is managed by the `ConfigurationManager` class in the `memopt` namespace.

### Configuration Categories

#### Generic Settings
```json
"generic": {
  "optimize": false,        // Enable memory optimization
  "verify": false,          // Enable result verification
  "repeat": 1,              // Number of times to repeat the algorithm
  "useUM": false,           // Use CUDA Unified Memory
  "availableMemoryForUMInMiB": 40960  // Memory limit for Unified Memory
}
```

#### Optimization Settings
```json
"optimization": {
  "loadExistingPlan": false,   // Load pre-computed optimization plan
  "planPath": "optimizationPlan.json",  // Path to plan file
  "mergeConcurrentCudaGraphNodes": true,  // Merge nodes optimization
  "prefetchingBandwidthInGB": 281.0,      // Bandwidth assumption
  "acceptableRunningTimeFactor": 0.0,     // Acceptable performance loss
  "minManagedArraySize": 0,     // Minimum size of arrays to manage
  "solver": "SCIP",             // Optimization solver to use
  "maxPeakMemoryUsageInMiB": 0.0,  // Memory usage constraint
  "weightOfPeakMemoryUsage": 1.0,  // Optimization objective weights
  "weightOfTotalRunningTime": 0.0001,
  "weightOfNumberOfMigrations": 0.00001
}
```

#### Execution Settings
```json
"execution": {
  "useNvlink": false,            // Use NVLink for multi-GPU operations
  "measurePeakMemoryUsage": false,  // Enable memory usage profiling
  "enableDebugOutput": false,    // Enable debug output (dot files, etc.)
  "enableVerboseOutput": false,  // Enable detailed operation logs
  "mainDeviceId": 1,             // Main GPU device ID
  "storageDeviceId": 2           // Storage GPU device ID for multi-GPU
}
```

#### Algorithm-Specific Settings (tiledCholesky)
```json
"tiledCholesky": {
  "n": 256,    // Matrix size (N×N)
  "t": 4,      // Number of tiles (T×T)
  "mode": 0    // Operational mode: 0=normal, 1=dump matrix, 2=read matrix
}
```

### Using Configuration Files

1. **Default configuration**: We include a default configuration in `config.json` which can be used as a reference.

2. **Loading configuration**: A custom configuration can be loaded using:
   ```
   ./program --configFile=myConfig.json
   ```

3. **Special modes**: The configuration allows for special execution modes, such as:
   - Normal execution
   - Dumping input matrices to files for later use
   - Reading matrices from files (useful for matrices beyond device memory capacity)

### Configuration Validation

The system validates configuration settings to ensure they are consistent and compatible. For example, it prevents enabling both `optimize` and `useUM` simultaneously since they are mutually exclusive features.

### Benefits

- **Experimentation**: Easily test different parameters and strategies
- **Reproducibility**: Save configurations for experimental results
- **Flexibility**: Change behavior without recompiling code
- **Documentation**: JSON format provides self-documenting configuration

### Example

To run tiledCholesky with a custom configuration:
```bash
./build/userApplications/tiledCholesky --configFile=myConfig.json
```

## Domain Enlargement

The project includes **domain enlargement** functionality that enables optimizing with small data sizes while executing with larger data sizes. This allows reusing optimization plans across different problem scales without re-optimization.

### Concept

Domain enlargement works by:
1. **Optimization Phase**: Use small domain (e.g., 1024×1024 matrix) that fits in GPU memory
2. **Enlargement Phase**: Re-register arrays with larger CPU-allocated data (e.g., 4096×4096 matrix)  
3. **Execution Phase**: Execute with enlarged domain using the same optimization plan
4. **Memory Management**: Automatically handle data transfers between CPU and GPU as needed

### Implementation: tiledCholeskyDomainEnlarge

The `tiledCholeskyDomainEnlarge` application demonstrates domain enlargement on tiled Cholesky decomposition:

```bash
# Basic usage: small_domain large_domain tile_count
./build/userApplications/tiledCholeskyDomainEnlarge 1024 4096 4

# Arguments:
# - small_domain: Matrix size for optimization (e.g., 1024 = 1024×1024)
# - large_domain: Matrix size for execution (e.g., 4096 = 4096×4096) 
# - tile_count: Number of tiles per dimension (same for both domains)
```

### Key Features

- **Execution-Oblivious Design**: No changes required to executor logic
- **Global Pointer Mechanism**: Dynamic block size switching during execution
- **Workspace Reallocation**: Automatic cuSOLVER workspace resizing for different block sizes
- **GPU Memory Profiling**: Continuous monitoring with `PeakMemoryUsageProfiler`
- **Cold Start Pattern**: CPU→GPU→CPU execution with proper assertions
- **Verification**: Compare tiled vs cuSOLVER direct Cholesky for validation

### Output

The application provides comprehensive reporting:
- Memory optimization results (peak usage reduction)
- Domain enlargement metrics (block size switching)
- GPU memory profiling (peak usage during execution)
- Verification results (tiled vs direct Cholesky comparison)

```
Original peak memory usage (MiB): 5.00
Optimized peak memory usage (MiB): 2.00
🔄 Switched kernel block size from 256 to 1024
📊 Peak GPU memory usage during execution: 960.00 MB
✅ SUCCESS: Tiled and Direct Cholesky produce IDENTICAL results
```

## CUDA Graph Dependency System

The project includes a sophisticated dependency tracking system for CUDA graph construction that ensures proper synchronization between memory operations.

### PointerDependencyCudaGraphConstructor

The `PointerDependencyCudaGraphConstructor` class provides automatic dependency management based on memory pointer analysis. It supports all three types of data dependencies:

#### Supported Dependency Types

1. **Read-After-Write (RAW)** - True dependency
   - Ensures operations wait for previous writes to complete before reading
   - Prevents reading stale or uninitialized data

2. **Write-After-Write (WAW)** - Output dependency  
   - Ensures operations wait for previous writes to complete before writing
   - Prevents write ordering conflicts

3. **Write-After-Read (WAR)** - Anti-dependency
   - Ensures operations wait for all reads to complete before overwriting
   - Prevents premature destruction of data being read

#### Usage Example

```cpp
// Create dependency tracker
auto graphConstructor = std::make_unique<PointerDependencyCudaGraphConstructor>(stream, graph);

// Define memory dependencies for an operation
std::vector<void*> inputs = {inputPtr1, inputPtr2};    // Memory locations to read
std::vector<void*> outputs = {outputPtr};             // Memory locations to write

// Begin operation with automatic dependency resolution
graphConstructor->beginCaptureOperation(inputs, outputs);

// Execute your CUDA operations here
// (kernel launches, cuBLAS/cuSOLVER calls, etc.)

// End capture and update dependency tracking
graphConstructor->endCaptureOperation();
```

#### Key Features

- **Automatic dependency detection**: Analyzes input/output memory patterns
- **Read-modify-write support**: Properly handles operations that both read and write to the same location
- **Efficient tracking**: Uses maps to track last readers and writers for each memory location
- **Race condition prevention**: Ensures proper synchronization for complex computation graphs

#### Implementation Details

- **Reader tracking**: Maintains `lastReadByMap` to track current readers of each memory location
- **Writer tracking**: Maintains `lastModifiedByMap` to track the last writer of each memory location
- **Dependency resolution**: Automatically establishes edges in the CUDA graph based on memory access patterns
- **Cleanup optimization**: Clears reader lists when establishing WAR dependencies to prevent memory buildup

This system is particularly useful for complex applications like tiled matrix decompositions where operations frequently perform in-place updates on shared memory blocks.
