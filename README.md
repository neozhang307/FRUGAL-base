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
