# CUDA Memory Optimization Project Guidelines

## Build Commands
- Build: `make config && make build`
- Debug build: `make clean && make config-debug && make build`
- Quick build: `make build-sequential`
- Verbose build: `make build-verbose`
- Clean: `make clean`
- Run sample: `make run` (runs helloWorld)
- Run experiments: `./data/bandwidthTest/run.sh` or `./data/splitDataMovement/run.sh`
- Test changes: For tiledCholesky, run `./build/userApplications/tiledCholesky`

## Coding Standards
- **Namespace**: All code must be in the `memopt` namespace
- **Naming Conventions**:
  - Classes: CamelCase (e.g., `PeakMemoryUsageProfiler`)
  - Functions/Methods: camelCase (e.g., `optimizeMemoryUsage`)
  - Variables: camelCase (e.g., `devicePointer`)
- **Error Handling**: Always use `checkCudaErrors` macro for CUDA calls
- **Logging**: Use `LOG_TRACE()` for function tracing or `LOG_TRACE_WITH_INFO()` for parameterized logging
- **Language Standards**: C++17 with CUDA standard 17

## Documentation
- Document all public APIs with detailed parameter descriptions
- Use Doxygen-style documentation for classes and methods
- Explain complex algorithms with comments that describe the purpose

## Best Practices
- Prefer strong typing and avoid magic numbers
- Use smart pointers for memory management (std::unique_ptr)
- Prefer TaskManager for task execution and MemoryManager for memory tracking
- Use the MemoryManager singleton with getInstance() for most operations
- Always synchronize CUDA operations properly with cudaDeviceSynchronize()