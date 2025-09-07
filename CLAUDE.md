# CUDA Memory Optimization Project Guidelines

## Git Usage Rules (IMPORTANT - ALWAYS FOLLOW)
- **NEVER use `git add -A` or `git add .`** - Only add specific files that were modified
- **NEVER commit unrelated files** - Only commit files directly related to the current task
- **ALWAYS use `git add <specific-file>`** for each file you want to stage
- **NEVER commit TODO.md, configs/, or other temporary/generated files**
- **ALWAYS check `git status` before committing** to ensure only intended files are staged

## Build Commands
- **Environment Setup**: 
  - x86 Conda activation (preferred): `source ~/miniconda3_x86/bin/activate && conda activate frugal`
  - x64 Conda activation: `source ~/miniconda3/bin/activate && conda activate frugal`
- Build: `source ~/miniconda3_x86/bin/activate && conda activate frugal && make config && make build`
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