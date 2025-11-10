# CUDA Memory Optimization Project Guidelines

## Project Status (Updated: November 2024)
- **Branch**: CGO26/master (main development)
- **Phase 1 Optimization**: ✅ SOLVED - Beam search with configurable width provides fast solutions
- **Phase 2 Optimization**: ✅ Improved - Variable reduction using dependencies as constraints
- **Current Focus**: Warm start implementation, Top-K solutions, validation experiments

## Git Usage Rules (IMPORTANT - ALWAYS FOLLOW)
- **NEVER use `git add -A` or `git add .`** - Only add specific files that were modified
- **NEVER commit unrelated files** - Only commit files directly related to the current task
- **ALWAYS use `git add <specific-file>`** for each file you want to stage
- **NEVER commit TODO.md, TODO_TIMELINE.md, configs/, or other temporary/generated files**
- **ALWAYS check `git status` before committing** to ensure only intended files are staged

## Build Commands
- **Environment Setup**: 
  - Conda activation: `source ~/miniconda3/bin/activate && conda activate frugal`
  - Alternative (if available): `enable miniconda3 frugal` (x64) or `enable miniconda3_x86 frugal` (x86)
- Build: `source ~/miniconda3/bin/activate && conda activate frugal && make config && make build`
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

## Known Issues
- **Stage Logic Bug**: Different behavior between CGO26/master (working) and CGO26/stage-showcase (buggy) for out-of-core initialization
- **Minor**: cudaFreeHost error in domain enlargement (doesn't affect functionality)

## Upcoming Features (In Development)
- **Warm Start for Gurobi**: Heuristic initial solution to speed up optimization
- **Top-K Solutions**: Using Gurobi solution pool to explore alternative schedules
- **Minimal Memory Calculation**: Theoretical lower bound for memory usage
- **Batch Testing Infrastructure**: Automated parameter sweeps and evaluation

## Configuration Parameters
- **Solver**: `firstStepSolverType` should be "BEAM_SEARCH" (note: typo "BEAN_SEARCH" in some configs)
- **Beam Width**: Default 100, configurable for quality/speed tradeoff
- **Gurobi Settings**: 60s timeout, 10% MIP gap (proven sufficient)
- **Weights**: Set to 0 for pure memory optimization, adjust for runtime optimization

## Best Practices
- Prefer strong typing and avoid magic numbers
- Use smart pointers for memory management (std::unique_ptr)
- Prefer TaskManager_v2 for task execution (enhanced with annotation support)
- Use the MemoryManager singleton with getInstance() for most operations
- Always synchronize CUDA operations properly with cudaDeviceSynchronize()
- For experiments, use JSON configuration instead of recompiling