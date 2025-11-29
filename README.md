# FRUGAL: Framework for Reducing GPU memory via Unified Graph-based AnaLysis

## Overview

FRUGAL is a sophisticated CUDA memory optimization framework that enables GPU applications to exceed physical device memory limits through intelligent scheduling and data movement. The system uses a novel two-phase optimization approach to minimize memory usage while maintaining performance.

### Key Features
- **Automatic memory management** for out-of-core GPU applications
- **Two-phase optimization**: Task scheduling (Phase 1) and memory optimization (Phase 2)
- **CUDA graph integration** with automatic dependency tracking
- **Domain enlargement** for scaling optimization plans across problem sizes
- **Multiple solver backends**: SCIP (open-source) and Gurobi (commercial)

## Project Status (November 2025)

- ✅ **Phase 1 Optimization**: Solved with beam search (configurable width for quality/speed tradeoff)
- ✅ **Phase 2 Optimization**: Variable reduction using dependencies as constraints
- 🚧 **In Development**: Warm start for Gurobi, Top-K solutions, validation experiments
- 📊 **Benchmarks**: Tiled Cholesky, LU Decomposition, LULESH

## Quick Start

### Build Instructions
```bash
# Setup environment
source ~/miniconda3/bin/activate && conda activate frugal

# Configure and build
make config
make build

# Run example (tiled Cholesky)
./build/userApplications/tiledCholesky 8192 16
```

### Basic Usage
```bash
# Run with optimization
./build/userApplications/tiledCholesky --configFile=config.json

# Run with Unified Memory baseline
./build/userApplications/tiledCholesky --useUM
```

## Architecture

### Two-Phase Optimization

#### Phase 1: Task Scheduling (FirstStepSolver)
- **Algorithm**: Beam search with configurable width
- **Objective**: Maximize data reuse between tasks
- **Output**: Optimal task execution order

#### Phase 2: Memory Optimization (SecondStepSolver)
- **Method**: Mixed Integer Programming (MIP)
- **Decisions**: Array placement (GPU vs storage), prefetch/offload scheduling
- **Solvers**: Gurobi (recommended) or SCIP

### Core Components

- **TaskManager_v2**: Enhanced task management with annotation support
- **MemoryManager**: Centralized memory tracking and address resolution
- **CudaGraphConstructor**: Automatic dependency detection for CUDA graphs
- **ConfigurationManager**: JSON-based configuration system

## File Structure

- `include/`: Third party libraries
- `utilities/`: Common utilities and helpers
- `optimization/`: Memory optimization and execution logic
- `profiling/`: Application profiling and memory tracking
- `userApplications/`: Example applications (Cholesky, LU, LULESH)
- `experiments/`: Benchmarks and validation tests
- `playground/`: Small test programs for API verification

## Configuration System

The project uses JSON-based configuration for flexibility without recompilation.

### Key Configuration Parameters

```json
{
  "optimization": {
    "firstStepSolverType": "BEAM_SEARCH",  // Note: Some configs have typo "BEAN_SEARCH"
    "beamWidth": 100,                      // Beam search width (higher = better quality)
    "enableGapOverlap": true,              // ⚠️ CRITICAL: Dramatically affects quality (see below)
    "gapOverlapDecayFactor": 0.5,          // Gap overlap decay (0.0-1.0)
    "solver": "GUROBI_MIXED_INTEGER_PROGRAMMING",
    "gurobiTimeLimitSeconds": 60,
    "gurobiMipGap": 0.10,
    "maxPeakMemoryUsageInMiB": 15000,      // Memory constraint
    "weightOfPeakMemoryUsage": 0,          // 0 = optimize runtime at memory constraint
    "weightOfTotalRunningTime": 1          // 1 = prioritize performance
  },
  "execution": {
    "enableDebugOutput": false,
    "enableVerboseOutput": false,
    "measurePeakMemoryUsage": true
  }
}
```

**⚠️ See "Critical Configuration: Gap Overlap" section below for important findings about `enableGapOverlap`**

## User Applications

### 1. Tiled Cholesky Decomposition

Three implementations demonstrating different APIs:

```bash
# Basic implementation with explicit dependencies
./build/userApplications/tiledCholesky N T optimize verify

# Memory-optimized production version
./build/userApplications/tiledCholeskyMemoryOptimized N T

# Naive graph API (simplest)
./build/userApplications/tiledCholeskyNaiveGraph N T
```

### 2. LU Decomposition

```bash
./build/userApplications/lu_def N T
```

### 3. LULESH Benchmark

```bash
./build/userApplications/lulesh -s 45
```

## Domain Enlargement

Optimize with small data, execute with large data:

```bash
# Optimize with 1024×1024, execute with 4096×4096
./build/userApplications/tiledCholeskyDomainEnlarge 1024 4096 4
```

## CUDA Graph Dependency System

### PointerDependencyCudaGraphConstructor

Automatic dependency management based on memory access patterns:

```cpp
auto graphConstructor = std::make_unique<PointerDependencyCudaGraphConstructor>(stream, graph);

// Define memory dependencies
std::vector<void*> inputs = {inputPtr1, inputPtr2};
std::vector<void*> outputs = {outputPtr};

// Automatic dependency resolution
graphConstructor->beginCaptureOperation(inputs, outputs);
// ... CUDA operations ...
graphConstructor->endCaptureOperation();
```

Supports all dependency types:
- **RAW** (Read-After-Write): True dependencies
- **WAW** (Write-After-Write): Output dependencies
- **WAR** (Write-After-Read): Anti-dependencies

## Critical Configuration: Gap Overlap (IMPORTANT!)

### Impact of `enableGapOverlap` on Optimization Quality

⚠️ **CRITICAL FINDING** (November 2025): The `enableGapOverlap` setting has a **dramatic impact** on beam search quality and MIP feasibility.

#### Experimental Evidence (Tiled Cholesky, N=102400, T=4, 15GB constraint)

| Config Setting | Beam Search Score | MIP Feasibility at 15GB | Memory Impact |
|----------------|------------------|------------------------|---------------|
| `enableGapOverlap: false` | 92.77 GB data reuse | ❌ **INFEASIBLE** | Requires >35GB |
| `enableGapOverlap: true` | 119.62 GB data reuse | ✅ **FEASIBLE** | Achieves 15GB |

**Key Insights:**
- **29% improvement** in data reuse score (92.77 GB → 119.62 GB)
- **2.33x reduction** in memory requirements (35GB → 15GB)
- Beam search is **highly sensitive** to this setting
- Gap overlap enables better task scheduling by considering execution time gaps

#### Recommended Configuration

For **memory-constrained optimization** (achieving theoretical minimum memory):
```json
{
  "optimization": {
    "enableGapOverlap": true,           // CRITICAL for tight memory constraints
    "gapOverlapDecayFactor": 0.5,       // Decay factor for gap scoring
    "weightOfPeakMemoryUsage": 0,       // Focus on runtime, not memory
    "weightOfTotalRunningTime": 1,      // Prioritize performance
    "maxPeakMemoryUsageInMiB": 15000,   // Set to theoretical minimum
    "beamWidth": 100                     // Higher = better quality
  }
}
```

For **pure memory minimization** (may sacrifice performance):
```json
{
  "optimization": {
    "enableGapOverlap": false,           // Traditional approach
    "weightOfPeakMemoryUsage": 1.0,      // Minimize memory usage
    "weightOfTotalRunningTime": 0.0      // Ignore runtime overhead
  }
}
```

#### Technical Details

**Gap Overlap** in beam search considers temporal gaps between task completions and starts, enabling:
- Better exploitation of GPU idle time for data movement
- More effective prefetching by identifying available time windows
- Improved task ordering that creates larger reuse opportunities

**Profile Sensitivity:** Different profiling runs produce 0.2%-5% timing variations, which can lead to:
- Different beam search scores (up to 29% variance observed)
- Different MIP feasibility outcomes
- This is an inherent characteristic of GPU timing, not a bug

#### Validation

Full pipeline testing confirms:
- ✅ End-to-end and standalone optimizer produce **identical results** with same profile
- ✅ Algorithm correctness verified across all optimization paths
- ✅ Profile quality (not implementation) determines optimization success

For detailed investigation, see: [docs/GAP_OVERLAP_CRITICAL_FINDING.md](docs/GAP_OVERLAP_CRITICAL_FINDING.md)

## Known Issues

- **Stage Logic Bug**: Different behavior between CGO26/master and CGO26/stage-showcase branches
- **Minor**: cudaFreeHost error in domain enlargement (non-critical)
- **Profile Sensitivity**: GPU timing variance (0.2%-5%) affects beam search quality - consider multiple profiling runs for critical applications

## Upcoming Features

- **Warm Start**: Heuristic initial solution for faster Gurobi convergence
- **Top-K Solutions**: Gurobi solution pool for exploring alternatives
- **Minimal Memory Calculation**: Theoretical lower bound computation
- **Batch Testing**: Automated parameter sweeps and evaluation

## Development Guidelines

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines, coding standards, and best practices.

## Documentation

### Main Documentation
- [PROGRAM.md](PROGRAM.md): High-level program architecture and design
- [userApplications/CholeskyShowCase.md](userApplications/CholeskyShowCase.md): Comparison of different Cholesky implementations

### Technical Documentation (docs/)
- [docs/ABLATION.md](docs/ABLATION.md): Ablation study guide and experiment structure
- [docs/METRICS.md](docs/METRICS.md): Performance metrics explanation
- [docs/GAP_OVERLAP_CRITICAL_FINDING.md](docs/GAP_OVERLAP_CRITICAL_FINDING.md): Gap overlap analysis findings
- [experiments/performance_validation/README.md](experiments/performance_validation/README.md): Validation system usage guide

### Reference Documentation (docs/)
- [docs/cuda_graph_notation.md](docs/cuda_graph_notation.md): CUDA graph notation reference
- [docs/secondStepSolver_MIP_Reference.md](docs/secondStepSolver_MIP_Reference.md): MIP solver reference

## Requirements

- CUDA 12.x
- C++17 compiler
- CMake 3.21+
- Gurobi (optional, for MIP optimization)
- SCIP (alternative to Gurobi)

## Contributors

FRUGAL Project - CUDA Memory Optimization Framework

## License

[License information to be added]
