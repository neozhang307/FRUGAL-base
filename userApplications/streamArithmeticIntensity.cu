#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <fmt/core.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include "../include/argh.h"
#include "memopt.hpp"

using namespace memopt;

// Global variables
size_t N; // Total data size
size_t T; // Number of tasks (tiles)
size_t B; // Chunk size (N/T)
int intensity; // Arithmetic intensity

// Custom kernel with tunable arithmetic intensity
__global__ void tunable_streaming_kernel(double *a, double *b, double *c, size_t size, int intensity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double val_b = b[idx];
        double val_c = c[idx];
        double result = a[idx];

        for (int i = 0; i < intensity; i++) {
            result += val_b * val_c;
        }
        a[idx] = result;
    }
}


void generateRandomData(double *h_A, const size_t num_elements) {
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
  curandGenerateUniformDouble(prng, h_A, num_elements);
  curandDestroyGenerator(prng);
}

void initializeDeviceData(double *h_data, double* d_tile, size_t size) {
  fmt::print("Initializing device data of size {}\n", size);
  auto& memManager = MemoryManager::getInstance();
  void* srcAddress = memManager.getAddress(d_tile);

  if (srcAddress == d_tile) {
    srcAddress = memManager.getStoragePtr(d_tile);
    if (srcAddress == nullptr) {
      srcAddress = d_tile;
    }
  }
  checkCudaErrors(cudaMemcpy(srcAddress, h_data, size, cudaMemcpyDefault));
  checkCudaErrors(cudaDeviceSynchronize());
}

// Simplified verification
bool verifyResult(double* d_tile, size_t size) {
  auto& memManager = MemoryManager::getInstance();
  fmt::print("Performing simplified verification...\n");

  double* h_tile;
  checkCudaErrors(cudaMallocHost(&h_tile, size));

  bool copySuccess = memManager.copyManagedArrayToHost(d_tile, h_tile, size);
  if (!copySuccess) {
    fmt::print("ERROR: Failed to copy tile\n");
    checkCudaErrors(cudaFreeHost(h_tile));
    return false;
  }

  bool all_zeros = true;
  for (int k = 0; k < size / sizeof(double); k++) {
    if (h_tile[k] != 0.0) {
      all_zeros = false;
      break;
    }
  }

  checkCudaErrors(cudaFreeHost(h_tile));

  if (all_zeros) {
    fmt::print("❌ VERIFICATION FAILED: Tile is all zeros.\n");
    return false;
  }

  fmt::print("✅ VERIFICATION PASSED: No all-zero tiles found.\n");
  return true;
}

void streamArithmeticIntensityTest() {
  fmt::print("=== Stream Arithmetic Intensity Test ===\n");
  fmt::print("N={}, T={}, B={}, intensity={}\n", N, T, B, intensity);

  initializeCudaDevice();

  // =========================================================================
  // PHASE 1: SETUP AND ALLOCATE MEMORY
  // =========================================================================
  fmt::print("\n--- PHASE 1: Setup and Allocate Memory ---\n");

  const size_t chunkSize = B * sizeof(double);

  // Allocate 3 tiles for A, B, C
  double *d_tile_A, *d_tile_B, *d_tile_C;
  checkCudaErrors(cudaMalloc(&d_tile_A, chunkSize));
  checkCudaErrors(cudaMalloc(&d_tile_B, chunkSize));
  checkCudaErrors(cudaMalloc(&d_tile_C, chunkSize));

  auto& memManager = MemoryManager::getInstance();
  memManager.registerManagedMemoryAddress(d_tile_A, chunkSize);
  memManager.registerManagedMemoryAddress(d_tile_B, chunkSize);
  memManager.registerManagedMemoryAddress(d_tile_C, chunkSize);

  // Generate random data for B and C
  double* h_data_B;
  double* h_data_C;
  checkCudaErrors(cudaMallocHost(&h_data_B, chunkSize));
  checkCudaErrors(cudaMallocHost(&h_data_C, chunkSize));
  generateRandomData(h_data_B, B);
  generateRandomData(h_data_C, B);

  // Initialize A to zeros
  double* h_data_A;
  checkCudaErrors(cudaMallocHost(&h_data_A, chunkSize));
  memset(h_data_A, 0, chunkSize);


  initializeDeviceData(h_data_A, d_tile_A, chunkSize);
  initializeDeviceData(h_data_B, d_tile_B, chunkSize);
  initializeDeviceData(h_data_C, d_tile_C, chunkSize);


  double totalManagedMemoryMB = memManager.GetMemoryManagedSizeInMB();
  fmt::print("Total managed memory: {:.2f} MB\n", totalManagedMemoryMB);


  // =========================================================================
  // PHASE 2: BUILD NAIVE CUDA GRAPH USING TASKMANAGER_V2
  // =========================================================================
  fmt::print("\n--- PHASE 2: Build Naive CUDA Graph ---\n");

  cudaStream_t s;
  checkCudaErrors(cudaStreamCreate(&s));

  // Initialize TaskManager_v2 for naive graph construction
  TaskManager_v2 tmanager_v2(true);

  // Register T tasks
  fmt::print("Registering {} tasks...\n", T);

  for (int i = 0; i < T; i++) {
      std::vector<void*> inputs = {static_cast<void*>(d_tile_A), static_cast<void*>(d_tile_B), static_cast<void*>(d_tile_C)};
      std::vector<void*> outputs = {static_cast<void*>(d_tile_A)};

      TaskId taskId = tmanager_v2.registerTask<std::function<void(cudaStream_t, double*, double*, double*, int)>, double*, double*, double*, int>(
        [](cudaStream_t stream, double* a, double* b, double* c, int local_intensity) {
          int threads = 256;
          int blocks = (B + threads - 1) / threads;
          tunable_streaming_kernel<<<blocks, threads, 0, stream>>>(a, b, c, B, local_intensity);
        },
        inputs, outputs,
        TaskManager_v2::makeArgs(d_tile_A, d_tile_B, d_tile_C, intensity),
        "TunableKernel_task_" + std::to_string(i)
      );
      fmt::print("✓ Registered task {}\n", i);
  }


  fmt::print("✅ Task registration completed! Total tasks: {}\n", tmanager_v2.taskCount());

  // Generate naive CUDA graph from registered tasks
  fmt::print("Generating naive CUDA graph from task sequence...\n");
  cudaGraph_t graph = tmanager_v2.generateNaiveGraph(s);

  // Get graph stats
  size_t numNodes;
  checkCudaErrors(cudaGraphGetNodes(graph, nullptr, &numNodes));
  fmt::print("📊 Generated graph contains {} nodes\n", numNodes);

  // =========================================================================
  // PHASE 3: OPTIMIZE MEMORY ALLOCATION WITH NAIVE GRAPH
  // =========================================================================
  fmt::print("\n--- PHASE 3: Memory Optimization with Naive Graph ---\n");

  double initialPeakMemory = memManager.GetMemoryManagedSizeInMB();
  fmt::print("Initial peak memory: {:.2f} MB\n", initialPeakMemory);

  // Initialize data before optimization
  initializeDeviceData(h_data_A, d_tile_A, chunkSize);
  initializeDeviceData(h_data_B, d_tile_B, chunkSize);
  initializeDeviceData(h_data_C, d_tile_C, chunkSize);

  // Profile and optimize the naive graph
  fmt::print("Profiling and optimizing naive CUDA graph...\n");
  auto optimizedGraph = profileAndOptimize(graph);

  fmt::print("Original peak memory usage (MiB): {:.2f}\n", optimizedGraph.originalMemoryUsage);
  fmt::print("Optimized peak memory usage (MiB): {:.2f}\n", optimizedGraph.anticipatedPeakMemoryUsage);
  fmt::print("Memory reduction: {:.2f} MiB ({:.1f}%)\n",
             optimizedGraph.originalMemoryUsage - optimizedGraph.anticipatedPeakMemoryUsage,
             ((optimizedGraph.originalMemoryUsage - optimizedGraph.anticipatedPeakMemoryUsage) / optimizedGraph.originalMemoryUsage) * 100);

  // Move all data to storage before execution (required for optimized execution)
  fmt::print("Moving all data to storage for optimized execution...\n");
  memManager.offloadAllManagedMemoryToStorage();
  fmt::print("✅ All data moved to CPU/storage\n");

  // =========================================================================
  // PHASE 4: EXECUTE WITH OPTIMIZED NAIVE GRAPH
  // =========================================================================
  fmt::print("\n--- PHASE 4: Execute with Optimized Naive Graph ---\n");

  // Reinitialize data after optimization
  initializeDeviceData(h_data_A, d_tile_A, chunkSize);
  initializeDeviceData(h_data_B, d_tile_B, chunkSize);
  initializeDeviceData(h_data_C, d_tile_C, chunkSize);

  // Get current GPU memory info
  size_t free_mem, total_mem;
  checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem));
  fmt::print("GPU Memory - Total: {:.2f} MB, Free: {:.2f} MB\n",
             (double)total_mem / (1024.0 * 1024.0), (double)free_mem / (1024.0 * 1024.0));

  // Start peak memory monitoring
  fmt::print("🔍 Starting continuous GPU memory monitoring during execution...\n");
  PeakMemoryUsageProfiler peakProfiler(10); // Sample every 10ms
  peakProfiler.start();

  // Run the optimized naive graph
  float runningTime;
  executeOptimizedGraph(
    optimizedGraph,
    [&tmanager_v2](int taskId, std::map<void*, void*> addressMapping, cudaStream_t stream) {
      tmanager_v2.execute(taskId, stream);
    },
    runningTime,
    memManager
  );

  // Get peak memory usage
  size_t peakMemoryBytes = peakProfiler.end();
  double peakMemoryMB = (double)peakMemoryBytes / (1024.0 * 1024.0);

  fmt::print("✅ Optimized naive graph execution completed!\n");
  fmt::print("Execution time: {:.3f} ms\n", runningTime * 1000.0f);
  fmt::print("📊 Peak GPU memory usage during execution: {:.2f} MB\n", peakMemoryMB);

  // =========================================================================
  // PHASE 5: VERIFY RESULTS
  // =========================================================================
  fmt::print("\n--- PHASE 5: Verify Results ---\n");

  bool result = verifyResult(d_tile_A, chunkSize);

  // =========================================================================
  // PHASE 6: CLEANUP
  // =========================================================================
  fmt::print("\n--- PHASE 6: Cleanup ---\n");

  checkCudaErrors(cudaGraphDestroy(graph));
  checkCudaErrors(cudaStreamDestroy(s));

  memManager.freeManagedMemory(d_tile_A);
  memManager.freeManagedMemory(d_tile_B);
  memManager.freeManagedMemory(d_tile_C);

  checkCudaErrors(cudaFreeHost(h_data_A));
  checkCudaErrors(cudaFreeHost(h_data_B));
  checkCudaErrors(cudaFreeHost(h_data_C));


  fmt::print("\nFinal result: {}\n", result ? "SUCCESS" : "FAILED");
}

int main(int argc, char *argv[]) {
  argh::parser cmdl(argc, argv);

  if (cmdl[{"-h", "--help"}]) {
    fmt::print("Usage: streamArithmeticIntensity [options]\n");
    fmt::print("Options:\n");
    fmt::print("  -N, --total-size <elements>  Total number of elements in the stream (default: {})\n", 1024 * 1024);
    fmt::print("  -T, --num-tasks <tasks>      Number of tasks to divide the stream into (default: {})\n", 16);
    fmt::print("  --intensity <value>          Arithmetic intensity per element (default: {})\n", 1);
    fmt::print("  -h, --help                   Show this help message\n");
    return 0;
  }

  cmdl({"-N", "--total-size"}) >> N;
  if (!cmdl({"-N", "--total-size"})) {
    N = 1024 * 1024; // 1M elements
  }

  cmdl({"-T", "--num-tasks"}) >> T;
  if (!cmdl({"-T", "--num-tasks"})) {
    T = 16;
  }
  
  cmdl({"--intensity"}) >> intensity;
  if (!cmdl({"--intensity"})) {
    intensity = 1;
  }

  // Calculate block size
  B = N / T;

  // Validation
  if (N % T != 0) {
    fmt::print("ERROR: Total size must be divisible by num tasks\n");
    return -1;
  }

  ConfigurationManager::exportDefaultConfiguration();
  ConfigurationManager::loadConfiguration("config.json");

  fmt::print("Configuration: N={}, T={}, B={}, intensity={}\n", N, T, B, intensity);

  // Run the test
  streamArithmeticIntensityTest();

  return 0;
}