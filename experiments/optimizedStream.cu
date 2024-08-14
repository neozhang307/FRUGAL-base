#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "../include/argh.h"
#include "../include/csv.hpp"
#include "memopt.hpp"

using namespace memopt;

constexpr int COMPUTE_DEVICE_ID = 0;
constexpr int STORAGE_DEVICE_ID = cudaCpuDeviceId;

template <typename T>
__global__ void initializeArrayKernel(T *array, T initialValue, size_t count) {
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < count) {
    array[i] = initialValue;
  }
}

template <typename T>
__global__ void addKernel(const T *a, const T *b, T *c) {
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i] + b[i];
}

template <typename T>
__global__ void checkResultKernel(const T *c, const T expectedValue) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (c[i] != expectedValue) {
    printf("[checkResultKernel] found c[%d] = %f, while expectedValue = %f\n", i, c[i], expectedValue);
  }
}

void warmUpDevice(const int deviceId) {
  checkCudaErrors(cudaSetDevice(deviceId));
  constexpr size_t WARMUP_ARRAY_SIZE = 1024ull * 1024 * 1024;  // 1GiB
  constexpr size_t WARMUP_ARRAY_LENGTH = WARMUP_ARRAY_SIZE / sizeof(float);
  constexpr size_t BLOCK_SIZE = 1024;
  constexpr size_t GRID_SIZE = WARMUP_ARRAY_LENGTH / BLOCK_SIZE;

  constexpr float initA = 1;
  constexpr float initB = 2;
  constexpr float expectedC = initA + initB;

  float *a, *b, *c;
  checkCudaErrors(cudaMalloc(&a, WARMUP_ARRAY_SIZE));
  checkCudaErrors(cudaMalloc(&b, WARMUP_ARRAY_SIZE));
  checkCudaErrors(cudaMalloc(&c, WARMUP_ARRAY_SIZE));

  initializeArrayKernel<<<GRID_SIZE, BLOCK_SIZE>>>(a, initA, WARMUP_ARRAY_SIZE);
  initializeArrayKernel<<<GRID_SIZE, BLOCK_SIZE>>>(b, initB, WARMUP_ARRAY_SIZE);

  addKernel<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c);

  checkResultKernel<<<GRID_SIZE, BLOCK_SIZE>>>(c, expectedC);

  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(a));
  checkCudaErrors(cudaFree(b));
  checkCudaErrors(cudaFree(c));

  checkCudaErrors(cudaDeviceSynchronize());
}

void warmUpDataMovement(int deviceA, int deviceB) {
  const size_t ARRAY_SIZE = 1ull << 30;
  const size_t ARRAY_LENGTH = ARRAY_SIZE / sizeof(int);

  int *arrayOnA;
  if (deviceA == cudaCpuDeviceId) {
    arrayOnA = (int *)malloc(ARRAY_SIZE);
    memset(arrayOnA, 0, ARRAY_SIZE);
  } else {
    checkCudaErrors(cudaSetDevice(deviceA));
    checkCudaErrors(cudaMalloc(&arrayOnA, ARRAY_SIZE));
    initializeArrayKernel<<<ARRAY_LENGTH / 1024, 1024>>>(arrayOnA, 0, ARRAY_LENGTH);
  }

  int *arrayOnB;
  if (deviceB == cudaCpuDeviceId) {
    arrayOnB = (int *)malloc(ARRAY_SIZE);
  } else {
    checkCudaErrors(cudaSetDevice(deviceB));
    checkCudaErrors(cudaMalloc(&arrayOnB, ARRAY_SIZE));
    initializeArrayKernel<<<ARRAY_LENGTH / 1024, 1024>>>(arrayOnB, 0, ARRAY_LENGTH);
  }

  cudaStream_t stream;
  checkCudaErrors(cudaMemcpy(arrayOnA, arrayOnB, ARRAY_SIZE, cudaMemcpyDefault));
  checkCudaErrors(cudaDeviceSynchronize());

  if (deviceA == cudaCpuDeviceId) {
    free(arrayOnA);
  } else {
    checkCudaErrors(cudaFree(arrayOnA));
  }

  if (deviceB == cudaCpuDeviceId) {
    free(arrayOnB);
  } else {
    checkCudaErrors(cudaFree(arrayOnB));
  }
}

void runOptimizedStream(size_t arraySize, int numberOfKernels, int prefetchCycleLength) {
  const size_t arrayLength = arraySize / sizeof(float);
  constexpr size_t BLOCK_SIZE = 1024;
  const size_t GRID_SIZE = arrayLength / BLOCK_SIZE;

  assert(arrayLength % BLOCK_SIZE == 0ull);

  warmUpDevice(COMPUTE_DEVICE_ID);

  warmUpDataMovement(STORAGE_DEVICE_ID, COMPUTE_DEVICE_ID);
  warmUpDataMovement(COMPUTE_DEVICE_ID, STORAGE_DEVICE_ID);

  checkCudaErrors(cudaSetDevice(COMPUTE_DEVICE_ID));

  // Initialize data
  auto aOnComputeDevice = std::make_unique<float *[]>(numberOfKernels);
  auto bOnComputeDevice = std::make_unique<float *[]>(numberOfKernels);
  auto cOnComputeDevice = std::make_unique<float *[]>(numberOfKernels);

  auto aOnStorageDevice = std::make_unique<float *[]>(numberOfKernels);
  auto bOnStorageDevice = std::make_unique<float *[]>(numberOfKernels);

  checkCudaErrors(cudaSetDevice(COMPUTE_DEVICE_ID));
  cudaStream_t computeStream, dataMovementStream;
  checkCudaErrors(cudaStreamCreate(&computeStream));
  checkCudaErrors(cudaStreamCreate(&dataMovementStream));

  for (int i = 0; i < numberOfKernels; i++) {
    if (i != 1 && i % prefetchCycleLength == 1) {
      LOG_TRACE_WITH_INFO("Kernel %d is prefetched", i);

      aOnStorageDevice[i] = (float *)malloc(arraySize);
      bOnStorageDevice[i] = (float *)malloc(arraySize);
      memset(aOnStorageDevice[i], 0, arraySize);
      memset(bOnStorageDevice[i], 0, arraySize);
    } else {
      checkCudaErrors(cudaMallocAsync(&aOnComputeDevice[i], arraySize, dataMovementStream));
      checkCudaErrors(cudaMallocAsync(&bOnComputeDevice[i], arraySize, dataMovementStream));
      initializeArrayKernel<<<GRID_SIZE, BLOCK_SIZE, 0, dataMovementStream>>>(aOnComputeDevice[i], (float)0, arrayLength);
      initializeArrayKernel<<<GRID_SIZE, BLOCK_SIZE, 0, dataMovementStream>>>(bOnComputeDevice[i], (float)0, arrayLength);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }

  // Compute
  auto prefetchEvents = std::make_unique<cudaEvent_t[]>(numberOfKernels / prefetchCycleLength);
  for (int i = 0; i < numberOfKernels / prefetchCycleLength; i++) {
    checkCudaErrors(cudaEventCreate(&prefetchEvents[i]));
  }

  cudaEvent_t endOfKernelEvent;
  checkCudaErrors(cudaEventCreate(&endOfKernelEvent));

  CudaEventClock clock;
  clock.start(computeStream);

  for (int i = 0; i < numberOfKernels; i++) {
    if (i % prefetchCycleLength == 1) {
      if (i + prefetchCycleLength < numberOfKernels) {
        checkCudaErrors(cudaEventRecord(endOfKernelEvent, computeStream));
        checkCudaErrors(cudaStreamWaitEvent(dataMovementStream, endOfKernelEvent));

        checkCudaErrors(cudaMallocAsync(&aOnComputeDevice[i + prefetchCycleLength], arraySize, dataMovementStream));
        checkCudaErrors(cudaMallocAsync(&bOnComputeDevice[i + prefetchCycleLength], arraySize, dataMovementStream));
        checkCudaErrors(cudaMemcpyAsync(aOnComputeDevice[i + prefetchCycleLength], aOnStorageDevice[i + prefetchCycleLength], arraySize, cudaMemcpyDefault, dataMovementStream));
        checkCudaErrors(cudaMemcpyAsync(bOnComputeDevice[i + prefetchCycleLength], bOnStorageDevice[i + prefetchCycleLength], arraySize, cudaMemcpyDefault, dataMovementStream));
        checkCudaErrors(cudaEventRecord(prefetchEvents[(i - 1) / prefetchCycleLength], dataMovementStream));
      }
      if (i != 1) {
        checkCudaErrors(cudaStreamWaitEvent(computeStream, prefetchEvents[(i - 1) / prefetchCycleLength - 1]));
      }
    }

    checkCudaErrors(cudaMallocAsync(&cOnComputeDevice[i], arraySize, computeStream));
    addKernel<<<GRID_SIZE, BLOCK_SIZE, 0, computeStream>>>(aOnComputeDevice[i], bOnComputeDevice[i], cOnComputeDevice[i]);
    checkCudaErrors(cudaFreeAsync(aOnComputeDevice[i], computeStream));
    checkCudaErrors(cudaFreeAsync(bOnComputeDevice[i], computeStream));
  }

  clock.end(computeStream);
  checkCudaErrors(cudaStreamSynchronize(computeStream));

  const float runningTime = clock.getTimeInSeconds();
  const float bandwidth = static_cast<float>(arraySize) * 3.0 * numberOfKernels / 1e9 / runningTime;
  LOG_TRACE_WITH_INFO("Total running time (s): %.6f", runningTime);
  LOG_TRACE_WITH_INFO("Bandwidth (GB/s): %.2f", bandwidth);

  for (int i = 0; i < numberOfKernels / prefetchCycleLength; i++) {
    checkCudaErrors(cudaEventDestroy(prefetchEvents[i]));
  }

  checkCudaErrors(cudaEventDestroy(endOfKernelEvent));

  checkCudaErrors(cudaStreamDestroy(computeStream));
  checkCudaErrors(cudaStreamDestroy(dataMovementStream));
}

int main(int argc, char **argv) {
  auto cmdl = argh::parser(argc, argv);

  size_t arraySize;
  cmdl("array-size", 1'073'741'824ull) >> arraySize;  // 1GiB by default

  int numberOfKernels;
  cmdl("number-of-kernels", 22) >> numberOfKernels;  // 22 kernels in total by default: 1th kernel, 2nd kernel, ..., 22th kernel.

  int prefetchCycleLength;
  cmdl("prefetch-cycle-length", 4) >> prefetchCycleLength;  // Prefetch the 6th, 10th,..., 22nd kernels by default

  runOptimizedStream(arraySize, numberOfKernels, prefetchCycleLength);

  return 0;
}
