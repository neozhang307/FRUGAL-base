#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "../profiling/memoryManager.hpp"

using namespace memopt;

int main() {
    printf("=== Domain Enlargement Test ===\n");
    
    // Step 1: Initialize memory manager
    MemoryManager& memManager = MemoryManager::getInstance();
    
    // Step 2: Allocate small arrays on GPU (simulate original small domain)
    printf("\nStep 1: Allocate small domain arrays\n");
    
    size_t smallSize = 1024 * sizeof(float);  // 1K elements
    float* smallArrayA;
    float* smallArrayB;
    
    cudaMalloc(&smallArrayA, smallSize);
    cudaMalloc(&smallArrayB, smallSize);
    
    printf("  Small arrays allocated: A=%p (%zu bytes), B=%p (%zu bytes)\n", 
           smallArrayA, smallSize, smallArrayB, smallSize);
    
    // Step 3: Register arrays with memory manager
    printf("\nStep 2: Register arrays with MemoryManager\n");
    
    memManager.registerManagedMemoryAddress(smallArrayA, smallSize);
    memManager.registerManagedMemoryAddress(smallArrayB, smallSize);
    
    // Verify registration
    bool isAManaged = memManager.isManaged(smallArrayA);
    bool isBManaged = memManager.isManaged(smallArrayB);
    size_t retrievedSizeA = memManager.getSize(smallArrayA);
    size_t retrievedSizeB = memManager.getSize(smallArrayB);
    
    printf("  Registration results:\n");
    printf("    Array A: managed=%s, size=%zu\n", isAManaged ? "YES" : "NO", retrievedSizeA);
    printf("    Array B: managed=%s, size=%zu\n", isBManaged ? "YES" : "NO", retrievedSizeB);
    
    if (!isAManaged || !isBManaged) {
        printf("ERROR: Arrays not properly registered!\n");
        return -1;
    }
    
    // Step 4: Create larger CPU arrays (4x larger)
    printf("\nStep 3: Create enlarged CPU arrays\n");
    
    size_t largeSize = smallSize * 4;  // 4x larger
    float* largeCpuArrayA = (float*)malloc(largeSize);
    float* largeCpuArrayB = (float*)malloc(largeSize);
    
    // Initialize with some test data
    for (size_t i = 0; i < largeSize/sizeof(float); i++) {
        largeCpuArrayA[i] = (float)i;
        largeCpuArrayB[i] = (float)(i * 2);
    }
    
    printf("  Large CPU arrays created: A=%p (%zu bytes), B=%p (%zu bytes)\n", 
           largeCpuArrayA, largeSize, largeCpuArrayB, largeSize);
    
    // Step 5: Test re-registration
    printf("\nStep 4: Re-register with larger CPU data\n");
    
    bool reregisterA = memManager.reregisterManagedArrayWithLargerCPUData(smallArrayA, largeCpuArrayA, largeSize);
    bool reregisterB = memManager.reregisterManagedArrayWithLargerCPUData(smallArrayB, largeCpuArrayB, largeSize);
    
    printf("  Re-registration results: A=%s, B=%s\n", 
           reregisterA ? "SUCCESS" : "FAILED", reregisterB ? "SUCCESS" : "FAILED");
    
    if (!reregisterA || !reregisterB) {
        printf("ERROR: Re-registration failed!\n");
        return -1;
    }
    
    // Step 6: Verify enlarged sizes are reflected
    printf("\nStep 5: Verify enlarged sizes\n");
    
    size_t newSizeA = memManager.getSize(smallArrayA);  // Should return largeSize now
    size_t newSizeB = memManager.getSize(smallArrayB);  // Should return largeSize now
    
    printf("  New sizes: A=%zu bytes, B=%zu bytes\n", newSizeA, newSizeB);
    printf("  Expected: %zu bytes\n", largeSize);
    
    bool testPassed = (newSizeA == largeSize) && (newSizeB == largeSize);
    
    if (testPassed) {
        printf("\n✅ SUCCESS: Domain enlargement working correctly!\n");
        printf("   - Original addresses preserved: A=%p, B=%p\n", smallArrayA, smallArrayB);
        printf("   - Memory manager now reports larger sizes: %zu -> %zu bytes\n", smallSize, largeSize);
        printf("   - Storage points to larger CPU data\n");
    } else {
        printf("\n❌ FAILED: Size verification failed!\n");
        printf("   Expected: A=%zu, B=%zu\n", largeSize, largeSize);
        printf("   Actual:   A=%zu, B=%zu\n", newSizeA, newSizeB);
    }
    
    // Step 7: Test batch re-registration
    printf("\nStep 6: Test batch re-registration\n");
    
    // Create more test arrays
    size_t mediumSize = smallSize * 2;
    float* testArrayC;
    float* testArrayD;
    cudaMalloc(&testArrayC, smallSize);
    cudaMalloc(&testArrayD, smallSize);
    
    memManager.registerManagedMemoryAddress(testArrayC, smallSize);
    memManager.registerManagedMemoryAddress(testArrayD, smallSize);
    
    float* mediumCpuArrayC = (float*)malloc(mediumSize);
    float* mediumCpuArrayD = (float*)malloc(mediumSize);
    
    // Batch re-registration
    std::map<void*, MemoryManager::ArrayReplacement> replacements;
    replacements[testArrayC] = {mediumCpuArrayC, mediumSize};
    replacements[testArrayD] = {mediumCpuArrayD, mediumSize};
    
    int batchSuccess = memManager.reregisterMultipleArraysWithLargerCPUData(replacements);
    printf("  Batch re-registration: %d out of %zu arrays succeeded\n", batchSuccess, replacements.size());
    
    // Cleanup
    printf("\nStep 7: Cleanup\n");
    free(largeCpuArrayA);
    free(largeCpuArrayB);
    free(mediumCpuArrayC);
    free(mediumCpuArrayD);
    cudaFree(smallArrayA);
    cudaFree(smallArrayB);
    cudaFree(testArrayC);
    cudaFree(testArrayD);
    
    printf("=== Test Complete ===\n");
    return testPassed ? 0 : -1;
}