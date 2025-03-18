#pragma once

#include <cassert>
#include <initializer_list>
#include <map>
#include <set>
#include <vector>

#include "../utilities/configurationManager.hpp"
#include "../utilities/types.hpp"

namespace memopt {

/**
 * @brief Memory management system for tracking and optimizing GPU memory usage
 *
 * The MemoryManager serves as a central registry for GPU memory allocations, tracking:
 * - Memory addresses of GPU allocations
 * - Sizes of each allocation
 * - Application inputs and outputs (for data dependency tracking)
 * - Mappings between original device pointers and relocated host/device pointers
 * 
 * This system enables memory optimization strategies like:
 * - Data offloading (moving data between GPU and host/secondary GPU)
 * - Prefetching (loading data back to GPU before needed)
 * - Memory usage scheduling (to reduce peak memory consumption)
 * 
 * The MemoryManager is used by the optimization system to enable execution of
 * workloads that might otherwise exceed available GPU memory.
 */
struct MemoryManager {
  /** @brief Ordered list of all managed memory addresses */
  inline static std::vector<void *> managedMemoryAddresses; // Original Memory
  
  /** @brief Maps memory addresses to their array IDs (index in managedMemoryAddresses) */
  inline static std::map<void *, ArrayId> managedMemoryAddressToIndexMap;
  
  /** @brief Maps memory addresses to their allocation sizes in bytes */
  inline static std::map<void *, size_t> managedMemoryAddressToSizeMap;
  
  /** @brief Tracks which arrays are application inputs/outputs (for data dependency analysis) */
  inline static std::set<void *> applicationInputs, applicationOutputs;
  
  /** @brief Maps original device pointers to their current relocated pointers */
  inline static std::map<void *, void *> deviceToStorageArrayMap;

  /** @brief Maps original device pointers to their current relocated pointers */
  inline static std::map<void *, void *> managedMemoryAddressToAssignedMap;

  /**
   * @brief Gets the updated address of a pointer if it has been relocated
   *
   * This function checks if the given pointer exists in the address update map
   * and returns the updated address if found, otherwise returns the original address.
   * 
   * @tparam T Type of data stored in the array
   * @param addressUpdate Map containing original addresses mapped to their new locations
   * @param oldAddress Original pointer address to check for updates
   * @return Updated pointer address if found in map, otherwise the original address
   */
  template <typename T>
  static T* getAddress(T *oldAddress) {
    auto it = managedMemoryAddressToAssignedMap.find(static_cast<void *>(oldAddress));
    if (it != managedMemoryAddressToAssignedMap.end()) {
      return static_cast<T *>(it->second);
    }
    return oldAddress;
  }
};

/**
 * @brief Registers a GPU memory allocation for management by the optimization system
 *
 * This function adds a device pointer to the tracked memory addresses if:
 * 1. Its size exceeds the minimum threshold defined in configuration
 * 2. It hasn't already been registered
 *
 * @tparam T Type of data stored in the GPU array
 * @param devPtr Pointer to device memory
 * @param size Size of allocation in bytes
 */
template <typename T>
void registerManagedMemoryAddress(T *devPtr, size_t size) {
  // Skip small allocations based on configuration threshold: overhead of managing bunch of small arrays is non trival. 
  if (size < ConfigurationManager::getConfig().optimization.minManagedArraySize) {
    return;
  }

  auto ptr = static_cast<void *>(devPtr);
  // Only register if not already tracked
  if (MemoryManager::managedMemoryAddressToIndexMap.count(ptr) == 0) {
    MemoryManager::managedMemoryAddresses.push_back(ptr);
    MemoryManager::managedMemoryAddressToIndexMap[ptr] = MemoryManager::managedMemoryAddresses.size() - 1;
    MemoryManager::managedMemoryAddressToSizeMap[ptr] = size;
    
    // Initialize the mapping to point to itself (no relocation yet)
    MemoryManager::deviceToStorageArrayMap[ptr] = ptr;
  }
}

/**
 * @brief Marks a device pointer as an application input
 *
 * Application inputs are arrays that are initialized from outside the
 * optimization system and must be available at the start of execution.
 *
 * @tparam T Type of data stored in the GPU array
 * @param devPtr Pointer to device memory containing input data
 */
template <typename T>
void registerApplicationInput(T *devPtr) {
  MemoryManager::applicationInputs.insert(static_cast<void *>(devPtr));
}

/**
 * @brief Marks a device pointer as an application output
 *
 * Application outputs are arrays that contain results needed after
 * optimization system execution completes, and must be preserved.
 *
 * @tparam T Type of data stored in the GPU array
 * @param devPtr Pointer to device memory that will store output data
 */
template <typename T>
void registerApplicationOutput(T *devPtr) {
  MemoryManager::applicationOutputs.insert(static_cast<void *>(devPtr));
}

/**
 * @brief Updates memory address mappings after memory is reallocated
 *
 * Used when original pointers are changed (e.g., during reallocation or
 * when switching between different memory spaces). This function:
 * 1. Updates all internal tracking structures with new addresses
 * 2. Maintains the relationship between old and new addresses
 * 3. Updates application input/output registrations
 *
 * @param oldAddressToNewAddressMap Mapping from original addresses to new addresses
 */
inline void updateManagedMemoryAddress(const std::map<void *, void *> oldAddressToNewAddressMap) {
  // Save the old size mapping for reference
  auto oldManagedMemoryAddressToSizeMap = MemoryManager::managedMemoryAddressToSizeMap;

  // Clear existing mappings to rebuild them
  MemoryManager::managedMemoryAddressToSizeMap.clear();
  MemoryManager::managedMemoryAddressToIndexMap.clear();

  // Update all addresses and rebuild mappings
  for (int i = 0; i < MemoryManager::managedMemoryAddresses.size(); i++) {
    // Ensure every old address has a corresponding new address
    assert(oldAddressToNewAddressMap.count(MemoryManager::managedMemoryAddresses[i]) == 1);

    const auto newAddr = oldAddressToNewAddressMap.at(MemoryManager::managedMemoryAddresses[i]);
    const auto oldAddr = MemoryManager::managedMemoryAddresses[i];

    // Update address in the main list
    MemoryManager::managedMemoryAddresses[i] = newAddr;
    // Transfer the size information to the new address
    MemoryManager::managedMemoryAddressToSizeMap[newAddr] = oldManagedMemoryAddressToSizeMap[oldAddr];
    // Maintain the same array ID
    MemoryManager::managedMemoryAddressToIndexMap[newAddr] = i;
    // Update the device-to-host mapping
    MemoryManager::deviceToStorageArrayMap[oldAddr] = newAddr;

    // Update application input registry if this was an input
    if (MemoryManager::applicationInputs.count(oldAddr) > 0) {
      MemoryManager::applicationInputs.erase(oldAddr);
      MemoryManager::applicationInputs.insert(newAddr);
    }
    // Update application output registry if this was an output
    if (MemoryManager::applicationOutputs.count(oldAddr) > 0) {
      MemoryManager::applicationOutputs.erase(oldAddr);
      MemoryManager::applicationOutputs.insert(newAddr);
    }
  }
}

}  // namespace memopt
