#pragma once

#include <cassert>
#include <initializer_list>
#include <map>
#include <set>
#include <vector>
#include <memory>
#include <mutex>

#include <cuda_runtime.h>

#include "../utilities/configurationManager.hpp"
#include "../utilities/types.hpp"
#include "../utilities/cudaUtilities.hpp"

namespace memopt {

// Forward declaration
class MemoryManager;

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
void registerManagedMemoryAddress(T *devPtr, size_t size);

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
void registerApplicationInput(T *devPtr);

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
void registerApplicationOutput(T *devPtr);

/**
 * @brief Updates memory address mappings after memory is reallocated
 *
 * Used when original pointers are changed (e.g., during reallocation or
 * when switching between different memory spaces).
 *
 * @param oldAddressToNewAddressMap Mapping from original addresses to new addresses
 */
void updateManagedMemoryAddress(const std::map<void *, void *> oldAddressToNewAddressMap);

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
class MemoryManager {
private:
  /** @brief Ordered list of all managed memory addresses */
  inline static std::vector<void *> managedMemoryAddresses; // Original Memory
  
  /** @brief Maps memory addresses to their array IDs (index in managedMemoryAddresses) */
  inline static std::map<void *, ArrayId> managedMemoryAddressToIndexMap;
  
  /** @brief Maps memory addresses to their allocation sizes in bytes */
  inline static std::map<void *, size_t> managedMemoryAddressToSizeMap;
  
  /** @brief Tracks which arrays are application inputs/outputs (for data dependency analysis) */
  inline static std::set<void *> applicationInputs, applicationOutputs;
  
  /** @brief Maps original device pointers to their current relocated pointers in storage */
  inline static std::map<void *, void *> deviceToStorageArrayMap;

  /** @brief Maps original device pointers to their current relocated pointers on device */
  inline static std::map<void *, void *> managedMemoryAddressToAssignedMap;
  
  /** @brief Maps device pointers to host/storage array pointers (internal version of managedDeviceArrayToHostArrayMap) */
  inline static std::map<void *, void *> managedDeviceArrayToHostArrayMap;
  
  /** @brief Storage configuration parameters */
  struct StorageConfig {
    int mainDeviceId;                  // Main GPU device ID
    int storageDeviceId;               // Storage device ID (CPU=-1, other values for secondary GPU)
    bool useNvlink;                    // Whether to use NVLink for GPU-to-GPU transfers
    cudaMemcpyKind prefetchMemcpyKind; // Type of memory copy for prefetching operations
    cudaMemcpyKind offloadMemcpyKind;  // Type of memory copy for offloading operations
    
    StorageConfig() 
      : mainDeviceId(0), 
        storageDeviceId(-1), 
        useNvlink(false),
        prefetchMemcpyKind(cudaMemcpyHostToDevice),
        offloadMemcpyKind(cudaMemcpyDeviceToHost) {}
  };
  inline static StorageConfig storageConfig;

public:
  /**
   * @brief Gets the updated address of a pointer if it has been relocated
   *
   * This function checks if the given pointer exists in the address update map
   * and returns the updated address if found, otherwise returns the original address.
   * 
   * @tparam T Type of data stored in the array
   * @param oldAddress Original pointer address to check for updates
   * @return Updated pointer address if found in map, otherwise the original address
   */
  template <typename T>
  static T* getAddress(T *oldAddress) {
    auto& instance = getInstance();
    auto& mapping = instance.getCurrentAddressMap();
    auto it = mapping.find(static_cast<void *>(oldAddress));
    if (it != mapping.end()) {
      return static_cast<T *>(it->second);
    }
    return oldAddress;
  }

  /**
   * @brief Gets the singleton instance of the MemoryManager
   * 
   * @return Reference to the MemoryManager instance
   */
  static MemoryManager& getInstance();

  // Accessor methods for the private members
  const std::vector<void*>& getManagedAddresses() const;
  const std::map<void*, ArrayId>& getAddressToIndexMap() const;
  const std::map<void*, size_t>& getAddressToSizeMap() const;
  const std::set<void*>& getApplicationInputs() const;
  const std::set<void*>& getApplicationOutputs() const;
  const std::map<void*, void*>& getDeviceToStorageMap() const;
  const std::map<void*, void*>& getCurrentAddressMap() const;
  const std::map<void*, void*>& getDeviceToHostArrayMap() const;
  
  // Editable accessors for methods that need to modify the members
  std::vector<void*>& getEditableManagedAddresses();
  std::map<void*, ArrayId>& getEditableAddressToIndexMap();
  std::map<void*, size_t>& getEditableAddressToSizeMap();
  std::set<void*>& getEditableApplicationInputs();
  std::set<void*>& getEditableApplicationOutputs();
  std::map<void*, void*>& getEditableDeviceToStorageMap();
  std::map<void*, void*>& getEditableCurrentAddressMap();
  std::map<void*, void*>& getEditableDeviceToHostArrayMap();

  // Memory management methods
  size_t getSize(void* addr) const;
  ArrayId getArrayId(void* addr) const;
  bool isManaged(void* addr) const;
  
  // Mapping management
  void updateCurrentMapping(void* originalAddr, void* currentAddr);
  void removeCurrentMapping(void* originalAddr);
  void clearCurrentMappings();
  
  // Storage configuration and management
  void configureStorage(int mainDeviceId, int storageDeviceId, bool useNvlink);
  void cleanStorage();
  
  // Memory movement operations
  void prefetchAllDataToDeviceAsync(const std::vector<ArrayId>& arrayIds, 
                             const std::map<void*, void*>& storageMap,
                             std::map<void*, void*>& currentMap,
                             cudaMemcpyKind memcpyKind,
                             cudaStream_t stream);
  
  void prefetchAllDataToDeviceAsync(const std::vector<ArrayId>& arrayIds, cudaStream_t stream);
  void prefetchToDeviceAsync(const ArrayId arrayId, cudaStream_t stream);
  void offloadFromDeviceAsync(const ArrayId arrayId, cudaStream_t stream);
  
  void prefetchAllDataToDeviceAsync(const std::vector<ArrayId>& arrayIds,
                             cudaMemcpyKind memcpyKind,
                             cudaStream_t stream);
  
  void* allocateInStorage(void* ptr, int storageDeviceId, bool useNvlink);
  void transferData(void* srcPtr, void* dstPtr, cudaMemcpyKind memcpyKind);
  
  void* offloadToStorage(void* ptr, int storageDeviceId, bool useNvlink, 
                       std::map<void*, void*>& storageMap);
  
  // Memory storage operations
  void moveAllManagedMemoryToStorage();
  void moveAllManagedMemoryToStorage(std::map<void*, void*>& storageMap);
  void moveRemainedManagedMemoryToStorage();
  void moveRemainedManagedMemoryToStorageAsync(cudaStream_t stream);
  
  void moveAllManagedMemoryToStorage(int mainDeviceId, int storageDeviceId, bool useNvlink,
                                   std::map<void*, void*>& storageMap);
  void moveAllManagedMemoryToStorage(int mainDeviceId, int storageDeviceId, bool useNvlink);
  
  // Asynchronous operations
  void offloadDataAsync(void* originalPtr, std::map<void*, void*>& storageMap, 
                       std::map<void*, void*>& currentMap,
                       cudaStream_t stream);
  
  void offloadDataAsync(void* originalPtr, std::map<void*, void*>& storageMap, 
                       std::map<void*, void*>& currentMap,
                       cudaMemcpyKind offloadMemcpyKind,
                       cudaStream_t stream);
  
  // Array ID utility methods
  void* getPointerByArrayId(int arrayId) const;
  size_t getSizeByArrayId(int arrayId) const;

  // Make these functions friends so they can access private members
  template <typename T>
  friend void registerManagedMemoryAddress(T *devPtr, size_t size);
  
  template <typename T>
  friend void registerApplicationInput(T *devPtr);
  
  template <typename T>
  friend void registerApplicationOutput(T *devPtr);
  
  friend void updateManagedMemoryAddress(const std::map<void *, void *> oldAddressToNewAddressMap);
};

// Template implementation of helper functions
template <typename T>
void registerManagedMemoryAddress(T *devPtr, size_t size) {
  // Skip small allocations based on configuration threshold
  if (size < ConfigurationManager::getConfig().optimization.minManagedArraySize) {
    return;
  }

  auto ptr = static_cast<void *>(devPtr);
  auto& memManager = MemoryManager::getInstance();
  
  // Only register if not already tracked
  if (memManager.getAddressToIndexMap().count(ptr) == 0) {
    memManager.getEditableManagedAddresses().push_back(ptr);
    memManager.getEditableAddressToIndexMap()[ptr] = memManager.getManagedAddresses().size() - 1;
    memManager.getEditableAddressToSizeMap()[ptr] = size;
    
    // Initialize the mapping to point to itself (no relocation yet)
    memManager.getEditableDeviceToStorageMap()[ptr] = ptr;
  }
}

template <typename T>
void registerApplicationInput(T *devPtr) {
  MemoryManager::getInstance().getEditableApplicationInputs().insert(static_cast<void *>(devPtr));
}

template <typename T>
void registerApplicationOutput(T *devPtr) {
  MemoryManager::getInstance().getEditableApplicationOutputs().insert(static_cast<void *>(devPtr));
}

} // namespace memopt