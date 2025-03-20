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
    int mainDeviceId;      // Main GPU device ID
    int storageDeviceId;   // Storage device ID (CPU=-1, other values for secondary GPU)
    bool useNvlink;        // Whether to use NVLink for GPU-to-GPU transfers
    
    StorageConfig() : mainDeviceId(0), storageDeviceId(-1), useNvlink(false) {}
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
   * @param addressUpdate Map containing original addresses mapped to their new locations
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

  // === Enhanced API for future code ===

  /**
   * @brief Gets the singleton instance of the MemoryManager
   * 
   * @return Reference to the MemoryManager instance
   */
  static MemoryManager& getInstance() {
    static MemoryManager instance;
    return instance;
  }

  // Accessor methods that wrap the static members for modern code
  const std::vector<void*>& getManagedAddresses() const { return managedMemoryAddresses; }
  const std::map<void*, ArrayId>& getAddressToIndexMap() const { return managedMemoryAddressToIndexMap; }
  const std::map<void*, size_t>& getAddressToSizeMap() const { return managedMemoryAddressToSizeMap; }
  const std::set<void*>& getApplicationInputs() const { return applicationInputs; }
  const std::set<void*>& getApplicationOutputs() const { return applicationOutputs; }
  const std::map<void*, void*>& getDeviceToStorageMap() const { return deviceToStorageArrayMap; }
  const std::map<void*, void*>& getCurrentAddressMap() const { return managedMemoryAddressToAssignedMap; }
  const std::map<void*, void*>& getDeviceToHostArrayMap() const { return managedDeviceArrayToHostArrayMap; }
  
  // Editable accessors for methods that need to modify the members
  std::vector<void*>& getEditableManagedAddresses() { return managedMemoryAddresses; }
  std::map<void*, ArrayId>& getEditableAddressToIndexMap() { return managedMemoryAddressToIndexMap; }
  std::map<void*, size_t>& getEditableAddressToSizeMap() { return managedMemoryAddressToSizeMap; }
  std::set<void*>& getEditableApplicationInputs() { return applicationInputs; }
  std::set<void*>& getEditableApplicationOutputs() { return applicationOutputs; }
  std::map<void*, void*>& getEditableDeviceToStorageMap() { return deviceToStorageArrayMap; }
  std::map<void*, void*>& getEditableCurrentAddressMap() { return managedMemoryAddressToAssignedMap; }
  std::map<void*, void*>& getEditableDeviceToHostArrayMap() { return managedDeviceArrayToHostArrayMap; }

  // Get size for a specific managed address
  size_t getSize(void* addr) const {
    auto it = managedMemoryAddressToSizeMap.find(addr);
    if (it != managedMemoryAddressToSizeMap.end()) {
      return it->second;
    }
    return 0; // Return 0 for unmanaged memory
  }

  // Get array ID for a specific managed address
  ArrayId getArrayId(void* addr) const {
    auto it = managedMemoryAddressToIndexMap.find(addr);
    if (it != managedMemoryAddressToIndexMap.end()) {
      return it->second;
    }
    return -1; // Invalid ID for unmanaged memory
  }

  // Check if an address is managed
  bool isManaged(void* addr) const {
    return managedMemoryAddressToIndexMap.count(addr) > 0;
  }

  // For new code to update current mappings
  void updateCurrentMapping(void* originalAddr, void* currentAddr) {
    managedMemoryAddressToAssignedMap[originalAddr] = currentAddr;
  }

  // Remove a mapping
  void removeCurrentMapping(void* originalAddr) {
    managedMemoryAddressToAssignedMap.erase(originalAddr);
  }

  // Clear all mappings
  void clearCurrentMappings() {
    managedMemoryAddressToAssignedMap.clear();
  }
  
  /**
   * @brief Configures storage parameters for memory offloading
   *
   * This function sets the parameters used for memory movements between device and storage.
   * It should be called before using moveAllManagedMemoryToStorage or other memory operations.
   *
   * @param mainDeviceId The ID of the main GPU device
   * @param storageDeviceId The ID of the storage device (-1 for CPU, other values for secondary GPU)
   * @param useNvlink Whether to use NVLink for GPU-to-GPU transfers
   */
  void configureStorage(int mainDeviceId, int storageDeviceId, bool useNvlink) {
    storageConfig.mainDeviceId = mainDeviceId;
    storageConfig.storageDeviceId = storageDeviceId;
    storageConfig.useNvlink = useNvlink;
  }
  
  /**
   * @brief Prefetches data from storage to device for needed arrays
   *
   * This method:
   * 1. Allocates memory on the device
   * 2. Copies data from storage to the device
   * 3. Updates the current address mapping
   *
   * @param arrayIds Array IDs to prefetch
   * @param storageMap Map containing storage locations
   * @param currentMap Map to update with current device locations
   * @param memcpyKind Type of memory copy (host-to-device or device-to-device)
   * @param stream CUDA stream to use for async operations
   */
  void prefetchToDevice(const std::vector<ArrayId>& arrayIds,
                         const std::map<void*, void*>& storageMap,
                         std::map<void*, void*>& currentMap,
                         cudaMemcpyKind memcpyKind,
                         cudaStream_t stream) {
    for (auto arrayId : arrayIds) {
      void* originalPtr = managedMemoryAddresses[arrayId];
      void* storagePtr = storageMap.at(originalPtr);
      size_t size = getSize(originalPtr);
      
      // Allocate on device and copy data from storage
      void* devicePtr;
      checkCudaErrors(cudaMallocAsync(&devicePtr, size, stream));
      checkCudaErrors(cudaMemcpyAsync(
        devicePtr, 
        storagePtr, 
        size, 
        memcpyKind, 
        stream
      ));
      
      // Update the current mapping
      currentMap[originalPtr] = devicePtr;
    }
  }
  
  /**
   * @brief Prefetches data from storage to device for needed arrays using internal maps
   *
   * This overload uses the internal managedDeviceToStorageMap and managedMemoryAddressToAssignedMap
   * for simplicity when using the MemoryManager's built-in storage.
   *
   * @param arrayIds Array IDs to prefetch
   * @param memcpyKind Type of memory copy (host-to-device or device-to-device)
   * @param stream CUDA stream to use for async operations
   */
  void prefetchToDevice(const std::vector<ArrayId>& arrayIds,
                         cudaMemcpyKind memcpyKind,
                         cudaStream_t stream) {
    for (auto arrayId : arrayIds) {
      void* originalPtr = managedMemoryAddresses[arrayId];
      void* storagePtr = managedDeviceArrayToHostArrayMap.at(originalPtr);
      size_t size = getSize(originalPtr);
      
      // Allocate on device and copy data from storage
      void* devicePtr;
      checkCudaErrors(cudaMallocAsync(&devicePtr, size, stream));
      checkCudaErrors(cudaMemcpyAsync(
        devicePtr, 
        storagePtr, 
        size, 
        memcpyKind, 
        stream
      ));
      
      // Update the current mapping
      managedMemoryAddressToAssignedMap[originalPtr] = devicePtr;
    }
  }
  
  /**
   * @brief Allocates memory for a managed array in storage (host or GPU)
   * 
   * @param ptr Original pointer to managed memory
   * @param storageDeviceId Device ID for storage (cudaCpuDeviceId for host)
   * @param useNvlink Whether to use NVLink for GPU-to-GPU transfers
   * @return void* Pointer to newly allocated memory in storage
   */
  void* allocateInStorage(void* ptr, int storageDeviceId, bool useNvlink) {
    void* newPtr = nullptr;
    size_t size = getSize(ptr);
    
    if (useNvlink) {
      // Allocate on secondary GPU
      checkCudaErrors(cudaSetDevice(storageDeviceId));
      checkCudaErrors(cudaMalloc(&newPtr, size));
    } else {
      // Allocate on host memory
      checkCudaErrors(cudaMallocHost(&newPtr, size));
    }
    
    return newPtr;
  }
  
  /**
   * @brief Transfers data from device to storage (host or secondary GPU)
   * 
   * @param srcPtr Source pointer (on device)
   * @param dstPtr Destination pointer (in storage)
   * @param memcpyKind Type of memory copy operation
   */
  void transferData(void* srcPtr, void* dstPtr, cudaMemcpyKind memcpyKind) {
    size_t size = getSize(srcPtr);
    checkCudaErrors(cudaMemcpy(dstPtr, srcPtr, size, memcpyKind));
  }
  
  /**
   * @brief Moves data from device to storage and frees device memory
   * 
   * This operation:
   * 1. Allocates memory in storage
   * 2. Copies data from device to storage
   * 3. Frees the original device memory
   * 4. Updates the storage mapping
   * 
   * @param ptr Device pointer to managed memory
   * @param storageDeviceId Device ID for storage
   * @param useNvlink Whether to use NVLink
   * @param storageMap Map for tracking device-to-storage relationships
   * @return void* Pointer to memory in storage
   */
  void* offloadToStorage(void* ptr, int storageDeviceId, bool useNvlink, 
                        std::map<void*, void*>& storageMap) {
    // Allocate in storage
    void* storagePtr = allocateInStorage(ptr, storageDeviceId, useNvlink);
    
    // Copy data from device to storage
    transferData(ptr, storagePtr, cudaMemcpyDefault);
    
    // Update mapping
    storageMap[ptr] = storagePtr;
    
    // Free original device memory
    checkCudaErrors(cudaFree(ptr));
    
    return storagePtr;
  }
  
  /**
   * @brief Moves all managed memory to storage (host or secondary GPU)
   *
   * Iterates through all managed memory addresses and offloads them to storage,
   * either host memory or a secondary GPU depending on configuration.
   * Uses the storage parameters previously set with configureStorage().
   * Stores mappings in internal managedDeviceArrayToHostArrayMap.
   */
  void moveAllManagedMemoryToStorage() {
    // Ensure the storage map starts empty
    managedDeviceArrayToHostArrayMap.clear();
    
    // Move each managed memory address to storage
    for (auto ptr : managedMemoryAddresses) {
      offloadToStorage(ptr, storageConfig.storageDeviceId, storageConfig.useNvlink, managedDeviceArrayToHostArrayMap);
    }
    
    // Switch back to main GPU
    checkCudaErrors(cudaSetDevice(storageConfig.mainDeviceId));
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  /**
   * @brief Moves all managed memory to storage (host or secondary GPU)
   *
   * Iterates through all managed memory addresses and offloads them to storage.
   * Uses the storage parameters previously set with configureStorage().
   * This overload provides the same functionality but with an external map for compatibility.
   *
   * @param storageMap Map for tracking device-to-storage relationships
   */
  void moveAllManagedMemoryToStorage(std::map<void*, void*>& storageMap) {
    // Ensure the storage map starts empty
    storageMap.clear();
    
    // Move each managed memory address to storage
    for (auto ptr : managedMemoryAddresses) {
      offloadToStorage(ptr, storageConfig.storageDeviceId, storageConfig.useNvlink, storageMap);
    }
    
    // Also update internal map
    managedDeviceArrayToHostArrayMap = storageMap;
    
    // Switch back to main GPU
    checkCudaErrors(cudaSetDevice(storageConfig.mainDeviceId));
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  /**
   * @brief Moves all managed memory to storage (host or secondary GPU)
   *
   * Overloaded version that takes explicit parameters.
   * The original version remains for backward compatibility.
   *
   * @param mainDeviceId Device ID to return to after offloading
   * @param storageDeviceId Device ID for storage
   * @param useNvlink Whether to use NVLink
   * @param storageMap Map for tracking device-to-storage relationships
   */
  void moveAllManagedMemoryToStorage(int mainDeviceId, int storageDeviceId, bool useNvlink,
                                     std::map<void*, void*>& storageMap) {
    // Configure storage with the provided parameters
    configureStorage(mainDeviceId, storageDeviceId, useNvlink);
    
    // Call the simpler version that uses the configured parameters and external map
    moveAllManagedMemoryToStorage(storageMap);
  }
  
  /**
   * @brief Moves all managed memory to storage (host or secondary GPU)
   *
   * Overloaded version that takes explicit parameters but uses internal map.
   *
   * @param mainDeviceId Device ID to return to after offloading
   * @param storageDeviceId Device ID for storage
   * @param useNvlink Whether to use NVLink
   */
  void moveAllManagedMemoryToStorage(int mainDeviceId, int storageDeviceId, bool useNvlink) {
    // Configure storage with the provided parameters
    configureStorage(mainDeviceId, storageDeviceId, useNvlink);
    
    // Call the simpler version that uses the configured parameters and internal map
    moveAllManagedMemoryToStorage();
  }
  
  /**
   * @brief Offloads data from device to storage asynchronously
   *
   * Similar to offloadToStorage but runs asynchronously on a stream
   *
   * @param originalPtr Original device pointer
   * @param storageMap Map of storage locations
   * @param currentMap Map of current device locations
   * @param offloadMemcpyKind Type of memory copy for offloading
   * @param stream CUDA stream for async operations
   */
  void offloadDataAsync(void* originalPtr, std::map<void*, void*>& storageMap, 
                       std::map<void*, void*>& currentMap, cudaMemcpyKind offloadMemcpyKind,
                       cudaStream_t stream) {
    void* devicePtr = currentMap[originalPtr];
    void* storagePtr = storageMap[originalPtr];
    size_t size = getSize(originalPtr);
    
    // Copy data from device to storage and free device memory
    checkCudaErrors(cudaMemcpyAsync(storagePtr, devicePtr, size, offloadMemcpyKind, stream));
    checkCudaErrors(cudaFreeAsync(devicePtr, stream));
    
    // Remove from current map
    currentMap.erase(originalPtr);
  }
  
  /**
   * @brief Prefetches data for specific array IDs
   *
   * This method initializes device memory for a set of arrays by:
   * 1. Looking up array addresses by ID
   * 2. Allocating device memory of the appropriate size
   * 3. Copying data from storage to device 
   * 4. Updating the address mapping
   *
   * @param arrayIds Array IDs to initialize
   * @param storageMap Map of storage locations
   * @param memcpyKind Type of memory copy for prefetching
   * @param stream CUDA stream for async operations
   */
  void prefetchArraysByIds(const std::vector<int>& arrayIds,
                         std::map<void*, void*>& storageMap,
                         cudaMemcpyKind memcpyKind,
                         cudaStream_t stream) {
    for (auto arrayId : arrayIds) {
      auto ptr = managedMemoryAddresses[arrayId];
      auto size = managedMemoryAddressToSizeMap[ptr];
      auto newPtr = storageMap[ptr];
      
      // Allocate on device and copy data from storage
      void *devicePtr;
      checkCudaErrors(cudaMallocAsync(&devicePtr, size, stream));
      checkCudaErrors(cudaMemcpyAsync(devicePtr, newPtr, size, memcpyKind, stream));
      managedMemoryAddressToAssignedMap[ptr] = devicePtr;  // Update the address mapping
    }
  }
  
  /**
   * @brief Gets the pointer to managed memory for a specific array ID
   *
   * @param arrayId The ID of the array to look up
   * @return void* Pointer to the managed memory
   */
  void* getPointerByArrayId(int arrayId) const {
    if (arrayId >= 0 && arrayId < managedMemoryAddresses.size()) {
      return managedMemoryAddresses[arrayId];
    }
    return nullptr;
  }
  
  /**
   * @brief Gets the size of managed memory for a specific array ID
   *
   * @param arrayId The ID of the array to look up
   * @return size_t Size of the managed memory in bytes
   */
  size_t getSizeByArrayId(int arrayId) const {
    if (arrayId >= 0 && arrayId < managedMemoryAddresses.size()) {
      void* ptr = managedMemoryAddresses[arrayId];
      return getSize(ptr);
    }
    return 0;
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
  MemoryManager::getInstance().getEditableApplicationInputs().insert(static_cast<void *>(devPtr));
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
  MemoryManager::getInstance().getEditableApplicationOutputs().insert(static_cast<void *>(devPtr));
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
  auto& memManager = MemoryManager::getInstance();
  
  // Save the old size mapping for reference
  auto oldManagedMemoryAddressToSizeMap = memManager.getAddressToSizeMap();

  // Clear existing mappings to rebuild them
  memManager.getEditableAddressToSizeMap().clear();
  memManager.getEditableAddressToIndexMap().clear();

  // Update all addresses and rebuild mappings
  for (int i = 0; i < memManager.getManagedAddresses().size(); i++) {
    // Ensure every old address has a corresponding new address
    assert(oldAddressToNewAddressMap.count(memManager.getManagedAddresses()[i]) == 1);

    const auto newAddr = oldAddressToNewAddressMap.at(memManager.getManagedAddresses()[i]);
    const auto oldAddr = memManager.getManagedAddresses()[i];

    // Update address in the main list
    memManager.getEditableManagedAddresses()[i] = newAddr;
    // Transfer the size information to the new address
    memManager.getEditableAddressToSizeMap()[newAddr] = oldManagedMemoryAddressToSizeMap.at(oldAddr);
    // Maintain the same array ID
    memManager.getEditableAddressToIndexMap()[newAddr] = i;
    // Update the device-to-host mapping
    memManager.getEditableDeviceToStorageMap()[oldAddr] = newAddr;

    // Update application input registry if this was an input
    auto& appInputs = memManager.getEditableApplicationInputs();
    if (memManager.getApplicationInputs().count(oldAddr) > 0) {
      appInputs.erase(oldAddr);
      appInputs.insert(newAddr);
    }
    
    // Update application output registry if this was an output
    auto& appOutputs = memManager.getEditableApplicationOutputs();
    if (memManager.getApplicationOutputs().count(oldAddr) > 0) {
      appOutputs.erase(oldAddr);
      appOutputs.insert(newAddr);
    }
  }
}

}  // namespace memopt