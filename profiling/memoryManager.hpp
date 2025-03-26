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
  /** @brief Memory array information structure */
  struct MemoryArrayInfo {
    void* managedMemoryAddress;  // Original memory address
    void* deviceAddress;         // Current device address (could be same as original)
    void* storageAddress;        // Address in storage (host or secondary GPU)
    size_t size;                 // Size of allocation in bytes
  };

  /** @brief Stores complete memory array information */
  inline static std::vector<MemoryArrayInfo> memoryArrayInfos;
  
  /** @brief Ordered list of all managed memory addresses */
  inline static std::vector<void *> managedMemoryAddresses; // Original Memory
  
  /** @brief Maps memory addresses to their array IDs (index in managedMemoryAddresses)
   * @deprecated This map is being phased out in favor of memoryArrayInfos.
   * All new code should access address-to-ID mapping by searching in memoryArrayInfos.
   */
  inline static std::map<void *, ArrayId> managedMemoryAddressToIndexMap;
  
  /** @brief Maps memory addresses to their allocation sizes in bytes
   * @deprecated This map is being phased out in favor of memoryArrayInfos.
   * All new code should access size information through memoryArrayInfos via the array ID.
   */
  // inline static std::map<void *, size_t> managedMemoryAddressToSizeMap;
  
  /** @brief Tracks which arrays are application inputs/outputs (for data dependency analysis) */
  inline static std::set<void *> applicationInputs, applicationOutputs;
  
  // Removed deviceToStorageArrayMap as it's redundant with managedDeviceArrayToHostArrayMap

  /** @brief Maps original device pointers to their current relocated pointers on device 
   * @deprecated This map is being phased out in favor of memoryArrayInfos.deviceAddress.
   * All new code should access device address mappings through memoryArrayInfos instead.
   */
  // inline static std::map<void *, void *> managedMemoryAddressToAssignedMap;
  
  /** @brief Maps device pointers to host/storage array pointers (internal version of managedDeviceArrayToHostArrayMap) */
  // inline static std::map<void *, void *> managedDeviceArrayToHostArrayMap;
  
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
    void* addr = static_cast<void *>(oldAddress);
    
    // First try to find the address in memoryArrayInfos (preferred method)
    ArrayId arrayId = instance.findArrayIdByAddress(addr);
    if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
      void* deviceAddr = memoryArrayInfos[arrayId].deviceAddress;
      if (deviceAddr != nullptr) {
        fprintf(stderr, "[DEBUG-GETADDR] Found address %p in memoryArrayInfos[%d].deviceAddress = %p\n", 
                addr, arrayId, deviceAddr);
        return static_cast<T *>(deviceAddr);
      } else {
        fprintf(stderr, "[DEBUG-GETADDR] Found address %p in memoryArrayInfos[%d] but deviceAddress is nullptr\n", 
                addr, arrayId);
      }
    } else {
      fprintf(stderr, "[DEBUG-GETADDR] Address %p not found in memoryArrayInfos (arrayId = %d)\n", 
              addr, arrayId);
    }
    
    // // Fall back to the legacy map if not found in memoryArrayInfos
    // auto& mapping = instance.getCurrentAddressMap();
    // auto it = mapping.find(addr);
    // if (it != mapping.end()) {
    //   fprintf(stderr, "[DEBUG-GETADDR] Found address %p in legacy map => %p\n", 
    //           addr, it->second);
    //   return static_cast<T *>(it->second);
    // } else {
    //   fprintf(stderr, "[DEBUG-GETADDR] Address %p not found in legacy map, returning original\n", addr);
    // }
    
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
  
  /**
   * @deprecated This method is being phased out. Find address in memoryArrayInfos instead.
   * @return Reference to the map of memory addresses to array IDs
   */
  const std::map<void*, ArrayId>& getAddressToIndexMap() const;
  
  /**
   * @deprecated This method is being phased out. Use getMemoryArrayInfo(arrayId).size instead.
   * @return Reference to the map of memory addresses to sizes
   */
  // const std::map<void*, size_t>& getAddressToSizeMap() const;
  const std::set<void*>& getApplicationInputs() const;
  const std::set<void*>& getApplicationOutputs() const;
  
  /**
   * @deprecated This method is being phased out. Use memoryArrayInfos[arrayId].deviceAddress instead.
   * @return Reference to the map of original addresses to current device addresses
   * 
   * Note: This method now dynamically builds the map from memoryArrayInfos for compatibility
   */
  const std::map<void*, void*>& getCurrentAddressMap() const;
  // const std::map<void*, void*>& getDeviceToHostArrayMap() const;
  
  // Editable accessors for methods that need to modify the members
  std::vector<void*>& getEditableManagedAddresses();
  
  /**
   * @deprecated This method is being phased out. Manipulate memoryArrayInfos directly instead.
   * @return Reference to the map of memory addresses to array IDs for modification
   */
  std::map<void*, ArrayId>& getEditableAddressToIndexMap();
  
  /**
   * @deprecated This method is being phased out. Use memoryArrayInfos vector directly instead.
   * @return Reference to the map of memory addresses to sizes for modification
   */
  // std::map<void*, size_t>& getEditableAddressToSizeMap();
  std::set<void*>& getEditableApplicationInputs();
  std::set<void*>& getEditableApplicationOutputs();
  
  /**
   * @deprecated This method is being phased out. Manipulate memoryArrayInfos[arrayId].deviceAddress directly instead.
   * @return Reference to the map of original addresses to current device addresses for modification
   * 
   * Note: This method now dynamically builds the map from memoryArrayInfos for compatibility
   */
  std::map<void*, void*>& getEditableCurrentAddressMap();
  // std::map<void*, void*>& getEditableDeviceToHostArrayMap();

  // Memory management methods
  size_t getSize(void* addr) const;
  ArrayId getArrayId(void* addr) const;
  bool isManaged(void* addr) const;
  
  // Mapping management
  void updateCurrentMapping(void* originalAddr, void* currentAddr);
  void removeCurrentMapping(void* originalAddr);//note that before this step the pointer would have already been freed
  void clearCurrentMappings();
  
  // Storage configuration and management
  void configureStorage(int mainDeviceId, int storageDeviceId, bool useNvlink);
  void ResetStorageConfig();
  
  /**
   * @brief Releases all storage pointers, freeing memory and setting addresses to nullptr
   * 
   * This function iterates through all MemoryArrayInfo entries, frees the
   * memory at storageAddress if not nullptr, and then sets it to nullptr.
   * Use this to clean up all storage memory at once.
   */
  void releaseStoragePointers();
  
  /**
   * @brief Frees both device and storage memory for a given managed memory address
   * 
   * This function will:
   * 1. Find the array ID associated with the given pointer
   * 2. Free the device memory if deviceAddress is not nullptr
   * 3. Free the storage memory if storageAddress is not nullptr
   * 4. Set both addresses to nullptr in the MemoryArrayInfo structure
   * 
   * @param managedMemoryAddress The original managed memory address to free resources for
   * @return True if the address was found and memory was freed, false otherwise
   */
  bool freeManagedMemory(void* managedMemoryAddress);

  void prefetchAllDataToDeviceAsync(const std::vector<ArrayId>& arrayIds, cudaStream_t stream);
  void prefetchAllDataToDevice();
  void prefetchToDeviceAsync(const ArrayId arrayId, cudaStream_t stream);
  void offloadFromDeviceAsync(const ArrayId arrayId, cudaStream_t stream);
  
  void prefetchAllDataToDeviceAsync(const std::vector<ArrayId>& arrayIds,
                             cudaMemcpyKind memcpyKind,
                             cudaStream_t stream);
  
  void* allocateInStorage(void* ptr, int storageDeviceId, bool useNvlink);
  /**
   * @brief Copies memory from device to storage
   * 
   * This function is specifically designed to copy data from a device address to a storage address
   * (either host memory or secondary GPU memory). It retrieves the size from the memory manager
   * and handles proper address resolution.
   * 
   * @param managedMemoryAddress Managed memory address associated with the data
   * @param storageAddress Storage memory address to copy to
   * @param memcpyKind Type of memory copy (typically cudaMemcpyDefault or cudaMemcpyDeviceToHost)
   */
  void copyMemoryDeviceToStorage(void* managedMemoryAddress, void* storageAddress, cudaMemcpyKind memcpyKind);
  
  /**
   * @brief Frees memory in storage (host or secondary GPU)
   * 
   * This function determines whether to use cudaFree or cudaFreeHost based on 
   * the storage configuration's useNvlink setting.
   * 
   * @param storageAddress The storage pointer to free
   */
  void freeStorage(void* storageAddress);
  
  /**
   * @brief Offloads memory from device to storage and tracks it in a map
   * 
   * This function allocates storage memory, copies data from device to storage,
   * and updates the storage map to track the relationship between addresses.
   * 
   * @param managedMemoryAddress The managed memory address to offload to storage
   * @param storageDeviceId The device ID for storage (-1 for host)
   * @param useNvlink Whether to use NVLink for GPU-to-GPU transfers
   * @param storageMap Map to update with managed-to-storage address mappings
   * @return Pointer to the allocated storage memory
   */
  void* offloadToStorage(void* managedMemoryAddress, int storageDeviceId, bool useNvlink, 
                       std::map<void*, void*>& storageMap);

  /**
   * @brief Offloads memory from device to storage
   * 
   * This overload doesn't require a storage map and updates memoryArrayInfos directly.
   * 
   * @param managedMemoryAddress The managed memory address to offload to storage
   * @param storageDeviceId The device ID for storage (-1 for host)
   * @param useNvlink Whether to use NVLink for GPU-to-GPU transfers
   */
  void offloadToStorage(void* managedMemoryAddress, int storageDeviceId, bool useNvlink);

  // Memory storage operations
  void offloadAllManagedMemoryToStorage();
  void offloadAllManagedMemoryToStorage(std::map<void*, void*>& storageMap);
  void offloadRemainedManagedMemoryToStorage();
  void offloadRemainedManagedMemoryToStorageAsync(cudaStream_t stream);
  
  void offloadAllManagedMemoryToStorage(int mainDeviceId, int storageDeviceId, bool useNvlink,
                                   std::map<void*, void*>& storageMap);
  void offloadAllManagedMemoryToStorage(int mainDeviceId, int storageDeviceId, bool useNvlink);
  
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
  std::vector<ArrayId> getArrayIds() const;
  
  /**
   * @brief Calculate the total size of all registered memory in MB
   * 
   * This function sums the sizes of all memory arrays that have been
   * registered with the MemoryManager and returns the result in megabytes.
   *
   * @return Total size of all registered memory in MB
   */
  double GetMemoryManagedSizeInMB() const;
  
  /**
   * @brief Get the storage pointer for a given managed memory address
   * 
   * This function retrieves the storage address associated with a managed memory address
   * by looking up its array ID and then accessing the corresponding entry in memoryArrayInfos.
   * 
   * @param managedMemAddress The managed memory address to look up
   * @return The storage pointer if found, nullptr otherwise
   */
  void* getStoragePtr(void* managedMemAddress) const;
  
  // MemoryArrayInfo methods
  const MemoryArrayInfo& getMemoryArrayInfo(ArrayId arrayId) const;
  const std::vector<MemoryArrayInfo>& getMemoryArrayInfos() const;
  size_t getMemoryArrayInfosSize() const;
  
  /**
   * @brief Find array ID for a given memory address using the memoryArrayInfos structure
   * 
   * This method replaces the lookup in managedMemoryAddressToIndexMap with a search
   * in the memoryArrayInfos vector.
   * 
   * @param addr Memory address to find
   * @return Array ID if found, -1 if not found, -2 if addr is nullptr
   */
  ArrayId findArrayIdByAddress(void* addr) const;

  // Make these functions friends so they can access private members
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
  
  friend void updateManagedMemoryAddress(const std::map<void *, void *> oldAddressToNewAddressMap);
};



} // namespace memopt