#include "memoryManager.hpp"

namespace memopt {

// Memory management methods specific to the optimized executor (executor_v2.cu)

// Memory information methods used by executor_v2
size_t MemoryManager::getSize(void* addr) const {

  // First try to find the array ID using the mapping
  auto it = managedMemoryAddressToIndexMap.find(addr);

  if (it != managedMemoryAddressToIndexMap.end()) {
    ArrayId arrayId = it->second;
    

    // Use the MemoryArrayInfo if available
    if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
      return memoryArrayInfos[arrayId].size;
    }
  }
  // Only print debug information if verbose output is enabled
  if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
    for(auto it:managedMemoryAddressToIndexMap)
    {
      fprintf(stderr, "[GET-SIZE-ERR]  %p at %d\n", it.first,it.second);
    }
  }
  // // Fall back to the old method if not found in memoryArrayInfos
  // auto sizeIt = managedMemoryAddressToSizeMap.find(addr);
  // if (sizeIt != managedMemoryAddressToSizeMap.end()) {
  //   return sizeIt->second;
  // }
  
  return 0; // Return 0 for unmanaged memory
}

ArrayId MemoryManager::getArrayId(void* addr) const {
  // First try to find the array ID using the new method
  ArrayId arrayId = findArrayIdByAddress(addr);
  if (arrayId >= 0) {
    return arrayId;
  }
  
  // // Fall back to the old method if not found in memoryArrayInfos
  // auto it = managedMemoryAddressToIndexMap.find(addr);
  // if (it != managedMemoryAddressToIndexMap.end()) {
  //   return it->second;
  // }
  
  return -1; // Invalid ID for unmanaged memory
}

bool MemoryManager::isManaged(void* addr) const {
  // First try the new method
  if (findArrayIdByAddress(addr) >= 0) {
    return true;
  }
  return false;
  // Fall back to old method
  // return managedMemoryAddressToIndexMap.count(addr) > 0;
}

// Array ID utility methods
void* MemoryManager::getPointerByArrayId(int arrayId) const {
  // Use the original implementation for now
  if (arrayId >= 0 && arrayId < managedMemoryAddresses.size()) {
    return managedMemoryAddresses[arrayId];
  }
  return nullptr;
}

size_t MemoryManager::getSizeByArrayId(int arrayId) const {
  // Use the original implementation for now
  if (arrayId >= 0 && arrayId < managedMemoryAddresses.size()) {
    void* ptr = managedMemoryAddresses[arrayId];
    return getSize(ptr);
  }
  return 0;
}

std::vector<ArrayId> MemoryManager::getArrayIds() const {
  std::vector<ArrayId> arrayIds;
  arrayIds.reserve(managedMemoryAddresses.size());
  
  for (size_t i = 0; i < managedMemoryAddresses.size(); ++i) {
    arrayIds.push_back(static_cast<ArrayId>(i));
  }
  
  return arrayIds;
}

void* MemoryManager::allocateInStorage(void* ptr, int storageDeviceId, bool useNvlink) {
  // ArrayId id = getArrayId(ptr);
  
  void* newPtr = getStoragePtr(ptr);
  if(newPtr!=nullptr)return newPtr;
  size_t size = getSize(ptr);
  
  // Only print debug information if verbose output is enabled
  if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
    fprintf(stderr, "[DEBUG-OFFLOAD] location 0 address %p device %p for storage %p with size %ld\n", 
                    memoryArrayInfos[0].managedMemoryAddress,memoryArrayInfos[0].deviceAddress,
                    memoryArrayInfos[0].storageAddress,memoryArrayInfos[0].size);
  }
  
  if (useNvlink) {
    // Allocate on secondary GPU
    checkCudaErrors(cudaSetDevice(storageDeviceId));
    checkCudaErrors(cudaMalloc(&newPtr, size));
  } else {
    // Allocate on host memory
    checkCudaErrors(cudaMallocHost(&newPtr, size));
  }
  // Only print debug information if verbose output is enabled
  if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
    fprintf(stderr, "[DEBUG-OFFLOAD] Allocated storage %p for address %p with size %ld\n", newPtr, ptr,size);
  }

  return newPtr;
}

void MemoryManager::copyMemoryDeviceToStorage(void* deviceAddress, void* storageAddress, cudaMemcpyKind memcpyKind, cudaStream_t stream) {
  size_t size = getSize(deviceAddress);
  if (size == 0) {
    // Only print debug information if verbose output is enabled
    if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
      fprintf(stderr, "[DEBUG-COPY] Warning: Size of device address %p is 0, no memory will be copied\n", deviceAddress);
    }
    return;
  }
  
  void* actualDeviceAddress = getAddress(deviceAddress);
  // Only print debug information if verbose output is enabled
  if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
    fprintf(stderr, "[DEBUG-COPY] Copying %lu bytes from device %p to storage %p (actual device address: %p)\n", 
            size, deviceAddress, storageAddress, actualDeviceAddress);
  }
          
  checkCudaErrors(cudaMemcpyAsync(storageAddress, actualDeviceAddress, size, memcpyKind,stream));
  checkCudaErrors(cudaStreamSynchronize(stream));
}

void MemoryManager::freeStorage(void* storageAddress) {
  if (storageAddress == nullptr) {
    return;
  }
  
  // Try both deallocation methods to handle allocation method mismatches
  if (storageConfig.useNvlink) {
    // Try cudaFree first for device allocations
    cudaError_t err = cudaFree(storageAddress);
    if (err != cudaSuccess) {
      // Fallback: try cudaFreeHost in case it was allocated with cudaMallocHost
      cudaError_t fallbackErr = cudaFreeHost(storageAddress);
      if (fallbackErr != cudaSuccess) {
        // Only print debug information if verbose output is enabled
        if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
          fprintf(stderr, "[DEBUG-STORAGE] Failed to free storage address %p - cudaFree: %s, cudaFreeHost: %s\n",
                  storageAddress, cudaGetErrorString(err), cudaGetErrorString(fallbackErr));
        }
      }
    }
  } else {
    // Try cudaFreeHost first for host allocations
    cudaError_t err = cudaFreeHost(storageAddress);
    if (err != cudaSuccess) {
      // Fallback: try cudaFree in case it was allocated with cudaMalloc
      cudaError_t fallbackErr = cudaFree(storageAddress);
      if (fallbackErr != cudaSuccess) {
        // Only print debug information if verbose output is enabled
        if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
          fprintf(stderr, "[DEBUG-STORAGE] Failed to free storage address %p - cudaFreeHost: %s, cudaFree: %s\n",
                  storageAddress, cudaGetErrorString(err), cudaGetErrorString(fallbackErr));
        }
      }
    }
  }
}

bool MemoryManager::freeManagedMemory(void* managedMemoryAddress, cudaStream_t stream) {
  // Only print debug information if verbose output is enabled
  bool verbose = ConfigurationManager::getConfig().execution.enableVerboseOutput;
  
  if (managedMemoryAddress == nullptr) {
    if (verbose) {
      fprintf(stderr, "[DEBUG-FREE] Cannot free nullptr\n");
    }
    return false;
  }
  
  // Find the array ID for this pointer
  ArrayId arrayId = findArrayIdByAddress(managedMemoryAddress);
  if (arrayId < 0 || arrayId >= memoryArrayInfos.size()) {
    if (verbose) {
      fprintf(stderr, "[DEBUG-FREE] Address %p not found in memoryArrayInfos (arrayId = %d)\n", 
              managedMemoryAddress, arrayId);
    }
    return false;
  }
  
  // Free device memory if it exists
  if (memoryArrayInfos[arrayId].deviceAddress != nullptr) {
    if (verbose) {
      fprintf(stderr, "[DEBUG-FREE] Freeing device memory %p for address %p\n", 
              memoryArrayInfos[arrayId].deviceAddress, managedMemoryAddress);
    }
    checkCudaErrors(cudaFreeAsync(memoryArrayInfos[arrayId].deviceAddress,stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    memoryArrayInfos[arrayId].deviceAddress = nullptr;
  }
  
  // Free storage memory if it exists
  if (memoryArrayInfos[arrayId].storageAddress != nullptr) {
    if (verbose) {
      fprintf(stderr, "[DEBUG-FREE] Freeing storage memory %p for address %p\n", 
              memoryArrayInfos[arrayId].storageAddress, managedMemoryAddress);
    }
    freeStorage(memoryArrayInfos[arrayId].storageAddress);
    memoryArrayInfos[arrayId].storageAddress = nullptr;
  }
  
  if (verbose) {
    fprintf(stderr, "[DEBUG-FREE] Successfully freed all memory for address %p (arrayId = %d)\n", 
            managedMemoryAddress, arrayId);
  }
  return true;
}


void MemoryManager::offloadToStorage(
    void* managedMemoryAddress, 
    int storageDeviceId, 
    bool useNvlink,
    cudaStream_t stream) 
{
  // Only print debug information if verbose output is enabled
  bool verbose = ConfigurationManager::getConfig().execution.enableVerboseOutput;
  
  if (verbose) {
    fprintf(stderr, "[DEBUG-OFFLOAD] Starting offload for address %p\n", managedMemoryAddress);
  }

  ArrayId arrayId = getArrayId(managedMemoryAddress);
  if (verbose) {
    fprintf(stderr, "[DEBUG-OFFLOAD] getArrayId returned %d for address %p\n", arrayId, managedMemoryAddress);
  }
  
  if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
    // Allocate in storage
    void* storagePtr = allocateInStorage(managedMemoryAddress, storageDeviceId, useNvlink);//if it is not empty
    if (verbose) {
      fprintf(stderr, "[DEBUG-OFFLOAD] Allocated storage %p for address %p\n", storagePtr, managedMemoryAddress);
    }
    
    
    
    // Update MemoryArrayInfo structure
    memoryArrayInfos[arrayId].storageAddress = storagePtr;
    if (verbose) {
      fprintf(stderr, "[DEBUG-OFFLOAD] Updated memoryArrayInfos[%d].storageAddress = %p\n", 
              arrayId, storagePtr);
    }
    if(memoryArrayInfos[arrayId].deviceAddress!=nullptr)
    {
      // Copy data from device to storage
      copyMemoryDeviceToStorage(managedMemoryAddress, storagePtr, cudaMemcpyDefault);
      
      if (verbose) {
        fprintf(stderr, "[DEBUG-OFFLOAD] Data transferred from %p to %p\n", managedMemoryAddress, storagePtr);
      }

      checkCudaErrors(cudaFreeAsync(memoryArrayInfos[arrayId].deviceAddress,stream));
      
      checkCudaErrors(cudaStreamSynchronize(stream));
      memoryArrayInfos[arrayId].deviceAddress=nullptr;
      if (verbose) {
        fprintf(stderr, "[DEBUG-OFFLOAD] Freed original device memory %p of %p\n",
                memoryArrayInfos[arrayId].deviceAddress, managedMemoryAddress);
      }
    }
    
  } else {
    if (verbose) {
      fprintf(stderr, "[DEBUG-OFFLOAD] WARNING: Could not update memoryArrayInfos for address %p, arrayId=%d\n", 
              managedMemoryAddress, arrayId);
    }
  }
  
  // Verify storageAddress was set
  if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
    if (verbose) {
      fprintf(stderr, "[DEBUG-OFFLOAD] Verification: memoryArrayInfos[%d].storageAddress = %p\n", 
              arrayId, memoryArrayInfos[arrayId].storageAddress);
    }
  }
}

// Configuration
void MemoryManager::configureStorage(int mainDeviceId, int storageDeviceId, bool useNvlink) {
  storageConfig.mainDeviceId = mainDeviceId;
  storageConfig.storageDeviceId = storageDeviceId;
  storageConfig.useNvlink = useNvlink;
  
  // Set appropriate memcpy kinds based on the configuration
  if (useNvlink) {
    storageConfig.prefetchMemcpyKind = cudaMemcpyDeviceToDevice;
    storageConfig.offloadMemcpyKind = cudaMemcpyDeviceToDevice;
  } else {
    storageConfig.prefetchMemcpyKind = cudaMemcpyHostToDevice;
    storageConfig.offloadMemcpyKind = cudaMemcpyDeviceToHost;
  }
}

void MemoryManager::ResetStorageConfig() {
  if (storageConfig.useNvlink) {
    disablePeerAccessForNvlink(storageConfig.mainDeviceId, storageConfig.storageDeviceId);
  }
  
  // Clear storage addresses in MemoryArrayInfo
  // for (auto& info : memoryArrayInfos) {
  //   info.storageAddress = nullptr;
  // }
  
  // Also clear the legacy mapping to maintain consistency
  // This is crucial when running multiple iterations to prevent stale mappings
  // managedDeviceArrayToHostArrayMap.clear();
  
  // fprintf(stderr, "[DEBUG-CLEAN] cleanStorage cleared both memoryArrayInfos.storageAddress and managedDeviceArrayToHostArrayMap\n");
}

void MemoryManager::releaseStoragePointers() {
  // Only print debug information if verbose output is enabled
  bool verbose = ConfigurationManager::getConfig().execution.enableVerboseOutput;
  
  // Iterate through all memory array infos, free storage memory and set storageAddress to nullptr
  for (auto& info : memoryArrayInfos) {
    if(info.storageAddress != nullptr)
    {
      if(info.deviceAddress==nullptr)
      {
        if (verbose) {
          fprintf(stderr,"[DEBUG] releaseStoragePointers not working on %p as data stored in storage now\n",info.managedMemoryAddress);
        }
        continue;
      }
      // Store the address before setting to nullptr to avoid double-free issues
      void* storageAddress = info.storageAddress;
      // Set to nullptr first, then free the memory
      info.storageAddress = nullptr;
      
      if (verbose) {
        fprintf(stderr, "[DEBUG-STORAGE] Releasing storage pointer %p for managed address %p\n",
                storageAddress, info.managedMemoryAddress);
      }
      
      freeStorage(storageAddress);
      ArrayId arrayId = findArrayIdByAddress(info.managedMemoryAddress);
      memoryArrayInfos[arrayId].storageAddress = nullptr;
      
    }
  }
  checkCudaErrors(cudaDeviceSynchronize());
  
  // Legacy cleanup commented out as part of transition
  // managedDeviceArrayToHostArrayMap.clear();
}

// Mapping management
void MemoryManager::updateCurrentMapping(void* originalAddr, void* currentAddr) {
  // managedMemoryAddressToAssignedMap[originalAddr] = currentAddr;
  
  // Also update the MemoryArrayInfo structure
  ArrayId arrayId = getArrayId(originalAddr);
  if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
    memoryArrayInfos[arrayId].deviceAddress = currentAddr;
  }
}
//note that before this step the pointer would have already been freed
void MemoryManager::removeCurrentMapping(void* originalAddr) {
  // managedMemoryAddressToAssignedMap.erase(originalAddr);
  
  // Also update the MemoryArrayInfo structure
  ArrayId arrayId = getArrayId(originalAddr);
  if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
    // When removing device mapping, set deviceAddress to nullptr
    memoryArrayInfos[arrayId].deviceAddress = nullptr;
  }
}

void MemoryManager::clearCurrentMappings() {
  // managedMemoryAddressToAssignedMap.clear();
  
  // Clear device pointers in all MemoryArrayInfo entries
  for (auto& info : memoryArrayInfos) {
    info.deviceAddress = nullptr;
  }
}

void MemoryManager::prefetchAllDataToDeviceAsync(
    const std::vector<ArrayId>& arrayIds,
    cudaStream_t stream) {
  for (auto arrayId : arrayIds) {
    if(memoryArrayInfos[arrayId].deviceAddress!=nullptr)continue;//already in device
    void* originalPtr = managedMemoryAddresses[arrayId];
    void* storagePtr = getStoragePtr(originalPtr);//managedDeviceArrayToHostArrayMap.at(originalPtr);
    if(storagePtr==nullptr)continue;//not in sotrage 

    size_t size = getSize(originalPtr);
    
    // Allocate on device and copy data from storage
    // void* devicePtr;
    if (memoryArrayInfos[arrayId].deviceAddress==nullptr)
      checkCudaErrors(cudaMallocAsync(&memoryArrayInfos[arrayId].deviceAddress, size, stream));

    checkCudaErrors(cudaMemcpyAsync(
      memoryArrayInfos[arrayId].deviceAddress, 
      storagePtr, 
      size, 
      storageConfig.prefetchMemcpyKind, 
      stream
    ));
    
    // Update the current mapping
    // managedMemoryAddressToAssignedMap[originalPtr] = devicePtr;
    // memoryArrayInfos[arrayId].deviceAddress=devicePtr;
  }
}

void MemoryManager::prefetchAllDataToDevice(cudaStream_t stream) {
  auto arrayIds=this->getArrayIds();
  for (auto arrayId : arrayIds) {
    if(memoryArrayInfos[arrayId].deviceAddress!=nullptr)continue;//already in device
    
    void* originalPtr = managedMemoryAddresses[arrayId];
    void* storagePtr = getStoragePtr(originalPtr);//managedDeviceArrayToHostArrayMap.at(originalPtr);
    size_t size = getSize(originalPtr);
    
    // Allocate on device and copy data from storage
    void* devicePtr;
    checkCudaErrors(cudaMallocAsync(&devicePtr, size,stream));
    checkCudaErrors(cudaMemcpyAsync(
      devicePtr, 
      storagePtr, 
      size, 
      storageConfig.prefetchMemcpyKind, stream
    ));
    cudaStreamSynchronize(stream);
    memoryArrayInfos[arrayId].deviceAddress=devicePtr;
  }
}

void MemoryManager::prefetchToDeviceAsync(const ArrayId arrayId, cudaStream_t stream) {
  void *devicePtr;
  auto dataMovementSize = this->getSizeByArrayId(arrayId);
  auto dataMovementAddress = this->getPointerByArrayId(arrayId);
  auto storageAddress = this->memoryArrayInfos[arrayId].storageAddress;

  checkCudaErrors(cudaMallocAsync(&devicePtr, dataMovementSize, stream));
  checkCudaErrors(cudaMemcpyAsync(
    devicePtr,
    storageAddress,//this->getDeviceToHostArrayMap().at(dataMovementAddress),
    dataMovementSize,
    storageConfig.prefetchMemcpyKind,
    stream
  ));
  this->updateCurrentMapping(dataMovementAddress, devicePtr);
}

void MemoryManager::offloadFromDeviceAsync(const ArrayId arrayId, cudaStream_t stream) {
  auto dataMovementSize = this->getSizeByArrayId(arrayId);
  auto dataMovementAddress = this->getPointerByArrayId(arrayId);
  // void *devicePtr = this->getCurrentAddressMap().at(dataMovementAddress);
  auto devicePtr = this->memoryArrayInfos[arrayId].deviceAddress;
  auto storageAddress = this->memoryArrayInfos[arrayId].storageAddress;

  checkCudaErrors(cudaMemcpyAsync(
    storageAddress,//this->getDeviceToHostArrayMap().at(dataMovementAddress),
    devicePtr,
    dataMovementSize,
    storageConfig.offloadMemcpyKind,
    stream
  ));
  checkCudaErrors(cudaFreeAsync(devicePtr, stream));
  this->memoryArrayInfos[arrayId].deviceAddress=nullptr;
}

// Memory storage operations
void MemoryManager::offloadAllManagedMemoryToStorage() {
  // Only print debug information if verbose output is enabled
  bool verbose = ConfigurationManager::getConfig().execution.enableVerboseOutput;
  
  if (verbose) {
    fprintf(stderr, "[DEBUG-OFFLOAD-ALL] Starting offloadAllManagedMemoryToStorage, address count: %zu\n", 
            managedMemoryAddresses.size());
  }
  
  // Ensure the storage map starts empty
  // managedDeviceArrayToHostArrayMap.clear();
  // if (verbose) {
  //   fprintf(stderr, "[DEBUG-OFFLOAD-ALL] Cleared managedDeviceArrayToHostArrayMap\n");
  // }
  
  releaseStoragePointers();//only if devicememory is not null
  if (verbose) {
    fprintf(stderr, "[DEBUG-OFFLOAD-ALL] Called releaseStoragePointers() to reset memoryArrayInfos\n");
  }
  
  // Move each managed memory address to storage
  for (size_t i = 0; i < managedMemoryAddresses.size(); i++) {
    void* ptr = memoryArrayInfos[i].managedMemoryAddress;//managedMemoryAddresses[i];
    if (verbose) {
      fprintf(stderr, "[DEBUG-OFFLOAD-ALL] Processing address %zu/%zu: %p\n", 
              i+1, managedMemoryAddresses.size(), ptr);
    }
    
    // device -> storage 
    offloadToStorage(ptr, storageConfig.storageDeviceId, 
                     storageConfig.useNvlink);                                   
  }
  if (verbose) {
    fprintf(stderr, "[DEBUG-OFFLOAD-ALL] Summary - memoryArrayInfos entries: %zu\n", 
            memoryArrayInfos.size());
  }
          
  // Switch back to main GPU
  checkCudaErrors(cudaSetDevice(storageConfig.mainDeviceId));
  checkCudaErrors(cudaDeviceSynchronize());
  
  if (verbose) {
    fprintf(stderr, "[DEBUG-OFFLOAD-ALL] Completed offloadAllManagedMemoryToStorage\n");
  }
}

void MemoryManager::offloadRemainedManagedMemoryToStorage(cudaStream_t stream) {
  // Only print debug information if verbose output is enabled
  bool verbose = ConfigurationManager::getConfig().execution.enableVerboseOutput;

  for (auto& info : memoryArrayInfos) {
    if(info.deviceAddress==nullptr)continue;//already in storage
    
    // Handle case where storageAddress is nullptr
    if(info.storageAddress == nullptr) {
      if (verbose) {
        fprintf(stderr, "[DEBUG-OFFLOAD] Storage address is nullptr for %p, allocating now\n", 
                info.managedMemoryAddress);
      }
      
      // Allocate storage space first
      void* storagePtr = allocateInStorage(info.managedMemoryAddress, 
                                           storageConfig.storageDeviceId, 
                                           storageConfig.useNvlink);
      
      // Update the storage address in memoryArrayInfos
      ArrayId arrayId = findArrayIdByAddress(info.managedMemoryAddress);
      if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
        memoryArrayInfos[arrayId].storageAddress = storagePtr;
        
        if (verbose) {
          fprintf(stderr, "[DEBUG-OFFLOAD] Updated memoryArrayInfos[%d].storageAddress = %p\n", 
                  arrayId, storagePtr);
        }
      }
      
      // Use updated storage address
      info.storageAddress = storagePtr;
    }
    
    // Now copy data from device to storage
    checkCudaErrors(cudaMemcpyAsync(
      info.storageAddress,
      info.deviceAddress,
      info.size,
      storageConfig.offloadMemcpyKind,
      stream
    ));
    checkCudaErrors(cudaFreeAsync(info.deviceAddress, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    
    ArrayId arrayId = findArrayIdByAddress(info.managedMemoryAddress);
    memoryArrayInfos[arrayId].deviceAddress = nullptr;
  }
  
  // Free unused graph memory back to the OS
  int currentDevice;
  checkCudaErrors(cudaGetDevice(&currentDevice));
  checkCudaErrors(cudaDeviceGraphMemTrim(currentDevice));
}

void MemoryManager::offloadRemainedManagedMemoryToStorageAsync(cudaStream_t stream) {
  // Only print debug information if verbose output is enabled
  bool verbose = ConfigurationManager::getConfig().execution.enableVerboseOutput;

  for (auto& info : memoryArrayInfos) {
    if(info.deviceAddress==nullptr)continue;//already in storage
    
    // Handle case where storageAddress is nullptr
    if(info.storageAddress == nullptr) {
      if (verbose) {
        fprintf(stderr, "[DEBUG-OFFLOAD-ASYNC] Storage address is nullptr for %p, allocating now\n", 
                info.managedMemoryAddress);
      }
      
      // Allocate storage space first
      void* storagePtr = allocateInStorage(info.managedMemoryAddress, 
                                          storageConfig.storageDeviceId, 
                                          storageConfig.useNvlink);
      
      // Update the storage address in memoryArrayInfos
      ArrayId arrayId = findArrayIdByAddress(info.managedMemoryAddress);
      if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
        memoryArrayInfos[arrayId].storageAddress = storagePtr;
        
        if (verbose) {
          fprintf(stderr, "[DEBUG-OFFLOAD-ASYNC] Updated memoryArrayInfos[%d].storageAddress = %p\n", 
                  arrayId, storagePtr);
        }
      }
      
      // Use updated storage address
      info.storageAddress = storagePtr;
    }

    checkCudaErrors(cudaMemcpyAsync(
      info.storageAddress,
      info.deviceAddress,
      info.size,
      storageConfig.offloadMemcpyKind,
      stream
    ));
    checkCudaErrors(cudaFreeAsync(info.deviceAddress, stream));
    
    ArrayId arrayId = findArrayIdByAddress(info.managedMemoryAddress);
    memoryArrayInfos[arrayId].deviceAddress = nullptr;
  }
}

// MemoryArrayInfo methods
const MemoryManager::MemoryArrayInfo& MemoryManager::getMemoryArrayInfo(ArrayId arrayId) const {
  assert(arrayId >= 0 && arrayId < memoryArrayInfos.size());
  return memoryArrayInfos[arrayId];
}

const std::vector<MemoryManager::MemoryArrayInfo>& MemoryManager::getMemoryArrayInfos() const {
  return memoryArrayInfos;
}

size_t MemoryManager::getMemoryArrayInfosSize() const {
  return memoryArrayInfos.size();
}

ArrayId MemoryManager::findArrayIdByAddress(void* addr) const {
  // Handle nullptr case
  if (addr == nullptr) {
    return -2;
  }
  
  // Search through memoryArrayInfos to find the matching address
  for (size_t i = 0; i < memoryArrayInfos.size(); ++i) {
    if (memoryArrayInfos[i].managedMemoryAddress == addr) {
      return static_cast<ArrayId>(i);
    }
  }
  return -1; // Not found
}

void* MemoryManager::getStoragePtr(void* managedMemAddress) const {
  // Only print debug information if verbose output is enabled
  bool verbose = ConfigurationManager::getConfig().execution.enableVerboseOutput;
  
  // Handle nullptr case
  if (managedMemAddress == nullptr) {
    if (verbose) {
      fprintf(stderr, "[DEBUG] getStoragePtr: nullptr input\n");
    }
    return nullptr;
  }

  // Try to find the array ID using the new method
  ArrayId arrayId = findArrayIdByAddress(managedMemAddress);
  
  // If found in memoryArrayInfos, return its storage address
  if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
    void* storageAddr = memoryArrayInfos[arrayId].storageAddress;
    if (storageAddr == nullptr) {
      if (verbose) {
        fprintf(stderr, "[DEBUG] getStoragePtr: Found arrayId %d for address %p but storageAddress is nullptr\n", 
                arrayId, managedMemAddress);
      }
      
      // Special case - legacy map is empty but we found the array ID
      // Check if applicationInputs has this address, which would indicate it's a valid array
      // This helps recover after cleanStorage() clears the addresses
      // if (managedDeviceArrayToHostArrayMap.empty() && 
      //     (applicationInputs.count(managedMemAddress) > 0 || applicationOutputs.count(managedMemAddress) > 0)) {
      //   if (verbose) {
      //     fprintf(stderr, "[DEBUG] getStoragePtr: Address %p is a known application input/output but storage was cleared (cleanStorage ran)\n", 
      //             managedMemAddress);
      //   }
      // }
    } else {
      // Success case
      if (verbose) {
        fprintf(stderr, "[DEBUG] getStoragePtr: Successfully found storage address %p for managed address %p using arrayId %d\n", 
                storageAddr, managedMemAddress, arrayId);
      }
    }
    return storageAddr;
  } else {
    if (verbose) {
      fprintf(stderr, "[DEBUG] getStoragePtr: findArrayIdByAddress failed with arrayId %d for address %p\n", 
              arrayId, managedMemAddress);
    }
  }
  
  // Fall back to the old index map method
  // auto it = managedMemoryAddressToIndexMap.find(managedMemAddress);
  // if (it != managedMemoryAddressToIndexMap.end()) {
  //   arrayId = it->second;
  //   if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
  //     void* storageAddr = memoryArrayInfos[arrayId].storageAddress;
  //     if (storageAddr == nullptr) {
  //       fprintf(stderr, "[DEBUG] getStoragePtr: Found address in indexMap with arrayId %d but storageAddress is nullptr\n", 
  //               arrayId);
  //     } else {
  //       fprintf(stderr, "[DEBUG] getStoragePtr: Successfully found storage address %p using indexMap with arrayId %d\n", 
  //               storageAddr, arrayId);
  //     }
  //     return storageAddr;
  //   } else {
  //     fprintf(stderr, "[DEBUG] getStoragePtr: Found in indexMap but arrayId %d is invalid\n", arrayId);
  //   }
  // } else {
  //   fprintf(stderr, "[DEBUG] getStoragePtr: Address %p not found in managedMemoryAddressToIndexMap\n", managedMemAddress);
  // }

  
  // Legacy map lookup first (fastest direct lookup)
  // This ensures backward compatibility and protects against array index changes
  // auto hostIt = managedDeviceArrayToHostArrayMap.find(managedMemAddress);
  // if (hostIt != managedDeviceArrayToHostArrayMap.end()) {
  //   if (verbose) {
  //     fprintf(stderr, "[DEBUG] getStoragePtr: Found in legacy hostArrayMap: %p -> %p\n", 
  //             managedMemAddress, hostIt->second);
  //   }
  //   return hostIt->second;
  // }  
  // Not found in any map
  if (verbose) {
    fprintf(stderr, "[DEBUG] getStoragePtr: CRITICAL! Address %p not found in any map!\n", managedMemAddress);
  }
  return nullptr;
}

// Template implementation of member function
template <typename T>
void MemoryManager::registerManagedMemoryAddress(T *devPtr, size_t size) {
  // Skip small allocations based on configuration threshold
  if (size < ConfigurationManager::getConfig().optimization.minManagedArraySize) {
    return;
  }

  auto ptr = static_cast<void *>(devPtr);
  
  // Only register if not already tracked
  if (this->findArrayIdByAddress(ptr) < 0 && this->getAddressToIndexMap().count(ptr) == 0) {
    // Update existing data structures
    this->getEditableManagedAddresses().push_back(ptr);
    ArrayId arrayId = this->getManagedAddresses().size() - 1;
    
    // Maintain old index map for backward compatibility
    // Will be removed in the future once transition to memoryArrayInfos is complete
    this->getEditableAddressToIndexMap()[ptr] = arrayId;
    
    //malloc memory for device address
    void* devicePtr;
    // checkCudaErrors(cudaMalloc(&devicePtr, size));
    //copy data from managed memory to device memory
    // checkCudaErrors(cudaMemcpy(devicePtr, ptr, size, cudaMemcpyDeviceToDevice));
    //free managed memory
    // checkCudaErrors(cudaFree(ptr));

    // Create and populate new MemoryArrayInfo structure
    MemoryArrayInfo info;
    info.managedMemoryAddress = ptr;
    info.deviceAddress = ptr;//devicePtr;         // Initially, the device address is the same as the original
    info.storageAddress = nullptr;    // Initially, no storage allocated
    info.size = size;
    
    // Expand the vector if needed
    if (memoryArrayInfos.size() <= arrayId) {
      memoryArrayInfos.resize(arrayId + 1);
    }
    memoryArrayInfos[arrayId] = info;
  }
}

// Implementation of application input/output registration
template <typename T>
void MemoryManager::registerApplicationInput(T *devPtr) {
  this->getEditableApplicationInputs().insert(static_cast<void *>(devPtr));
}

template <typename T>
void MemoryManager::registerApplicationOutput(T *devPtr) {
  this->getEditableApplicationOutputs().insert(static_cast<void *>(devPtr));
}

// Explicit template instantiation for common types
template void MemoryManager::registerManagedMemoryAddress<float>(float*, size_t);
template void MemoryManager::registerManagedMemoryAddress<double>(double*, size_t);
template void MemoryManager::registerManagedMemoryAddress<int>(int*, size_t);
template void MemoryManager::registerManagedMemoryAddress<char>(char*, size_t);
template void MemoryManager::registerManagedMemoryAddress<unsigned int>(unsigned int*, size_t);
template void MemoryManager::registerManagedMemoryAddress<void>(void*, size_t);

// Explicit template instantiation for registerApplicationInput
template void MemoryManager::registerApplicationInput<float>(float*);
template void MemoryManager::registerApplicationInput<double>(double*);
template void MemoryManager::registerApplicationInput<int>(int*);
template void MemoryManager::registerApplicationInput<char>(char*);
template void MemoryManager::registerApplicationInput<unsigned int>(unsigned int*);
template void MemoryManager::registerApplicationInput<void>(void*);

// Explicit template instantiation for registerApplicationOutput
template void MemoryManager::registerApplicationOutput<float>(float*);
template void MemoryManager::registerApplicationOutput<double>(double*);
template void MemoryManager::registerApplicationOutput<int>(int*);
template void MemoryManager::registerApplicationOutput<char>(char*);
template void MemoryManager::registerApplicationOutput<unsigned int>(unsigned int*);
template void MemoryManager::registerApplicationOutput<void>(void*);

// GB-based memory management functions
bool MemoryManager::consumeGPUMemory(size_t sizeInGB) {
  // Convert GB to bytes
  size_t sizeInBytes = sizeInGB * 1024ULL * 1024ULL * 1024ULL;
  void* ptr = nullptr;
  
  // We use the main GPU device
  int currentDevice;
  cudaGetDevice(&currentDevice);
  cudaSetDevice(storageConfig.mainDeviceId);
  
  // Attempt to allocate memory
  cudaError_t result = cudaMalloc(&ptr, sizeInBytes);
  
  // Restore original device
  cudaSetDevice(currentDevice);
  
  if (result != cudaSuccess) {
    if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
      fprintf(stderr, "[MEMORY-INFO] Failed to allocate %zu GB of dummy memory: %s\n", 
              sizeInGB, cudaGetErrorString(result));
    }
    return false;
  }
  
  // Add to our tracking vector
  dummyAllocations.emplace_back(ptr, sizeInBytes);
  
  if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
    fprintf(stderr, "[MEMORY-INFO] Successfully allocated %zu GB of dummy memory (%p)\n", 
            sizeInGB, ptr);
  }
  
  // Initialize the memory to prevent GPU driver optimizations that might not actually allocate
  // The device will actually allocate the memory when it's first written to
  cudaMemset(ptr, 0xAA, sizeInBytes);
  
  return true;
}

double MemoryManager::getConsumedGPUMemory() const {
  size_t totalSizeInBytes = 0;
  
  // Sum up all dummy allocations
  for (const auto& alloc : dummyAllocations) {
    totalSizeInBytes += alloc.sizeInBytes;
  }
  
  // Convert to GB
  return static_cast<double>(totalSizeInBytes) / (1024.0 * 1024.0 * 1024.0);
}

bool MemoryManager::releaseDummyMemory() {
  if (dummyAllocations.empty()) {
    return false;
  }
  
  // Free all dummy allocations
  for (const auto& alloc : dummyAllocations) {
    cudaFree(alloc.ptr);
  }
  
  size_t count = dummyAllocations.size();
  double totalGB = getConsumedGPUMemory();
  
  // Clear the vector
  dummyAllocations.clear();
  
  if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
    fprintf(stderr, "[MEMORY-INFO] Released %zu dummy allocations (%.2f GB total)\n", 
            count, totalGB);
  }
  
  return true;
}

bool MemoryManager::claimNecessaryMemory(size_t sizeInGB) {
  // Release any existing dummy allocations
  if (!dummyAllocations.empty()) {
    releaseDummyMemory();
  }
  
  // Get total and free memory on the device
  size_t free, total;
  int currentDevice;
  cudaGetDevice(&currentDevice);
  cudaSetDevice(storageConfig.mainDeviceId);
  
  cudaError_t result = cudaMemGetInfo(&free, &total);
  if (result != cudaSuccess) {
    if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
      fprintf(stderr, "[MEMORY-ERROR] Failed to get memory info: %s\n", 
              cudaGetErrorString(result));
    }
    cudaSetDevice(currentDevice);
    return false;
  }
  
  // Convert to GB for calculations and logging
  double totalGB = static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0);
  double freeGB = static_cast<double>(free) / (1024.0 * 1024.0 * 1024.0);
  
  // Calculate how much memory to consume to leave exactly the requested amount
  if (sizeInGB > freeGB) {
    if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
      fprintf(stderr, "[MEMORY-ERROR] Cannot claim %.2f GB - only %.2f GB available\n", 
              static_cast<double>(sizeInGB), freeGB);
    }
    cudaSetDevice(currentDevice);
    return false;
  }
  
  double consumeGB = freeGB - sizeInGB;
  
  // Ensure we don't consume everything by leaving at least 1GB headroom
  if (consumeGB > 1.0) {
    consumeGB -= 1.0;
  }
  
  // Log what we're doing (verbose only)
  if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
    fprintf(stderr, "[MEMORY-INFO] Total: %.2f GB, Free: %.2f GB\n", totalGB, freeGB);
    fprintf(stderr, "[MEMORY-INFO] Consuming %.2f GB to leave %.2f GB available\n", 
            consumeGB, static_cast<double>(sizeInGB));
  }
  
  // Restore original device
  cudaSetDevice(currentDevice);
  
  // Consume the calculated amount of memory
  if (consumeGB > 0) {
    return consumeGPUMemory(static_cast<size_t>(consumeGB));
  }
  
  return true;
}

// Domain enlargement implementation
bool MemoryManager::reregisterManagedArrayWithLargerCPUData(void* originalAddress, void* newLargerCpuPtr, size_t newSize) {
  // Find the ArrayId for the original address
  ArrayId arrayId = getArrayId(originalAddress);
  if (arrayId < 0 || arrayId >= memoryArrayInfos.size()) {
    return false; // Address not found
  }
  
  // Update the MemoryArrayInfo with larger CPU data
  memoryArrayInfos[arrayId].managedMemoryAddress = originalAddress;  // Keep same original address
  memoryArrayInfos[arrayId].storageAddress = newLargerCpuPtr;        // Point storage to new larger CPU data
  memoryArrayInfos[arrayId].deviceAddress = nullptr;                // Not on device initially
  memoryArrayInfos[arrayId].size = newSize;                         // Update to larger size
  
  // Update address mappings - keep same original address in vectors/maps
  // managedMemoryAddresses[arrayId] already contains originalAddress
  // managedMemoryAddressToIndexMap[originalAddress] already maps to arrayId
  
  // Only print debug information if verbose output is enabled
  if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
    fprintf(stderr, "[DOMAIN-ENLARGE] Re-registered array %d: originalAddr=%p, newCpuPtr=%p, newSize=%zu bytes\n", 
            arrayId, originalAddress, newLargerCpuPtr, newSize);
  }
  
  return true;
}

int MemoryManager::reregisterMultipleArraysWithLargerCPUData(const std::map<void*, ArrayReplacement>& replacements) {
  int successCount = 0;
  
  for (const auto& [originalAddress, replacement] : replacements) {
    if (reregisterManagedArrayWithLargerCPUData(originalAddress, replacement.cpuPtr, replacement.size)) {
      successCount++;
    } else {
      // Only print debug information if verbose output is enabled
      if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
        fprintf(stderr, "[DOMAIN-ENLARGE] Failed to re-register array at address %p\n", originalAddress);
      }
    }
  }
  
  if (ConfigurationManager::getConfig().execution.enableVerboseOutput) {
    fprintf(stderr, "[DOMAIN-ENLARGE] Successfully re-registered %d out of %zu arrays\n", 
            successCount, replacements.size());
  }
  
  return successCount;
}

bool MemoryManager::copyManagedArrayToHost(void* managedMemoryAddress, void* hostBuffer, size_t size) {
  
  // Find the array ID for this managed address
  ArrayId arrayId = getArrayId(managedMemoryAddress);
  if (arrayId < 0 || arrayId >= memoryArrayInfos.size()) {
    fprintf(stderr, "ERROR: Invalid array ID %d for address %p in copyManagedArrayToHost\n", 
            arrayId, managedMemoryAddress);
    return false;
  }
  
  const auto& arrayInfo = memoryArrayInfos[arrayId];
  
  // Check if data is currently on device or in storage
  if (arrayInfo.deviceAddress != nullptr) {
    // Data is on device - use CUDA device-to-host copy
    cudaError_t err = cudaMemcpy(hostBuffer, arrayInfo.deviceAddress, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      fprintf(stderr, "ERROR: cudaMemcpy DeviceToHost failed for address %p: %s\n", 
              managedMemoryAddress, cudaGetErrorString(err));
      return false;
    }
  } else if (arrayInfo.storageAddress != nullptr) {
    // Data is in storage/host - use standard memcpy
    memcpy(hostBuffer, arrayInfo.storageAddress, size);
  } else {
    // Fallback: try the managed memory address directly
    // This handles cases where data might still be accessible via unified memory
    cudaError_t err = cudaMemcpy(hostBuffer, managedMemoryAddress, size, cudaMemcpyDefault);
    if (err != cudaSuccess) {
      fprintf(stderr, "ERROR: Fallback cudaMemcpy failed for address %p: %s\n", 
              managedMemoryAddress, cudaGetErrorString(err));
      return false;
    }
  }
  
  return true;
}

} // namespace memopt