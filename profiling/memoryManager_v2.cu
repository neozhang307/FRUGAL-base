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
  for(auto it:managedMemoryAddressToIndexMap)
  {
    fprintf(stderr, "[GET-SIZE-ERR]  %p at %d\n", it.first,it.second);
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
  void* newPtr = nullptr;
  size_t size = getSize(ptr);
  fprintf(stderr, "[DEBUG-OFFLOAD] location 0 address %p device %p for storage %p with size %ld\n", 
                      memoryArrayInfos[0].managedMemoryAddress,memoryArrayInfos[0].deviceAddress,
                     memoryArrayInfos[0].storageAddress,memoryArrayInfos[0].size);
  
  if (useNvlink) {
    // Allocate on secondary GPU
    checkCudaErrors(cudaSetDevice(storageDeviceId));
    checkCudaErrors(cudaMalloc(&newPtr, size));
  } else {
    // Allocate on host memory
    checkCudaErrors(cudaMallocHost(&newPtr, size));
  }
  fprintf(stderr, "[DEBUG-OFFLOAD] Allocated storage %p for address %p with size %ld\n", newPtr, ptr,size);

  return newPtr;
}

void MemoryManager::transferData(void* ptr, void* dstPtr, cudaMemcpyKind memcpyKind) {
  size_t size = getSize(ptr);
  checkCudaErrors(cudaMemcpy(dstPtr, getAddress(ptr), size, memcpyKind));
}

void MemoryManager::freeStorage(void* ptr) {
  if (ptr == nullptr) {
    return;
  }
  
  if (storageConfig.useNvlink) {
    checkCudaErrors(cudaFree(ptr));
  } else {
    checkCudaErrors(cudaFreeHost(ptr));
  }
}

void* MemoryManager::offloadToStorage(
    void* ptr, 
    int storageDeviceId, 
    bool useNvlink, 
    std::map<void*, void*>& storageMap) 
{
  fprintf(stderr, "[DEBUG-OFFLOAD] Starting offload for address %p\n", ptr);
  
  // Allocate in storage
  void* storagePtr = allocateInStorage(ptr, storageDeviceId, useNvlink);
  fprintf(stderr, "[DEBUG-OFFLOAD] Allocated storage %p for address %p\n", storagePtr, ptr);
  
  // Copy data from device to storage
  transferData(ptr, storagePtr, cudaMemcpyDefault);
  fprintf(stderr, "[DEBUG-OFFLOAD] Data transferred from %p to %p\n", ptr, storagePtr);
  
  // Update mapping
  storageMap[ptr] = storagePtr;
  fprintf(stderr, "[DEBUG-OFFLOAD] Updated storageMap: %p -> %p\n", ptr, storagePtr);
  
  // Update MemoryArrayInfo structure
  ArrayId arrayId = getArrayId(ptr);
  fprintf(stderr, "[DEBUG-OFFLOAD] getArrayId returned %d for address %p\n", arrayId, ptr);
  
  if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
    memoryArrayInfos[arrayId].storageAddress = storagePtr;
    fprintf(stderr, "[DEBUG-OFFLOAD] Updated memoryArrayInfos[%d].storageAddress = %p\n", 
            arrayId, storagePtr);
  } else {
    fprintf(stderr, "[DEBUG-OFFLOAD] WARNING: Could not update memoryArrayInfos for address %p, arrayId=%d\n", 
            ptr, arrayId);
  }
  
  // Verify storageAddress was set
  if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
    fprintf(stderr, "[DEBUG-OFFLOAD] Verification: memoryArrayInfos[%d].storageAddress = %p\n", 
            arrayId, memoryArrayInfos[arrayId].storageAddress);
  }
  
  // Free original device memory
  checkCudaErrors(cudaFree(ptr));
  fprintf(stderr, "[DEBUG-OFFLOAD] Freed original device memory %p\n", ptr);
  
  return storagePtr;
}

void MemoryManager::offloadToStorage(
    void* ptr, 
    int storageDeviceId, 
    bool useNvlink) 
{
  fprintf(stderr, "[DEBUG-OFFLOAD] Starting offload for address %p\n", ptr);

  ArrayId arrayId = getArrayId(ptr);
  fprintf(stderr, "[DEBUG-OFFLOAD] getArrayId returned %d for address %p\n", arrayId, ptr);
  if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
    // Allocate in storage
    void* storagePtr = allocateInStorage(ptr, storageDeviceId, useNvlink);
    fprintf(stderr, "[DEBUG-OFFLOAD] Allocated storage %p for address %p\n", storagePtr, ptr);
    
    // Copy data from device to storage
    transferData(ptr, storagePtr, cudaMemcpyDefault);
    
    fprintf(stderr, "[DEBUG-OFFLOAD] Data transferred from %p to %p\n", ptr, storagePtr);
    
    // Update MemoryArrayInfo structure

  
 
    memoryArrayInfos[arrayId].storageAddress = storagePtr;
    fprintf(stderr, "[DEBUG-OFFLOAD] Updated memoryArrayInfos[%d].storageAddress = %p\n", 
            arrayId, storagePtr);
    
    checkCudaErrors(cudaFree(memoryArrayInfos[arrayId].deviceAddress));
    memoryArrayInfos[arrayId].deviceAddress=nullptr;
    fprintf(stderr, "[DEBUG-OFFLOAD] Freed original device memory %p of %p\n",memoryArrayInfos[arrayId].deviceAddress, ptr);
  } else {
    fprintf(stderr, "[DEBUG-OFFLOAD] WARNING: Could not update memoryArrayInfos for address %p, arrayId=%d\n", 
            ptr, arrayId);
  }
  
  // Verify storageAddress was set
  if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
    fprintf(stderr, "[DEBUG-OFFLOAD] Verification: memoryArrayInfos[%d].storageAddress = %p\n", 
            arrayId, memoryArrayInfos[arrayId].storageAddress);
  }
  
  // Free original device memory
  
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

void MemoryManager::cleanStorage() {
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

void MemoryManager::clearStorage() {
  // Iterate through all memory array infos and set storageAddress to nullptr
  for (auto& info : memoryArrayInfos) {
    if(info.storageAddress!=nullptr)
    {
      // Store the address before setting to nullptr so we can print it
      void* addr = info.storageAddress;
      // Set to nullptr first, then call freeStorage to avoid double-free issues
      info.storageAddress = nullptr;
      fprintf(stderr, "[DEBUG-CLEAR] clearStorage cleared memoryArrayInfos.storageAddress for %p at %p \n",info.managedMemoryAddress, addr);
      freeStorage(addr);
    }
    
  }
  checkCudaErrors(cudaDeviceSynchronize());
  // Also clear the legacy mapping to maintain consistency
  // This ensures both data structures stay in sync
  // managedDeviceArrayToHostArrayMap.clear();
  
  // fprintf(stderr, "[DEBUG-CLEAR] clearStorage cleared both memoryArrayInfos.storageAddress and managedDeviceArrayToHostArrayMap\n");
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
    void* originalPtr = managedMemoryAddresses[arrayId];
    void* storagePtr = getStoragePtr(originalPtr);//managedDeviceArrayToHostArrayMap.at(originalPtr);
    size_t size = getSize(originalPtr);
    
    // Allocate on device and copy data from storage
    void* devicePtr;
    checkCudaErrors(cudaMallocAsync(&devicePtr, size, stream));
    checkCudaErrors(cudaMemcpyAsync(
      devicePtr, 
      storagePtr, 
      size, 
      storageConfig.prefetchMemcpyKind, 
      stream
    ));
    
    // Update the current mapping
    // managedMemoryAddressToAssignedMap[originalPtr] = devicePtr;
    memoryArrayInfos[arrayId].deviceAddress=devicePtr;
  }
}

void MemoryManager::prefetchAllDataToDevice() {
  auto arrayIds=this->getArrayIds();
  for (auto arrayId : arrayIds) {
    void* originalPtr = managedMemoryAddresses[arrayId];
    void* storagePtr = getStoragePtr(originalPtr);//managedDeviceArrayToHostArrayMap.at(originalPtr);
    size_t size = getSize(originalPtr);
    
    // Allocate on device and copy data from storage
    void* devicePtr;
    checkCudaErrors(cudaMalloc(&devicePtr, size));
    checkCudaErrors(cudaMemcpy(
      devicePtr, 
      storagePtr, 
      size, 
      storageConfig.prefetchMemcpyKind
    ));
    
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
  fprintf(stderr, "[DEBUG-OFFLOAD-ALL] Starting offloadAllManagedMemoryToStorage, address count: %zu\n", 
          managedMemoryAddresses.size());
  
  // Ensure the storage map starts empty
  // managedDeviceArrayToHostArrayMap.clear();
  // fprintf(stderr, "[DEBUG-OFFLOAD-ALL] Cleared managedDeviceArrayToHostArrayMap\n");
  
  clearStorage();
  fprintf(stderr, "[DEBUG-OFFLOAD-ALL] Called clearStorage() to reset memoryArrayInfos\n");
  
  // Move each managed memory address to storage
  for (size_t i = 0; i < managedMemoryAddresses.size(); i++) {
    void* ptr = memoryArrayInfos[i].managedMemoryAddress;//managedMemoryAddresses[i];
    fprintf(stderr, "[DEBUG-OFFLOAD-ALL] Processing address %zu/%zu: %p\n", 
            i+1, managedMemoryAddresses.size(), ptr);
    
    // device -> storage 
    offloadToStorage(ptr, storageConfig.storageDeviceId, 
                                       storageConfig.useNvlink);                                   
  }
  fprintf(stderr, "[DEBUG-OFFLOAD-ALL] Summary - memoryArrayInfos entries: %zu\n", 
          memoryArrayInfos.size());
          
  // Switch back to main GPU
  checkCudaErrors(cudaSetDevice(storageConfig.mainDeviceId));
  checkCudaErrors(cudaDeviceSynchronize());
  fprintf(stderr, "[DEBUG-OFFLOAD-ALL] Completed offloadAllManagedMemoryToStorage\n");
}

void MemoryManager::offloadRemainedManagedMemoryToStorage() {
  // auto currentAddressMap = this->getEditableCurrentAddressMap();

  // for (auto &[oldAddr, newAddr] : currentAddressMap) {
  //   checkCudaErrors(cudaMemcpy(
  //     // this->getDeviceToHostArrayMap().at(oldAddr),
  //     this->getStoragePtr(oldAddr),
  //     newAddr,
  //     this->getSize(oldAddr),
  //     storageConfig.offloadMemcpyKind
  //   ));
  //   checkCudaErrors(cudaFree(newAddr));
  // }

  for (auto info : memoryArrayInfos) {
    if(info.deviceAddress==nullptr)continue;

    checkCudaErrors(cudaMemcpy(
      // this->getDeviceToHostArrayMap().at(oldAddr),
      info.storageAddress,
      info.deviceAddress,
      info.size,
      storageConfig.offloadMemcpyKind
    ));
    checkCudaErrors(cudaFree(info.deviceAddress));
    info.deviceAddress=nullptr;
  }
}

void MemoryManager::offloadRemainedManagedMemoryToStorageAsync(cudaStream_t stream) {
  // auto currentAddressMap = this->getEditableCurrentAddressMap();
  // for (auto &[oldAddr, newAddr] : currentAddressMap) {
  //   checkCudaErrors(cudaMemcpyAsync(
  //     // this->getDeviceToHostArrayMap().at(oldAddr),
  //     this->getStoragePtr(oldAddr),
  //     newAddr,
  //     this->getSize(oldAddr),
  //     storageConfig.offloadMemcpyKind,
  //     stream
  //   ));
  //   checkCudaErrors(cudaFreeAsync(newAddr, stream));
  //   this->removeCurrentMapping(oldAddr);
  // }

  for (auto info : memoryArrayInfos) {
    if(info.deviceAddress==nullptr)continue;

    checkCudaErrors(cudaMemcpyAsync(
      // this->getDeviceToHostArrayMap().at(oldAddr),
      info.storageAddress,
      info.deviceAddress,
      info.size,
      storageConfig.offloadMemcpyKind,
      stream
    ));
    checkCudaErrors(cudaFreeAsync(info.deviceAddress,stream));
    info.deviceAddress=nullptr;
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
  // Handle nullptr case
  if (managedMemAddress == nullptr) {
    fprintf(stderr, "[DEBUG] getStoragePtr: nullptr input\n");
    return nullptr;
  }

  
  // Then try to find the array ID using the new method
  ArrayId arrayId = findArrayIdByAddress(managedMemAddress);
  
  // If found in memoryArrayInfos, return its storage address
  if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
    void* storageAddr = memoryArrayInfos[arrayId].storageAddress;
    if (storageAddr == nullptr) {
      fprintf(stderr, "[DEBUG] getStoragePtr: Found arrayId %d for address %p but storageAddress is nullptr\n", 
              arrayId, managedMemAddress);
      
      // Special case - legacy map is empty but we found the array ID
      // Check if applicationInputs has this address, which would indicate it's a valid array
      // This helps recover after cleanStorage() clears the addresses
      // if (managedDeviceArrayToHostArrayMap.empty() && 
      //     (applicationInputs.count(managedMemAddress) > 0 || applicationOutputs.count(managedMemAddress) > 0)) {
      //   fprintf(stderr, "[DEBUG] getStoragePtr: Address %p is a known application input/output but storage was cleared (cleanStorage ran)\n", 
      //           managedMemAddress);
      // }
    } else {
      // Success case
      fprintf(stderr, "[DEBUG] getStoragePtr: Successfully found storage address %p for managed address %p using arrayId %d\n", 
              storageAddr, managedMemAddress, arrayId);
    }
    return storageAddr;
  } else {
    fprintf(stderr, "[DEBUG] getStoragePtr: findArrayIdByAddress failed with arrayId %d for address %p\n", 
            arrayId, managedMemAddress);
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
  //   fprintf(stderr, "[DEBUG] getStoragePtr: Found in legacy hostArrayMap: %p -> %p\n", 
  //           managedMemAddress, hostIt->second);
  //   return hostIt->second;
  // }  
  // Not found in any map
  fprintf(stderr, "[DEBUG] getStoragePtr: CRITICAL! Address %p not found in any map!\n", managedMemAddress);
  return nullptr;
}

} // namespace memopt