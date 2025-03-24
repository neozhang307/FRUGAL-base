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
  
  // Fall back to the old method if not found in memoryArrayInfos
  auto it = managedMemoryAddressToIndexMap.find(addr);
  if (it != managedMemoryAddressToIndexMap.end()) {
    return it->second;
  }
  
  return -1; // Invalid ID for unmanaged memory
}

bool MemoryManager::isManaged(void* addr) const {
  // First try the new method
  if (findArrayIdByAddress(addr) >= 0) {
    return true;
  }
  
  // Fall back to old method
  return managedMemoryAddressToIndexMap.count(addr) > 0;
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

void MemoryManager::transferData(void* srcPtr, void* dstPtr, cudaMemcpyKind memcpyKind) {
  size_t size = getSize(srcPtr);
  checkCudaErrors(cudaMemcpy(dstPtr, srcPtr, size, memcpyKind));
}

void* MemoryManager::offloadToStorage(
    void* ptr, 
    int storageDeviceId, 
    bool useNvlink, 
    std::map<void*, void*>& storageMap) {
  // Allocate in storage
  void* storagePtr = allocateInStorage(ptr, storageDeviceId, useNvlink);
  
  // Copy data from device to storage
  transferData(ptr, storagePtr, cudaMemcpyDefault);
  
  // Update mapping
  storageMap[ptr] = storagePtr;
  
  // Update MemoryArrayInfo structure
  ArrayId arrayId = getArrayId(ptr);
  if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
    memoryArrayInfos[arrayId].storageAddress = storagePtr;
  }
  
  // Free original device memory
  checkCudaErrors(cudaFree(ptr));
  
  return storagePtr;
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
  for (auto& info : memoryArrayInfos) {
    info.storageAddress = nullptr;
  }
}

void MemoryManager::clearStorage() {
  // Iterate through all memory array infos and set storageAddress to nullptr
  for (auto& info : memoryArrayInfos) {
    info.storageAddress = nullptr;
  }
}

// Mapping management
void MemoryManager::updateCurrentMapping(void* originalAddr, void* currentAddr) {
  managedMemoryAddressToAssignedMap[originalAddr] = currentAddr;
  
  // Also update the MemoryArrayInfo structure
  ArrayId arrayId = getArrayId(originalAddr);
  if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
    memoryArrayInfos[arrayId].deviceAddress = currentAddr;
  }
}

void MemoryManager::removeCurrentMapping(void* originalAddr) {
  managedMemoryAddressToAssignedMap.erase(originalAddr);
  
  // Also update the MemoryArrayInfo structure
  ArrayId arrayId = getArrayId(originalAddr);
  if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
    // When removing device mapping, set deviceAddress to nullptr
    memoryArrayInfos[arrayId].deviceAddress = nullptr;
  }
}

void MemoryManager::clearCurrentMappings() {
  managedMemoryAddressToAssignedMap.clear();
  
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
    void* storagePtr = managedDeviceArrayToHostArrayMap.at(originalPtr);
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
    managedMemoryAddressToAssignedMap[originalPtr] = devicePtr;
  }
}

void MemoryManager::prefetchAllDataToDevice() {
  auto arrayIds=this->getArrayIds();
  for (auto arrayId : arrayIds) {
    void* originalPtr = managedMemoryAddresses[arrayId];
    void* storagePtr = managedDeviceArrayToHostArrayMap.at(originalPtr);
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
     // Free temporary storage
    if (storageConfig.useNvlink) {
      checkCudaErrors(cudaFree(storagePtr));
    } else {
      checkCudaErrors(cudaFreeHost(storagePtr));
    }
    // Update the current mapping
    managedMemoryAddressToAssignedMap[originalPtr] = devicePtr;
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
  void *devicePtr = this->getCurrentAddressMap().at(dataMovementAddress);
  auto storageAddress = this->memoryArrayInfos[arrayId].storageAddress;

  checkCudaErrors(cudaMemcpyAsync(
    storageAddress,//this->getDeviceToHostArrayMap().at(dataMovementAddress),
    devicePtr,
    dataMovementSize,
    storageConfig.offloadMemcpyKind,
    stream
  ));
  checkCudaErrors(cudaFreeAsync(devicePtr, stream));
  this->removeCurrentMapping(dataMovementAddress);
}

// Memory storage operations
void MemoryManager::offloadAllManagedMemoryToStorage() {
  // Ensure the storage map starts empty
  managedDeviceArrayToHostArrayMap.clear();
  clearStorage();
  // Move each managed memory address to storage
  for (auto ptr : managedMemoryAddresses) {
    // would automatically update MemoryArrayInfo
    offloadToStorage(ptr, storageConfig.storageDeviceId, storageConfig.useNvlink, managedDeviceArrayToHostArrayMap);
  }
  
  // Switch back to main GPU
  checkCudaErrors(cudaSetDevice(storageConfig.mainDeviceId));
  checkCudaErrors(cudaDeviceSynchronize());
}

void MemoryManager::offloadRemainedManagedMemoryToStorage() {
  auto currentAddressMap = this->getEditableCurrentAddressMap();

  for (auto &[oldAddr, newAddr] : currentAddressMap) {
    checkCudaErrors(cudaMemcpy(
      // this->getDeviceToHostArrayMap().at(oldAddr),
      this->getStoragePtr(oldAddr),
      newAddr,
      this->getSize(oldAddr),
      storageConfig.offloadMemcpyKind
    ));
    checkCudaErrors(cudaFree(newAddr));
  }
}

void MemoryManager::offloadRemainedManagedMemoryToStorageAsync(cudaStream_t stream) {
  auto currentAddressMap = this->getEditableCurrentAddressMap();
  for (auto &[oldAddr, newAddr] : currentAddressMap) {
    checkCudaErrors(cudaMemcpyAsync(
      // this->getDeviceToHostArrayMap().at(oldAddr),
      this->getStoragePtr(oldAddr),
      newAddr,
      this->getSize(oldAddr),
      storageConfig.offloadMemcpyKind,
      stream
    ));
    checkCudaErrors(cudaFreeAsync(newAddr, stream));
    this->removeCurrentMapping(oldAddr);
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
    return nullptr;
  }
  
  // First try to find the array ID using the new method
  ArrayId arrayId = findArrayIdByAddress(managedMemAddress);
  
  // If found in memoryArrayInfos, return its storage address
  if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
    return memoryArrayInfos[arrayId].storageAddress;
  }
  
  // Fall back to the old method if not found in memoryArrayInfos
  auto it = managedMemoryAddressToIndexMap.find(managedMemAddress);
  if (it != managedMemoryAddressToIndexMap.end()) {
    arrayId = it->second;
    if (arrayId >= 0 && arrayId < memoryArrayInfos.size()) {
      return memoryArrayInfos[arrayId].storageAddress;
    }
  }
  
  // If we have the legacy host array mapping, try that
  auto hostIt = managedDeviceArrayToHostArrayMap.find(managedMemAddress);
  if (hostIt != managedDeviceArrayToHostArrayMap.end()) {
    return hostIt->second;
  }
  
  // Not found in any map
  return nullptr;
}

} // namespace memopt