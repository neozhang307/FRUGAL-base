#include "memoryManager.hpp"

namespace memopt {

// Memory management methods specific to the optimized executor (executor_v2.cu)

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
}

// Mapping management
void MemoryManager::updateCurrentMapping(void* originalAddr, void* currentAddr) {
  managedMemoryAddressToAssignedMap[originalAddr] = currentAddr;
}

void MemoryManager::removeCurrentMapping(void* originalAddr) {
  managedMemoryAddressToAssignedMap.erase(originalAddr);
}

void MemoryManager::clearCurrentMappings() {
  managedMemoryAddressToAssignedMap.clear();
}

// Memory prefetching methods
void MemoryManager::prefetchAllDataToDevice(
    const std::vector<ArrayId>& arrayIds,
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

void MemoryManager::prefetchAllDataToDevice(
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

void MemoryManager::prefetchToDevice(const ArrayId arrayId, cudaStream_t stream) {
  void *devicePtr;
  auto dataMovementSize = this->getSizeByArrayId(arrayId);
  auto dataMovementAddress = this->getPointerByArrayId(arrayId);

  checkCudaErrors(cudaMallocAsync(&devicePtr, dataMovementSize, stream));
  checkCudaErrors(cudaMemcpyAsync(
    devicePtr,
    this->getDeviceToHostArrayMap().at(dataMovementAddress),
    dataMovementSize,
    storageConfig.prefetchMemcpyKind,
    stream
  ));
  this->updateCurrentMapping(dataMovementAddress, devicePtr);
}

void MemoryManager::offloadFromDevice(const ArrayId arrayId, cudaStream_t stream) {
  auto dataMovementSize = this->getSizeByArrayId(arrayId);
  auto dataMovementAddress = this->getPointerByArrayId(arrayId);
  void *devicePtr = this->getCurrentAddressMap().at(dataMovementAddress);

  checkCudaErrors(cudaMemcpyAsync(
    this->getDeviceToHostArrayMap().at(dataMovementAddress),
    devicePtr,
    dataMovementSize,
    storageConfig.offloadMemcpyKind,
    stream
  ));
  checkCudaErrors(cudaFreeAsync(devicePtr, stream));
  this->removeCurrentMapping(dataMovementAddress);
}

// Memory storage operations
void MemoryManager::moveAllManagedMemoryToStorage() {
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

void MemoryManager::moveRemainedManagedMemoryToStorage() {
  auto currentAddressMap = this->getEditableCurrentAddressMap();
  for (auto &[oldAddr, newAddr] : currentAddressMap) {
    checkCudaErrors(cudaMemcpy(
      this->getDeviceToHostArrayMap().at(oldAddr),
      newAddr,
      this->getSize(oldAddr),
      storageConfig.offloadMemcpyKind
    ));
    checkCudaErrors(cudaFree(newAddr));
  }
}

void MemoryManager::moveRemainedManagedMemoryToStorage(cudaStream_t stream) {
  auto currentAddressMap = this->getEditableCurrentAddressMap();
  for (auto &[oldAddr, newAddr] : currentAddressMap) {
    checkCudaErrors(cudaMemcpyAsync(
      this->getDeviceToHostArrayMap().at(oldAddr),
      newAddr,
      this->getSize(oldAddr),
      storageConfig.offloadMemcpyKind,
      stream
    ));
    checkCudaErrors(cudaFreeAsync(newAddr, stream));
    this->removeCurrentMapping(oldAddr);
  }
}

} // namespace memopt