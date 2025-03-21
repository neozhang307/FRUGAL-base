#include "memoryManager.hpp"

namespace memopt {

// Singleton instance access
MemoryManager& MemoryManager::getInstance() {
  static MemoryManager instance;
  return instance;
}

// Const accessor methods
const std::vector<void*>& MemoryManager::getManagedAddresses() const { 
  return managedMemoryAddresses; 
}

const std::map<void*, ArrayId>& MemoryManager::getAddressToIndexMap() const { 
  return managedMemoryAddressToIndexMap; 
}

const std::map<void*, size_t>& MemoryManager::getAddressToSizeMap() const { 
  return managedMemoryAddressToSizeMap; 
}

const std::set<void*>& MemoryManager::getApplicationInputs() const { 
  return applicationInputs; 
}

const std::set<void*>& MemoryManager::getApplicationOutputs() const { 
  return applicationOutputs; 
}

const std::map<void*, void*>& MemoryManager::getDeviceToStorageMap() const { 
  return deviceToStorageArrayMap; 
}

const std::map<void*, void*>& MemoryManager::getCurrentAddressMap() const { 
  return managedMemoryAddressToAssignedMap; 
}

const std::map<void*, void*>& MemoryManager::getDeviceToHostArrayMap() const { 
  return managedDeviceArrayToHostArrayMap; 
}

// Editable accessor methods
std::vector<void*>& MemoryManager::getEditableManagedAddresses() { 
  return managedMemoryAddresses; 
}

std::map<void*, ArrayId>& MemoryManager::getEditableAddressToIndexMap() { 
  return managedMemoryAddressToIndexMap; 
}

std::map<void*, size_t>& MemoryManager::getEditableAddressToSizeMap() { 
  return managedMemoryAddressToSizeMap; 
}

std::set<void*>& MemoryManager::getEditableApplicationInputs() { 
  return applicationInputs; 
}

std::set<void*>& MemoryManager::getEditableApplicationOutputs() { 
  return applicationOutputs; 
}

std::map<void*, void*>& MemoryManager::getEditableDeviceToStorageMap() { 
  return deviceToStorageArrayMap; 
}

std::map<void*, void*>& MemoryManager::getEditableCurrentAddressMap() { 
  return managedMemoryAddressToAssignedMap; 
}

std::map<void*, void*>& MemoryManager::getEditableDeviceToHostArrayMap() { 
  return managedDeviceArrayToHostArrayMap; 
}

// Memory information
size_t MemoryManager::getSize(void* addr) const {
  auto it = managedMemoryAddressToSizeMap.find(addr);
  if (it != managedMemoryAddressToSizeMap.end()) {
    return it->second;
  }
  return 0; // Return 0 for unmanaged memory
}

ArrayId MemoryManager::getArrayId(void* addr) const {
  auto it = managedMemoryAddressToIndexMap.find(addr);
  if (it != managedMemoryAddressToIndexMap.end()) {
    return it->second;
  }
  return -1; // Invalid ID for unmanaged memory
}

bool MemoryManager::isManaged(void* addr) const {
  return managedMemoryAddressToIndexMap.count(addr) > 0;
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

// Storage configuration
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

// Data prefetching and movement
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

void MemoryManager::prefetchAllDataToDevice(
    const std::vector<ArrayId>& arrayIds,
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
  
  // Free original device memory
  checkCudaErrors(cudaFree(ptr));
  
  return storagePtr;
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

void MemoryManager::moveAllManagedMemoryToStorage(std::map<void*, void*>& storageMap) {
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

void MemoryManager::moveAllManagedMemoryToStorage(
    int mainDeviceId, 
    int storageDeviceId, 
    bool useNvlink,
    std::map<void*, void*>& storageMap) {
  // Configure storage with the provided parameters
  configureStorage(mainDeviceId, storageDeviceId, useNvlink);
  
  // Call the simpler version that uses the configured parameters and external map
  moveAllManagedMemoryToStorage(storageMap);
}

void MemoryManager::moveAllManagedMemoryToStorage(
    int mainDeviceId, 
    int storageDeviceId, 
    bool useNvlink) {
  // Configure storage with the provided parameters
  configureStorage(mainDeviceId, storageDeviceId, useNvlink);
  
  // Call the simpler version that uses the configured parameters and internal map
  moveAllManagedMemoryToStorage();
}

// Asynchronous operations
void MemoryManager::offloadDataAsync(
    void* originalPtr, 
    std::map<void*, void*>& storageMap, 
    std::map<void*, void*>& currentMap,
    cudaStream_t stream) {
  void* devicePtr = currentMap[originalPtr];
  void* storagePtr = storageMap[originalPtr];
  size_t size = getSize(originalPtr);
  
  // Copy data from device to storage and free device memory
  checkCudaErrors(cudaMemcpyAsync(storagePtr, devicePtr, size, storageConfig.offloadMemcpyKind, stream));
  checkCudaErrors(cudaFreeAsync(devicePtr, stream));
  
  // Remove from current map
  currentMap.erase(originalPtr);
}

void MemoryManager::offloadDataAsync(
    void* originalPtr, 
    std::map<void*, void*>& storageMap, 
    std::map<void*, void*>& currentMap, 
    cudaMemcpyKind offloadMemcpyKind,
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

// Array ID utility methods
void* MemoryManager::getPointerByArrayId(int arrayId) const {
  if (arrayId >= 0 && arrayId < managedMemoryAddresses.size()) {
    return managedMemoryAddresses[arrayId];
  }
  return nullptr;
}

size_t MemoryManager::getSizeByArrayId(int arrayId) const {
  if (arrayId >= 0 && arrayId < managedMemoryAddresses.size()) {
    void* ptr = managedMemoryAddresses[arrayId];
    return getSize(ptr);
  }
  return 0;
}

// Implementation of standalone function
void updateManagedMemoryAddress(const std::map<void *, void *> oldAddressToNewAddressMap) {
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

} // namespace memopt