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

// const std::map<void*, size_t>& MemoryManager::getAddressToSizeMap() const { 
//   return managedMemoryAddressToSizeMap; 
// }

const std::set<void*>& MemoryManager::getApplicationInputs() const { 
  return applicationInputs; 
}

const std::set<void*>& MemoryManager::getApplicationOutputs() const { 
  return applicationOutputs; 
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

// std::map<void*, size_t>& MemoryManager::getEditableAddressToSizeMap() { 
//   return managedMemoryAddressToSizeMap; 
// }

std::set<void*>& MemoryManager::getEditableApplicationInputs() { 
  return applicationInputs; 
}

std::set<void*>& MemoryManager::getEditableApplicationOutputs() { 
  return applicationOutputs; 
}


std::map<void*, void*>& MemoryManager::getEditableCurrentAddressMap() { 
  return managedMemoryAddressToAssignedMap; 
}

std::map<void*, void*>& MemoryManager::getEditableDeviceToHostArrayMap() { 
  return managedDeviceArrayToHostArrayMap; 
}

// Memory information methods are now in memoryManager_v2.cu

void MemoryManager::prefetchAllDataToDeviceAsync(
    const std::vector<ArrayId>& arrayIds,
    cudaMemcpyKind memcpyKind,
    cudaStream_t stream) {
  for (auto arrayId : arrayIds) {
    void* originalPtr = managedMemoryAddresses[arrayId];
    void* storagePtr = memoryArrayInfos[arrayId].storageAddress;// managedDeviceArrayToHostArrayMap.at(originalPtr);
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

// Memory allocation and data transfer methods moved to memoryManager_v2.cu

void MemoryManager::offloadAllManagedMemoryToStorage(std::map<void*, void*>& storageMap) {
  // Ensure the storage map starts empty
  storageMap.clear();
  clearStorage();
  // Move each managed memory address to storage
  for (auto ptr : managedMemoryAddresses) {
     // would automatically update MemoryArrayInfo
    offloadToStorage(ptr, storageConfig.storageDeviceId, storageConfig.useNvlink, storageMap);
  }
  
  // Also update internal map
  managedDeviceArrayToHostArrayMap = storageMap;
  
  // Switch back to main GPU
  checkCudaErrors(cudaSetDevice(storageConfig.mainDeviceId));
  checkCudaErrors(cudaDeviceSynchronize());
}

void MemoryManager::offloadAllManagedMemoryToStorage(
    int mainDeviceId, 
    int storageDeviceId, 
    bool useNvlink,
    std::map<void*, void*>& storageMap) {
  // Configure storage with the provided parameters
  configureStorage(mainDeviceId, storageDeviceId, useNvlink);
  
  // Call the simpler version that uses the configured parameters and external map
  offloadAllManagedMemoryToStorage(storageMap);
}

void MemoryManager::offloadAllManagedMemoryToStorage(
    int mainDeviceId, 
    int storageDeviceId, 
    bool useNvlink) {
  // Configure storage with the provided parameters
  configureStorage(mainDeviceId, storageDeviceId, useNvlink);
  
  // Call the simpler version that uses the configured parameters and internal map
  offloadAllManagedMemoryToStorage();
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

// Array ID utility methods moved to memoryManager_v2.cu

// Implementation of standalone function
void updateManagedMemoryAddress(const std::map<void *, void *> oldAddressToNewAddressMap) {
  auto& memManager = MemoryManager::getInstance();
  
  // Save the old size mapping for reference
  // auto oldManagedMemoryAddressToSizeMap = memManager.getAddressToSizeMap();

  // Clear existing mappings to rebuild them
  // memManager.getEditableAddressToSizeMap().clear();
  memManager.getEditableAddressToIndexMap().clear();

  // Update all addresses and rebuild mappings
  for (int i = 0; i < memManager.getManagedAddresses().size(); i++) {
    // Ensure every old address has a corresponding new address
    assert(oldAddressToNewAddressMap.count(memManager.getManagedAddresses()[i]) == 1);

    const auto newAddr = oldAddressToNewAddressMap.at(memManager.getManagedAddresses()[i]);
    const auto oldAddr = memManager.getManagedAddresses()[i];

    // Update address in the main list
    memManager.getEditableManagedAddresses()[i] = newAddr;
    
    // Transfer the size information to the new address (legacy map for backward compatibility)
    // memManager.getEditableAddressToSizeMap()[newAddr] = oldManagedMemoryAddressToSizeMap.at(oldAddr);
    
    // Maintain the same array ID
    memManager.getEditableAddressToIndexMap()[newAddr] = i;
    
    // Update the MemoryArrayInfo structure
    if (i < memManager.memoryArrayInfos.size()) {
      memManager.memoryArrayInfos[i].managedMemoryAddress = newAddr;
    }

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