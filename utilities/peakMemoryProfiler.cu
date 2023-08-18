#include <algorithm>
#include <cassert>
#include <cstdio>

#include "cudaUtilities.hpp"
#include "peakMemoryProfiler.hpp"

// CUPTI related constants and macros
constexpr size_t BUF_SIZE = 8 * 1024;
constexpr size_t ALIGN_SIZE = 8;

#define ALIGN_BUFFER(buffer, align) \
  (((uintptr_t)(buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) : (buffer))

#define CUPTI_CALL(call)                                                                                       \
  do {                                                                                                         \
    CUptiResult _status = call;                                                                                \
    if (_status != CUPTI_SUCCESS) {                                                                            \
      const char *errstr;                                                                                      \
      cuptiGetResultString(_status, &errstr);                                                                  \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__, #call, errstr); \
      exit(-1);                                                                                                \
    }                                                                                                          \
  } while (0)

PeakMemoryProfiler *PeakMemoryProfiler::instance = nullptr;

PeakMemoryProfiler *PeakMemoryProfiler::getInstance() {
  if (instance == nullptr) {
    instance = new PeakMemoryProfiler();
  }
  return instance;
}

uint64_t PeakMemoryProfiler::getPeakMemoryUsage() {
  assert(this->finalized);
  return this->currentPeakAllocatedMemory;
}

void PeakMemoryProfiler::consumeActivityRecord(CUpti_Activity *record) {
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_MEMORY2: {
      auto memoryActivityRecord = reinterpret_cast<CUpti_ActivityMemory2 *>(record);
      int sign = 0;
      if (memoryActivityRecord->memoryOperationType == CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION) {
        sign = 1;
      } else if (memoryActivityRecord->memoryOperationType == CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE) {
        sign = -1;
      }
      this->memoryActivityRecords.push_back(
        std::make_tuple(
          static_cast<uint64_t>(memoryActivityRecord->timestamp),
          static_cast<uint64_t>(memoryActivityRecord->bytes),
          sign
        )
      );
      break;
    }
    default:
      printf("Warning: Unknown CUPTI activity (%d)\n", record->kind);
      break;
  }
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  auto rawBuffer = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
  if (rawBuffer == nullptr) {
    printf("Error: Out of memory\n");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
  CUptiResult status;
  CUpti_Activity *record = nullptr;

  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      PeakMemoryProfiler::getInstance()->consumeActivityRecord(record);
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      CUPTI_CALL(status);
    }
  } while (1);

  // Report any records dropped from the queue
  size_t dropped;
  CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
  if (dropped != 0) {
    printf("Dropped %u activity records\n", (unsigned int)dropped);
  }

  free(buffer);
}

// Initialize CUDA driver and CUPTI
void PeakMemoryProfiler::initialize() {
  this->finalized = false;

  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2));
}

void PeakMemoryProfiler::finalize() {
  CUPTI_CALL(cuptiGetLastError());

  CUPTI_CALL(cuptiActivityFlushAll(1));

  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMORY2));

  std::sort(this->memoryActivityRecords.begin(), this->memoryActivityRecords.end());

  this->currentAllocatedMemory = 0;
  this->currentPeakAllocatedMemory = 0;

  for (auto &&record : this->memoryActivityRecords) {
    if (std::get<2>(record) == 1) {
      this->currentAllocatedMemory += std::get<1>(record);
    } else if (std::get<2>(record) == -1) {
      this->currentAllocatedMemory -= std::get<1>(record);
    }
    this->currentPeakAllocatedMemory = std::max(this->currentAllocatedMemory, this->currentPeakAllocatedMemory);
  }

  this->finalized = true;
}
