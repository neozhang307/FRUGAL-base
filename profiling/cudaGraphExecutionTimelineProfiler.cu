#include <cassert>
#include <cstdio>
#include <memory>

#include "../utilities/cudaUtilities.hpp"
#include "../utilities/cuptiUtilities.hpp"
#include "cudaGraphExecutionTimelineProfiler.hpp"

namespace memopt {

// Singleton instance pointer initialization
CudaGraphExecutionTimelineProfiler *CudaGraphExecutionTimelineProfiler::instance = nullptr;

/**
 * Returns the singleton instance of the profiler, creating it if necessary
 */
CudaGraphExecutionTimelineProfiler *CudaGraphExecutionTimelineProfiler::getInstance() {
  if (instance == nullptr) {
    instance = new CudaGraphExecutionTimelineProfiler();
  }
  return instance;
}

/**
 * Processes a CUPTI activity record and stores node execution time data
 * 
 * This function is the core of the timeline profiling system. It receives activity records 
 * from CUPTI (via the bufferCompleted callback) and extracts timing information from them.
 * Each record contains data about a specific CUDA operation that was executed, including 
 * timestamps for when it started and ended.
 * 
 * The function:
 * 1. Identifies the type of CUDA operation (kernel execution or memory set)
 * 2. Extracts the unique graph node ID that performed the operation
 * 3. Records the start and end timestamps for that operation
 * 4. Stores this timing data in the graphNodeIdToLifetimeMap for later analysis
 * 
 * @param record Pointer to a CUPTI activity record containing execution data
 */
void CudaGraphExecutionTimelineProfiler::consumeActivityRecord(CUpti_Activity *record) {
  // The record's 'kind' field tells us what type of CUDA operation this record represents
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      // This is a kernel execution record (CUDA computational work)
      // We need to cast to the specific kernel record type to access its fields
      auto kernelActivityRecord = reinterpret_cast<CUpti_ActivityKernel9 *>(record);
      
      // Extract the relevant data:
      // - graphNodeId: Identifies which node in the CUDA graph performed this kernel execution
      // - start: Timestamp (in ns) when the kernel started executing
      // - end: Timestamp (in ns) when the kernel finished executing
      this->graphNodeIdToLifetimeMap[static_cast<uint64_t>(kernelActivityRecord->graphNodeId)] = std::make_pair(
        static_cast<uint64_t>(kernelActivityRecord->start),
        static_cast<uint64_t>(kernelActivityRecord->end)
      );
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMSET: {
      // This is a memory set record (CUDA memory initialization)
      // We need to cast to the specific memset record type to access its fields
      auto memsetActivityRecord = reinterpret_cast<CUpti_ActivityMemset4 *>(record);
      
      // Extract the relevant data:
      // - graphNodeId: Identifies which node in the CUDA graph performed this memset operation
      // - start: Timestamp (in ns) when the memset started executing
      // - end: Timestamp (in ns) when the memset finished executing
      this->graphNodeIdToLifetimeMap[static_cast<uint64_t>(memsetActivityRecord->graphNodeId)] = std::make_pair(
        static_cast<uint64_t>(memsetActivityRecord->start),
        static_cast<uint64_t>(memsetActivityRecord->end)
      );
      break;
    }
    default: {
      // We encounter an activity type we're not handling
      // This could happen if CUPTI provides records for other operations like memory copies
      // that we're not explicitly tracking in this profiler
      printf("Warning: Unknown CUPTI activity (%d)\n", record->kind);
      break;
    }
  }
}

/**
 * CUPTI callback for buffer allocation requests
 * 
 * Called by CUPTI when it needs a buffer for storing activity records
 * Allocates memory for the buffer and returns its address and size
 */
void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  auto rawBuffer = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
  if (rawBuffer == nullptr) {
    printf("Error: Out of memory\n");
    exit(-1);
  }

  // Set buffer properties with proper alignment
  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
  *maxNumRecords = 0;  // Let CUPTI determine how many records fit
}

/**
 * CUPTI callback for buffer completion
 * 
 * Called by CUPTI when a buffer has been filled with activity records
 * Processes each record in the buffer and frees the buffer memory
 */
void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
  CUptiResult status;
  CUpti_Activity *record = nullptr;

  // Process all records in the buffer
  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      // Send record to profiler for processing
      CudaGraphExecutionTimelineProfiler::getInstance()->consumeActivityRecord(record);
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

  free(buffer);  // Free the buffer after processing all records
}

/**
 * Callback handler for graph node cloning events
 * 
 * Maintains a mapping between original graph nodes and their cloned versions
 * This mapping is essential because CUDA creates clones of nodes during execution
 */
void CudaGraphExecutionTimelineProfiler::graphNodeClonedCallback(CUpti_GraphData *graphData) {
  uint64_t clonedNodeId, originalNodeId;
  // Get IDs for both the original node and its clone
  CUPTI_CALL(cuptiGetGraphNodeId(graphData->node, &clonedNodeId));
  CUPTI_CALL(cuptiGetGraphNodeId(graphData->originalNode, &originalNodeId));
  // Store the mapping for later use
  this->originalNodeIdToClonedNodeIdMap[originalNodeId] = clonedNodeId;
}

/**
 * Global CUPTI callback handler - central event processing function for CUPTI events
 * 
 * This function is registered with CUPTI's callback API and gets called whenever
 * a CUPTI event of interest occurs. It routes different event types to the appropriate
 * specialized handlers based on the domain and callback ID.
 * 
 * Currently focused on tracking graph node cloning events, which are critical for
 * mapping between the original graph nodes and their runtime clones.
 * 
 * @param pUserData      User-provided data pointer (nullptr in our case)
 * @param domain         The callback domain (e.g., RESOURCE, RUNTIME, DRIVER)
 * @param callbackId     Specific event ID within the domain
 * @param pCallbackInfo  Data structure containing event-specific information
 */
void CUPTIAPI
callbackHandler(
  void *pUserData,
  CUpti_CallbackDomain domain,
  CUpti_CallbackId callbackId,
  const CUpti_CallbackData *pCallbackInfo
) {
  // Check for any pending CUPTI errors before processing this callback
  // This ensures we're not missing any error conditions
  CUPTI_CALL(cuptiGetLastError());

  // Process events based on their domain category
  switch (domain) {
    // RESOURCE domain handles hardware and runtime resource management
    case CUPTI_CB_DOMAIN_RESOURCE: {
      // Cast callback data to the appropriate type for RESOURCE domain
      CUpti_ResourceData *pResourceData = (CUpti_ResourceData *)(pCallbackInfo);
      
      // Process specific events within the RESOURCE domain
      switch (callbackId) {
        // Handle node cloning - occurs when CUDA runtime clones graph nodes during instantiation
        case CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED: {
          // For cloning events, the resource descriptor contains graph node information
          CUpti_GraphData *callbackData = (CUpti_GraphData *)(pResourceData->resourceDescriptor);
          
          // Forward to our specialized handler that will capture the node ID mapping
          // This mapping is critical because timing data is recorded for the cloned nodes,
          // but we want to associate it with the original graph nodes
          CudaGraphExecutionTimelineProfiler::getInstance()->graphNodeClonedCallback(callbackData);
          break;
        }
        default:
          // Ignore other resource events we're not interested in
          break;
      }
      break;
    }
    default:
      // Ignore other domains we're not interested in (RUNTIME, DRIVER, etc.)
      break;
  }
}

/**
 * Initialize the profiler for a specific CUDA graph
 * 
 * Sets up CUPTI activity and callback API to track graph execution.
 * This must be called before the graph is executed to properly capture all events.
 * 
 * @param graph The CUDA graph to be profiled
 */
void CudaGraphExecutionTimelineProfiler::initialize(cudaGraph_t graph) {
  // CUPTI Activity API setup - tracks timing of operations
  // Register callbacks for buffer management:
  // - bufferRequested: Called when CUPTI needs memory to store activity records
  // - bufferCompleted: Called when a buffer is filled with activity data and ready to process
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  
  // Enable tracking of kernel executions - records when each CUDA kernel starts and ends
  // This captures the exact timestamps of computational operations in the graph
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  
  // Enable tracking of memory set operations - records timing of memory initialization
  // Important because some application contains memset nodes for initializing memory
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));

  // CUPTI Callback API setup - tracks structural events in the graph's lifecycle
  // Subscribe to receive callback events with our callbackHandler function
  // The nullptr parameter means we're not passing any user data to the callback
  CUPTI_CALL(cuptiSubscribe(&this->subscriberHandle, (CUpti_CallbackFunc)(callbackHandler), nullptr));
  
  // Enable notifications specifically for graph node cloning events
  // This is critical because CUDA creates clones when executing graph nodes
  // Allows tracking the relationship between original graph nodes and runtime instances
  CUPTI_CALL(cuptiEnableCallback(1, this->subscriberHandle, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED));

  // Initialize internal state for this profiling session
  this->finalized = false;         // Mark profiler as active and not yet finalized
  this->graph = graph;             // Store reference to the graph being profiled
  this->graphNodeIdToLifetimeMap.clear();       // Clear any previous node timing data
  this->originalNodeIdToClonedNodeIdMap.clear(); // Clear node mapping from previous runs
}

/**
 * Finalize profiling session and clean up resources
 * 
 * Must be called after graph execution is complete
 * Flushes all pending activity records and disables profiling
 */
void CudaGraphExecutionTimelineProfiler::finalize() {
  CUPTI_CALL(cuptiGetLastError());  // Check for any pending errors

  // CUPTI Activity API cleanup
  CUPTI_CALL(cuptiActivityFlushAll(1));  // Flush any pending activity records
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));

  // CUPTI Callback API cleanup
  CUPTI_CALL(cuptiUnsubscribe(this->subscriberHandle));

  this->finalized = true;  // Mark profiler as finalized
}

/**
 * Retrieve the execution timeline for the profiled graph
 * 
 * Maps each graph node to its execution time period (start/end timestamps)
 * Must be called after finalize() has been called
 */
CudaGraphExecutionTimeline CudaGraphExecutionTimelineProfiler::getTimeline() {
  assert(this->finalized);  // Ensure profiler has been finalized

  // Get all nodes from the graph
  size_t numNodes;
  checkCudaErrors(cudaGraphGetNodes(this->graph, nullptr, &numNodes));
  auto nodes = std::make_unique<cudaGraphNode_t[]>(numNodes);
  checkCudaErrors(cudaGraphGetNodes(this->graph, nodes.get(), &numNodes));

  CudaGraphExecutionTimeline timeline;

  // For each node, look up its execution timestamps
  uint64_t originalNodeId;
  for (int i = 0; i < numNodes; i++) {
    // Get the original node ID
    CUPTI_CALL(cuptiGetGraphNodeId(nodes[i], &originalNodeId));
    
    // Map from original node ID -> cloned node ID -> execution timestamps
    // This two-step mapping is necessary because:
    // 1. We execute with cloned nodes, not original nodes
    // 2. Timing data is recorded for cloned nodes
    // 3. We need to map this back to original nodes for API consistency
    timeline[nodes[i]] = this->graphNodeIdToLifetimeMap[originalNodeIdToClonedNodeIdMap[originalNodeId]];
  }

  return timeline;
}

}  // namespace memopt
