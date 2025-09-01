#pragma once

#include <cuda_runtime.h>
#include <algorithm>  // for std::sort, std::unique, std::distance
#include <vector>
#include <map>
#include <memory>
#include <set>

namespace memopt {

/**
 * @brief Abstract base class for CUDA graph construction
 * 
 * Provides common functionality for creating and managing CUDA graphs
 * with dependency tracking.
 */
class CudaGraphConstructor {
public:
    /**
     * @brief Constructor
     * 
     * @param stream CUDA stream used for capturing operations
     * @param graph CUDA graph where operations will be recorded
     */
    CudaGraphConstructor(cudaStream_t stream, cudaGraph_t graph) 
        : stream(stream), graph(graph) {}
    
    /**
     * @brief Virtual destructor
     */
    virtual ~CudaGraphConstructor() = default;
    
    /**
     * @brief End capturing operation and update the dependency tracking
     * 
     * @return Vector of new leaf nodes added during this capture
     */
    virtual std::vector<cudaGraphNode_t> endCaptureOperation() = 0;
    
    /**
     * @brief Get the current CUDA graph
     * 
     * @return The CUDA graph that has been built
     */
    cudaGraph_t getGraph() const {
        return graph;
    }
    
protected:
    cudaStream_t stream;         ///< CUDA stream for capturing operations
    cudaGraph_t graph;           ///< CUDA graph being constructed
    std::map<cudaGraphNode_t, bool> visited;  ///< Tracks which nodes have been processed
    
    /**
     * @brief Find leaf nodes (nodes with no outgoing edges)
     * 
     * @return Vector of nodes with no outgoing edges
     */
    std::vector<cudaGraphNode_t> getNodesWithZeroOutDegree() const {
        // Get all edges in the graph
        size_t numEdges;
        cudaGraphGetEdges(this->graph, nullptr, nullptr, &numEdges);
        auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
        auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);
        cudaGraphGetEdges(this->graph, from.get(), to.get(), &numEdges);

        // Track which nodes have outgoing edges
        std::map<cudaGraphNode_t, bool> hasOutGoingEdge;
        std::set<cudaGraphNode_t> noOutGoingEdgeNodes;
        
        // Get all nodes in the graph
        size_t numNodes;
        cudaGraphGetNodes(this->graph, nullptr, &numNodes);
        auto nodes = std::make_unique<cudaGraphNode_t[]>(numNodes);
        cudaGraphGetNodes(this->graph, nodes.get(), &numNodes);
        
        // Add all nodes to the set initially
        for (size_t i = 0; i < numNodes; i++) {
            noOutGoingEdgeNodes.insert(nodes[i]);
        }
        
        // Remove nodes that have outgoing edges
        for (size_t i = 0; i < numEdges; i++) {
            hasOutGoingEdge[from[i]] = true;
            noOutGoingEdgeNodes.erase(from[i]);
        }
        
        // Convert set to vector for return
        return std::vector<cudaGraphNode_t>(noOutGoingEdgeNodes.begin(), noOutGoingEdgeNodes.end());
    }
    
    /**
     * @brief Find new leaf nodes added since the last capture
     * 
     * @return Vector of new leaf nodes
     */
    std::vector<cudaGraphNode_t> getNewLeafNodesAddedByLastCapture() {
        // Get all nodes in the graph
        size_t numNodes;
        cudaGraphGetNodes(this->graph, nullptr, &numNodes);
        auto nodes = std::make_unique<cudaGraphNode_t[]>(numNodes);
        cudaGraphGetNodes(this->graph, nodes.get(), &numNodes);

        // Get all edges in the graph
        size_t numEdges;
        cudaGraphGetEdges(this->graph, nullptr, nullptr, &numEdges);
        auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
        auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);
        cudaGraphGetEdges(this->graph, from.get(), to.get(), &numEdges);

        // Track which nodes have outgoing edges
        std::map<cudaGraphNode_t, bool> hasOutGoingEdge;
        for (size_t i = 0; i < numEdges; i++) {
            hasOutGoingEdge[from[i]] = true;
        }

        // Find new leaf nodes (not visited before and have no outgoing edges)
        std::vector<cudaGraphNode_t> newLeafNodes;
        for (size_t i = 0; i < numNodes; i++) {
            auto& node = nodes[i];
            if (!visited[node]) {
                visited[node] = true;
                if (!hasOutGoingEdge[node]) {
                    newLeafNodes.push_back(node);
                }
            }
        }

        return newLeafNodes;
    }
};

/**
 * @brief CUDA graph constructor implementation using memory pointers for dependency tracking
 * 
 * Tracks dependencies between operations based on memory addresses accessed.
 */
class PointerDependencyCudaGraphConstructor : public CudaGraphConstructor {
public:
    /**
     * @brief Constructor
     * 
     * @param stream CUDA stream used for capturing operations
     * @param graph CUDA graph where operations will be recorded
     */
    PointerDependencyCudaGraphConstructor(cudaStream_t stream, cudaGraph_t graph) 
        : CudaGraphConstructor(stream, graph) {}
    
    /**
     * @brief Begin capturing operation with input and output memory dependencies
     * 
     * @param inputs Vector of input memory pointers (read from)
     * @param outputs Vector of output memory pointers (written to)
     */
    void beginCaptureOperation(const std::vector<void*>& inputs, const std::vector<void*>& outputs) {
        // Calculate dependencies based on memory addresses
        auto dependencies = getDependencies(inputs, outputs);
        
        // Store inputs and outputs for later tracking
        currentInputs = inputs;
        currentOutputs = outputs;
        lastDependencies = dependencies;
        
        // Begin stream capture with dependencies
        if (dependencies.empty()) {
            cudaStreamBeginCaptureToGraph(this->stream, this->graph, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal);
        } else {
            cudaStreamBeginCaptureToGraph(this->stream, this->graph, dependencies.data(), nullptr, dependencies.size(), cudaStreamCaptureModeGlobal);
        }
    }
    
    /**
     * @brief End capturing operation and update the dependency tracking
     * 
     * @return Vector of new leaf nodes added during this capture
     */
    std::vector<cudaGraphNode_t> endCaptureOperation() override {
        // End the capture
        cudaStreamEndCapture(this->stream, &this->graph);
        
        // Find the newly added leaf nodes
        auto leafNodes = getNewLeafNodesAddedByLastCapture();
        
        if (!leafNodes.empty()) {
            // Update readers: ALL inputs are readers
            for (auto& input : currentInputs) {
                for (auto& node : leafNodes) {
                    lastReadByMap[input].push_back(node);
                }
            }
            
            // Update writers: outputs update last writer
            for (auto& output : currentOutputs) {
                lastModifiedByMap[output] = leafNodes[0];
            }
        }
        
        // Clear current inputs and outputs
        currentInputs.clear();
        currentOutputs.clear();
        
        return leafNodes;
    }
    
private:
    std::map<void*, cudaGraphNode_t> lastModifiedByMap;  ///< Maps memory addresses to last modifying node
    std::map<void*, std::vector<cudaGraphNode_t>> lastReadByMap;  ///< Maps memory addresses to last reading nodes
    std::vector<void*> currentInputs;                    ///< Current inputs being read
    std::vector<void*> currentOutputs;                   ///< Current outputs being modified
    std::vector<cudaGraphNode_t> lastDependencies;       ///< Dependencies of the current operation
    
    /**
     * @brief Get dependencies for given inputs and outputs
     * 
     * @param inputs Memory addresses that will be read
     * @param outputs Memory addresses that will be written
     * @return Vector of dependent nodes
     */
    std::vector<cudaGraphNode_t> getDependencies(
        const std::vector<void*>& inputs, 
        const std::vector<void*>& outputs) 
    {
        std::vector<cudaGraphNode_t> dependencies;
        
        // Add dependencies for inputs (read-after-write)
        for (auto& input : inputs) {
            auto it = lastModifiedByMap.find(input);
            if (it != lastModifiedByMap.end()) {
                dependencies.push_back(it->second);
            }
        }
        
        // Add dependencies for outputs (write-after-write and write-after-read)
        for (auto& output : outputs) {
            // Write-after-write dependency
            auto it = lastModifiedByMap.find(output);
            if (it != lastModifiedByMap.end()) {
                dependencies.push_back(it->second);
            }
            
            // Write-after-read dependencies (WAR)
            auto readers_it = lastReadByMap.find(output);
            if (readers_it != lastReadByMap.end()) {
                // Add all readers as dependencies (must wait for all reads to complete)
                for (auto& reader : readers_it->second) {
                    dependencies.push_back(reader);
                }
                // Clear readers - they become irrelevant after this write
                readers_it->second.clear();
            }
        }
        
        // Remove duplicate dependencies
        std::sort(dependencies.begin(), dependencies.end());
        auto it = std::unique(dependencies.begin(), dependencies.end());
        dependencies.resize(std::distance(dependencies.begin(), it));
        
        return dependencies;
    }
};

} // namespace memopt