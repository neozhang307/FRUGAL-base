#include "optimizationSerializer.hpp"

#include <fstream>
#include <vector>
#include <ctime>
#include <chrono>
#include <algorithm>
#include "../include/json.hpp"

#include "../memory/memoryManager.hpp"
#include "../utilities/logger.hpp"

namespace memopt {

// ========== Helper Structure for Serialization ==========

/**
 * @brief Helper structure for serializing OptimizationOutput nodes to JSON
 *
 * This structure flattens the node information for easier JSON serialization
 */
struct SerializableOptimizationOutputNode {
  int nodeId;
  std::vector<int> edges;
  OptimizationOutput::NodeType nodeType;
  int taskId;
  OptimizationOutput::DataMovement::Direction direction;
  int arrayId;

  // Define JSON serialization using nlohmann json macro
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(
    SerializableOptimizationOutputNode,
    nodeId,
    nodeType,
    edges,
    taskId,
    direction,
    arrayId
  );
};

// ========== OptimizationInput Serialization ==========

/**
 * @brief Save OptimizationInput to JSON file
 *
 * Converts the profiling data to a JSON format suitable for offline optimization.
 * Pointer-based dependencies are converted to ArrayId-based dependencies using MemoryManager.
 */
void saveOptimizationInput(const OptimizationInput& input, const std::string& path) {
    LOG_TRACE();

    nlohmann::json j;
    auto& memManager = MemoryManager::getInstance();

    // Serialize task groups (nodes)
    j["taskGroups"] = nlohmann::json::array();
    for (size_t i = 0; i < input.nodes.size(); i++) {
        const auto& taskGroup = input.nodes[i];
        nlohmann::json tg;

        tg["id"] = i;
        tg["taskIds"] = std::vector<TaskId>(taskGroup.nodes.begin(), taskGroup.nodes.end());
        tg["runningTime"] = taskGroup.runningTime;

        // Convert internal task dependencies
        tg["internalEdges"] = nlohmann::json::object();
        for (const auto& [fromTask, toTasks] : taskGroup.edges) {
            tg["internalEdges"][std::to_string(fromTask)] = toTasks;
        }

        // Convert void* dependencies to ArrayIds
        std::vector<ArrayId> inputArrayIds;
        for (void* ptr : taskGroup.dataDependency.inputs) {
            ArrayId arrayId = memManager.getArrayId(ptr);
            if (arrayId >= 0) {
                inputArrayIds.push_back(arrayId);
            }
        }
        tg["inputArrays"] = inputArrayIds;

        std::vector<ArrayId> outputArrayIds;
        for (void* ptr : taskGroup.dataDependency.outputs) {
            ArrayId arrayId = memManager.getArrayId(ptr);
            if (arrayId >= 0) {
                outputArrayIds.push_back(arrayId);
            }
        }
        tg["outputArrays"] = outputArrayIds;

        j["taskGroups"].push_back(tg);
    }

    // Serialize task group dependencies (edges between task groups)
    j["taskGroupEdges"] = nlohmann::json::object();
    for (const auto& [from, toList] : input.edges) {
        j["taskGroupEdges"][std::to_string(from)] = toList;
    }

    // Serialize array information
    j["arrays"] = nlohmann::json::array();

    // If arraySizes is populated (new path), use it
    if (!input.arraySizes.empty()) {
        // Build array list from arraySizes map
        std::map<ArrayId, size_t> arrayIdToSize;
        for (const auto& [ptr, size] : input.arraySizes) {
            ArrayId arrayId = memManager.getArrayId(ptr);
            if (arrayId >= 0) {
                arrayIdToSize[arrayId] = size;
            }
        }

        for (const auto& [arrayId, size] : arrayIdToSize) {
            nlohmann::json arr;
            arr["id"] = arrayId;
            arr["size"] = size;
            j["arrays"].push_back(arr);
        }
    } else {
        // Fallback to old path using MemoryManager (for backward compatibility)
        const auto& addressToIndexMap = memManager.getAddressToIndexMap();
        for (const auto& [ptr, arrayId] : addressToIndexMap) {
            nlohmann::json arr;
            arr["id"] = arrayId;
            arr["size"] = memManager.getSize(ptr);
            j["arrays"].push_back(arr);
        }
    }

    // Serialize application inputs/outputs
    std::vector<ArrayId> appInputIds;
    for (void* ptr : memManager.getApplicationInputs()) {
        ArrayId arrayId = memManager.getArrayId(ptr);
        if (arrayId >= 0) {
            appInputIds.push_back(arrayId);
        }
    }
    j["applicationInputArrays"] = appInputIds;

    std::vector<ArrayId> appOutputIds;
    for (void* ptr : memManager.getApplicationOutputs()) {
        ArrayId arrayId = memManager.getArrayId(ptr);
        if (arrayId >= 0) {
            appOutputIds.push_back(arrayId);
        }
    }
    j["applicationOutputArrays"] = appOutputIds;

    // Metadata
    j["metadata"]["originalTotalRunningTime"] = input.originalTotalRunningTime;
    j["metadata"]["forceAllArraysToResideOnHostInitiallyAndFinally"] =
        input.forceAllArraysToResideOnHostInitiallyAndFinally;
    j["metadata"]["stageIndex"] = input.stageIndex;

    // Save timestamp
    j["metadata"]["timestamp"] = std::time(nullptr);

    // Write to file
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    file << j.dump(2);
    file.close();

    LOG_TRACE_WITH_INFO("Saved OptimizationInput to %s", path.c_str());
    printf("Profiling data saved to: %s\n", path.c_str());
    printf("  Task groups: %zu\n", input.nodes.size());
    // Count arrays based on what we serialized
    size_t numArrays = j["arrays"].size();
    printf("  Arrays: %zu\n", numArrays);
}

/**
 * @brief Load OptimizationInput from JSON file
 *
 * Deserializes profiling data from JSON. Note that pointer-based dependencies
 * are not reconstructed (they're not needed for offline optimization).
 */
OptimizationInput loadOptimizationInput(const std::string& path) {
    LOG_TRACE();

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    auto j = nlohmann::json::parse(file);
    file.close();

    OptimizationInput input;

    // Deserialize task groups
    for (const auto& tg : j["taskGroups"]) {
        OptimizationInput::TaskGroup taskGroup;

        auto taskIds = tg["taskIds"].get<std::vector<TaskId>>();
        taskGroup.nodes = std::set<TaskId>(taskIds.begin(), taskIds.end());
        taskGroup.runningTime = tg["runningTime"];

        // Deserialize internal edges
        if (tg.contains("internalEdges")) {
            for (const auto& [fromStr, toTasks] : tg["internalEdges"].items()) {
                TaskId from = std::stoi(fromStr);
                taskGroup.edges[from] = toTasks.get<std::vector<TaskId>>();
            }
        }

        // Populate dataDependency with fake pointers based on array IDs
        // This allows the existing optimization code to work unchanged
        if (tg.contains("inputArrays")) {
            for (ArrayId id : tg["inputArrays"].get<std::vector<ArrayId>>()) {
                void* fakePtr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1000000 + id * 0x1000));
                taskGroup.dataDependency.inputs.insert(fakePtr);
            }
        }
        if (tg.contains("outputArrays")) {
            for (ArrayId id : tg["outputArrays"].get<std::vector<ArrayId>>()) {
                void* fakePtr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1000000 + id * 0x1000));
                taskGroup.dataDependency.outputs.insert(fakePtr);
            }
        }

        input.nodes.push_back(taskGroup);
    }

    // Deserialize task group edges
    if (j.contains("taskGroupEdges")) {
        for (const auto& [fromStr, toList] : j["taskGroupEdges"].items()) {
            TaskGroupId from = std::stoi(fromStr);
            input.edges[from] = toList.get<std::vector<TaskGroupId>>();
        }
    }

    // Metadata
    input.originalTotalRunningTime = j["metadata"]["originalTotalRunningTime"];
    input.forceAllArraysToResideOnHostInitiallyAndFinally =
        j["metadata"]["forceAllArraysToResideOnHostInitiallyAndFinally"];
    input.stageIndex = j["metadata"]["stageIndex"];

    // CRITICAL FIX: Register arrays with MemoryManager for standalone mode
    // This allows the optimization to work without actual memory pointers
    // Also populate arraySizes map in OptimizationInput
    if (j.contains("arrays")) {
        auto& memManager = MemoryManager::getInstance();
        for (const auto& arr : j["arrays"]) {
            ArrayId id = arr["id"];
            size_t size = arr["size"];
            // Create a fake pointer for this array ID
            // The actual address doesn't matter since we won't dereference it
            void* fakePtr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1000000 + id * 0x1000));
            memManager.registerManagedMemoryAddress(fakePtr, size);

            // NEW: Populate arraySizes map for offline optimization
            input.arraySizes[fakePtr] = size;
        }
    }

    // Register application inputs/outputs with MemoryManager
    if (j.contains("applicationInputArrays")) {
        auto& memManager = MemoryManager::getInstance();
        for (ArrayId id : j["applicationInputArrays"].get<std::vector<ArrayId>>()) {
            void* fakePtr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1000000 + id * 0x1000));
            memManager.registerApplicationInput(fakePtr);
        }
    }
    if (j.contains("applicationOutputArrays")) {
        auto& memManager = MemoryManager::getInstance();
        for (ArrayId id : j["applicationOutputArrays"].get<std::vector<ArrayId>>()) {
            void* fakePtr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1000000 + id * 0x1000));
            memManager.registerApplicationOutput(fakePtr);
        }
    }

    LOG_TRACE_WITH_INFO("Loaded OptimizationInput from %s", path.c_str());
    printf("Profiling data loaded from: %s\n", path.c_str());
    printf("  Task groups: %zu\n", input.nodes.size());
    printf("  Arrays: %zu\n", j["arrays"].size());

    return input;
}

// ========== OptimizationOutput Serialization ==========

/**
 * @brief Save OptimizationOutput to JSON file
 *
 * Serializes the optimized execution plan including task nodes, data movement
 * operations, and their dependencies. The JSON format matches the internal
 * format used by optimizer.cu for consistency.
 */
void saveOptimizationOutput(const OptimizationOutput& output, const std::string& path) {
    LOG_TRACE_WITH_INFO("Saving optimization plan to %s", path.c_str());

    // Convert nodes to serializable format
    std::vector<SerializableOptimizationOutputNode> serializableNodes;
    for (auto i : output.nodes) {
        SerializableOptimizationOutputNode node;
        node.nodeId = i;
        // Initialize all fields to avoid undefined behavior with uninitialized memory
        node.taskId = -1;
        node.arrayId = -1;
        node.direction = OptimizationOutput::DataMovement::Direction::hostToDevice;

        // Get edges if they exist
        auto edgeIt = output.edges.find(i);
        if (edgeIt != output.edges.end()) {
            node.edges = edgeIt->second;
        }

        // Get node type
        auto typeIt = output.nodeIdToNodeTypeMap.find(i);
        if (typeIt != output.nodeIdToNodeTypeMap.end()) {
            node.nodeType = typeIt->second;

            // Get task-specific data
            if (node.nodeType == OptimizationOutput::NodeType::task) {
                auto taskIt = output.nodeIdToTaskIdMap.find(i);
                if (taskIt != output.nodeIdToTaskIdMap.end()) {
                    node.taskId = taskIt->second;
                }
            }
            // Get data movement-specific data
            else if (node.nodeType == OptimizationOutput::NodeType::dataMovement) {
                auto moveIt = output.nodeIdToDataMovementMap.find(i);
                if (moveIt != output.nodeIdToDataMovementMap.end()) {
                    node.direction = moveIt->second.direction;
                    node.arrayId = moveIt->second.arrayId;
                }
            }
        } else {
            node.nodeType = OptimizationOutput::NodeType::empty;
        }

        serializableNodes.push_back(node);
    }

    // Create JSON document
    nlohmann::json j;
    j["nodes"] = serializableNodes;
    j["arraysInitiallyAllocatedOnDevice"] = output.arraysInitiallyAllocatedOnDevice;

    // Add memory usage information if available
    j["originalMemoryUsage"] = output.originalMemoryUsage;
    j["anticipatedPeakMemoryUsage"] = output.anticipatedPeakMemoryUsage;
    j["optimal"] = output.optimal;

    // Write to file
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    file << j.dump(2) << std::endl;
    file.close();

    LOG_TRACE_WITH_INFO("Saved OptimizationOutput to %s", path.c_str());
    printf("Optimization plan saved to: %s\n", path.c_str());
    printf("  Nodes: %zu\n", output.nodes.size());
    printf("  Initial device arrays: %zu\n", output.arraysInitiallyAllocatedOnDevice.size());
    printf("  Original memory: %.2f MiB\n", output.originalMemoryUsage);
    printf("  Optimized memory: %.2f MiB\n", output.anticipatedPeakMemoryUsage);
}

/**
 * @brief Load OptimizationOutput from JSON file
 *
 * Deserializes an optimized execution plan from JSON. This allows loading
 * pre-computed optimization plans for direct execution without re-optimization.
 */
OptimizationOutput loadOptimizationOutput(const std::string& path) {
    LOG_TRACE_WITH_INFO("Loading optimization plan from %s", path.c_str());

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    auto j = nlohmann::json::parse(file);
    file.close();

    // Deserialize nodes
    auto serializableNodes = j.at("nodes").get<std::vector<SerializableOptimizationOutputNode>>();

    OptimizationOutput output;

    // Initialize memory fields to zero to avoid garbage values
    output.optimal = false;
    output.originalMemoryUsage = 0.0;
    output.anticipatedPeakMemoryUsage = 0.0;

    // Reconstruct the graph structure
    for (const auto& node : serializableNodes) {
        output.nodes.push_back(node.nodeId);
        output.edges[node.nodeId] = node.edges;
        output.nodeIdToNodeTypeMap[node.nodeId] = node.nodeType;

        // Add task-specific data
        if (node.nodeType == OptimizationOutput::NodeType::task) {
            output.nodeIdToTaskIdMap[node.nodeId] = node.taskId;
        }
        // Add data movement-specific data
        else if (node.nodeType == OptimizationOutput::NodeType::dataMovement) {
            OptimizationOutput::DataMovement dataMovement;
            dataMovement.direction = node.direction;
            dataMovement.arrayId = node.arrayId;
            output.nodeIdToDataMovementMap[node.nodeId] = dataMovement;
        }
    }

    // Load arrays initially on device
    output.arraysInitiallyAllocatedOnDevice =
        j.at("arraysInitiallyAllocatedOnDevice").get<std::vector<ArrayId>>();

    // Load memory usage information if available
    if (j.contains("originalMemoryUsage")) {
        output.originalMemoryUsage = j["originalMemoryUsage"];
    }
    if (j.contains("anticipatedPeakMemoryUsage")) {
        output.anticipatedPeakMemoryUsage = j["anticipatedPeakMemoryUsage"];
    }
    if (j.contains("optimal")) {
        output.optimal = j["optimal"];
    }

    LOG_TRACE_WITH_INFO("Loaded OptimizationOutput from %s", path.c_str());
    printf("Optimization plan loaded from: %s\n", path.c_str());
    printf("  Nodes: %zu\n", output.nodes.size());
    printf("  Initial device arrays: %zu\n", output.arraysInitiallyAllocatedOnDevice.size());
    if (j.contains("originalMemoryUsage")) {
        printf("  Original memory: %.2f MiB\n", output.originalMemoryUsage);
    }
    if (j.contains("anticipatedPeakMemoryUsage")) {
        printf("  Optimized memory: %.2f MiB\n", output.anticipatedPeakMemoryUsage);
    }

    return output;
}

// ========== FirstStepSolver::Output Serialization Implementation ==========

void saveFirstStepOutput(const FirstStepSolver::Output& output, const std::string& path) {
    nlohmann::json j;
    j["taskGroupExecutionOrder"] = output.taskGroupExecutionOrder;

    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    file << j.dump(2);
    file.close();

    LOG_TRACE_WITH_INFO("Saved FirstStepOutput to %s", path.c_str());
    printf("Saved FirstStepOutput to %s (tasks: %zu)\n", path.c_str(),
           output.taskGroupExecutionOrder.size());
}

FirstStepSolver::Output loadFirstStepOutput(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    nlohmann::json j;
    file >> j;
    file.close();

    FirstStepSolver::Output output;
    output.taskGroupExecutionOrder = j["taskGroupExecutionOrder"].get<std::vector<TaskGroupId>>();

    LOG_TRACE_WITH_INFO("Loaded FirstStepOutput from %s", path.c_str());
    printf("Loaded FirstStepOutput from %s (tasks: %zu)\n", path.c_str(),
           output.taskGroupExecutionOrder.size());

    return output;
}

// ========== FirstStepSolver::TopKOutput Serialization ==========

void saveTopKSolutions(const FirstStepSolver::TopKOutput& topK, const std::string& path) {
    nlohmann::json j;

    // Save metadata
    j["numSolutions"] = topK.solutions.size();
    j["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();

    // Save all solutions
    j["solutions"] = nlohmann::json::array();
    for (size_t i = 0; i < topK.solutions.size(); i++) {
        nlohmann::json sol;
        sol["rank"] = i + 1;
        sol["taskGroupExecutionOrder"] = topK.solutions[i].taskGroupExecutionOrder;
        sol["dataReuseScore"] = topK.solutions[i].dataReuseScore;
        j["solutions"].push_back(sol);
    }

    // Write to file
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    file << j.dump(2);
    file.close();

    LOG_TRACE_WITH_INFO("Saved TopKSolutions to %s", path.c_str());
    printf("Saved %zu Top-K solutions to %s\n", topK.solutions.size(), path.c_str());

    // Print summary of scores
    printf("  Solution scores: ");
    for (size_t i = 0; i < std::min(size_t(5), topK.solutions.size()); i++) {
        printf("%zu ", topK.solutions[i].dataReuseScore);
    }
    if (topK.solutions.size() > 5) {
        printf("... (showing first 5 of %zu)", topK.solutions.size());
    }
    printf("\n");
}

FirstStepSolver::TopKOutput loadTopKSolutions(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    nlohmann::json j;
    file >> j;
    file.close();

    FirstStepSolver::TopKOutput topK;

    // Load all solutions
    for (const auto& solJson : j["solutions"]) {
        FirstStepSolver::Output solution;
        solution.taskGroupExecutionOrder = solJson["taskGroupExecutionOrder"].get<std::vector<TaskGroupId>>();
        solution.dataReuseScore = solJson["dataReuseScore"];
        topK.solutions.push_back(solution);
    }

    LOG_TRACE_WITH_INFO("Loaded TopKSolutions from %s", path.c_str());
    printf("Loaded %zu Top-K solutions from %s\n", topK.solutions.size(), path.c_str());

    // Print summary of scores
    printf("  Solution scores: ");
    for (size_t i = 0; i < std::min(size_t(5), topK.solutions.size()); i++) {
        printf("%zu ", topK.solutions[i].dataReuseScore);
    }
    if (topK.solutions.size() > 5) {
        printf("... (showing first 5 of %zu)", topK.solutions.size());
    }
    printf("\n");

    return topK;
}

}  // namespace memopt