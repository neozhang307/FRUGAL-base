#include "optimizationInputSerializer.hpp"

#include <fstream>
#include <vector>
#include "../include/json.hpp"

#include "../profiling/memoryManager.hpp"
#include "../utilities/logger.hpp"

namespace memopt {

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

    // Serialize array information from MemoryManager
    j["arrays"] = nlohmann::json::array();
    const auto& addressToIndexMap = memManager.getAddressToIndexMap();
    for (const auto& [ptr, arrayId] : addressToIndexMap) {
        nlohmann::json arr;
        arr["id"] = arrayId;
        arr["size"] = memManager.getSize(ptr);
        j["arrays"].push_back(arr);
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
    printf("  Arrays: %zu\n", addressToIndexMap.size());
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

        // Note: We don't populate dataDependency.inputs/outputs (void* pointers)
        // because they're not available in offline mode and not needed by solvers
        // The ArrayId-based information will be used directly when constructing
        // SecondStepSolver::Input

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

    LOG_TRACE_WITH_INFO("Loaded OptimizationInput from %s", path.c_str());
    printf("Profiling data loaded from: %s\n", path.c_str());
    printf("  Task groups: %zu\n", input.nodes.size());
    printf("  Arrays: %zu\n", j["arrays"].size());

    return input;
}

}  // namespace memopt
