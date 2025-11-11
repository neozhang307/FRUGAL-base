#include "greedyScheduler.hpp"

#include <algorithm>
#include <numeric>
#include <fmt/core.h>

#include "../../utilities/configurationManager.hpp"
#include "../../utilities/logger.hpp"
#include "../../utilities/utilities.hpp"

namespace memopt {

GreedyScheduler::GreedyScheduler() : config_() {}

GreedyScheduler::GreedyScheduler(const Config& config) : config_(config) {}

SecondStepSolver::Output GreedyScheduler::generateSchedule(const SecondStepSolver::Input& input) {
  LOG_TRACE();

  if (config_.mode == Mode::MIN_MEMORY) {
    LOG_TRACE_WITH_INFO("Using MIN_MEMORY mode (prefetch per task, offload everything)");
    return generateMinMemorySchedule(input);
  } else {
    LOG_TRACE_WITH_INFO("Using MAX_PERFORMANCE mode (prefetch all at start, keep everything)");
    return generateMaxPerformanceSchedule(input);
  }
}

SecondStepSolver::Output GreedyScheduler::generateMinMemorySchedule(const SecondStepSolver::Input& input) {
  LOG_TRACE();

  const int numTasks = input.taskGroupRunningTimes.size();
  const int numArrays = input.arraySizes.size();

  SecondStepSolver::Output output;
  output.optimal = true;

  // Cold start: no arrays initially on device (avoids CUDA graph memory issues)
  output.indicesOfArraysInitiallyOnDevice.clear();

  // Track which arrays are currently on device
  std::set<ArrayId> arraysOnDevice;

  // For each task in execution order
  for (int taskId = 0; taskId < numTasks; taskId++) {
    LOG_TRACE_WITH_INFO("Processing task %d", taskId);

    // Step 1: Prefetch all arrays needed by this task (if not already on device)
    for (int arrayId = 0; arrayId < numArrays; arrayId++) {
      if (isArrayUsedByTask(taskId, arrayId, input)) {
        if (arraysOnDevice.find(arrayId) == arraysOnDevice.end()) {
          // Array not on device, need to prefetch
          output.prefetches.push_back(std::make_tuple(taskId, arrayId));
          arraysOnDevice.insert(arrayId);
          LOG_TRACE_WITH_INFO("  Prefetch array %d at task %d", arrayId, taskId);
        }
      }
    }

    // Step 2: After task completes, offload ALL arrays
    // Offload destination = taskId + 1 (must complete before next task)
    int offloadDestination = taskId + 1;

    for (ArrayId arrayId : arraysOnDevice) {
      output.offloadings.push_back(std::make_tuple(taskId, arrayId, offloadDestination));
      LOG_TRACE_WITH_INFO("  Offload array %d at task %d (destination: %d)", arrayId, taskId, offloadDestination);
    }

    // Clear device state (all arrays offloaded)
    arraysOnDevice.clear();
  }

  // Calculate memory usage
  output.anticipatedPeakMemoryUsage = calculatePeakMemory(output, input);

  // Calculate original memory usage (all arrays on device)
  double totalMemory = 0.0;
  for (size_t arraySize : input.arraySizes) {
    totalMemory += arraySize;
  }
  output.originalMemoryUsage = totalMemory / (1024.0 * 1024.0);  // Convert to MiB

  LOG_TRACE_WITH_INFO("Generated MIN_MEMORY schedule:");
  LOG_TRACE_WITH_INFO("  Prefetches: %zu", output.prefetches.size());
  LOG_TRACE_WITH_INFO("  Offloads: %zu", output.offloadings.size());
  LOG_TRACE_WITH_INFO("  Original memory: %.2f MiB", output.originalMemoryUsage);
  LOG_TRACE_WITH_INFO("  Peak memory: %.2f MiB", output.anticipatedPeakMemoryUsage);
  LOG_TRACE_WITH_INFO("  Reduction: %.2f%%", (1.0 - output.anticipatedPeakMemoryUsage / output.originalMemoryUsage) * 100.0);

  return output;
}

SecondStepSolver::Output GreedyScheduler::generateMaxPerformanceSchedule(const SecondStepSolver::Input& input) {
  LOG_TRACE();

  const int numTasks = input.taskGroupRunningTimes.size();
  const int numArrays = input.arraySizes.size();

  SecondStepSolver::Output output;
  output.optimal = true;

  // Strategy: Prefetch ALL arrays at the beginning (before first task)
  // No offloads - keep everything on device

  output.indicesOfArraysInitiallyOnDevice.clear();

  // Prefetch all arrays at task 0
  for (int arrayId = 0; arrayId < numArrays; arrayId++) {
    output.prefetches.push_back(std::make_tuple(0, arrayId));
    LOG_TRACE_WITH_INFO("  Prefetch array %d at task 0", arrayId);
  }

  // No offloads in MAX_PERFORMANCE mode
  output.offloadings.clear();

  // Calculate memory usage (all arrays on device)
  double totalMemory = 0.0;
  for (size_t arraySize : input.arraySizes) {
    totalMemory += arraySize;
  }
  output.originalMemoryUsage = totalMemory / (1024.0 * 1024.0);  // Convert to MiB
  output.anticipatedPeakMemoryUsage = output.originalMemoryUsage;  // Same as original

  LOG_TRACE_WITH_INFO("Generated MAX_PERFORMANCE schedule:");
  LOG_TRACE_WITH_INFO("  Prefetches: %zu (all at task 0)", output.prefetches.size());
  LOG_TRACE_WITH_INFO("  Offloads: 0 (keep all arrays on device)");
  LOG_TRACE_WITH_INFO("  Peak memory: %.2f MiB", output.anticipatedPeakMemoryUsage);

  return output;
}

bool GreedyScheduler::isArrayUsedByTask(
    TaskGroupId taskId,
    ArrayId arrayId,
    const SecondStepSolver::Input& input) const {

  // Check if array is in task's input arrays
  if (input.taskGroupInputArrays[taskId].find(arrayId) != input.taskGroupInputArrays[taskId].end()) {
    return true;
  }

  // Check if array is in task's output arrays
  if (input.taskGroupOutputArrays[taskId].find(arrayId) != input.taskGroupOutputArrays[taskId].end()) {
    return true;
  }

  return false;
}

double GreedyScheduler::calculatePeakMemory(
    const SecondStepSolver::Output& schedule,
    const SecondStepSolver::Input& input) const {

  const int numTasks = input.taskGroupRunningTimes.size();
  const int numArrays = input.arraySizes.size();

  // Simulate the schedule to find peak memory
  std::set<ArrayId> arraysOnDevice;
  double peakMemory = 0.0;

  // Start with initially allocated arrays
  for (ArrayId arrayId : schedule.indicesOfArraysInitiallyOnDevice) {
    arraysOnDevice.insert(arrayId);
  }

  // Calculate initial memory
  double currentMemory = 0.0;
  for (ArrayId arrayId : arraysOnDevice) {
    currentMemory += input.arraySizes[arrayId];
  }
  peakMemory = std::max(peakMemory, currentMemory);

  // Process each task
  for (int taskId = 0; taskId < numTasks; taskId++) {
    // Apply prefetches for this task
    for (const auto& prefetch : schedule.prefetches) {
      if (std::get<0>(prefetch) == taskId) {
        ArrayId arrayId = std::get<1>(prefetch);
        if (arraysOnDevice.find(arrayId) == arraysOnDevice.end()) {
          arraysOnDevice.insert(arrayId);
          currentMemory += input.arraySizes[arrayId];
        }
      }
    }

    // Update peak
    peakMemory = std::max(peakMemory, currentMemory);

    // Apply offloads for this task
    for (const auto& offload : schedule.offloadings) {
      if (std::get<0>(offload) == taskId) {
        ArrayId arrayId = std::get<1>(offload);
        if (arraysOnDevice.find(arrayId) != arraysOnDevice.end()) {
          arraysOnDevice.erase(arrayId);
          currentMemory -= input.arraySizes[arrayId];
        }
      }
    }

    // Update peak after offloads
    peakMemory = std::max(peakMemory, currentMemory);
  }

  // Convert to MiB
  return peakMemory / (1024.0 * 1024.0);
}

std::map<std::string, double> GreedyScheduler::generateWarmStart(const SecondStepSolver::Input& input) {
  LOG_TRACE();
  LOG_TRACE_WITH_INFO("Generating warm start solution for MIP solver");

  const int numTasks = input.taskGroupRunningTimes.size();
  const int numArrays = input.arraySizes.size();

  // Decide which mode to use based on memory constraints
  Config warmStartConfig;

  // Calculate total memory needed (all arrays on device)
  double totalMemoryBytes = 0.0;
  for (size_t arraySize : input.arraySizes) {
    totalMemoryBytes += arraySize;
  }
  double totalMemoryMiB = totalMemoryBytes / (1024.0 * 1024.0);

  // Check if there's a real memory constraint
  auto& config = ConfigurationManager::getConfig().optimization;
  bool hasMemoryConstraint = (config.maxPeakMemoryUsageInMiB > 0.0) &&
                             (config.maxPeakMemoryUsageInMiB < totalMemoryMiB);

  if (hasMemoryConstraint) {
    LOG_TRACE_WITH_INFO("Memory constraint detected (%.2f MiB < %.2f MiB total), using MIN_MEMORY mode for warm start",
                        config.maxPeakMemoryUsageInMiB, totalMemoryMiB);
    warmStartConfig.mode = Mode::MIN_MEMORY;
  } else {
    LOG_TRACE_WITH_INFO("No effective memory constraint (%.2f MiB >= %.2f MiB total), using MAX_PERFORMANCE mode for warm start",
                        config.maxPeakMemoryUsageInMiB, totalMemoryMiB);
    warmStartConfig.mode = Mode::MAX_PERFORMANCE;
  }

  // Generate greedy schedule with selected mode
  GreedyScheduler tempScheduler(warmStartConfig);
  auto schedule = tempScheduler.generateSchedule(input);

  // Convert greedy schedule to Gurobi variable format
  std::map<std::string, double> warmStart;

  // 1. Initial allocation variables: I_{j}
  // Arrays in indicesOfArraysInitiallyOnDevice should have I_{j} = 1
  std::set<ArrayId> initialArrays(schedule.indicesOfArraysInitiallyOnDevice.begin(),
                                    schedule.indicesOfArraysInitiallyOnDevice.end());
  for (int j = 0; j < numArrays; j++) {
    std::string varName = fmt::format("I_{{{}}}", j);
    warmStart[varName] = initialArrays.count(j) > 0 ? 1.0 : 0.0;
  }

  // 2. Prefetch variables: p_{i,j}
  // Set to 1 if (i,j) is in prefetches list
  for (int i = 0; i < numTasks; i++) {
    for (int j = 0; j < numArrays; j++) {
      std::string varName = fmt::format("p_{{{}, {}}}", i, j);
      bool isPrefetched = false;
      for (const auto& prefetch : schedule.prefetches) {
        if (std::get<0>(prefetch) == i && std::get<1>(prefetch) == j) {
          isPrefetched = true;
          break;
        }
      }
      warmStart[varName] = isPrefetched ? 1.0 : 0.0;
    }
  }

  // 3. Offload variables: o_{i,j,k}
  // Set to 1 if (i,j,k) is in offloadings list
  for (int i = 0; i < numTasks; i++) {
    for (int j = 0; j < numArrays; j++) {
      for (int k = 0; k < numTasks; k++) {
        std::string varName = fmt::format("o_{{{},{},{}}}", i, j, k);
        bool isOffloaded = false;
        for (const auto& offload : schedule.offloadings) {
          if (std::get<0>(offload) == i &&
              std::get<1>(offload) == j &&
              std::get<2>(offload) == k) {
            isOffloaded = true;
            break;
          }
        }
        warmStart[varName] = isOffloaded ? 1.0 : 0.0;
      }
    }
  }

  // 4. State variables: x_{i,j} and y_{i,j}
  // These track which arrays are allocated/available at each task
  // Simulate the schedule to determine these states
  std::set<ArrayId> arraysOnDevice;

  // Start with initial arrays
  for (ArrayId arrayId : schedule.indicesOfArraysInitiallyOnDevice) {
    arraysOnDevice.insert(arrayId);
  }

  for (int i = 0; i < numTasks; i++) {
    // Apply prefetches for this task
    for (const auto& prefetch : schedule.prefetches) {
      if (std::get<0>(prefetch) == i) {
        arraysOnDevice.insert(std::get<1>(prefetch));
      }
    }

    // Set y_{i,j} = 1 if array j is available at start of task i
    for (int j = 0; j < numArrays; j++) {
      std::string yVarName = fmt::format("y_{{{}, {}}}", i, j);
      warmStart[yVarName] = arraysOnDevice.count(j) > 0 ? 1.0 : 0.0;
    }

    // Set x_{i,j} = 1 if array j is allocated on device at task i
    for (int j = 0; j < numArrays; j++) {
      std::string xVarName = fmt::format("x_{{{}, {}}}", i, j);
      warmStart[xVarName] = arraysOnDevice.count(j) > 0 ? 1.0 : 0.0;
    }

    // Apply offloads after this task
    for (const auto& offload : schedule.offloadings) {
      if (std::get<0>(offload) == i) {
        arraysOnDevice.erase(std::get<1>(offload));
      }
    }
  }

  LOG_TRACE_WITH_INFO("Generated warm start with %zu variable assignments", warmStart.size());
  LOG_TRACE_WITH_INFO("  Initial arrays: %zu", schedule.indicesOfArraysInitiallyOnDevice.size());
  LOG_TRACE_WITH_INFO("  Prefetches: %zu", schedule.prefetches.size());
  LOG_TRACE_WITH_INFO("  Offloads: %zu", schedule.offloadings.size());

  return warmStart;
}

}  // namespace memopt
