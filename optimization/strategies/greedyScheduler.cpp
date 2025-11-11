#include "greedyScheduler.hpp"

#include <algorithm>
#include <numeric>

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

  // Generate a greedy schedule
  auto schedule = generateSchedule(input);

  // TODO: Convert schedule to Gurobi variable format
  // This requires understanding the variable naming scheme used in secondStepSolver.cpp
  // For now, return empty map (warm start not yet implemented)

  std::map<std::string, double> warmStart;

  LOG_TRACE_WITH_INFO("Warm start generation not yet fully implemented");

  return warmStart;
}

}  // namespace memopt
