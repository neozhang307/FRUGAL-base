#include <map>
#include <string>
#include <vector>

#include "../optimization/optimizationInput.hpp"

namespace memopt {

/**
 * @brief Parser for MemRed memory analysis output files
 * 
 * MemRedAnalysisParser reads and processes analysis data produced by the MemRed tool,
 * which provides detailed information about kernel memory access patterns and data dependencies.
 * This information is used to optimize CUDA memory usage.
 */
struct MemRedAnalysisParser {
  /** @brief Path to the MemRed analysis output file */
  static constexpr std::string_view ANALYSIS_FILE = "./.memred.memory.analysis.out";

  /**
   * @brief Holds information about a CUDA kernel's memory access patterns
   *
   * Stores kernel function name, overall memory effect classification,
   * and detailed information about each pointer argument's memory access patterns.
   */
  struct KernelInfo {
    /** @brief Kernel function name as extracted from the analysis file */
    std::string funcName;
    
    /** @brief Overall memory effect (e.g., "ArgMemOnly", "AnyMem") */
    std::string memoryEffect;
    
    /** @brief Vector of pairs containing argument index and memory effect type
     *  Memory effect types include: "ReadOnly", "WriteOnly", "ReadWrite", "None"
     */
    std::vector<std::pair<int, std::string>> ptrArgInfos;
  };

  /** @brief Mapping from kernel function names to their corresponding memory access information */
  std::map<std::string, KernelInfo> funcNameToKernelInfoMap;

  /**
   * @brief Constructor parses the MemRed analysis file and builds the kernel info map
   */
  MemRedAnalysisParser();

  /**
   * @brief Extracts data dependencies from a CUDA graph kernel node
   *
   * Identifies the input and output memory regions accessed by the kernel
   * based on the memory effect analysis from MemRed.
   *
   * @param kernelNode The CUDA graph node representing a kernel execution
   * @return DataDependency structure containing sets of input and output memory pointers
   */
  OptimizationInput::TaskGroup::DataDependency getKernelDataDependency(cudaGraphNode_t kernelNode);
};

}  // namespace memopt