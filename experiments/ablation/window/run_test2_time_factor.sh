#!/bin/bash
# Test 2: Relative Time Factor Ablation
# Variable: Time factors (1, 5, 10, 20, 30, 40, 50)
# Fixed: Distance limits = 1000 (effectively infinite)

set -e

PROJECT_ROOT="/home/users/lingqi.zhang/workspace_x64/FRUGAL/BASE/optimize-cuda-memory-usage-v1"
cd "$PROJECT_ROOT"

# Configuration
PROFILE="results/ablation/exp1/profile_gapoverlap_enabled.json"
FIRST_STEP_SOLUTION="results/ablation/exp1/topk_extensive/first_step/topk100.json"
SOLUTION_ID=84  # 6th fastest solution
CONFIG="experiments/ablation/window/config.json"
EXP_DIR="results/ablation/exp3/test2_time_factor"

# Create directories
mkdir -p "$EXP_DIR/logs"
mkdir -p "$EXP_DIR/plans"

# Output CSV
OUTPUT_CSV="$EXP_DIR/window_time_factor_results.csv"

# Time factors to test
TIME_FACTORS=(1 5 10 20 30 40 50)

# Fixed distance limit (set to very large = infinite)
DIST_LIMIT=1000

echo "=========================================="
echo "Test 2: Relative Time Factor Ablation"
echo "=========================================="
echo "Base solution: Top-100 Solution #$SOLUTION_ID"
echo "Profile: $PROFILE"
echo "Time factors: ${TIME_FACTORS[@]}"
echo "Distance limits (fixed): $DIST_LIMIT"
echo "Output: $EXP_DIR"
echo ""

# Create CSV header
echo "time_factor,mip_solve_time_s,predicted_runtime_s,peak_memory_mib,status" > "$OUTPUT_CSV"

# Test each time factor
for FACTOR in "${TIME_FACTORS[@]}"; do
    echo ""
    echo "=== Testing Time Factor: $FACTOR ==="

    PLAN_FILE="$EXP_DIR/plans/plan_factor${FACTOR}.json"
    LOG_FILE="$EXP_DIR/logs/factor${FACTOR}.log"
    CONFIG_TEMP="$EXP_DIR/config_factor${FACTOR}.json"

    # Create temporary config with current window settings
    jq "
        .optimization.prefetchLookbackDistanceLimit = $DIST_LIMIT |
        .optimization.offloadLookaheadDistanceLimit = $DIST_LIMIT |
        .optimization.prefetchLookbackTimeBudgetFactor = $FACTOR |
        .optimization.offloadLookaheadComputeTimeFactor = $FACTOR
    " "$CONFIG" > "$CONFIG_TEMP"

    echo "Running optimization with time factor = $FACTOR..."

    # Run standalone optimizer with specific first-step solution
    build/tools/standaloneOptimizer \
        "$PROFILE" \
        "$PLAN_FILE" \
        --config="$CONFIG_TEMP" \
        --load-topk="$FIRST_STEP_SOLUTION" \
        --solution-index=$SOLUTION_ID \
        > "$LOG_FILE" 2>&1

    # Extract key metrics
    if grep -q "OPTIMAL" "$LOG_FILE"; then
        STATUS="OPTIMAL"
    elif grep -q "INFEASIBLE" "$LOG_FILE"; then
        STATUS="INFEASIBLE"
    else
        STATUS="UNKNOWN"
    fi

    MIP_TIME=$(grep "Time for solving the MIP problem" "$LOG_FILE" | awk '{print $NF}' || echo "N/A")
    PREDICTED=$(grep "Total running time (s):" "$LOG_FILE" | head -1 | awk '{print $NF}' || echo "N/A")
    PEAK_MEM=$(grep "Optimal peak memory usage (MiB):" "$LOG_FILE" | awk '{print $NF}' || echo "N/A")

    echo "  Time factor: $FACTOR"
    echo "  MIP solve time: ${MIP_TIME}s"
    echo "  Predicted runtime: ${PREDICTED}s"
    echo "  Peak memory: ${PEAK_MEM} MiB"
    echo "  Status: $STATUS"

    # Append to CSV
    echo "$FACTOR,$MIP_TIME,$PREDICTED,$PEAK_MEM,$STATUS" >> "$OUTPUT_CSV"

    # Clean up temp config
    rm "$CONFIG_TEMP"
done

echo ""
echo "=========================================="
echo "✅ Test 2 complete"
echo "Results saved to: $OUTPUT_CSV"
echo "=========================================="
echo ""
echo "Summary:"
column -t -s',' "$OUTPUT_CSV"
