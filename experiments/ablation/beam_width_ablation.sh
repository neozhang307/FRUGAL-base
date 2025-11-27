#!/bin/bash
# Beam width ablation study: Test different beam widths (1, 10, 20, 30, ..., 100)
# Uses the same profile from exp1 for fair comparison

set -e

PROJECT_ROOT="/home/users/lingqi.zhang/workspace_x64/FRUGAL/BASE/optimize-cuda-memory-usage-v1"
cd "$PROJECT_ROOT"

# Configuration
PROFILE="results/ablation/exp1/profile_gapoverlap_enabled.json"
CONFIG="config.json"
EXP_DIR="results/ablation/exp2"
OUTPUT_CSV="$EXP_DIR/beam_width_results.csv"

# Beam widths to test
BEAM_WIDTHS=(1 10 20 30 40 50 60 70 80 90 100)

echo "=========================================="
echo "Beam Width Ablation Study"
echo "=========================================="
echo "Profile: $PROFILE"
echo "Config: $CONFIG"
echo "Beam widths: ${BEAM_WIDTHS[@]}"
echo "Output: $OUTPUT_CSV"
echo ""

# Create CSV header
echo "beam_width,score_bytes,score_gb,solve_time_ms,predicted_runtime_s,peak_memory_mib,mip_solve_time_s,status" > "$OUTPUT_CSV"

# Test each beam width
for BEAM_WIDTH in "${BEAM_WIDTHS[@]}"; do
    echo ""
    echo "=== Testing Beam Width: $BEAM_WIDTH ==="

    PLAN_FILE="$EXP_DIR/beam_analysis/plan_beam${BEAM_WIDTH}.json"
    LOG_FILE="$EXP_DIR/logs/beam${BEAM_WIDTH}.log"

    # Update config with current beam width
    CONFIG_TEMP="$EXP_DIR/config_beam${BEAM_WIDTH}.json"
    jq ".optimization.beamWidth = $BEAM_WIDTH" "$CONFIG" > "$CONFIG_TEMP"

    echo "Running optimization with beam width $BEAM_WIDTH..."

    # Run standalone optimizer
    build/tools/standaloneOptimizer \
        "$PROFILE" \
        "$PLAN_FILE" \
        --config="$CONFIG_TEMP" \
        > "$LOG_FILE" 2>&1

    # Extract metrics from log
    SCORE=$(grep "Solution found with total overlap:" "$LOG_FILE" | awk '{print $7}')
    SOLVE_TIME=$(grep "Solver execution completed in" "$LOG_FILE" | awk '{print $5}')
    PREDICTED_TIME=$(grep "Total running time (s):" "$LOG_FILE" | head -1 | awk '{print $5}')
    PEAK_MEM=$(grep "Optimal peak memory usage (MiB):" "$LOG_FILE" | awk '{print $6}')
    MIP_TIME=$(grep "Time for solving the MIP problem" "$LOG_FILE" | awk '{print $8}')

    # Check MIP status
    if grep -q "OPTIMAL" "$LOG_FILE"; then
        STATUS="OPTIMAL"
    elif grep -q "INFEASIBLE" "$LOG_FILE"; then
        STATUS="INFEASIBLE"
    else
        STATUS="UNKNOWN"
    fi

    # Convert to appropriate units
    SCORE_GB=$(echo "scale=2; $SCORE / 1024^3" | bc)
    SOLVE_TIME_MS=$(echo "scale=3; $SOLVE_TIME * 1000" | bc)

    echo "  Score: $SCORE_GB GB"
    echo "  First-step solve time: ${SOLVE_TIME_MS} ms"
    echo "  Predicted runtime: ${PREDICTED_TIME} s"
    echo "  Peak memory: ${PEAK_MEM} MiB"
    echo "  MIP solve time: ${MIP_TIME} s"
    echo "  Status: $STATUS"

    # Append to CSV
    echo "$BEAM_WIDTH,$SCORE,$SCORE_GB,$SOLVE_TIME_MS,$PREDICTED_TIME,$PEAK_MEM,$MIP_TIME,$STATUS" >> "$OUTPUT_CSV"

    # Clean up temp config
    rm "$CONFIG_TEMP"
done

echo ""
echo "=========================================="
echo "✅ Beam width ablation complete"
echo "Results saved to: $OUTPUT_CSV"
echo "=========================================="
echo ""

# Display summary
echo "Summary:"
column -t -s',' "$OUTPUT_CSV"
