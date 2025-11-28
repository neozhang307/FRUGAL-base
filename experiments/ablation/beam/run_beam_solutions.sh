#!/bin/bash
# Execute all beam width solutions on GPU to get actual runtime
# Compare MIP predictions across different beam widths

set -e

PROJECT_ROOT="/home/users/lingqi.zhang/workspace_x64/FRUGAL/BASE/optimize-cuda-memory-usage-v1"
cd "$PROJECT_ROOT"

# Configuration
N=102400
T=4
EXP_DIR="results/ablation/exp2"
PLAN_DIR="$EXP_DIR/beam_analysis"
OUTPUT_LOG="$EXP_DIR/beam_execution_results.log"

# Create directories
mkdir -p "$EXP_DIR"

# Beam widths to test
BEAM_WIDTHS=(1 10 20 30 40 50 60 70 80 90 100)

echo "==========================================" | tee "$OUTPUT_LOG"
echo "Execute Beam Width Solutions on GPU" | tee -a "$OUTPUT_LOG"
echo "==========================================" | tee -a "$OUTPUT_LOG"
echo "Workload: Tiled Cholesky N=$N, T=$T" | tee -a "$OUTPUT_LOG"
echo "Plans directory: $PLAN_DIR" | tee -a "$OUTPUT_LOG"
echo "Output log: $OUTPUT_LOG" | tee -a "$OUTPUT_LOG"
echo "" | tee -a "$OUTPUT_LOG"

# Execute each beam width solution
for BEAM_WIDTH in "${BEAM_WIDTHS[@]}"; do
    PLAN_FILE="$PLAN_DIR/plan_beam${BEAM_WIDTH}.json"

    if [ ! -f "$PLAN_FILE" ]; then
        echo "⚠️  Plan file not found: $PLAN_FILE" | tee -a "$OUTPUT_LOG"
        continue
    fi

    echo "=== Beam Width: $BEAM_WIDTH ===" | tee -a "$OUTPUT_LOG"

    # Extract MIP predicted runtime from beam_middle_results.csv
    CSV_FILE="$EXP_DIR/beam_middle_results.csv"
    if [ -f "$CSV_FILE" ]; then
        PREDICTED=$(awk -F',' -v bw="$BEAM_WIDTH" '$1 == bw {print $5}' "$CSV_FILE")
    else
        PREDICTED="N/A"
    fi
    echo "MIP predicted runtime: ${PREDICTED}s" | tee -a "$OUTPUT_LOG"

    echo "Executing solution..." | tee -a "$OUTPUT_LOG"

    # Run on GPU
    build/userApplications/tiledCholeskyAblation \
        --N=$N --T=$T \
        --run-plan \
        --load-plan="$PLAN_FILE" \
        2>&1 | tee -a "$OUTPUT_LOG"

    echo "✅ Beam width $BEAM_WIDTH completed" | tee -a "$OUTPUT_LOG"
    echo "" | tee -a "$OUTPUT_LOG"
done

echo "==========================================" | tee -a "$OUTPUT_LOG"
echo "✅ All beam width solutions executed" | tee -a "$OUTPUT_LOG"
echo "==========================================" | tee -a "$OUTPUT_LOG"
echo "" | tee -a "$OUTPUT_LOG"
echo "Next step: Parse results and generate plots" | tee -a "$OUTPUT_LOG"
echo "  python experiments/ablation/parse_beam_execution.py" | tee -a "$OUTPUT_LOG"
