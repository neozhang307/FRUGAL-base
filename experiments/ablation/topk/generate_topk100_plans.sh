#!/bin/bash
# Generate optimized plans for all 100 Top-K solutions

set -e  # Exit on error

PROJECT_ROOT="/home/users/lingqi.zhang/workspace_x64/FRUGAL/BASE/optimize-cuda-memory-usage-v1"
cd "$PROJECT_ROOT"

# Experiment settings
EXP_DIR="results/ablation/exp1/topk_extensive"
PROFILE="results/ablation/exp1/profile_gapoverlap_enabled.json"
TOPK_FILE="$EXP_DIR/first_step/topk100.json"
PLAN_DIR="$EXP_DIR/second_step_plans"
CONFIG="experiments/ablation/topk/config.json"

echo "=========================================="
echo "Generate Plans for All Top-100 Solutions"
echo "=========================================="
echo "Profile: $PROFILE"
echo "Top-K file: $TOPK_FILE"
echo "Output directory: $PLAN_DIR"
echo ""

# Verify Top-K file exists
if [ ! -f "$TOPK_FILE" ]; then
    echo "❌ ERROR: Top-K file not found: $TOPK_FILE"
    echo "Run generate_topk100.sh first"
    exit 1
fi

# Create output directory
mkdir -p "$PLAN_DIR"

# Count solutions in TopK file
NUM_SOLUTIONS=$(jq '.numSolutions' "$TOPK_FILE")
echo "Found $NUM_SOLUTIONS solutions in Top-K file"
echo ""

# Generate plan for each solution
echo "=== Step 2: Generate Optimized Plans for All Solutions ==="
START_TIME=$(date +%s)

for i in $(seq 0 $((NUM_SOLUTIONS - 1))); do
    PLAN_FILE="$PLAN_DIR/plan_sol${i}.json"

    echo -n "Solution #$i: "

    # Run standaloneOptimizer with specific solution index
    build/tools/standaloneOptimizer \
        "$PROFILE" \
        "$PLAN_FILE" \
        --config="$CONFIG" \
        --load-topk="$TOPK_FILE" \
        --solution-index=$i \
        > "$PLAN_DIR/optimization_sol${i}.log" 2>&1

    if [ -f "$PLAN_FILE" ]; then
        # Extract key metrics from plan
        PEAK_MEM=$(jq -r '.anticipatedPeakMemoryUsage' "$PLAN_FILE" 2>/dev/null || echo "N/A")
        # Runtime is extracted from log since it's not in plan JSON
        RUNTIME=$(grep -oP 'Total running time \(s\): \K[\d.]+' "$PLAN_DIR/optimization_sol${i}.log" 2>/dev/null || echo "N/A")
        echo "✅ Peak: ${PEAK_MEM} MiB, Runtime: ${RUNTIME}s"
    else
        echo "❌ FAILED (see log: $PLAN_DIR/optimization_sol${i}.log)"
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "✅ All plans generated in ${ELAPSED} seconds"
echo "=========================================="

# Summary statistics
echo ""
echo "=== Summary Statistics ==="
echo "Total solutions: $NUM_SOLUTIONS"
echo "Plans generated: $(ls -1 $PLAN_DIR/plan_sol*.json 2>/dev/null | wc -l)"
echo ""

# Analyze peak memory distribution
echo "Peak Memory Distribution:"
for plan in $PLAN_DIR/plan_sol*.json; do
    if [ -f "$plan" ]; then
        jq -r '.anticipatedPeakMemoryUsage' "$plan" 2>/dev/null
    fi
done | sort -n | uniq -c | awk '{printf "  %s solutions: %.0f MiB\n", $1, $2}'

echo ""
echo "Runtime Distribution (seconds):"
for log in $PLAN_DIR/optimization_sol*.log; do
    if [ -f "$log" ]; then
        grep -oP 'Total running time \(s\): \K[\d.]+' "$log" 2>/dev/null
    fi
done | sort -n | awk 'BEGIN{min=999999; max=0; sum=0; count=0}
    {
        if ($1 < min) min = $1;
        if ($1 > max) max = $1;
        sum += $1;
        count++;
    }
    END {
        if (count > 0) {
            avg = sum / count;
            printf "  Min: %.3f s\n  Max: %.3f s\n  Avg: %.3f s\n  Range: %.3f s (%.1f%%)\n",
                   min, max, avg, max-min, ((max-min)/min)*100
        }
    }'

echo ""
echo "Output files:"
echo "  Plans: $PLAN_DIR/plan_sol{0..99}.json"
echo "  Logs:  $PLAN_DIR/optimization_sol{0..99}.log"
