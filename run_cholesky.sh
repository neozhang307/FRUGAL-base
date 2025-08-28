#!/bin/bash

# Script to run tiledCholesky 10 times and collect metrics

# Create CSV output file
OUTPUT_FILE="cholesky_results.csv"
echo "Run,AnticipateRunTime,OriginalRunTime,RealRunTime,OptimalPeakMemory,OriginalPeakMemory,RealPeakMemory,GFlops,Error" > $OUTPUT_FILE

# Run 10 times
for i in {1..10}; do
    echo "Starting run $i..."
    
    # Run the application and capture output
    LOG_FILE="cholesky_run_${i}.log"
    ./build/userApplications/tiledCholesky > "$LOG_FILE" 2>&1
    
    # Extract metrics from log file
    REAL_TIME=$(grep "Total time used (s):" "$LOG_FILE" | awk '{print $5}')
    PEAK_MEMORY=$(grep "\[executor.cu/executeOptimizedGraph\] Peak memory usage (MiB):" "$LOG_FILE" | awk '{print $6}')
    ERROR=$(grep "error = " "$LOG_FILE" | awk '{print $3}')
    GFLOPS=$(grep "\[PDPOTRF\]" "$LOG_FILE" | awk '{print $6}')
    
    # Extract optimization metrics
    ORIGINAL_RUNTIME=$(grep "Original total running time (s):" "$LOG_FILE" | awk '{print $6}')
    ANTICIPATED_RUNTIME=$(grep "Anticipate Total running time (s):" "$LOG_FILE" | awk '{print $6}')
    ORIGINAL_PEAK_MEMORY=$(grep "Original peak memory usage (MiB):" "$LOG_FILE" | awk '{print $6}')
    OPTIMAL_PEAK_MEMORY=$(grep "Optimal peak memory usage (MiB):" "$LOG_FILE" | awk '{print $6}')
    
    # Output to screen
    echo "Run $i results:"
    echo "  Real execution time: $REAL_TIME s"
    echo "  Original runtime: $ORIGINAL_RUNTIME s"
    echo "  Anticipated runtime: $ANTICIPATED_RUNTIME s"
    echo "  GFLOPS: $GFLOPS"
    echo "  Error: $ERROR"
    echo "  Real Peak Memory: $PEAK_MEMORY MiB"
    echo "  Original Peak Memory: $ORIGINAL_PEAK_MEMORY MiB"
    echo "  Optimal Peak Memory: $OPTIMAL_PEAK_MEMORY MiB"
    
    # Write to CSV
    echo "$i,$ANTICIPATED_RUNTIME,$ORIGINAL_RUNTIME,$REAL_TIME,$OPTIMAL_PEAK_MEMORY,$ORIGINAL_PEAK_MEMORY,$PEAK_MEMORY,$GFLOPS,$ERROR" >> $OUTPUT_FILE
    
    # Sleep a bit between runs to let system cool down
    sleep 2
done

# Compute averages and append to CSV
echo "Computing averages..."
echo "Averages:" >> $OUTPUT_FILE

# Use awk to compute averages for numeric columns
awk -F, '
BEGIN {
    antic_sum=0; orig_time_sum=0; real_time_sum=0; opt_mem_sum=0; 
    orig_mem_sum=0; real_mem_sum=0; gflops_sum=0; error_sum=0; 
    count=0
} 
NR>1 && $1!="Averages:" {
    antic_sum+=$2; orig_time_sum+=$3; real_time_sum+=$4; opt_mem_sum+=$5; 
    orig_mem_sum+=$6; real_mem_sum+=$7; gflops_sum+=$8; error_sum+=$9; 
    count++
} 
END {
    if(count>0) {
        printf ",%f,%f,%f,%f,%f,%f,%f,%f\n", 
        antic_sum/count, orig_time_sum/count, real_time_sum/count, opt_mem_sum/count, 
        orig_mem_sum/count, real_mem_sum/count, gflops_sum/count, error_sum/count
    }
}' $OUTPUT_FILE >> $OUTPUT_FILE

echo "Results written to $OUTPUT_FILE"
echo "Done!"