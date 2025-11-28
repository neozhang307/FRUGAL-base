#!/usr/bin/env python3
"""
Analyze Top-100 execution results
Extract metrics from TopK solutions and optimized plans
"""

import json
import re
from pathlib import Path
import pandas as pd
import argparse

def parse_topk_file(topk_path):
    """Parse Top-K solutions file"""
    with open(topk_path, 'r') as f:
        data = json.load(f)

    solutions = []
    for sol in data['solutions']:
        solutions.append({
            'solution_id': sol['rank'] - 1,  # 0-indexed
            'rank': sol['rank'],
            'score_bytes': sol['dataReuseScore'],
            'score_gb': sol['dataReuseScore'] / (1024**3)
        })

    return pd.DataFrame(solutions)

def parse_optimization_log(log_path):
    """Extract metrics from optimization log"""
    with open(log_path, 'r') as f:
        content = f.read()

    metrics = {}

    # Extract runtime
    match = re.search(r'Total running time \(s\): ([\d.]+)', content)
    if match:
        metrics['predicted_time_s'] = float(match.group(1))

    # Extract peak memory
    match = re.search(r'Optimal peak memory usage \(MiB\): ([\d.]+)', content)
    if match:
        metrics['peak_memory_mib'] = float(match.group(1))

    # Extract optimality - try multiple patterns
    match = re.search(r'Optimization completed: (OPTIMAL|INFEASIBLE|FEASIBLE)', content)
    if not match:
        match = re.search(r'MIP status: (OPTIMAL|INFEASIBLE|FEASIBLE)', content)
    if match:
        metrics['status'] = match.group(1)
    else:
        metrics['status'] = 'UNKNOWN'

    # Extract MIP solve time
    match = re.search(r'Time for solving the MIP problem \(seconds\): ([\d.]+)', content)
    if match:
        metrics['mip_solve_time_s'] = float(match.group(1))

    return metrics

def parse_plan_file(plan_path):
    """Extract metrics from plan JSON"""
    with open(plan_path, 'r') as f:
        data = json.load(f)

    return {
        'anticipated_peak_mib': data.get('anticipatedPeakMemoryUsage', None),
        'original_memory_mib': data.get('originalMemoryUsage', None),
        'optimal': data.get('optimal', False)
    }

def analyze_all_solutions(exp_dir):
    """Analyze all Top-100 solutions"""
    exp_path = Path(exp_dir)

    # Load Top-K solutions
    topk_path = exp_path / 'first_step' / 'topk100.json'
    print(f"Loading Top-K solutions from: {topk_path}")
    df = parse_topk_file(topk_path)

    # Parse all optimization logs and plans
    plan_dir = exp_path / 'second_step_plans'

    predicted_times = []
    peak_memories = []
    statuses = []
    mip_times = []

    for i in range(len(df)):
        log_path = plan_dir / f'optimization_sol{i}.log'
        plan_path = plan_dir / f'plan_sol{i}.json'

        # Parse log
        if log_path.exists():
            log_metrics = parse_optimization_log(log_path)
            predicted_times.append(log_metrics.get('predicted_time_s', None))
            peak_memories.append(log_metrics.get('peak_memory_mib', None))
            statuses.append(log_metrics.get('status', 'UNKNOWN'))
            mip_times.append(log_metrics.get('mip_solve_time_s', None))
        else:
            predicted_times.append(None)
            peak_memories.append(None)
            statuses.append('NO_LOG')
            mip_times.append(None)

    df['predicted_time_s'] = predicted_times
    df['peak_memory_mib'] = peak_memories
    df['status'] = statuses
    df['mip_solve_time_s'] = mip_times

    return df

def print_summary(df):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("Top-100 Solutions Analysis Summary")
    print("="*60)

    # Score distribution
    print("\nScore Distribution:")
    score_counts = df.groupby('score_gb').size().sort_index(ascending=False)
    for score, count in score_counts.items():
        print(f"  {score:.2f} GB: {count} solutions")

    # MIP status
    print("\nMIP Optimization Status:")
    status_counts = df['status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count} solutions")

    # Runtime statistics
    print("\nPredicted Runtime Statistics (seconds):")
    runtime_stats = df['predicted_time_s'].describe()
    print(f"  Min:  {runtime_stats['min']:.3f}")
    print(f"  Max:  {runtime_stats['max']:.3f}")
    print(f"  Mean: {runtime_stats['mean']:.3f}")
    print(f"  Std:  {runtime_stats['std']:.3f}")
    print(f"  Range: {runtime_stats['max'] - runtime_stats['min']:.3f} s")
    print(f"  Variance: {((runtime_stats['max'] - runtime_stats['min']) / runtime_stats['min'] * 100):.1f}%")

    # MIP solve time
    print("\nMIP Solve Time Statistics (seconds):")
    mip_stats = df['mip_solve_time_s'].describe()
    print(f"  Min:  {mip_stats['min']:.3f}")
    print(f"  Max:  {mip_stats['max']:.3f}")
    print(f"  Mean: {mip_stats['mean']:.3f}")

    # Top 10 by score
    print("\nTop 10 Solutions by Beam Score:")
    print("  Rank  Score (GB)  Predicted (s)  Status")
    for idx, row in df.head(10).iterrows():
        print(f"  #{row['solution_id']:3d}  {row['score_gb']:10.2f}  {row['predicted_time_s']:12.3f}  {row['status']}")

    # Bottom 10 by score
    print("\nBottom 10 Solutions by Beam Score:")
    print("  Rank  Score (GB)  Predicted (s)  Status")
    for idx, row in df.tail(10).iterrows():
        print(f"  #{row['solution_id']:3d}  {row['score_gb']:10.2f}  {row['predicted_time_s']:12.3f}  {row['status']}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Top-100 solutions")
    parser.add_argument('--exp-dir', type=str,
                       default='results/ablation/exp1/topk_extensive',
                       help='Experiment directory')
    parser.add_argument('--output-csv', type=str,
                       default='results/ablation/exp1/topk_extensive/topk100_middle_results.csv',
                       help='Output CSV file')

    args = parser.parse_args()

    # Analyze solutions
    df = analyze_all_solutions(args.exp_dir)

    # Save to CSV
    df.to_csv(args.output_csv, index=False)
    print(f"\n✅ Saved analysis to: {args.output_csv}")

    # Print summary
    print_summary(df)

if __name__ == '__main__':
    main()
