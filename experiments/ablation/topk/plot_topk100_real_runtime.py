#!/usr/bin/env python3
"""
Plot Top-100 results based on real GPU execution runtime
Focuses on: predicted vs actual runtime, and score vs actual runtime
"""

import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def parse_execution_log(log_path, topk_path):
    """Parse execution log to extract all metrics"""

    # Load TopK file for scores
    with open(topk_path, 'r') as f:
        topk_data = json.load(f)

    # Read log file
    with open(log_path, 'r') as f:
        content = f.read()

    # Split by solution
    solutions = []
    solution_blocks = re.split(r'=== Solution #(\d+) ===', content)[1:]  # Skip first empty

    for i in range(0, len(solution_blocks), 2):
        sol_id = int(solution_blocks[i])
        sol_content = solution_blocks[i + 1]

        # Extract metrics
        score_match = re.search(r'Beam search score: (\d+)', sol_content)
        predicted_match = re.search(r'Predicted runtime: ([\d.]+)s', sol_content)
        actual_match = re.search(r'Execution time: ([\d.]+) ms', sol_content)
        memory_match = re.search(r'Peak GPU memory usage during execution: ([\d.]+) MB', sol_content)

        if score_match and predicted_match and actual_match and memory_match:
            score = int(score_match.group(1))
            predicted_s = float(predicted_match.group(1))
            actual_ms = float(actual_match.group(1))
            peak_mem_mb = float(memory_match.group(1))

            solutions.append({
                'solution_id': sol_id,
                'score_bytes': score,
                'score_gb': score / (1024**3),
                'predicted_time_s': predicted_s,
                'predicted_time_ms': predicted_s * 1000,
                'actual_time_ms': actual_ms,
                'actual_time_s': actual_ms / 1000,
                'peak_memory_mb': peak_mem_mb,
                'prediction_error_ms': abs(predicted_s * 1000 - actual_ms),
                'prediction_error_pct': abs(predicted_s * 1000 - actual_ms) / actual_ms * 100
            })

    return pd.DataFrame(solutions)

def plot_predicted_vs_actual(df, output_dir):
    """Main plot: Predicted vs Actual runtime with perfect prediction line"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap-style coloring: red (high score) to green (low score)
    # Normalize scores to 0-1 range for colormap
    score_min = df['score_gb'].min()
    score_max = df['score_gb'].max()
    norm_scores = (df['score_gb'] - score_min) / (score_max - score_min)

    # Use RdYlGn colormap: red=high score, green=low score
    colors = plt.cm.RdYlGn_r(norm_scores)

    # Scatter plot with heatmap coloring
    scatter = ax.scatter(df['predicted_time_ms'], df['actual_time_ms'],
                        c=df['score_gb'], cmap='RdYlGn_r',
                        s=150, alpha=0.8,
                        edgecolors='black', linewidth=1.0,
                        vmin=score_min, vmax=score_max)

    # Perfect prediction line (diagonal)
    # Set explicit axis ranges as requested
    x_min, x_max = 6700, 7200
    y_min, y_max = 6700, 7000

    ax.plot([x_min, x_max], [x_min, x_max], '--',
           color='black', label='Perfect Prediction', linewidth=3.5, alpha=0.9, zorder=1000)

    ax.set_xlabel('Predicted Runtime (ms)', fontweight='bold', fontsize=22)
    ax.set_ylabel('Actual GPU Runtime (ms)', fontweight='bold', fontsize=22)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set custom axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add colorbar for score heatmap
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Task Scheduling Score (Data Reuse) (GB)', fontweight='bold', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    # Calculate and display statistics (without correlation)
    avg_error = df['prediction_error_pct'].mean()
    max_error = df['prediction_error_pct'].max()

    # Runtime statistics
    runtime_mean = df['actual_time_ms'].mean()
    runtime_std = df['actual_time_ms'].std()

    stats_text = (
        f'Prediction Accuracy:\n'
        f'Avg Error: {avg_error:.2f}%\n'
        f'Max Error: {max_error:.2f}%\n'
        f'\n'
        f'Actual Runtime:\n'
        f'Mean: {runtime_mean:.1f} ms\n'
        f'Std Dev: {runtime_std:.1f} ms\n'
        f'\n'
        f'Top-{len(df)} Solutions'
    )
    # Move to upper left
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes, ha='left', va='top',
           fontsize=16, family='monospace', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                    edgecolor='gray', linewidth=2))

    # Add legend for diagonal line (also upper area, right side)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='black', fontsize=16)

    plt.tight_layout()
    output_base = output_dir / 'topk100_predicted_vs_actual_execution'
    plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_base}.pdf and .png")
    plt.close()

def plot_score_vs_actual_runtime(df, output_dir):
    """Plot: Beam score vs actual runtime (showing score doesn't matter)"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Color by score tier
    unique_scores = sorted(df['score_gb'].unique(), reverse=True)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(unique_scores)))
    color_map = {score: colors[i] for i, score in enumerate(unique_scores)}

    for score in unique_scores:
        mask = df['score_gb'] == score
        subset = df[mask]
        ax.scatter(subset['score_gb'], subset['actual_time_ms'],
                  s=150, alpha=0.7,
                  color=color_map[score],
                  edgecolors='black', linewidth=1,
                  label=f'{score:.2f} GB (n={len(subset)})')

    ax.set_xlabel('Beam Search Score (GB)', fontweight='bold', fontsize=18)
    ax.set_ylabel('Actual GPU Runtime (ms)', fontweight='bold', fontsize=18)
    ax.legend(loc='best', framealpha=0.95, edgecolor='black', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Calculate correlation
    correlation = df['score_gb'].corr(df['actual_time_ms'])

    # Add annotation showing NO correlation
    ax.text(0.02, 0.98,
           f'Score vs Runtime:\n'
           f'Correlation: {correlation:.4f}\n'
           f'Score does NOT predict runtime!',
           transform=ax.transAxes, ha='left', va='top',
           fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9,
                    edgecolor='red', linewidth=2))

    plt.tight_layout()
    output_base = output_dir / 'topk100_score_vs_actual_runtime'
    plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_base}.pdf and .png")
    plt.close()

def print_summary(df):
    """Print comprehensive summary"""
    print("\n" + "="*70)
    print("TOP-100 EXECUTION RESULTS SUMMARY")
    print("="*70)

    print("\n📊 Prediction Accuracy:")
    print(f"  Correlation (predicted vs actual): {df['predicted_time_ms'].corr(df['actual_time_ms']):.4f}")
    print(f"  Average prediction error: {df['prediction_error_pct'].mean():.2f}%")
    print(f"  Max prediction error: {df['prediction_error_pct'].max():.2f}%")
    print(f"  Min prediction error: {df['prediction_error_pct'].min():.2f}%")

    print("\n⏱️  Actual Runtime Statistics (ms):")
    print(f"  Min:  {df['actual_time_ms'].min():.2f}")
    print(f"  Max:  {df['actual_time_ms'].max():.2f}")
    print(f"  Mean: {df['actual_time_ms'].mean():.2f}")
    print(f"  Std:  {df['actual_time_ms'].std():.2f}")
    variance_ms = df['actual_time_ms'].max() - df['actual_time_ms'].min()
    variance_pct = 100 * variance_ms / df['actual_time_ms'].min()
    print(f"  Range: {variance_ms:.2f} ms ({variance_pct:.2f}%)")

    print("\n🎯 Score vs Runtime Correlation:")
    score_runtime_corr = df['score_gb'].corr(df['actual_time_ms'])
    print(f"  Correlation: {score_runtime_corr:.4f}")
    if abs(score_runtime_corr) < 0.3:
        print(f"  ⚠️  WEAK/NO correlation - beam score does NOT predict runtime!")

    print("\n🏆 Top 10 Fastest Solutions:")
    print("  Rank  Sol#  Score (GB)  Predicted (ms)  Actual (ms)  Error (%)")
    df_sorted = df.sort_values('actual_time_ms')
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        print(f"  {i+1:4d}  #{row['solution_id']:3.0f}  {row['score_gb']:10.2f}  "
              f"{row['predicted_time_ms']:13.1f}  {row['actual_time_ms']:11.1f}  "
              f"{row['prediction_error_pct']:7.2f}")

    print("\n🐌 Bottom 10 Slowest Solutions:")
    print("  Rank  Sol#  Score (GB)  Predicted (ms)  Actual (ms)  Error (%)")
    for i, (idx, row) in enumerate(df_sorted.tail(10).iterrows()):
        rank = len(df) - 9 + i
        print(f"  {rank:4d}  #{row['solution_id']:3.0f}  {row['score_gb']:10.2f}  "
              f"{row['predicted_time_ms']:13.1f}  {row['actual_time_ms']:11.1f}  "
              f"{row['prediction_error_pct']:7.2f}")

def main():
    parser = argparse.ArgumentParser(description="Parse Top-100 execution results")
    parser.add_argument('--log-file', type=str,
                       default='results/ablation/exp1/topk_extensive/all_topk100_execution.log',
                       help='Execution log file')
    parser.add_argument('--topk-file', type=str,
                       default='results/ablation/exp1/topk_extensive/first_step/topk100.json',
                       help='TopK solutions file')
    parser.add_argument('--output-dir', type=str,
                       default='results/ablation/exp1/topk_extensive/plots/real',
                       help='Output directory for plots')
    parser.add_argument('--output-csv', type=str,
                       default='results/ablation/exp1/topk_extensive/topk100_results.csv',
                       help='Output CSV file')

    args = parser.parse_args()

    # Parse log
    print(f"Parsing execution log: {args.log_file}")
    df = parse_execution_log(args.log_file, args.topk_file)
    print(f"✅ Parsed {len(df)} solutions")

    # Save CSV
    df.to_csv(args.output_csv, index=False)
    print(f"✅ Saved results to: {args.output_csv}")

    # Create plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📊 Generating plots...")
    plot_predicted_vs_actual(df, output_dir)
    plot_score_vs_actual_runtime(df, output_dir)

    print(f"\n✅ All plots saved to: {output_dir}")

    # Print summary
    print_summary(df)

if __name__ == '__main__':
    main()
