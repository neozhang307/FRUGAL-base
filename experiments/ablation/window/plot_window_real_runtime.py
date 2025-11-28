#!/usr/bin/env python3
"""
Parse window size execution results and generate final results CSV with actual runtime.
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# Paths
PROJECT_ROOT = Path("/home/users/lingqi.zhang/workspace_x64/FRUGAL/BASE/optimize-cuda-memory-usage-v1")

def parse_execution_log(log_file, test_type):
    """Parse execution log to extract parameter and actual time."""
    results = []

    with open(log_file, 'r') as f:
        content = f.read()

    if test_type == 'test1':
        # Split by distance limit sections
        sections = re.split(r'=== Distance Limit: (\d+) ===', content)
        param_name = 'distance_limit'
    else:  # test2
        # Split by time factor sections
        sections = re.split(r'=== Time Factor: (\d+) ===', content)
        param_name = 'time_factor'

    for i in range(1, len(sections), 2):
        param_value = int(sections[i])
        section_content = sections[i + 1]

        # Extract actual execution time
        actual_match = re.search(r'Execution time: ([\d.]+) ms', section_content)
        if not actual_match:
            print(f"Warning: Could not find actual runtime for {param_name}={param_value}")
            continue
        actual_time_ms = float(actual_match.group(1))

        results.append({
            param_name: param_value,
            'actual_time_ms': actual_time_ms
        })

    return pd.DataFrame(results)


def load_middle_results(middle_csv, test_type):
    """Load middle results (predicted runtime, MIP time, etc.)."""
    df = pd.read_csv(middle_csv)
    return df


def plot_predicted_vs_actual(df, output_dir, test_type, param_name):
    """Plot predicted vs actual runtime."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with parameter as color
    scatter = ax.scatter(df['predicted_time_ms'], df['actual_time_ms'],
                        c=df[param_name], cmap='viridis',
                        s=300, alpha=0.8,
                        edgecolors='black', linewidth=1.5)

    # Add labels to each point
    for idx, row in df.iterrows():
        ax.annotate(f"{param_name.split('_')[0]}={row[param_name]}",
                   (row['predicted_time_ms'], row['actual_time_ms']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')

    # Perfect prediction line
    min_val = min(df['predicted_time_ms'].min(), df['actual_time_ms'].min())
    max_val = max(df['predicted_time_ms'].max(), df['actual_time_ms'].max())
    ax.plot([min_val, max_val], [min_val, max_val],
           'r--', linewidth=2, alpha=0.7, label='Perfect Prediction')

    # Calculate prediction errors
    df['error_pct'] = abs(df['predicted_time_ms'] - df['actual_time_ms']) / df['actual_time_ms'] * 100
    avg_error = df['error_pct'].mean()

    ax.text(0.02, 0.98, f'Avg Prediction Error: {avg_error:.2f}%',
           transform=ax.transAxes, ha='left', va='top',
           fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Predicted Runtime (ms)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Actual GPU Runtime (ms)', fontsize=16, fontweight='bold')
    ax.set_title(f'Window {test_type.upper()}: Predicted vs Actual Runtime',
                fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='lower right')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(param_name.replace('_', ' ').title(), fontweight='bold', fontsize=14)

    plt.tight_layout()
    output_base = output_dir / f'window_{test_type}_predicted_vs_actual'
    plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_base}.pdf and .png")
    plt.close()


def plot_mip_time_vs_param(df, output_dir, test_type, param_name):
    """Plot MIP solve time vs parameter."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df[param_name], df['mip_solve_time_s'],
           'o-', linewidth=2, markersize=10,
           color='steelblue', markeredgecolor='black', markeredgewidth=1.5)

    ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=16, fontweight='bold')
    ax.set_ylabel('MIP Solve Time (s)', fontsize=16, fontweight='bold')
    ax.set_title(f'Window {test_type.upper()}: MIP Solve Time vs {param_name.replace("_", " ").title()}',
                fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_base = output_dir / f'window_{test_type}_mip_time'
    plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_base}.pdf and .png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Parse window execution results")
    parser.add_argument('--test', type=str, required=True, choices=['test1', 'test2'],
                       help='Which test to process (test1=distance, test2=time_factor)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same as results)')
    args = parser.parse_args()

    if args.test == 'test1':
        exp_dir = PROJECT_ROOT / "results/ablation/exp3/test1_abstract_window"
        middle_csv = exp_dir / "window_distance_middle_results.csv"
        output_csv = exp_dir / "window_distance_results.csv"
        param_name = 'distance_limit'
    else:
        exp_dir = PROJECT_ROOT / "results/ablation/exp3/test2_time_factor"
        middle_csv = exp_dir / "window_time_factor_middle_results.csv"
        output_csv = exp_dir / "window_time_factor_results.csv"
        param_name = 'time_factor'

    log_file = exp_dir / "execution.log"
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Parsing Window {args.test.upper()} Execution Results")
    print("=" * 60)

    # Parse execution log
    print(f"\n📖 Parsing: {log_file}")
    df_exec = parse_execution_log(log_file, args.test)

    if df_exec.empty:
        print("❌ No results found in log file")
        return

    print(f"✅ Found {len(df_exec)} execution results")

    # Load middle results
    print(f"\n📖 Loading middle results from {middle_csv}")
    df_middle = load_middle_results(middle_csv, args.test)

    # Merge
    df = pd.merge(df_exec, df_middle, on=param_name, how='left')
    df['predicted_time_ms'] = df['predicted_runtime_s'] * 1000

    # Reorder columns
    df = df[[param_name, 'mip_solve_time_s', 'predicted_runtime_s', 'predicted_time_ms',
             'actual_time_ms', 'peak_memory_mib', 'status']]

    print(f"✅ Merged with middle results")

    # Save final results
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(df.to_string(index=False))

    # Calculate errors
    df['error_pct'] = abs(df['predicted_time_ms'] - df['actual_time_ms']) / df['actual_time_ms'] * 100
    print(f"\nPrediction Error: Mean={df['error_pct'].mean():.2f}%, Max={df['error_pct'].max():.2f}%")

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)

    plot_predicted_vs_actual(df, output_dir, args.test, param_name)
    plot_mip_time_vs_param(df, output_dir, args.test, param_name)

    print("\n" + "=" * 60)
    print("✅ Analysis Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
