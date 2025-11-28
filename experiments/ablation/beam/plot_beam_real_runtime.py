#!/usr/bin/env python3
"""
Parse beam width execution results and generate plots showing:
1. MIP prediction accuracy across different beam widths
2. Actual runtime vs beam width (to show beam width doesn't affect final performance)
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path("/home/users/lingqi.zhang/workspace_x64/FRUGAL/BASE/optimize-cuda-memory-usage-v1")
EXP_DIR = PROJECT_ROOT / "results/ablation/exp2"
LOG_FILE = EXP_DIR / "beam_execution_results.log"
OUTPUT_CSV = EXP_DIR / "beam_results.csv"
PLOT_DIR = EXP_DIR / "plots"

# Create output directory
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def parse_execution_log(log_file):
    """Parse execution log to extract beam width and actual time."""
    results = []

    with open(log_file, 'r') as f:
        content = f.read()

    # Split by beam width sections
    sections = re.split(r'=== Beam Width: (\d+) ===', content)

    for i in range(1, len(sections), 2):
        beam_width = int(sections[i])
        section_content = sections[i + 1]

        # Extract actual execution time
        actual_match = re.search(r'Execution time: ([\d.]+) ms', section_content)
        if not actual_match:
            print(f"Warning: Could not find actual runtime for beam={beam_width}")
            continue
        actual_time_ms = float(actual_match.group(1))

        results.append({
            'beam_width': beam_width,
            'actual_time_ms': actual_time_ms
        })

    return pd.DataFrame(results)

def load_predicted_times():
    """Load predicted runtimes and beam search time from beam_middle_results.csv."""
    beam_results_csv = EXP_DIR / "beam_middle_results.csv"
    df = pd.read_csv(beam_results_csv)
    return df[['beam_width', 'predicted_runtime_s', 'solve_time_ms', 'score_gb']].copy()

def plot_mip_prediction_by_beam(df, output_dir):
    """Plot MIP predicted vs actual runtime, colored by beam width."""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create scatter plot with beam width as color
    scatter = ax.scatter(df['predicted_time_ms'], df['actual_time_ms'],
                        c=df['beam_width'], cmap='viridis',
                        s=300, alpha=0.8,
                        edgecolors='black', linewidth=1.5)

    # Add beam width labels to each point
    for idx, row in df.iterrows():
        ax.annotate(f"Beam={row['beam_width']}",
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
    max_error = df['error_pct'].max()

    # Calculate runtime statistics
    runtime_mean = df['actual_time_ms'].mean()
    runtime_std = df['actual_time_ms'].std()
    runtime_cv = (runtime_std / runtime_mean) * 100  # Coefficient of variation

    # Add statistics text box
    stats_text = (
        f'Prediction Accuracy:\n'
        f'Avg Error: {avg_error:.2f}%\n'
        f'Max Error: {max_error:.2f}%\n'
        f'\n'
        f'Actual Runtime:\n'
        f'Mean: {runtime_mean:.1f} ms\n'
        f'Std Dev: {runtime_std:.1f} ms\n'
        f'CV: {runtime_cv:.2f}%\n'
        f'\n'
        f'Beam Widths: {len(df)} tested'
    )
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes, ha='left', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=14, family='monospace', fontweight='bold')

    # Labels and formatting
    ax.set_xlabel('MIP Predicted Runtime (ms)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Actual GPU Runtime (ms)', fontsize=18, fontweight='bold')
    ax.set_title('MIP Prediction Accuracy Across Beam Widths',
                fontsize=20, fontweight='bold', pad=20)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=14, loc='lower right')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Beam Width', fontweight='bold', fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save
    pdf_path = output_dir / 'beam_mip_prediction_accuracy.pdf'
    png_path = output_dir / 'beam_mip_prediction_accuracy.png'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {pdf_path}")
    print(f"✅ Saved: {png_path}")
    plt.close()

def plot_runtime_vs_beam(df, output_dir):
    """Plot actual runtime vs beam width to show beam doesn't affect final performance."""

    fig, ax = plt.subplots(figsize=(10, 7))

    # Bar plot
    bars = ax.bar(df['beam_width'], df['actual_time_ms'],
                  color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add mean line
    mean_runtime = df['actual_time_ms'].mean()
    ax.axhline(mean_runtime, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {mean_runtime:.1f} ms')

    # Add std dev band
    std_runtime = df['actual_time_ms'].std()
    ax.axhspan(mean_runtime - std_runtime, mean_runtime + std_runtime,
              alpha=0.2, color='red', label=f'±1 Std Dev: {std_runtime:.1f} ms')

    # Labels and formatting
    ax.set_xlabel('Beam Width', fontsize=16, fontweight='bold')
    ax.set_ylabel('Actual GPU Runtime (ms)', fontsize=16, fontweight='bold')
    ax.set_title('Beam Width Independence: Final Runtime Not Affected by Beam Width',
                fontsize=18, fontweight='bold', pad=20)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(fontsize=12, loc='upper right')

    # Set x-axis ticks
    ax.set_xticks(df['beam_width'])

    plt.tight_layout()

    # Save
    pdf_path = output_dir / 'beam_runtime_independence.pdf'
    png_path = output_dir / 'beam_runtime_independence.png'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {pdf_path}")
    print(f"✅ Saved: {png_path}")
    plt.close()

def main():
    print("="*60)
    print("Parsing Beam Width Execution Results")
    print("="*60)

    # Parse log file
    print(f"\n📖 Parsing: {LOG_FILE}")
    df_exec = parse_execution_log(LOG_FILE)

    if df_exec.empty:
        print("❌ No results found in log file")
        return

    print(f"✅ Found {len(df_exec)} beam width execution results")

    # Load predicted times from beam_middle_results.csv
    print(f"\n📖 Loading predicted times from beam_middle_results.csv")
    df_pred = load_predicted_times()

    # Merge execution results with predictions
    df = pd.merge(df_exec, df_pred, on='beam_width', how='left')
    df['predicted_time_ms'] = df['predicted_runtime_s'] * 1000  # Convert to ms
    df['beam_search_time_ms'] = df['solve_time_ms']  # Rename for clarity
    df = df[['beam_width', 'score_gb', 'beam_search_time_ms', 'predicted_time_ms', 'actual_time_ms']]  # Reorder columns

    print(f"✅ Merged with predicted runtimes and beam search times")

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved: {OUTPUT_CSV}")

    # Display summary
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(df.to_string(index=False))

    print("\nRuntime Statistics:")
    print(f"  Mean: {df['actual_time_ms'].mean():.2f} ms")
    print(f"  Std Dev: {df['actual_time_ms'].std():.2f} ms")
    print(f"  Min: {df['actual_time_ms'].min():.2f} ms")
    print(f"  Max: {df['actual_time_ms'].max():.2f} ms")
    print(f"  Range: {df['actual_time_ms'].max() - df['actual_time_ms'].min():.2f} ms")

    print("\nPrediction Error Statistics:")
    df['error_pct'] = abs(df['predicted_time_ms'] - df['actual_time_ms']) / df['actual_time_ms'] * 100
    print(f"  Mean Error: {df['error_pct'].mean():.2f}%")
    print(f"  Max Error: {df['error_pct'].max():.2f}%")

    # Generate plots
    print("\n" + "="*60)
    print("Generating Plots")
    print("="*60)

    plot_mip_prediction_by_beam(df, PLOT_DIR)
    plot_runtime_vs_beam(df, PLOT_DIR)

    print("\n" + "="*60)
    print("✅ Analysis Complete")
    print("="*60)

if __name__ == '__main__':
    main()
