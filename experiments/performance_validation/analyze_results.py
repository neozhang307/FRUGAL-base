#!/usr/bin/env python3
"""
Analyze performance validation results and generate plots

This script analyzes the data collected by run_validation.py and generates
visualizations for the saturation assumption validation study.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json

class PerformanceAnalyzer:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        # Load results
        csv_path = self.results_dir / "performance_validation_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Results file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} results from {csv_path}")

        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    def calculate_statistics(self):
        """Calculate summary statistics"""
        print("\n=== Summary Statistics ===")

        # Overall stats
        print(f"\nMemory Reduction:")
        print(f"  Mean: {self.df['memory_reduction'].mean():.1f}%")
        print(f"  Std: {self.df['memory_reduction'].std():.1f}%")
        print(f"  Min: {self.df['memory_reduction'].min():.1f}%")
        print(f"  Max: {self.df['memory_reduction'].max():.1f}%")

        print(f"\nPrediction Error:")
        print(f"  Mean: {self.df['prediction_error'].mean():.1f}%")
        print(f"  Std: {self.df['prediction_error'].std():.1f}%")
        print(f"  Min: {self.df['prediction_error'].min():.1f}%")
        print(f"  Max: {self.df['prediction_error'].max():.1f}%")

        print(f"\nActual Slowdown:")
        print(f"  Mean: {self.df['actual_slowdown'].mean():.2f}x")
        print(f"  Std: {self.df['actual_slowdown'].std():.2f}x")
        print(f"  Min: {self.df['actual_slowdown'].min():.2f}x")
        print(f"  Max: {self.df['actual_slowdown'].max():.2f}x")

        # Categorize by domain size
        self.df['category'] = pd.cut(self.df['n'],
                                      bins=[0, 8192, 32768, np.inf],
                                      labels=['Small', 'Transition', 'Saturated'])

        print(f"\nBy Category:")
        for cat in ['Small', 'Transition', 'Saturated']:
            subset = self.df[self.df['category'] == cat]
            if len(subset) > 0:
                print(f"\n{cat}:")
                print(f"  Prediction Error: {subset['prediction_error'].mean():.1f}% ± {subset['prediction_error'].std():.1f}%")
                print(f"  Slowdown: {subset['actual_slowdown'].mean():.2f}x ± {subset['actual_slowdown'].std():.2f}x")

    def plot_prediction_error_vs_domain_size(self):
        """Plot prediction error vs domain size"""
        plt.figure(figsize=(14, 6))

        # Plot 1: Line plot with markers
        plt.subplot(1, 2, 1)
        plt.plot(self.df['n'], self.df['prediction_error'], 'o-', linewidth=2, markersize=8)
        plt.xlabel('Domain Size (N)', fontsize=14)
        plt.ylabel('Prediction Error (%)', fontsize=14)
        plt.title('Prediction Error vs Domain Size', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')

        # Add category regions
        plt.axvspan(0, 8192, alpha=0.1, color='red', label='Not Saturated')
        plt.axvspan(8192, 32768, alpha=0.1, color='yellow', label='Transition')
        plt.axvspan(32768, self.df['n'].max()*1.1, alpha=0.1, color='green', label='Saturated')
        plt.legend()

        # Plot 2: Bar plot by category
        plt.subplot(1, 2, 2)
        category_stats = self.df.groupby('category')['prediction_error'].agg(['mean', 'std'])
        category_stats['mean'].plot(kind='bar', yerr=category_stats['std'], capsize=5)
        plt.xlabel('Category', fontsize=14)
        plt.ylabel('Prediction Error (%)', fontsize=14)
        plt.title('Prediction Error by Category', fontsize=16, fontweight='bold')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'prediction_error_vs_domain_size.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.plots_dir}/prediction_error_vs_domain_size.png")

    def plot_memory_reduction(self):
        """Plot memory reduction rates"""
        plt.figure(figsize=(14, 6))

        # Plot 1: Phase 1 vs Phase 2 memory reduction
        plt.subplot(1, 2, 1)
        x = np.arange(len(self.df))
        width = 0.35
        plt.bar(x - width/2, self.df['memory_reduction'], width, label='Phase 1 (Minimized)', alpha=0.8)
        plt.bar(x + width/2, self.df['memory_reduction_maintained'], width, label='Phase 2 (Maintained)', alpha=0.8)
        plt.xlabel('Domain Size', fontsize=14)
        plt.ylabel('Memory Reduction (%)', fontsize=14)
        plt.title('Memory Reduction: Phase 1 vs Phase 2', fontsize=16, fontweight='bold')
        plt.xticks(x, self.df['n'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        # Plot 2: Memory usage comparison
        plt.subplot(1, 2, 2)
        plt.plot(self.df['n'], self.df['baseline_memory'], 'o-', label='Baseline', linewidth=2, markersize=8)
        plt.plot(self.df['n'], self.df['min_memory'], 's-', label='Minimized', linewidth=2, markersize=8)
        plt.plot(self.df['n'], self.df['final_memory'], '^-', label='Optimized', linewidth=2, markersize=8)
        plt.xlabel('Domain Size (N)', fontsize=14)
        plt.ylabel('Memory Usage (MB)', fontsize=14)
        plt.title('Memory Usage Comparison', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'memory_reduction.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.plots_dir}/memory_reduction.png")

    def plot_slowdown_analysis(self):
        """Plot slowdown analysis"""
        plt.figure(figsize=(14, 6))

        # Plot 1: Predicted vs actual slowdown
        plt.subplot(1, 2, 1)
        plt.scatter(self.df['predicted_slowdown'], self.df['actual_slowdown'], s=100, alpha=0.6)
        for i, n in enumerate(self.df['n']):
            plt.annotate(f'{n}', (self.df['predicted_slowdown'].iloc[i], self.df['actual_slowdown'].iloc[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

        # Add diagonal line (perfect prediction)
        max_val = max(self.df['predicted_slowdown'].max(), self.df['actual_slowdown'].max())
        plt.plot([1, max_val], [1, max_val], 'r--', label='Perfect Prediction', linewidth=2)

        plt.xlabel('Predicted Slowdown (x)', fontsize=14)
        plt.ylabel('Actual Slowdown (x)', fontsize=14)
        plt.title('Predicted vs Actual Slowdown', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Slowdown vs domain size
        plt.subplot(1, 2, 2)
        plt.plot(self.df['n'], self.df['actual_slowdown'], 'o-', linewidth=2, markersize=8)
        plt.xlabel('Domain Size (N)', fontsize=14)
        plt.ylabel('Actual Slowdown (x)', fontsize=14)
        plt.title('Slowdown vs Domain Size', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')

        # Add category regions
        plt.axvspan(0, 8192, alpha=0.1, color='red')
        plt.axvspan(8192, 32768, alpha=0.1, color='yellow')
        plt.axvspan(32768, self.df['n'].max()*1.1, alpha=0.1, color='green')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'slowdown_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.plots_dir}/slowdown_analysis.png")

    def plot_tflops_performance(self):
        """Plot TFLOPS performance"""
        plt.figure(figsize=(14, 6))

        # Plot 1: TFLOPS comparison
        plt.subplot(1, 2, 1)
        x = np.arange(len(self.df))
        width = 0.35
        plt.bar(x - width/2, self.df['baseline_tflops'], width, label='Baseline', alpha=0.8)
        plt.bar(x + width/2, self.df['optimized_tflops'], width, label='Optimized', alpha=0.8)
        plt.xlabel('Domain Size', fontsize=14)
        plt.ylabel('TFLOPS', fontsize=14)
        plt.title('TFLOPS: Baseline vs Optimized', fontsize=16, fontweight='bold')
        plt.xticks(x, self.df['n'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        # Plot 2: TFLOPS vs domain size (log-log)
        plt.subplot(1, 2, 2)
        plt.plot(self.df['n'], self.df['baseline_tflops'], 'o-', label='Baseline', linewidth=2, markersize=8)
        plt.plot(self.df['n'], self.df['optimized_tflops'], 's-', label='Optimized', linewidth=2, markersize=8)
        plt.xlabel('Domain Size (N)', fontsize=14)
        plt.ylabel('TFLOPS', fontsize=14)
        plt.title('TFLOPS vs Domain Size', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'tflops_performance.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.plots_dir}/tflops_performance.png")

    def generate_summary_report(self):
        """Generate markdown summary report"""
        report = []
        report.append("# Performance Model Validation Results\n")
        report.append(f"**Generated**: {pd.Timestamp.now()}\n")
        report.append(f"**Total Experiments**: {len(self.df)}\n\n")

        report.append("## Summary Statistics\n\n")
        report.append("### Memory Reduction\n")
        report.append(f"- Mean: {self.df['memory_reduction'].mean():.1f}%\n")
        report.append(f"- Std: {self.df['memory_reduction'].std():.1f}%\n")
        report.append(f"- Range: [{self.df['memory_reduction'].min():.1f}%, {self.df['memory_reduction'].max():.1f}%]\n\n")

        report.append("### Prediction Error\n")
        report.append(f"- Mean: {self.df['prediction_error'].mean():.1f}%\n")
        report.append(f"- Std: {self.df['prediction_error'].std():.1f}%\n")
        report.append(f"- Range: [{self.df['prediction_error'].min():.1f}%, {self.df['prediction_error'].max():.1f}%]\n\n")

        report.append("### Slowdown\n")
        report.append(f"- Mean: {self.df['actual_slowdown'].mean():.2f}x\n")
        report.append(f"- Std: {self.df['actual_slowdown'].std():.2f}x\n")
        report.append(f"- Range: [{self.df['actual_slowdown'].min():.2f}x, {self.df['actual_slowdown'].max():.2f}x]\n\n")

        report.append("## Results by Category\n\n")
        for cat in ['Small', 'Transition', 'Saturated']:
            subset = self.df[self.df['category'] == cat]
            if len(subset) > 0:
                report.append(f"### {cat} ({len(subset)} experiments)\n")
                report.append(f"- Prediction Error: {subset['prediction_error'].mean():.1f}% ± {subset['prediction_error'].std():.1f}%\n")
                report.append(f"- Slowdown: {subset['actual_slowdown'].mean():.2f}x ± {subset['actual_slowdown'].std():.2f}x\n")
                report.append(f"- Memory Reduction: {subset['memory_reduction'].mean():.1f}% ± {subset['memory_reduction'].std():.1f}%\n\n")

        report.append("## Detailed Results\n\n")
        report.append(self.df.to_markdown(index=False))

        # Save report
        report_path = self.results_dir / "SUMMARY_REPORT.md"
        with open(report_path, 'w') as f:
            f.writelines(report)

        print(f"✅ Saved: {report_path}")

    def run_all_analyses(self):
        """Run all analyses and generate plots"""
        print("\n🔬 Starting Analysis...")

        # Calculate statistics
        self.calculate_statistics()

        # Generate plots
        print("\n📊 Generating plots...")
        self.plot_prediction_error_vs_domain_size()
        self.plot_memory_reduction()
        self.plot_slowdown_analysis()
        self.plot_tflops_performance()

        # Generate report
        print("\n📝 Generating summary report...")
        self.generate_summary_report()

        print(f"\n✅ Analysis complete! Results in: {self.plots_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze performance validation results")
    parser.add_argument('--input-dir', type=str, default="results/saturation_study",
                        help="Directory containing results")

    args = parser.parse_args()

    # Create analyzer
    analyzer = PerformanceAnalyzer(results_dir=args.input_dir)

    # Run analyses
    analyzer.run_all_analyses()

if __name__ == "__main__":
    main()
