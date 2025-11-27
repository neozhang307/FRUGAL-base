#!/usr/bin/env python3
"""
Generate publication-quality plots for paper
Separate plots without titles, PDF format, better labeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

class PaperPlotGenerator:
    def __init__(self, results_dir: str, output_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        csv_path = self.results_dir / "performance_validation_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Results file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} results from {csv_path}")

        # Set publication style
        sns.set_style("whitegrid")
        # Use larger fonts for readability in papers
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['figure.figsize'] = (8, 6)

        # Use PDF-friendly backend
        plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for editability
        plt.rcParams['ps.fonttype'] = 42

        # Black axes and labels
        plt.rcParams['axes.labelcolor'] = 'black'
        plt.rcParams['axes.edgecolor'] = 'black'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'
        plt.rcParams['text.color'] = 'black'
        plt.rcParams['legend.edgecolor'] = 'black'

    def format_domain_size(self, n):
        """Format domain size for better readability

        Options:
        1. "1024×1024" - explicit 2D notation
        2. "1K×1K" - short with K/M notation
        3. "N=1024" - mathematical notation
        4. "1024²" - superscript notation
        """
        if n >= 1000:
            # Use K/M notation with × for clarity
            k = n // 1024
            if k * 1024 == n:
                return f"{k}K×{k}K"
            else:
                return f"{n}×{n}"
        else:
            return f"{n}×{n}"

    def plot_slowdown_scatter(self):
        """Plot predicted vs actual slowdown (scatter plot only)"""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter plot with larger markers - using less saturated blue
        scatter = ax.scatter(self.df['predicted_slowdown'],
                            self.df['actual_slowdown'],
                            s=150, alpha=0.8, c='#5A7FA5', edgecolors='black', linewidth=1.5)

        # Annotate each point with formatted domain size
        for i, n in enumerate(self.df['n']):
            label = self.format_domain_size(n)
            ax.annotate(label,
                       (self.df['predicted_slowdown'].iloc[i],
                        self.df['actual_slowdown'].iloc[i]),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=12, ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='gray', alpha=0.7))

        # Add diagonal line (perfect prediction) - using darker red
        max_val = max(self.df['predicted_slowdown'].max(), self.df['actual_slowdown'].max()) * 1.05
        min_val = min(self.df['predicted_slowdown'].min(), self.df['actual_slowdown'].min()) * 0.95
        ax.plot([min_val, max_val], [min_val, max_val], '--',
               color='#B85450', label='Perfect Prediction', linewidth=2.5, alpha=0.9)

        ax.set_xlabel('Predicted Slowdown', fontweight='bold', color='black')
        ax.set_ylabel('Actual Slowdown', fontweight='bold', color='black')
        ax.legend(loc='upper left', framealpha=0.95, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--', color='gray')

        # Set equal aspect ratio for better visual comparison
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()

        # Save as both PNG and PDF
        output_base = self.output_dir / 'slowdown_scatter'
        plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_base}.pdf and .png")
        plt.close()

    def plot_slowdown_vs_domain_size(self):
        """Plot slowdown vs domain size (line plot only)"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot predicted and actual slowdown - using less saturated colors
        ax.plot(self.df['n'], self.df['predicted_slowdown'],
               'o--', linewidth=2.5, markersize=10,
               label='Predicted', color='#D4825C', alpha=0.9)
        ax.plot(self.df['n'], self.df['actual_slowdown'],
               's-', linewidth=2.5, markersize=10,
               label='Actual', color='#5A7FA5', alpha=0.9)

        ax.set_xlabel('Matrix Dimension (N)', fontweight='bold', color='black')
        ax.set_ylabel('Slowdown Factor', fontweight='bold', color='black')
        ax.legend(loc='upper right', framealpha=0.95, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--', color='gray')
        ax.set_xscale('log')

        # Better x-axis labels with formatted domain sizes
        xticks = self.df['n'].values
        xticklabels = [self.format_domain_size(n) for n in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, ha='right')

        # Add horizontal line at 1.0 (no slowdown)
        ax.axhline(y=1.0, color='#555555', linestyle=':', linewidth=2, alpha=0.6, label='Baseline')

        plt.tight_layout()

        output_base = self.output_dir / 'slowdown_vs_size'
        plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_base}.pdf and .png")
        plt.close()

    def plot_memory_reduction_vs_domain_size(self):
        """Plot memory reduction vs domain size"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot both managed and peak memory reduction - using less saturated colors
        ax.plot(self.df['n'], self.df['managed_reduction'],
               'o-', linewidth=2.5, markersize=10,
               label='Managed Memory', color='#5B8A72', alpha=0.9)
        ax.plot(self.df['n'], self.df['peak_reduction'],
               's-', linewidth=2.5, markersize=10,
               label='Peak Memory', color='#D4825C', alpha=0.9)

        ax.set_xlabel('Matrix Dimension (N)', fontweight='bold', color='black')
        ax.set_ylabel('Memory Reduction (%)', fontweight='bold', color='black')
        ax.legend(loc='lower right', framealpha=0.95, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--', color='gray')
        ax.set_xscale('log')

        # Better x-axis labels
        xticks = self.df['n'].values
        xticklabels = [self.format_domain_size(n) for n in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, ha='right')

        # Add horizontal line at 0 (no reduction)
        ax.axhline(y=0, color='#555555', linestyle=':', linewidth=2, alpha=0.6)

        plt.tight_layout()

        output_base = self.output_dir / 'memory_reduction_vs_size'
        plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_base}.pdf and .png")
        plt.close()

    def plot_prediction_error_vs_domain_size(self):
        """Plot prediction error vs domain size"""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.df['n'], self.df['prediction_error'],
               'o-', linewidth=2.5, markersize=10,
               color='#B85450', alpha=0.9)

        ax.set_xlabel('Matrix Dimension (N)', fontweight='bold', color='black')
        ax.set_ylabel('Prediction Error (%)', fontweight='bold', color='black')
        ax.grid(True, alpha=0.3, linestyle='--', color='gray')
        ax.set_xscale('log')

        # Better x-axis labels
        xticks = self.df['n'].values
        xticklabels = [self.format_domain_size(n) for n in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, ha='right')

        plt.tight_layout()

        output_base = self.output_dir / 'prediction_error_vs_size'
        plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_base}.pdf and .png")
        plt.close()

    def plot_tflops_comparison(self):
        """Plot TFLOPS comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.df['n'], self.df['baseline_tflops'],
               'o-', linewidth=2.5, markersize=10,
               label='Baseline', color='#4A5F7F', alpha=0.9)
        ax.plot(self.df['n'], self.df['optimized_tflops'],
               's-', linewidth=2.5, markersize=10,
               label='Optimized', color='#5B8A8A', alpha=0.9)

        ax.set_xlabel('Matrix Dimension (N)', fontweight='bold', color='black')
        ax.set_ylabel('TFLOPS', fontweight='bold', color='black')
        ax.legend(loc='lower right', framealpha=0.95, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--', color='gray')
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Better x-axis labels
        xticks = self.df['n'].values
        xticklabels = [self.format_domain_size(n) for n in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, ha='right')

        plt.tight_layout()

        output_base = self.output_dir / 'tflops_comparison'
        plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_base}.pdf and .png")
        plt.close()

    def generate_all_plots(self):
        """Generate all publication-quality plots"""
        print("\n📊 Generating publication-quality plots...\n")

        print("Format options for domain size labels:")
        print("  Current: Using 'NxN' notation (e.g., '1K×1K' for 1024)")
        print("  You can modify format_domain_size() to change this\n")

        self.plot_slowdown_scatter()
        self.plot_slowdown_vs_domain_size()
        self.plot_memory_reduction_vs_domain_size()
        self.plot_prediction_error_vs_domain_size()
        self.plot_tflops_comparison()

        print(f"\n✅ All plots saved to: {self.output_dir}")
        print(f"   Both PDF (vector) and PNG (raster) formats generated")

def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality plots")
    parser.add_argument('--input-dir', type=str, default="results/saturation_study",
                        help="Directory containing validation results")
    parser.add_argument('--output-dir', type=str, default="results/saturation_study/paper_plots",
                        help="Directory to save publication plots")

    args = parser.parse_args()

    generator = PaperPlotGenerator(
        results_dir=args.input_dir,
        output_dir=args.output_dir
    )

    generator.generate_all_plots()

if __name__ == "__main__":
    main()
