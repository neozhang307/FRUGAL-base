#!/usr/bin/env python3
"""
Analyze beam width ablation study results and generate plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

class BeamWidthAnalyzer:
    def __init__(self, csv_path: str, output_dir: str):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} beam width configurations")

        # Set publication style
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        plt.rcParams['legend.fontsize'] = 16
        plt.rcParams['figure.figsize'] = (12, 7)
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

    def plot_score_vs_beam_width(self):
        """Plot: Beam width vs task scheduling score"""
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(self.df['beam_width'], self.df['score_gb'],
               'o-', linewidth=3, markersize=12,
               color='#5A7FA5', markeredgecolor='black', markeredgewidth=1.5)

        ax.set_xlabel('Beam Width', fontweight='bold', fontsize=20)
        ax.set_ylabel('Task Scheduling Score (GB)', fontweight='bold', fontsize=20)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add horizontal line at max score
        max_score = self.df['score_gb'].max()
        ax.axhline(y=max_score, color='red', linestyle='--', linewidth=2, alpha=0.7,
                  label=f'Max Score: {max_score:.2f} GB')

        # Annotate convergence point (where score stops improving significantly)
        score_diff = self.df['score_gb'].diff()
        if len(score_diff) > 1:
            # Find where improvement is < 1%
            pct_improvement = (score_diff / self.df['score_gb'].shift(1) * 100).abs()
            converge_idx = None
            for i in range(1, len(pct_improvement)):
                if pct_improvement.iloc[i] < 0.5:  # Less than 0.5% improvement
                    converge_idx = i
                    break

            if converge_idx:
                converge_width = self.df.iloc[converge_idx]['beam_width']
                converge_score = self.df.iloc[converge_idx]['score_gb']
                ax.axvline(x=converge_width, color='orange', linestyle=':', linewidth=2,
                          alpha=0.7, label=f'Convergence: Beam={converge_width}')

        ax.legend(loc='lower right', framealpha=0.95, edgecolor='black')

        # Statistics
        min_score = self.df['score_gb'].min()
        improvement = ((max_score - min_score) / min_score) * 100
        stats_text = (
            f'Score Range:\n'
            f'Min: {min_score:.2f} GB (beam=1)\n'
            f'Max: {max_score:.2f} GB (beam={self.df.loc[self.df["score_gb"].idxmax(), "beam_width"]:.0f})\n'
            f'Improvement: {improvement:.1f}%'
        )
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes, ha='left', va='top',
               fontsize=14, family='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))

        plt.tight_layout()
        output_base = self.output_dir / 'beam_width_vs_score'
        plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_base}.pdf and .png")
        plt.close()

    def plot_solve_time_vs_beam_width(self):
        """Plot: Beam width vs first-step solve time"""
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(self.df['beam_width'], self.df['solve_time_ms'],
               's-', linewidth=3, markersize=12,
               color='#D4825C', markeredgecolor='black', markeredgewidth=1.5)

        ax.set_xlabel('Beam Width', fontweight='bold', fontsize=20)
        ax.set_ylabel('First-Step Solve Time (ms)', fontweight='bold', fontsize=20)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add trend line
        z = np.polyfit(self.df['beam_width'], self.df['solve_time_ms'], 1)
        p = np.poly1d(z)
        ax.plot(self.df['beam_width'], p(self.df['beam_width']),
               "--", linewidth=2, color='red', alpha=0.7, label='Linear Trend')

        ax.legend(loc='upper left', framealpha=0.95, edgecolor='black')

        # Statistics
        min_time = self.df['solve_time_ms'].min()
        max_time = self.df['solve_time_ms'].max()
        stats_text = (
            f'Solve Time Range:\n'
            f'Min: {min_time:.2f} ms (beam={self.df.loc[self.df["solve_time_ms"].idxmin(), "beam_width"]:.0f})\n'
            f'Max: {max_time:.2f} ms (beam={self.df.loc[self.df["solve_time_ms"].idxmax(), "beam_width"]:.0f})\n'
            f'Overhead: {max_time - min_time:.2f} ms'
        )
        ax.text(0.98, 0.02, stats_text,
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=14, family='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))

        plt.tight_layout()
        output_base = self.output_dir / 'beam_width_vs_solve_time'
        plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_base}.pdf and .png")
        plt.close()

    def plot_efficiency_frontier(self):
        """Plot: Score vs solve time (Pareto frontier)"""
        fig, ax = plt.subplots(figsize=(12, 7))

        # Color points by beam width
        scatter = ax.scatter(self.df['solve_time_ms'], self.df['score_gb'],
                           c=self.df['beam_width'], cmap='viridis',
                           s=250, alpha=0.8,
                           edgecolors='black', linewidth=1.5)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Beam Width', fontweight='bold', fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        # Annotate key points
        for beam in [1, 10, 50, 100]:
            row = self.df[self.df['beam_width'] == beam]
            if not row.empty:
                ax.annotate(f'Beam={beam}',
                           (row['solve_time_ms'].values[0], row['score_gb'].values[0]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', lw=1.5))

        ax.set_xlabel('First-Step Solve Time (ms)', fontweight='bold', fontsize=20)
        ax.set_ylabel('Task Scheduling Score (GB)', fontweight='bold', fontsize=20)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_base = self.output_dir / 'beam_width_efficiency_frontier'
        plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_base}.pdf and .png")
        plt.close()

    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*70)
        print("BEAM WIDTH ABLATION STUDY - SUMMARY")
        print("="*70)

        print("\n📊 Score Analysis:")
        print(f"  Min Score: {self.df['score_gb'].min():.2f} GB (beam={self.df.loc[self.df['score_gb'].idxmin(), 'beam_width']:.0f})")
        print(f"  Max Score: {self.df['score_gb'].max():.2f} GB (beam={self.df.loc[self.df['score_gb'].idxmax(), 'beam_width']:.0f})")
        improvement = ((self.df['score_gb'].max() - self.df['score_gb'].min()) / self.df['score_gb'].min()) * 100
        print(f"  Improvement: {improvement:.2f}%")

        print("\n⏱️  Solve Time Analysis:")
        print(f"  Min Time: {self.df['solve_time_ms'].min():.3f} ms (beam={self.df.loc[self.df['solve_time_ms'].idxmin(), 'beam_width']:.0f})")
        print(f"  Max Time: {self.df['solve_time_ms'].max():.3f} ms (beam={self.df.loc[self.df['solve_time_ms'].idxmax(), 'beam_width']:.0f})")
        print(f"  Overhead: {self.df['solve_time_ms'].max() - self.df['solve_time_ms'].min():.3f} ms")

        print("\n🎯 Memory Constraint Achievement:")
        optimal_count = len(self.df[self.df['status'] == 'OPTIMAL'])
        print(f"  Optimal solutions: {optimal_count}/{len(self.df)} ({100*optimal_count/len(self.df):.1f}%)")

        print("\n📈 Detailed Results:")
        print("  Beam  Score(GB)  Solve(ms)  Predicted(s)  Peak(MiB)  Status")
        for _, row in self.df.iterrows():
            print(f"  {row['beam_width']:4.0f}  {row['score_gb']:8.2f}  {row['solve_time_ms']:9.3f}  "
                  f"{row['predicted_runtime_s']:11.2f}  {row['peak_memory_mib']:9.0f}  {row['status']}")

        print("\n" + "="*70)

    def generate_all_plots(self):
        """Generate all plots"""
        print("\n📊 Generating beam width ablation plots...\n")

        self.plot_score_vs_beam_width()
        self.plot_solve_time_vs_beam_width()
        self.plot_efficiency_frontier()

        print(f"\n✅ All plots saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze beam width ablation study")
    parser.add_argument('--csv', type=str,
                       default='results/ablation/exp2/beam_width_results.csv',
                       help='Path to results CSV')
    parser.add_argument('--output-dir', type=str,
                       default='results/ablation/exp2/plots',
                       help='Directory to save plots')

    args = parser.parse_args()

    analyzer = BeamWidthAnalyzer(args.csv, args.output_dir)
    analyzer.generate_all_plots()
    analyzer.print_summary()

if __name__ == '__main__':
    main()
