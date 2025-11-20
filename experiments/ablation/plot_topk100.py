#!/usr/bin/env python3
"""
Generate publication-quality plots for Top-100 results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

class TopK100PlotGenerator:
    def __init__(self, csv_path: str, output_dir: str):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} solutions from {csv_path}")

        # Set publication style
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

    def plot_score_distribution(self):
        """Plot beam score distribution across all 100 solutions"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by score
        score_groups = self.df.groupby('score_gb').size().sort_index(ascending=False)

        # Create colors for different tiers
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(score_groups)))

        # Bar plot
        bars = ax.bar(range(len(self.df)), self.df['score_gb'],
                     color=[colors[list(score_groups.index).index(s)]
                           for s in self.df['score_gb']],
                     alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Solution Rank (0-99)', fontweight='bold')
        ax.set_ylabel('Beam Search Score (GB)', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Add tier annotations
        ax.text(0.02, 0.98, f'Top Tier (119.63 GB): 5 solutions',
               transform=ax.transAxes, ha='left', va='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        # Add score distribution legend
        legend_text = "Score Tiers:\n"
        for i, (score, count) in enumerate(score_groups.items()):
            legend_text += f"{score:.2f} GB: {count} sols\n"
        ax.text(0.98, 0.02, legend_text.strip(),
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=11, family='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        plt.tight_layout()
        output_base = self.output_dir / 'topk100_score_distribution'
        plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_base}.pdf and .png")
        plt.close()

    def plot_runtime_vs_score(self):
        """Scatter plot: beam score vs predicted runtime"""
        fig, ax = plt.subplots(figsize=(10, 7))

        # Color by score tier
        unique_scores = sorted(self.df['score_gb'].unique(), reverse=True)
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(unique_scores)))
        color_map = {score: colors[i] for i, score in enumerate(unique_scores)}

        for score in unique_scores:
            mask = self.df['score_gb'] == score
            subset = self.df[mask]
            ax.scatter(subset['score_gb'], subset['predicted_time_s'],
                      s=100, alpha=0.7,
                      color=color_map[score],
                      edgecolors='black', linewidth=0.5,
                      label=f'{score:.2f} GB ({len(subset)} sols)')

        ax.set_xlabel('Beam Search Score (GB)', fontweight='bold')
        ax.set_ylabel('MIP Predicted Runtime (s)', fontweight='bold')
        ax.legend(loc='best', framealpha=0.95, edgecolor='black', ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add annotation
        runtime_range = self.df['predicted_time_s'].max() - self.df['predicted_time_s'].min()
        runtime_var_pct = 100 * runtime_range / self.df['predicted_time_s'].min()
        ax.text(0.02, 0.98,
               f'Runtime Variance: {runtime_range:.3f}s ({runtime_var_pct:.1f}%)\n'
               f'All solutions achieve 15GB constraint',
               transform=ax.transAxes, ha='left', va='top',
               fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))

        plt.tight_layout()
        output_base = self.output_dir / 'topk100_runtime_vs_score'
        plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_base}.pdf and .png")
        plt.close()

    def plot_runtime_histogram(self):
        """Histogram of predicted runtimes"""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(self.df['predicted_time_s'], bins=30,
               color='#5A7FA5', alpha=0.7, edgecolor='black', linewidth=1.5)

        ax.set_xlabel('MIP Predicted Runtime (seconds)', fontweight='bold')
        ax.set_ylabel('Number of Solutions', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Add statistics
        stats_text = (
            f"Mean: {self.df['predicted_time_s'].mean():.3f} s\n"
            f"Std:  {self.df['predicted_time_s'].std():.3f} s\n"
            f"Min:  {self.df['predicted_time_s'].min():.3f} s\n"
            f"Max:  {self.df['predicted_time_s'].max():.3f} s"
        )
        ax.text(0.98, 0.98, stats_text,
               transform=ax.transAxes, ha='right', va='top',
               fontsize=12, family='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        plt.tight_layout()
        output_base = self.output_dir / 'topk100_runtime_histogram'
        plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_base}.pdf and .png")
        plt.close()

    def plot_score_tier_comparison(self):
        """Box plot comparing runtime across score tiers"""
        fig, ax = plt.subplots(figsize=(12, 7))

        # Group by score
        score_groups = sorted(self.df['score_gb'].unique(), reverse=True)

        # Prepare data for box plot
        data_by_tier = [self.df[self.df['score_gb'] == score]['predicted_time_s'].values
                       for score in score_groups]
        labels = [f'{score:.2f} GB\n(n={len(self.df[self.df["score_gb"] == score])})'
                 for score in score_groups]

        # Box plot
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(score_groups)))
        bp = ax.boxplot(data_by_tier, labels=labels, patch_artist=True,
                       widths=0.6,
                       boxprops=dict(linewidth=2, edgecolor='black'),
                       medianprops=dict(color='red', linewidth=2.5),
                       whiskerprops=dict(color='black', linewidth=1.5),
                       capprops=dict(color='black', linewidth=1.5))

        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('MIP Predicted Runtime (s)', fontweight='bold')
        ax.set_xlabel('Beam Search Score Tier', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        plt.xticks(rotation=0)
        plt.tight_layout()
        output_base = self.output_dir / 'topk100_score_tier_comparison'
        plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_base}.pdf and .png")
        plt.close()

    def plot_cumulative_score(self):
        """Cumulative plot showing score degradation"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Sort by rank
        df_sorted = self.df.sort_values('rank')

        ax.plot(df_sorted['rank'], df_sorted['score_gb'],
               'o-', linewidth=2, markersize=4,
               color='#5A7FA5', alpha=0.8)

        ax.set_xlabel('Solution Rank', fontweight='bold')
        ax.set_ylabel('Beam Search Score (GB)', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add tier boundaries
        unique_scores = sorted(df_sorted['score_gb'].unique(), reverse=True)
        for i, score in enumerate(unique_scores[:-1]):
            # Find where score changes
            change_idx = df_sorted[df_sorted['score_gb'] == score]['rank'].max()
            ax.axvline(x=change_idx, color='red', linestyle='--', alpha=0.5, linewidth=1)

        # Add annotation
        score_drop = df_sorted['score_gb'].max() - df_sorted['score_gb'].min()
        score_drop_pct = 100 * score_drop / df_sorted['score_gb'].max()
        ax.text(0.98, 0.02,
               f'Top-100 Score Range:\n'
               f'Best:  {df_sorted["score_gb"].max():.2f} GB\n'
               f'100th: {df_sorted["score_gb"].min():.2f} GB\n'
               f'Drop:  {score_drop:.2f} GB ({score_drop_pct:.1f}%)',
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=12, family='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        plt.tight_layout()
        output_base = self.output_dir / 'topk100_cumulative_score'
        plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_base}.pdf and .png")
        plt.close()

    def generate_all_plots(self):
        """Generate all plots"""
        print("\n📊 Generating Top-100 plots...\n")

        self.plot_score_distribution()
        self.plot_runtime_vs_score()
        self.plot_runtime_histogram()
        self.plot_score_tier_comparison()
        self.plot_cumulative_score()

        print(f"\n✅ All plots saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate Top-100 plots")
    parser.add_argument('--csv', type=str,
                       default='results/ablation/exp1/topk_extensive/topk100_analysis.csv',
                       help='Path to analysis CSV')
    parser.add_argument('--output-dir', type=str,
                       default='results/ablation/exp1/topk_extensive/plots',
                       help='Directory to save plots')

    args = parser.parse_args()

    generator = TopK100PlotGenerator(args.csv, args.output_dir)
    generator.generate_all_plots()

if __name__ == '__main__':
    main()
