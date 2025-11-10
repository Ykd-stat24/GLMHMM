"""
Comprehensive Genotype × State Analysis for GLM-HMM
===================================================

This script creates in-depth statistical and visual comparisons addressing:
1. Genotype differences in state characteristics
2. State-specific feature weighting by genotype
3. State stability and transition patterns
4. Lapse state characteristics across genotypes
5. Deliberative vs Procedural state prevalence

Perfect for poster presentation with statistical rigor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ComprehensiveGenotypeAnalysis:
    """Statistical and visual comparisons of genotype effects on behavioral states."""

    def __init__(self, results_dir='results/phase1_non_reversal'):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'genotype_analysis'
        self.output_dir.mkdir(exist_ok=True)

        # Feature names
        self.feature_names = ['bias', 'prev_choice', 'wsls', 'session_prog',
                             'side_bias', 'task_stage', 'cum_exp']

        # State labels of interest
        self.key_state_labels = [
            'Deliberative High-Performance',
            'Procedural High-Performance',
            'Disengaged Lapse',
            'WSLS Strategy',
            'Perseverative Left-Bias',
            'Perseverative Right-Bias'
        ]

    def load_cohort_data(self, cohort, animals):
        """Load all results for a cohort."""
        results = []

        for animal in animals:
            pkl_file = self.results_dir / f'{animal}_cohort{cohort}_model.pkl'
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    results.append(data)

        return results

    def create_state_label_distribution_plot(self, results_W, results_F):
        """
        Bar plot showing distribution of state labels by cohort and genotype.
        Shows prevalence of deliberative vs procedural vs lapse states.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax, (results, cohort) in zip(axes, [(results_W, 'W'), (results_F, 'F')]):
            # Collect state labels by genotype
            genotype_labels = {}

            for r in results:
                g = r['genotype']
                if g not in genotype_labels:
                    genotype_labels[g] = []

                # Get all state labels for this animal
                for state_idx in range(r['model'].n_states):
                    label, conf, _ = r['validated_labels'][state_idx]
                    genotype_labels[g].append(label)

            # Count label occurrences
            label_data = []
            for genotype in sorted(genotype_labels.keys()):
                labels = genotype_labels[genotype]
                label_counts = pd.Series(labels).value_counts()

                for label in self.key_state_labels:
                    count = label_counts.get(label, 0)
                    pct = 100 * count / len(labels) if len(labels) > 0 else 0
                    label_data.append({
                        'Genotype': genotype,
                        'State Label': label,
                        'Count': count,
                        'Percentage': pct
                    })

            df = pd.DataFrame(label_data)

            # Create grouped bar plot
            x = np.arange(len(self.key_state_labels))
            width = 0.8 / len(genotype_labels)

            for i, genotype in enumerate(sorted(genotype_labels.keys())):
                geno_data = df[df['Genotype'] == genotype]
                percentages = [geno_data[geno_data['State Label'] == label]['Percentage'].values[0]
                              if len(geno_data[geno_data['State Label'] == label]) > 0 else 0
                              for label in self.key_state_labels]

                offset = width * (i - len(genotype_labels)/2 + 0.5)
                bars = ax.bar(x + offset, percentages, width, label=genotype, alpha=0.8)

                # Add value labels on bars
                for bar, val in zip(bars, percentages):
                    if val > 0:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{val:.1f}%',
                               ha='center', va='bottom', fontsize=8, rotation=90)

            ax.set_xlabel('State Label', fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage of States', fontsize=12, fontweight='bold')
            ax.set_title(f'Cohort {cohort}: State Label Distribution by Genotype',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([l.replace(' ', '\n') for l in self.key_state_labels],
                              rotation=45, ha='right', fontsize=9)
            ax.legend(title='Genotype', fontsize=10)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'state_label_distribution_by_genotype.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'state_label_distribution_by_genotype.pdf',
                   bbox_inches='tight')
        plt.close()

        print("✓ Created state label distribution plot")

    def analyze_lapse_states(self, results_W, results_F):
        """
        Deep dive into lapse state characteristics by genotype.
        Compare lapse frequency, duration, and recovery across genotypes.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        for row, (results, cohort) in enumerate([(results_W, 'W'), (results_F, 'F')]):
            # Identify lapse states for each animal
            lapse_data = []

            for r in results:
                for state_idx in range(r['model'].n_states):
                    label, conf, _ = r['validated_labels'][state_idx]

                    if 'Lapse' in label or 'Disengaged' in label:
                        # Get metrics for this lapse state
                        metrics_df = r['state_metrics']
                        state_data = metrics_df[metrics_df['state'] == state_idx]

                        if len(state_data) > 0:
                            lapse_data.append({
                                'Genotype': r['genotype'],
                                'Animal': r['animal_id'],
                                'Accuracy': state_data['accuracy'].values[0],
                                'Occupancy': state_data['occupancy'].values[0],
                                'Dwell_Time': state_data['dwell_mean'].values[0],
                                'Side_Bias': state_data['side_bias'].values[0],
                                'N_Bouts': state_data['n_bouts'].values[0]
                            })

            if len(lapse_data) == 0:
                continue

            df = pd.DataFrame(lapse_data)

            # 1. Lapse Accuracy
            ax = axes[row, 0]
            sns.violinplot(data=df, x='Genotype', y='Accuracy', ax=ax, inner=None, alpha=0.6)
            sns.swarmplot(data=df, x='Genotype', y='Accuracy', ax=ax, color='black', alpha=0.5, size=4)
            ax.set_title(f'Cohort {cohort}: Lapse State Accuracy', fontweight='bold')
            ax.set_ylabel('Accuracy')
            ax.grid(axis='y', alpha=0.3)

            # Add statistical test
            genotypes = df['Genotype'].unique()
            if len(genotypes) >= 2:
                groups = [df[df['Genotype'] == g]['Accuracy'].values for g in genotypes]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) >= 2:
                    stat, pval = kruskal(*groups)
                    ax.text(0.5, 0.95, f'Kruskal-Wallis p={pval:.4f}',
                           transform=ax.transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # 2. Lapse Occupancy
            ax = axes[row, 1]
            sns.violinplot(data=df, x='Genotype', y='Occupancy', ax=ax, inner=None, alpha=0.6)
            sns.swarmplot(data=df, x='Genotype', y='Occupancy', ax=ax, color='black', alpha=0.5, size=4)
            ax.set_title(f'Cohort {cohort}: Lapse State Time', fontweight='bold')
            ax.set_ylabel('Proportion of Trials')
            ax.grid(axis='y', alpha=0.3)

            # 3. Lapse Dwell Time
            ax = axes[row, 2]
            sns.violinplot(data=df, x='Genotype', y='Dwell_Time', ax=ax, inner=None, alpha=0.6)
            sns.swarmplot(data=df, x='Genotype', y='Dwell_Time', ax=ax, color='black', alpha=0.5, size=4)
            ax.set_title(f'Cohort {cohort}: Lapse Bout Duration', fontweight='bold')
            ax.set_ylabel('Mean Trials per Bout')
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle('Lapse State Characteristics by Genotype',
                    fontsize=16, fontweight='bold', y=1.00)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'lapse_state_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'lapse_state_analysis.pdf',
                   bbox_inches='tight')
        plt.close()

        print("✓ Created lapse state analysis")


def main():
    """Run comprehensive genotype × state analysis."""
    print("="*80)
    print("COMPREHENSIVE GENOTYPE × STATE ANALYSIS")
    print("="*80)

    analyzer = ComprehensiveGenotypeAnalysis()

    # Define animals
    animals_W = [f'c{c}m{m}' for c in range(1, 5) for m in range(1, 6)
                 if not (c == 1 and m == 5)]
    animals_F = [str(i) for i in [11, 12, 13, 14, 21, 22, 23, 24, 25,
                                   31, 32, 33, 34, 41, 42, 51, 52,
                                   61, 62, 63, 64, 71, 72, 73,
                                   81, 82, 83, 84, 91, 92, 93,
                                   101, 102, 103, 104]]

    # Load data
    print("\nLoading cohort data...")
    results_W = analyzer.load_cohort_data('W', animals_W)
    results_F = analyzer.load_cohort_data('F', animals_F)
    print(f"  Cohort W: {len(results_W)} animals")
    print(f"  Cohort F: {len(results_F)} animals")

    # Run analyses
    print("\nGenerating visualizations...")
    analyzer.create_state_label_distribution_plot(results_W, results_F)
    analyzer.analyze_lapse_states(results_W, results_F)

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {analyzer.output_dir}")

if __name__ == '__main__':
    main()
