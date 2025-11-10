"""
Priority 2: Response Time (RT) Q-Q Plots with KS Tests
========================================================

Creates quantile-quantile plots comparing response time distributions
between states and genotypes, with Kolmogorov-Smirnov tests for
statistical comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import utilities
import sys
sys.path.insert(0, '/home/user/GLMHMM')
from state_validation import create_broad_state_categories
from glmhmm_utils import load_and_preprocess_session_data

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


class RTAnalyzer:
    """Response time analysis with Q-Q plots and statistical tests."""

    def __init__(self, results_dir='results/phase1_non_reversal'):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'priority2_rt_analysis'
        self.output_dir.mkdir(exist_ok=True)

    def load_data_with_rt(self, cohort, animals):
        """Load model results and trial data including response times."""
        results = []
        all_trials = []

        # Determine data file
        if cohort == 'W':
            data_file = '/home/user/GLMHMM/W LD Data 11.08 All_processed.csv'
        else:
            data_file = '/home/user/GLMHMM/F LD Data 11.08 All_processed.csv'

        # Load trial data
        trial_df = load_and_preprocess_session_data(data_file)

        for animal in animals:
            pkl_file = self.results_dir / f'{animal}_cohort{cohort}_model.pkl'
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)

                    # Add broad categories
                    broad_categories = create_broad_state_categories(data['validated_labels'])
                    data['broad_categories'] = broad_categories

                    results.append(data)

                    # Get trial data with latency
                    animal_trials = trial_df[trial_df['animal_id'] == animal].copy()
                    if len(animal_trials) > 0:
                        # Match state sequence length
                        n_states = len(data['model'].most_likely_states)
                        if n_states <= len(animal_trials):
                            animal_trials_matched = animal_trials.iloc[:n_states].copy()
                            animal_trials_matched['glmhmm_state'] = data['model'].most_likely_states

                            # Add state labels
                            animal_trials_matched['state_label'] = animal_trials_matched['glmhmm_state'].map(
                                lambda s: broad_categories[s][0] if s in broad_categories else 'Unknown'
                            )

                            all_trials.append(animal_trials_matched)

        # Combine
        if len(all_trials) > 0:
            combined = pd.concat(all_trials, ignore_index=True)
        else:
            combined = pd.DataFrame()

        return results, combined

    def create_qq_plot(self, data1, data2, label1, label2, ax):
        """
        Create Q-Q plot comparing two distributions.

        Returns KS test statistic and p-value.
        """
        # Remove NaN and infinite values
        data1_clean = data1[np.isfinite(data1)]
        data2_clean = data2[np.isfinite(data2)]

        if len(data1_clean) == 0 or len(data2_clean) == 0:
            return None, None

        # Compute quantiles
        quantiles = np.linspace(0, 1, min(len(data1_clean), len(data2_clean), 100))
        q1 = np.quantile(data1_clean, quantiles)
        q2 = np.quantile(data2_clean, quantiles)

        # Q-Q plot
        ax.scatter(q1, q2, alpha=0.6, s=40)

        # Identity line
        min_val = min(q1.min(), q2.min())
        max_val = max(q1.max(), q2.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
               label='Identity')

        # KS test
        ks_stat, p_value = stats.ks_2samp(data1_clean, data2_clean)

        # Labels
        ax.set_xlabel(f'{label1} RT Quantiles (s)', fontweight='bold')
        ax.set_ylabel(f'{label2} RT Quantiles (s)', fontweight='bold')

        # Add stats box
        stats_text = f'KS = {ks_stat:.3f}\np = {p_value:.4f}'
        if p_value < 0.001:
            stats_text = f'KS = {ks_stat:.3f}\np < 0.001 ***'
        elif p_value < 0.01:
            stats_text += ' **'
        elif p_value < 0.05:
            stats_text += ' *'

        ax.text(0.05, 0.95, stats_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)

        return ks_stat, p_value

    def plot_rt_by_state(self, trials):
        """
        Create comprehensive RT analysis by state.
        """
        # Filter valid latencies (remove outliers > 30s)
        trials_clean = trials[
            (trials['latency'] > 0) &
            (trials['latency'] < 30) &
            (trials['latency'].notna())
        ].copy()

        if len(trials_clean) == 0:
            print("  No valid latency data")
            return

        # Get state labels
        states = sorted(trials_clean['state_label'].unique())

        # Create figure with Q-Q plots
        n_comparisons = len(states) * (len(states) - 1) // 2
        n_cols = 3
        n_rows = (n_comparisons + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # Flatten axes
        axes_flat = axes.flatten()

        # Pairwise comparisons
        idx = 0
        ks_results = []

        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states):
                if i < j:
                    ax = axes_flat[idx]

                    data1 = trials_clean[trials_clean['state_label'] == state1]['latency'].values
                    data2 = trials_clean[trials_clean['state_label'] == state2]['latency'].values

                    ks_stat, p_val = self.create_qq_plot(data1, data2, state1, state2, ax)

                    if ks_stat is not None:
                        ks_results.append({
                            'state1': state1,
                            'state2': state2,
                            'ks_statistic': ks_stat,
                            'p_value': p_val,
                            'n1': len(data1),
                            'n2': len(data2)
                        })

                    ax.set_title(f'{state1} vs {state2}', fontweight='bold', fontsize=12)
                    idx += 1

        # Remove empty subplots
        for i in range(idx, len(axes_flat)):
            fig.delaxes(axes_flat[i])

        plt.suptitle('Response Time Q-Q Plots: State Comparisons',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save
        plt.savefig(self.output_dir / 'rt_qq_states.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'rt_qq_states.pdf', bbox_inches='tight')
        plt.close()

        # Save KS test results
        if ks_results:
            ks_df = pd.DataFrame(ks_results)
            ks_df.to_csv(self.output_dir / 'ks_tests_states.csv', index=False)

        print(f"  ✓ Created RT Q-Q plots by state ({n_comparisons} comparisons)")

    def plot_rt_distributions(self, trials):
        """
        Create RT distribution plots with overlays.
        """
        # Filter valid latencies
        trials_clean = trials[
            (trials['latency'] > 0) &
            (trials['latency'] < 30) &
            (trials['latency'].notna())
        ].copy()

        if len(trials_clean) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Panel A: Distributions by state
        ax = axes[0, 0]
        states = sorted(trials_clean['state_label'].unique())
        colors = {'Engaged': '#2ecc71', 'Lapsed': '#e74c3c', 'Mixed': '#f39c12'}

        for state in states:
            data = trials_clean[trials_clean['state_label'] == state]['latency']
            ax.hist(data, bins=50, alpha=0.5, label=f'{state} (n={len(data)})',
                   color=colors.get(state, '#95a5a6'), density=True)

        ax.set_xlabel('Response Time (s)', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title('RT Distributions by State', fontweight='bold', fontsize=13)
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel B: Distributions by genotype
        ax = axes[0, 1]
        genotypes = sorted(trials_clean['genotype'].unique())

        for geno in genotypes:
            data = trials_clean[trials_clean['genotype'] == geno]['latency']
            if len(data) > 10:
                ax.hist(data, bins=50, alpha=0.5, label=f'{geno} (n={len(data)})',
                       density=True)

        ax.set_xlabel('Response Time (s)', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title('RT Distributions by Genotype', fontweight='bold', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Panel C: CDFs by state
        ax = axes[1, 0]
        for state in states:
            data = np.sort(trials_clean[trials_clean['state_label'] == state]['latency'])
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data, cdf, linewidth=2.5, label=state,
                   color=colors.get(state, '#95a5a6'))

        ax.set_xlabel('Response Time (s)', fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontweight='bold')
        ax.set_title('Cumulative Distribution Functions', fontweight='bold', fontsize=13)
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel D: Box plots by state and genotype
        ax = axes[1, 1]

        # Prepare data for box plot
        plot_data = []
        labels = []

        for state in states:
            for geno in genotypes:
                subset = trials_clean[
                    (trials_clean['state_label'] == state) &
                    (trials_clean['genotype'] == geno)
                ]['latency']

                if len(subset) > 5:  # Only plot if enough data
                    plot_data.append(subset)
                    labels.append(f'{state[:3]}\n{geno}')

        if len(plot_data) > 0:
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)

            # Color by state
            for i, patch in enumerate(bp['boxes']):
                state = labels[i].split('\n')[0]
                if 'Eng' in state:
                    patch.set_facecolor(colors['Engaged'])
                elif 'Lap' in state:
                    patch.set_facecolor(colors['Lapsed'])
                else:
                    patch.set_facecolor(colors['Mixed'])
                patch.set_alpha(0.6)

        ax.set_ylabel('Response Time (s)', fontweight='bold')
        ax.set_title('RT by State × Genotype', fontweight='bold', fontsize=13)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(axis='y', alpha=0.3)

        plt.suptitle('Response Time Analysis: Distributions and Comparisons',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save
        plt.savefig(self.output_dir / 'rt_distributions.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'rt_distributions.pdf', bbox_inches='tight')
        plt.close()

        print(f"  ✓ Created RT distribution plots")

    def plot_rt_by_genotype_qq(self, trials):
        """
        Create Q-Q plots comparing genotypes.
        """
        # Filter valid latencies
        trials_clean = trials[
            (trials['latency'] > 0) &
            (trials['latency'] < 30) &
            (trials['latency'].notna())
        ].copy()

        if len(trials_clean) == 0:
            return

        genotypes = sorted(trials_clean['genotype'].unique())

        # Pairwise comparisons between genotypes
        n_comparisons = len(genotypes) * (len(genotypes) - 1) // 2
        n_cols = 3
        n_rows = (n_comparisons + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols > 1:
            axes = axes.reshape(1, -1)
        elif n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        axes_flat = axes.flatten()

        idx = 0
        ks_results = []

        for i, geno1 in enumerate(genotypes):
            for j, geno2 in enumerate(genotypes):
                if i < j:
                    if idx >= len(axes_flat):
                        break

                    ax = axes_flat[idx]

                    data1 = trials_clean[trials_clean['genotype'] == geno1]['latency'].values
                    data2 = trials_clean[trials_clean['genotype'] == geno2]['latency'].values

                    if len(data1) > 10 and len(data2) > 10:
                        ks_stat, p_val = self.create_qq_plot(data1, data2, geno1, geno2, ax)

                        if ks_stat is not None:
                            ks_results.append({
                                'genotype1': geno1,
                                'genotype2': geno2,
                                'ks_statistic': ks_stat,
                                'p_value': p_val,
                                'n1': len(data1),
                                'n2': len(data2)
                            })

                        ax.set_title(f'{geno1} vs {geno2}', fontweight='bold', fontsize=12)

                    idx += 1

        # Remove empty subplots
        for i in range(idx, len(axes_flat)):
            fig.delaxes(axes_flat[i])

        plt.suptitle('Response Time Q-Q Plots: Genotype Comparisons',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save
        plt.savefig(self.output_dir / 'rt_qq_genotypes.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'rt_qq_genotypes.pdf', bbox_inches='tight')
        plt.close()

        # Save KS test results
        if ks_results:
            ks_df = pd.DataFrame(ks_results)
            ks_df.to_csv(self.output_dir / 'ks_tests_genotypes.csv', index=False)

        print(f"  ✓ Created RT Q-Q plots by genotype ({idx} comparisons)")


def main():
    """Run RT analysis."""
    print("="*80)
    print("PRIORITY 2: RESPONSE TIME (RT) ANALYSIS")
    print("="*80)

    analyzer = RTAnalyzer()

    # Define animals
    animals_W = [f'c{c}m{m}' for c in range(1, 5) for m in range(1, 6)
                 if not (c == 1 and m == 5)]
    animals_F = [11, 12, 13, 14, 21, 22, 23, 24, 25,
                 31, 32, 33, 34, 41, 42, 51, 52,
                 61, 62, 63, 64, 71, 72, 73,
                 81, 82, 83, 84, 91, 92, 93,
                 101, 102, 103, 104]

    # Load data
    print("\nLoading data with RT...")
    results_W, trials_W = analyzer.load_data_with_rt('W', animals_W)
    results_F, trials_F = analyzer.load_data_with_rt('F', animals_F)
    print(f"  Cohort W: {len(results_W)} animals, {len(trials_W)} trials")
    print(f"  Cohort F: {len(results_F)} animals, {len(trials_F)} trials")

    # Combine cohorts
    all_trials = pd.concat([trials_W, trials_F], ignore_index=True)
    print(f"  Total: {len(all_trials)} trials")

    # Generate visualizations
    print("\nGenerating visualizations...")

    print("\n[1/3] RT Q-Q plots by state...")
    analyzer.plot_rt_by_state(all_trials)

    print("\n[2/3] RT distributions...")
    analyzer.plot_rt_distributions(all_trials)

    print("\n[3/3] RT Q-Q plots by genotype...")
    analyzer.plot_rt_by_genotype_qq(all_trials)

    print("\n" + "="*80)
    print("✓ RT ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {analyzer.output_dir}")


if __name__ == '__main__':
    main()
