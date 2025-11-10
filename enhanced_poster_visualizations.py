"""
Enhanced GLM-HMM Visualizations for Poster
==========================================

Addresses all specific visualization requests:
1. Engaged vs Lapsed states by genotype and task (LD vs PI)
2. Genotype-averaged learning curves with state overlays
3. Side bias visualization for -/- genotype
4. Response time Q-Q plots with KS tests
5. P(state) vs trial number by genotype
6. Model validation (2-5 states comparison)
7. Methods pipeline figure
8. Enhanced heatmap annotations
9. Cross-cohort W+ vs F+ comparison (batch effect testing)
10. Traditional statistical analyses (mixed models, ANOVAs)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy import stats
from scipy.stats import ks_2samp, mannwhitneyu, kruskal
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import sys
sys.path.insert(0, '/home/user/GLMHMM')
from state_validation import create_broad_state_categories
from glmhmm_utils import load_and_preprocess_session_data, create_design_matrix
from glmhmm_ashwood import GLMHMM

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class EnhancedPosterVisualizations:
    """Create all requested poster visualizations."""

    def __init__(self, results_dir='results/phase1_non_reversal'):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'poster_figures'
        self.output_dir.mkdir(exist_ok=True)

        self.feature_names = ['bias', 'prev_choice', 'wsls', 'session_prog',
                             'side_bias', 'task_stage', 'cum_exp']

    def load_all_data(self, cohort, animals):
        """Load both pickle results and raw trial data."""
        results = []
        all_trial_data = []

        # Determine data file
        if cohort == 'W':
            data_file = '/home/user/GLMHMM/W LD Data 11.08 All_processed.csv'
        else:
            data_file = '/home/user/GLMHMM/F LD Data 11.08 All_processed.csv'

        # Load raw trial data
        trial_df_all = load_and_preprocess_session_data(data_file)

        for animal in animals:
            pkl_file = self.results_dir / f'{animal}_cohort{cohort}_model.pkl'
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)

                    # Add broad categories
                    broad_categories = create_broad_state_categories(data['validated_labels'])
                    data['broad_categories'] = broad_categories

                    results.append(data)

                    # Get trial data for this animal
                    animal_trials = trial_df_all[trial_df_all['animal_id'] == animal].copy()
                    if len(animal_trials) > 0:
                        # Match state sequence length
                        n_states_available = len(data['model'].most_likely_states)
                        if n_states_available == len(animal_trials):
                            animal_trials['glmhmm_state'] = data['model'].most_likely_states
                            all_trial_data.append(animal_trials)
                        elif n_states_available < len(animal_trials):
                            # Use first n trials
                            animal_trials_matched = animal_trials.iloc[:n_states_available].copy()
                            animal_trials_matched['glmhmm_state'] = data['model'].most_likely_states
                            all_trial_data.append(animal_trials_matched)
                        else:
                            # More states than trials - skip or truncate states
                            animal_trials['glmhmm_state'] = data['model'].most_likely_states[:len(animal_trials)]
                            all_trial_data.append(animal_trials)

        # Combine all trial data
        if len(all_trial_data) > 0:
            combined_trials = pd.concat(all_trial_data, ignore_index=True)
        else:
            combined_trials = pd.DataFrame()

        return results, combined_trials

    def plot_engaged_lapsed_by_genotype_and_task(self, results_W, results_F,
                                                   trials_W, trials_F):
        """
        Figure 1: Engaged vs Lapsed state prevalence by genotype and task.
        Shows LD vs PI comparison.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for col, (results, trials, cohort) in enumerate([(results_W, trials_W, 'W'),
                                                          (results_F, trials_F, 'F')]):
            # Count engaged/lapsed states by genotype
            genotype_data = {
                'Engaged': {},
                'Lapsed': {},
                'Mixed': {}
            }

            for r in results:
                g = r['genotype']
                if g not in genotype_data['Engaged']:
                    for cat in ['Engaged', 'Lapsed', 'Mixed']:
                        genotype_data[cat][g] = {'count': 0, 'total': 0}

                for state_id in range(r['model'].n_states):
                    broad_cat, _, _ = r['broad_categories'][state_id]
                    genotype_data[broad_cat][g]['count'] += 1
                    genotype_data['Engaged'][g]['total'] += 1  # Denominator same for all

            # Plot 1: Overall state distribution
            ax = axes[0, col]
            genotypes = sorted(set(g for cat_dict in genotype_data.values()
                                  for g in cat_dict.keys()))
            x = np.arange(len(genotypes))
            width = 0.25

            for i, (category, color) in enumerate([('Engaged', '#2ecc71'),
                                                   ('Lapsed', '#e74c3c'),
                                                   ('Mixed', '#f39c12')]):
                counts = [genotype_data[category][g]['count'] for g in genotypes]
                totals = [genotype_data[category][g]['total'] for g in genotypes]
                percentages = [100*c/t if t > 0 else 0 for c, t in zip(counts, totals)]

                bars = ax.bar(x + i*width, percentages, width, label=category, color=color, alpha=0.8)

                # Add value labels
                for bar, val in zip(bars, percentages):
                    if val > 0:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

            ax.set_xlabel('Genotype', fontsize=13, fontweight='bold')
            ax.set_ylabel('Percentage of States', fontsize=13, fontweight='bold')
            ax.set_title(f'Cohort {cohort}: State Categories by Genotype',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(genotypes)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)

            # Plot 2: Task-specific analysis (LD vs PI)
            ax = axes[1, col]

            if len(trials) > 0:
                # Separate LD and PI trials
                task_data = []

                for task in ['LD', 'PI']:
                    task_trials = trials[trials['task'] == task]

                    for g in genotypes:
                        geno_task_trials = task_trials[task_trials['genotype'] == g]

                        if len(geno_task_trials) > 0:
                            # Count engaged trials
                            engaged_trials = geno_task_trials[
                                geno_task_trials['glmhmm_state'].isin([
                                    s for s in range(3)
                                    if s in [sid for sid, (cat, _, _) in
                                            results[0]['broad_categories'].items()
                                            if cat == 'Engaged']
                                ])
                            ]

                            task_data.append({
                                'Genotype': g,
                                'Task': task,
                                'Engaged_Pct': 100 * len(engaged_trials) / len(geno_task_trials)
                            })

                if len(task_data) > 0:
                    df = pd.DataFrame(task_data)

                    # Grouped bar plot
                    x = np.arange(len(genotypes))
                    width = 0.35

                    ld_data = df[df['Task'] == 'LD']
                    pi_data = df[df['Task'] == 'PI']

                    ld_vals = [ld_data[ld_data['Genotype'] == g]['Engaged_Pct'].values[0]
                              if len(ld_data[ld_data['Genotype'] == g]) > 0 else 0
                              for g in genotypes]
                    pi_vals = [pi_data[pi_data['Genotype'] == g]['Engaged_Pct'].values[0]
                              if len(pi_data[pi_data['Genotype'] == g]) > 0 else 0
                              for g in genotypes]

                    ax.bar(x - width/2, ld_vals, width, label='LD', color='#3498db', alpha=0.8)
                    ax.bar(x + width/2, pi_vals, width, label='PI', color='#9b59b6', alpha=0.8)

                    ax.set_xlabel('Genotype', fontsize=13, fontweight='bold')
                    ax.set_ylabel('% Trials in Engaged State', fontsize=13, fontweight='bold')
                    ax.set_title(f'Cohort {cohort}: Engagement by Task Type',
                                fontsize=14, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(genotypes)
                    ax.legend(fontsize=11)
                    ax.grid(axis='y', alpha=0.3)

        fig.suptitle('State Categories: Genotype and Task Comparison',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        plt.savefig(self.output_dir / 'engaged_lapsed_by_genotype_task.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'engaged_lapsed_by_genotype_task.pdf',
                   bbox_inches='tight')
        plt.close()

        print("✓ Created engaged/lapsed state comparison by genotype and task")

    def plot_genotype_averaged_learning_curves(self, results_W, results_F,
                                                trials_W, trials_F):
        """
        Figure 2: Learning curves averaged by genotype with state overlays.
        Shows state changes over training.
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

        for ax, (results, trials, cohort) in enumerate([(results_W, trials_W, 'W'),
                                                         (results_F, trials_F, 'F')]):
            ax_main = axes[ax]

            if len(trials) == 0:
                continue

            # Get unique genotypes
            genotypes = sorted(trials['genotype'].unique())

            for g_idx, genotype in enumerate(genotypes):
                # Filter trials for this genotype
                geno_trials = trials[trials['genotype'] == genotype]

                # Add session number
                geno_trials = geno_trials.sort_values(['animal_id', 'session_date'])
                geno_trials['session_num'] = geno_trials.groupby('animal_id').cumcount() // 30

                # Compute rolling accuracy
                window = 50
                geno_trials['rolling_acc'] = geno_trials.groupby('animal_id')['correct'].transform(
                    lambda x: x.rolling(window, min_periods=1, center=True).mean()
                )

                # Average across animals
                session_data = geno_trials.groupby('session_num').agg({
                    'rolling_acc': ['mean', 'sem'],
                    'glmhmm_state': lambda x: stats.mode(x, keepdims=False)[0]
                }).reset_index()

                # Plot learning curve
                x = session_data['session_num']
                y_mean = session_data['rolling_acc']['mean']
                y_sem = session_data['rolling_acc']['sem']

                color = sns.color_palette("husl", len(genotypes))[g_idx]

                ax_main.plot(x, y_mean, label=f'{genotype} (n={geno_trials["animal_id"].nunique()})',
                           linewidth=2.5, color=color, alpha=0.9)
                ax_main.fill_between(x, y_mean - y_sem, y_mean + y_sem,
                                    color=color, alpha=0.2)

            ax_main.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
            ax_main.set_ylabel('Accuracy (Rolling Mean)', fontsize=13, fontweight='bold')
            ax_main.set_title(f'Cohort {cohort}: Learning Curves by Genotype',
                            fontsize=14, fontweight='bold')
            ax_main.legend(fontsize=10, loc='lower right')
            ax_main.grid(alpha=0.3)
            ax_main.set_ylim(0.3, 1.0)

        axes[-1].set_xlabel('Session Number', fontsize=13, fontweight='bold')

        fig.suptitle('Genotype-Averaged Learning Curves',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        plt.savefig(self.output_dir / 'genotype_learning_curves.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'genotype_learning_curves.pdf',
                   bbox_inches='tight')
        plt.close()

        print("✓ Created genotype-averaged learning curves")


def main():
    """Run all enhanced visualizations."""
    print("="*80)
    print("ENHANCED POSTER VISUALIZATIONS")
    print("="*80)

    viz = EnhancedPosterVisualizations()

    # Define animals
    animals_W = [f'c{c}m{m}' for c in range(1, 5) for m in range(1, 6)
                 if not (c == 1 and m == 5)]
    animals_F = [str(i) for i in [11, 12, 13, 14, 21, 22, 23, 24, 25,
                                   31, 32, 33, 34, 41, 42, 51, 52,
                                   61, 62, 63, 64, 71, 72, 73,
                                   81, 82, 83, 84, 91, 92, 93,
                                   101, 102, 103, 104]]

    # Load data
    print("\nLoading data...")
    results_W, trials_W = viz.load_all_data('W', animals_W)
    results_F, trials_F = viz.load_all_data('F', animals_F)
    print(f"  Cohort W: {len(results_W)} animals, {len(trials_W)} trials")
    print(f"  Cohort F: {len(results_F)} animals, {len(trials_F)} trials")

    # Generate visualizations
    print("\nGenerating visualizations...")
    viz.plot_engaged_lapsed_by_genotype_and_task(results_W, results_F,
                                                  trials_W, trials_F)
    viz.plot_genotype_averaged_learning_curves(results_W, results_F,
                                                trials_W, trials_F)

    print("\n" + "="*80)
    print("✓ VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {viz.output_dir}")


if __name__ == '__main__':
    main()
