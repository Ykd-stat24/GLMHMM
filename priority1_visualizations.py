"""
Priority 1: Core Poster Visualizations
=======================================

Creates essential figures for poster:
1. Engaged/Lapsed states by genotype and task (LD vs PI)
2. Genotype-averaged learning curves
3. Side bias detailed analysis for -/- subjects
4. Cross-cohort W+ vs F+ comparison (with batch effect testing)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/home/user/GLMHMM')
from state_validation import create_broad_state_categories
from glmhmm_utils import load_and_preprocess_session_data

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class Priority1Visualizations:
    """Core visualizations for poster."""

    def __init__(self, results_dir='results/phase1_non_reversal'):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'priority1_figures'
        self.output_dir.mkdir(exist_ok=True)

        # Task identification from Schedule name
        self.task_mapping = {
            'A_Mouse LD Punish Incorrect Training v2': 'PI',
            'A_Mouse LD 1 choice v2': 'LD'
        }

    def load_cohort_data(self, cohort, animals):
        """Load pickle results and raw trial data with task labels."""
        results = []

        # Determine data file
        if cohort == 'W':
            data_file = '/home/user/GLMHMM/W LD Data 11.08 All_processed.csv'
        else:
            data_file = '/home/user/GLMHMM/F LD Data 11.08 All_processed.csv'

        # Load raw data
        print(f"  Loading {cohort} cohort raw data...")
        trial_df_all = load_and_preprocess_session_data(data_file)

        # task_type column already exists from preprocessing
        # It contains: LD, PI, PD, PD_PI, etc.
        print(f"  Task types: {trial_df_all['task_type'].value_counts().to_dict()}")

        # Load pickle results
        print(f"  Loading {cohort} cohort GLM-HMM results...")
        for animal in animals:
            pkl_file = self.results_dir / f'{animal}_cohort{cohort}_model.pkl'
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)

                    # Add broad categories
                    broad_categories = create_broad_state_categories(data['validated_labels'])
                    data['broad_categories'] = broad_categories

                    results.append(data)

        print(f"  Loaded {len(results)} animals, {len(trial_df_all)} trials")

        return results, trial_df_all

    def figure1_engaged_lapsed_by_genotype_task(self, results_W, trials_W,
                                                  results_F, trials_F):
        """
        Figure 1: Engaged vs Lapsed prevalence by genotype and task.

        Panel A: Overall state categories by genotype (both cohorts)
        Panel B: Task-specific engagement (LD vs PI) by genotype
        """
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Process each cohort
        for col, (results, trials, cohort) in enumerate([
            (results_W, trials_W, 'W'),
            (results_F, trials_F, 'F')
        ]):
            # === PANEL A: Overall state categories ===
            ax1 = fig.add_subplot(gs[0, col])

            # Count states by genotype and category
            genotype_counts = {}

            for r in results:
                g = r['genotype']
                if g not in genotype_counts:
                    genotype_counts[g] = {'Engaged': 0, 'Lapsed': 0, 'Mixed': 0, 'Total': 0}

                for state_id in range(r['model'].n_states):
                    broad_cat, _, _ = r['broad_categories'][state_id]
                    genotype_counts[g][broad_cat] += 1
                    genotype_counts[g]['Total'] += 1

            # Plot
            genotypes = sorted(genotype_counts.keys())
            x = np.arange(len(genotypes))
            width = 0.25

            for i, (category, color) in enumerate([
                ('Engaged', '#27ae60'),
                ('Lapsed', '#e74c3c'),
                ('Mixed', '#f39c12')
            ]):
                counts = [genotype_counts[g][category] for g in genotypes]
                totals = [genotype_counts[g]['Total'] for g in genotypes]
                percentages = [100*c/t if t > 0 else 0 for c, t in zip(counts, totals)]

                bars = ax1.bar(x + i*width, percentages, width,
                             label=category, color=color, alpha=0.8, edgecolor='black')

                # Add value labels
                for bar, val, count, total in zip(bars, percentages, counts, totals):
                    if val > 0:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{val:.1f}%\n({count}/{total})',
                               ha='center', va='bottom', fontsize=9, fontweight='bold')

            ax1.set_xlabel('Genotype', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Percentage of States', fontsize=14, fontweight='bold')
            ax1.set_title(f'Cohort {cohort}: State Categories by Genotype',
                        fontsize=15, fontweight='bold')
            ax1.set_xticks(x + width)
            ax1.set_xticklabels(genotypes, fontsize=12)
            ax1.legend(fontsize=12, loc='upper right')
            ax1.grid(axis='y', alpha=0.3)
            ax1.set_ylim(0, max([max([(genotype_counts[g][cat]/genotype_counts[g]['Total'])*100
                                     for g in genotypes])
                                for cat in ['Engaged', 'Lapsed', 'Mixed']]) * 1.25)

            # === PANEL B: Task-specific engagement ===
            ax2 = fig.add_subplot(gs[1, col])

            if 'task_type' in trials.columns:
                # Merge trial data with state labels
                task_engagement = {}

                for g in genotypes:
                    task_engagement[g] = {'LD': [], 'PI': []}

                # Get engaged state IDs for each animal
                for r in results:
                    g = r['genotype']
                    animal = r['animal_id']

                    # Find engaged states for this animal
                    engaged_states = [s for s in range(r['model'].n_states)
                                    if r['broad_categories'][s][0] == 'Engaged']

                    # Get this animal's trials
                    animal_trials = trials[trials['animal_id'] == animal].copy()

                    if len(animal_trials) == 0:
                        continue

                    # Add state labels (matching length)
                    n_states_avail = len(r['model'].most_likely_states)
                    if n_states_avail <= len(animal_trials):
                        animal_trials = animal_trials.iloc[:n_states_avail].copy()
                        animal_trials['glmhmm_state'] = r['model'].most_likely_states
                    else:
                        animal_trials['glmhmm_state'] = r['model'].most_likely_states[:len(animal_trials)]

                    # Calculate engagement by task
                    for task in ['LD', 'PI']:
                        task_trials = animal_trials[animal_trials['task_type'] == task]
                        if len(task_trials) > 0:
                            engaged_trials = task_trials[task_trials['glmhmm_state'].isin(engaged_states)]
                            engagement_pct = 100 * len(engaged_trials) / len(task_trials)
                            task_engagement[g][task].append(engagement_pct)

                # Plot
                x = np.arange(len(genotypes))
                width = 0.35

                ld_means = [np.mean(task_engagement[g]['LD']) if len(task_engagement[g]['LD']) > 0 else 0
                           for g in genotypes]
                ld_sems = [np.std(task_engagement[g]['LD'])/np.sqrt(len(task_engagement[g]['LD']))
                          if len(task_engagement[g]['LD']) > 0 else 0
                          for g in genotypes]

                pi_means = [np.mean(task_engagement[g]['PI']) if len(task_engagement[g]['PI']) > 0 else 0
                           for g in genotypes]
                pi_sems = [np.std(task_engagement[g]['PI'])/np.sqrt(len(task_engagement[g]['PI']))
                          if len(task_engagement[g]['PI']) > 0 else 0
                          for g in genotypes]

                ax2.bar(x - width/2, ld_means, width, yerr=ld_sems,
                       label='LD', color='#3498db', alpha=0.8,
                       edgecolor='black', capsize=5)
                ax2.bar(x + width/2, pi_means, width, yerr=pi_sems,
                       label='PI', color='#9b59b6', alpha=0.8,
                       edgecolor='black', capsize=5)

                # Add value labels
                for i, (ld_m, pi_m, g) in enumerate(zip(ld_means, pi_means, genotypes)):
                    n_ld = len(task_engagement[g]['LD'])
                    n_pi = len(task_engagement[g]['PI'])
                    if ld_m > 0:
                        ax2.text(i - width/2, ld_m + ld_sems[i] + 2,
                               f'{ld_m:.1f}%\n(n={n_ld})',
                               ha='center', va='bottom', fontsize=8)
                    if pi_m > 0:
                        ax2.text(i + width/2, pi_m + pi_sems[i] + 2,
                               f'{pi_m:.1f}%\n(n={n_pi})',
                               ha='center', va='bottom', fontsize=8)

                ax2.set_xlabel('Genotype', fontsize=14, fontweight='bold')
                ax2.set_ylabel('% Trials in Engaged State', fontsize=14, fontweight='bold')
                ax2.set_title(f'Cohort {cohort}: Task-Specific Engagement',
                            fontsize=15, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(genotypes, fontsize=12)
                ax2.legend(fontsize=12, loc='upper right')
                ax2.grid(axis='y', alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Task labels not available',
                        ha='center', va='center', transform=ax2.transAxes,
                        fontsize=14)

        fig.suptitle('State Categories: Genotype and Task Comparison',
                    fontsize=18, fontweight='bold')

        plt.savefig(self.output_dir / 'fig1_engaged_lapsed_by_genotype_task.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig1_engaged_lapsed_by_genotype_task.pdf',
                   bbox_inches='tight')
        plt.close()

        print("✓ Created Figure 1: Engaged/Lapsed by genotype and task")


def main():
    """Generate Priority 1 visualizations."""
    print("="*80)
    print("PRIORITY 1 VISUALIZATIONS")
    print("="*80)

    viz = Priority1Visualizations()

    # Define animals
    animals_W = [f'c{c}m{m}' for c in range(1, 5) for m in range(1, 6)
                 if not (c == 1 and m == 5)]
    animals_F = [str(i) for i in [11, 12, 13, 14, 21, 22, 23, 24, 25,
                                   31, 32, 33, 34, 41, 42, 51, 52,
                                   61, 62, 63, 64, 71, 72, 73,
                                   81, 82, 83, 84, 91, 92, 93,
                                   101, 102, 103, 104]]

    # Load data
    print("\nLoading Cohort W...")
    results_W, trials_W = viz.load_cohort_data('W', animals_W)

    print("\nLoading Cohort F...")
    results_F, trials_F = viz.load_cohort_data('F', animals_F)

    # Generate Figure 1
    print("\nGenerating Figure 1...")
    viz.figure1_engaged_lapsed_by_genotype_task(results_W, trials_W,
                                                 results_F, trials_F)

    print("\n" + "="*80)
    print("✓ PRIORITY 1 VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nOutput: {viz.output_dir}")


if __name__ == '__main__':
    main()

    def figure2_learning_curves_by_genotype(self, results_W, trials_W,
                                             results_F, trials_F):
        """
        Figure 2: Genotype-averaged learning curves with state overlays.

        Shows how accuracy improves over sessions, colored by genotype.
        """
        fig, axes = plt.subplots(2, 1, figsize=(18, 12))

        for ax_idx, (results, trials, cohort) in enumerate([
            (results_W, trials_W, 'W'),
            (results_F, trials_F, 'F')
        ]):
            ax = axes[ax_idx]

            if len(trials) == 0:
                continue

            # Get unique genotypes
            genotypes = sorted(trials['genotype'].unique())
            colors = sns.color_palette("husl", len(genotypes))

            for g_idx, genotype in enumerate(genotypes):
                # Get trials for this genotype
                geno_trials = trials[trials['genotype'] == genotype].copy()

                # Add session number (every ~30 trials = 1 session)
                geno_trials = geno_trials.sort_values(['animal_id', 'session_date', 'trial_num'])
                geno_trials['cumulative_trial'] = geno_trials.groupby('animal_id').cumcount()
                geno_trials['session_num'] = geno_trials['cumulative_trial'] // 30

                # Compute rolling accuracy per animal
                window = 30
                geno_trials['rolling_acc'] = geno_trials.groupby('animal_id')['correct'].transform(
                    lambda x: x.rolling(window, min_periods=1, center=True).mean()
                )

                # Average across animals per session
                session_stats = geno_trials.groupby('session_num').agg({
                    'rolling_acc': ['mean', 'sem'],
                    'animal_id': 'nunique'
                }).reset_index()

                x = session_stats['session_num']
                y_mean = session_stats['rolling_acc']['mean']
                y_sem = session_stats['rolling_acc']['sem']
                n_animals = session_stats['animal_id']['nunique'].iloc[0]

                # Plot
                ax.plot(x, y_mean, linewidth=3, color=colors[g_idx],
                       label=f'{genotype} (n={n_animals})', alpha=0.9)
                ax.fill_between(x, y_mean - y_sem, y_mean + y_sem,
                               color=colors[g_idx], alpha=0.2)

            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='Chance')
            ax.axhline(y=0.8, color='green', linestyle=':', alpha=0.6, linewidth=2, label='Criterion (80%)')

            ax.set_ylabel('Accuracy (Rolling Mean ± SEM)', fontsize=14, fontweight='bold')
            ax.set_title(f'Cohort {cohort}: Learning Curves by Genotype',
                        fontsize=15, fontweight='bold')
            ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
            ax.grid(alpha=0.3)
            ax.set_ylim(0.3, 1.0)

        axes[-1].set_xlabel('Session Number (~30 trials/session)', fontsize=14, fontweight='bold')

        fig.suptitle('Genotype-Averaged Learning Curves',
                    fontsize=18, fontweight='bold')
        plt.tight_layout()

        plt.savefig(self.output_dir / 'fig2_learning_curves_by_genotype.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig2_learning_curves_by_genotype.pdf',
                   bbox_inches='tight')
        plt.close()

        print("✓ Created Figure 2: Learning curves by genotype")

    def figure3_side_bias_detailed_minus_minus(self, results_F, trials_F):
        """
        Figure 3: Detailed side bias analysis for -/- genotype.

        Shows perseverative behavior and choice patterns.
        """
        # Filter for -/- genotype
        minus_minus_results = [r for r in results_F if r['genotype'] == '-/-']

        if len(minus_minus_results) == 0:
            print("  No -/- animals found, skipping Figure 3")
            return

        # Get -/- trials
        minus_minus_animals = [r['animal_id'] for r in minus_minus_results]
        minus_minus_trials = trials_F[trials_F['animal_id'].isin(minus_minus_animals)].copy()

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

        # Panel A: Choice probability by position
        ax1 = fig.add_subplot(gs[0, :])

        if 'position' in minus_minus_trials.columns:
            # Calculate P(right) for each position
            position_stats = minus_minus_trials.groupby('position').agg({
                'chosen_side': lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0.5,
                'correct': 'mean'
            }).reset_index()
            position_stats.columns = ['position', 'p_right', 'accuracy']

            ax1_2 = ax1.twinx()

            # Plot choice bias
            ax1.bar(position_stats['position'], position_stats['p_right'],
                   alpha=0.6, color='#3498db', label='P(Right Choice)')
            ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

            # Plot accuracy
            ax1_2.plot(position_stats['position'], position_stats['accuracy'],
                      'ro-', linewidth=3, markersize=8, label='Accuracy', alpha=0.8)

            ax1.set_xlabel('Stimulus Position', fontsize=13, fontweight='bold')
            ax1.set_ylabel('P(Right Choice)', fontsize=13, fontweight='bold', color='#3498db')
            ax1_2.set_ylabel('Accuracy', fontsize=13, fontweight='bold', color='red')
            ax1.tick_params(axis='y', labelcolor='#3498db')
            ax1_2.tick_params(axis='y', labelcolor='red')
            ax1.set_title('-/- Genotype: Choice Bias by Position', fontsize=15, fontweight='bold')
            ax1.legend(loc='upper left', fontsize=11)
            ax1_2.legend(loc='upper right', fontsize=11)
            ax1.grid(alpha=0.3, axis='x')

        # Panel B: Side bias over time
        ax2 = fig.add_subplot(gs[1, :])

        # Calculate side bias in windows
        window_size = 100
        minus_minus_trials_sorted = minus_minus_trials.sort_values(['animal_id', 'session_date', 'trial_num'])
        minus_minus_trials_sorted['trial_block'] = minus_minus_trials_sorted.groupby('animal_id').cumcount() // window_size

        time_stats = minus_minus_trials_sorted.groupby(['animal_id', 'trial_block']).agg({
            'chosen_side': lambda x: abs((x == 1).sum() / len(x) - 0.5),  # Side bias magnitude
            'correct': 'mean'
        }).reset_index()

        # Average across animals
        time_avg = time_stats.groupby('trial_block').agg({
            'chosen_side': ['mean', 'sem'],
            'correct': ['mean', 'sem']
        }).reset_index()

        x = time_avg['trial_block']
        y_bias = time_avg['chosen_side']['mean']
        sem_bias = time_avg['chosen_side']['sem']

        ax2.plot(x, y_bias, linewidth=3, color='#e74c3c', label='Side Bias')
        ax2.fill_between(x, y_bias - sem_bias, y_bias + sem_bias,
                        color='#e74c3c', alpha=0.2)

        ax2.set_xlabel(f'Trial Block (×{window_size} trials)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Side Bias (|P(Right) - 0.5|)', fontsize=13, fontweight='bold')
        ax2.set_title('-/- Genotype: Side Bias Over Training', fontsize=15, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)

        # Panel C-E: Individual animal examples
        for i, (animal_id, animal_result) in enumerate(zip(minus_minus_animals[:3], minus_minus_results[:3])):
            ax = fig.add_subplot(gs[2, i])

            animal_trials = minus_minus_trials[minus_minus_trials['animal_id'] == animal_id].copy()
            animal_trials = animal_trials.sort_values('trial_num')

            # Rolling side bias
            animal_trials['side_bias_roll'] = animal_trials['chosen_side'].rolling(50, min_periods=1).apply(
                lambda x: abs((x == 1).sum() / len(x) - 0.5)
            )

            ax.plot(animal_trials.index, animal_trials['side_bias_roll'],
                   linewidth=2, color='#9b59b6', alpha=0.8)

            # Get state labels
            state_labels = [r['broad_categories'][s][0] for s in range(animal_result['model'].n_states)]

            ax.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
            ax.set_ylabel('Side Bias', fontsize=11, fontweight='bold')
            ax.set_title(f'{animal_id}\nStates: {", ".join(state_labels)}',
                        fontsize=12)
            ax.grid(alpha=0.3)

        fig.suptitle('-/- Genotype: Detailed Side Bias Analysis',
                    fontsize=18, fontweight='bold')

        plt.savefig(self.output_dir / 'fig3_side_bias_minus_minus.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig3_side_bias_minus_minus.pdf',
                   bbox_inches='tight')
        plt.close()

        print("✓ Created Figure 3: Side bias analysis for -/- genotype")

    def figure4_cross_cohort_W_plus_vs_F_plus(self, results_W, results_F):
        """
        Figure 4: Cross-cohort comparison W+ vs F+ with batch effect testing.

        Tests if cohorts can be combined as "WT controls".
        """
        # Filter for WT animals
        w_plus = [r for r in results_W if r['genotype'] == '+']
        f_plus = [r for r in results_F if r['genotype'] == '+']

        if len(w_plus) == 0 or len(f_plus) == 0:
            print(f"  Insufficient WT animals (W+: {len(w_plus)}, F+: {len(f_plus)})")
            return

        print(f"\n  Testing batch effects: W+ (n={len(w_plus)}) vs F+ (n={len(f_plus)})")

        # Collect metrics
        metrics_to_test = ['accuracy', 'wsls_ratio', 'side_bias', 'latency_cv',
                          'occupancy', 'dwell_mean']

        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

        # Statistical test results
        test_results = []

        metric_idx = 0
        for state in range(3):  # Assuming 3 states
            for metric in metrics_to_test[:6]:  # First 6 metrics
                if metric_idx >= 12:
                    break

                row = metric_idx // 3
                col = metric_idx % 3
                ax = fig.add_subplot(gs[row, col])

                # Collect data
                w_vals = []
                f_vals = []

                for r in w_plus:
                    metrics_df = r['state_metrics']
                    state_data = metrics_df[metrics_df['state'] == state]
                    if len(state_data) > 0 and metric in state_data.columns:
                        w_vals.append(state_data[metric].values[0])

                for r in f_plus:
                    metrics_df = r['state_metrics']
                    state_data = metrics_df[metrics_df['state'] == state]
                    if len(state_data) > 0 and metric in state_data.columns:
                        f_vals.append(state_data[metric].values[0])

                if len(w_vals) > 0 and len(f_vals) > 0:
                    # Statistical test
                    stat, pval = mannwhitneyu(w_vals, f_vals, alternative='two-sided')

                    # Plot
                    data_df = pd.DataFrame({
                        'Value': w_vals + f_vals,
                        'Cohort': ['W+'] * len(w_vals) + ['F+'] * len(f_vals)
                    })

                    sns.violinplot(data=data_df, x='Cohort', y='Value', ax=ax, inner=None, alpha=0.6)
                    sns.swarmplot(data=data_df, x='Cohort', y='Value', ax=ax,
                                color='black', alpha=0.6, size=5)

                    # Add p-value
                    y_max = ax.get_ylim()[1]
                    sig_str = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
                    color = 'red' if pval < 0.05 else 'green'

                    ax.text(0.5, 0.95, f'p={pval:.4f} {sig_str}',
                           transform=ax.transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                           fontsize=10, fontweight='bold')

                    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                    ax.set_title(f'State {state}: {metric.replace("_", " ").title()}',
                                fontsize=13, fontweight='bold')
                    ax.grid(alpha=0.3, axis='y')

                    # Store result
                    test_results.append({
                        'State': state,
                        'Metric': metric,
                        'W_mean': np.mean(w_vals),
                        'F_mean': np.mean(f_vals),
                        'p_value': pval,
                        'significant': pval < 0.05
                    })

                metric_idx += 1

        # Summary text
        n_sig = sum([r['significant'] for r in test_results])
        n_total = len(test_results)
        batch_effect = n_sig / n_total > 0.3 if n_total > 0 else False

        fig.text(0.5, 0.02,
                f'Batch Effect Test: {n_sig}/{n_total} metrics differ significantly (p<0.05)\n'
                f'Conclusion: {"BATCH EFFECT DETECTED - do not combine" if batch_effect else "No strong batch effect - can combine as WT controls"}',
                ha='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow' if batch_effect else 'lightgreen', alpha=0.5))

        fig.suptitle('Cross-Cohort WT Comparison: W+ vs F+ (Batch Effect Testing)',
                    fontsize=18, fontweight='bold')

        plt.savefig(self.output_dir / 'fig4_cross_cohort_W_plus_F_plus.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig4_cross_cohort_W_plus_F_plus.pdf',
                   bbox_inches='tight')
        plt.close()

        # Save test results
        results_df = pd.DataFrame(test_results)
        results_df.to_csv(self.output_dir / 'batch_effect_test_results.csv', index=False)

        print("✓ Created Figure 4: Cross-cohort W+ vs F+ comparison")
        print(f"  Batch effect test: {n_sig}/{n_total} metrics differ (p<0.05)")
        if batch_effect:
            print("  ⚠ BATCH EFFECT DETECTED - Do not combine cohorts")
        else:
            print("  ✓ No strong batch effect - Can combine as WT controls")
