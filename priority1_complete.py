"""
Priority 1: Complete Core Poster Visualizations
================================================

All 4 essential figures for poster:
1. Engaged/Lapsed by genotype and task
2. Learning curves by genotype  
3. Side bias analysis for -/-
4. Cross-cohort W+ vs F+ comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy import stats
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/home/user/GLMHMM')
from state_validation import create_broad_state_categories
from glmhmm_utils import load_and_preprocess_session_data

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class CompletePriority1:
    """All Priority 1 visualizations in one class."""

    def __init__(self):
        self.results_dir = Path('results/phase1_non_reversal')
        self.output_dir = self.results_dir / 'priority1_complete'
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self, cohort, animals):
        """Load GLM-HMM results and trial data."""
        results = []

        # Load trial data
        data_file = f'{cohort} LD Data 11.08 All_processed.csv'
        print(f"  Loading {cohort} raw data...")
        trials = load_and_preprocess_session_data(data_file)
        print(f"    {len(trials)} trials, tasks: {trials['task_type'].value_counts().to_dict()}")

        # Load GLM-HMM results
        print(f"  Loading {cohort} GLM-HMM results...")
        for animal in animals:
            pkl = self.results_dir / f'{animal}_cohort{cohort}_model.pkl'
            if pkl.exists():
                with open(pkl, 'rb') as f:
                    data = pickle.load(f)
                    data['broad_categories'] = create_broad_state_categories(data['validated_labels'])
                    results.append(data)

        print(f"    {len(results)} animals loaded")
        return results, trials

    def run_all(self, results_W, trials_W, results_F, trials_F):
        """Generate all 4 Priority 1 figures."""
        print("\n" + "="*80)
        print("Generating all Priority 1 figures...")
        print("="*80)

        self.fig1_engaged_lapsed(results_W, trials_W, results_F, trials_F)
        self.fig2_learning_curves(results_W, trials_W, results_F, trials_F)
        self.fig3_side_bias(results_F, trials_F)
        self.fig4_cross_cohort(results_W, results_F)

        print("\n" + "="*80)
        print("✓ ALL PRIORITY 1 FIGURES COMPLETE!")
        print("="*80)
        print(f"\nOutput: {self.output_dir}")

    def fig1_engaged_lapsed(self, rW, tW, rF, tF):
        """Figure 1: Engaged/Lapsed by genotype and task."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))

        for col, (res, tri, cohort) in enumerate([(rW, tW, 'W'), (rF, tF, 'F')]):
            # Panel A: State categories
            ax = axes[0, col]
            geno_stats = {}

            for r in res:
                g = r['genotype']
                if g not in geno_stats:
                    geno_stats[g] = {'Engaged': 0, 'Lapsed': 0, 'Mixed': 0, 'Total': 0}
                for s in range(r['model'].n_states):
                    cat = r['broad_categories'][s][0]
                    geno_stats[g][cat] += 1
                    geno_stats[g]['Total'] += 1

            genotypes = sorted(geno_stats.keys())
            x = np.arange(len(genotypes))
            width = 0.25

            for i, (cat, color) in enumerate([('Engaged', '#27ae60'), ('Lapsed', '#e74c3c'), ('Mixed', '#f39c12')]):
                vals = [100 * geno_stats[g][cat] / geno_stats[g]['Total'] for g in genotypes]
                bars = ax.bar(x + i*width, vals, width, label=cat, color=color, alpha=0.8, edgecolor='black')

                for bar, val, g in zip(bars, vals, genotypes):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                               f'{val:.1f}%\n({geno_stats[g][cat]}/{geno_stats[g]["Total"]})',
                               ha='center', va='bottom', fontsize=9, fontweight='bold')

            ax.set_xlabel('Genotype', fontsize=14, fontweight='bold')
            ax.set_ylabel('Percentage of States', fontsize=14, fontweight='bold')
            ax.set_title(f'Cohort {cohort}: State Categories', fontsize=15, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(genotypes, fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(axis='y', alpha=0.3)

            # Panel B: Task-specific
            ax = axes[1, col]
            task_stats = {g: {'LD': [], 'PI': []} for g in genotypes}

            for r in res:
                g, aid = r['genotype'], r['animal_id']
                engaged_states = [s for s in range(r['model'].n_states) if r['broad_categories'][s][0] == 'Engaged']

                a_tri = tri[tri['animal_id'] == aid].copy()
                if len(a_tri) == 0: continue

                n = min(len(a_tri), len(r['model'].most_likely_states))
                a_tri = a_tri.iloc[:n].copy()
                a_tri['state'] = r['model'].most_likely_states[:n]

                for task in ['LD', 'PI']:
                    t_tri = a_tri[a_tri['task_type'] == task]
                    if len(t_tri) > 0:
                        eng_pct = 100 * len(t_tri[t_tri['state'].isin(engaged_states)]) / len(t_tri)
                        task_stats[g][task].append(eng_pct)

            x = np.arange(len(genotypes))
            width = 0.35

            ld_m = [np.mean(task_stats[g]['LD']) if task_stats[g]['LD'] else 0 for g in genotypes]
            ld_s = [np.std(task_stats[g]['LD'])/np.sqrt(len(task_stats[g]['LD'])) if task_stats[g]['LD'] else 0 for g in genotypes]
            pi_m = [np.mean(task_stats[g]['PI']) if task_stats[g]['PI'] else 0 for g in genotypes]
            pi_s = [np.std(task_stats[g]['PI'])/np.sqrt(len(task_stats[g]['PI'])) if task_stats[g]['PI'] else 0 for g in genotypes]

            ax.bar(x - width/2, ld_m, width, yerr=ld_s, label='LD', color='#3498db', alpha=0.8, edgecolor='black', capsize=5)
            ax.bar(x + width/2, pi_m, width, yerr=pi_s, label='PI', color='#9b59b6', alpha=0.8, edgecolor='black', capsize=5)

            ax.set_xlabel('Genotype', fontsize=14, fontweight='bold')
            ax.set_ylabel('% Trials in Engaged State', fontsize=14, fontweight='bold')
            ax.set_title(f'Cohort {cohort}: Task-Specific Engagement', fontsize=15, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(genotypes, fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle('State Categories: Genotype and Task Comparison', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_engaged_lapsed.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig1_engaged_lapsed.pdf', bbox_inches='tight')
        plt.close()

        print("✓ Figure 1: Engaged/Lapsed by genotype and task")

    def fig2_learning_curves(self, rW, tW, rF, tF):
        """Figure 2: Learning curves by genotype."""
        fig, axes = plt.subplots(2, 1, figsize=(18, 12))

        for ax, (res, tri, cohort) in zip(axes, [(rW, tW, 'W'), (rF, tF, 'F')]):
            genotypes = sorted(tri['genotype'].unique())
            colors = sns.color_palette("husl", len(genotypes))

            for g_idx, geno in enumerate(genotypes):
                g_tri = tri[tri['genotype'] == geno].copy()
                g_tri = g_tri.sort_values(['animal_id', 'session_date', 'trial_num'])
                g_tri['session'] = g_tri.groupby('animal_id').cumcount() // 30
                g_tri['rolling_acc'] = g_tri.groupby('animal_id')['correct'].transform(
                    lambda x: x.rolling(30, min_periods=1, center=True).mean())

                sess_stats = g_tri.groupby('session').agg({
                    'rolling_acc': ['mean', 'sem'],
                    'animal_id': 'nunique'
                }).reset_index()

                x = sess_stats['session']
                y = sess_stats['rolling_acc']['mean']
                sem = sess_stats['rolling_acc']['sem']
                n = sess_stats['animal_id']['nunique'].iloc[0]

                ax.plot(x, y, linewidth=3, color=colors[g_idx], label=f'{geno} (n={n})', alpha=0.9)
                ax.fill_between(x, y - sem, y + sem, color=colors[g_idx], alpha=0.2)

            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='Chance')
            ax.axhline(y=0.8, color='green', linestyle=':', alpha=0.6, linewidth=2, label='80% Criterion')

            ax.set_ylabel('Accuracy ± SEM', fontsize=14, fontweight='bold')
            ax.set_title(f'Cohort {cohort}: Learning Curves', fontsize=15, fontweight='bold')
            ax.legend(fontsize=11, loc='lower right')
            ax.grid(alpha=0.3)
            ax.set_ylim(0.3, 1.0)

        axes[-1].set_xlabel('Session (~30 trials)', fontsize=14, fontweight='bold')
        fig.suptitle('Genotype-Averaged Learning Curves', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig2_learning_curves.pdf', bbox_inches='tight')
        plt.close()

        print("✓ Figure 2: Learning curves")

    def fig3_side_bias(self, rF, tF):
        """Figure 3: Side bias for -/- genotype."""
        mm_res = [r for r in rF if r['genotype'] == '-/-']
        if not mm_res:
            print("  No -/- animals, skipping Fig 3")
            return

        mm_ids = [r['animal_id'] for r in mm_res]
        mm_tri = tF[tF['animal_id'].isin(mm_ids)].copy()

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Panel A: Position bias
        ax = axes[0, 0]
        if 'position' in mm_tri.columns:
            pos_stats = mm_tri.groupby('position').agg({
                'chosen_side': lambda x: (x == 1).sum() / len(x),
                'correct': 'mean'
            }).reset_index()

            ax.bar(pos_stats['position'], pos_stats['chosen_side'], alpha=0.6, color='#3498db')
            ax.axhline(y=0.5, color='gray', linestyle='--')
            ax.set_xlabel('Position', fontsize=12, fontweight='bold')
            ax.set_ylabel('P(Right)', fontsize=12, fontweight='bold')
            ax.set_title('-/- Genotype: Choice Bias by Position', fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3)

        # Panel B: Bias over time
        ax = axes[0, 1]
        mm_tri_s = mm_tri.sort_values(['animal_id', 'session_date'])
        mm_tri_s['block'] = mm_tri_s.groupby('animal_id').cumcount() // 100

        block_stats = mm_tri_s.groupby(['animal_id', 'block']).agg({
            'chosen_side': lambda x: abs((x==1).sum()/len(x) - 0.5)
        }).reset_index()

        block_avg = block_stats.groupby('block').agg({'chosen_side': ['mean', 'sem']}).reset_index()
        x = block_avg['block']
        y = block_avg['chosen_side']['mean']
        sem = block_avg['chosen_side']['sem']

        ax.plot(x, y, linewidth=3, color='#e74c3c')
        ax.fill_between(x, y-sem, y+sem, color='#e74c3c', alpha=0.2)
        ax.set_xlabel('Block (×100 trials)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Side Bias', fontsize=12, fontweight='bold')
        ax.set_title('-/- Genotype: Side Bias Over Time', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)

        # Panels C-D: Individual examples
        for i, (aid, ares) in enumerate(list(zip(mm_ids, mm_res))[:2]):
            ax = axes[1, i]
            a_tri = mm_tri[mm_tri['animal_id'] == aid].copy().sort_values('trial_num')
            a_tri['bias_roll'] = a_tri['chosen_side'].rolling(50, min_periods=1).apply(
                lambda x: abs((x==1).sum()/len(x) - 0.5))

            ax.plot(a_tri.index, a_tri['bias_roll'], linewidth=2, color='#9b59b6', alpha=0.8)
            ax.set_xlabel('Trial', fontsize=11, fontweight='bold')
            ax.set_ylabel('Side Bias', fontsize=11, fontweight='bold')

            states = [ares['broad_categories'][s][0] for s in range(ares['model'].n_states)]
            ax.set_title(f'{aid}\nStates: {", ".join(states)}', fontsize=12)
            ax.grid(alpha=0.3)

        fig.suptitle('-/- Genotype: Detailed Side Bias Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_side_bias.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig3_side_bias.pdf', bbox_inches='tight')
        plt.close()

        print("✓ Figure 3: Side bias for -/-")

    def fig4_cross_cohort(self, rW, rF):
        """Figure 4: W+ vs F+ batch effect test."""
        w_plus = [r for r in rW if r['genotype'] == '+']
        f_plus = [r for r in rF if r['genotype'] == '+']

        if not w_plus or not f_plus:
            print(f"  Insufficient WT (W+:{len(w_plus)}, F+:{len(f_plus)})")
            return

        print(f"\n  Batch test: W+ (n={len(w_plus)}) vs F+ (n={len(f_plus)})")

        metrics = ['accuracy', 'wsls_ratio', 'side_bias', 'latency_cv']
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()

        test_results = []
        idx = 0

        for state in range(3):
            for metric in metrics:
                ax = axes[idx]

                w_vals = []
                f_vals = []

                for r in w_plus:
                    sd = r['state_metrics'][r['state_metrics']['state'] == state]
                    if len(sd) > 0 and metric in sd.columns:
                        w_vals.append(sd[metric].values[0])

                for r in f_plus:
                    sd = r['state_metrics'][r['state_metrics']['state'] == state]
                    if len(sd) > 0 and metric in sd.columns:
                        f_vals.append(sd[metric].values[0])

                if w_vals and f_vals:
                    stat, pval = mannwhitneyu(w_vals, f_vals, alternative='two-sided')

                    df = pd.DataFrame({
                        'Value': w_vals + f_vals,
                        'Cohort': ['W+']*len(w_vals) + ['F+']*len(f_vals)
                    })

                    sns.violinplot(data=df, x='Cohort', y='Value', ax=ax, inner=None, alpha=0.6)
                    sns.swarmplot(data=df, x='Cohort', y='Value', ax=ax, color='black', alpha=0.6, size=5)

                    sig = '***' if pval<0.001 else '**' if pval<0.01 else '*' if pval<0.05 else 'n.s.'
                    col = 'red' if pval<0.05 else 'green'

                    ax.text(0.5, 0.95, f'p={pval:.4f} {sig}', transform=ax.transAxes,
                           ha='center', va='top', bbox=dict(boxstyle='round', facecolor=col, alpha=0.3),
                           fontsize=10, fontweight='bold')

                    ax.set_ylabel(metric.replace('_',' ').title(), fontsize=12, fontweight='bold')
                    ax.set_title(f'State {state}: {metric.replace("_"," ").title()}', fontsize=13, fontweight='bold')
                    ax.grid(alpha=0.3, axis='y')

                    test_results.append({
                        'State': state, 'Metric': metric,
                        'W_mean': np.mean(w_vals), 'F_mean': np.mean(f_vals),
                        'p_value': pval, 'significant': pval<0.05
                    })

                idx += 1

        n_sig = sum([r['significant'] for r in test_results])
        n_tot = len(test_results)
        batch = n_sig/n_tot > 0.3

        fig.text(0.5, 0.02,
                f'Batch Effect Test: {n_sig}/{n_tot} metrics differ (p<0.05)\n'
                f'Conclusion: {"BATCH EFFECT - do not combine" if batch else "No batch effect - can combine as WT controls"}',
                ha='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow' if batch else 'lightgreen', alpha=0.5))

        fig.suptitle('Cross-Cohort WT Comparison: W+ vs F+', fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(self.output_dir / 'fig4_cross_cohort.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig4_cross_cohort.pdf', bbox_inches='tight')
        plt.close()

        pd.DataFrame(test_results).to_csv(self.output_dir / 'batch_test.csv', index=False)

        print(f"✓ Figure 4: Cross-cohort (batch: {n_sig}/{n_tot} differ)")
        if batch:
            print("  ⚠ BATCH EFFECT DETECTED")
        else:
            print("  ✓ No batch effect")


def main():
    print("="*80)
    print("PRIORITY 1: COMPLETE VISUALIZATIONS")
    print("="*80)

    viz = CompletePriority1()

    animals_W = [f'c{c}m{m}' for c in range(1,5) for m in range(1,6) if not (c==1 and m==5)]
    animals_F = [str(i) for i in [11,12,13,14,21,22,23,24,25,31,32,33,34,41,42,51,52,
                                   61,62,63,64,71,72,73,81,82,83,84,91,92,93,101,102,103,104]]

    print("\nCohort W:")
    rW, tW = viz.load_data('W', animals_W)

    print("\nCohort F:")
    rF, tF = viz.load_data('F', animals_F)

    viz.run_all(rW, tW, rF, tF)


if __name__ == '__main__':
    main()
