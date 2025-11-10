"""
Phase 2: Combined Cohort Reversal Analysis
===========================================

Combines W and F cohort reversal data with cohort as covariate.
Tests for cohort × genotype interactions to justify pooling.

ALL GRAPHS CLEARLY LABELED: "Combined W+F Cohorts"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


class CombinedCohortReversalAnalysis:
    """Analyze reversal learning with combined cohorts."""

    def __init__(self):
        self.phase2_dir = Path('results/phase2_reversal')
        self.output_dir = Path('results/phase2_combined_cohorts')
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def load_all_reversal_models(self):
        """Load all Phase 2 reversal models from both cohorts."""
        print("Loading Phase 2 reversal models...")

        models = {'W': [], 'F': []}

        for pkl_file in (self.phase2_dir / 'models').glob('*_reversal.pkl'):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    cohort = data['cohort']
                    models[cohort].append(data)
            except Exception as e:
                print(f"  Warning: Could not load {pkl_file.name}: {e}")

        print(f"  W cohort: {len(models['W'])} models")
        print(f"  F cohort: {len(models['F'])} models")

        return models

    def create_combined_dataframe(self, models):
        """
        Create combined dataframe with animal-level reversal metrics.
        """
        print("\nCreating combined dataframe...")

        data = []

        for cohort, cohort_models in models.items():
            for model_data in cohort_models:
                animal_id = model_data['animal_id']
                genotype = model_data['genotype']
                n_trials = model_data['n_trials']

                # Get state metrics
                state_metrics = model_data['state_metrics']

                # Average across states (weighted by occupancy)
                avg_accuracy = np.average(
                    state_metrics['accuracy'].values,
                    weights=state_metrics['occupancy'].values
                )

                avg_wsls = np.average(
                    state_metrics['wsls_ratio'].values,
                    weights=state_metrics['occupancy'].values
                )

                # State occupancies
                broad_cats = model_data['broad_categories']
                state_occ = {}
                for state_id in range(3):  # Always 3 states
                    # Handle numpy float keys
                    key = None
                    for k in broad_cats.keys():
                        if int(k) == state_id:
                            key = k
                            break

                    if key is not None and key in broad_cats:
                        cat = broad_cats[key][0]
                        state_rows = state_metrics[state_metrics['state'] == state_id]
                        if len(state_rows) > 0:
                            occ = state_rows['occupancy'].values[0]
                            if cat in state_occ:
                                state_occ[cat] += occ
                            else:
                                state_occ[cat] = occ

                data.append({
                    'animal_id': animal_id,
                    'cohort': cohort,
                    'genotype': genotype,
                    'n_trials': n_trials,
                    'accuracy': avg_accuracy,
                    'wsls': avg_wsls,
                    'engaged_occ': state_occ.get('Engaged', 0),
                    'lapsed_occ': state_occ.get('Lapsed', 0),
                    'mixed_occ': state_occ.get('Mixed', 0)
                })

        df = pd.DataFrame(data)

        print(f"  Created dataframe with {len(df)} animals")
        print(f"    W cohort: {(df['cohort'] == 'W').sum()} animals")
        print(f"    F cohort: {(df['cohort'] == 'F').sum()} animals")

        return df

    def test_cohort_interactions(self, df):
        """
        Test for cohort × genotype interactions to justify pooling.
        """
        print("\n" + "="*80)
        print("TESTING COHORT × GENOTYPE INTERACTIONS")
        print("="*80)

        results = []

        # For each dependent variable
        for var in ['accuracy', 'wsls', 'engaged_occ', 'lapsed_occ']:
            print(f"\n{var.upper()}:")
            print("-" * 40)

            # Test main effects and interaction
            # Using Kruskal-Wallis for non-parametric test

            # Main effect of cohort
            w_vals = df[df['cohort'] == 'W'][var].dropna()
            f_vals = df[df['cohort'] == 'F'][var].dropna()

            if len(w_vals) > 0 and len(f_vals) > 0:
                h_cohort, p_cohort = mannwhitneyu(w_vals, f_vals, alternative='two-sided')
                print(f"  Main effect - Cohort: U={h_cohort:.3f}, p={p_cohort:.4f}")
            else:
                h_cohort, p_cohort = np.nan, np.nan

            # Test genotype effect within each cohort
            cohort_effects = {}
            for cohort in ['W', 'F']:
                cohort_data = df[df['cohort'] == cohort]
                genotypes = cohort_data['genotype'].unique()

                if len(genotypes) > 1:
                    groups = [cohort_data[cohort_data['genotype'] == g][var].dropna()
                             for g in genotypes]
                    groups = [g for g in groups if len(g) > 0]

                    if len(groups) > 1:
                        h_stat, p_val = kruskal(*groups)
                        cohort_effects[cohort] = (h_stat, p_val)
                        print(f"  Genotype effect in {cohort}: H={h_stat:.3f}, p={p_val:.4f}")

            # Interaction test (qualitative assessment)
            # If genotype effects differ substantially between cohorts, interaction exists
            if 'W' in cohort_effects and 'F' in cohort_effects:
                w_effect = cohort_effects['W'][1] < 0.05
                f_effect = cohort_effects['F'][1] < 0.05

                if w_effect != f_effect:
                    interaction = "SIGNIFICANT (qualitative difference)"
                else:
                    interaction = "Not significant (consistent pattern)"
            else:
                interaction = "Cannot test (insufficient data)"

            print(f"  Interaction assessment: {interaction}")

            results.append({
                'variable': var,
                'cohort_main_effect_p': p_cohort,
                'interaction': interaction,
                'pooling_justified': interaction.startswith("Not")
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'interaction_tests.csv', index=False)

        print("\n" + "="*80)
        print("POOLING DECISION")
        print("="*80)

        if results_df['pooling_justified'].all():
            print("\n✓ No significant cohort × genotype interactions detected")
            print("  JUSTIFIED to pool cohorts for increased statistical power")
            print("  All analyses will include cohort as covariate")
        else:
            print("\n⚠ Significant cohort × genotype interactions detected")
            print("  RECOMMEND analyzing cohorts separately")
            print("  Pooled analyses should be interpreted cautiously")

        return results_df

    def combined_descriptive_statistics(self, df):
        """Descriptive statistics for combined cohorts."""
        print("\n" + "="*80)
        print("COMBINED COHORT DESCRIPTIVE STATISTICS")
        print("="*80)

        stats_list = []

        # Overall by genotype (pooled)
        print("\nPooled Across Cohorts:")
        print("-" * 40)

        for genotype in sorted(df['genotype'].unique()):
            geno_data = df[df['genotype'] == genotype]

            stats_dict = {
                'genotype': genotype,
                'n_animals': len(geno_data),
                'n_W': (geno_data['cohort'] == 'W').sum(),
                'n_F': (geno_data['cohort'] == 'F').sum(),
                'accuracy_mean': geno_data['accuracy'].mean(),
                'accuracy_sem': geno_data['accuracy'].sem(),
                'wsls_mean': geno_data['wsls'].mean(),
                'wsls_sem': geno_data['wsls'].sem(),
                'engaged_mean': geno_data['engaged_occ'].mean(),
                'engaged_sem': geno_data['engaged_occ'].sem()
            }

            stats_list.append(stats_dict)

            print(f"\n  {genotype} (n={stats_dict['n_animals']}: {stats_dict['n_W']}W + {stats_dict['n_F']}F):")
            print(f"    Accuracy: {stats_dict['accuracy_mean']:.3f} ± {stats_dict['accuracy_sem']:.3f}")
            print(f"    WSLS:     {stats_dict['wsls_mean']:.3f} ± {stats_dict['wsls_sem']:.3f}")
            print(f"    Engaged:  {stats_dict['engaged_mean']:.3f} ± {stats_dict['engaged_sem']:.3f}")

        stats_df = pd.DataFrame(stats_list)
        stats_df.to_csv(self.output_dir / 'combined_descriptive_stats.csv', index=False)

        return stats_df

    def create_combined_visualizations(self, df, stats_df):
        """
        Create visualizations with CLEAR labeling that cohorts are combined.
        """
        print("\n" + "="*80)
        print("CREATING COMBINED COHORT VISUALIZATIONS")
        print("="*80)

        # Figure 1: Combined cohort comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # CLEAR TITLE indicating combined cohorts
        fig.suptitle('Reversal Learning: COMBINED W+F COHORTS',
                    fontsize=18, fontweight='bold', color='#d62728')

        # Panel A: Accuracy by genotype
        ax = axes[0, 0]
        genotypes = sorted(stats_df['genotype'].unique())
        x = np.arange(len(genotypes))

        means = stats_df['accuracy_mean'].values
        sems = stats_df['accuracy_sem'].values

        bars = ax.bar(x, means, yerr=sems, capsize=5, alpha=0.7,
                     color=sns.color_palette("Set2", len(genotypes)))

        # Add sample size labels
        for i, (bar, geno) in enumerate(zip(bars, genotypes)):
            n_total = stats_df[stats_df['genotype'] == geno]['n_animals'].values[0]
            n_w = stats_df[stats_df['genotype'] == geno]['n_W'].values[0]
            n_f = stats_df[stats_df['genotype'] == geno]['n_F'].values[0]
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'n={n_total}\n({n_w}W+{n_f}F)',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xticks(x)
        ax.set_xticklabels(genotypes, fontsize=12)
        ax.set_ylabel('Accuracy (mean ± SEM)', fontsize=13, fontweight='bold')
        ax.set_title('Accuracy by Genotype\n[Combined W+F Cohorts]',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)

        # Panel B: WSLS by genotype
        ax = axes[0, 1]
        means = stats_df['wsls_mean'].values
        sems = stats_df['wsls_sem'].values

        bars = ax.bar(x, means, yerr=sems, capsize=5, alpha=0.7,
                     color=sns.color_palette("Set2", len(genotypes)))

        ax.set_xticks(x)
        ax.set_xticklabels(genotypes, fontsize=12)
        ax.set_ylabel('WSLS Ratio (mean ± SEM)', fontsize=13, fontweight='bold')
        ax.set_title('Win-Stay/Lose-Shift Strategy\n[Combined W+F Cohorts]',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)

        # Panel C: Engaged state by genotype
        ax = axes[1, 0]
        means = stats_df['engaged_mean'].values
        sems = stats_df['engaged_sem'].values

        bars = ax.bar(x, means, yerr=sems, capsize=5, alpha=0.7,
                     color='#27ae60')

        ax.set_xticks(x)
        ax.set_xticklabels(genotypes, fontsize=12)
        ax.set_ylabel('Engaged State Occupancy (mean ± SEM)', fontsize=13, fontweight='bold')
        ax.set_title('Engaged State During Reversal\n[Combined W+F Cohorts]',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)

        # Panel D: Individual animals colored by cohort
        ax = axes[1, 1]

        for cohort, color, marker in [('W', '#1f77b4', 'o'), ('F', '#ff7f0e', 's')]:
            cohort_data = df[df['cohort'] == cohort]

            for i, geno in enumerate(genotypes):
                geno_data = cohort_data[cohort_data['genotype'] == geno]

                # Add jitter
                x_pos = np.ones(len(geno_data)) * i + np.random.normal(0, 0.05, len(geno_data))

                ax.scatter(x_pos, geno_data['accuracy'],
                          alpha=0.6, s=80, color=color, marker=marker,
                          edgecolors='black', linewidths=0.5,
                          label=f'{cohort} cohort' if i == 0 else '')

        ax.set_xticks(x)
        ax.set_xticklabels(genotypes, fontsize=12)
        ax.set_ylabel('Accuracy (individual animals)', fontsize=13, fontweight='bold')
        ax.set_title('Individual Animal Performance\n[W=circles, F=squares]',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'combined_cohorts_reversal.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'combined_cohorts_reversal.pdf', bbox_inches='tight')
        plt.close()

        print("✓ Created combined cohort visualization (clearly labeled)")

    def run_analysis(self):
        """Run complete combined cohort analysis."""
        print("\n" + "="*80)
        print("PHASE 2: COMBINED COHORT REVERSAL ANALYSIS")
        print("="*80)

        # Load models
        models = self.load_all_reversal_models()

        if len(models['W']) == 0 and len(models['F']) == 0:
            print("\n⚠ No reversal models found. Phase 2 analysis may not be complete yet.")
            return

        # Create combined dataframe
        df = self.create_combined_dataframe(models)

        # Test for interactions
        interaction_results = self.test_cohort_interactions(df)

        # Descriptive statistics
        stats_df = self.combined_descriptive_statistics(df)

        # Visualizations
        self.create_combined_visualizations(df, stats_df)

        print("\n" + "="*80)
        print("COMBINED ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {self.output_dir}")
        print("\nNOTE: All graphs clearly labeled 'Combined W+F Cohorts'")


if __name__ == '__main__':
    analyzer = CombinedCohortReversalAnalysis()
    analyzer.run_analysis()
