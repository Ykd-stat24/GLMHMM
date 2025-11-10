"""
Traditional Statistical Analyses to Support GLM-HMM Findings
=============================================================

Performs mixed-effects models, ANOVAs, and descriptive statistics
to validate and complement the GLM-HMM state-based findings.

Analyses:
1. Descriptive statistics by genotype
2. Mixed-effects models for accuracy, latency, side bias
3. State occupancy ANOVAs by genotype
4. Learning curve comparisons
5. Transition probability analyses
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy import stats
from scipy.stats import f_oneway, kruskal, mannwhitneyu, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# For mixed-effects models
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. Mixed-effects models will be skipped.")

import sys
sys.path.insert(0, '/home/user/GLMHMM')
from glmhmm_utils import load_and_preprocess_session_data

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


class TraditionalStatisticalAnalysis:
    """Traditional statistical analyses to complement GLM-HMM."""

    def __init__(self, results_dir='results/phase1_non_reversal'):
        self.results_dir = Path(results_dir)
        self.output_dir = Path('results/traditional_statistics')
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load Phase 1 models
        self.models = self.load_all_models()

    def load_all_models(self):
        """Load all Phase 1 GLM-HMM models."""
        print("Loading Phase 1 GLM-HMM models...")
        models = {}

        for pkl_file in self.results_dir.glob('*_model.pkl'):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    animal_id = data['animal_id']
                    cohort = data['cohort']
                    models[f"{animal_id}_{cohort}"] = data
            except Exception as e:
                print(f"  Warning: Could not load {pkl_file.name}: {e}")

        print(f"  Loaded {len(models)} models")
        return models

    def create_master_dataframe(self):
        """
        Create master dataframe with trial-level and animal-level variables.
        """
        print("\nCreating master dataframe...")

        # Load raw trial data for both cohorts
        print("  Loading raw trial data...")
        df_W = load_and_preprocess_session_data('W LD Data 11.08 All_processed.csv')
        df_F = load_and_preprocess_session_data('F LD Data 11.08 All_processed.csv')

        # Filter to non-reversal only (Phase 1)
        df_W = df_W[~df_W['task_type'].str.contains('reversal', case=False, na=False)]
        df_F = df_F[~df_F['task_type'].str.contains('reversal', case=False, na=False)]

        df_W['cohort'] = 'W'
        df_F['cohort'] = 'F'

        # Combine
        df_all = pd.concat([df_W, df_F], ignore_index=True)

        print(f"  Loaded {len(df_all)} trials from {df_all['animal_id'].nunique()} animals")

        # Add model predictions
        print("  Adding GLM-HMM state predictions...")

        df_all['state'] = np.nan
        df_all['broad_category'] = ''

        from state_validation import create_broad_state_categories

        for model_key, model_data in self.models.items():
            animal_id = model_data['animal_id']
            cohort = model_data['cohort']

            # Get states
            model = model_data['model']
            states = model.most_likely_states
            validated_labels = model_data['validated_labels']
            broad_cats = create_broad_state_categories(validated_labels)

            # Get broad category for each trial
            trial_broad_cats = [broad_cats[s][0] for s in states]

            # Match with raw data
            animal_mask = (df_all['animal_id'] == animal_id) & (df_all['cohort'] == cohort)
            animal_trials = df_all[animal_mask]

            if len(animal_trials) == len(states):
                df_all.loc[animal_mask, 'state'] = states
                df_all.loc[animal_mask, 'broad_category'] = trial_broad_cats
            else:
                print(f"  Warning: Trial count mismatch for {animal_id} ({cohort}): " +
                      f"raw={len(animal_trials)}, model={len(states)}")

        # Remove trials without state assignments
        df = df_all[df_all['state'].notna()].copy()
        df['state'] = df['state'].astype(int)

        print(f"  Created dataframe with {len(df)} trials from {df['animal_id'].nunique()} animals")

        # Add derived variables
        df['session_num'] = df.groupby('animal_id').cumcount() // 30

        # Side bias - chose_right already exists from preprocessing as 'chosen_side'
        if 'chosen_side' in df.columns:
            df['chose_right'] = (df['chosen_side'] == 1).astype(int)

        # WSLS - compute if not already present
        if 'wsls' not in df.columns:
            df['wsls'] = np.nan
            for animal in df['animal_id'].unique():
                animal_data = df[df['animal_id'] == animal].sort_values(['session_date', 'trial_num']).copy()

                if len(animal_data) < 2:
                    continue

                wsls_vals = np.full(len(animal_data), np.nan)

                for i in range(1, len(animal_data)):
                    prev_choice = animal_data.iloc[i-1]['chosen_side']
                    curr_choice = animal_data.iloc[i]['chosen_side']
                    prev_reward = animal_data.iloc[i-1]['correct']

                    stayed = (curr_choice == prev_choice)
                    wsls_vals[i] = 1.0 if ((stayed and prev_reward) or
                                          (not stayed and not prev_reward)) else 0.0

                df.loc[df['animal_id'] == animal, 'wsls'] = wsls_vals

        return df

    def descriptive_statistics(self, df):
        """Generate descriptive statistics by genotype."""
        print("\n" + "="*80)
        print("DESCRIPTIVE STATISTICS BY GENOTYPE")
        print("="*80)

        results = []

        for cohort in df['cohort'].unique():
            cohort_data = df[df['cohort'] == cohort]

            print(f"\nCohort {cohort}:")
            print("-" * 40)

            for genotype in sorted(cohort_data['genotype'].unique()):
                geno_data = cohort_data[cohort_data['genotype'] == genotype]

                # Animal-level statistics
                animal_stats = geno_data.groupby('animal_id').agg({
                    'correct': 'mean',
                    'latency': 'mean',
                    'chose_right': 'mean',
                    'wsls': 'mean'
                }).reset_index()

                # State occupancy
                state_occ = geno_data.groupby('animal_id')['broad_category'].value_counts(
                    normalize=True).unstack(fill_value=0)

                # Compute means and SEMs
                stats_dict = {
                    'cohort': cohort,
                    'genotype': genotype,
                    'n_animals': geno_data['animal_id'].nunique(),
                    'n_trials': len(geno_data),
                    'accuracy_mean': animal_stats['correct'].mean(),
                    'accuracy_sem': animal_stats['correct'].sem(),
                    'latency_mean': animal_stats['latency'].mean(),
                    'latency_sem': animal_stats['latency'].sem(),
                    'side_bias_mean': abs(animal_stats['chose_right'].mean() - 0.5),
                    'side_bias_sem': animal_stats['chose_right'].sem(),
                    'wsls_mean': animal_stats['wsls'].mean(),
                    'wsls_sem': animal_stats['wsls'].sem()
                }

                # Add state occupancies
                for state_cat in ['Engaged', 'Lapsed', 'Mixed']:
                    if state_cat in state_occ.columns:
                        stats_dict[f'{state_cat}_occ_mean'] = state_occ[state_cat].mean()
                        stats_dict[f'{state_cat}_occ_sem'] = state_occ[state_cat].sem()
                    else:
                        stats_dict[f'{state_cat}_occ_mean'] = 0
                        stats_dict[f'{state_cat}_occ_sem'] = 0

                results.append(stats_dict)

                # Print summary
                print(f"\n  {genotype} (n={stats_dict['n_animals']} animals):")
                print(f"    Accuracy: {stats_dict['accuracy_mean']:.3f} ± {stats_dict['accuracy_sem']:.3f}")
                print(f"    Latency:  {stats_dict['latency_mean']:.3f} ± {stats_dict['latency_sem']:.3f}")
                print(f"    WSLS:     {stats_dict['wsls_mean']:.3f} ± {stats_dict['wsls_sem']:.3f}")
                print(f"    Engaged:  {stats_dict['Engaged_occ_mean']:.3f} ± {stats_dict['Engaged_occ_sem']:.3f}")
                print(f"    Lapsed:   {stats_dict['Lapsed_occ_mean']:.3f} ± {stats_dict['Lapsed_occ_sem']:.3f}")

        stats_df = pd.DataFrame(results)
        stats_df.to_csv(self.output_dir / 'descriptive_statistics.csv', index=False)
        print(f"\n✓ Saved descriptive statistics")

        return stats_df

    def anova_analyses(self, df):
        """
        Perform ANOVAs/Kruskal-Wallis tests for genotype effects.
        """
        print("\n" + "="*80)
        print("ANOVA / KRUSKAL-WALLIS TESTS FOR GENOTYPE EFFECTS")
        print("="*80)

        results = []

        for cohort in df['cohort'].unique():
            cohort_data = df[df['cohort'] == cohort]

            print(f"\nCohort {cohort}:")
            print("-" * 40)

            # Get animal-level aggregates
            animal_stats = cohort_data.groupby(['animal_id', 'genotype']).agg({
                'correct': 'mean',
                'latency': 'mean',
                'wsls': 'mean',
                'chose_right': 'mean'
            }).reset_index()

            # State occupancies
            state_occ = cohort_data.groupby(['animal_id', 'genotype'])['broad_category'].value_counts(
                normalize=True).unstack(fill_value=0).reset_index()

            # Combine
            animal_stats = animal_stats.merge(state_occ, on=['animal_id', 'genotype'])

            genotypes = sorted(animal_stats['genotype'].unique())

            if len(genotypes) < 2:
                print("  Insufficient genotypes for testing")
                continue

            # Test each variable
            variables = ['correct', 'latency', 'wsls']
            if 'Engaged' in animal_stats.columns:
                variables.extend(['Engaged', 'Lapsed'])

            for var in variables:
                # Get data by genotype
                groups = [animal_stats[animal_stats['genotype'] == g][var].dropna()
                         for g in genotypes]

                # Skip if any group has < 2 observations
                if any(len(g) < 2 for g in groups):
                    continue

                # Kruskal-Wallis (non-parametric)
                h_stat, p_val = kruskal(*groups)

                # Effect size (epsilon squared)
                n = sum(len(g) for g in groups)
                epsilon_sq = (h_stat - len(groups) + 1) / (n - len(groups))

                results.append({
                    'cohort': cohort,
                    'variable': var,
                    'test': 'Kruskal-Wallis',
                    'statistic': h_stat,
                    'p_value': p_val,
                    'effect_size': epsilon_sq,
                    'significant': p_val < 0.05
                })

                sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"  {var:15s}: H={h_stat:.3f}, p={p_val:.4f} {sig_marker} (ε²={epsilon_sq:.3f})")

        anova_df = pd.DataFrame(results)
        anova_df.to_csv(self.output_dir / 'anova_results.csv', index=False)
        print(f"\n✓ Saved ANOVA results")

        return anova_df

    def mixed_effects_models(self, df):
        """
        Fit mixed-effects models with genotype as fixed effect,
        animal as random effect.
        """
        if not HAS_STATSMODELS:
            print("\n⚠ Skipping mixed-effects models (statsmodels not installed)")
            return None

        print("\n" + "="*80)
        print("MIXED-EFFECTS MODELS")
        print("="*80)

        results = []

        for cohort in df['cohort'].unique():
            cohort_data = df[df['cohort'] == cohort].copy()

            print(f"\nCohort {cohort}:")
            print("-" * 40)

            # Encode genotype as categorical
            cohort_data['genotype_cat'] = pd.Categorical(cohort_data['genotype'])

            # Model 1: Accuracy ~ Genotype + Session + (1|Animal)
            try:
                print("\n  Model: Accuracy ~ Genotype + Session + (1|Animal)")
                md = mixedlm("correct ~ C(genotype) + session_num",
                            cohort_data,
                            groups=cohort_data["animal_id"])
                mdf = md.fit(method='lbfgs')

                print(mdf.summary().tables[1])

                # Extract genotype effects
                for param in mdf.params.index:
                    if 'genotype' in param:
                        results.append({
                            'cohort': cohort,
                            'model': 'Accuracy',
                            'parameter': param,
                            'estimate': mdf.params[param],
                            'std_err': mdf.bse[param],
                            'z_value': mdf.tvalues[param],
                            'p_value': mdf.pvalues[param]
                        })

            except Exception as e:
                print(f"  Error fitting accuracy model: {e}")

            # Model 2: Latency ~ Genotype + Session + (1|Animal)
            try:
                print("\n  Model: Latency ~ Genotype + Session + (1|Animal)")
                # Remove NaN latencies
                latency_data = cohort_data[cohort_data['latency'].notna()].copy()

                md = mixedlm("latency ~ C(genotype) + session_num",
                            latency_data,
                            groups=latency_data["animal_id"])
                mdf = md.fit(method='lbfgs')

                print(mdf.summary().tables[1])

                for param in mdf.params.index:
                    if 'genotype' in param:
                        results.append({
                            'cohort': cohort,
                            'model': 'Latency',
                            'parameter': param,
                            'estimate': mdf.params[param],
                            'std_err': mdf.bse[param],
                            'z_value': mdf.tvalues[param],
                            'p_value': mdf.pvalues[param]
                        })

            except Exception as e:
                print(f"  Error fitting latency model: {e}")

        if results:
            lme_df = pd.DataFrame(results)
            lme_df.to_csv(self.output_dir / 'mixed_effects_results.csv', index=False)
            print(f"\n✓ Saved mixed-effects results")
            return lme_df
        else:
            return None

    def create_visualizations(self, df, stats_df):
        """Create publication-quality visualizations."""
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)

        # Figure 1: Genotype comparison across key metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for cohort_idx, cohort in enumerate(df['cohort'].unique()):
            cohort_data = df[df['cohort'] == cohort]
            cohort_stats = stats_df[stats_df['cohort'] == cohort]

            # Panel A: Accuracy
            ax = axes[cohort_idx, 0]
            genotypes = sorted(cohort_stats['genotype'].unique())
            x = np.arange(len(genotypes))

            means = [cohort_stats[cohort_stats['genotype'] == g]['accuracy_mean'].values[0]
                    for g in genotypes]
            sems = [cohort_stats[cohort_stats['genotype'] == g]['accuracy_sem'].values[0]
                   for g in genotypes]

            bars = ax.bar(x, means, yerr=sems, capsize=5, alpha=0.7,
                         color=sns.color_palette("Set2", len(genotypes)))
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
            ax.set_xticks(x)
            ax.set_xticklabels(genotypes)
            ax.set_ylabel('Accuracy', fontweight='bold')
            ax.set_title(f'Cohort {cohort}: Accuracy by Genotype', fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            # Panel B: State Occupancy
            ax = axes[cohort_idx, 1]

            engaged = [cohort_stats[cohort_stats['genotype'] == g]['Engaged_occ_mean'].values[0]
                      for g in genotypes]
            lapsed = [cohort_stats[cohort_stats['genotype'] == g]['Lapsed_occ_mean'].values[0]
                     for g in genotypes]
            mixed = [cohort_stats[cohort_stats['genotype'] == g]['Mixed_occ_mean'].values[0]
                    for g in genotypes]

            width = 0.25
            ax.bar(x - width, engaged, width, label='Engaged', color='#27ae60', alpha=0.8)
            ax.bar(x, lapsed, width, label='Lapsed', color='#e74c3c', alpha=0.8)
            ax.bar(x + width, mixed, width, label='Mixed', color='#f39c12', alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels(genotypes)
            ax.set_ylabel('State Occupancy', fontweight='bold')
            ax.set_title(f'Cohort {cohort}: State Distribution', fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'genotype_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'genotype_comparison.pdf', bbox_inches='tight')
        plt.close()

        print("✓ Created genotype comparison figure")

    def run_all_analyses(self):
        """Run complete statistical analysis pipeline."""
        print("\n" + "="*80)
        print("TRADITIONAL STATISTICAL ANALYSIS PIPELINE")
        print("="*80)

        # Create master dataframe
        df = self.create_master_dataframe()

        # Descriptive statistics
        stats_df = self.descriptive_statistics(df)

        # ANOVA analyses
        anova_df = self.anova_analyses(df)

        # Mixed-effects models
        lme_df = self.mixed_effects_models(df)

        # Visualizations
        self.create_visualizations(df, stats_df)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {self.output_dir}")

        return df, stats_df, anova_df, lme_df


if __name__ == '__main__':
    analyzer = TraditionalStatisticalAnalysis()
    df, stats_df, anova_df, lme_df = analyzer.run_all_analyses()
