"""
Phase 2 Reversal Learning Visualizations
=========================================

Creates comprehensive visualizations for reversal learning analysis,
mirroring Phase 1 visualizations but adapted for reversal paradigms.

Figures Generated:
1. State characteristics during reversal (Engaged/Lapsed/Mixed)
2. Psychometric learning curves (pre-reversal vs post-reversal)
3. Reversal adaptation by genotype
4. Cross-cohort comparisons
5. Performance trajectories through reversal
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

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class Phase2Visualizations:
    """Create comprehensive Phase 2 reversal learning visualizations."""

    def __init__(self):
        self.results_dir = Path('results/phase2_reversal/visualizations')
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.models_dir = Path('results/phase2_reversal/models')

    def load_all_models(self):
        """Load all Phase 2 reversal models."""
        print("Loading Phase 2 reversal models...")
        models = {'W': {}, 'F': {}}

        for pkl_file in self.models_dir.glob('*_reversal.pkl'):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                animal_id = data['animal_id']
                cohort = data['cohort']
                models[cohort][animal_id] = data
            except Exception as e:
                print(f"  Warning: Could not load {pkl_file.name}: {e}")

        print(f"  Loaded W cohort: {len(models['W'])} animals")
        print(f"  Loaded F cohort: {len(models['F'])} animals")
        return models

    def create_figure1_state_characteristics(self, models):
        """
        Figure 1: State Characteristics During Reversal
        Similar to Phase 1 engaged/lapsed analysis
        """
        print("\nCreating Figure 1: State Characteristics During Reversal...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Phase 2: Behavioral State Characteristics During Reversal Learning',
                    fontsize=16, fontweight='bold')

        # Collect data for all animals
        data = []
        for cohort in ['W', 'F']:
            for animal_id, model_data in models[cohort].items():
                state_metrics = model_data.get('state_metrics')
                if state_metrics is None or not isinstance(state_metrics, pd.DataFrame):
                    continue

                genotype = model_data['genotype']
                broad_cats = model_data.get('broad_categories', {})

                for _, row in state_metrics.iterrows():
                    state_id = int(row['state'])

                    # Get broad category
                    cat = 'Unknown'
                    for k, v in broad_cats.items():
                        if int(k) == state_id:
                            cat = v[0]
                            break

                    data.append({
                        'animal_id': animal_id,
                        'cohort': cohort,
                        'genotype': genotype,
                        'state': state_id,
                        'category': cat,
                        'accuracy': row['accuracy'],
                        'wsls': row['wsls_ratio'],
                        'occupancy': row['occupancy'],
                        'perseveration': row.get('perseveration_ratio', 0),
                        'win_stay': row.get('win_stay', 0),
                        'lose_shift': row.get('lose_shift', 0)
                    })

        df = pd.DataFrame(data)

        # Plot 1: Accuracy by state category
        ax = axes[0, 0]
        sns.boxplot(data=df, x='category', y='accuracy', hue='genotype', ax=ax)
        ax.set_title('Accuracy by State Category', fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('State Category')
        ax.legend(title='Genotype', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot 2: WSLS by state category
        ax = axes[0, 1]
        sns.boxplot(data=df, x='category', y='wsls', hue='genotype', ax=ax)
        ax.set_title('Win-Stay-Lose-Shift by State Category', fontweight='bold')
        ax.set_ylabel('WSLS Ratio')
        ax.set_xlabel('State Category')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.legend(title='Genotype', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot 3: State occupancy by genotype
        ax = axes[0, 2]
        occ_summary = df.groupby(['genotype', 'category'])['occupancy'].mean().unstack()
        occ_summary.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
        ax.set_title('State Occupancy by Genotype', fontweight='bold')
        ax.set_ylabel('Mean Occupancy')
        ax.set_xlabel('Genotype')
        ax.legend(title='State Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Plot 4: Perseveration comparison
        ax = axes[1, 0]
        sns.violinplot(data=df, x='category', y='perseveration', hue='genotype', ax=ax)
        ax.set_title('Perseveration by State', fontweight='bold')
        ax.set_ylabel('Perseveration Ratio')
        ax.set_xlabel('State Category')

        # Plot 5: Win-Stay behavior
        ax = axes[1, 1]
        sns.violinplot(data=df, x='category', y='win_stay', hue='genotype', ax=ax)
        ax.set_title('Win-Stay Behavior by State', fontweight='bold')
        ax.set_ylabel('Win-Stay Probability')
        ax.set_xlabel('State Category')

        # Plot 6: Lose-Shift behavior
        ax = axes[1, 2]
        sns.violinplot(data=df, x='category', y='lose_shift', hue='genotype', ax=ax)
        ax.set_title('Lose-Shift Behavior by State', fontweight='bold')
        ax.set_ylabel('Lose-Shift Probability')
        ax.set_xlabel('State Category')

        plt.tight_layout()
        fig.savefig(self.results_dir / 'fig1_reversal_state_characteristics.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.results_dir / 'fig1_reversal_state_characteristics.pdf', bbox_inches='tight')
        print("  ✓ Saved Figure 1")
        plt.close()

    def create_figure2_psychometric_curves(self, models):
        """
        Figure 2: Psychometric Learning Curves Through Reversal
        Shows performance trajectories before and after reversal point
        """
        print("\nCreating Figure 2: Psychometric Learning Curves...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Phase 2: Psychometric Curves Through Reversal Learning',
                    fontsize=16, fontweight='bold')

        # Collect trial-by-trial data
        all_data = []
        for cohort in ['W', 'F']:
            for animal_id, model_data in models[cohort].items():
                # Get trial data
                y = model_data.get('y')
                X = model_data.get('X')
                states = model_data['model'].most_likely_states
                metadata = model_data.get('metadata', {})
                genotype = model_data['genotype']

                if y is None or len(y) == 0:
                    continue

                # Create trial-level dataframe
                for trial_idx in range(len(y)):
                    all_data.append({
                        'animal_id': animal_id,
                        'cohort': cohort,
                        'genotype': genotype,
                        'trial_num': trial_idx,
                        'correct': int(y[trial_idx]),
                        'state': states[trial_idx],
                        'session': metadata.get('session', [None] * len(y))[trial_idx] if 'session' in metadata else None
                    })

        df = pd.DataFrame(all_data)

        # Plot 1: Overall performance curves by genotype
        ax = axes[0, 0]
        for genotype in df['genotype'].unique():
            geno_df = df[df['genotype'] == genotype]
            # Rolling average
            window = 50
            rolling_acc = geno_df.groupby('trial_num')['correct'].mean().rolling(window=window, center=True).mean()
            ax.plot(rolling_acc.index, rolling_acc.values, label=f'{genotype} (n={geno_df["animal_id"].nunique()})',
                   linewidth=2, alpha=0.8)

        ax.set_title('Performance Throughout Reversal', fontweight='bold')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Accuracy (50-trial rolling average)')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Performance by state
        ax = axes[0, 1]
        for state in sorted(df['state'].unique()):
            state_df = df[df['state'] == state]
            window = 50
            rolling_acc = state_df.groupby('trial_num')['correct'].mean().rolling(window=window, center=True).mean()
            ax.plot(rolling_acc.index, rolling_acc.values, label=f'State {state}',
                   linewidth=2, alpha=0.8)

        ax.set_title('Performance by Behavioral State', fontweight='bold')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Accuracy (50-trial rolling average)')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Cohort comparison
        ax = axes[1, 0]
        for cohort in ['W', 'F']:
            cohort_df = df[df['cohort'] == cohort]
            window = 50
            rolling_acc = cohort_df.groupby('trial_num')['correct'].mean().rolling(window=window, center=True).mean()
            ax.plot(rolling_acc.index, rolling_acc.values,
                   label=f'Cohort {cohort} (n={cohort_df["animal_id"].nunique()})',
                   linewidth=2, alpha=0.8)

        ax.set_title('Performance by Cohort', fontweight='bold')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Accuracy (50-trial rolling average)')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Early vs Late reversal performance
        ax = axes[1, 1]
        # Split trials into quartiles
        df['trial_quartile'] = pd.qcut(df['trial_num'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        quartile_summary = df.groupby(['genotype', 'trial_quartile'])['correct'].mean().unstack()
        quartile_summary.plot(kind='bar', ax=ax, colormap='viridis')
        ax.set_title('Performance Across Reversal Stages', fontweight='bold')
        ax.set_ylabel('Mean Accuracy')
        ax.set_xlabel('Genotype')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.legend(title='Reversal Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        fig.savefig(self.results_dir / 'fig2_psychometric_curves.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.results_dir / 'fig2_psychometric_curves.pdf', bbox_inches='tight')
        print("  ✓ Saved Figure 2")
        plt.close()

    def create_figure3_reversal_adaptation(self, models):
        """
        Figure 3: Reversal Adaptation by Genotype
        """
        print("\nCreating Figure 3: Reversal Adaptation Analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Phase 2: Reversal Adaptation by Genotype',
                    fontsize=16, fontweight='bold')

        # Load Phase 1 vs Phase 2 comparison data
        comparison_files = {
            'W': Path('results/phase2_reversal/summary/W_phase1_vs_phase2_comparison.csv'),
            'F': Path('results/phase2_reversal/summary/F_phase1_vs_phase2_comparison.csv')
        }

        comp_data = []
        for cohort, file_path in comparison_files.items():
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['cohort'] = cohort
                comp_data.append(df)

        if comp_data:
            comp_df = pd.concat(comp_data, ignore_index=True)

            # Plot 1: Accuracy change by genotype
            ax = axes[0, 0]
            sns.boxplot(data=comp_df, x='genotype', y='accuracy_change', ax=ax)
            sns.swarmplot(data=comp_df, x='genotype', y='accuracy_change',
                         color='black', alpha=0.5, size=4, ax=ax)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='No change')
            ax.set_title('Accuracy Change: Phase 1 → Phase 2', fontweight='bold')
            ax.set_ylabel('Δ Accuracy (Phase 2 - Phase 1)')
            ax.set_xlabel('Genotype')
            ax.legend()

            # Plot 2: WSLS change by genotype
            ax = axes[0, 1]
            sns.boxplot(data=comp_df, x='genotype', y='wsls_change', ax=ax)
            sns.swarmplot(data=comp_df, x='genotype', y='wsls_change',
                         color='black', alpha=0.5, size=4, ax=ax)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='No change')
            ax.set_title('WSLS Change: Phase 1 → Phase 2', fontweight='bold')
            ax.set_ylabel('Δ WSLS (Phase 2 - Phase 1)')
            ax.set_xlabel('Genotype')
            ax.legend()

            # Plot 3: Phase 1 vs Phase 2 accuracy scatter
            ax = axes[1, 0]
            for genotype in comp_df['genotype'].unique():
                geno_df = comp_df[comp_df['genotype'] == genotype]
                ax.scatter(geno_df['p1_accuracy'], geno_df['p2_accuracy'],
                          label=f'{genotype} (n={len(geno_df)})',
                          s=100, alpha=0.6)

            # Unity line
            max_val = max(comp_df['p1_accuracy'].max(), comp_df['p2_accuracy'].max())
            min_val = min(comp_df['p1_accuracy'].min(), comp_df['p2_accuracy'].min())
            ax.plot([min_val, max_val], [min_val, max_val],
                   'k--', linewidth=2, alpha=0.5, label='Unity')

            ax.set_title('Phase 1 vs Phase 2 Accuracy', fontweight='bold')
            ax.set_xlabel('Phase 1 Accuracy (Non-Reversal)')
            ax.set_ylabel('Phase 2 Accuracy (Reversal)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 4: Correlation analysis
            ax = axes[1, 1]
            # Calculate reversal deficit (how much worse in Phase 2)
            comp_df['reversal_deficit'] = comp_df['p1_accuracy'] - comp_df['p2_accuracy']

            sns.violinplot(data=comp_df, x='genotype', y='reversal_deficit', ax=ax)
            sns.swarmplot(data=comp_df, x='genotype', y='reversal_deficit',
                         color='black', alpha=0.5, size=4, ax=ax)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax.set_title('Reversal Learning Deficit by Genotype', fontweight='bold')
            ax.set_ylabel('Reversal Deficit\n(Phase 1 Acc - Phase 2 Acc)')
            ax.set_xlabel('Genotype')

            # Add statistics
            for i, genotype in enumerate(comp_df['genotype'].unique()):
                geno_data = comp_df[comp_df['genotype'] == genotype]['reversal_deficit']
                mean_deficit = geno_data.mean()
                ax.text(i, ax.get_ylim()[1] * 0.9,
                       f'μ={mean_deficit:.3f}',
                       ha='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        fig.savefig(self.results_dir / 'fig3_reversal_adaptation.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.results_dir / 'fig3_reversal_adaptation.pdf', bbox_inches='tight')
        print("  ✓ Saved Figure 3")
        plt.close()

    def create_figure4_cross_cohort_combined(self, models):
        """
        Figure 4: Cross-Cohort Reversal Analysis
        """
        print("\nCreating Figure 4: Cross-Cohort Analysis...")

        # Load combined cohort data if available
        combined_file = Path('results/phase2_combined_cohorts/combined_descriptive_stats.csv')

        if combined_file.exists():
            combined_df = pd.read_csv(combined_file)

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Phase 2: Cross-Cohort Reversal Learning (Combined W+F)',
                        fontsize=16, fontweight='bold', color='#d62728')

            # Plot 1: Accuracy comparison
            ax = axes[0, 0]
            combined_df.plot(x='genotype', y='accuracy_mean', kind='bar',
                           yerr='accuracy_sem', ax=ax, capsize=5, color='steelblue')
            ax.set_title('[Combined W+F] Accuracy by Genotype', fontweight='bold')
            ax.set_ylabel('Mean Accuracy')
            ax.set_xlabel('Genotype')
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            # Add sample sizes
            for i, (idx, row) in enumerate(combined_df.iterrows()):
                ax.text(i, row['accuracy_mean'] + row['accuracy_sem'] + 0.02,
                       f"n={row['n_animals']}\n({row['n_W']}W+{row['n_F']}F)",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

            # Plot 2: WSLS comparison
            ax = axes[0, 1]
            combined_df.plot(x='genotype', y='wsls_mean', kind='bar',
                           yerr='wsls_sem', ax=ax, capsize=5, color='coral')
            ax.set_title('[Combined W+F] WSLS by Genotype', fontweight='bold')
            ax.set_ylabel('Mean WSLS')
            ax.set_xlabel('Genotype')
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            # Plot 3: Engaged state occupancy
            ax = axes[1, 0]
            combined_df.plot(x='genotype', y='engaged_mean', kind='bar',
                           yerr='engaged_sem', ax=ax, capsize=5, color='seagreen')
            ax.set_title('[Combined W+F] Engaged State Occupancy', fontweight='bold')
            ax.set_ylabel('Mean Engaged Occupancy')
            ax.set_xlabel('Genotype')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            # Plot 4: Summary table
            ax = axes[1, 1]
            ax.axis('tight')
            ax.axis('off')

            table_data = []
            for _, row in combined_df.iterrows():
                table_data.append([
                    row['genotype'],
                    f"{row['n_animals']}\n({row['n_W']}W+{row['n_F']}F)",
                    f"{row['accuracy_mean']:.3f}±{row['accuracy_sem']:.3f}",
                    f"{row['wsls_mean']:.3f}±{row['wsls_sem']:.3f}" if not pd.isna(row['wsls_mean']) else "N/A",
                    f"{row['engaged_mean']:.3f}±{row['engaged_sem']:.3f}"
                ])

            table = ax.table(cellText=table_data,
                           colLabels=['Genotype', 'N (W+F)', 'Accuracy', 'WSLS', 'Engaged'],
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.15, 0.15, 0.23, 0.23, 0.23])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            # Style header
            for i in range(5):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')

            ax.set_title('[Combined W+F] Summary Statistics', fontweight='bold', pad=20)

        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.text(0.5, 0.5, 'Combined cohort data not available\nRun phase2_combined_cohort_analysis.py first',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')

        plt.tight_layout()
        fig.savefig(self.results_dir / 'fig4_cross_cohort_combined.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.results_dir / 'fig4_cross_cohort_combined.pdf', bbox_inches='tight')
        print("  ✓ Saved Figure 4")
        plt.close()

    def create_summary_report(self, models):
        """Create a text summary of Phase 2 analysis."""
        print("\nCreating summary report...")

        report = []
        report.append("="*80)
        report.append("PHASE 2: REVERSAL LEARNING ANALYSIS SUMMARY")
        report.append("="*80)
        report.append("")

        # Count animals
        w_count = len(models['W'])
        f_count = len(models['F'])
        report.append(f"Total animals analyzed: {w_count + f_count}")
        report.append(f"  W Cohort: {w_count} animals")
        report.append(f"  F Cohort: {f_count} animals")
        report.append("")

        # Genotype distribution
        report.append("Genotype Distribution:")
        for cohort in ['W', 'F']:
            genotypes = {}
            for animal_id, data in models[cohort].items():
                geno = data['genotype']
                genotypes[geno] = genotypes.get(geno, 0) + 1

            report.append(f"  {cohort} Cohort:")
            for geno, count in sorted(genotypes.items()):
                report.append(f"    {geno}: {count} animals")

        report.append("")
        report.append("Visualizations Created:")
        report.append("  1. fig1_reversal_state_characteristics - Behavioral states during reversal")
        report.append("  2. fig2_psychometric_curves - Performance trajectories")
        report.append("  3. fig3_reversal_adaptation - Phase 1 vs Phase 2 comparison")
        report.append("  4. fig4_cross_cohort_combined - Combined cohort analysis")
        report.append("")
        report.append("="*80)

        summary_file = self.results_dir / 'analysis_summary.txt'
        with open(summary_file, 'w') as f:
            f.write('\n'.join(report))

        print("  ✓ Saved summary report")
        print('\n'.join(report))

    def run_all(self):
        """Run all visualization generation."""
        print("\n" + "="*80)
        print("PHASE 2 REVERSAL LEARNING - COMPREHENSIVE VISUALIZATIONS")
        print("="*80)

        # Load models
        models = self.load_all_models()

        if not models['W'] and not models['F']:
            print("ERROR: No Phase 2 models found!")
            return

        # Create all figures
        self.create_figure1_state_characteristics(models)
        self.create_figure2_psychometric_curves(models)
        self.create_figure3_reversal_adaptation(models)
        self.create_figure4_cross_cohort_combined(models)
        self.create_summary_report(models)

        print("\n" + "="*80)
        print("VISUALIZATION GENERATION COMPLETE")
        print("="*80)
        print(f"\nAll figures saved to: {self.results_dir}")
        print("\nGenerated files:")
        for f in sorted(self.results_dir.glob('*')):
            print(f"  - {f.name}")


if __name__ == '__main__':
    viz = Phase2Visualizations()
    viz.run_all()
