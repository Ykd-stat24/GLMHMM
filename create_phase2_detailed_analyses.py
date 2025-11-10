"""
Phase 2 Detailed Analyses
=========================

Comprehensive analyses including:
1. Individual animal reversal learning curves
2. Deliberative vs Procedural strategy changes
3. High vs Low performer comparisons
4. Lapse/Engaged state dynamics Phase 1→2
5. Statistical tests for all comparisons
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

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


class Phase2DetailedAnalyses:
    """Comprehensive detailed analyses for Phase 2 reversal learning."""

    def __init__(self):
        self.results_dir = Path('results/phase2_reversal/detailed_analyses')
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.models_dir_p2 = Path('results/phase2_reversal/models')
        self.models_dir_p1 = Path('results/phase1_non_reversal')

    def load_phase2_models(self):
        """Load all Phase 2 models."""
        print("Loading Phase 2 models...")
        models = {'W': {}, 'F': {}}

        for pkl_file in self.models_dir_p2.glob('*_reversal.pkl'):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                animal_id = data['animal_id']
                cohort = data['cohort']
                models[cohort][animal_id] = data
            except Exception as e:
                print(f"  Warning: {pkl_file.name}: {e}")

        print(f"  Loaded: {len(models['W'])} W, {len(models['F'])} F")
        return models

    def load_phase1_models(self):
        """Load Phase 1 models for comparison."""
        print("Loading Phase 1 models...")
        models = {'W': {}, 'F': {}}

        for pkl_file in self.models_dir_p1.glob('*_model.pkl'):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                animal_id = data['animal_id']
                cohort = data['cohort']
                models[cohort][animal_id] = data
            except Exception as e:
                pass

        print(f"  Loaded: {len(models['W'])} W, {len(models['F'])} F")
        return models

    def create_individual_animal_curves(self, p2_models):
        """Individual animal reversal learning curves."""
        print("\nCreating individual animal reversal curves...")

        # Create separate figures for each cohort
        for cohort in ['W', 'F']:
            animals = list(p2_models[cohort].keys())
            if not animals:
                continue

            n_animals = len(animals)
            n_cols = 4
            n_rows = int(np.ceil(n_animals / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
            fig.suptitle(f'Individual Reversal Learning Curves - Cohort {cohort}',
                        fontsize=16, fontweight='bold')

            axes = axes.flatten() if n_animals > 1 else [axes]

            for idx, animal_id in enumerate(sorted(animals)):
                ax = axes[idx]
                model_data = p2_models[cohort][animal_id]

                y = model_data.get('y')
                states = model_data['model'].most_likely_states
                genotype = model_data['genotype']

                if y is None or len(y) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(f'{animal_id} ({genotype})')
                    continue

                # Create trial-level accuracy
                window = 30
                accuracy = pd.Series(y).rolling(window=window, center=True).mean()

                # Plot accuracy
                ax.plot(accuracy.index, accuracy.values, 'k-', linewidth=2, alpha=0.7,
                       label='Accuracy')

                # Color by state
                for state in np.unique(states):
                    state_mask = states == state
                    ax.scatter(np.where(state_mask)[0],
                             accuracy[state_mask],
                             alpha=0.3, s=10, label=f'State {state}')

                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                ax.set_title(f'{animal_id} ({genotype})\nn={len(y)} trials',
                           fontweight='bold')
                ax.set_xlabel('Trial')
                ax.set_ylabel(f'Accuracy ({window}-trial avg)')
                ax.set_ylim([0, 1])
                ax.legend(fontsize=8, loc='best')
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for idx in range(n_animals, len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            fig.savefig(self.results_dir / f'individual_curves_cohort{cohort}.png',
                       dpi=300, bbox_inches='tight')
            fig.savefig(self.results_dir / f'individual_curves_cohort{cohort}.pdf',
                       bbox_inches='tight')
            print(f"  ✓ Saved individual curves for cohort {cohort}")
            plt.close()

    def analyze_state_transitions_p1_to_p2(self, p1_models, p2_models):
        """Analyze how Engaged/Lapsed states change from Phase 1 to Phase 2."""
        print("\nAnalyzing state transitions Phase 1 → Phase 2...")

        comparison_data = []

        for cohort in ['W', 'F']:
            # Find animals with both Phase 1 and 2
            common = set(p1_models[cohort].keys()) & set(p2_models[cohort].keys())

            for animal_id in common:
                p1 = p1_models[cohort][animal_id]
                p2 = p2_models[cohort][animal_id]

                genotype = p1['genotype']

                # Extract state metrics
                p1_metrics = p1.get('state_metrics')
                p2_metrics = p2.get('state_metrics')

                if p1_metrics is None or p2_metrics is None:
                    continue

                # Handle DataFrame format
                if isinstance(p1_metrics, pd.DataFrame):
                    p1_engaged = p1_metrics[p1_metrics['accuracy'] > 0.6]['occupancy'].sum()
                    p1_lapsed = p1_metrics[p1_metrics['accuracy'] < 0.4]['occupancy'].sum()
                else:
                    continue

                if isinstance(p2_metrics, pd.DataFrame):
                    p2_engaged = p2_metrics[p2_metrics['accuracy'] > 0.6]['occupancy'].sum()
                    p2_lapsed = p2_metrics[p2_metrics['accuracy'] < 0.4]['occupancy'].sum()
                else:
                    continue

                comparison_data.append({
                    'animal_id': animal_id,
                    'cohort': cohort,
                    'genotype': genotype,
                    'p1_engaged': p1_engaged,
                    'p2_engaged': p2_engaged,
                    'engaged_change': p2_engaged - p1_engaged,
                    'p1_lapsed': p1_lapsed,
                    'p2_lapsed': p2_lapsed,
                    'lapsed_change': p2_lapsed - p1_lapsed
                })

        df = pd.DataFrame(comparison_data)

        if len(df) == 0:
            print("  No comparable data found")
            return None

        # Save data
        df.to_csv(self.results_dir / 'state_transitions_p1_to_p2.csv', index=False)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('State Transitions: Phase 1 → Phase 2 Reversal',
                    fontsize=16, fontweight='bold')

        # Plot 1: Engaged state changes
        ax = axes[0, 0]
        sns.boxplot(data=df, x='genotype', y='engaged_change', ax=ax)
        sns.swarmplot(data=df, x='genotype', y='engaged_change',
                     color='black', alpha=0.5, size=4, ax=ax)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_title('Change in Engaged State Occupancy', fontweight='bold')
        ax.set_ylabel('Δ Engaged (Phase 2 - Phase 1)')
        ax.set_xlabel('Genotype')

        # Add statistics
        for i, geno in enumerate(df['genotype'].unique()):
            geno_data = df[df['genotype'] == geno]['engaged_change']
            mean_change = geno_data.mean()
            # One-sample t-test vs 0
            if len(geno_data) > 2:
                t_stat, p_val = stats.ttest_1samp(geno_data, 0)
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                ax.text(i, ax.get_ylim()[1] * 0.9,
                       f'μ={mean_change:.3f}\n{sig}',
                       ha='center', fontsize=9, fontweight='bold')

        # Plot 2: Lapsed state changes
        ax = axes[0, 1]
        sns.boxplot(data=df, x='genotype', y='lapsed_change', ax=ax)
        sns.swarmplot(data=df, x='genotype', y='lapsed_change',
                     color='black', alpha=0.5, size=4, ax=ax)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_title('Change in Lapsed State Occupancy', fontweight='bold')
        ax.set_ylabel('Δ Lapsed (Phase 2 - Phase 1)')
        ax.set_xlabel('Genotype')

        # Plot 3: Phase 1 vs Phase 2 Engaged scatter
        ax = axes[1, 0]
        for geno in df['genotype'].unique():
            geno_df = df[df['genotype'] == geno]
            ax.scatter(geno_df['p1_engaged'], geno_df['p2_engaged'],
                      label=f'{geno} (n={len(geno_df)})', s=100, alpha=0.6)

        # Unity line
        max_val = max(df['p1_engaged'].max(), df['p2_engaged'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.5)
        ax.set_xlabel('Phase 1 Engaged Occupancy')
        ax.set_ylabel('Phase 2 Engaged Occupancy')
        ax.set_title('Engaged State: Phase 1 vs Phase 2', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Summary statistics table
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')

        summary = df.groupby('genotype').agg({
            'engaged_change': ['mean', 'sem', 'count'],
            'lapsed_change': ['mean', 'sem']
        }).round(3)

        table_data = []
        for geno in summary.index:
            table_data.append([
                geno,
                f"{int(summary.loc[geno, ('engaged_change', 'count')])}",
                f"{summary.loc[geno, ('engaged_change', 'mean')]:.3f}±{summary.loc[geno, ('engaged_change', 'sem')]:.3f}",
                f"{summary.loc[geno, ('lapsed_change', 'mean')]:.3f}±{summary.loc[geno, ('lapsed_change', 'sem')]:.3f}"
            ])

        table = ax.table(cellText=table_data,
                        colLabels=['Genotype', 'N', 'Δ Engaged', 'Δ Lapsed'],
                        cellLoc='center', loc='center',
                        colWidths=[0.2, 0.15, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Summary: State Transitions', fontweight='bold', pad=20)

        plt.tight_layout()
        fig.savefig(self.results_dir / 'state_transitions_p1_to_p2.png',
                   dpi=300, bbox_inches='tight')
        fig.savefig(self.results_dir / 'state_transitions_p1_to_p2.pdf',
                   bbox_inches='tight')
        print("  ✓ Saved state transition analysis")
        plt.close()

        return df

    def analyze_deliberative_vs_procedural(self, p2_models):
        """Analyze Deliberative HP vs Procedural HP strategies."""
        print("\nAnalyzing Deliberative vs Procedural strategies...")

        strategy_data = []

        for cohort in ['W', 'F']:
            for animal_id, model_data in p2_models[cohort].items():
                validated_labels = model_data.get('validated_labels', {})
                state_metrics = model_data.get('state_metrics')
                genotype = model_data['genotype']

                if state_metrics is None or not isinstance(state_metrics, pd.DataFrame):
                    continue

                # Classify states as Deliberative HP or Procedural HP
                for state_id, label in validated_labels.items():
                    state_id = int(state_id)

                    if 'Deliberative HP' in label:
                        strategy_type = 'Deliberative HP'
                    elif 'Procedural HP' in label:
                        strategy_type = 'Procedural HP'
                    else:
                        continue

                    # Get metrics for this state
                    state_row = state_metrics[state_metrics['state'] == state_id]
                    if len(state_row) == 0:
                        continue

                    row = state_row.iloc[0]

                    strategy_data.append({
                        'animal_id': animal_id,
                        'cohort': cohort,
                        'genotype': genotype,
                        'strategy': strategy_type,
                        'state': state_id,
                        'occupancy': row['occupancy'],
                        'accuracy': row['accuracy'],
                        'wsls': row['wsls_ratio'],
                        'perseveration': row.get('perseveration_ratio', 0)
                    })

        df = pd.DataFrame(strategy_data)

        if len(df) == 0:
            print("  No strategy data found")
            return None

        # Save data
        df.to_csv(self.results_dir / 'deliberative_vs_procedural.csv', index=False)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Deliberative HP vs Procedural HP Strategies in Reversal',
                    fontsize=16, fontweight='bold')

        # Plot 1: Strategy occupancy by genotype
        ax = axes[0, 0]
        occ_pivot = df.groupby(['genotype', 'strategy'])['occupancy'].mean().unstack()
        occ_pivot.plot(kind='bar', ax=ax, colormap='Set2')
        ax.set_title('Strategy Occupancy by Genotype', fontweight='bold')
        ax.set_ylabel('Mean Occupancy')
        ax.set_xlabel('Genotype')
        ax.legend(title='Strategy')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Plot 2: Accuracy comparison
        ax = axes[0, 1]
        sns.boxplot(data=df, x='strategy', y='accuracy', hue='genotype', ax=ax)
        ax.set_title('Accuracy by Strategy Type', fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Strategy Type')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        # Plot 3: WSLS comparison
        ax = axes[1, 0]
        sns.boxplot(data=df, x='strategy', y='wsls', hue='genotype', ax=ax)
        ax.set_title('WSLS by Strategy Type', fontweight='bold')
        ax.set_ylabel('WSLS Ratio')
        ax.set_xlabel('Strategy Type')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        # Plot 4: Perseveration comparison
        ax = axes[1, 1]
        sns.violinplot(data=df, x='strategy', y='perseveration', hue='genotype', ax=ax)
        ax.set_title('Perseveration by Strategy Type', fontweight='bold')
        ax.set_ylabel('Perseveration Ratio')
        ax.set_xlabel('Strategy Type')

        plt.tight_layout()
        fig.savefig(self.results_dir / 'deliberative_vs_procedural.png',
                   dpi=300, bbox_inches='tight')
        fig.savefig(self.results_dir / 'deliberative_vs_procedural.pdf',
                   bbox_inches='tight')
        print("  ✓ Saved Deliberative vs Procedural analysis")
        plt.close()

        return df

    def analyze_high_vs_low_performers(self, p2_models):
        """Compare strategies between high and low performing animals."""
        print("\nAnalyzing High vs Low performer strategies...")

        # First, calculate overall performance for each animal
        performance = []

        for cohort in ['W', 'F']:
            for animal_id, model_data in p2_models[cohort].items():
                y = model_data.get('y')
                genotype = model_data['genotype']
                state_metrics = model_data.get('state_metrics')

                if y is None or len(y) == 0:
                    continue

                overall_acc = np.mean(y)

                # Get state information
                if isinstance(state_metrics, pd.DataFrame):
                    engaged_occ = state_metrics[state_metrics['accuracy'] > 0.6]['occupancy'].sum()
                    lapsed_occ = state_metrics[state_metrics['accuracy'] < 0.4]['occupancy'].sum()
                    avg_wsls = np.average(state_metrics['wsls_ratio'].values,
                                         weights=state_metrics['occupancy'].values)
                else:
                    engaged_occ = 0
                    lapsed_occ = 0
                    avg_wsls = np.nan

                performance.append({
                    'animal_id': animal_id,
                    'cohort': cohort,
                    'genotype': genotype,
                    'accuracy': overall_acc,
                    'engaged_occ': engaged_occ,
                    'lapsed_occ': lapsed_occ,
                    'wsls': avg_wsls,
                    'n_trials': len(y)
                })

        df = pd.DataFrame(performance)

        # Classify as high vs low performer (median split)
        median_acc = df['accuracy'].median()
        df['performer_group'] = df['accuracy'].apply(
            lambda x: 'High Performer' if x >= median_acc else 'Low Performer'
        )

        # Save data
        df.to_csv(self.results_dir / 'high_vs_low_performers.csv', index=False)

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('High vs Low Performers: Strategy Comparison',
                    fontsize=16, fontweight='bold')

        # Plot 1: Accuracy distribution
        ax = axes[0, 0]
        sns.boxplot(data=df, x='performer_group', y='accuracy', hue='genotype', ax=ax)
        ax.axhline(y=median_acc, color='red', linestyle='--', linewidth=2,
                  label=f'Median={median_acc:.3f}')
        ax.set_title('Accuracy Distribution', fontweight='bold')
        ax.set_ylabel('Overall Accuracy')
        ax.set_xlabel('')
        ax.legend()

        # Plot 2: Engaged state occupancy
        ax = axes[0, 1]
        sns.boxplot(data=df, x='performer_group', y='engaged_occ', ax=ax)
        sns.swarmplot(data=df, x='performer_group', y='engaged_occ',
                     hue='genotype', alpha=0.6, size=6, ax=ax)
        ax.set_title('Engaged State Occupancy', fontweight='bold')
        ax.set_ylabel('Engaged Occupancy')
        ax.set_xlabel('')

        # Statistics
        high = df[df['performer_group'] == 'High Performer']['engaged_occ']
        low = df[df['performer_group'] == 'Low Performer']['engaged_occ']
        if len(high) > 0 and len(low) > 0:
            u_stat, p_val = mannwhitneyu(high, low)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            ax.text(0.5, 0.95, f'Mann-Whitney: p={p_val:.4f} {sig}',
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 3: Lapsed state occupancy
        ax = axes[0, 2]
        sns.boxplot(data=df, x='performer_group', y='lapsed_occ', ax=ax)
        sns.swarmplot(data=df, x='performer_group', y='lapsed_occ',
                     hue='genotype', alpha=0.6, size=6, ax=ax)
        ax.set_title('Lapsed State Occupancy', fontweight='bold')
        ax.set_ylabel('Lapsed Occupancy')
        ax.set_xlabel('')

        # Statistics
        high = df[df['performer_group'] == 'High Performer']['lapsed_occ']
        low = df[df['performer_group'] == 'Low Performer']['lapsed_occ']
        if len(high) > 0 and len(low) > 0:
            u_stat, p_val = mannwhitneyu(high, low)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            ax.text(0.5, 0.95, f'Mann-Whitney: p={p_val:.4f} {sig}',
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 4: WSLS strategy
        ax = axes[1, 0]
        sns.boxplot(data=df, x='performer_group', y='wsls', ax=ax)
        sns.swarmplot(data=df, x='performer_group', y='wsls',
                     hue='genotype', alpha=0.6, size=6, ax=ax)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('WSLS Strategy Use', fontweight='bold')
        ax.set_ylabel('WSLS Ratio')
        ax.set_xlabel('')

        # Plot 5: Genotype composition
        ax = axes[1, 1]
        comp = df.groupby(['performer_group', 'genotype']).size().unstack(fill_value=0)
        comp.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
        ax.set_title('Genotype Composition', fontweight='bold')
        ax.set_ylabel('Number of Animals')
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title='Genotype', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot 6: Summary table
        ax = axes[1, 2]
        ax.axis('tight')
        ax.axis('off')

        summary = df.groupby('performer_group').agg({
            'accuracy': ['mean', 'sem', 'count'],
            'engaged_occ': ['mean', 'sem'],
            'lapsed_occ': ['mean', 'sem'],
            'wsls': ['mean', 'sem']
        }).round(3)

        table_data = []
        for group in summary.index:
            table_data.append([
                group,
                f"{int(summary.loc[group, ('accuracy', 'count')])}",
                f"{summary.loc[group, ('accuracy', 'mean')]:.3f}±{summary.loc[group, ('accuracy', 'sem')]:.3f}",
                f"{summary.loc[group, ('engaged_occ', 'mean')]:.3f}±{summary.loc[group, ('engaged_occ', 'sem')]:.3f}",
                f"{summary.loc[group, ('lapsed_occ', 'mean')]:.3f}±{summary.loc[group, ('lapsed_occ', 'sem')]:.3f}"
            ])

        table = ax.table(cellText=table_data,
                        colLabels=['Group', 'N', 'Accuracy', 'Engaged', 'Lapsed'],
                        cellLoc='center', loc='center',
                        colWidths=[0.25, 0.1, 0.25, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Summary Statistics', fontweight='bold', pad=20)

        plt.tight_layout()
        fig.savefig(self.results_dir / 'high_vs_low_performers.png',
                   dpi=300, bbox_inches='tight')
        fig.savefig(self.results_dir / 'high_vs_low_performers.pdf',
                   bbox_inches='tight')
        print("  ✓ Saved High vs Low performer analysis")
        plt.close()

        return df

    def generate_comprehensive_summary(self, state_trans_df, strategy_df, performer_df):
        """Generate comprehensive text summary."""
        print("\nGenerating comprehensive summary...")

        report = []
        report.append("="*80)
        report.append("PHASE 2 REVERSAL LEARNING - COMPREHENSIVE ANALYSIS SUMMARY")
        report.append("="*80)
        report.append("")

        # STATE TRANSITIONS
        if state_trans_df is not None and len(state_trans_df) > 0:
            report.append("="*80)
            report.append("1. STATE TRANSITIONS: PHASE 1 → PHASE 2")
            report.append("="*80)
            report.append("")

            for geno in state_trans_df['genotype'].unique():
                geno_data = state_trans_df[state_trans_df['genotype'] == geno]
                n = len(geno_data)

                engaged_change = geno_data['engaged_change'].mean()
                engaged_sem = geno_data['engaged_change'].sem()
                lapsed_change = geno_data['lapsed_change'].mean()
                lapsed_sem = geno_data['lapsed_change'].sem()

                # Test if change is significant
                if len(geno_data) > 2:
                    t_eng, p_eng = stats.ttest_1samp(geno_data['engaged_change'], 0)
                    t_lap, p_lap = stats.ttest_1samp(geno_data['lapsed_change'], 0)

                    report.append(f"{geno} Genotype (n={n}):")
                    report.append(f"  Engaged State Change: {engaged_change:+.3f} ± {engaged_sem:.3f}")
                    report.append(f"    Significance: p={p_eng:.4f} {'***' if p_eng<0.001 else '**' if p_eng<0.01 else '*' if p_eng<0.05 else 'ns'}")
                    report.append(f"  Lapsed State Change: {lapsed_change:+.3f} ± {lapsed_sem:.3f}")
                    report.append(f"    Significance: p={p_lap:.4f} {'***' if p_lap<0.001 else '**' if p_lap<0.01 else '*' if p_lap<0.05 else 'ns'}")
                    report.append("")

        # STRATEGY ANALYSIS
        if strategy_df is not None and len(strategy_df) > 0:
            report.append("="*80)
            report.append("2. DELIBERATIVE vs PROCEDURAL STRATEGIES")
            report.append("="*80)
            report.append("")

            for geno in strategy_df['genotype'].unique():
                geno_data = strategy_df[strategy_df['genotype'] == geno]
                report.append(f"{geno} Genotype:")

                for strategy in ['Deliberative HP', 'Procedural HP']:
                    strat_data = geno_data[geno_data['strategy'] == strategy]
                    if len(strat_data) == 0:
                        continue

                    occ = strat_data['occupancy'].mean()
                    acc = strat_data['accuracy'].mean()
                    wsls = strat_data['wsls'].mean()

                    report.append(f"  {strategy}:")
                    report.append(f"    Occupancy: {occ:.3f}")
                    report.append(f"    Accuracy: {acc:.3f}")
                    report.append(f"    WSLS: {wsls:.3f}")

                report.append("")

        # PERFORMER COMPARISON
        if performer_df is not None and len(performer_df) > 0:
            report.append("="*80)
            report.append("3. HIGH vs LOW PERFORMER COMPARISON")
            report.append("="*80)
            report.append("")

            median_acc = performer_df['accuracy'].median()
            report.append(f"Performance Split: Median Accuracy = {median_acc:.3f}")
            report.append("")

            for group in ['High Performer', 'Low Performer']:
                group_data = performer_df[performer_df['performer_group'] == group]
                n = len(group_data)

                acc = group_data['accuracy'].mean()
                acc_sem = group_data['accuracy'].sem()
                engaged = group_data['engaged_occ'].mean()
                engaged_sem = group_data['engaged_occ'].sem()
                lapsed = group_data['lapsed_occ'].mean()
                lapsed_sem = group_data['lapsed_occ'].sem()
                wsls = group_data['wsls'].mean()
                wsls_sem = group_data['wsls'].sem()

                report.append(f"{group} (n={n}):")
                report.append(f"  Accuracy: {acc:.3f} ± {acc_sem:.3f}")
                report.append(f"  Engaged Occupancy: {engaged:.3f} ± {engaged_sem:.3f}")
                report.append(f"  Lapsed Occupancy: {lapsed:.3f} ± {lapsed_sem:.3f}")
                report.append(f"  WSLS: {wsls:.3f} ± {wsls_sem:.3f}")
                report.append("")

                # Genotype breakdown
                geno_counts = group_data['genotype'].value_counts()
                report.append("  Genotype Distribution:")
                for geno, count in geno_counts.items():
                    pct = 100 * count / n
                    report.append(f"    {geno}: {count} ({pct:.1f}%)")
                report.append("")

            # Statistical comparisons
            high = performer_df[performer_df['performer_group'] == 'High Performer']
            low = performer_df[performer_df['performer_group'] == 'Low Performer']

            report.append("Statistical Comparisons (Mann-Whitney U test):")

            if len(high) > 0 and len(low) > 0:
                u_eng, p_eng = mannwhitneyu(high['engaged_occ'], low['engaged_occ'])
                u_lap, p_lap = mannwhitneyu(high['lapsed_occ'], low['lapsed_occ'])

                report.append(f"  Engaged Occupancy: U={u_eng:.1f}, p={p_eng:.4f} {'***' if p_eng<0.001 else '**' if p_eng<0.01 else '*' if p_eng<0.05 else 'ns'}")
                report.append(f"  Lapsed Occupancy: U={u_lap:.1f}, p={p_lap:.4f} {'***' if p_lap<0.001 else '**' if p_lap<0.01 else '*' if p_lap<0.05 else 'ns'}")

        report.append("")
        report.append("="*80)
        report.append("KEY FINDINGS")
        report.append("="*80)
        report.append("")

        # Extract key findings
        findings = []

        if state_trans_df is not None and len(state_trans_df) > 0:
            # Which genotype shows biggest engaged state drop?
            worst_geno = state_trans_df.groupby('genotype')['engaged_change'].mean().idxmin()
            worst_change = state_trans_df.groupby('genotype')['engaged_change'].mean().min()
            findings.append(f"• Reversal learning causes biggest engaged state loss in {worst_geno} genotype ({worst_change:+.3f})")

        if performer_df is not None and len(performer_df) > 0:
            high = performer_df[performer_df['performer_group'] == 'High Performer']
            low = performer_df[performer_df['performer_group'] == 'Low Performer']

            eng_diff = high['engaged_occ'].mean() - low['engaged_occ'].mean()
            findings.append(f"• High performers spend {eng_diff:.3f} more time in engaged states")

            lap_diff = low['lapsed_occ'].mean() - high['lapsed_occ'].mean()
            findings.append(f"• Low performers spend {lap_diff:.3f} more time in lapsed states")

        for finding in findings:
            report.append(finding)

        report.append("")
        report.append("="*80)

        # Save report
        summary_file = self.results_dir / 'comprehensive_summary.txt'
        with open(summary_file, 'w') as f:
            f.write('\n'.join(report))

        print("  ✓ Saved comprehensive summary")
        print('\n'.join(report))

    def run_all(self):
        """Run all detailed analyses."""
        print("\n" + "="*80)
        print("PHASE 2 DETAILED ANALYSES - COMPREHENSIVE")
        print("="*80)

        # Load models
        p2_models = self.load_phase2_models()
        p1_models = self.load_phase1_models()

        # Run analyses
        self.create_individual_animal_curves(p2_models)
        state_trans_df = self.analyze_state_transitions_p1_to_p2(p1_models, p2_models)
        strategy_df = self.analyze_deliberative_vs_procedural(p2_models)
        performer_df = self.analyze_high_vs_low_performers(p2_models)

        # Generate summary
        self.generate_comprehensive_summary(state_trans_df, strategy_df, performer_df)

        print("\n" + "="*80)
        print("DETAILED ANALYSES COMPLETE")
        print("="*80)
        print(f"\nAll results saved to: {self.results_dir}")
        print("\nGenerated files:")
        for f in sorted(self.results_dir.glob('*')):
            print(f"  - {f.name}")


if __name__ == '__main__':
    analyzer = Phase2DetailedAnalyses()
    analyzer.run_all()
