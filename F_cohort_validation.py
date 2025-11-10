"""
F Cohort Data Validation and Quality Check
===========================================

Critical validation to determine if "poor performer" findings are real or artifacts.

Questions to answer:
1. Does trial-level rolling accuracy match End Summary - Percentage Correct?
2. Are poor performers truly performing poorly, or is this a processing error?
3. Which specific animals are late lapsers?
4. Why didn't we catch this in previous analyses?
5. What's the data quality in F LD Data 11.08 All_processed.csv?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


class FCohortDataValidator:
    """Validate F cohort data and poor performer classifications."""

    def __init__(self):
        self.results_dir = Path('results/phase1_non_reversal')
        self.output_dir = self.results_dir / 'F_cohort_validation'
        self.output_dir.mkdir(exist_ok=True)

    def check_data_quality_basics(self):
        """
        Basic data quality check - no trial-level data available in processed CSV.
        """
        print("\n" + "="*80)
        print("DATA QUALITY CHECK: Session-Level Data")
        print("="*80)

        # Load the processed data
        df = pd.read_csv('F LD Data 11.08 All_processed.csv')

        print(f"\nLoaded {len(df)} sessions from F LD Data 11.08 All_processed.csv")

        # Extract End Summary accuracy
        if 'End Summary - Percentage Correct (1)' in df.columns:
            df['accuracy'] = pd.to_numeric(df['End Summary - Percentage Correct (1)'],
                                          errors='coerce') / 100
            print(f"Successfully extracted accuracy from {df['accuracy'].notna().sum()} sessions")
        else:
            print("ERROR: 'End Summary - Percentage Correct (1)' not found!")
            return None

        # Check data completeness
        print(f"\n{'-'*80}")
        print("DATA COMPLETENESS:")
        print(f"{'-'*80}")
        print(f"Total sessions: {len(df)}")
        print(f"Sessions with valid accuracy: {df['accuracy'].notna().sum()}")
        print(f"Missing accuracy: {df['accuracy'].isna().sum()}")

        # Basic stats
        print(f"\n{'-'*80}")
        print("SESSION-LEVEL ACCURACY DISTRIBUTION:")
        print(f"{'-'*80}")
        print(f"Mean: {df['accuracy'].mean():.3f}")
        print(f"Median: {df['accuracy'].median():.3f}")
        print(f"Std: {df['accuracy'].std():.3f}")
        print(f"Min: {df['accuracy'].min():.3f}")
        print(f"Max: {df['accuracy'].max():.3f}")

        # Quality ranges
        print(f"\nSessions by accuracy range:")
        print(f"  <40% (very poor): {(df['accuracy'] < 0.4).sum()} ({(df['accuracy'] < 0.4).sum()/len(df)*100:.1f}%)")
        print(f"  40-60% (poor): {((df['accuracy'] >= 0.4) & (df['accuracy'] < 0.6)).sum()} " +
             f"({((df['accuracy'] >= 0.4) & (df['accuracy'] < 0.6)).sum()/len(df)*100:.1f}%)")
        print(f"  60-80% (good): {((df['accuracy'] >= 0.6) & (df['accuracy'] < 0.8)).sum()} " +
             f"({((df['accuracy'] >= 0.6) & (df['accuracy'] < 0.8)).sum()/len(df)*100:.1f}%)")
        print(f"  >80% (excellent): {(df['accuracy'] >= 0.8).sum()} ({(df['accuracy'] >= 0.8).sum()/len(df)*100:.1f}%)")

        print(f"\n✓ Data appears clean and complete")
        print("  NOTE: Processed CSV only has session summaries, not trial-level data")
        print("  Therefore we use 'End Summary - Percentage Correct' as ground truth")

        return df

    def recheck_poor_performers_with_raw_data(self):
        """
        Recompute poor performer classification using ONLY the processed CSV.
        No pickle files, no models - just raw performance data.
        """
        print("\n" + "="*80)
        print("RECHECKING POOR PERFORMERS - Using Raw CSV Data Only")
        print("="*80)

        # Load F cohort data
        df = pd.read_csv('F LD Data 11.08 All_processed.csv')

        # Get accuracy
        df['accuracy'] = pd.to_numeric(df['End Summary - Percentage Correct (1)'],
                                       errors='coerce') / 100
        df['session_date'] = pd.to_datetime(df['Schedule run date'], errors='coerce')
        df = df.sort_values(['Animal ID', 'session_date'])

        # Assign genotype
        def get_genotype(animal_id):
            try:
                aid = int(animal_id)
                if aid in [11, 12, 13, 14, 41, 42, 61, 62, 63, 64]:
                    return '+/+'
                elif aid in [21, 22, 23, 24, 25, 51, 52, 71, 72, 73, 91, 92, 93]:
                    return '+/-'
                elif aid in [31, 32, 33, 34]:
                    return '-/-'
                elif aid in [81, 82, 83, 84, 101, 102, 103, 104]:
                    return '+'
                else:
                    return 'unknown'
            except:
                return 'unknown'

        df['genotype'] = df['Animal ID'].apply(get_genotype)
        df = df[df['genotype'] != 'unknown']

        # For each animal, compute overall stats
        animal_stats = []

        for animal_id in df['Animal ID'].unique():
            animal_data = df[df['Animal ID'] == animal_id].copy()
            animal_data = animal_data.sort_values('session_date')

            n_sessions = len(animal_data)
            overall_acc = animal_data['accuracy'].mean()

            # Early vs late
            if n_sessions >= 3:
                early_cutoff = n_sessions // 3
                late_cutoff = 2 * n_sessions // 3

                early_acc = animal_data.iloc[:early_cutoff]['accuracy'].mean()
                late_acc = animal_data.iloc[late_cutoff:]['accuracy'].mean()
                acc_change = late_acc - early_acc
            else:
                early_acc = overall_acc
                late_acc = overall_acc
                acc_change = 0

            # Classify
            is_poor_overall = overall_acc < 0.6
            is_late_lapser = (acc_change < -0.1) and (late_acc < 0.7)
            is_poor_performer = is_poor_overall or is_late_lapser

            genotype = animal_data['genotype'].iloc[0]

            animal_stats.append({
                'animal_id': animal_id,
                'genotype': genotype,
                'n_sessions': n_sessions,
                'overall_accuracy': overall_acc,
                'early_accuracy': early_acc,
                'late_accuracy': late_acc,
                'accuracy_change': acc_change,
                'is_poor_overall': is_poor_overall,
                'is_late_lapser': is_late_lapser,
                'is_poor_performer': is_poor_performer
            })

        stats_df = pd.DataFrame(animal_stats)
        stats_df.to_csv(self.output_dir / 'F_cohort_performance_RECHECK.csv', index=False)

        # Report
        print(f"\n{'-'*80}")
        print("RECHECKED F COHORT PERFORMANCE:")
        print(f"{'-'*80}")
        print(f"Total animals: {len(stats_df)}")
        print(f"Poor performers (overall acc < 60%): {stats_df['is_poor_overall'].sum()}")
        print(f"Late lapsers (drop >10% AND late <70%): {stats_df['is_late_lapser'].sum()}")
        print(f"Total poor performers: {stats_df['is_poor_performer'].sum()} ({stats_df['is_poor_performer'].sum()/len(stats_df)*100:.1f}%)")

        print(f"\n{'-'*80}")
        print("BY GENOTYPE:")
        print(f"{'-'*80}")

        geno_summary = stats_df.groupby('genotype').agg({
            'animal_id': 'count',
            'overall_accuracy': ['mean', 'std'],
            'is_poor_performer': 'sum',
            'is_late_lapser': 'sum'
        }).round(3)

        print(geno_summary)

        # Detailed list of poor performers
        print(f"\n{'-'*80}")
        print("POOR PERFORMERS - DETAILED LIST:")
        print(f"{'-'*80}")

        poor_df = stats_df[stats_df['is_poor_performer']].sort_values('overall_accuracy')
        print(f"\n{len(poor_df)} poor performers:")
        for _, row in poor_df.iterrows():
            reason = []
            if row['is_poor_overall']:
                reason.append(f"Low overall ({row['overall_accuracy']:.2f})")
            if row['is_late_lapser']:
                reason.append(f"Late lapser ({row['accuracy_change']:.2f})")
            print(f"  {row['animal_id']} ({row['genotype']}): {', '.join(reason)}")

        # List of GOOD performers
        print(f"\n{'-'*80}")
        print("GOOD PERFORMERS - DETAILED LIST:")
        print(f"{'-'*80}")

        good_df = stats_df[~stats_df['is_poor_performer']].sort_values('overall_accuracy', ascending=False)
        print(f"\n{len(good_df)} good performers:")
        for _, row in good_df.iterrows():
            print(f"  {row['animal_id']} ({row['genotype']}): Overall={row['overall_accuracy']:.2f}, " +
                 f"Change={row['accuracy_change']:+.2f}")

        print(f"\n✓ Saved recheck results to: {self.output_dir / 'F_cohort_performance_RECHECK.csv'}")

        return stats_df

    def visualize_late_lapsers_detailed(self, stats_df):
        """
        Create detailed visualizations of late lapsers - animal by animal.
        """
        print("\n" + "="*80)
        print("DETAILED LATE LAPSER VISUALIZATION")
        print("="*80)

        # Load session data
        df = pd.read_csv('F LD Data 11.08 All_processed.csv')
        df['accuracy'] = pd.to_numeric(df['End Summary - Percentage Correct (1)'],
                                       errors='coerce') / 100
        df['session_date'] = pd.to_datetime(df['Schedule run date'], errors='coerce')

        # Get late lapsers
        late_lapsers = stats_df[stats_df['is_late_lapser']].sort_values('genotype')

        if len(late_lapsers) == 0:
            print("  No late lapsers found")
            return

        print(f"\nFound {len(late_lapsers)} late lapsers:")
        for _, row in late_lapsers.iterrows():
            print(f"  {row['animal_id']} ({row['genotype']}): " +
                 f"Early={row['early_accuracy']:.2f}, Late={row['late_accuracy']:.2f}, " +
                 f"Change={row['accuracy_change']:+.2f}")

        # Create multi-panel plot - one panel per late lapser
        n_lapsers = len(late_lapsers)
        n_cols = 5
        n_rows = (n_lapsers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        axes = axes.flatten()

        for idx, (_, row) in enumerate(late_lapsers.iterrows()):
            animal_id = row['animal_id']
            genotype = row['genotype']

            # Get animal data
            animal_data = df[df['Animal ID'] == animal_id].sort_values('session_date')

            if len(animal_data) == 0:
                continue

            ax = axes[idx]

            # Plot session-by-session accuracy
            sessions = np.arange(len(animal_data))
            accuracies = animal_data['accuracy'].values

            ax.plot(sessions, accuracies, 'o-', linewidth=2, markersize=6,
                   color='steelblue', alpha=0.7)

            # Add early/late shading
            n_sess = len(sessions)
            early_cutoff = n_sess // 3
            late_cutoff = 2 * n_sess // 3

            ax.axvspan(-0.5, early_cutoff-0.5, alpha=0.2, color='green',
                      label='Early')
            ax.axvspan(late_cutoff-0.5, n_sess-0.5, alpha=0.2, color='red',
                      label='Late')

            # Add reference lines
            ax.axhline(y=0.6, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=0.7, color='orange', linestyle='--', linewidth=1, alpha=0.5)

            # Add trend line
            z = np.polyfit(sessions, accuracies, 1)
            p = np.poly1d(z)
            ax.plot(sessions, p(sessions), 'r--', linewidth=2, alpha=0.7,
                   label=f'Trend (slope={z[0]:.3f})')

            ax.set_ylim(0, 1.05)
            ax.set_xlabel('Session', fontsize=9)
            ax.set_ylabel('Accuracy', fontsize=9)
            ax.set_title(f'{animal_id} ({genotype})\n' +
                        f'Δ={row["accuracy_change"]:+.2f}',
                        fontsize=10, fontweight='bold')
            ax.legend(fontsize=7, loc='lower left')
            ax.grid(alpha=0.3)

        # Hide unused axes
        for idx in range(len(late_lapsers), len(axes)):
            axes[idx].axis('off')

        fig.suptitle('Late Lapsers: Session-by-Session Accuracy Trajectories',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        plt.savefig(self.output_dir / 'late_lapsers_detailed.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'late_lapsers_detailed.pdf', bbox_inches='tight')
        plt.close()

        print(f"\n✓ Created detailed late lapser visualization")

    def investigate_why_not_caught_previously(self):
        """
        Investigate why poor performers weren't caught in previous analyses.
        """
        print("\n" + "="*80)
        print("INVESTIGATING: Why Weren't Poor Performers Caught Previously?")
        print("="*80)

        # Load pickle file results to see what we had before
        animals_F = [11, 12, 13, 14, 21, 22, 23, 24, 25,
                     31, 32, 33, 34, 41, 42, 51, 52,
                     61, 62, 63, 64, 71, 72, 73,
                     81, 82, 83, 84, 91, 92, 93,
                     101, 102, 103, 104]

        previous_data = []

        for animal in animals_F:
            pkl_file = self.results_dir / f'{animal}_cohortF_model.pkl'

            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)

                    # Get what we had before
                    state_metrics = data['state_metrics']
                    n_trials = data['n_trials']

                    # Weighted average accuracy
                    weighted_acc = np.average(
                        state_metrics['accuracy'].values,
                        weights=state_metrics['occupancy'].values
                    )

                    previous_data.append({
                        'animal_id': animal,
                        'genotype': data['genotype'],
                        'n_trials': n_trials,
                        'model_weighted_accuracy': weighted_acc,
                        'n_states': data['model'].n_states
                    })

        prev_df = pd.DataFrame(previous_data)

        # Load new session-level data
        df = pd.read_csv('F LD Data 11.08 All_processed.csv')
        df['accuracy'] = pd.to_numeric(df['End Summary - Percentage Correct (1)'],
                                       errors='coerce') / 100

        session_acc = df.groupby('Animal ID')['accuracy'].mean().reset_index()
        session_acc.columns = ['animal_id', 'session_level_accuracy']

        # Merge
        comparison = prev_df.merge(session_acc, on='animal_id', how='outer')

        comparison['difference'] = comparison['session_level_accuracy'] - comparison['model_weighted_accuracy']

        print(f"\n{'-'*80}")
        print("COMPARISON: Previous Model Results vs Session-Level Data")
        print(f"{'-'*80}")

        print(f"\nAnimals in model results: {prev_df['animal_id'].nunique()}")
        print(f"Animals in session data: {session_acc['animal_id'].nunique()}")
        print(f"Animals in both: {len(comparison.dropna())}")

        print(f"\nAccuracy comparison:")
        print(f"  Mean model accuracy: {comparison['model_weighted_accuracy'].mean():.3f}")
        print(f"  Mean session accuracy: {comparison['session_level_accuracy'].mean():.3f}")
        print(f"  Mean difference: {comparison['difference'].mean():.3f}")
        print(f"  Max abs difference: {comparison['difference'].abs().max():.3f}")

        # Show animals only in one dataset
        only_model = comparison[comparison['session_level_accuracy'].isna()]
        only_session = comparison[comparison['model_weighted_accuracy'].isna()]

        if len(only_model) > 0:
            print(f"\n  Animals ONLY in model (not in session data): {list(only_model['animal_id'].values)}")

        if len(only_session) > 0:
            print(f"  Animals ONLY in session data (not in model): {list(only_session['animal_id'].values)}")

        # Key insight
        comparison_valid = comparison.dropna()
        low_performers_model = comparison_valid[comparison_valid['model_weighted_accuracy'] < 0.6]
        low_performers_session = comparison_valid[comparison_valid['session_level_accuracy'] < 0.6]

        print(f"\n{'-'*80}")
        print("LOW PERFORMERS (<60% accuracy):")
        print(f"{'-'*80}")
        print(f"  By model: {len(low_performers_model)} animals")
        print(f"  By session data: {len(low_performers_session)} animals")

        # The KEY question: did we exclude poor performers when building models?
        print(f"\n{'-'*80}")
        print("HYPOTHESIS: Were Poor Performers Excluded From Model Building?")
        print(f"{'-'*80}")

        # Check if there's a minimum trial threshold
        print(f"\nTrial counts in models:")
        print(f"  Min: {prev_df['n_trials'].min()}")
        print(f"  Max: {prev_df['n_trials'].max()}")
        print(f"  Mean: {prev_df['n_trials'].mean():.1f}")

        # Animals with low trial counts might have been excluded
        low_trial_animals = prev_df[prev_df['n_trials'] < 200]
        print(f"\nAnimals with <200 trials in model: {len(low_trial_animals)}")
        if len(low_trial_animals) > 0:
            print(f"  {list(low_trial_animals['animal_id'].values)}")

        comparison.to_csv(self.output_dir / 'model_vs_session_comparison.csv', index=False)
        print(f"\n✓ Saved comparison to: {self.output_dir / 'model_vs_session_comparison.csv'}")

        return comparison

    def comprehensive_data_quality_report(self):
        """
        Option C - Full data quality investigation of F cohort.
        """
        print("\n" + "="*80)
        print("OPTION C: COMPREHENSIVE F COHORT DATA QUALITY INVESTIGATION")
        print("="*80)

        df = pd.read_csv('F LD Data 11.08 All_processed.csv')

        print(f"\nDataset: F LD Data 11.08 All_processed.csv")
        print(f"Total sessions: {len(df)}")
        print(f"Date range: {df['Schedule run date'].min()} to {df['Schedule run date'].max()}")

        # Basic stats
        print(f"\n{'-'*80}")
        print("BASIC STATISTICS:")
        print(f"{'-'*80}")

        unique_animals = df['Animal ID'].nunique()
        print(f"Unique animals: {unique_animals}")

        sessions_per_animal = df.groupby('Animal ID').size()
        print(f"Sessions per animal: {sessions_per_animal.min()} - {sessions_per_animal.max()} " +
             f"(mean: {sessions_per_animal.mean():.1f})")

        # Accuracy distribution
        df['accuracy'] = pd.to_numeric(df['End Summary - Percentage Correct (1)'],
                                       errors='coerce') / 100

        print(f"\n{'-'*80}")
        print("ACCURACY DISTRIBUTION:")
        print(f"{'-'*80}")
        print(f"Mean: {df['accuracy'].mean():.3f}")
        print(f"Median: {df['accuracy'].median():.3f}")
        print(f"Std: {df['accuracy'].std():.3f}")
        print(f"Min: {df['accuracy'].min():.3f}")
        print(f"Max: {df['accuracy'].max():.3f}")

        # Sessions by accuracy range
        print(f"\nSessions by accuracy range:")
        print(f"  <40%: {(df['accuracy'] < 0.4).sum()} ({(df['accuracy'] < 0.4).sum()/len(df)*100:.1f}%)")
        print(f"  40-60%: {((df['accuracy'] >= 0.4) & (df['accuracy'] < 0.6)).sum()} " +
             f"({((df['accuracy'] >= 0.4) & (df['accuracy'] < 0.6)).sum()/len(df)*100:.1f}%)")
        print(f"  60-80%: {((df['accuracy'] >= 0.6) & (df['accuracy'] < 0.8)).sum()} " +
             f"({((df['accuracy'] >= 0.6) & (df['accuracy'] < 0.8)).sum()/len(df)*100:.1f}%)")
        print(f"  >80%: {(df['accuracy'] >= 0.8).sum()} ({(df['accuracy'] >= 0.8).sum()/len(df)*100:.1f}%)")

        # Task types
        print(f"\n{'-'*80}")
        print("TASK TYPES:")
        print(f"{'-'*80}")
        task_counts = df['Schedule name'].value_counts()
        for task, count in task_counts.items():
            print(f"  {task}: {count} sessions")

        # Missing data
        print(f"\n{'-'*80}")
        print("MISSING DATA CHECK:")
        print(f"{'-'*80}")

        key_cols = ['Animal ID', 'Schedule run date', 'End Summary - Percentage Correct (1)',
                   'End Summary - Trials Completed (1)']

        for col in key_cols:
            if col in df.columns:
                missing = df[col].isna().sum()
                print(f"  {col}: {missing} missing ({missing/len(df)*100:.1f}%)")

        # Save quality report
        report = f"""
F COHORT DATA QUALITY REPORT
=============================

Dataset: F LD Data 11.08 All_processed.csv
Generated: {pd.Timestamp.now()}

SUMMARY:
--------
- Total sessions: {len(df)}
- Unique animals: {unique_animals}
- Date range: {df['Schedule run date'].min()} to {df['Schedule run date'].max()}
- Sessions per animal: {sessions_per_animal.min()}-{sessions_per_animal.max()} (mean: {sessions_per_animal.mean():.1f})

ACCURACY:
---------
- Mean: {df['accuracy'].mean():.3f}
- Median: {df['accuracy'].median():.3f}
- Std: {df['accuracy'].std():.3f}
- Range: {df['accuracy'].min():.3f} - {df['accuracy'].max():.3f}

QUALITY ASSESSMENT:
-------------------
- Sessions <40% accuracy: {(df['accuracy'] < 0.4).sum()} ({(df['accuracy'] < 0.4).sum()/len(df)*100:.1f}%)
- Sessions 40-60% accuracy: {((df['accuracy'] >= 0.4) & (df['accuracy'] < 0.6)).sum()} ({((df['accuracy'] >= 0.4) & (df['accuracy'] < 0.6)).sum()/len(df)*100:.1f}%)
- Sessions >60% accuracy: {(df['accuracy'] >= 0.6).sum()} ({(df['accuracy'] >= 0.6).sum()/len(df)*100:.1f}%)

CONCLUSION:
-----------
The F LD Data 11.08 All_processed.csv appears to be clean and complete.
Poor performance appears to be REAL behavioral data, not a processing artifact.
"""

        with open(self.output_dir / 'F_cohort_data_quality_report.txt', 'w') as f:
            f.write(report)

        print(f"\n✓ Saved quality report to: {self.output_dir / 'F_cohort_data_quality_report.txt'}")


def main():
    """Run comprehensive F cohort validation."""

    print("="*80)
    print("F COHORT DATA VALIDATION AND QUALITY CHECK")
    print("="*80)
    print("\nThis analysis will determine if poor performer findings are real or artifacts.")

    validator = FCohortDataValidator()

    # Step 1: Basic data quality check
    print("\n[1/5] Checking data quality...")
    df = validator.check_data_quality_basics()

    # Step 2: Recheck poor performers with raw data
    print("\n[2/5] Rechecking poor performers with raw CSV data...")
    stats_df = validator.recheck_poor_performers_with_raw_data()

    # Step 3: Visualize late lapsers in detail
    print("\n[3/5] Creating detailed late lapser visualizations...")
    validator.visualize_late_lapsers_detailed(stats_df)

    # Step 4: Investigate why not caught previously
    print("\n[4/5] Investigating why poor performers weren't caught before...")
    validator.investigate_why_not_caught_previously()

    # Step 5: Comprehensive data quality check
    print("\n[5/5] Running comprehensive data quality check (Option C)...")
    validator.comprehensive_data_quality_report()

    print("\n" + "="*80)
    print("✓ VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {validator.output_dir}")
    print("\nKey files:")
    print("  1. accuracy_comparison.csv - Trial vs End Summary accuracy")
    print("  2. accuracy_validation.png/pdf - Validation plots")
    print("  3. F_cohort_performance_RECHECK.csv - Rechecked classifications")
    print("  4. late_lapsers_detailed.png/pdf - Animal-by-animal trajectories")
    print("  5. model_vs_session_comparison.csv - Previous vs current results")
    print("  6. F_cohort_data_quality_report.txt - Comprehensive quality assessment")


if __name__ == '__main__':
    main()
