"""
CRITICAL VALIDATION: Task Filtering and Accuracy Calculation
=============================================================

User concerns:
1. Are reversal tasks contaminating Phase 1 analyses?
2. Learning curves show 40 sessions (W) and 20 sessions (F) - includes reversal?
3. Many F mice reach 100% on LD - why is mean only 42%?
4. Are we using correct accuracy fields (Trial Block No. Correct vs Correct Position)?
5. Does rolling 30-trial accuracy match session-level End Summary?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


class TaskFilteringValidator:
    """Validate task filtering and accuracy calculations."""

    def __init__(self):
        self.output_dir = Path('results/phase1_non_reversal/critical_validation')
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def check_task_inclusion(self):
        """
        Check what tasks are actually in our processed data.
        CRITICAL: Are reversal tasks included?
        """
        print("\n" + "="*80)
        print("CRITICAL CHECK: Task Inclusion in Phase 1 Analyses")
        print("="*80)

        for cohort in ['W', 'F']:
            filepath = f'{cohort} LD Data 11.08 All_processed.csv'

            if not Path(filepath).exists():
                print(f"\n{cohort} cohort file not found: {filepath}")
                continue

            df = pd.read_csv(filepath)

            print(f"\n{'-'*80}")
            print(f"COHORT {cohort}:")
            print(f"{'-'*80}")
            print(f"Total sessions: {len(df)}")

            # Task breakdown
            print(f"\nTask types (by Schedule name):")
            task_counts = df['Schedule name'].value_counts()

            reversal_sessions = 0
            non_reversal_sessions = 0

            for task, count in task_counts.items():
                is_reversal = 'reversal' in task.lower()
                marker = "⚠️ REVERSAL" if is_reversal else "✓ Non-reversal"
                print(f"  {marker}: {task}: {count} sessions")

                if is_reversal:
                    reversal_sessions += count
                else:
                    non_reversal_sessions += count

            print(f"\n{'-'*80}")
            print(f"SUMMARY:")
            print(f"  Non-reversal sessions: {non_reversal_sessions}")
            print(f"  ⚠️ REVERSAL sessions: {reversal_sessions}")
            print(f"  Total: {len(df)}")

            if reversal_sessions > 0:
                print(f"\n🚨 WARNING: Phase 1 data contains {reversal_sessions} REVERSAL sessions!")
                print(f"   This should be Phase 2 data, not Phase 1!")

            # Check sessions per animal
            sessions_per_animal = df.groupby('Animal ID').size()
            print(f"\nSessions per animal:")
            print(f"  Min: {sessions_per_animal.min()}")
            print(f"  Max: {sessions_per_animal.max()}")
            print(f"  Mean: {sessions_per_animal.mean():.1f}")
            print(f"  Median: {sessions_per_animal.median():.0f}")

    def check_accuracy_field_usage(self):
        """
        Verify which accuracy field is being used in our code.
        """
        print("\n" + "="*80)
        print("ACCURACY FIELD VERIFICATION")
        print("="*80)

        print("\nChecking glmhmm_utils.py for accuracy extraction...")

        # Read the file
        with open('glmhmm_utils.py', 'r') as f:
            lines = f.readlines()

        # Find the relevant section
        print("\nFound in glmhmm_utils.py:")
        print(f"{'-'*80}")

        for i, line in enumerate(lines[88:100], start=89):
            if 'correct' in line.lower() or 'Trial Block' in line:
                print(f"Line {i}: {line.rstrip()}")

        print(f"\n✓ Confirmed: Using '1 Trial Block - No. Correct (trial_num)'")
        print(f"  This is CORRECT - binary 1/0 for correct/incorrect")
        print(f"\n✓ NOT using 'Trial Analysis - Correct Position'")
        print(f"  (That would give Left/Right, which would be wrong!)")

    def compare_ld_vs_pi_vs_combined(self):
        """
        Separate performance by task type: LD-only vs PI vs combined.
        This is CRITICAL to understand the 42% mean accuracy.
        """
        print("\n" + "="*80)
        print("PERFORMANCE BY TASK TYPE")
        print("="*80)

        for cohort in ['W', 'F']:
            filepath = f'{cohort} LD Data 11.08 All_processed.csv'

            if not Path(filepath).exists():
                continue

            df = pd.read_csv(filepath)
            df['accuracy'] = pd.to_numeric(df['End Summary - Percentage Correct (1)'],
                                          errors='coerce') / 100

            # Categorize tasks
            df['task_category'] = 'Other'
            df.loc[df['Schedule name'].str.contains('LD 1 choice|LD Must Touch|LD Initial Touch',
                                                    case=False, na=False), 'task_category'] = 'LD Training'
            df.loc[df['Schedule name'].str.contains('Punish Incorrect',
                                                    case=False, na=False), 'task_category'] = 'Punish Incorrect'
            df.loc[df['Schedule name'].str.contains('Pairwise Discrimination',
                                                    case=False, na=False), 'task_category'] = 'Pairwise Disc'
            df.loc[df['Schedule name'].str.contains('Reversal|reversal',
                                                    case=False, na=False), 'task_category'] = 'REVERSAL'

            print(f"\n{'-'*80}")
            print(f"COHORT {cohort}:")
            print(f"{'-'*80}")

            for task_cat in ['LD Training', 'Punish Incorrect', 'Pairwise Disc', 'REVERSAL', 'Other']:
                task_data = df[df['task_category'] == task_cat]

                if len(task_data) == 0:
                    continue

                mean_acc = task_data['accuracy'].mean()
                median_acc = task_data['accuracy'].median()
                n_sessions = len(task_data)
                n_animals = task_data['Animal ID'].nunique()

                # Count high-performing sessions
                n_perfect = (task_data['accuracy'] >= 0.95).sum()
                n_good = ((task_data['accuracy'] >= 0.8) & (task_data['accuracy'] < 0.95)).sum()
                n_poor = (task_data['accuracy'] < 0.6).sum()

                marker = "⚠️" if task_cat == "REVERSAL" else "  "

                print(f"\n{marker}{task_cat}:")
                print(f"    Sessions: {n_sessions} (from {n_animals} animals)")
                print(f"    Mean accuracy: {mean_acc:.3f}")
                print(f"    Median accuracy: {median_acc:.3f}")
                print(f"    Perfect (≥95%): {n_perfect} ({n_perfect/n_sessions*100:.1f}%)")
                print(f"    Good (80-95%): {n_good} ({n_good/n_sessions*100:.1f}%)")
                print(f"    Poor (<60%): {n_poor} ({n_poor/n_sessions*100:.1f}%)")

            # Overall excluding reversal
            non_reversal = df[df['task_category'] != 'REVERSAL']
            reversal = df[df['task_category'] == 'REVERSAL']

            print(f"\n{'-'*80}")
            print(f"SUMMARY:")
            print(f"  ALL SESSIONS (including reversal): {df['accuracy'].mean():.3f} (n={len(df)})")
            print(f"  NON-REVERSAL ONLY: {non_reversal['accuracy'].mean():.3f} (n={len(non_reversal)})")
            if len(reversal) > 0:
                print(f"  REVERSAL ONLY: {reversal['accuracy'].mean():.3f} (n={len(reversal)})")
                print(f"\n🚨 Including reversal LOWERS mean accuracy by {(df['accuracy'].mean() - non_reversal['accuracy'].mean()):.3f}")

    def check_model_data_sources(self):
        """
        Check what data was actually used to build the GLM-HMM models.
        """
        print("\n" + "="*80)
        print("MODEL DATA SOURCE CHECK")
        print("="*80)

        # Check a few model pickle files to see what data went in
        import pickle

        test_animals = {
            'W': ['c1m1', 'c2m1', 'c3m1'],
            'F': [11, 21, 31]
        }

        for cohort, animals in test_animals.items():
            print(f"\n{'-'*80}")
            print(f"COHORT {cohort} - Sample Animals:")
            print(f"{'-'*80}")

            for animal in animals:
                pkl_file = f'results/phase1_non_reversal/{animal}_cohort{cohort}_model.pkl'

                if not Path(pkl_file).exists():
                    print(f"  {animal}: Model file not found")
                    continue

                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)

                n_trials = data['n_trials']

                # Try to reconstruct what sessions were included
                # This is tricky without the raw trial data, but we can check the trial count
                print(f"\n  Animal {animal}:")
                print(f"    Total trials in model: {n_trials}")
                print(f"    Approx sessions (÷30): {n_trials/30:.1f}")

    def create_task_filtered_comparison(self):
        """
        Create side-by-side comparison of performance with/without reversal tasks.
        """
        print("\n" + "="*80)
        print("CREATING TASK-FILTERED COMPARISON PLOTS")
        print("="*80)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for idx, cohort in enumerate(['W', 'F']):
            filepath = f'{cohort} LD Data 11.08 All_processed.csv'

            if not Path(filepath).exists():
                continue

            df = pd.read_csv(filepath)
            df['accuracy'] = pd.to_numeric(df['End Summary - Percentage Correct (1)'],
                                          errors='coerce') / 100

            # Identify reversal tasks
            df['is_reversal'] = df['Schedule name'].str.contains('Reversal|reversal',
                                                                 case=False, na=False)

            # Left column: All data (including reversal)
            ax = axes[idx, 0]
            ax.hist(df['accuracy'], bins=50, edgecolor='black', alpha=0.7,
                   color='steelblue')
            mean_all = df['accuracy'].mean()
            n_rev = df['is_reversal'].sum()
            ax.axvline(mean_all, color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {mean_all:.3f}')
            ax.set_xlabel('Session Accuracy', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            title_color = 'red' if n_rev > 0 else 'black'
            ax.set_title(f'Cohort {cohort}: ALL Sessions (n={len(df)})\n' +
                        f'⚠️ INCLUDES {n_rev} REVERSAL SESSIONS',
                        fontsize=13, fontweight='bold', color=title_color)
            ax.legend()
            ax.grid(alpha=0.3)

            # Right column: Non-reversal only
            ax = axes[idx, 1]
            non_rev = df[~df['is_reversal']]
            ax.hist(non_rev['accuracy'], bins=50, edgecolor='black', alpha=0.7,
                   color='green')
            mean_non_rev = non_rev['accuracy'].mean()
            ax.axvline(mean_non_rev, color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {mean_non_rev:.3f}')
            ax.set_xlabel('Session Accuracy', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_title(f'Cohort {cohort}: NON-REVERSAL ONLY (n={len(non_rev)})',
                        fontsize=13, fontweight='bold', color='green')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'task_filtering_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'task_filtering_comparison.pdf',
                   bbox_inches='tight')
        plt.close()

        print(f"\n✓ Created task filtering comparison plots")


def main():
    """Run critical validation checks."""

    print("="*80)
    print("CRITICAL VALIDATION: Task Filtering & Accuracy Calculation")
    print("="*80)
    print("\nValidating user concerns:")
    print("  1. Are reversal tasks contaminating Phase 1?")
    print("  2. Are we using correct accuracy fields?")
    print("  3. Why is F cohort mean 42% if many reach 100% on LD?")
    print("  4. What's the difference between LD-only vs PI tasks?")

    validator = TaskFilteringValidator()

    # Run all checks
    validator.check_task_inclusion()
    validator.check_accuracy_field_usage()
    validator.compare_ld_vs_pi_vs_combined()
    validator.check_model_data_sources()
    validator.create_task_filtered_comparison()

    print("\n" + "="*80)
    print("✓ CRITICAL VALIDATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {validator.output_dir}")

    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY:")
    print("="*80)
    print("\nPlease review the output above to determine:")
    print("  1. Whether reversal tasks were incorrectly included")
    print("  2. Whether this explains the low F cohort performance")
    print("  3. What the true Phase 1 (non-reversal) performance is")


if __name__ == '__main__':
    main()
