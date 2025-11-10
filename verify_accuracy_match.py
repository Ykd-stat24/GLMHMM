"""
Verify Rolling Accuracy vs End Summary Percentage Correct
==========================================================

User concern: Does rolling accuracy calculated from trial-level 'correct' field
match the session-level 'End Summary - Percentage Correct'?

This is CRITICAL to validate our analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from glmhmm_utils import load_and_preprocess_session_data

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def verify_accuracy_calculations():
    """
    Compare rolling trial-level accuracy with End Summary session accuracy.
    """
    print("="*80)
    print("VERIFYING ACCURACY CALCULATIONS")
    print("="*80)

    for cohort in ['W', 'F']:
        print(f"\n{'-'*80}")
        print(f"COHORT {cohort}")
        print(f"{'-'*80}")

        # Load data
        if cohort == 'W':
            csv_file = 'W LD Data 11.08 All_processed.csv'
        else:
            csv_file = 'F LD Data 11.08 All_processed.csv'

        # Load raw CSV to get End Summary
        df_sessions = pd.read_csv(csv_file)
        df_sessions['session_accuracy'] = pd.to_numeric(
            df_sessions['End Summary - Percentage Correct (1)'],
            errors='coerce'
        ) / 100
        df_sessions['session_date'] = pd.to_datetime(df_sessions['Schedule run date'], errors='coerce')
        df_sessions = df_sessions.sort_values(['Animal ID', 'session_date'])

        # Load trial-level data
        df_trials = load_and_preprocess_session_data(csv_file)

        # For each session, compute accuracy from trials and compare with End Summary
        comparisons = []

        for _, session_row in df_sessions.iterrows():
            animal_id = session_row['Animal ID']
            session_date = session_row['session_date']
            session_acc = session_row['session_accuracy']

            # Find trials for this session
            # Trials have animal_id and session_date columns
            session_trials = df_trials[
                (df_trials['animal_id'] == animal_id) &
                (df_trials['session_date'] == session_date)
            ]

            if len(session_trials) == 0:
                continue

            # Compute accuracy from trial-level 'correct' field
            trial_acc = session_trials['correct'].mean()

            # Compare
            match = np.abs(trial_acc - session_acc) < 0.01  # Allow 1% difference

            comparisons.append({
                'animal_id': animal_id,
                'session_date': session_date,
                'session_acc': session_acc,
                'trial_acc': trial_acc,
                'difference': trial_acc - session_acc,
                'match': match,
                'n_trials': len(session_trials)
            })

        comp_df = pd.DataFrame(comparisons)

        # Report findings
        print(f"\nTotal sessions compared: {len(comp_df)}")
        print(f"Sessions with matching accuracy: {comp_df['match'].sum()} ({comp_df['match'].sum()/len(comp_df)*100:.1f}%)")
        print(f"Sessions with mismatch: {(~comp_df['match']).sum()} ({(~comp_df['match']).sum()/len(comp_df)*100:.1f}%)")

        if (~comp_df['match']).sum() > 0:
            print(f"\nMismatch statistics:")
            print(f"  Mean difference: {comp_df['difference'].mean():.4f}")
            print(f"  Median difference: {comp_df['difference'].median():.4f}")
            print(f"  Max absolute difference: {comp_df['difference'].abs().max():.4f}")

            # Show examples of mismatches
            print(f"\nTop 10 largest mismatches:")
            mismatches = comp_df[~comp_df['match']].sort_values('difference', key=abs, ascending=False).head(10)
            print(mismatches[['animal_id', 'session_acc', 'trial_acc', 'difference', 'n_trials']].to_string())

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: Scatter plot
        ax = axes[0]
        ax.scatter(comp_df['session_acc'], comp_df['trial_acc'],
                  alpha=0.5, s=30, edgecolor='black', linewidth=0.5)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect match')
        ax.set_xlabel('End Summary - Percentage Correct', fontsize=12, fontweight='bold')
        ax.set_ylabel('Trial-Level Accuracy (mean)', fontsize=12, fontweight='bold')
        ax.set_title(f'Cohort {cohort}: Session vs Trial Accuracy', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Panel 2: Difference histogram
        ax = axes[1]
        ax.hist(comp_df['difference'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
        ax.set_xlabel('Difference (Trial - Session)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Differences', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel 3: Difference vs n_trials
        ax = axes[2]
        ax.scatter(comp_df['n_trials'], comp_df['difference'].abs(),
                  alpha=0.5, s=30, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Number of Trials', fontsize=12, fontweight='bold')
        ax.set_ylabel('|Difference|', fontsize=12, fontweight='bold')
        ax.set_title('Mismatch vs Trial Count', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)

        plt.tight_layout()

        output_dir = Path('results/phase1_non_reversal/critical_validation')
        output_dir.mkdir(exist_ok=True, parents=True)

        plt.savefig(output_dir / f'accuracy_verification_{cohort}.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'accuracy_verification_{cohort}.pdf', bbox_inches='tight')
        plt.close()

        print(f"\n✓ Created verification plots: {output_dir / f'accuracy_verification_{cohort}.png'}")

        # Save comparison data
        comp_df.to_csv(output_dir / f'accuracy_comparison_{cohort}.csv', index=False)
        print(f"✓ Saved comparison data: {output_dir / f'accuracy_comparison_{cohort}.csv'}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nIf trial-level accuracy matches End Summary, our analysis is correct.")
    print("If there are systematic mismatches, we need to investigate why.")


if __name__ == '__main__':
    verify_accuracy_calculations()
