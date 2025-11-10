"""
Investigate Accuracy Mismatches
================================

Find out WHICH tasks/schedules have mismatches between trial-level accuracy
and End Summary - Percentage Correct.
"""

import numpy as np
import pandas as pd
from pathlib import Path

print("="*80)
print("INVESTIGATING ACCURACY MISMATCHES BY TASK TYPE")
print("="*80)

for cohort in ['W', 'F']:
    print(f"\n{'-'*80}")
    print(f"COHORT {cohort}")
    print(f"{'-'*80}")

    # Load comparison data
    comp_file = f'results/phase1_non_reversal/critical_validation/accuracy_comparison_{cohort}.csv'
    comp_df = pd.read_csv(comp_file)

    # Load session data to get Schedule names
    if cohort == 'W':
        csv_file = 'W LD Data 11.08 All_processed.csv'
    else:
        csv_file = 'F LD Data 11.08 All_processed.csv'

    df_sessions = pd.read_csv(csv_file)
    df_sessions['session_date'] = pd.to_datetime(df_sessions['Schedule run date'], errors='coerce')

    # Merge to get Schedule names
    comp_df['session_date'] = pd.to_datetime(comp_df['session_date'])

    merged = comp_df.merge(
        df_sessions[['Animal ID', 'Schedule run date', 'Schedule name']],
        left_on=['animal_id', 'session_date'],
        right_on=['Animal ID', pd.to_datetime(df_sessions['Schedule run date'], errors='coerce')],
        how='left'
    )

    # Filter to mismatches
    mismatches = merged[~merged['match']].copy()

    print(f"\nMismatches by Schedule type:")
    print(f"{'-'*80}")

    schedule_summary = mismatches.groupby('Schedule name').agg({
        'animal_id': 'count',
        'difference': ['mean', 'median', 'min', 'max']
    }).round(3)

    schedule_summary.columns = ['Count', 'Mean Diff', 'Median Diff', 'Min Diff', 'Max Diff']
    print(schedule_summary.sort_values('Count', ascending=False).to_string())

    # Check if mismatches are concentrated in specific task types
    print(f"\n{'-'*80}")
    print(f"Mismatch breakdown:")
    print(f"{'-'*80}")

    total_mismatches = len(mismatches)
    initial_touch = mismatches[mismatches['Schedule name'].str.contains('Initial Touch', case=False, na=False)]
    must_touch = mismatches[mismatches['Schedule name'].str.contains('Must Touch', case=False, na=False)]

    print(f"Total mismatches: {total_mismatches}")
    print(f"  LD Initial Touch: {len(initial_touch)} ({len(initial_touch)/total_mismatches*100:.1f}%)")
    print(f"  LD Must Touch: {len(must_touch)} ({len(must_touch)/total_mismatches*100:.1f}%)")
    print(f"  Other tasks: {total_mismatches - len(initial_touch) - len(must_touch)} " +
          f"({(total_mismatches - len(initial_touch) - len(must_touch))/total_mismatches*100:.1f}%)")

    # Check session_acc = 0 but trial_acc = 1 pattern
    zero_to_one = mismatches[(mismatches['session_acc'] == 0.0) & (mismatches['trial_acc'] == 1.0)]
    print(f"\nSessions with session_acc=0 but trial_acc=1: {len(zero_to_one)} ({len(zero_to_one)/total_mismatches*100:.1f}% of mismatches)")

    if len(zero_to_one) > 0:
        print(f"\nSchedule types for session_acc=0 / trial_acc=1 cases:")
        print(zero_to_one['Schedule name'].value_counts().to_string())

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nIf mismatches are concentrated in LD Initial Touch / LD Must Touch,")
print("then 'End Summary - Percentage Correct' may be incorrect for those tasks.")
print("We should use trial-level accuracy instead.")
