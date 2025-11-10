"""
Regenerate Phase 1 analyses for both cohorts with pickle file outputs.
This enables summary visualizations and cross-cohort comparisons.
"""

import sys
sys.path.insert(0, '/home/user/GLMHMM')

from run_phase1_analysis import (
    analyze_single_animal_phase1,
    create_cohort_summary
)
from glmhmm_utils import load_and_preprocess_session_data
from pathlib import Path

# Paths
W_DATA = '/home/user/GLMHMM/W LD Data 11.08 All_processed.csv'
F_DATA = '/home/user/GLMHMM/F LD Data 11.08 All_processed.csv'
OUTPUT_DIR = Path('/home/user/GLMHMM/results/phase1_non_reversal')

def regenerate_cohort_W():
    """Regenerate W cohort with pickle files."""
    print("="*70)
    print("COHORT W PHASE 1 - REGENERATING WITH PICKLE FILES")
    print("="*70)
    print("\nLoading W cohort data...")

    trial_df = load_and_preprocess_session_data(W_DATA)
    animal_ids = trial_df['animal_id'].unique()

    print(f"\nAnalyzing {len(animal_ids)} W cohort animals...")
    print(f"Animals: {sorted(animal_ids)}")

    w_results = []
    for i, animal_id in enumerate(sorted(animal_ids), 1):
        print(f"\n[{i}/{len(animal_ids)}] Processing {animal_id}...")

        result = analyze_single_animal_phase1(
            trial_df=trial_df,
            animal_id=animal_id,
            cohort='W',
            n_states=3
        )

        if result is not None:
            w_results.append(result)
            genotype = result.get('genotype', 'Unknown')
            print(f"✓ {animal_id} - {genotype} genotype - {result['n_trials']} trials")

    print(f"\n{'='*70}")
    print(f"W Cohort Analysis Complete")
    print(f"Successfully analyzed: {len(w_results)}/{len(animal_ids)} animals")
    print(f"{'='*70}")

    # Create cohort summary
    print("\nGenerating W cohort summaries and genotype comparisons...")
    create_cohort_summary(w_results, 'W', OUTPUT_DIR)

    return w_results


def regenerate_cohort_F():
    """Regenerate F cohort with pickle files and correct genotypes."""
    print("\n" + "="*70)
    print("COHORT F PHASE 1 - REGENERATING WITH PICKLE FILES")
    print("="*70)
    print("\nLoading F cohort data...")

    trial_df = load_and_preprocess_session_data(F_DATA)

    # Filter out animal 105 (duplicate of 104)
    print(f"Total animals before filtering: {trial_df['animal_id'].nunique()}")
    trial_df = trial_df[trial_df['animal_id'] != 105].copy()
    print(f"Total animals after removing 105: {trial_df['animal_id'].nunique()}")

    # Check genotypes
    print("\nGenotype distribution:")
    geno_counts = trial_df.groupby('genotype')['animal_id'].nunique()
    print(geno_counts)

    animal_ids = trial_df['animal_id'].unique()
    print(f"\nAnalyzing {len(animal_ids)} F cohort animals...")
    print(f"Animals: {sorted(animal_ids)}")

    f_results = []
    for i, animal_id in enumerate(sorted(animal_ids), 1):
        print(f"\n[{i}/{len(animal_ids)}] Processing {animal_id}...")

        result = analyze_single_animal_phase1(
            trial_df=trial_df,
            animal_id=animal_id,
            cohort='F',
            n_states=3
        )

        if result is not None:
            f_results.append(result)
            print(f"✓ {animal_id} - {result['genotype']} genotype - {result['n_trials']} trials")

    print(f"\n{'='*70}")
    print(f"F Cohort Analysis Complete")
    print(f"Successfully analyzed: {len(f_results)}/{len(animal_ids)} animals")
    print(f"{'='*70}")

    # Create cohort summary
    print("\nGenerating F cohort summaries and genotype comparisons...")
    create_cohort_summary(f_results, 'F', OUTPUT_DIR)

    return f_results


def main():
    """Regenerate both cohorts."""
    print("\n" + "="*80)
    print("REGENERATING PHASE 1 ANALYSES WITH PICKLE FILES")
    print("="*80)
    print("\nThis will enable comprehensive summary visualizations and comparisons.")
    print("Existing plots will be overwritten.\n")

    # Regenerate W cohort
    w_results = regenerate_cohort_W()

    # Regenerate F cohort
    f_results = regenerate_cohort_F()

    # Final summary
    print("\n" + "="*80)
    print("REGENERATION COMPLETE!")
    print("="*80)
    print(f"\nCohort W: {len(w_results)} animals analyzed")
    print(f"Cohort F: {len(f_results)} animals analyzed")
    print(f"Total: {len(w_results) + len(f_results)} animals")
    print(f"\nPickle files saved to: {OUTPUT_DIR}")
    print("\nYou can now run: python create_cohort_summaries.py")
    print("="*80)


if __name__ == '__main__':
    main()
