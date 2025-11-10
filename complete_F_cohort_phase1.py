"""
Complete Phase 1 F Cohort Analysis with Correct Genotypes
==========================================================

Re-analyzes F cohort with:
- Animal 105 excluded (duplicate of 104)
- Correct genotype handling: +/+, +/-, -/-, +
- Full genotype comparison analyses
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
F_DATA = '/home/user/GLMHMM/F LD Data 11.08 All_processed.csv'
OUTPUT_DIR = Path('/home/user/GLMHMM/results/phase1_non_reversal')

print("="*70)
print("F COHORT PHASE 1 ANALYSIS - CORRECTED GENOTYPES")
print("="*70)
print("\nLoading F cohort data...")

# Load data
trial_df = load_and_preprocess_session_data(F_DATA)

# Filter out animal 105 (duplicate of 104 in PI session)
print(f"Total animals before filtering: {trial_df['animal_id'].nunique()}")
trial_df = trial_df[trial_df['animal_id'] != 105].copy()  # Use integer, not string
print(f"Total animals after removing 105: {trial_df['animal_id'].nunique()}")

# Check genotypes
print("\nGenotype distribution:")
geno_counts = trial_df.groupby('genotype')['animal_id'].nunique()
print(geno_counts)

# Get unique animals
animal_ids = trial_df['animal_id'].unique()
print(f"\nAnalyzing {len(animal_ids)} F cohort animals...")
print(f"Animals: {sorted(animal_ids)}")

# Analyze each animal
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

# Create cohort summary with genotype comparisons
print("\nGenerating F cohort summaries and genotype comparisons...")
create_cohort_summary(f_results, 'F', OUTPUT_DIR)

print("\n" + "="*70)
print("F COHORT PHASE 1 COMPLETE")
print("="*70)
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"Genotype comparisons: {OUTPUT_DIR}/cohort_F_genotype_comparisons/")
