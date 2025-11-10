"""
Generate F Cohort Summaries from Successfully Analyzed Results
===============================================================

All 35 F cohort animals (11-104) were successfully analyzed.
This script loads those results and generates cohort-level summaries
and genotype comparisons.
"""

import pickle
import numpy as np
from pathlib import Path
from run_phase1_analysis import create_cohort_summary

# Paths
OUTPUT_DIR = Path('/home/user/GLMHMM/results/phase1_non_reversal')

print("="*70)
print("GENERATING F COHORT SUMMARIES FROM ANALYZED RESULTS")
print("="*70)

# Load results from individual animal pickle files
f_animals = list(range(11, 105))  # Animals 11-104 (35 total)
f_results = []

print(f"\nLoading results for {len(f_animals)} F cohort animals...")

for animal_id in f_animals:
    animal_dir = OUTPUT_DIR / 'figures' / f'{animal_id}_cohortF'
    result_file = animal_dir / f'{animal_id}_results.pkl'

    if result_file.exists():
        with open(result_file, 'rb') as f:
            result = pickle.load(f)
            f_results.append(result)
            print(f"  ✓ Loaded {animal_id} - {result['genotype']} genotype")
    else:
        print(f"  ✗ Missing results for {animal_id}")

print(f"\nLoaded {len(f_results)}/{len(f_animals)} animal results")

# Generate cohort summary with genotype comparisons
print("\nGenerating F cohort summaries and genotype comparisons...")
create_cohort_summary(f_results, 'F', OUTPUT_DIR)

print("\n" + "="*70)
print("F COHORT SUMMARIES COMPLETE")
print("="*70)
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"Genotype comparisons: {OUTPUT_DIR}/cohort_F_genotype_comparisons/")
