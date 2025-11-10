"""
Comprehensive Cross-Validation for Both Cohorts
================================================

Tests 2, 3, 4, and 5 state models with larger representative samples
from each cohort to validate our choice of 3 states.

Strategy:
- W Cohort: 5-6 animals per genotype (total ~10 animals)
- F Cohort: 5-6 animals per genotype (total ~18-20 animals)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys
sys.path.insert(0, '/home/user/GLMHMM')

from priority2_model_validation import ModelValidator

# Load existing models to get animal-genotype mapping
results_dir = Path('results/phase1_non_reversal')
animal_genotypes = {}

print("Loading existing models to identify animals and genotypes...")
for pkl_file in results_dir.glob('*_model.pkl'):
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            animal_id = data['animal_id']
            genotype = data['genotype']
            cohort = data['cohort']
            n_trials = data['n_trials']
            animal_genotypes[animal_id] = {
                'genotype': genotype,
                'cohort': cohort,
                'n_trials': n_trials
            }
    except:
        continue

# Organize by cohort and genotype
W_animals = {g: [] for g in ['+', '-', '+/-']}
F_animals = {g: [] for g in ['+', '-', '+/-', '-/-']}

for animal, info in animal_genotypes.items():
    cohort = info['cohort']
    genotype = info['genotype']
    n_trials = info['n_trials']

    if cohort == 'W':
        if genotype in W_animals:
            W_animals[genotype].append((animal, n_trials))
    elif cohort == 'F':
        if genotype in F_animals:
            F_animals[genotype].append((animal, n_trials))

# Print distribution
print("\n" + "="*80)
print("ANIMAL DISTRIBUTION BY GENOTYPE")
print("="*80)
print("\nW Cohort:")
for genotype in ['+', '-', '+/-']:
    animals = W_animals[genotype]
    print(f"  {genotype:5s}: {len(animals):2d} animals")

print("\nF Cohort:")
for genotype in ['+', '-', '+/-', '-/-']:
    animals = F_animals[genotype]
    print(f"  {genotype:5s}: {len(animals):2d} animals")

# Select larger representative samples
print("\n" + "="*80)
print("SELECTING COMPREHENSIVE SUBSET")
print("="*80)

selected_animals = []

# W Cohort - select 4-5 animals per genotype for balanced representation
print("\nW Cohort:")
for genotype in ['+', '-']:
    animals = W_animals[genotype]
    if len(animals) > 0:
        # Select top 5 by trial count (or all if fewer than 5)
        n_select = min(5, len(animals))
        sorted_animals = sorted(animals, key=lambda x: x[1], reverse=True)[:n_select]
        for animal, n_trials in sorted_animals:
            selected_animals.append((animal, 'W'))
            print(f"  {str(animal):10s} ({genotype:5s}): {n_trials} trials")

# F Cohort - select 5-6 animals per genotype
print("\nF Cohort:")
for genotype in ['+', '+/-', '-/-']:
    animals = F_animals[genotype]
    if len(animals) > 0:
        # For +/- and -/- select more animals (up to 6), for + select 5
        n_select = min(6 if genotype in ['+/-', '-/-'] else 5, len(animals))
        sorted_animals = sorted(animals, key=lambda x: x[1], reverse=True)[:n_select]
        for animal, n_trials in sorted_animals:
            selected_animals.append((animal, 'F'))
            print(f"  {str(animal):10s} ({genotype:5s}): {n_trials} trials")

print(f"\nTotal selected: {len(selected_animals)} animals")
print(f"  W Cohort: {sum(1 for a, c in selected_animals if c == 'W')} animals")
print(f"  F Cohort: {sum(1 for a, c in selected_animals if c == 'F')} animals")

# Run cross-validation
print("\n" + "="*80)
print("RUNNING COMPREHENSIVE CROSS-VALIDATION")
print("="*80)
print("\nNote: This will take several hours. Results will be saved incrementally.")
print("You can stop and restart - completed animals won't be re-run.\n")

cv_study = ModelValidator()
results_df = cv_study.run_validation_study(selected_animals)

if len(results_df) > 0:
    print("\n" + "="*80)
    print("CREATING SUMMARY VISUALIZATIONS")
    print("="*80)
    cv_study.plot_validation_results(results_df)

    # Create cohort-specific summaries
    print("\n" + "="*80)
    print("COHORT-SPECIFIC SUMMARIES")
    print("="*80)

    for cohort in ['W', 'F']:
        cohort_data = results_df[results_df['cohort'] == cohort]
        if len(cohort_data) > 0:
            print(f"\n{cohort} Cohort Summary:")
            print(f"  Animals tested: {cohort_data['animal_id'].nunique()}")
            print(f"  Total trials: {cohort_data['n_trials'].sum()}")

            # Best model by AIC
            best_by_aic = cohort_data.groupby('n_states')['aic'].mean().idxmin()
            print(f"  Best model (AIC): {best_by_aic} states")

            # Best model by BIC
            best_by_bic = cohort_data.groupby('n_states')['bic'].mean().idxmin()
            print(f"  Best model (BIC): {best_by_bic} states")

            # Best accuracy
            best_acc_states = cohort_data.groupby('n_states')['accuracy'].mean().idxmax()
            best_acc_val = cohort_data.groupby('n_states')['accuracy'].mean().max()
            print(f"  Best accuracy: {best_acc_states} states ({best_acc_val:.3f})")

    print("\n" + "="*80)
    print("CROSS-VALIDATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {cv_study.output_dir}")
else:
    print("\nERROR: No results obtained!")
