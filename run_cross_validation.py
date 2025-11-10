"""
Run Cross-Validation on Representative Subset
==============================================

Tests 2, 3, 4, and 5 state models to validate our choice of 3 states.
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

# Select representative subset (2-3 animals per genotype, prioritizing high trial counts)
print("\n" + "="*80)
print("SELECTING REPRESENTATIVE SUBSET")
print("="*80)

selected_animals = []

# W Cohort - select 2 animals per genotype
print("\nW Cohort:")
for genotype in ['+', '-', '+/-']:
    animals = W_animals[genotype]
    if len(animals) > 0:
        # Sort by trial count, select top 2
        sorted_animals = sorted(animals, key=lambda x: x[1], reverse=True)[:2]
        for animal, n_trials in sorted_animals:
            selected_animals.append((animal, 'W'))
            print(f"  {str(animal):10s} ({genotype:5s}): {n_trials} trials")

# F Cohort - select 2-3 animals per genotype
print("\nF Cohort:")
for genotype in ['+', '-', '+/-', '-/-']:
    animals = F_animals[genotype]
    if len(animals) > 0:
        # For -/- (largest group), select 3; others select 2
        n_select = 3 if genotype == '-/-' else 2
        sorted_animals = sorted(animals, key=lambda x: x[1], reverse=True)[:n_select]
        for animal, n_trials in sorted_animals:
            selected_animals.append((animal, 'F'))
            print(f"  {str(animal):10s} ({genotype:5s}): {n_trials} trials")

print(f"\nTotal selected: {len(selected_animals)} animals")

# Run cross-validation
print("\n" + "="*80)
print("RUNNING CROSS-VALIDATION")
print("="*80)

cv_study = ModelValidator()
results_df = cv_study.run_validation_study(selected_animals)

if len(results_df) > 0:
    print("\n" + "="*80)
    print("CREATING SUMMARY VISUALIZATIONS")
    print("="*80)
    cv_study.create_summary_plots(results_df)

    print("\n" + "="*80)
    print("CROSS-VALIDATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {cv_study.output_dir}")
else:
    print("\nERROR: No results obtained!")
