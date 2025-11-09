"""
Test Phase 1 Analysis on Both W and F Cohorts
=============================================

Analyze animals from both cohorts to show diverse examples.
"""

import pandas as pd
from run_phase1_analysis import analyze_single_animal_phase1
from glmhmm_utils import load_and_preprocess_session_data

# Load both cohorts
W_DATA = '/home/user/GLMHMM/W LD Data 11.08 All_processed.csv'
F_DATA = '/home/user/GLMHMM/F LD Data 11.08 All_processed.csv'

print("="*70)
print("PHASE 1 ANALYSIS: W AND F COHORTS")
print("="*70)

results_all = []

# ===== W COHORT =====
print("\n" + "="*70)
print("W COHORT ANALYSIS")
print("="*70)

w_trial_df = load_and_preprocess_session_data(W_DATA)
w_animals = w_trial_df['animal_id'].unique()
w_trial_counts = w_trial_df.groupby('animal_id').size().sort_values(ascending=False)

print(f"\nW Cohort: {len(w_animals)} animals")
print(f"Top animals by trial count:")
print(w_trial_counts.head(10).to_string())

# Analyze top 3 W animals (excluding c1m1)
w_animals_to_analyze = [a for a in w_trial_counts.index if a != 'c1m1'][:3]

for i, animal_id in enumerate(w_animals_to_analyze, 1):
    print(f"\n[W {i}/3] Analyzing {animal_id}...")

    results = analyze_single_animal_phase1(
        trial_df=w_trial_df,
        animal_id=animal_id,
        cohort='W',
        n_states=3
    )

    if results is not None:
        results_all.append(results)
        print(f"\n✓ {animal_id} (W cohort) - {results['genotype']} genotype")
        print(f"  Trials: {results['n_trials']}, LL: {results['model'].log_likelihood_history[-1]:.2f}")
        for state in range(3):
            label, conf, _ = results['validated_labels'][state]
            print(f"  State {state}: {label} (conf={conf})")

# ===== F COHORT =====
print("\n" + "="*70)
print("F COHORT ANALYSIS")
print("="*70)

f_trial_df = load_and_preprocess_session_data(F_DATA)
f_animals = f_trial_df['animal_id'].unique()
f_trial_counts = f_trial_df.groupby('animal_id').size().sort_values(ascending=False)

print(f"\nF Cohort: {len(f_animals)} animals")
print(f"Top animals by trial count:")
print(f_trial_counts.head(10).to_string())

# Analyze top 3 F animals
f_animals_to_analyze = f_trial_counts.index[:3]

for i, animal_id in enumerate(f_animals_to_analyze, 1):
    print(f"\n[F {i}/3] Analyzing {animal_id}...")

    results = analyze_single_animal_phase1(
        trial_df=f_trial_df,
        animal_id=animal_id,
        cohort='F',
        n_states=3
    )

    if results is not None:
        results_all.append(results)
        print(f"\n✓ {animal_id} (F cohort) - {results['genotype']} genotype")
        print(f"  Trials: {results['n_trials']}, LL: {results['model'].log_likelihood_history[-1]:.2f}")
        for state in range(3):
            label, conf, _ = results['validated_labels'][state]
            print(f"  State {state}: {label} (conf={conf})")

# ===== FINAL SUMMARY =====
print("\n" + "="*70)
print("COMBINED ANALYSIS SUMMARY")
print("="*70)
print(f"\nTotal animals analyzed: {len(results_all)}")
print(f"  W cohort: {sum(1 for r in results_all if r['cohort'] == 'W')}")
print(f"  F cohort: {sum(1 for r in results_all if r['cohort'] == 'F')}")

print(f"\n{'Animal':<10} {'Cohort':<8} {'Genotype':<12} {'Trials':<8} {'LL':<10} {'State Labels'}")
print("-" * 100)

for r in results_all:
    labels = []
    for state in range(3):
        label, conf, _ = r['validated_labels'][state]
        labels.append(f"{label[:18]}")

    label_str = " | ".join(labels)

    print(f"{r['animal_id']:<10} {r['cohort']:<8} {r['genotype']:<12} "
          f"{r['n_trials']:<8} {r['model'].log_likelihood_history[-1]:<10.2f} {label_str}")

# Group by genotype
print("\n" + "="*70)
print("STATE LABELS BY GENOTYPE")
print("="*70)

genotypes = {}
for r in results_all:
    geno = r['genotype']
    if geno not in genotypes:
        genotypes[geno] = []

    for state in range(3):
        label, conf, _ = r['validated_labels'][state]
        if conf > 0:  # Only count validated states
            genotypes[geno].append(label)

for geno in sorted(genotypes.keys()):
    print(f"\n{geno} genotype:")
    from collections import Counter
    label_counts = Counter(genotypes[geno])
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")

print("\n" + "="*70)
print("Output locations:")
print("  W cohort: /home/user/GLMHMM/results/phase1_non_reversal/figures/*_cohortW/")
print("  F cohort: /home/user/GLMHMM/results/phase1_non_reversal/figures/*_cohortF/")
print("="*70)
