"""
Test Phase 1 Analysis on Single Animal
=======================================

Quick test to verify Phase 1 pipeline works correctly.
"""

import sys
import pandas as pd
from run_phase1_analysis import analyze_single_animal_phase1
from glmhmm_utils import load_and_preprocess_session_data

# Test on single W cohort animal
W_DATA = '/home/user/GLMHMM/W LD Data 11.08 All_processed.csv'

# Load data
print("Loading data...")
trial_df = load_and_preprocess_session_data(W_DATA)

# Get first animal ID
first_animal = trial_df['animal_id'].iloc[0]

print(f"\nTesting Phase 1 analysis on animal: {first_animal}")

# Run analysis
results = analyze_single_animal_phase1(
    trial_df=trial_df,
    animal_id=first_animal,
    cohort='W',
    n_states=3
)

if results is not None:
    print("\n" + "="*70)
    print("TEST SUCCESSFUL")
    print("="*70)
    print(f"Analyzed {results['n_trials']} trials")
    print(f"Model log-likelihood: {results['model'].log_likelihood_history[-1]:.2f}")
    print("\nValidated state labels:")
    for state in range(3):
        label, confidence, _ = results['validated_labels'][state]
        print(f"  State {state}: {label} (confidence={confidence})")
else:
    print("\n" + "="*70)
    print("TEST FAILED")
    print("="*70)
    sys.exit(1)
