#!/usr/bin/env python3
"""
Generate additional visualization figures beyond the main analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from glmhmm_utils import load_and_preprocess_session_data, create_design_matrix
from glmhmm_ashwood import GLMHMM
from additional_visualizations import (
    plot_session_by_session_learning,
    plot_task_specific_analysis,
    plot_weight_comparison_across_animals,
    plot_genotype_learning_curves
)

print("="*70)
print("GENERATING ADDITIONAL VISUALIZATION FIGURES")
print("="*70)

BASE_DIR = Path('/home/user/GLMHMM')
DATA_DIR = BASE_DIR
FIG_DIR = BASE_DIR / 'figures' / 'additional_analyses'

# Load data
print("\n[1] Loading data...")
w_file = DATA_DIR / 'W LD Data 11.08 All_processed.csv'
f_file = DATA_DIR / 'F LD Data 11.08 All_processed.csv'

if w_file.exists():
    print(f"✓ Loading W cohort: {w_file.name}")
    w_trials = load_and_preprocess_session_data(str(w_file))
    print(f"  Loaded {len(w_trials)} trials from {w_trials['animal_id'].nunique()} animals")
else:
    raise FileNotFoundError(f"W cohort file not found: {w_file}")

if f_file.exists():
    print(f"✓ Loading F cohort: {f_file.name}")
    f_trials = load_and_preprocess_session_data(str(f_file))
    print(f"  Loaded {len(f_trials)} trials from {f_trials['animal_id'].nunique()} animals")
else:
    print(f"⚠ F cohort file not found: {f_file}")
    f_trials = None

# Select animal with most trials
print("\n[2] Selecting animal for detailed analysis...")
trials_per_animal = w_trials.groupby('animal_id').size()
test_animal = trials_per_animal.idxmax()
print(f"Selected animal: {test_animal} ({trials_per_animal[test_animal]} trials)")

# Fit model
print("\n[3] Fitting GLM-HMM model...")
X, y, feature_names, metadata, animal_data = create_design_matrix(
    w_trials,
    animal_id=test_animal,
    include_position=True,
    include_session_progression=True
)

model = GLMHMM(
    n_states=3,
    feature_names=feature_names,
    normalize_features=True,
    regularization_strength=1.0,
    random_state=42
)
model.fit(X, y, n_iter=100, tolerance=1e-4, verbose=False)
print(f"✓ Model converged in {len(model.log_likelihood_history)} iterations")

# Create filtered trial_df for the test animal
animal_trial_df = w_trials[w_trials['animal_id'] == test_animal].copy()
animal_trial_df = animal_trial_df.reset_index(drop=True)

# Generate additional figures
print("\n[4] Generating additional figures...")

# Figure 1: Session-by-session learning
try:
    print("  Creating session-by-session learning figure...")
    fig = plot_session_by_session_learning(animal_trial_df, model, metadata, figsize=(16, 10))
    fig.savefig(FIG_DIR / '04_session_by_session_learning.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("    ✓ Saved: 04_session_by_session_learning.png")
except Exception as e:
    print(f"    ✗ Error: {e}")

# Figure 2: Task-specific analysis
try:
    print("  Creating task-specific analysis figure...")
    fig = plot_task_specific_analysis(animal_trial_df, model, metadata, figsize=(14, 10))
    fig.savefig(FIG_DIR / '05_task_specific_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("    ✓ Saved: 05_task_specific_analysis.png")
except Exception as e:
    print(f"    ✗ Error: {e}")

# Figure 3: Weight comparison across animals (requires multi-animal fitting)
try:
    print("  Creating weight comparison across animals...")
    print("    Fitting models for top 5 animals...")

    n_animals = min(5, w_trials['animal_id'].nunique())
    top_animals = trials_per_animal.nlargest(n_animals).index.tolist()

    multi_results = {}
    for animal in top_animals:
        X_i, y_i, fn_i, meta_i, data_i = create_design_matrix(
            w_trials, animal_id=animal, include_position=True
        )

        model_i = GLMHMM(
            n_states=3,
            feature_names=fn_i,
            normalize_features=True,
            regularization_strength=1.0
        )
        model_i.fit(X_i, y_i, n_iter=50, verbose=False)

        multi_results[animal] = {
            'model': model_i,
            'X': X_i,
            'y': y_i,
            'metadata': meta_i,
            'genotype': data_i['genotype'].iloc[0]
        }
        print(f"      ✓ Fitted model for {animal}")

    fig = plot_weight_comparison_across_animals(multi_results, figsize=(14, 8))
    fig.savefig(FIG_DIR / '06_weight_comparison_across_animals.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("    ✓ Saved: 06_weight_comparison_across_animals.png")
except Exception as e:
    print(f"    ✗ Error: {e}")

# Figure 4: Genotype learning curves
if f_trials is not None:
    try:
        print("  Creating genotype learning curves...")
        fig = plot_genotype_learning_curves(w_trials, f_trials, figsize=(14, 8))
        fig.savefig(FIG_DIR / '07_genotype_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("    ✓ Saved: 07_genotype_learning_curves.png")
    except Exception as e:
        print(f"    ✗ Error: {e}")

print("\n" + "="*70)
print("ADDITIONAL FIGURES GENERATION COMPLETE!")
print("="*70)
print(f"\nFigures saved to: {FIG_DIR}")
