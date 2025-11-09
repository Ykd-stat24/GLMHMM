"""
Test Phase 1 Analysis on Multiple Animals
=========================================

Analyze several animals from W cohort to show diverse examples.
"""

import pandas as pd
from run_phase1_analysis import analyze_single_animal_phase1
from glmhmm_utils import load_and_preprocess_session_data

# Load W cohort data
W_DATA = '/home/user/GLMHMM/W LD Data 11.08 All_processed.csv'

print("Loading W cohort data...")
trial_df = load_and_preprocess_session_data(W_DATA)

# Get animal list
animals = trial_df['animal_id'].unique()
print(f"\nAvailable animals in W cohort: {len(animals)}")
print(f"Animal IDs: {list(animals)}")

# Count trials per animal
trial_counts = trial_df.groupby('animal_id').size().sort_values(ascending=False)
print(f"\nTrials per animal:")
print(trial_counts.to_string())

# Analyze top 5 animals by trial count (excluding c1m1)
animals_to_analyze = [a for a in trial_counts.index if a != 'c1m1'][:5]

print(f"\n{'='*70}")
print(f"Analyzing {len(animals_to_analyze)} animals: {animals_to_analyze}")
print(f"{'='*70}")

results_summary = []

for i, animal_id in enumerate(animals_to_analyze, 1):
    print(f"\n{'*'*70}")
    print(f"[{i}/{len(animals_to_analyze)}] Analyzing {animal_id}...")
    print(f"{'*'*70}")

    results = analyze_single_animal_phase1(
        trial_df=trial_df,
        animal_id=animal_id,
        cohort='W',
        n_states=3
    )

    if results is not None:
        # Collect summary
        summary = {
            'animal_id': animal_id,
            'genotype': results['genotype'],
            'n_trials': results['n_trials'],
            'log_likelihood': results['model'].log_likelihood_history[-1],
            'states': {}
        }

        for state in range(3):
            label, confidence, evidence = results['validated_labels'][state]
            summary['states'][state] = {
                'label': label,
                'confidence': confidence,
                'evidence_count': len(evidence)
            }

        results_summary.append(summary)

        print(f"\n✓ {animal_id} Analysis Complete")
        print(f"  Genotype: {summary['genotype']}")
        print(f"  Trials: {summary['n_trials']}")
        print(f"  Log-likelihood: {summary['log_likelihood']:.2f}")
        print(f"  Validated States:")
        for state, info in summary['states'].items():
            print(f"    State {state}: {info['label']} (confidence={info['confidence']})")

# Print final summary
print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)
print(f"\nSuccessfully analyzed {len(results_summary)} animals\n")

# Create summary table
print(f"{'Animal':<10} {'Genotype':<15} {'Trials':<8} {'LL':<10} {'State Labels'}")
print("-" * 90)
for summary in results_summary:
    state_labels = " | ".join([
        f"{summary['states'][s]['label'][:20]}"
        for s in range(3)
    ])
    print(f"{summary['animal_id']:<10} {summary['genotype']:<15} {summary['n_trials']:<8} "
          f"{summary['log_likelihood']:<10.2f} {state_labels}")

print("\n" + "="*70)
print("Output directories:")
print("  Results: /home/user/GLMHMM/results/phase1_non_reversal/figures/")
print("  Figures: <animal_id>_cohortW/*.png")
print("="*70)
