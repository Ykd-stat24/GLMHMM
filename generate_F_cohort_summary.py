"""
Generate F Cohort Summaries from Existing Results
=================================================

Creates group-level summaries for F cohort since the main script
errored before completing F cohort analysis.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from run_phase1_analysis import create_cohort_summary
from glmhmm_utils import load_and_preprocess_session_data

# Load F cohort data to get results
F_DATA = '/home/user/GLMHMM/F LD Data 11.08 All_processed.csv'
OUTPUT_DIR = Path('/home/user/GLMHMM/results/phase1_non_reversal')

print("Loading F cohort trial data...")
trial_df = load_and_preprocess_session_data(F_DATA)

# Get all F cohort animals that were successfully analyzed
figures_dir = OUTPUT_DIR / 'figures'
f_animals = [d.name.split('_cohort')[0] for d in figures_dir.iterdir()
             if d.is_dir() and 'cohortF' in d.name]

print(f"Found {len(f_animals)} F cohort animals with results")
print(f"Animals: {sorted(f_animals)}")

# We need to reconstruct results from the saved figures
# Since we don't have the model objects saved, we'll create a simplified summary
# by reading the validation plots and extracting information

print("\nCreating F cohort summary from available data...")

# Count state labels from figure directories
from collections import defaultdict
state_labels_by_animal = {}

for animal_id in f_animals:
    animal_dir = figures_dir / f'{animal_id}_cohortF'
    # We can infer some info from filenames, but ideally we need to re-run
    # For now, create a basic summary
    state_labels_by_animal[animal_id] = {
        'analyzed': True,
        'genotype': trial_df[trial_df['animal_id'] == animal_id]['genotype'].iloc[0] if len(trial_df[trial_df['animal_id'] == animal_id]) > 0 else 'Unknown'
    }

# Create basic summary
summary_text = f"Cohort F - Phase 1 Summary\n"
summary_text += f"{'='*50}\n"
summary_text += f"Animals analyzed: {len(f_animals)}\n"
summary_text += f"States per animal: 3\n\n"

# Count genotypes
genotypes = [info['genotype'] for info in state_labels_by_animal.values()]
from collections import Counter
geno_counts = Counter(genotypes)

summary_text += f"Genotype Distribution:\n"
for geno, count in sorted(geno_counts.items()):
    summary_text += f"  {geno}: {count} animals\n"

summary_text += f"\nNote: Full state label distribution requires re-running analysis\n"
summary_text += f"Individual animal results available in figures/ directory\n"

# Save summary
summary_file = OUTPUT_DIR / 'cohort_F_phase1_summary.txt'
with open(summary_file, 'w') as f:
    f.write(summary_text)

print(f"\n✓ Saved F cohort summary to: {summary_file.name}")
print(summary_text)

print("\n" + "="*70)
print("F COHORT SUMMARY COMPLETE")
print("="*70)
print("\nTo generate full genotype comparisons with transition matrices,")
print("we need to re-run the analysis on F cohort animals.")
print("This would create:")
print("  - F cohort transition summary")
print("  - F cohort genotype comparison figures (4 genotypes: +/+, +/-, +, -)")
print("  - Cross-cohort comparison analyses")
