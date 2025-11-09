"""
Complete Analysis Runner - Generate All Figures

This script runs the complete GLM-HMM analysis pipeline and generates
all publication-quality figures organized by category.

Author: Claude (Anthropic)
Date: 2025-11-08
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set paths
BASE_DIR = Path('/home/user/GLMHMM')
DATA_DIR = BASE_DIR
FIG_DIR = BASE_DIR / 'figures'

# Import custom modules
import sys
sys.path.insert(0, str(BASE_DIR))

from glmhmm_ashwood import GLMHMM
from glmhmm_utils import (
    load_and_preprocess_session_data,
    create_design_matrix,
    compute_psychometric_curves,
    plot_psychometric_curves,
    plot_glmhmm_summary,
    cross_validate_n_states
)
from genotype_sex_comparisons import (
    plot_genotype_sex_learning_curves,
    plot_state_occupancy_by_groups,
    statistical_group_comparisons
)
from advanced_analysis import (
    compute_latency_variability_metrics,
    identify_vte_states,
    analyze_lapse_discreteness,
    test_deliberation_learning_hypothesis,
    analyze_state_transitions_at_reversals,
    compute_state_dwell_times,
    create_learning_efficiency_score,
    create_flexibility_index
)
from advanced_visualization import (
    plot_latency_variability_over_learning,
    plot_state_classification_dual_process,
    plot_lapse_discreteness_analysis,
    plot_deliberation_learning_correlation,
    plot_state_transitions_at_reversals,
    plot_state_dwell_times
)

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11

print("="*70)
print("COMPLETE GLM-HMM ANALYSIS PIPELINE")
print("="*70)
print(f"\nBase directory: {BASE_DIR}")
print(f"Figure directory: {FIG_DIR}")
print(f"\nAnalysis will generate 30+ figures organized by category")
print("="*70)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[STEP 1] Loading data...")

# Use new 11.08 files with position data
w_file = DATA_DIR / 'W LD Data 11.08 All_processed.csv'
f_file = DATA_DIR / 'F LD Data 11.08 All_processed.csv'

if w_file.exists():
    print(f"✓ Loading W cohort: {w_file.name}")
    w_trials = load_and_preprocess_session_data(str(w_file))
    print(f"  Loaded {len(w_trials)} trials from {w_trials['animal_id'].nunique()} animals")
    has_w_data = True
else:
    print(f"✗ W cohort file not found: {w_file}")
    has_w_data = False

if f_file.exists():
    print(f"✓ Loading F cohort: {f_file.name}")
    f_trials = load_and_preprocess_session_data(str(f_file))
    print(f"  Loaded {len(f_trials)} trials from {f_trials['animal_id'].nunique()} animals")
    has_f_data = True
else:
    print(f"✗ F cohort file not found: {f_file}")
    has_f_data = False

# Use W cohort as primary
if has_w_data:
    trial_df = w_trials
    cohort_name = 'W'
elif has_f_data:
    trial_df = f_trials
    cohort_name = 'F'
else:
    raise FileNotFoundError("No data files found!")

print(f"\nUsing {cohort_name} cohort for analysis")

# Check for position data
has_position = trial_df['position'].notna().any()
print(f"Position data available: {has_position}")

# ============================================================================
# STEP 2: Select Animal for Detailed Analysis
# ============================================================================
print("\n[STEP 2] Selecting animal for detailed analysis...")

trials_per_animal = trial_df.groupby('animal_id').size()
test_animal = trials_per_animal.idxmax()

print(f"Selected animal: {test_animal}")
print(f"  Trials: {trials_per_animal[test_animal]}")
print(f"  Genotype: {trial_df[trial_df['animal_id']==test_animal]['genotype'].iloc[0]}")

# ============================================================================
# STEP 3: Create Design Matrix and Fit Model
# ============================================================================
print("\n[STEP 3] Creating design matrix and fitting GLM-HMM...")

X, y, feature_names, metadata, animal_data = create_design_matrix(
    trial_df,
    animal_id=test_animal,
    include_position=has_position,
    include_session_progression=True
)

print(f"Design matrix: {X.shape}")
print(f"Features: {feature_names}")

# Fit GLM-HMM
n_states = 3
model = GLMHMM(
    n_states=n_states,
    feature_names=feature_names,
    normalize_features=True,
    regularization_strength=1.0,
    random_state=42
)

print(f"\nFitting {n_states}-state GLM-HMM...")
model.fit(X, y, n_iter=100, tolerance=1e-4, verbose=False)

print(f"✓ Model converged in {len(model.log_likelihood_history)} iterations")
print(f"  Final log-likelihood: {model.log_likelihood_history[-1]:.2f}")

# Create filtered trial_df for hypothesis testing (only test animal)
animal_trial_df = trial_df[trial_df['animal_id'] == test_animal].copy()
animal_trial_df = animal_trial_df.reset_index(drop=True)  # Reset index to match model states

# ============================================================================
# STEP 4: Generate Core Analysis Figures
# ============================================================================
print("\n[STEP 4] Generating core analysis figures...")

save_dir = FIG_DIR / 'core_analysis'

# Figure 1: Comprehensive GLM-HMM summary
print("  Creating comprehensive summary figure...")
fig = plot_glmhmm_summary(model, X, y, metadata, feature_names=feature_names, figsize=(18, 14))
fig.suptitle(f'GLM-HMM Analysis: {test_animal} ({cohort_name} Cohort)', fontsize=16, y=0.995)
fig.savefig(save_dir / f'01_glmhmm_summary_{test_animal}.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"    ✓ Saved: 01_glmhmm_summary_{test_animal}.png")

# Figure 2: Psychometric curves
print("  Creating psychometric curves...")
curves = compute_psychometric_curves(model, X, y, metadata, n_bins=7)
fig, ax = plt.subplots(figsize=(10, 7))
plot_psychometric_curves(curves, model, ax=ax,
                         title=f"State-Specific Psychometric Curves - {test_animal}")
fig.tight_layout()
fig.savefig(save_dir / f'02_psychometric_curves_{test_animal}.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"    ✓ Saved: 02_psychometric_curves_{test_animal}.png")

# Figure 3: State summary table
print("  Creating state summary table...")
state_summary = model.get_state_summary(y=metadata['correct'], metadata={'latency': metadata['latency']})
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
table_data = state_summary[['state', 'n_trials', 'proportion', 'accuracy', 'intercept']].round(3)
table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
# Color header
for i in range(len(table_data.columns)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')
plt.title(f'State Summary - {test_animal}', fontsize=14, fontweight='bold', pad=20)
fig.savefig(save_dir / f'03_state_summary_table_{test_animal}.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"    ✓ Saved: 03_state_summary_table_{test_animal}.png")

print(f"\n✓ Core analysis figures saved to: {save_dir}")

# ============================================================================
# STEP 5: Hypothesis 1 - Discrete Lapse Analysis
# ============================================================================
print("\n[STEP 5] Testing Hypothesis 1: Discrete Lapse States...")

save_dir = FIG_DIR / 'hypothesis1_discrete_lapses'

# Analyze lapse discreteness
lapse_results = analyze_lapse_discreteness(model, metadata, min_lapse_run=3)

print(f"\nLapse Analysis Results:")
print(f"  Lapse state: {lapse_results['lapse_state'] + 1}")
print(f"  Lapse probability: {lapse_results['lapse_probability']:.2%}")
print(f"  Run length ratio: {lapse_results['run_length_ratio']:.2f}x")
print(f"  Autocorrelation: {lapse_results['autocorrelation_lag1']:.3f}")
print(f"  Interpretation: {lapse_results['interpretation']}")

# Generate comprehensive lapse figure
fig = plot_lapse_discreteness_analysis(lapse_results, model, metadata, figsize=(18, 12))
fig.savefig(save_dir / f'01_discrete_lapse_comprehensive_{test_animal}.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"\n✓ Saved: 01_discrete_lapse_comprehensive_{test_animal}.png")

print(f"✓ Hypothesis 1 figures saved to: {save_dir}")

# ============================================================================
# STEP 6: Hypothesis 2 - Dual-Process VTE Analysis
# ============================================================================
print("\n[STEP 6] Testing Hypothesis 2: Dual-Process Decision Making...")

save_dir = FIG_DIR / 'hypothesis2_dual_process'

# Compute latency variability metrics
print("  Computing latency variability metrics...")
latency_metrics = compute_latency_variability_metrics(animal_trial_df, window_size=20)

# Identify VTE states
print("  Identifying deliberative vs procedural states...")
state_classification = identify_vte_states(model, metadata, latency_metrics, cv_threshold=0.5)

print(f"\nState Classifications:")
for idx, row in state_classification.iterrows():
    print(f"  State {row['state']+1}: {row['state_type']} ({row['process_type']})")
    print(f"    Accuracy: {row['accuracy']:.2%}, CV: {row['cv_latency']:.2f}")

# Test deliberation-learning hypothesis
print("\n  Testing deliberation → learning correlation...")
delib_results = test_deliberation_learning_hypothesis(animal_trial_df, model, metadata, early_phase_sessions=5)

corr = delib_results.attrs.get('correlation_performance', 0)
p_val = delib_results.attrs.get('p_value_performance', 1)
print(f"  Early deliberation → Final performance: r={corr:.3f}, p={p_val:.4f}")

if p_val < 0.05:
    print(f"  ✓ HYPOTHESIS SUPPORTED: Early deliberation predicts better learning!")
else:
    print(f"  ✗ Not significant")

# Generate figures
print("\n  Generating dual-process figures...")

# Figure 1: Latency variability over learning
fig = plot_latency_variability_over_learning(latency_metrics, figsize=(14, 8))
fig.savefig(save_dir / '01_latency_variability_over_learning.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"    ✓ Saved: 01_latency_variability_over_learning.png")

# Figure 2: State classification
fig = plot_state_classification_dual_process(state_classification, figsize=(14, 6))
fig.savefig(save_dir / '02_state_classification_dual_process.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"    ✓ Saved: 02_state_classification_dual_process.png")

# Figure 3: Deliberation-learning correlation
fig = plot_deliberation_learning_correlation(delib_results, figsize=(14, 10))
fig.savefig(save_dir / '03_deliberation_learning_correlation.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"    ✓ Saved: 03_deliberation_learning_correlation.png")

print(f"\n✓ Hypothesis 2 figures saved to: {save_dir}")

# ============================================================================
# STEP 7: State Transition Analysis
# ============================================================================
print("\n[STEP 7] Analyzing state transitions at reversals...")

save_dir = FIG_DIR / 'additional_analyses'

reversal_df = analyze_state_transitions_at_reversals(animal_trial_df, model, metadata,
                                                     window_before=20, window_after=20)

if len(reversal_df) > 0:
    fig = plot_state_transitions_at_reversals(reversal_df, figsize=(14, 6))
    if fig is not None:
        fig.savefig(save_dir / '01_state_transitions_reversals.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: 01_state_transitions_reversals.png")
else:
    print("  ✗ No reversal data available")

# ============================================================================
# STEP 8: State Dwell Time Analysis
# ============================================================================
print("\n[STEP 8] Computing state dwell times...")

dwell_df, dwell_times_raw = compute_state_dwell_times(model)

print("\nDwell Time Summary:")
for idx, row in dwell_df.iterrows():
    print(f"  State {row['state']+1}: Mean={row['mean_dwell']:.1f} trials, "
          f"Median={row['median_dwell']:.1f}, n_bouts={row['n_bouts']}")

fig = plot_state_dwell_times(dwell_df, dwell_times_raw, figsize=(12, 8))
fig.savefig(save_dir / '02_state_dwell_times.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ Saved: 02_state_dwell_times.png")

# ============================================================================
# STEP 9: Learning Efficiency Scores
# ============================================================================
print("\n[STEP 9] Computing learning efficiency scores...")

efficiency_df = create_learning_efficiency_score(animal_trial_df, model, metadata)

print("\nLearning Efficiency Scores:")
print(efficiency_df[['animal_id', 'genotype', 'learning_efficiency_score',
                     'final_accuracy']].to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
colors = {'+': 'blue', '-': 'red', np.nan: 'gray'}
for geno in efficiency_df['genotype'].unique():
    if pd.isna(geno):
        continue
    geno_data = efficiency_df[efficiency_df['genotype'] == geno]
    ax.scatter(geno_data.index, geno_data['learning_efficiency_score'],
              s=100, alpha=0.7, label=f'Genotype {geno}',
              color=colors.get(geno, 'gray'), edgecolors='black')

ax.set_xlabel('Animal Index', fontsize=12)
ax.set_ylabel('Learning Efficiency Score', fontsize=12)
ax.set_title('Learning Efficiency by Animal', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(save_dir / '03_learning_efficiency_scores.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ Saved: 03_learning_efficiency_scores.png")

# ============================================================================
# STEP 10: Multi-Animal Comparison
# ============================================================================
print("\n[STEP 10] Fitting GLM-HMM to multiple animals...")

save_dir = FIG_DIR / 'individual_animals'

n_animals_to_analyze = min(5, trial_df['animal_id'].nunique())
animals_to_analyze = trials_per_animal.nlargest(n_animals_to_analyze).index.tolist()

multi_results = {}

for animal in animals_to_analyze:
    print(f"  Processing {animal}...")

    X_i, y_i, fn_i, meta_i, data_i = create_design_matrix(
        trial_df, animal_id=animal, include_position=has_position
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
        'genotype': data_i['genotype'].iloc[0],
        'sex': data_i['sex'].iloc[0]  # Add sex data
    }

    # Generate individual summary
    fig = plot_glmhmm_summary(model_i, X_i, y_i, meta_i, feature_names=fn_i, figsize=(16, 12))
    fig.suptitle(f'GLM-HMM Analysis: {animal}', fontsize=16, y=0.995)
    fig.savefig(save_dir / f'summary_{animal}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

print(f"\n✓ Individual animal figures saved to: {save_dir}")

# ============================================================================
# STEP 11: Genotype Comparison
# ============================================================================
print("\n[STEP 11] Comparing genotypes...")

save_dir = FIG_DIR / 'genotype_comparison'

# Compile state statistics
state_comparison = []
for animal, results in multi_results.items():
    model_i = results['model']
    meta_i = results['metadata']

    for state in range(3):
        mask = model_i.most_likely_states == state
        if mask.sum() > 0:
            state_comparison.append({
                'animal': animal,
                'genotype': results['genotype'],
                'state': state + 1,
                'occupancy': mask.sum() / len(mask),
                'accuracy': meta_i['correct'][mask].mean()
            })

state_comp_df = pd.DataFrame(state_comparison)

# Plot genotype comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# State occupancy by genotype
ax = axes[0]
geno_occ = state_comp_df.groupby(['genotype', 'state'])['occupancy'].mean().unstack()
geno_occ.plot(kind='bar', ax=ax, alpha=0.7)
ax.set_xlabel('Genotype', fontsize=12)
ax.set_ylabel('Mean State Occupancy', fontsize=12)
ax.set_title('State Usage by Genotype', fontsize=14, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='State', labels=[f'State {i}' for i in geno_occ.columns])
ax.grid(axis='y', alpha=0.3)

# Accuracy by genotype and state
ax = axes[1]
geno_acc = state_comp_df.groupby(['genotype', 'state'])['accuracy'].mean().unstack()
geno_acc.plot(kind='bar', ax=ax, alpha=0.7)
ax.set_xlabel('Genotype', fontsize=12)
ax.set_ylabel('Mean Accuracy', fontsize=12)
ax.set_title('Performance by Genotype and State', fontsize=14, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.axhline(0.8, color='green', linestyle='--', alpha=0.3)
ax.legend(title='State', labels=[f'State {i}' for i in geno_acc.columns])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(save_dir / '01_genotype_state_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ Saved: 01_genotype_state_comparison.png")

# NEW: Enhanced genotype × sex comparisons
print("\n  Generating enhanced genotype and sex comparison figures...")

# Figure 2: Learning curves by genotype and sex
try:
    fig = plot_genotype_sex_learning_curves(trial_df, figsize=(16, 10))
    fig.savefig(save_dir / '02_genotype_sex_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: 02_genotype_sex_learning_curves.png")
except Exception as e:
    print(f"  ✗ Error generating learning curves: {e}")

# Figure 3: State occupancy by genotype and sex
try:
    fig = plot_state_occupancy_by_groups(multi_results, figsize=(14, 10))
    fig.savefig(save_dir / '03_state_occupancy_genotype_sex.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: 03_state_occupancy_genotype_sex.png")
except Exception as e:
    print(f"  ✗ Error generating state occupancy: {e}")

# Statistical comparisons
print("\n  Running statistical tests for group differences...")
try:
    stats_df = statistical_group_comparisons(trial_df, multi_results)
    if len(stats_df) > 0:
        print("\nStatistical Comparisons:")
        print(stats_df[['comparison', 'factor', 'group1', 'group2',
                       'mean1', 'mean2', 'p_value', 'significant']].to_string(index=False))
        # Save to CSV
        stats_df.to_csv(save_dir / 'statistical_comparisons.csv', index=False)
        print(f"\n  ✓ Saved: statistical_comparisons.csv")
except Exception as e:
    print(f"  ✗ Error in statistical tests: {e}")

print(f"\n✓ Genotype/sex comparison figures saved to: {save_dir}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)

# Count figures
total_figs = 0
for subdir in FIG_DIR.iterdir():
    if subdir.is_dir():
        n_figs = len(list(subdir.glob('*.png')))
        if n_figs > 0:
            print(f"\n{subdir.name}:")
            print(f"  {n_figs} figures generated")
            total_figs += n_figs

print(f"\n{'='*70}")
print(f"TOTAL: {total_figs} figures generated")
print(f"Saved to: {FIG_DIR}")
print(f"{'='*70}")

# Print key results
print("\n" + "="*70)
print("KEY RESULTS SUMMARY")
print("="*70)

print(f"\nHypothesis 1 (Discrete Lapses):")
print(f"  {lapse_results['interpretation']}")
print(f"  Run length ratio: {lapse_results['run_length_ratio']:.2f}x")
print(f"  Autocorrelation: {lapse_results['autocorrelation_lag1']:.3f}")

print(f"\nHypothesis 2 (Dual-Process VTE):")
if 'correlation_performance' in delib_results.attrs:
    corr = delib_results.attrs['correlation_performance']
    p_val = delib_results.attrs['p_value_performance']
    print(f"  Early deliberation → Final performance:")
    print(f"    Correlation: r={corr:.3f}, p={p_val:.4f}")
    if p_val < 0.05:
        print(f"    ✓ SIGNIFICANT - Hypothesis supported!")
    else:
        print(f"    ✗ Not significant")
else:
    print(f"  Insufficient data for correlation analysis")

print(f"\nState Classifications:")
for idx, row in state_classification.iterrows():
    print(f"  State {row['state']+1}: {row['state_type']}")

print("\n" + "="*70)
print("Analysis script completed successfully!")
print("="*70)
