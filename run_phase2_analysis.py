"""
Phase 2 Analysis: Reversal Tasks
=================================

Analyzes reversal tasks where stimulus varies dynamically:
- A_Mouse LD 1 choice reversal v3 (W cohort - rolling 7/8 criterion)
- A_Mouse LD 1 Reversal 9 (F cohort - single reversal)

Key differences from Phase 1:
1. INCLUDES corrected stimulus feature (from reversal detection)
2. Analyzes state transitions around reversal points
3. Generates psychometric curves (valid here!)
4. Measures adaptation speed post-reversal
5. Runs separately for W and F cohorts

Features used (8 total):
1. stimulus_correct_side (CORRECTED for reversals)
2. bias (constant)
3. prev_choice
4. wsls (win-stay/lose-shift)
5. session_progression
6. recent_side_bias
7. task_stage
8. cumulative_experience
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from glmhmm_utils import (
    load_mouse_data,
    create_design_matrix,
    prepare_glmhmm_inputs
)
from glmhmm_ashwood import GLMHMM
from reversal_detection import (
    add_reversal_info_to_trials,
    compute_reversal_adaptation_metrics
)
from state_validation import (
    compute_performance_trajectory,
    compute_comprehensive_state_metrics,
    validate_state_labels
)

# Set random seed for reproducibility
np.random.seed(42)

# Create output directories
OUTPUT_DIR = Path('/home/user/GLMHMM/results/phase2_reversal')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)


def create_phase2_design_matrix(trial_df):
    """
    Create design matrix for Phase 2 (reversal tasks).

    INCLUDES stimulus feature (corrected for reversals).

    Features (8 total):
    1. stimulus_correct_side (CORRECTED)
    2. bias (constant = 1.0)
    3. prev_choice (-1=left, +1=right)
    4. wsls (win-stay/lose-shift)
    5. session_progression (0 to 1)
    6. recent_side_bias (proportion right in last 10)
    7. task_stage (training progression)
    8. cumulative_experience (overall trials)

    Returns:
        design_matrix: (n_trials, 8) array
        feature_names: list of feature names
    """
    # Use standard design matrix creation
    design_matrix, feature_names = create_design_matrix(trial_df)

    # Verify stimulus has variance
    stimulus_idx = feature_names.index('stimulus_correct_side')
    stimulus_var = np.var(design_matrix[:, stimulus_idx])

    print(f"\nPhase 2 Design Matrix:")
    print(f"  Shape: {design_matrix.shape}")
    print(f"  Features: {feature_names}")
    print(f"  Stimulus variance: {stimulus_var:.4f}")

    if stimulus_var < 0.01:
        print(f"  WARNING: Stimulus has very low variance!")

    return design_matrix, feature_names


def run_glmhmm_phase2(trial_df, metadata, n_states=3, n_iterations=200,
                      l2_penalty=1.0, random_seed=42):
    """
    Run GLM-HMM on Phase 2 data (reversal tasks).

    Args:
        trial_df: Trial data with corrected stimulus
        metadata: Metadata dictionary
        n_states: Number of hidden states
        n_iterations: Number of EM iterations
        l2_penalty: L2 regularization strength
        random_seed: Random seed

    Returns:
        model: Fitted GLMHMM model
        design_matrix: Design matrix used
        feature_names: Feature names
    """
    print(f"\n{'='*70}")
    print(f"Running GLM-HMM: Phase 2 (Reversal)")
    print(f"{'='*70}")
    print(f"Animal: {metadata['animal_id']}")
    print(f"Genotype: {metadata['genotype']}")
    print(f"Cohort: {metadata['cohort']}")
    print(f"Tasks: Reversal (LD reversal v3 or LD Reversal 9)")
    print(f"Trials: {len(trial_df)}")
    print(f"Reversal sessions: {trial_df['reversal_session'].sum()}")
    print(f"States: {n_states}")
    print(f"L2 penalty: {l2_penalty}")

    # Create Phase 2 design matrix (WITH corrected stimulus)
    design_matrix, feature_names = create_phase2_design_matrix(trial_df)

    # Prepare inputs
    choices_list, inputs_list, session_ids = prepare_glmhmm_inputs(
        trial_df, design_matrix
    )

    # Initialize and fit model
    n_features = design_matrix.shape[1]
    model = GLMHMM(
        n_states=n_states,
        n_features=n_features,
        observations="bernoulli",
        l2_penalty=l2_penalty,
        random_seed=random_seed
    )

    print(f"\nFitting model...")
    model.fit(choices_list, inputs_list, num_iters=n_iterations)

    print(f"✓ Model fitted successfully")
    print(f"  Final log-likelihood: {model.log_likelihood:.2f}")

    return model, design_matrix, feature_names


def plot_phase2_weights(model, feature_names, metadata, save_path):
    """
    Plot GLM weights for Phase 2 analysis.

    Highlights stimulus weight (most important for reversal learning).
    """
    n_states = model.n_states
    weights = model.weights  # (n_states, n_features)

    fig, axes = plt.subplots(1, n_states, figsize=(5*n_states, 6))
    if n_states == 1:
        axes = [axes]

    stimulus_idx = feature_names.index('stimulus_correct_side')

    for state in range(n_states):
        ax = axes[state]

        # Plot weights
        y_pos = np.arange(len(feature_names))
        colors = []
        for i, w in enumerate(weights[state]):
            if i == stimulus_idx:
                colors.append('green' if w > 0 else 'orange')  # Highlight stimulus
            else:
                colors.append('red' if w < 0 else 'blue')

        ax.barh(y_pos, weights[state], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('GLM Weight')
        ax.set_title(f'State {state}\nStimulus weight: {weights[state, stimulus_idx]:.3f}')
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle(f'Phase 2 GLM Weights (Reversal)\n{metadata["animal_id"]} - {metadata["genotype"]}',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved GLM weights: {save_path.name}")


def plot_psychometric_curves(trial_df, model, feature_names, metadata, save_path):
    """
    Plot state-specific psychometric curves.

    Shows P(choose right) vs stimulus (correct side).
    Only valid for Phase 2 where stimulus varies!
    """
    n_states = model.n_states

    # Get stimulus index
    stimulus_idx = feature_names.index('stimulus_correct_side')

    # Compute state probabilities
    design_matrix, _ = create_phase2_design_matrix(trial_df)
    posterior_probs = model.expected_states(
        [trial_df['choice_encoding'].values],
        [design_matrix]
    )[0]

    # Assign each trial to most likely state
    most_likely_states = np.argmax(posterior_probs, axis=1)

    fig, axes = plt.subplots(1, n_states, figsize=(5*n_states, 5))
    if n_states == 1:
        axes = [axes]

    for state in range(n_states):
        ax = axes[state]

        # Filter trials for this state
        state_mask = most_likely_states == state
        state_trials = trial_df[state_mask].copy()

        if len(state_trials) < 10:
            ax.text(0.5, 0.5, f'State {state}\n(insufficient data)',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Bin by stimulus value
        stimulus_values = state_trials['stimulus'].values
        choice_right = (state_trials['choice_encoding'].values == 1).astype(int)

        # Create bins
        bins = [-1.5, -0.5, 0.5, 1.5]  # Left (-1), Right (+1)
        bin_centers = [-1, 1]
        p_right = []
        n_trials = []

        for i in range(len(bins) - 1):
            mask = (stimulus_values >= bins[i]) & (stimulus_values < bins[i+1])
            if mask.sum() > 0:
                p_right.append(choice_right[mask].mean())
                n_trials.append(mask.sum())
            else:
                p_right.append(np.nan)
                n_trials.append(0)

        # Plot
        ax.plot(bin_centers, p_right, 'o-', markersize=10, linewidth=2, color='steelblue')

        # Add trial counts
        for x, y, n in zip(bin_centers, p_right, n_trials):
            if not np.isnan(y):
                ax.text(x, y + 0.05, f'n={n}', ha='center', fontsize=9)

        # Reference lines
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.axvline(0, color='gray', linestyle='--', alpha=0.3)

        ax.set_xlabel('Stimulus (Correct Side)\n-1=Left, +1=Right')
        ax.set_ylabel('P(Choose Right)')
        ax.set_ylim(0, 1)
        ax.set_xticks([-1, 1])
        ax.set_xticklabels(['Left\nCorrect', 'Right\nCorrect'])
        ax.set_title(f'State {state}\nStimulus Following')
        ax.grid(alpha=0.3)
        ax.legend()

    plt.suptitle(f'Psychometric Curves: {metadata["animal_id"]} - {metadata["genotype"]}',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved psychometric curves: {save_path.name}")


def plot_reversal_transitions(trial_df, model, metadata, save_path):
    """
    Plot state transitions around reversal points.

    Shows how state occupancy changes before/during/after reversal.
    """
    # Filter to reversal sessions only
    reversal_sessions = trial_df[trial_df['reversal_session']].copy()

    if len(reversal_sessions) == 0:
        print("No reversal sessions found, skipping reversal transition plot")
        return

    # Compute state probabilities
    design_matrix, _ = create_phase2_design_matrix(reversal_sessions)
    posterior_probs = model.expected_states(
        [reversal_sessions['choice_encoding'].values],
        [design_matrix]
    )[0]

    # Find reversal trial indices
    reversal_trials = reversal_sessions[reversal_sessions['trials_since_reversal'] == 0].index.tolist()

    if len(reversal_trials) == 0:
        print("No reversal trials found, skipping reversal transition plot")
        return

    fig, axes = plt.subplots(len(reversal_trials), 1,
                             figsize=(12, 4*len(reversal_trials)),
                             squeeze=False)

    for i, rev_idx in enumerate(reversal_trials[:5]):  # Plot up to 5 reversals
        ax = axes[i, 0]

        # Get window around reversal
        window = 20
        start_idx = max(0, rev_idx - window)
        end_idx = min(len(reversal_sessions), rev_idx + window)

        trial_nums = np.arange(start_idx, end_idx)
        window_probs = posterior_probs[start_idx:end_idx]

        # Plot state probabilities
        for state in range(model.n_states):
            ax.plot(trial_nums, window_probs[:, state], label=f'State {state}',
                   linewidth=2, alpha=0.7)

        # Mark reversal point
        ax.axvline(rev_idx, color='red', linestyle='--', linewidth=2,
                  label='Reversal', alpha=0.7)

        ax.set_xlabel('Trial Number')
        ax.set_ylabel('State Probability')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_title(f'Reversal {i+1}')

    plt.suptitle(f'State Transitions at Reversals\n{metadata["animal_id"]} - {metadata["genotype"]}',
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved reversal transitions: {save_path.name}")


def plot_adaptation_metrics(trial_df, metadata, save_path):
    """
    Plot reversal adaptation metrics.

    Shows:
    1. Trials to criterion after reversal
    2. Accuracy trajectory post-reversal
    3. Perseveration errors
    """
    reversal_sessions = trial_df[trial_df['reversal_session']].copy()

    if len(reversal_sessions) == 0:
        print("No reversal sessions found, skipping adaptation metrics plot")
        return

    # Compute adaptation metrics
    adaptation_metrics = compute_reversal_adaptation_metrics(reversal_sessions)

    if len(adaptation_metrics) == 0:
        print("Could not compute adaptation metrics")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Trials to criterion distribution
    ax = axes[0, 0]
    trials_to_crit = [m['trials_to_criterion'] for m in adaptation_metrics
                     if not np.isnan(m['trials_to_criterion'])]
    if len(trials_to_crit) > 0:
        ax.hist(trials_to_crit, bins=10, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(trials_to_crit), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(trials_to_crit):.1f}')
        ax.set_xlabel('Trials to Criterion')
        ax.set_ylabel('Count')
        ax.set_title('Reversal Adaptation Speed')
        ax.legend()
        ax.grid(alpha=0.3)

    # 2. Accuracy trajectory post-reversal
    ax = axes[0, 1]
    max_trials = 20
    acc_by_trial = {t: [] for t in range(max_trials)}

    for metric in adaptation_metrics:
        post_acc = metric['post_reversal_accuracy']
        for t, acc in enumerate(post_acc[:max_trials]):
            acc_by_trial[t].append(acc)

    trials = []
    mean_acc = []
    sem_acc = []
    for t in range(max_trials):
        if len(acc_by_trial[t]) > 0:
            trials.append(t)
            mean_acc.append(np.mean(acc_by_trial[t]))
            sem_acc.append(np.std(acc_by_trial[t]) / np.sqrt(len(acc_by_trial[t])))

    if len(trials) > 0:
        ax.plot(trials, mean_acc, 'o-', linewidth=2, markersize=6, color='steelblue')
        ax.fill_between(trials,
                       np.array(mean_acc) - np.array(sem_acc),
                       np.array(mean_acc) + np.array(sem_acc),
                       alpha=0.3, color='steelblue')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('Trials Since Reversal')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.set_title('Post-Reversal Learning Curve')
        ax.legend()
        ax.grid(alpha=0.3)

    # 3. Perseveration errors
    ax = axes[1, 0]
    persev_errors = [m['perseveration_errors'] for m in adaptation_metrics]
    if len(perseveration_errors) > 0:
        ax.hist(persev_errors, bins=10, alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(np.mean(persev_errors), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(persev_errors):.1f}')
        ax.set_xlabel('Perseveration Errors (first 5 trials)')
        ax.set_ylabel('Count')
        ax.set_title('Perseveration After Reversal')
        ax.legend()
        ax.grid(alpha=0.3)

    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = "Reversal Adaptation Summary\n"
    summary_text += "=" * 30 + "\n\n"
    summary_text += f"Total reversals: {len(adaptation_metrics)}\n\n"

    if len(trials_to_crit) > 0:
        summary_text += f"Trials to criterion:\n"
        summary_text += f"  Mean: {np.mean(trials_to_crit):.1f}\n"
        summary_text += f"  Median: {np.median(trials_to_crit):.1f}\n"
        summary_text += f"  Range: {np.min(trials_to_crit):.0f}-{np.max(trials_to_crit):.0f}\n\n"

    if len(persev_errors) > 0:
        summary_text += f"Perseveration errors:\n"
        summary_text += f"  Mean: {np.mean(persev_errors):.1f}\n"
        summary_text += f"  Median: {np.median(persev_errors):.1f}\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'Reversal Adaptation Metrics\n{metadata["animal_id"]} - {metadata["genotype"]}',
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved adaptation metrics: {save_path.name}")


def analyze_single_animal_phase2(data_file, ldr_file, animal_id, cohort, n_states=3):
    """
    Run complete Phase 2 analysis for a single animal.

    Args:
        data_file: Path to data file
        ldr_file: Path to LDR criterion file
        animal_id: Animal ID
        cohort: 'W' or 'F'
        n_states: Number of states

    Returns:
        results: Dictionary with model, metrics, and labels
    """
    print(f"\n{'#'*70}")
    print(f"# Analyzing Animal: {animal_id} (Cohort {cohort})")
    print(f"{'#'*70}")

    # Load data for this animal
    trial_df, metadata = load_mouse_data(data_file)
    trial_df_animal = trial_df[trial_df['animal_id'] == animal_id].copy()

    if len(trial_df_animal) == 0:
        print(f"WARNING: No trials found for {animal_id}")
        return None

    # Add reversal information
    print(f"Adding reversal information from LDR file...")
    trial_df_animal = add_reversal_info_to_trials(trial_df_animal, ldr_file, cohort=cohort)

    # Filter to reversal tasks only
    reversal_tasks = ['LD_reversal', 'PD_reversal']
    trial_df_rev = trial_df_animal[trial_df_animal['task'].isin(reversal_tasks)].copy()

    if len(trial_df_rev) == 0:
        print(f"WARNING: No reversal trials found for {animal_id}")
        return None

    # Update stimulus with corrected values
    if 'stimulus_corrected' in trial_df_rev.columns:
        trial_df_rev['stimulus'] = trial_df_rev['stimulus_corrected']

    print(f"Reversal trials: {len(trial_df_rev)}")
    print(f"Reversal sessions: {trial_df_rev['reversal_session'].sum()}")
    print(f"Total reversals: {trial_df_rev['n_reversals_in_session'].sum()}")

    # Run GLM-HMM
    model, design_matrix, feature_names = run_glmhmm_phase2(
        trial_df_rev, metadata, n_states=n_states
    )

    # State validation
    print(f"\nValidating states...")
    state_metrics = compute_comprehensive_state_metrics(trial_df_rev, model, metadata)
    trajectory_df = compute_performance_trajectory(trial_df_rev, model)
    validated_labels = validate_state_labels(state_metrics, trajectory_df)

    # Print validation results
    print(f"\nValidated State Labels:")
    for state in range(n_states):
        label, confidence, evidence = validated_labels[state]
        print(f"\n  State {state}: {label} (confidence={confidence})")
        for key, value in evidence.items():
            print(f"    - {key}: {value}")

    # Create animal-specific output directory
    animal_dir = FIGURES_DIR / f'{animal_id}_cohort{cohort}'
    animal_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_phase2_weights(model, feature_names, metadata,
                       animal_dir / f'{animal_id}_weights.png')
    plot_psychometric_curves(trial_df_rev, model, feature_names, metadata,
                            animal_dir / f'{animal_id}_psychometric.png')
    plot_reversal_transitions(trial_df_rev, model, metadata,
                             animal_dir / f'{animal_id}_reversal_transitions.png')
    plot_adaptation_metrics(trial_df_rev, metadata,
                           animal_dir / f'{animal_id}_adaptation.png')

    # Return results
    results = {
        'animal_id': animal_id,
        'cohort': cohort,
        'metadata': metadata,
        'model': model,
        'feature_names': feature_names,
        'state_metrics': state_metrics,
        'trajectory_df': trajectory_df,
        'validated_labels': validated_labels,
        'n_trials': len(trial_df_rev),
        'n_reversals': trial_df_rev['n_reversals_in_session'].sum()
    }

    return results


def run_cohort_analysis_phase2(data_file, ldr_file, cohort, n_states=3):
    """
    Run Phase 2 analysis for entire cohort.

    Args:
        data_file: Path to data file
        ldr_file: Path to LDR criterion file
        cohort: 'W' or 'F'
        n_states: Number of states

    Returns:
        cohort_results: List of results dictionaries
    """
    print(f"\n{'='*70}")
    print(f"PHASE 2 COHORT ANALYSIS: Cohort {cohort}")
    print(f"{'='*70}")

    # Load data
    trial_df, _ = load_mouse_data(data_file)

    # Get unique animals
    animal_ids = trial_df['animal_id'].unique()
    print(f"Total animals in cohort: {len(animal_ids)}")

    # Analyze each animal
    cohort_results = []
    for i, animal_id in enumerate(animal_ids, 1):
        print(f"\n[{i}/{len(animal_ids)}] Processing {animal_id}...")

        results = analyze_single_animal_phase2(data_file, ldr_file, animal_id, cohort, n_states)
        if results is not None:
            cohort_results.append(results)

    print(f"\n{'='*70}")
    print(f"Cohort {cohort} Analysis Complete")
    print(f"Successfully analyzed: {len(cohort_results)}/{len(animal_ids)} animals")
    print(f"{'='*70}")

    return cohort_results


def create_cohort_summary(cohort_results, cohort, save_dir):
    """
    Create summary statistics and plots for cohort.
    """
    print(f"\nCreating cohort summary for Cohort {cohort}...")

    n_animals = len(cohort_results)
    n_states = cohort_results[0]['model'].n_states

    # Aggregate state labels
    label_counts = {}
    for state in range(n_states):
        label_counts[state] = {}

    for results in cohort_results:
        for state in range(n_states):
            label, _, _ = results['validated_labels'][state]
            label_counts[state][label] = label_counts[state].get(label, 0) + 1

    # Aggregate stimulus weights
    stimulus_weights_by_state = {state: [] for state in range(n_states)}
    for results in cohort_results:
        feature_names = results['feature_names']
        stimulus_idx = feature_names.index('stimulus_correct_side')
        weights = results['model'].weights
        for state in range(n_states):
            stimulus_weights_by_state[state].append(weights[state, stimulus_idx])

    # Print summary
    summary_text = f"Cohort {cohort} Phase 2 Summary\n"
    summary_text += f"{'='*50}\n"
    summary_text += f"Animals analyzed: {n_animals}\n"
    summary_text += f"States per animal: {n_states}\n\n"
    summary_text += f"State Label Distribution:\n"
    for state in range(n_states):
        summary_text += f"\nState {state}:\n"
        for label, count in sorted(label_counts[state].items(), key=lambda x: -x[1]):
            pct = 100 * count / n_animals
            summary_text += f"  {label}: {count}/{n_animals} ({pct:.1f}%)\n"

        # Add stimulus weight statistics
        weights = stimulus_weights_by_state[state]
        summary_text += f"\n  Stimulus weight (mean ± std): {np.mean(weights):.3f} ± {np.std(weights):.3f}\n"

    print(summary_text)

    # Save summary
    summary_file = save_dir / f'cohort_{cohort}_phase2_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(summary_text)

    print(f"✓ Saved cohort summary: {summary_file.name}")


def main():
    """
    Main execution function for Phase 2 analysis.
    """
    print("="*70)
    print("GLM-HMM PHASE 2 ANALYSIS: REVERSAL TASKS")
    print("="*70)
    print("\nAnalysis configuration:")
    print("  Tasks: LD reversal (v3 for W, Reversal 9 for F)")
    print("  Features: 8 (including corrected stimulus)")
    print("  States: 3")
    print("  Validation: Performance trajectories + behavioral metrics")
    print("  Special: Psychometric curves, reversal transitions, adaptation")
    print("  Cohorts: W and F (analyzed separately)")

    # File paths
    W_DATA = '/home/user/GLMHMM/W LD Data 11.08 All_processed.csv'
    F_DATA = '/home/user/GLMHMM/F LD Data 11.08 All_processed.csv'
    LDR_FILE = '/home/user/GLMHMM/LDR 2025 data1_processed_withSecondCriterion.csv'

    # Run W cohort
    print("\n" + "="*70)
    print("COHORT W ANALYSIS")
    print("="*70)
    w_results = run_cohort_analysis_phase2(W_DATA, LDR_FILE, cohort='W', n_states=3)
    create_cohort_summary(w_results, 'W', OUTPUT_DIR)

    # Run F cohort
    print("\n" + "="*70)
    print("COHORT F ANALYSIS")
    print("="*70)
    f_results = run_cohort_analysis_phase2(F_DATA, LDR_FILE, cohort='F', n_states=3)
    create_cohort_summary(f_results, 'F', OUTPUT_DIR)

    print("\n" + "="*70)
    print("PHASE 2 ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("\nNext step: Cross-cohort comparison and hypothesis testing")


if __name__ == '__main__':
    main()
