"""
Phase 1 Analysis: Non-Reversal Tasks
====================================

Analyzes non-reversal tasks where stimulus has zero variance:
- A_Mouse LD 1 choice v2 (position 8 only)
- A_Mouse LD Punish Incorrect Training v2

Key differences from standard analysis:
1. EXCLUDES stimulus feature (zero variance)
2. Focuses on strategy states (WSLS, perseveration, engagement)
3. Uses comprehensive state validation with performance trajectories
4. Runs separately for W and F cohorts

Features used (7 total):
1. bias (constant)
2. prev_choice
3. wsls (win-stay/lose-shift)
4. session_progression
5. recent_side_bias
6. task_stage
7. cumulative_experience
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from glmhmm_utils import (
    load_and_preprocess_session_data,
    create_design_matrix
)
from glmhmm_ashwood import GLMHMM
from state_validation import (
    compute_performance_trajectory,
    compute_comprehensive_state_metrics,
    validate_state_labels
)
from state_transitions import (
    analyze_single_animal_transitions,
    compare_genotype_transitions,
    plot_genotype_comparison,
    create_transition_summary_report
)

# Set random seed for reproducibility
np.random.seed(42)

# Create output directories
OUTPUT_DIR = Path('/home/user/GLMHMM/results/phase1_non_reversal')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)


def create_phase1_design_matrix(trial_df, animal_id):
    """
    Create design matrix for Phase 1 (non-reversal tasks).

    EXCLUDES stimulus feature (zero variance in non-reversal tasks).

    Returns:
        X_no_stim: Design matrix without stimulus (n_trials, 7)
        y: Binary choices
        feature_names_no_stim: Feature names without stimulus
        metadata: Metadata dict
        animal_data: Filtered trial dataframe
    """
    # Get full design matrix first
    X, y, feature_names, metadata, animal_data = create_design_matrix(
        trial_df,
        animal_id=animal_id,
        include_session_progression=True
    )

    # Remove stimulus column (first column)
    stimulus_idx = feature_names.index('stimulus_correct_side')
    feature_indices = [i for i in range(len(feature_names)) if i != stimulus_idx]

    X_no_stim = X[:, feature_indices]
    feature_names_no_stim = [feature_names[i] for i in feature_indices]

    print(f"\nPhase 1 Design Matrix (stimulus excluded):")
    print(f"  Shape: {X_no_stim.shape}")
    print(f"  Features: {feature_names_no_stim}")

    return X_no_stim, y, feature_names_no_stim, metadata, animal_data


def run_glmhmm_phase1(trial_df, animal_id, cohort, n_states=3, n_iter=100):
    """
    Run GLM-HMM on Phase 1 data (non-reversal tasks).

    Args:
        trial_df: Trial data from load_and_preprocess_session_data
        animal_id: Animal ID to analyze
        cohort: 'W' or 'F'
        n_states: Number of hidden states
        n_iter: Number of EM iterations

    Returns:
        model: Fitted GLMHMM model
        X: Design matrix used
        y: Choices
        feature_names: Feature names
        metadata: Metadata dict
        animal_data: Filtered trial data
    """
    # Filter to non-reversal tasks
    non_reversal_tasks = ['LD', 'PI', 'PD', 'PD_PI']
    trial_df_nr = trial_df[
        (trial_df['animal_id'] == animal_id) &
        (trial_df['task_type'].isin(non_reversal_tasks))
    ].copy()

    if len(trial_df_nr) == 0:
        print(f"WARNING: No non-reversal trials for {animal_id}")
        return None, None, None, None, None, None

    print(f"\n{'='*70}")
    print(f"Running GLM-HMM: Phase 1 (Non-Reversal)")
    print(f"{'='*70}")
    print(f"Animal: {animal_id}")
    print(f"Cohort: {cohort}")
    print(f"Tasks: {trial_df_nr['task_type'].value_counts().to_dict()}")
    print(f"Trials: {len(trial_df_nr)}")
    print(f"States: {n_states}")

    # Create design matrix (without stimulus)
    X, y, feature_names, metadata, animal_data = create_phase1_design_matrix(
        trial_df_nr, animal_id=None  # Already filtered
    )

    # Initialize and fit model
    model = GLMHMM(
        n_states=n_states,
        feature_names=feature_names,
        normalize_features=True,
        regularization_strength=1.0,
        random_state=42
    )

    print(f"\nFitting model...")
    model.fit(X, y, n_iter=n_iter, tolerance=1e-4, verbose=False)

    print(f"✓ Model fitted successfully")
    print(f"  Iterations: {len(model.log_likelihood_history)}")
    print(f"  Final log-likelihood: {model.log_likelihood_history[-1]:.2f}")

    return model, X, y, feature_names, metadata, animal_data


def plot_phase1_weights(model, feature_names, animal_id, cohort, save_path):
    """Plot GLM weights for Phase 1 analysis."""
    n_states = model.n_states
    weights = model.glm_weights  # (n_states, n_features)

    fig, axes = plt.subplots(1, n_states, figsize=(5*n_states, 6))
    if n_states == 1:
        axes = [axes]

    for state in range(n_states):
        ax = axes[state]

        y_pos = np.arange(len(feature_names))
        colors = ['red' if w < 0 else 'blue' for w in weights[state]]

        ax.barh(y_pos, weights[state], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('GLM Weight')
        ax.set_title(f'State {state}')
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle(f'Phase 1 GLM Weights (Non-Reversal)\n{animal_id} - Cohort {cohort}',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved weights plot")


def plot_state_validation(state_metrics, trajectory_df, validated_labels,
                          animal_id, cohort, save_path):
    """Plot comprehensive state validation results."""
    n_states = len(state_metrics)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, n_states, hspace=0.3, wspace=0.3)

    for state in range(n_states):
        label, confidence, evidence = validated_labels[state]

        # 1. Performance trajectory
        ax1 = fig.add_subplot(gs[0, state])
        state_traj = trajectory_df[trajectory_df['state'] == state]

        if len(state_traj) > 0:
            acc_before = state_traj['pre_accuracy'].dropna().values
            acc_during = state_traj['during_accuracy'].dropna().values
            acc_after = state_traj['post_accuracy'].dropna().values

            positions = [1, 2, 3]
            data = [acc_before, acc_during, acc_after]
            labels_x = ['Before', 'During', 'After']

            bp = ax1.boxplot(data, positions=positions, widths=0.5,
                            patch_artist=True, showfliers=False)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')

            ax1.set_xticks(positions)
            ax1.set_xticklabels(labels_x)
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            ax1.set_title(f'State {state}: Performance Trajectory')
            ax1.grid(axis='y', alpha=0.3)

        # 2. Key metrics
        ax2 = fig.add_subplot(gs[1, state])

        # Get metrics row for this state
        state_row = state_metrics[state_metrics['state'] == state].iloc[0]

        metric_names = ['Accuracy', 'Latency CV', 'WSLS Ratio', 'Side Bias']
        metric_values = [
            state_row['accuracy'],
            state_row['latency_cv'],
            state_row['wsls_ratio'],
            state_row['side_bias']
        ]

        y_pos = np.arange(len(metric_names))
        ax2.barh(y_pos, metric_values, alpha=0.7, color='steelblue')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(metric_names)
        ax2.set_xlabel('Value')
        ax2.set_title(f'State {state}: Key Metrics')
        ax2.grid(axis='x', alpha=0.3)

        # 3. Validated label and evidence
        ax3 = fig.add_subplot(gs[2, state])
        ax3.axis('off')

        label_text = f"State {state}: {label}\n(Confidence: {confidence}/3)\n\n"
        label_text += "Evidence:\n"
        for key, value in evidence.items():
            label_text += f"• {key}: {value}\n"

        ax3.text(0.05, 0.95, label_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'State Validation: {animal_id} - Cohort {cohort}\nPhase 1 (Non-Reversal)',
                 fontsize=14, y=0.995)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved validation plot")


def plot_learning_curves(animal_data, model, animal_id, cohort, save_path):
    """Plot learning curves with state occupancy overlay."""
    # Compute rolling accuracy
    window = 50
    rolling_acc = pd.Series(animal_data['correct'].values).rolling(window, min_periods=10).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    # 1. Learning curve
    ax1.plot(rolling_acc, linewidth=2, color='black')
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.set_title(f'Learning Curve: {animal_id} - Cohort {cohort}', fontsize=14)

    # 2. State occupancy
    trials = np.arange(len(animal_data))
    state_probs = model.state_probabilities

    for state in range(model.n_states):
        ax2.fill_between(trials, 0, state_probs[:, state],
                         alpha=0.6, label=f'State {state}')

    ax2.set_xlabel('Trial Number', fontsize=12)
    ax2.set_ylabel('State Probability', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved learning curves")


def analyze_single_animal_phase1(trial_df, animal_id, cohort, n_states=3):
    """
    Run complete Phase 1 analysis for a single animal.

    Returns:
        results: Dictionary with model, metrics, and labels (or None if failed)
    """
    print(f"\n{'#'*70}")
    print(f"# Analyzing: {animal_id} (Cohort {cohort})")
    print(f"{'#'*70}")

    # Run GLM-HMM
    result = run_glmhmm_phase1(trial_df, animal_id, cohort, n_states=n_states)

    if result[0] is None:  # model is None
        return None

    model, X, y, feature_names, metadata, animal_data = result

    # State validation
    print(f"\nValidating states...")

    # Need to create trial_df compatible with state_validation
    # The animal_data already has the necessary columns
    state_metrics = compute_comprehensive_state_metrics(animal_data, model, metadata)
    trajectory_df = compute_performance_trajectory(animal_data, model)
    validated_labels = validate_state_labels(state_metrics, trajectory_df)

    # Print validation results
    print(f"\nValidated State Labels:")
    for state in range(n_states):
        label, confidence, evidence = validated_labels[state]
        print(f"  State {state}: {label} (confidence={confidence})")

    # Create output directory
    animal_dir = FIGURES_DIR / f'{animal_id}_cohort{cohort}'
    animal_dir.mkdir(exist_ok=True, parents=True)

    # Generate plots
    print(f"\nGenerating plots...")
    plot_phase1_weights(model, feature_names, animal_id, cohort,
                       animal_dir / f'{animal_id}_weights.png')
    plot_state_validation(state_metrics, trajectory_df, validated_labels,
                         animal_id, cohort,
                         animal_dir / f'{animal_id}_validation.png')
    plot_learning_curves(animal_data, model, animal_id, cohort,
                        animal_dir / f'{animal_id}_learning.png')

    # Transition analysis
    transition_metrics = analyze_single_animal_transitions(
        model, validated_labels, animal_id, animal_dir
    )

    # Return results
    genotype = animal_data['genotype'].iloc[0] if 'genotype' in animal_data.columns else 'Unknown'

    results = {
        'animal_id': animal_id,
        'cohort': cohort,
        'genotype': genotype,
        'model': model,
        'feature_names': feature_names,
        'state_metrics': state_metrics,
        'trajectory_df': trajectory_df,
        'validated_labels': validated_labels,
        'transition_metrics': transition_metrics,
        'n_trials': len(animal_data)
    }

    # Save results as pickle for summary analysis
    import pickle
    pickle_path = OUTPUT_DIR / f'{animal_id}_cohort{cohort}_model.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  ✓ Saved model results to {pickle_path.name}")

    return results


def run_cohort_analysis_phase1(data_file, cohort, n_states=3):
    """
    Run Phase 1 analysis for entire cohort.

    Args:
        data_file: Path to data file
        cohort: 'W' or 'F'
        n_states: Number of states

    Returns:
        cohort_results: List of results dictionaries
    """
    print(f"\n{'='*70}")
    print(f"PHASE 1 COHORT ANALYSIS: Cohort {cohort}")
    print(f"{'='*70}")

    # Load data
    trial_df = load_and_preprocess_session_data(data_file)

    # Get unique animals
    animal_ids = trial_df['animal_id'].unique()
    print(f"Total animals in cohort: {len(animal_ids)}")

    # Analyze each animal
    cohort_results = []
    for i, animal_id in enumerate(animal_ids, 1):
        print(f"\n[{i}/{len(animal_ids)}] Processing {animal_id}...")

        results = analyze_single_animal_phase1(trial_df, animal_id, cohort, n_states)
        if results is not None:
            cohort_results.append(results)

    print(f"\n{'='*70}")
    print(f"Cohort {cohort} Analysis Complete")
    print(f"Successfully analyzed: {len(cohort_results)}/{len(animal_ids)} animals")
    print(f"{'='*70}")

    return cohort_results


def create_cohort_summary(cohort_results, cohort, save_dir):
    """Create summary statistics for cohort including transitions."""
    print(f"\nCreating cohort summary for Cohort {cohort}...")

    n_animals = len(cohort_results)
    n_states = cohort_results[0]['model'].n_states

    # Aggregate state labels
    label_counts = {state: {} for state in range(n_states)}

    for results in cohort_results:
        for state in range(n_states):
            label, _, _ = results['validated_labels'][state]
            label_counts[state][label] = label_counts[state].get(label, 0) + 1

    # Print and save summary
    summary_text = f"Cohort {cohort} - Phase 1 Summary\n"
    summary_text += f"{'='*50}\n"
    summary_text += f"Animals analyzed: {n_animals}\n"
    summary_text += f"States per animal: {n_states}\n\n"
    summary_text += f"State Label Distribution:\n"

    for state in range(n_states):
        summary_text += f"\nState {state}:\n"
        for label, count in sorted(label_counts[state].items(), key=lambda x: -x[1]):
            pct = 100 * count / n_animals
            summary_text += f"  {label}: {count}/{n_animals} ({pct:.1f}%)\n"

    print(summary_text)

    summary_file = save_dir / f'cohort_{cohort}_phase1_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(summary_text)

    print(f"✓ Saved summary: {summary_file.name}")

    # Genotype transition analysis
    print(f"\nAnalyzing genotype differences in state transitions...")
    genotype_comparison = compare_genotype_transitions(cohort_results)

    # Create genotype comparison plots
    geno_dir = save_dir / f'cohort_{cohort}_genotype_comparisons'
    plot_genotype_comparison(genotype_comparison, geno_dir)

    # Create transition summary report
    transition_report = create_transition_summary_report(
        genotype_comparison,
        save_dir / f'cohort_{cohort}_transition_summary.txt'
    )

    print(f"✓ Genotype transition analysis complete")


def main():
    """Main execution function for Phase 1 analysis."""
    print("="*70)
    print("GLM-HMM PHASE 1 ANALYSIS: NON-REVERSAL TASKS")
    print("="*70)
    print("\nConfiguration:")
    print("  Tasks: LD 1 choice v2, Punish Incorrect")
    print("  Features: 7 (excluding stimulus)")
    print("  States: 3")
    print("  Validation: Performance trajectories + behavioral metrics")
    print("  Cohorts: W and F (analyzed separately)")

    # File paths
    W_DATA = '/home/user/GLMHMM/W LD Data 11.08 All_processed.csv'
    F_DATA = '/home/user/GLMHMM/F LD Data 11.08 All_processed.csv'

    # Run W cohort
    print("\n" + "="*70)
    print("COHORT W ANALYSIS")
    print("="*70)
    w_results = run_cohort_analysis_phase1(W_DATA, cohort='W', n_states=3)
    create_cohort_summary(w_results, 'W', OUTPUT_DIR)

    # Run F cohort
    print("\n" + "="*70)
    print("COHORT F ANALYSIS")
    print("="*70)
    f_results = run_cohort_analysis_phase1(F_DATA, cohort='F', n_states=3)
    create_cohort_summary(f_results, 'F', OUTPUT_DIR)

    print("\n" + "="*70)
    print("PHASE 1 ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("\nNext step: Run Phase 2 analysis (reversal tasks)")


if __name__ == '__main__':
    main()
