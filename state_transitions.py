"""
State Transition Analysis Module
=================================

Analyzes state transitions, stability, and genotype differences.

Key analyses:
1. Transition matrices (probability of state i → state j)
2. State stability (dwell times)
3. Lapse recovery metrics
4. Sequential patterns
5. Genotype comparisons

Author: Claude (Anthropic)
Date: November 9, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from collections import Counter, defaultdict
from pathlib import Path


def compute_state_transition_matrix(states):
    """
    Compute state transition probability matrix.

    Parameters:
    -----------
    states : array
        Most likely state sequence

    Returns:
    --------
    transition_matrix : ndarray (n_states, n_states)
        P(state_j | state_i) - probability of transitioning from i to j
    transition_counts : ndarray (n_states, n_states)
        Raw counts of transitions
    """
    n_states = len(np.unique(states))
    transitions = np.zeros((n_states, n_states))

    # Count transitions
    for i in range(len(states) - 1):
        from_state = states[i]
        to_state = states[i + 1]
        transitions[from_state, to_state] += 1

    # Normalize to probabilities
    row_sums = transitions.sum(axis=1, keepdims=True)
    transition_probs = np.divide(transitions, row_sums,
                                  where=row_sums > 0,
                                  out=np.zeros_like(transitions))

    return transition_probs, transitions


def compute_state_stability(states):
    """
    Compute state stability metrics (dwell times).

    Parameters:
    -----------
    states : array
        Most likely state sequence

    Returns:
    --------
    stability_metrics : dict
        {state_id: {'mean_dwell': float, 'median_dwell': float,
                    'n_bouts': int, 'total_trials': int}}
    """
    n_states = len(np.unique(states))
    stability = {}

    for state_id in range(n_states):
        # Find bouts (contiguous blocks) of this state
        bout_lengths = []
        current_bout = 0

        for s in states:
            if s == state_id:
                current_bout += 1
            else:
                if current_bout > 0:
                    bout_lengths.append(current_bout)
                current_bout = 0

        # Don't forget last bout
        if current_bout > 0:
            bout_lengths.append(current_bout)

        stability[state_id] = {
            'mean_dwell': np.mean(bout_lengths) if bout_lengths else 0,
            'median_dwell': np.median(bout_lengths) if bout_lengths else 0,
            'n_bouts': len(bout_lengths),
            'total_trials': sum(bout_lengths)
        }

    return stability


def compute_lapse_recovery_metrics(states, validated_labels):
    """
    Analyze how animals recover from lapse states.

    Parameters:
    -----------
    states : array
        Most likely state sequence
    validated_labels : dict
        {state_id: (label, confidence, evidence)}

    Returns:
    --------
    recovery_metrics : dict
        Recovery statistics for lapse states
    """
    # Identify lapse states
    lapse_states = [s for s, (label, conf, _) in validated_labels.items()
                    if 'Lapse' in label and conf > 0]

    if not lapse_states:
        return {'n_lapse_bouts': 0, 'recovery_times': []}

    recovery_times = []

    # Find lapse bouts and measure recovery
    i = 0
    while i < len(states):
        if states[i] in lapse_states:
            # Start of lapse bout
            bout_start = i

            # Find end of lapse bout
            while i < len(states) and states[i] in lapse_states:
                i += 1

            bout_end = i
            bout_length = bout_end - bout_start

            # Check if recovered (reached non-lapse state)
            if bout_end < len(states):
                recovery_times.append(bout_length)
        else:
            i += 1

    return {
        'n_lapse_bouts': len(recovery_times),
        'recovery_times': recovery_times,
        'mean_lapse_duration': np.mean(recovery_times) if recovery_times else 0,
        'median_lapse_duration': np.median(recovery_times) if recovery_times else 0
    }


def extract_state_sequences(states, window=3):
    """
    Extract common state sequences (n-grams).

    Parameters:
    -----------
    states : array
        Most likely state sequence
    window : int
        Length of sequences to extract

    Returns:
    --------
    sequences : Counter
        {sequence_tuple: count}
    """
    sequences = []

    for i in range(len(states) - window + 1):
        seq = tuple(states[i:i+window])
        sequences.append(seq)

    return Counter(sequences)


def compare_genotype_transitions(results_list, genotype_key='genotype'):
    """
    Compare state transition patterns across genotypes.

    Parameters:
    -----------
    results_list : list of dict
        List of analysis results, each with 'model', 'genotype', 'validated_labels'

    Returns:
    --------
    comparison : dict
        Genotype comparison statistics
    """
    # Group by genotype
    by_genotype = defaultdict(list)

    for result in results_list:
        geno = result[genotype_key]
        states = result['model'].most_likely_states
        labels = result['validated_labels']

        by_genotype[geno].append({
            'states': states,
            'labels': labels,
            'animal_id': result['animal_id']
        })

    comparison = {}

    for geno, animal_data in by_genotype.items():
        # Aggregate transitions across all animals in genotype
        all_transitions = []
        all_stability = []
        all_lapse_recovery = []

        for data in animal_data:
            # Transition matrix
            trans_probs, trans_counts = compute_state_transition_matrix(data['states'])
            all_transitions.append(trans_probs)

            # Stability
            stability = compute_state_stability(data['states'])
            all_stability.append(stability)

            # Lapse recovery
            recovery = compute_lapse_recovery_metrics(data['states'], data['labels'])
            all_lapse_recovery.append(recovery)

        # Average transition matrix for genotype
        mean_transition = np.mean(all_transitions, axis=0)

        # Average stability
        mean_dwell_by_state = {}
        for state_id in range(3):
            dwell_times = [s[state_id]['mean_dwell'] for s in all_stability if state_id in s]
            mean_dwell_by_state[state_id] = np.mean(dwell_times) if dwell_times else 0

        # Average lapse recovery
        all_recovery_times = []
        for rec in all_lapse_recovery:
            all_recovery_times.extend(rec['recovery_times'])

        comparison[geno] = {
            'n_animals': len(animal_data),
            'mean_transition_matrix': mean_transition,
            'mean_dwell_times': mean_dwell_by_state,
            'lapse_recovery': {
                'mean_duration': np.mean(all_recovery_times) if all_recovery_times else 0,
                'median_duration': np.median(all_recovery_times) if all_recovery_times else 0,
                'n_bouts': len(all_recovery_times)
            }
        }

    return comparison


def plot_transition_matrix(transition_matrix, state_labels, title, save_path):
    """
    Plot state transition matrix as heatmap.

    Parameters:
    -----------
    transition_matrix : ndarray (n_states, n_states)
        Transition probabilities
    state_labels : list of str
        Label for each state
    title : str
        Plot title
    save_path : Path
        Where to save figure
    """
    n_states = transition_matrix.shape[0]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Create heatmap
    im = ax.imshow(transition_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Transition Probability', rotation=270, labelpad=20)

    # Set ticks and labels
    ax.set_xticks(np.arange(n_states))
    ax.set_yticks(np.arange(n_states))
    ax.set_xticklabels([f"State {i}\n{state_labels[i][:20]}" for i in range(n_states)])
    ax.set_yticklabels([f"State {i}\n{state_labels[i][:20]}" for i in range(n_states)])

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(n_states):
        for j in range(n_states):
            text = ax.text(j, i, f'{transition_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black" if transition_matrix[i, j] < 0.5 else "white",
                          fontsize=12, weight='bold')

    ax.set_xlabel('To State', fontsize=12)
    ax.set_ylabel('From State', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_genotype_comparison(genotype_comparison, save_dir):
    """
    Plot comprehensive genotype comparison.

    Creates:
    1. Transition matrices per genotype
    2. State stability comparison
    3. Lapse recovery comparison
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    genotypes = sorted(genotype_comparison.keys())
    n_genotypes = len(genotypes)

    # 1. Plot transition matrices
    for geno in genotypes:
        trans_matrix = genotype_comparison[geno]['mean_transition_matrix']

        # Get most common state label for each state
        state_labels = [f"State {i}" for i in range(trans_matrix.shape[0])]

        plot_transition_matrix(
            trans_matrix,
            state_labels,
            f'State Transitions: {geno} Genotype (n={genotype_comparison[geno]["n_animals"]})',
            save_dir / f'transition_matrix_{geno.replace("/", "_")}.png'
        )

    # 2. State stability comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(3)  # 3 states
    width = 0.8 / n_genotypes

    for i, geno in enumerate(genotypes):
        dwell_times = genotype_comparison[geno]['mean_dwell_times']
        dwell_values = [dwell_times.get(state, 0) for state in range(3)]

        offset = (i - n_genotypes/2) * width + width/2
        ax.bar(x + offset, dwell_values, width, label=geno, alpha=0.8)

    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Mean Dwell Time (trials)', fontsize=12)
    ax.set_title('State Stability by Genotype', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'State {i}' for i in range(3)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'state_stability_by_genotype.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Lapse recovery comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Mean lapse duration
    geno_names = []
    mean_durations = []
    n_bouts = []

    for geno in genotypes:
        recovery = genotype_comparison[geno]['lapse_recovery']
        if recovery['n_bouts'] > 0:
            geno_names.append(geno)
            mean_durations.append(recovery['mean_duration'])
            n_bouts.append(recovery['n_bouts'])

    if geno_names:
        x_pos = np.arange(len(geno_names))

        ax1.bar(x_pos, mean_durations, alpha=0.7, color='coral')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(geno_names, rotation=45)
        ax1.set_ylabel('Mean Lapse Duration (trials)', fontsize=12)
        ax1.set_title('Lapse Duration by Genotype', fontsize=14)
        ax1.grid(axis='y', alpha=0.3)

        # Add n_bouts as text
        for i, (dur, n) in enumerate(zip(mean_durations, n_bouts)):
            ax1.text(i, dur + 0.5, f'n={n}', ha='center', fontsize=10)

        ax2.bar(x_pos, n_bouts, alpha=0.7, color='steelblue')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(geno_names, rotation=45)
        ax2.set_ylabel('Number of Lapse Bouts', fontsize=12)
        ax2.set_title('Lapse Frequency by Genotype', fontsize=14)
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'lapse_recovery_by_genotype.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved genotype comparison plots to {save_dir}")


def create_transition_summary_report(genotype_comparison, save_path):
    """
    Create text summary report of transition analysis.
    """
    report = []
    report.append("="*70)
    report.append("STATE TRANSITION ANALYSIS SUMMARY")
    report.append("="*70)
    report.append("")

    for geno in sorted(genotype_comparison.keys()):
        data = genotype_comparison[geno]
        report.append(f"\n{geno} Genotype (n={data['n_animals']} animals)")
        report.append("-"*50)

        # Transition matrix
        report.append("\nTransition Probabilities:")
        trans = data['mean_transition_matrix']
        report.append(f"         → State 0  → State 1  → State 2")
        for i in range(trans.shape[0]):
            report.append(f"State {i}    {trans[i,0]:.3f}      {trans[i,1]:.3f}      {trans[i,2]:.3f}")

        # State stability
        report.append("\nState Stability (Mean Dwell Times):")
        dwell = data['mean_dwell_times']
        for state in range(3):
            report.append(f"  State {state}: {dwell.get(state, 0):.1f} trials")

        # Lapse recovery
        report.append("\nLapse Recovery:")
        recovery = data['lapse_recovery']
        report.append(f"  Number of lapse bouts: {recovery['n_bouts']}")
        if recovery['n_bouts'] > 0:
            report.append(f"  Mean lapse duration: {recovery['mean_duration']:.1f} trials")
            report.append(f"  Median lapse duration: {recovery['median_duration']:.1f} trials")

        report.append("")

    # Write to file
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"✓ Saved transition summary report to {save_path.name}")

    return '\n'.join(report)


def analyze_single_animal_transitions(model, validated_labels, animal_id, save_dir):
    """
    Analyze transitions for a single animal and create visualizations.

    Returns:
    --------
    metrics : dict
        Transition metrics for this animal
    """
    save_dir = Path(save_dir)
    states = model.most_likely_states

    # Compute metrics
    trans_probs, trans_counts = compute_state_transition_matrix(states)
    stability = compute_state_stability(states)
    recovery = compute_lapse_recovery_metrics(states, validated_labels)

    # Create state labels for plotting
    state_labels = [validated_labels[i][0] for i in range(len(validated_labels))]

    # Plot transition matrix
    plot_transition_matrix(
        trans_probs,
        state_labels,
        f'State Transitions: {animal_id}',
        save_dir / f'{animal_id}_transitions.png'
    )

    print(f"  ✓ Saved transition analysis")

    return {
        'transition_probs': trans_probs,
        'transition_counts': trans_counts,
        'stability': stability,
        'lapse_recovery': recovery
    }
