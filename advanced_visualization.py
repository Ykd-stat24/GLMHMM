"""
Advanced Visualization Suite for Dual-Process and VTE Analyses

Creates publication-quality figures for:
1. Discrete lapse state analysis
2. Deliberative vs procedural state identification
3. VTE and latency variability
4. State transitions at reversals
5. Learning trajectories
6. Genotype comparisons

Author: Claude (Anthropic)
Date: 2025-11-08
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings


def plot_latency_variability_over_learning(latency_metrics, figsize=(14, 8)):
    """
    Plot how latency variability (CV) changes during learning.

    Shows:
    - CV over sessions (by genotype)
    - Deliberation index trajectory
    - Task phase annotations

    Parameters:
    -----------
    latency_metrics : DataFrame
        Output from compute_latency_variability_metrics
    figsize : tuple
        Figure size

    Returns:
    --------
    fig : Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Group by genotype
    genotypes = latency_metrics['genotype'].unique()
    colors = {'+': 'blue', '-': 'red', np.nan: 'gray'}

    # 1. CV over sessions
    ax = axes[0, 0]
    for geno in genotypes:
        if pd.isna(geno):
            continue
        geno_data = latency_metrics[latency_metrics['genotype'] == geno]
        # Average across animals
        session_cv = geno_data.groupby('session_index')['cv_latency'].mean()
        session_sem = geno_data.groupby('session_index')['cv_latency'].sem()

        ax.plot(session_cv.index, session_cv.values, 'o-', label=f'Genotype {geno}',
               color=colors.get(geno, 'gray'), linewidth=2, markersize=6, alpha=0.7)
        ax.fill_between(session_cv.index,
                        session_cv.values - session_sem.values,
                        session_cv.values + session_sem.values,
                        alpha=0.2, color=colors.get(geno, 'gray'))

    ax.set_xlabel('Session Index', fontsize=12)
    ax.set_ylabel('Latency Coefficient of Variation', fontsize=12)
    ax.set_title('Latency Variability Over Training', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='High variability threshold')

    # 2. Deliberation index over sessions
    ax = axes[0, 1]
    for geno in genotypes:
        if pd.isna(geno):
            continue
        geno_data = latency_metrics[latency_metrics['genotype'] == geno]
        session_delib = geno_data.groupby('session_index')['deliberation_index'].mean()
        session_sem = geno_data.groupby('session_index')['deliberation_index'].sem()

        ax.plot(session_delib.index, session_delib.values, 'o-', label=f'Genotype {geno}',
               color=colors.get(geno, 'gray'), linewidth=2, markersize=6, alpha=0.7)
        ax.fill_between(session_delib.index,
                        session_delib.values - session_sem.values,
                        session_delib.values + session_sem.values,
                        alpha=0.2, color=colors.get(geno, 'gray'))

    ax.set_xlabel('Session Index', fontsize=12)
    ax.set_ylabel('Deliberation Index', fontsize=12)
    ax.set_title('Deliberation (VTE) Over Training', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Task-dependent CV
    ax = axes[1, 0]
    task_types = latency_metrics['task_type'].unique()
    task_cv = latency_metrics.groupby(['task_type', 'genotype'])['cv_latency'].mean().unstack()

    task_cv.plot(kind='bar', ax=ax, color=[colors.get(g, 'gray') for g in task_cv.columns])
    ax.set_xlabel('Task Type', fontsize=12)
    ax.set_ylabel('Mean CV Latency', fontsize=12)
    ax.set_title('Latency Variability by Task', fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Genotype')
    ax.grid(axis='y', alpha=0.3)

    # 4. Distribution of CV values
    ax = axes[1, 1]
    for geno in genotypes:
        if pd.isna(geno):
            continue
        geno_data = latency_metrics[latency_metrics['genotype'] == geno]['cv_latency']
        ax.hist(geno_data, bins=30, alpha=0.6, label=f'Genotype {geno}',
               color=colors.get(geno, 'gray'), edgecolor='black')

    ax.set_xlabel('Latency CV', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Latency Variability', fontsize=14, fontweight='bold')
    ax.legend()
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_state_classification_dual_process(state_classification, figsize=(14, 6)):
    """
    Visualize state classification in dual-process framework.

    Shows:
    - State characteristics (accuracy vs CV)
    - Process type labels
    - Occupancy as bubble size

    Parameters:
    -----------
    state_classification : DataFrame
        Output from identify_vte_states
    figsize : tuple
        Figure size

    Returns:
    --------
    fig : Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Color map for state types
    type_colors = {
        'Deliberative/VTE': 'orange',
        'Procedural/Automatic': 'green',
        'Engaged': 'blue',
        'Perseverative': 'red',
        'Lapse/Random': 'gray',
        'Mixed': 'purple'
    }

    # 1. Accuracy vs CV scatter
    ax = axes[0]
    for idx, row in state_classification.iterrows():
        color = type_colors.get(row['state_type'], 'black')
        ax.scatter(row['cv_latency'], row['accuracy'],
                  s=row['occupancy'] * 2000,  # Size by occupancy
                  c=color, alpha=0.7, edgecolors='black', linewidth=2)
        ax.text(row['cv_latency'], row['accuracy'],
               f"S{row['state']+1}", fontsize=10, ha='center', va='center', fontweight='bold')

    ax.set_xlabel('Latency CV (Variability)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('State Classification: Dual-Process Framework', fontsize=14, fontweight='bold')
    ax.axhline(0.75, color='green', linestyle='--', alpha=0.5, label='High performance')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='High variability')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max(state_classification['cv_latency'].max() * 1.1, 1.0))
    ax.set_ylim(0, 1.05)

    # Add legend for state types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=stype, alpha=0.7)
                      for stype, color in type_colors.items()
                      if stype in state_classification['state_type'].values]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # 2. State occupancy by process type
    ax = axes[1]
    process_occ = state_classification.groupby('process_type')['occupancy'].sum()
    colors_proc = [type_colors.get(state_classification[state_classification['process_type'] == pt]['state_type'].iloc[0], 'gray')
                  for pt in process_occ.index]

    ax.bar(range(len(process_occ)), process_occ.values, color=colors_proc, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(process_occ)))
    ax.set_xticklabels(process_occ.index, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Proportion of Trials', fontsize=12)
    ax.set_title('State Occupancy by Process Type', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    return fig


def plot_lapse_discreteness_analysis(lapse_results, model, metadata, figsize=(16, 10)):
    """
    Comprehensive visualization for Hypothesis 1: Are lapses discrete?

    Shows:
    - Run length distribution
    - State sequence with lapse runs highlighted
    - Autocorrelation
    - Transition matrix

    Parameters:
    -----------
    lapse_results : dict
        Output from analyze_lapse_discreteness
    model : GLMHMM
        Fitted model
    metadata : dict
        Trial metadata
    figsize : tuple
        Figure size

    Returns:
    --------
    fig : Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    lapse_state = lapse_results['lapse_state']
    lapse_mask = model.most_likely_states == lapse_state

    # 1. Run length distribution
    ax1 = fig.add_subplot(gs[0, 0])
    runs = []
    current_run = 0
    for is_lapse in lapse_mask:
        if is_lapse:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)

    ax1.hist(runs, bins=range(1, max(runs)+2), alpha=0.7, color='gray', edgecolor='black')
    ax1.axvline(lapse_results['mean_run_length_observed'], color='red', linewidth=3,
               label=f"Observed mean: {lapse_results['mean_run_length_observed']:.1f}")
    ax1.axvline(lapse_results['mean_run_length_expected_random'], color='blue', linewidth=3, linestyle='--',
               label=f"Random expectation: {lapse_results['mean_run_length_expected_random']:.1f}")
    ax1.set_xlabel('Run Length (consecutive lapse trials)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Lapse Run Length Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # 2. State sequence over trials
    ax2 = fig.add_subplot(gs[0, 1:])
    n_plot = min(500, len(model.most_likely_states))
    states_plot = model.most_likely_states[:n_plot]

    # Color by state
    colors_state = plt.cm.Set2(np.linspace(0, 1, model.n_states))
    for state in range(model.n_states):
        state_trials = np.where(states_plot == state)[0]
        if state == lapse_state:
            ax2.scatter(state_trials, states_plot[state_trials], c='red', s=30, label=f'State {state+1} (LAPSE)', alpha=0.8, marker='s')
        else:
            ax2.scatter(state_trials, states_plot[state_trials], c=[colors_state[state]], s=15, label=f'State {state+1}', alpha=0.6)

    ax2.set_xlabel('Trial', fontsize=11)
    ax2.set_ylabel('State', fontsize=11)
    ax2.set_title(f'State Sequence (first {n_plot} trials)', fontsize=12, fontweight='bold')
    ax2.set_yticks(range(model.n_states))
    ax2.set_yticklabels([f'S{i+1}' for i in range(model.n_states)])
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)

    # 3. Autocorrelation
    ax3 = fig.add_subplot(gs[1, 0])
    lapse_indicator = lapse_mask.astype(int)
    max_lag = 50
    autocorrs = []
    for lag in range(max_lag):
        if lag == 0:
            autocorrs.append(1.0)
        else:
            autocorrs.append(np.corrcoef(lapse_indicator[:-lag], lapse_indicator[lag:])[0, 1])

    ax3.bar(range(max_lag), autocorrs, alpha=0.7, color='purple', edgecolor='black')
    ax3.axhline(0, color='black', linewidth=1)
    ax3.set_xlabel('Lag (trials)', fontsize=11)
    ax3.set_ylabel('Autocorrelation', fontsize=11)
    ax3.set_title('Lapse State Autocorrelation', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.text(0.95, 0.95, f"Lag-1: {lapse_results['autocorrelation_lag1']:.3f}",
            transform=ax3.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 4. Proportion of long runs
    ax4 = fig.add_subplot(gs[1, 1])
    long_runs = np.sum(np.array(runs) >= 3)
    short_runs = len(runs) - long_runs
    ax4.pie([long_runs, short_runs], labels=['Long runs (≥3)', 'Short runs (<3)'],
           autopct='%1.1f%%', colors=['red', 'lightgray'], startangle=90)
    ax4.set_title(f'Run Length Categories\n({lapse_results["proportion_long_runs"]:.1%} long runs)', fontsize=12, fontweight='bold')

    # 5. Interpretation box
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    interp_text = f"""
LAPSE DISCRETENESS ANALYSIS

Lapse State: {lapse_state + 1}
Lapse Probability: {lapse_results['lapse_probability']:.2%}
Accuracy: {lapse_results['lapse_accuracy']:.2%}

Run Length Analysis:
  Observed: {lapse_results['mean_run_length_observed']:.2f} trials
  Expected: {lapse_results['mean_run_length_expected_random']:.2f} trials
  Ratio: {lapse_results['run_length_ratio']:.2f}x

Temporal Structure:
  Autocorrelation: {lapse_results['autocorrelation_lag1']:.3f}
  Long runs (≥3): {lapse_results['proportion_long_runs']:.1%}
  Transition entropy: {lapse_results['transition_entropy']:.2f}

INTERPRETATION:
{lapse_results['interpretation']}
"""

    ax5.text(0.05, 0.95, interp_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # 6. Performance in lapse vs other states
    ax6 = fig.add_subplot(gs[2, :])
    for state in range(model.n_states):
        state_mask = model.most_likely_states == state
        if state_mask.sum() == 0:
            continue

        # Rolling accuracy
        window = 20
        state_trials = np.where(state_mask)[0]
        if len(state_trials) < window:
            continue

        state_accuracies = []
        for i in range(len(state_trials) - window + 1):
            trial_indices = state_trials[i:i+window]
            acc = metadata['correct'][trial_indices].mean()
            state_accuracies.append(acc)

        if state == lapse_state:
            ax6.plot(state_accuracies, linewidth=3, label=f'State {state+1} (LAPSE)',
                    color='red', alpha=0.8)
        else:
            ax6.plot(state_accuracies, linewidth=2, label=f'State {state+1}',
                    alpha=0.6)

    ax6.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax6.set_xlabel('Window', fontsize=11)
    ax6.set_ylabel('Rolling Accuracy (20 trials)', fontsize=11)
    ax6.set_title('Performance Stability by State', fontsize=12, fontweight='bold')
    ax6.legend(loc='best', fontsize=9)
    ax6.grid(alpha=0.3)
    ax6.set_ylim(0, 1.05)

    plt.suptitle('Hypothesis 1: Are Lapses Discrete States?', fontsize=16, fontweight='bold', y=0.995)

    return fig


def plot_deliberation_learning_correlation(delib_results, figsize=(14, 10)):
    """
    Test Hypothesis 2: Does early deliberation predict better learning?

    Shows:
    - Early CV vs final performance scatter
    - Early deliberation vs reversal speed
    - Genotype comparison
    - High vs low learners

    Parameters:
    -----------
    delib_results : DataFrame
        Output from test_deliberation_learning_hypothesis
    figsize : tuple
        Figure size

    Returns:
    --------
    fig : Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Check if results are empty or insufficient
    if len(delib_results) < 2:
        n_animals = len(delib_results) if len(delib_results) > 0 else 0
        fig.suptitle(f'Insufficient data for correlation analysis ({n_animals} animal(s), requires 2+)',
                    fontsize=16, fontweight='bold')
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'Insufficient data\n(requires 2+ animals for correlation)',
                   ha='center', va='center', fontsize=14, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        return fig

    colors = {'+': 'blue', '-': 'red', np.nan: 'gray'}

    # 1. Early deliberation vs final performance
    ax = axes[0, 0]
    if 'genotype' in delib_results.columns:
        for geno in delib_results['genotype'].unique():
            if pd.isna(geno):
                continue
            geno_data = delib_results[delib_results['genotype'] == geno]
            ax.scatter(geno_data['early_deliberation_index'], geno_data['final_accuracy'],
                  s=100, alpha=0.7, label=f'Genotype {geno}', color=colors.get(geno, 'gray'),
                  edgecolors='black', linewidth=1.5)

    # Regression line
    from scipy import stats
    x = delib_results['early_deliberation_index']
    y = delib_results['final_accuracy']
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    ax.plot(x, slope * x + intercept, 'k--', linewidth=2,
           label=f'r={r_value:.3f}, p={p_value:.3f}')

    ax.set_xlabel('Early Deliberation Index', fontsize=12)
    ax.set_ylabel('Final Accuracy', fontsize=12)
    ax.set_title('Deliberation → Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 2. Early CV vs reversal speed
    ax = axes[0, 1]
    valid_reversal = delib_results['reversal_speed'].notna()
    if valid_reversal.sum() > 0:
        for geno in delib_results['genotype'].unique():
            if pd.isna(geno):
                continue
            geno_data = delib_results[(delib_results['genotype'] == geno) & valid_reversal]
            if len(geno_data) > 0:
                ax.scatter(geno_data['early_cv_latency'], geno_data['reversal_speed'],
                          s=100, alpha=0.7, label=f'Genotype {geno}', color=colors.get(geno, 'gray'),
                          edgecolors='black', linewidth=1.5)

        # Regression
        x_rev = delib_results.loc[valid_reversal, 'early_cv_latency']
        y_rev = delib_results.loc[valid_reversal, 'reversal_speed']
        if len(x_rev) > 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_rev, y_rev)
            ax.plot(x_rev, slope * x_rev + intercept, 'k--', linewidth=2,
                   label=f'r={r_value:.3f}, p={p_value:.3f}')

    ax.set_xlabel('Early Latency CV', fontsize=12)
    ax.set_ylabel('Trials to Criterion (Reversal)', fontsize=12)
    ax.set_title('Variability → Reversal Learning', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Genotype comparison
    ax = axes[1, 0]
    geno_means = delib_results.groupby('genotype')[['early_deliberation_index', 'final_accuracy']].mean()
    geno_sems = delib_results.groupby('genotype')[['early_deliberation_index', 'final_accuracy']].sem()

    x_pos = np.arange(len(geno_means))
    width = 0.35

    for i, col in enumerate(['early_deliberation_index', 'final_accuracy']):
        offset = (i - 0.5) * width
        bars = ax.bar(x_pos + offset, geno_means[col], width,
                     yerr=geno_sems[col], capsize=5, alpha=0.7,
                     label=col.replace('_', ' ').title())

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Genotype {g}' for g in geno_means.index], rotation=0)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Genotype Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. High vs low learners
    ax = axes[1, 1]
    # Classify as high or low learners based on median split
    median_perf = delib_results['final_accuracy'].median()
    delib_results['learner_type'] = delib_results['final_accuracy'].apply(
        lambda x: 'High' if x >= median_perf else 'Low'
    )

    learner_delib = delib_results.groupby('learner_type')[['early_cv_latency', 'early_deliberation_index']].mean()
    learner_sem = delib_results.groupby('learner_type')[['early_cv_latency', 'early_deliberation_index']].sem()

    x_pos = np.arange(len(learner_delib))
    width = 0.35

    for i, col in enumerate(['early_cv_latency', 'early_deliberation_index']):
        offset = (i - 0.5) * width
        ax.bar(x_pos + offset, learner_delib[col], width,
              yerr=learner_sem[col], capsize=5, alpha=0.7,
              label=col.replace('_', ' ').title())

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{lt} Learners' for lt in learner_delib.index], rotation=0)
    ax.set_ylabel('Early Learning Metric', fontsize=12)
    ax.set_title('High vs Low Learners', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add interpretation if correlation is significant
    corr_perf = delib_results.attrs.get('correlation_performance', 0)
    p_perf = delib_results.attrs.get('p_value_performance', 1)

    interpretation = ""
    if p_perf < 0.05 and corr_perf > 0:
        interpretation = "✅ HYPOTHESIS SUPPORTED: Early deliberation predicts better learning"
    elif p_perf < 0.05 and corr_perf < 0:
        interpretation = "❌ OPPOSITE EFFECT: Early deliberation predicts worse learning"
    else:
        interpretation = "⚠️  NO SIGNIFICANT CORRELATION: Deliberation doesn't predict performance"

    fig.text(0.5, 0.02, interpretation, ha='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Hypothesis 2: Does Early Deliberation Predict Learning Success?',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    return fig


def plot_state_transitions_at_reversals(reversal_df, figsize=(14, 6)):
    """
    Visualize state changes when task contingencies reverse.

    Shows:
    - State occupancy before vs after reversal
    - Genotype differences in flexibility
    - Performance drop and recovery

    Parameters:
    -----------
    reversal_df : DataFrame
        Output from analyze_state_transitions_at_reversals
    figsize : tuple
        Figure size

    Returns:
    --------
    fig : Figure
    """
    if len(reversal_df) == 0:
        print("No reversal data available")
        return None

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. State occupancy change
    ax = axes[0]
    state_changes = reversal_df.groupby('state')['change_in_proportion'].mean()
    state_sems = reversal_df.groupby('state')['change_in_proportion'].sem()

    colors_change = ['green' if x > 0 else 'red' for x in state_changes.values]
    ax.bar(state_changes.index, state_changes.values, yerr=state_sems.values,
          capsize=5, alpha=0.7, color=colors_change, edgecolor='black')
    ax.axhline(0, color='black', linewidth=2)
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Change in Proportion\n(After - Before Reversal)', fontsize=12)
    ax.set_title('State Transitions at Reversal', fontsize=14, fontweight='bold')
    ax.set_xticks(state_changes.index)
    ax.set_xticklabels([f'State {s+1}' for s in state_changes.index])
    ax.grid(axis='y', alpha=0.3)

    # 2. Genotype comparison
    ax = axes[1]
    geno_changes = reversal_df.groupby(['genotype', 'state'])['change_in_proportion'].mean().unstack()

    geno_changes.plot(kind='bar', ax=ax, alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Genotype', fontsize=12)
    ax.set_ylabel('Change in State Proportion', fontsize=12)
    ax.set_title('Flexibility by Genotype', fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='State', labels=[f'S{i+1}' for i in geno_changes.columns])
    ax.grid(axis='y', alpha=0.3)

    # 3. Performance drop
    ax = axes[2]
    perf_before = reversal_df.groupby('state')['accuracy_before'].mean()
    perf_after = reversal_df.groupby('state')['accuracy_after'].mean()

    x = np.arange(len(perf_before))
    width = 0.35

    ax.bar(x - width/2, perf_before, width, label='Before Reversal', alpha=0.7, color='blue')
    ax.bar(x + width/2, perf_after, width, label='After Reversal', alpha=0.7, color='orange')

    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Performance at Reversal', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'State {s+1}' for s in perf_before.index])
    ax.legend()
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('State Dynamics During Reversal Learning', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_state_dwell_times(dwell_df, dwell_times_raw, figsize=(12, 8)):
    """
    Visualize how long animals persist in each state.

    Shows:
    - Distribution of dwell times for each state
    - Mean and median dwell times
    - State stability

    Parameters:
    -----------
    dwell_df : DataFrame
        Summary from compute_state_dwell_times
    dwell_times_raw : dict
        Raw dwell time lists
    figsize : tuple
        Figure size

    Returns:
    --------
    fig : Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    n_states = len(dwell_df)
    colors = plt.cm.Set2(np.linspace(0, 1, n_states))

    # 1. Dwell time distributions
    ax = axes[0, 0]
    for state in range(n_states):
        dwells = dwell_times_raw.get(state, [])
        if dwells:
            ax.hist(dwells, bins=range(1, min(max(dwells)+2, 100)), alpha=0.6,
                   label=f'State {state+1}', color=colors[state], edgecolor='black')

    ax.set_xlabel('Dwell Time (consecutive trials)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Dwell Time Distributions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 50)
    ax.grid(alpha=0.3)

    # 2. Mean dwell times
    ax = axes[0, 1]
    ax.bar(dwell_df['state'], dwell_df['mean_dwell'], alpha=0.7, color=colors, edgecolor='black')
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Mean Dwell Time (trials)', fontsize=12)
    ax.set_title('Average State Persistence', fontsize=14, fontweight='bold')
    ax.set_xticks(dwell_df['state'])
    ax.set_xticklabels([f'State {s+1}' for s in dwell_df['state']])
    ax.grid(axis='y', alpha=0.3)

    # 3. Number of state bouts
    ax = axes[1, 0]
    ax.bar(dwell_df['state'], dwell_df['n_bouts'], alpha=0.7, color=colors, edgecolor='black')
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Number of Bouts', fontsize=12)
    ax.set_title('State Switching Frequency', fontsize=14, fontweight='bold')
    ax.set_xticks(dwell_df['state'])
    ax.set_xticklabels([f'State {s+1}' for s in dwell_df['state']])
    ax.grid(axis='y', alpha=0.3)

    # 4. Stability index (mean/median ratio)
    ax = axes[1, 1]
    stability = dwell_df['mean_dwell'] / (dwell_df['median_dwell'] + 0.1)
    ax.bar(dwell_df['state'], stability, alpha=0.7, color=colors, edgecolor='black')
    ax.axhline(1, color='black', linestyle='--', linewidth=2, label='Mean = Median')
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Stability Index (Mean/Median)', fontsize=12)
    ax.set_title('State Stability\n(>1 = long-tailed distribution)', fontsize=14, fontweight='bold')
    ax.set_xticks(dwell_df['state'])
    ax.set_xticklabels([f'State {s+1}' for s in dwell_df['state']])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('State Dwell Time Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig
