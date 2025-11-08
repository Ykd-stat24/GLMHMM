"""
Additional Comprehensive Visualizations

Creates extra publication-quality figures for deeper insights:
- Session-by-session learning trajectories
- Task-specific state usage
- Weight evolution across animals
- Performance prediction models
- Genotype-specific learning curves

Author: Claude (Anthropic)
Date: 2025-11-08
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings


def plot_session_by_session_learning(trial_df, model, metadata, figsize=(16, 10)):
    """
    Plot learning trajectory session by session with state annotations.

    Shows:
    - Accuracy over sessions
    - State occupancy per session
    - Latency changes
    - Task transitions

    Parameters:
    -----------
    trial_df : DataFrame
        Trial data (for single animal)
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
    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # Get session-level data
    trial_df_subset = trial_df.copy()
    trial_df_subset['state'] = model.most_likely_states
    trial_df_subset['correct'] = metadata['correct']
    trial_df_subset['latency'] = metadata['latency']

    session_stats = trial_df_subset.groupby('session_index').agg({
        'correct': 'mean',
        'latency': 'mean',
        'state': lambda x: x.value_counts().to_dict(),
        'task_type': 'first'
    })

    # 1. Accuracy over sessions
    ax = axes[0, 0]
    ax.plot(session_stats.index, session_stats['correct'], 'o-', linewidth=2, markersize=6)
    ax.axhline(0.8, color='green', linestyle='--', alpha=0.5, label='Criterion')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Session', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Learning Curve', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Shade task regions
    for task in session_stats['task_type'].unique():
        task_sessions = session_stats[session_stats['task_type'] == task].index
        if len(task_sessions) > 0:
            ax.axvspan(task_sessions.min(), task_sessions.max(), alpha=0.1,
                      label=task)

    # 2. State occupancy per session
    ax = axes[0, 1]
    # Extract state occupancies
    state_occupancies = {s: [] for s in range(3)}
    sessions = []
    for sess_idx, state_dict in session_stats['state'].items():
        sessions.append(sess_idx)
        total = sum(state_dict.values())
        for s in range(3):
            state_occupancies[s].append(state_dict.get(s, 0) / total)

    # Stacked area plot
    bottom = np.zeros(len(sessions))
    colors = plt.cm.Set2(np.linspace(0, 1, 3))
    for s in range(3):
        ax.fill_between(sessions, bottom, bottom + state_occupancies[s],
                        alpha=0.7, label=f'State {s+1}', color=colors[s])
        bottom += state_occupancies[s]

    ax.set_xlabel('Session', fontsize=11)
    ax.set_ylabel('State Proportion', fontsize=11)
    ax.set_title('State Usage Over Sessions', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)

    # 3. Latency over sessions
    ax = axes[1, 0]
    ax.plot(session_stats.index, session_stats['latency'], 'o-',
           linewidth=2, markersize=6, color='purple')
    ax.set_xlabel('Session', fontsize=11)
    ax.set_ylabel('Mean Latency (s)', fontsize=11)
    ax.set_title('Response Speed Over Time', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    # 4. State-specific accuracy trends
    ax = axes[1, 1]
    for s in range(3):
        state_mask = trial_df_subset['state'] == s
        state_session_acc = trial_df_subset[state_mask].groupby('session_index')['correct'].mean()
        if len(state_session_acc) > 0:
            ax.plot(state_session_acc.index, state_session_acc.values, 'o-',
                   label=f'State {s+1}', alpha=0.7)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Session', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Performance by State Over Sessions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 5. Trials per session
    ax = axes[2, 0]
    trials_per_session = trial_df_subset.groupby('session_index').size()
    ax.bar(trials_per_session.index, trials_per_session.values, alpha=0.7)
    ax.set_xlabel('Session', fontsize=11)
    ax.set_ylabel('Number of Trials', fontsize=11)
    ax.set_title('Trial Count per Session', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 6. Cumulative performance
    ax = axes[2, 1]
    cumulative_correct = trial_df_subset['correct'].expanding().mean()
    ax.plot(cumulative_correct.values, linewidth=2)
    ax.axhline(0.8, color='green', linestyle='--', alpha=0.5, label='Criterion')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Cumulative Trial', fontsize=11)
    ax.set_ylabel('Cumulative Accuracy', fontsize=11)
    ax.set_title('Overall Learning Progress', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


def plot_task_specific_analysis(trial_df, model, metadata, figsize=(14, 10)):
    """
    Compare behavior across different tasks (PI, LD, LD Reversal).

    Shows:
    - State usage by task
    - Performance by task and state
    - State transitions between tasks

    Parameters:
    -----------
    trial_df : DataFrame
        Trial data
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
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Add states to dataframe
    trial_df_subset = trial_df.copy()
    trial_df_subset['state'] = model.most_likely_states
    trial_df_subset['correct'] = metadata['correct']

    # 1. State occupancy by task
    ax = axes[0, 0]
    task_state = trial_df_subset.groupby(['task_type', 'state']).size().unstack(fill_value=0)
    task_state_prop = task_state.div(task_state.sum(axis=1), axis=0)

    task_state_prop.plot(kind='bar', stacked=True, ax=ax, alpha=0.7,
                         color=plt.cm.Set2(np.linspace(0, 1, 3)))
    ax.set_xlabel('Task Type', fontsize=12)
    ax.set_ylabel('Proportion of Trials', fontsize=12)
    ax.set_title('State Usage by Task', fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='State', labels=[f'State {i+1}' for i in range(3)])
    ax.set_ylim(0, 1)

    # 2. Accuracy by task and state
    ax = axes[0, 1]
    task_state_acc = trial_df_subset.groupby(['task_type', 'state'])['correct'].mean().unstack()

    x = np.arange(len(task_state_acc))
    width = 0.25
    for i, state in enumerate(task_state_acc.columns):
        ax.bar(x + i * width, task_state_acc[state], width,
              label=f'State {state+1}', alpha=0.7)

    ax.set_xlabel('Task Type', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Performance by Task and State', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(task_state_acc.index, rotation=45, ha='right')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.8, color='green', linestyle='--', alpha=0.3)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 3. State transitions at task boundaries
    ax = axes[1, 0]
    task_changes = trial_df_subset['task_type'].ne(trial_df_subset['task_type'].shift())
    change_points = trial_df_subset[task_changes].index.tolist()

    # Plot state sequence with task changes highlighted
    states = model.most_likely_states
    ax.plot(states, 'o', markersize=3, alpha=0.5)

    for cp in change_points:
        ax.axvline(cp, color='red', linestyle='--', alpha=0.7, linewidth=2)

    ax.set_xlabel('Trial', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    ax.set_title('State Sequence with Task Transitions', fontsize=14, fontweight='bold')
    ax.set_yticks(range(3))
    ax.set_yticklabels([f'State {i+1}' for i in range(3)])

    # 4. Performance improvement by task
    ax = axes[1, 1]
    for task in trial_df_subset['task_type'].unique():
        task_data = trial_df_subset[trial_df_subset['task_type'] == task]
        if len(task_data) > 20:
            # Rolling accuracy
            rolling_acc = task_data['correct'].rolling(window=20, min_periods=5).mean()
            ax.plot(rolling_acc.values, label=task, alpha=0.7, linewidth=2)

    ax.set_xlabel('Trial within Task', fontsize=12)
    ax.set_ylabel('Rolling Accuracy (20 trials)', fontsize=12)
    ax.set_title('Learning Within Each Task', fontsize=14, fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.8, color='green', linestyle='--', alpha=0.3)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


def plot_weight_comparison_across_animals(multi_results, figsize=(14, 8)):
    """
    Compare GLM weights across multiple animals.

    Shows:
    - Weight distributions for each feature
    - Consistency across animals
    - Genotype differences

    Parameters:
    -----------
    multi_results : dict
        Dictionary of {animal_id: {model, metadata, genotype}}
    figsize : tuple
        Figure size

    Returns:
    --------
    fig : Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Extract weights
    all_weights = []
    for animal, results in multi_results.items():
        model = results['model']
        genotype = results['genotype']
        for state in range(model.n_states):
            for feat_idx, feat_name in enumerate(model.feature_names):
                all_weights.append({
                    'animal': animal,
                    'genotype': genotype,
                    'state': state + 1,
                    'feature': feat_name,
                    'weight': model.glm_weights[state, feat_idx],
                    'intercept': model.glm_intercepts[state]
                })

    weight_df = pd.DataFrame(all_weights)

    # 1. Weight distribution by feature
    ax = axes[0, 0]
    features = weight_df['feature'].unique()
    positions = np.arange(len(features))

    for state in sorted(weight_df['state'].unique()):
        state_data = weight_df[weight_df['state'] == state]
        means = [state_data[state_data['feature'] == f]['weight'].mean() for f in features]
        sems = [state_data[state_data['feature'] == f]['weight'].sem() for f in features]
        width = 0.25
        ax.bar(positions + (state-2) * width, means, width, yerr=sems,
              label=f'State {state}', alpha=0.7, capsize=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.set_ylabel('Mean Weight', fontsize=12)
    ax.set_title('GLM Weights Across Animals', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linewidth=1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Intercept distribution
    ax = axes[0, 1]
    for state in sorted(weight_df['state'].unique()):
        state_intercepts = weight_df[weight_df['state'] == state]['intercept'].unique()
        ax.hist(state_intercepts, bins=10, alpha=0.6, label=f'State {state}')

    ax.set_xlabel('Intercept Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Intercept Distribution Across Animals', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=2)
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Genotype comparison
    ax = axes[1, 0]
    for feat in features:
        geno_means = weight_df[weight_df['feature'] == feat].groupby('genotype')['weight'].mean()
        geno_sems = weight_df[weight_df['feature'] == feat].groupby('genotype')['weight'].sem()

        x = np.arange(len(geno_means))
        ax.errorbar(x + features.tolist().index(feat) * 0.1, geno_means.values,
                   yerr=geno_sems.values, marker='o', capsize=3,
                   label=feat, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Geno {g}' for g in geno_means.index])
    ax.set_ylabel('Mean Weight', fontsize=12)
    ax.set_title('Feature Weights by Genotype', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linewidth=1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4. Weight consistency (variance across animals)
    ax = axes[1, 1]
    weight_var = weight_df.groupby(['state', 'feature'])['weight'].std().unstack()

    weight_var.plot(kind='bar', ax=ax, alpha=0.7)
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Weight Std Dev', fontsize=12)
    ax.set_title('Weight Variability Across Animals', fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='Feature', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_genotype_learning_curves(trial_df_w, trial_df_f=None, figsize=(14, 8)):
    """
    Compare learning trajectories between genotypes.

    Shows:
    - Session-by-session accuracy by genotype
    - Time to criterion
    - Reversal performance

    Parameters:
    -----------
    trial_df_w : DataFrame
        W cohort data
    trial_df_f : DataFrame, optional
        F cohort data
    figsize : tuple
        Figure size

    Returns:
    --------
    fig : Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    colors = {'+': 'blue', '-': 'red', np.nan: 'gray'}

    # 1. Average learning curve by genotype
    ax = axes[0, 0]
    for geno in trial_df_w['genotype'].unique():
        if pd.isna(geno):
            continue
        geno_data = trial_df_w[trial_df_w['genotype'] == geno]

        session_acc = geno_data.groupby(['animal_id', 'session_index'])['correct'].mean()
        session_acc_df = session_acc.reset_index()

        # Average across animals
        mean_acc = session_acc_df.groupby('session_index')['correct'].mean()
        sem_acc = session_acc_df.groupby('session_index')['correct'].sem()

        ax.plot(mean_acc.index, mean_acc.values, 'o-', label=f'Genotype {geno}',
               color=colors[geno], linewidth=2, markersize=6, alpha=0.7)
        ax.fill_between(mean_acc.index,
                        mean_acc.values - sem_acc.values,
                        mean_acc.values + sem_acc.values,
                        alpha=0.2, color=colors[geno])

    ax.axhline(0.8, color='green', linestyle='--', alpha=0.5, label='Criterion')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Session', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Learning Curves by Genotype', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 2. Sessions to criterion
    ax = axes[0, 1]
    sessions_to_crit = []
    for animal in trial_df_w['animal_id'].unique():
        animal_data = trial_df_w[trial_df_w['animal_id'] == animal]
        session_accs = animal_data.groupby('session_index')['correct'].mean()
        criterion_sessions = session_accs[session_accs >= 0.8]

        if len(criterion_sessions) > 0:
            sessions_to_crit.append({
                'animal': animal,
                'genotype': animal_data['genotype'].iloc[0],
                'sessions_to_criterion': criterion_sessions.index[0]
            })

    if sessions_to_crit:
        crit_df = pd.DataFrame(sessions_to_crit)
        crit_by_geno = crit_df.groupby('genotype')['sessions_to_criterion'].agg(['mean', 'sem'])

        x = np.arange(len(crit_by_geno))
        ax.bar(x, crit_by_geno['mean'], yerr=crit_by_geno['sem'],
              color=[colors.get(g, 'gray') for g in crit_by_geno.index],
              alpha=0.7, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Genotype {g}' for g in crit_by_geno.index])
        ax.set_ylabel('Sessions to Criterion', fontsize=12)
        ax.set_title('Learning Speed by Genotype', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # 3. Task-specific performance
    ax = axes[1, 0]
    for task in ['PI', 'LD', 'LD_reversal']:
        task_data = trial_df_w[trial_df_w['task_type'] == task]
        if len(task_data) == 0:
            continue

        task_geno_acc = task_data.groupby('genotype')['correct'].agg(['mean', 'sem'])

        x = np.arange(len(task_geno_acc))
        width = 0.25
        offset = (['PI', 'LD', 'LD_reversal'].index(task) - 1) * width

        ax.bar(x + offset, task_geno_acc['mean'], width,
              yerr=task_geno_acc['sem'], label=task, alpha=0.7, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Geno {g}' for g in task_geno_acc.index])
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Performance by Task and Genotype', fontsize=14, fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.8, color='green', linestyle='--', alpha=0.3)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 4. Final performance distribution
    ax = axes[1, 1]
    final_accs = []
    for animal in trial_df_w['animal_id'].unique():
        animal_data = trial_df_w[trial_df_w['animal_id'] == animal]
        final_100 = animal_data.iloc[-100:] if len(animal_data) > 100 else animal_data
        final_accs.append({
            'animal': animal,
            'genotype': animal_data['genotype'].iloc[0],
            'final_accuracy': final_100['correct'].mean()
        })

    final_df = pd.DataFrame(final_accs)
    for geno in final_df['genotype'].unique():
        if pd.isna(geno):
            continue
        geno_accs = final_df[final_df['genotype'] == geno]['final_accuracy']
        ax.hist(geno_accs, bins=10, alpha=0.6, label=f'Genotype {geno}',
               color=colors.get(geno, 'gray'))

    ax.set_xlabel('Final Accuracy (last 100 trials)', fontsize=12)
    ax.set_ylabel('Number of Animals', fontsize=12)
    ax.set_title('Final Performance Distribution', fontsize=14, fontweight='bold')
    ax.axvline(0.8, color='green', linestyle='--', linewidth=2, label='Criterion')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig
