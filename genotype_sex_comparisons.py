"""
Enhanced genotype and sex comparison analyses for GLM-HMM.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def plot_genotype_sex_learning_curves(trial_df, figsize=(16, 10)):
    """
    Compare learning curves across genotype and sex.

    Shows:
    - Accuracy over sessions by genotype
    - Accuracy over sessions by sex
    - Latency over sessions by genotype/sex
    - State occupancy by genotype/sex
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Define colors
    genotype_colors = {'+': '#2E86AB', '-': '#A23B72', 'nan': 'gray'}
    sex_colors = {'M': '#06A77D', 'F': '#D4A574', 'nan': 'gray'}

    # 1. Accuracy by genotype
    ax = axes[0, 0]
    for geno in trial_df['genotype'].dropna().unique():
        geno_data = trial_df[trial_df['genotype'] == geno]
        session_acc = geno_data.groupby('session_index')['correct'].mean()
        session_sem = geno_data.groupby('session_index')['correct'].sem()

        ax.plot(session_acc.index, session_acc.values,
                label=f'Genotype {geno}', color=genotype_colors.get(geno, 'gray'),
                linewidth=2, alpha=0.8)
        ax.fill_between(session_acc.index,
                        session_acc.values - session_sem.values,
                        session_acc.values + session_sem.values,
                        alpha=0.2, color=genotype_colors.get(geno, 'gray'))

    ax.set_xlabel('Session', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Learning Curves by Genotype', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.3, label='Chance')

    # 2. Accuracy by sex
    ax = axes[0, 1]
    for sex in trial_df['sex'].dropna().unique():
        sex_data = trial_df[trial_df['sex'] == sex]
        session_acc = sex_data.groupby('session_index')['correct'].mean()
        session_sem = sex_data.groupby('session_index')['correct'].sem()

        ax.plot(session_acc.index, session_acc.values,
                label=f'{sex}', color=sex_colors.get(sex, 'gray'),
                linewidth=2, alpha=0.8)
        ax.fill_between(session_acc.index,
                        session_acc.values - session_sem.values,
                        session_acc.values + session_sem.values,
                        alpha=0.2, color=sex_colors.get(sex, 'gray'))

    ax.set_xlabel('Session', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Learning Curves by Sex', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.3)

    # 3. Genotype × Sex interaction
    ax = axes[0, 2]
    for geno in trial_df['genotype'].dropna().unique():
        for sex in trial_df['sex'].dropna().unique():
            group_data = trial_df[(trial_df['genotype'] == geno) & (trial_df['sex'] == sex)]
            if len(group_data) < 10:
                continue
            session_acc = group_data.groupby('session_index')['correct'].mean()

            linestyle = '-' if sex == 'M' else '--'
            ax.plot(session_acc.index, session_acc.values,
                    label=f'{geno}/{sex}',
                    color=genotype_colors.get(geno, 'gray'),
                    linestyle=linestyle, linewidth=2, alpha=0.7)

    ax.set_xlabel('Session', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Genotype × Sex Interaction', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 4. Latency by genotype
    ax = axes[1, 0]
    for geno in trial_df['genotype'].dropna().unique():
        geno_data = trial_df[trial_df['genotype'] == geno]
        session_lat = geno_data.groupby('session_index')['latency'].median()
        session_sem = geno_data.groupby('session_index')['latency'].sem()

        ax.plot(session_lat.index, session_lat.values,
                label=f'Genotype {geno}', color=genotype_colors.get(geno, 'gray'),
                linewidth=2, alpha=0.8)
        ax.fill_between(session_lat.index,
                        session_lat.values - session_sem.values,
                        session_lat.values + session_sem.values,
                        alpha=0.2, color=genotype_colors.get(geno, 'gray'))

    ax.set_xlabel('Session', fontsize=12)
    ax.set_ylabel('Median Latency (s)', fontsize=12)
    ax.set_title('Response Speed by Genotype', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 5. Latency by sex
    ax = axes[1, 1]
    for sex in trial_df['sex'].dropna().unique():
        sex_data = trial_df[trial_df['sex'] == sex]
        session_lat = sex_data.groupby('session_index')['latency'].median()
        session_sem = sex_data.groupby('session_index')['latency'].sem()

        ax.plot(session_lat.index, session_lat.values,
                label=f'{sex}', color=sex_colors.get(sex, 'gray'),
                linewidth=2, alpha=0.8)
        ax.fill_between(session_lat.index,
                        session_lat.values - session_sem.values,
                        session_lat.values + session_sem.values,
                        alpha=0.2, color=sex_colors.get(sex, 'gray'))

    ax.set_xlabel('Session', fontsize=12)
    ax.set_ylabel('Median Latency (s)', fontsize=12)
    ax.set_title('Response Speed by Sex', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 6. Final performance comparison (bar plot)
    ax = axes[1, 2]

    # Get final performance (last 20% of trials per animal)
    final_perf = []
    for animal in trial_df['animal_id'].unique():
        animal_data = trial_df[trial_df['animal_id'] == animal].copy()
        if len(animal_data) < 50:
            continue
        n_final = int(len(animal_data) * 0.2)
        final_acc = animal_data.iloc[-n_final:]['correct'].mean()
        final_perf.append({
            'animal_id': animal,
            'genotype': animal_data['genotype'].iloc[0],
            'sex': animal_data['sex'].iloc[0],
            'final_accuracy': final_acc
        })

    final_df = pd.DataFrame(final_perf)
    final_df = final_df.dropna(subset=['genotype', 'sex'])

    # Create grouped bar plot
    x_labels = []
    x_pos = []
    bar_heights = []
    bar_colors = []
    bar_errors = []

    pos = 0
    for geno in sorted(final_df['genotype'].unique()):
        for sex in sorted(final_df['sex'].unique()):
            group = final_df[(final_df['genotype'] == geno) & (final_df['sex'] == sex)]
            if len(group) > 0:
                x_labels.append(f'{geno}/{sex}')
                x_pos.append(pos)
                bar_heights.append(group['final_accuracy'].mean())
                bar_errors.append(group['final_accuracy'].sem())
                bar_colors.append(genotype_colors.get(geno, 'gray'))
                pos += 1
        pos += 0.5  # Gap between genotypes

    ax.bar(x_pos, bar_heights, yerr=bar_errors, capsize=5,
           color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylabel('Final Accuracy', fontsize=12)
    ax.set_title('Final Performance by Group', fontsize=14, fontweight='bold')
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Add sample sizes
    for i, (x, h, label) in enumerate(zip(x_pos, bar_heights, x_labels)):
        geno, sex = label.split('/')
        n = len(final_df[(final_df['genotype'] == geno) & (final_df['sex'] == sex)])
        ax.text(x, 0.05, f'n={n}', ha='center', fontsize=9, color='white', fontweight='bold')

    fig.suptitle('Genotype and Sex Comparisons: Learning and Performance',
                fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout()

    return fig


def plot_state_occupancy_by_groups(multi_results, figsize=(14, 10)):
    """
    Compare GLM-HMM state occupancy across genotype and sex.

    Parameters:
    -----------
    multi_results : dict
        Dictionary of animal_id -> {model, metadata, genotype, sex}
    """
    # Extract state data
    state_data = []
    for animal, results in multi_results.items():
        model = results['model']
        genotype = results.get('genotype', np.nan)
        sex = results.get('sex', np.nan)

        for state in range(model.n_states):
            mask = model.most_likely_states == state
            state_data.append({
                'animal': animal,
                'genotype': genotype,
                'sex': sex,
                'state': state + 1,
                'occupancy': mask.sum() / len(mask),
                'accuracy': results['metadata']['correct'][mask].mean()
            })

    state_df = pd.DataFrame(state_data)
    state_df = state_df.dropna(subset=['genotype', 'sex'])

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    genotype_colors = {'+': '#2E86AB', '-': '#A23B72'}
    sex_colors = {'M': '#06A77D', 'F': '#D4A574'}

    # 1. State occupancy by genotype
    ax = axes[0, 0]
    geno_state = state_df.groupby(['genotype', 'state'])['occupancy'].agg(['mean', 'sem']).reset_index()

    for geno in geno_state['genotype'].unique():
        geno_data = geno_state[geno_state['genotype'] == geno]
        ax.bar(geno_data['state'] + (0.2 if geno == '+' else -0.2),
               geno_data['mean'], width=0.35,
               yerr=geno_data['sem'], capsize=5,
               label=f'Genotype {geno}',
               color=genotype_colors.get(geno, 'gray'), alpha=0.7)

    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Mean Occupancy', fontsize=12)
    ax.set_title('State Usage by Genotype', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. State occupancy by sex
    ax = axes[0, 1]
    sex_state = state_df.groupby(['sex', 'state'])['occupancy'].agg(['mean', 'sem']).reset_index()

    for sex in sex_state['sex'].unique():
        sex_data = sex_state[sex_state['sex'] == sex]
        ax.bar(sex_data['state'] + (0.15 if sex == 'M' else -0.15),
               sex_data['mean'], width=0.3,
               yerr=sex_data['sem'], capsize=5,
               label=f'{sex}',
               color=sex_colors.get(sex, 'gray'), alpha=0.7)

    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Mean Occupancy', fontsize=12)
    ax.set_title('State Usage by Sex', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 3. State accuracy by genotype
    ax = axes[1, 0]
    geno_acc = state_df.groupby(['genotype', 'state'])['accuracy'].agg(['mean', 'sem']).reset_index()

    for geno in geno_acc['genotype'].unique():
        geno_data = geno_acc[geno_acc['genotype'] == geno]
        ax.bar(geno_data['state'] + (0.2 if geno == '+' else -0.2),
               geno_data['mean'], width=0.35,
               yerr=geno_data['sem'], capsize=5,
               label=f'Genotype {geno}',
               color=genotype_colors.get(geno, 'gray'), alpha=0.7)

    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Mean Accuracy', fontsize=12)
    ax.set_title('State-Specific Accuracy by Genotype', fontsize=14, fontweight='bold')
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.3, label='Chance')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 4. State accuracy by sex
    ax = axes[1, 1]
    sex_acc = state_df.groupby(['sex', 'state'])['accuracy'].agg(['mean', 'sem']).reset_index()

    for sex in sex_acc['sex'].unique():
        sex_data = sex_acc[sex_acc['sex'] == sex]
        ax.bar(sex_data['state'] + (0.15 if sex == 'M' else -0.15),
               sex_data['mean'], width=0.3,
               yerr=sex_data['sem'], capsize=5,
               label=f'{sex}',
               color=sex_colors.get(sex, 'gray'), alpha=0.7)

    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Mean Accuracy', fontsize=12)
    ax.set_title('State-Specific Accuracy by Sex', fontsize=14, fontweight='bold')
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.3)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.suptitle('GLM-HMM State Analysis by Genotype and Sex',
                fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout()

    return fig


def statistical_group_comparisons(trial_df, multi_results):
    """
    Perform statistical tests for genotype and sex differences.

    Returns summary DataFrame with test results.
    """
    results = []

    # 1. Final accuracy: Genotype comparison
    final_perf = []
    for animal in trial_df['animal_id'].unique():
        animal_data = trial_df[trial_df['animal_id'] == animal]
        if len(animal_data) < 50:
            continue
        n_final = int(len(animal_data) * 0.2)
        final_perf.append({
            'animal': animal,
            'genotype': animal_data['genotype'].iloc[0],
            'sex': animal_data['sex'].iloc[0],
            'final_acc': animal_data.iloc[-n_final:]['correct'].mean(),
            'avg_latency': animal_data['latency'].median()
        })

    perf_df = pd.DataFrame(final_perf).dropna()

    if len(perf_df) > 0:
        # Genotype comparison
        geno_groups = perf_df['genotype'].unique()
        if len(geno_groups) >= 2:
            group1 = perf_df[perf_df['genotype'] == geno_groups[0]]['final_acc']
            group2 = perf_df[perf_df['genotype'] == geno_groups[1]]['final_acc']
            t_stat, p_val = stats.ttest_ind(group1, group2)
            results.append({
                'comparison': 'Final Accuracy',
                'factor': 'Genotype',
                'group1': f'{geno_groups[0]} (n={len(group1)})',
                'group2': f'{geno_groups[1]} (n={len(group2)})',
                'mean1': group1.mean(),
                'mean2': group2.mean(),
                't_statistic': t_stat,
                'p_value': p_val,
                'significant': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            })

        # Sex comparison
        sex_groups = perf_df['sex'].unique()
        if len(sex_groups) >= 2:
            group1 = perf_df[perf_df['sex'] == sex_groups[0]]['final_acc']
            group2 = perf_df[perf_df['sex'] == sex_groups[1]]['final_acc']
            t_stat, p_val = stats.ttest_ind(group1, group2)
            results.append({
                'comparison': 'Final Accuracy',
                'factor': 'Sex',
                'group1': f'{sex_groups[0]} (n={len(group1)})',
                'group2': f'{sex_groups[1]} (n={len(group2)})',
                'mean1': group1.mean(),
                'mean2': group2.mean(),
                't_statistic': t_stat,
                'p_value': p_val,
                'significant': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            })

    return pd.DataFrame(results)
