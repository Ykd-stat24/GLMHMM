"""
Simple Figure Regenerator with New Labels
==========================================

Regenerates key figures using only data available in pickle files.
Uses new genotype labels and proper state naming.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/home/user/GLMHMM')
from genotype_labels import (
    relabel_genotype, get_state_label, get_state_color, get_genotype_color,
    GENOTYPE_ORDER, GENOTYPE_COLORS
)

plt.style.use('seaborn-v0_8-whitegrid')


def load_phase1_data():
    """Load Phase 1 results."""
    print("\nLoading Phase 1 data...")
    results_dir = Path('results/phase1_non_reversal')

    data_W, data_F = [], []

    for pkl_file in results_dir.glob('*_model.pkl'):
        with open(pkl_file, 'rb') as f:
            r = pickle.load(f)
            r['genotype'] = relabel_genotype(r['genotype'])

            if r['cohort'] == 'W':
                data_W.append(r)
            else:
                data_F.append(r)

    print(f"  W Cohort: {len(data_W)} animals")
    print(f"  F Cohort: {len(data_F)} animals")

    return data_W, data_F


def figure1_state_occupancy():
    """Figure 1: State occupancy by genotype."""
    print("\n" + "="*80)
    print("Generating Figure 1: State Occupancy by Genotype")
    print("="*80)

    data_W, data_F = load_phase1_data()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (cohort, data) in zip(axes, [('W', data_W), ('F', data_F)]):
        # Collect state occupancy by genotype
        geno_states = {}

        for r in data:
            g = r['genotype']
            if g not in geno_states:
                geno_states[g] = {0: [], 1: [], 2: []}

            # Calculate from trajectory_df
            traj = r['trajectory_df']
            total = traj['bout_length'].sum()

            for state in range(3):
                state_trials = traj[traj['state'] == state]['bout_length'].sum()
                pct = (state_trials / total * 100) if total > 0 else 0
                geno_states[g][state].append(pct)

        # Plot
        genotypes = [g for g in GENOTYPE_ORDER if g in geno_states]
        x = np.arange(len(genotypes))
        width = 0.25

        for i, state in enumerate([0, 1, 2]):
            means = [np.mean(geno_states[g][state]) for g in genotypes]
            sems = [stats.sem(geno_states[g][state]) if len(geno_states[g][state]) > 1 else 0
                    for g in genotypes]

            ax.bar(x + i*width, means, width, yerr=sems,
                   label=get_state_label(state), color=get_state_color(state),
                   alpha=0.8, capsize=5)

        ax.set_xlabel('Genotype', fontsize=14, fontweight='bold')
        ax.set_ylabel('% Time in State', fontsize=14, fontweight='bold')
        ax.set_title(f'Cohort {cohort}', fontsize=16, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(genotypes, rotation=45, ha='right', fontsize=12)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])

        # Add sample sizes
        for i, g in enumerate(genotypes):
            n = len(geno_states[g][0])
            ax.text(i, -8, f'n={n}', ha='center', fontsize=10)

    plt.suptitle('State Occupancy by Genotype (Phase 1: Non-Reversal)',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()

    output_dir = Path('results/updated_figures/phase1')
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'fig1_state_occupancy.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_state_occupancy.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_dir / 'fig1_state_occupancy.png'}")


def figure2_glm_weights():
    """Figure 2: GLM weights by state and genotype."""
    print("\n" + "="*80)
    print("Generating Figure 2: GLM Weights by State and Genotype")
    print("="*80)

    data_W, data_F = load_phase1_data()

    fig, axes = plt.subplots(3, 2, figsize=(18, 20))

    feature_names = ['bias', 'prev_choice', 'wsls', 'session\nprog',
                     'side_bias', 'task\nstage', 'cum_exp']

    for state in range(3):
        for col, (cohort, data) in enumerate([('W', data_W), ('F', data_F)]):
            ax = axes[state, col]

            # Collect weights by genotype
            geno_weights = {}

            for r in data:
                g = r['genotype']
                if g not in geno_weights:
                    geno_weights[g] = {f: [] for f in feature_names}

                weights = r['model'].glm_weights[state]
                for i, fname in enumerate(feature_names):
                    if i < len(weights):
                        geno_weights[g][fname].append(weights[i])

            # Plot
            genotypes = [g for g in GENOTYPE_ORDER if g in geno_weights]
            x = np.arange(len(feature_names))
            width = 0.8 / len(genotypes)

            for i, genotype in enumerate(genotypes):
                means = [np.mean(geno_weights[genotype][f]) for f in feature_names]
                sems = [stats.sem(geno_weights[genotype][f]) if len(geno_weights[genotype][f]) > 1 else 0
                       for f in feature_names]

                ax.bar(x + i*width, means, width, yerr=sems,
                      label=genotype, color=get_genotype_color(genotype),
                      alpha=0.7, capsize=3)

            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_xlabel('Feature', fontsize=13, fontweight='bold')
            ax.set_ylabel('GLM Weight', fontsize=13, fontweight='bold')
            ax.set_title(f'{get_state_label(state)} - Cohort {cohort}',
                        fontsize=14, fontweight='bold', pad=10)
            ax.set_xticks(x + width * (len(genotypes)-1) / 2)
            ax.set_xticklabels(feature_names, fontsize=10)

            if state == 0 and col == 1:
                ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
            ax.grid(axis='y', alpha=0.3)

    plt.suptitle('GLM Weights by State and Genotype (Phase 1)',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_dir = Path('results/updated_figures/phase1')
    plt.savefig(output_dir / 'fig2_glm_weights.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_glm_weights.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_dir / 'fig2_glm_weights.png'}")


def figure3_transition_matrices():
    """Figure 3: Transition matrices by genotype (F cohort)."""
    print("\n" + "="*80)
    print("Generating Figure 3: Transition Matrices by Genotype")
    print("="*80)

    _, data_F = load_phase1_data()

    # Group by genotype
    geno_data = {}
    for r in data_F:
        g = r['genotype']
        if g not in geno_data:
            geno_data[g] = []
        geno_data[g].append(r)

    genotypes = [g for g in GENOTYPE_ORDER if g in geno_data]
    n_genotypes = len(genotypes)

    fig, axes = plt.subplots(1, n_genotypes, figsize=(6*n_genotypes, 6))
    if n_genotypes == 1:
        axes = [axes]

    for ax, genotype in zip(axes, genotypes):
        # Average transition matrix
        trans_matrices = [r['model'].transition_matrix for r in geno_data[genotype]]
        avg_trans = np.mean(trans_matrices, axis=0)

        # Plot heatmap
        im = ax.imshow(avg_trans, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

        # Add text
        for i in range(3):
            for j in range(3):
                color = 'white' if avg_trans[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{avg_trans[i, j]:.3f}',
                       ha="center", va="center", color=color,
                       fontsize=14, fontweight='bold')

        # Labels
        state_labels = [get_state_label(i) for i in range(3)]
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(state_labels, fontsize=12)
        ax.set_yticklabels(state_labels, fontsize=12)
        ax.set_xlabel('To State', fontsize=13, fontweight='bold')
        ax.set_ylabel('From State', fontsize=13, fontweight='bold')
        ax.set_title(f'{genotype}\n(n={len(trans_matrices)} animals)',
                    fontsize=15, fontweight='bold', pad=10)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Transition Probability', fontsize=11, fontweight='bold')

    plt.suptitle('State Transition Matrices by Genotype (F Cohort)',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = Path('results/updated_figures/phase1')
    plt.savefig(output_dir / 'fig3_transition_matrices.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_transition_matrices.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_dir / 'fig3_transition_matrices.png'}")


def figure4_state_characteristics():
    """Figure 4: State characteristics (accuracy, latency, side bias)."""
    print("\n" + "="*80)
    print("Generating Figure 4: State Characteristics by Genotype")
    print("="*80)

    data_W, data_F = load_phase1_data()

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))

    metrics = ['during_accuracy', 'during_latency_mean', 'during_latency_cv']
    ylabels = ['Accuracy (%)', 'Mean Latency (s)', 'Latency CV']
    titles = ['Accuracy', 'Response Latency', 'Latency Variability']

    for row, (metric, ylabel, title) in enumerate(zip(metrics, ylabels, titles)):
        for col, (cohort, data) in enumerate([('W', data_W), ('F', data_F)]):
            ax = axes[row, col]

            # Collect metric by genotype and state
            geno_state_metric = {}

            for r in data:
                g = r['genotype']
                if g not in geno_state_metric:
                    geno_state_metric[g] = {0: [], 1: [], 2: []}

                for state in range(3):
                    if state in r['state_metrics']:
                        val = r['state_metrics'][state][metric]
                        if val is not None and not np.isnan(val):
                            if metric == 'during_accuracy':
                                val *= 100  # Convert to percentage
                            geno_state_metric[g][state].append(val)

            # Plot
            genotypes = [g for g in GENOTYPE_ORDER if g in geno_state_metric]
            x = np.arange(len(genotypes))
            width = 0.25

            for i, state in enumerate([0, 1, 2]):
                means = [np.mean(geno_state_metric[g][state]) if len(geno_state_metric[g][state]) > 0 else 0
                        for g in genotypes]
                sems = [stats.sem(geno_state_metric[g][state]) if len(geno_state_metric[g][state]) > 1 else 0
                       for g in genotypes]

                ax.bar(x + i*width, means, width, yerr=sems,
                      label=get_state_label(state), color=get_state_color(state),
                      alpha=0.8, capsize=4)

            ax.set_xlabel('Genotype', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(f'{title} - Cohort {cohort}', fontsize=14, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(genotypes, rotation=45, ha='right', fontsize=11)
            if row == 0 and col == 1:
                ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)

            if metric == 'during_accuracy':
                ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('State Characteristics by Genotype (Phase 1)',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_dir = Path('results/updated_figures/phase1')
    plt.savefig(output_dir / 'fig4_state_characteristics.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_state_characteristics.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_dir / 'fig4_state_characteristics.png'}")


def figure5_cross_cohort_comparison():
    """Figure 5: Cross-cohort comparison (B6 and C3H x B6)."""
    print("\n" + "="*80)
    print("Generating Figure 5: Cross-Cohort Comparison")
    print("="*80)

    data_W, data_F = load_phase1_data()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel A: Engaged state occupancy
    ax = axes[0]

    groups = {
        'B6 (W)': [r for r in data_W if r['genotype'] == 'B6'],
        'C3H x B6 (W)': [r for r in data_W if r['genotype'] == 'C3H x B6'],
        'B6 (F)': [r for r in data_F if r['genotype'] == 'B6']
    }

    engaged_pct = {}
    for label, group_data in groups.items():
        pcts = []
        for r in group_data:
            traj = r['trajectory_df']
            total = traj['bout_length'].sum()
            engaged_trials = traj[traj['state'] == 0]['bout_length'].sum()
            pcts.append((engaged_trials / total * 100) if total > 0 else 0)
        engaged_pct[label] = pcts

    labels = list(engaged_pct.keys())
    means = [np.mean(engaged_pct[l]) for l in labels]
    sems = [stats.sem(engaged_pct[l]) for l in labels]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=sems, capsize=8, alpha=0.7)
    bars[0].set_color(get_genotype_color('B6'))
    bars[1].set_color(get_genotype_color('C3H x B6'))
    bars[2].set_color(get_genotype_color('B6'))

    ax.set_ylabel('% Time Engaged', fontsize=14, fontweight='bold')
    ax.set_title('Engaged State: Cross-Cohort', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])

    for i, l in enumerate(labels):
        ax.text(i, -8, f'n={len(engaged_pct[l])}', ha='center', fontsize=10)

    # Panel B: F cohort genotype comparison
    ax = axes[1]

    geno_engaged = {}
    for genotype in ['B6', 'A1D_Wt', 'A1D_Het', 'A1D_KO']:
        group = [r for r in data_F if r['genotype'] == genotype]
        if len(group) == 0:
            continue

        pcts = []
        for r in group:
            traj = r['trajectory_df']
            total = traj['bout_length'].sum()
            engaged_trials = traj[traj['state'] == 0]['bout_length'].sum()
            pcts.append((engaged_trials / total * 100) if total > 0 else 0)
        geno_engaged[genotype] = pcts

    genotypes = list(geno_engaged.keys())
    means = [np.mean(geno_engaged[g]) for g in genotypes]
    sems = [stats.sem(geno_engaged[g]) for g in genotypes]

    x = np.arange(len(genotypes))
    bars = ax.bar(x, means, yerr=sems, capsize=8, alpha=0.7)
    for bar, genotype in zip(bars, genotypes):
        bar.set_color(get_genotype_color(genotype))

    ax.set_ylabel('% Time Engaged', fontsize=14, fontweight='bold')
    ax.set_title('Engaged State: F Cohort', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(genotypes, rotation=45, ha='right', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])

    for i, g in enumerate(genotypes):
        ax.text(i, -8, f'n={len(geno_engaged[g])}', ha='center', fontsize=10)

    # Panel C: All F cohort genotypes - all states
    ax = axes[2]

    geno_states = {}
    for genotype in ['B6', 'A1D_Wt', 'A1D_Het', 'A1D_KO']:
        group = [r for r in data_F if r['genotype'] == genotype]
        if len(group) == 0:
            continue

        geno_states[genotype] = {0: [], 1: [], 2: []}
        for r in group:
            traj = r['trajectory_df']
            total = traj['bout_length'].sum()
            for state in range(3):
                state_trials = traj[traj['state'] == state]['bout_length'].sum()
                geno_states[genotype][state].append((state_trials / total * 100) if total > 0 else 0)

    genotypes = list(geno_states.keys())
    x = np.arange(len(genotypes))
    width = 0.25

    for i, state in enumerate([0, 1, 2]):
        means = [np.mean(geno_states[g][state]) for g in genotypes]
        sems = [stats.sem(geno_states[g][state]) for g in genotypes]

        ax.bar(x + i*width, means, width, yerr=sems,
              label=get_state_label(state), color=get_state_color(state),
              alpha=0.8, capsize=4)

    ax.set_ylabel('% Time in State', fontsize=14, fontweight='bold')
    ax.set_title('All States: F Cohort', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(genotypes, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])

    plt.suptitle('Cross-Cohort and Genotype Comparisons',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir = Path('results/updated_figures/combined')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'fig5_cross_cohort.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_cross_cohort.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_dir / 'fig5_cross_cohort.png'}")


def main():
    print("\n" + "="*80)
    print("REGENERATING ALL FIGURES WITH NEW LABELS")
    print("="*80)
    print("\nNew genotype labels:")
    print("  +/+ → A1D_Wt")
    print("  +/- → A1D_Het")
    print("  -/- → A1D_KO")
    print("  +   → B6")
    print("  -   → C3H x B6")
    print("\nState labels:")
    print("  State 0 → Engaged (high accuracy)")
    print("  State 1 → Biased (side preference)")
    print("  State 2 → Lapsed (disengaged)")
    print("="*80)

    # Generate all figures
    figure1_state_occupancy()
    figure2_glm_weights()
    figure3_transition_matrices()
    figure4_state_characteristics()
    figure5_cross_cohort_comparison()

    print("\n" + "="*80)
    print("✓ ALL FIGURES REGENERATED SUCCESSFULLY!")
    print("="*80)
    print("\nOutput directories:")
    print("  Phase 1: results/updated_figures/phase1/")
    print("  Combined: results/updated_figures/combined/")
    print("\nGenerated 5 figures (PNG + PDF)")


if __name__ == '__main__':
    main()
