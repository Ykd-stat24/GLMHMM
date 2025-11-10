"""
Comprehensive Figure Regenerator - All Phases
==============================================

Regenerates ALL key figures from Phase 1 and Phase 2 with updated labels and colors.

Run this script locally after pulling the latest code:
    python regenerate_all_figures_locally.py

New genotype labels:
- +/+ → A1D_Wt (yellow)
- +/- → A1D_Het (blue)
- -/- → A1D_KO (maroon)
- +   → B6 (red)
- -   → C3H x B6 (black)

State labels:
- State 0 → Engaged (high accuracy)
- State 1 → Biased (side preference)
- State 2 → Lapsed (disengaged)
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
sys.path.insert(0, str(Path(__file__).parent))
from genotype_labels import (
    relabel_genotype, get_state_label, get_state_color, get_genotype_color,
    GENOTYPE_ORDER, GENOTYPE_COLORS, STATE_COLORS
)

plt.style.use('seaborn-v0_8-whitegrid')


class ComprehensiveFigureRegenerator:
    """Regenerate all figures from Phase 1 and Phase 2."""

    def __init__(self):
        self.results_dir = Path('results')
        self.phase1_dir = self.results_dir / 'phase1_non_reversal'
        self.phase2_dir = self.results_dir / 'phase2_reversal' / 'models'
        self.output_dir = self.results_dir / 'regenerated_figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create output subdirectories
        (self.output_dir / 'phase1').mkdir(exist_ok=True)
        (self.output_dir / 'phase2').mkdir(exist_ok=True)
        (self.output_dir / 'combined').mkdir(exist_ok=True)

    def load_phase1_data(self):
        """Load Phase 1 data."""
        print("\n" + "="*80)
        print("Loading Phase 1 Data...")
        print("="*80)

        data_W, data_F = [], []

        for pkl_file in self.phase1_dir.glob('*_model.pkl'):
            try:
                with open(pkl_file, 'rb') as f:
                    r = pickle.load(f)
                    r['genotype'] = relabel_genotype(r['genotype'])

                    if r['cohort'] == 'W':
                        data_W.append(r)
                    else:
                        data_F.append(r)
            except Exception as e:
                print(f"  Warning: Could not load {pkl_file.name}: {e}")

        print(f"  W Cohort: {len(data_W)} animals")
        print(f"  F Cohort: {len(data_F)} animals")

        return data_W, data_F

    def load_phase2_data(self):
        """Load Phase 2 data."""
        print("\n" + "="*80)
        print("Loading Phase 2 Data...")
        print("="*80)

        data_W, data_F = [], []

        if not self.phase2_dir.exists():
            print(f"  Phase 2 directory not found: {self.phase2_dir}")
            return data_W, data_F

        for pkl_file in self.phase2_dir.glob('*_reversal.pkl'):
            try:
                with open(pkl_file, 'rb') as f:
                    r = pickle.load(f)
                    r['genotype'] = relabel_genotype(r['genotype'])

                    if r['cohort'] == 'W':
                        data_W.append(r)
                    else:
                        data_F.append(r)
            except Exception as e:
                print(f"  Warning: Could not load {pkl_file.name}: {e}")

        print(f"  W Cohort: {len(data_W)} animals")
        print(f"  F Cohort: {len(data_F)} animals")

        return data_W, data_F

    # ==================== PHASE 1 FIGURES ====================

    def phase1_fig1_state_occupancy(self, data_W, data_F):
        """Phase 1 Figure 1: State occupancy by genotype."""
        print("\nGenerating Phase 1 Fig 1: State Occupancy...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        for ax, (cohort, data) in zip(axes, [('W', data_W), ('F', data_F)]):
            geno_states = {}

            for r in data:
                g = r['genotype']
                if g not in geno_states:
                    geno_states[g] = {0: [], 1: [], 2: []}

                traj = r['trajectory_df']
                total = traj['bout_length'].sum()

                for state in range(3):
                    state_trials = traj[traj['state'] == state]['bout_length'].sum()
                    pct = (state_trials / total * 100) if total > 0 else 0
                    geno_states[g][state].append(pct)

            genotypes = [g for g in GENOTYPE_ORDER if g in geno_states]
            x = np.arange(len(genotypes))
            width = 0.25

            for i, state in enumerate([0, 1, 2]):
                means = [np.mean(geno_states[g][state]) for g in genotypes]
                sems = [stats.sem(geno_states[g][state]) if len(geno_states[g][state]) > 1 else 0
                        for g in genotypes]

                ax.bar(x + i*width, means, width, yerr=sems,
                       label=get_state_label(state), color=get_state_color(state),
                       alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)

            ax.set_xlabel('Genotype', fontsize=14, fontweight='bold')
            ax.set_ylabel('% Time in State', fontsize=14, fontweight='bold')
            ax.set_title(f'Cohort {cohort}', fontsize=16, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(genotypes, rotation=45, ha='right', fontsize=12)
            ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 100])

            for i, g in enumerate(genotypes):
                n = len(geno_states[g][0])
                ax.text(i, -10, f'n={n}', ha='center', fontsize=11, fontweight='bold')

        plt.suptitle('State Occupancy by Genotype (Phase 1: Non-Reversal)',
                     fontsize=18, fontweight='bold')
        plt.tight_layout()

        self.save_figure('phase1/fig1_state_occupancy')
        print(f"  ✓ Saved")

    def phase1_fig2_glm_weights(self, data_W, data_F):
        """Phase 1 Figure 2: GLM weights by state and genotype."""
        print("\nGenerating Phase 1 Fig 2: GLM Weights...")

        fig, axes = plt.subplots(3, 2, figsize=(18, 20))

        feature_names = ['bias', 'prev\nchoice', 'wsls', 'session\nprog',
                        'side\nbias', 'task\nstage', 'cum\nexp']

        for state in range(3):
            for col, (cohort, data) in enumerate([('W', data_W), ('F', data_F)]):
                ax = axes[state, col]

                geno_weights = {}

                for r in data:
                    g = r['genotype']
                    if g not in geno_weights:
                        geno_weights[g] = {f: [] for f in feature_names}

                    weights = r['model'].glm_weights[state]
                    for i, fname in enumerate(feature_names):
                        if i < len(weights):
                            geno_weights[g][fname].append(weights[i])

                genotypes = [g for g in GENOTYPE_ORDER if g in geno_weights]
                x = np.arange(len(feature_names))
                width = 0.8 / len(genotypes) if len(genotypes) > 0 else 0.2

                for i, genotype in enumerate(genotypes):
                    means = [np.mean(geno_weights[genotype][f]) for f in feature_names]
                    sems = [stats.sem(geno_weights[genotype][f]) if len(geno_weights[genotype][f]) > 1 else 0
                           for f in feature_names]

                    ax.bar(x + i*width, means, width, yerr=sems,
                          label=genotype, color=get_genotype_color(genotype),
                          alpha=0.7, capsize=3, edgecolor='black', linewidth=1)

                ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
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

        self.save_figure('phase1/fig2_glm_weights')
        print(f"  ✓ Saved")

    def phase1_fig3_transition_matrices(self, data_F):
        """Phase 1 Figure 3: Transition matrices by genotype (F cohort)."""
        print("\nGenerating Phase 1 Fig 3: Transition Matrices...")

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
            trans_matrices = [r['model'].transition_matrix for r in geno_data[genotype]]
            avg_trans = np.mean(trans_matrices, axis=0)

            im = ax.imshow(avg_trans, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

            for i in range(3):
                for j in range(3):
                    color = 'white' if avg_trans[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{avg_trans[i, j]:.3f}',
                           ha="center", va="center", color=color,
                           fontsize=14, fontweight='bold')

            state_labels = [get_state_label(i) for i in range(3)]
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(state_labels, fontsize=12, fontweight='bold')
            ax.set_yticklabels(state_labels, fontsize=12, fontweight='bold')
            ax.set_xlabel('To State', fontsize=13, fontweight='bold')
            ax.set_ylabel('From State', fontsize=13, fontweight='bold')
            ax.set_title(f'{genotype}\n(n={len(trans_matrices)} animals)',
                        fontsize=15, fontweight='bold', pad=10,
                        color=get_genotype_color(genotype))

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Transition Probability', fontsize=12, fontweight='bold')

        plt.suptitle('State Transition Matrices by Genotype (F Cohort)',
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()

        self.save_figure('phase1/fig3_transition_matrices')
        print(f"  ✓ Saved")

    def phase1_fig4_state_characteristics(self, data_W, data_F):
        """Phase 1 Figure 4: State characteristics."""
        print("\nGenerating Phase 1 Fig 4: State Characteristics...")

        fig, axes = plt.subplots(3, 2, figsize=(16, 18))

        metrics = ['during_accuracy', 'during_latency_mean', 'during_latency_cv']
        ylabels = ['Accuracy (%)', 'Mean Latency (s)', 'Latency CV']
        titles = ['Accuracy by State', 'Response Latency by State', 'Latency Variability by State']

        for row, (metric, ylabel, title) in enumerate(zip(metrics, ylabels, titles)):
            for col, (cohort, data) in enumerate([('W', data_W), ('F', data_F)]):
                ax = axes[row, col]

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
                                    val *= 100
                                geno_state_metric[g][state].append(val)

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
                          alpha=0.8, capsize=4, edgecolor='black', linewidth=1.5)

                ax.set_xlabel('Genotype', fontsize=13, fontweight='bold')
                ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
                ax.set_title(f'{title} - Cohort {cohort}', fontsize=14, fontweight='bold')
                ax.set_xticks(x + width)
                ax.set_xticklabels(genotypes, rotation=45, ha='right', fontsize=11)
                if row == 0 and col == 1:
                    ax.legend(fontsize=11, framealpha=0.95)
                ax.grid(axis='y', alpha=0.3)

                if metric == 'during_accuracy':
                    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=2)

        plt.suptitle('State Characteristics by Genotype (Phase 1)',
                     fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()

        self.save_figure('phase1/fig4_state_characteristics')
        print(f"  ✓ Saved")

    # ==================== PHASE 2 FIGURES ====================

    def phase2_fig1_state_occupancy_reversal(self, data_W, data_F):
        """Phase 2 Figure 1: State occupancy during reversal."""
        print("\nGenerating Phase 2 Fig 1: State Occupancy (Reversal)...")

        if len(data_W) == 0 and len(data_F) == 0:
            print("  ⚠ No Phase 2 data available, skipping")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        for ax, (cohort, data) in zip(axes, [('W', data_W), ('F', data_F)]):
            if len(data) == 0:
                ax.text(0.5, 0.5, f'No Phase 2 data for Cohort {cohort}',
                       ha='center', va='center', fontsize=14)
                ax.set_title(f'Cohort {cohort}', fontsize=16, fontweight='bold')
                continue

            geno_states = {}

            for r in data:
                g = r['genotype']
                if g not in geno_states:
                    geno_states[g] = {0: [], 1: [], 2: []}

                traj = r['trajectory_df']
                total = traj['bout_length'].sum()

                for state in range(3):
                    state_trials = traj[traj['state'] == state]['bout_length'].sum()
                    pct = (state_trials / total * 100) if total > 0 else 0
                    geno_states[g][state].append(pct)

            genotypes = [g for g in GENOTYPE_ORDER if g in geno_states]
            x = np.arange(len(genotypes))
            width = 0.25

            for i, state in enumerate([0, 1, 2]):
                means = [np.mean(geno_states[g][state]) for g in genotypes]
                sems = [stats.sem(geno_states[g][state]) if len(geno_states[g][state]) > 1 else 0
                        for g in genotypes]

                ax.bar(x + i*width, means, width, yerr=sems,
                       label=get_state_label(state), color=get_state_color(state),
                       alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)

            ax.set_xlabel('Genotype', fontsize=14, fontweight='bold')
            ax.set_ylabel('% Time in State', fontsize=14, fontweight='bold')
            ax.set_title(f'Cohort {cohort}', fontsize=16, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(genotypes, rotation=45, ha='right', fontsize=12)
            ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 100])

            for i, g in enumerate(genotypes):
                if len(geno_states[g][0]) > 0:
                    n = len(geno_states[g][0])
                    ax.text(i, -10, f'n={n}', ha='center', fontsize=11, fontweight='bold')

        plt.suptitle('State Occupancy by Genotype (Phase 2: Reversal)',
                     fontsize=18, fontweight='bold')
        plt.tight_layout()

        self.save_figure('phase2/fig1_state_occupancy_reversal')
        print(f"  ✓ Saved")

    def phase2_fig2_reversal_performance(self, data_W, data_F):
        """Phase 2 Figure 2: Overall reversal performance by genotype."""
        print("\nGenerating Phase 2 Fig 2: Reversal Performance...")

        if len(data_W) == 0 and len(data_F) == 0:
            print("  ⚠ No Phase 2 data available, skipping")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        for ax, (cohort, data) in zip(axes, [('W', data_W), ('F', data_F)]):
            if len(data) == 0:
                ax.text(0.5, 0.5, f'No Phase 2 data for Cohort {cohort}',
                       ha='center', va='center', fontsize=14)
                ax.set_title(f'Cohort {cohort}', fontsize=16, fontweight='bold')
                continue

            geno_acc = {}

            for r in data:
                g = r['genotype']
                if g not in geno_acc:
                    geno_acc[g] = []

                # Calculate average accuracy from trajectory
                if 'trajectory_df' in r:
                    avg_acc = r['trajectory_df']['during_accuracy'].mean()
                    if not np.isnan(avg_acc):
                        geno_acc[g].append(avg_acc * 100)
                elif 'state_metrics' in r:
                    # Try to get from state_metrics
                    try:
                        if isinstance(r['state_metrics'], dict):
                            accs = [m.get('during_accuracy') for m in r['state_metrics'].values()
                                   if m.get('during_accuracy') is not None and not np.isnan(m.get('during_accuracy', np.nan))]
                            if len(accs) > 0:
                                geno_acc[g].append(np.mean(accs) * 100)
                    except:
                        pass

            genotypes = [g for g in GENOTYPE_ORDER if g in geno_acc]
            means = [np.mean(geno_acc[g]) for g in genotypes]
            sems = [stats.sem(geno_acc[g]) for g in genotypes]

            x = np.arange(len(genotypes))
            bars = ax.bar(x, means, yerr=sems, capsize=8, alpha=0.7, edgecolor='black', linewidth=2)

            for bar, genotype in zip(bars, genotypes):
                bar.set_color(get_genotype_color(genotype))

            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Chance')
            ax.set_xlabel('Genotype', fontsize=14, fontweight='bold')
            ax.set_ylabel('Overall Accuracy (%)', fontsize=14, fontweight='bold')
            ax.set_title(f'Cohort {cohort}', fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(genotypes, rotation=45, ha='right', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([40, 100])

            for i, g in enumerate(genotypes):
                n = len(geno_acc[g])
                ax.text(i, 42, f'n={n}', ha='center', fontsize=11, fontweight='bold')

        plt.suptitle('Reversal Learning Performance by Genotype (Phase 2)',
                     fontsize=18, fontweight='bold')
        plt.tight_layout()

        self.save_figure('phase2/fig2_reversal_performance')
        print(f"  ✓ Saved")

    # ==================== COMBINED FIGURES ====================

    def combined_fig1_cross_cohort_comparison(self, data_W_p1, data_F_p1):
        """Combined: Cross-cohort comparison."""
        print("\nGenerating Combined Fig 1: Cross-Cohort Comparison...")

        fig, axes = plt.subplots(1, 3, figsize=(22, 7))

        # Panel A: B6 vs C3H x B6 comparison
        ax = axes[0]

        groups = {
            'B6 (W)': [r for r in data_W_p1 if r['genotype'] == 'B6'],
            'C3H x B6 (W)': [r for r in data_W_p1 if r['genotype'] == 'C3H x B6'],
            'B6 (F)': [r for r in data_F_p1 if r['genotype'] == 'B6']
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
        bars = ax.bar(x, means, yerr=sems, capsize=8, alpha=0.7, edgecolor='black', linewidth=2)
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
            ax.text(i, -10, f'n={len(engaged_pct[l])}', ha='center', fontsize=11, fontweight='bold')

        # Panel B: F cohort genotypes
        ax = axes[1]

        geno_engaged = {}
        for genotype in ['B6', 'A1D_Wt', 'A1D_Het', 'A1D_KO']:
            group = [r for r in data_F_p1 if r['genotype'] == genotype]
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
        bars = ax.bar(x, means, yerr=sems, capsize=8, alpha=0.7, edgecolor='black', linewidth=2)
        for bar, genotype in zip(bars, genotypes):
            bar.set_color(get_genotype_color(genotype))

        ax.set_ylabel('% Time Engaged', fontsize=14, fontweight='bold')
        ax.set_title('Engaged State: F Cohort', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(genotypes, rotation=45, ha='right', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])

        for i, g in enumerate(genotypes):
            ax.text(i, -10, f'n={len(geno_engaged[g])}', ha='center', fontsize=11, fontweight='bold')

        # Panel C: All states F cohort
        ax = axes[2]

        geno_states = {}
        for genotype in ['B6', 'A1D_Wt', 'A1D_Het', 'A1D_KO']:
            group = [r for r in data_F_p1 if r['genotype'] == genotype]
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
                  alpha=0.8, capsize=4, edgecolor='black', linewidth=1.5)

        ax.set_ylabel('% Time in State', fontsize=14, fontweight='bold')
        ax.set_title('All States: F Cohort', fontsize=16, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(genotypes, rotation=45, ha='right', fontsize=12)
        ax.legend(fontsize=12, framealpha=0.95)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])

        plt.suptitle('Cross-Cohort and Genotype Comparisons',
                     fontsize=18, fontweight='bold')
        plt.tight_layout()

        self.save_figure('combined/fig1_cross_cohort_comparison')
        print(f"  ✓ Saved")

    def save_figure(self, filename):
        """Save figure as PNG and PDF."""
        output_path = self.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
        plt.close()

    def run_all(self):
        """Run all figure regeneration."""
        print("\n" + "="*80)
        print("COMPREHENSIVE FIGURE REGENERATION")
        print("="*80)
        print("\nNew genotype colors:")
        print("  B6 (+)        - RED")
        print("  C3H x B6 (-)  - BLACK")
        print("  A1D_Wt (+/+)  - YELLOW")
        print("  A1D_Het (+/-) - BLUE")
        print("  A1D_KO (-/-)  - MAROON")
        print("\nState labels:")
        print("  State 0 → Engaged (high accuracy)")
        print("  State 1 → Biased (side preference)")
        print("  State 2 → Lapsed (disengaged)")
        print("="*80)

        # Load all data
        data_W_p1, data_F_p1 = self.load_phase1_data()
        data_W_p2, data_F_p2 = self.load_phase2_data()

        # Phase 1 figures
        print("\n" + "="*80)
        print("PHASE 1 FIGURES")
        print("="*80)
        self.phase1_fig1_state_occupancy(data_W_p1, data_F_p1)
        self.phase1_fig2_glm_weights(data_W_p1, data_F_p1)
        self.phase1_fig3_transition_matrices(data_F_p1)
        self.phase1_fig4_state_characteristics(data_W_p1, data_F_p1)

        # Phase 2 figures
        print("\n" + "="*80)
        print("PHASE 2 FIGURES")
        print("="*80)
        self.phase2_fig1_state_occupancy_reversal(data_W_p2, data_F_p2)
        self.phase2_fig2_reversal_performance(data_W_p2, data_F_p2)

        # Combined figures
        print("\n" + "="*80)
        print("COMBINED FIGURES")
        print("="*80)
        self.combined_fig1_cross_cohort_comparison(data_W_p1, data_F_p1)

        # Summary
        print("\n" + "="*80)
        print("✓ ALL FIGURES REGENERATED SUCCESSFULLY!")
        print("="*80)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nGenerated figures:")
        print("  Phase 1: 4 figures (8 files)")
        print("  Phase 2: 2 figures (4 files)")
        print("  Combined: 1 figure (2 files)")
        print("\nAll figures saved as PNG (high-res) and PDF (publication-ready)")


def main():
    regenerator = ComprehensiveFigureRegenerator()
    regenerator.run_all()


if __name__ == '__main__':
    main()
