"""
Regenerate All Figures with New Genotype and State Labels
==========================================================

This script regenerates all key figures from Phase 1 and Phase 2 analyses
with updated genotype labels and properly labeled states.

New Labels:
- +/+ → A1D_Wt
- +/- → A1D_Het
- -/- → A1D_KO
- + → B6
- - → C3H x B6

States: Engaged, Biased, Lapsed (with descriptions)
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
    relabel_genotype, relabel_genotype_df,
    get_state_label, get_state_color, get_genotype_color,
    GENOTYPE_ORDER, STATE_COLORS, GENOTYPE_COLORS,
    sort_by_genotype_order
)
from state_validation import create_broad_state_categories
from glmhmm_utils import load_and_preprocess_session_data

plt.style.use('seaborn-v0_8-whitegrid')

class FigureRegenerator:
    """Regenerate all key figures with new labels."""

    def __init__(self):
        self.results_dir = Path('results')
        self.phase1_dir = self.results_dir / 'phase1_non_reversal'
        self.phase2_dir = self.results_dir / 'phase2_reversal'
        self.output_dir = self.results_dir / 'updated_figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / 'phase1').mkdir(exist_ok=True)
        (self.output_dir / 'phase2').mkdir(exist_ok=True)
        (self.output_dir / 'combined').mkdir(exist_ok=True)

    def load_phase1_data(self):
        """Load all Phase 1 data."""
        print("\n" + "="*80)
        print("Loading Phase 1 Data...")
        print("="*80)

        data = {'W': [], 'F': []}
        trials = {}

        for cohort in ['W', 'F']:
            print(f"\nLoading Cohort {cohort}...")
            data_file = f'{cohort} LD Data 11.08 All_processed.csv'
            trials[cohort] = load_and_preprocess_session_data(data_file)

            for pkl_file in self.phase1_dir.glob(f'*_cohort{cohort}_model.pkl'):
                with open(pkl_file, 'rb') as f:
                    result = pickle.load(f)
                    result['genotype'] = relabel_genotype(result['genotype'])
                    result['broad_categories'] = create_broad_state_categories(result['validated_labels'])
                    data[cohort].append(result)

            print(f"  Loaded {len(data[cohort])} animals")

        return data, trials

    def load_phase2_data(self):
        """Load all Phase 2 data."""
        print("\n" + "="*80)
        print("Loading Phase 2 Data...")
        print("="*80)

        data = {'W': [], 'F': []}

        for cohort in ['W', 'F']:
            print(f"\nLoading Cohort {cohort} Phase 2...")
            phase2_cohort_dir = self.phase2_dir / f'cohort{cohort}'

            if not phase2_cohort_dir.exists():
                print(f"  Warning: {phase2_cohort_dir} not found")
                continue

            for pkl_file in phase2_cohort_dir.glob('*_model.pkl'):
                try:
                    with open(pkl_file, 'rb') as f:
                        result = pickle.load(f)
                        result['genotype'] = relabel_genotype(result['genotype'])
                        data[cohort].append(result)
                except Exception as e:
                    print(f"  Error loading {pkl_file.name}: {e}")

            print(f"  Loaded {len(data[cohort])} animals")

        return data

    def phase1_fig1_state_occupancy_by_genotype(self, data_W, data_F):
        """Figure 1: State occupancy by genotype for both cohorts."""
        print("\nGenerating Phase 1 Figure 1: State Occupancy by Genotype...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax, (cohort, data) in zip(axes, [('W', data_W), ('F', data_F)]):
            # Aggregate state occupancy by genotype
            geno_states = {}

            for r in data:
                g = r['genotype']
                if g not in geno_states:
                    geno_states[g] = {0: [], 1: [], 2: []}

                # Calculate state occupancy from trajectory_df
                trajectory_df = r['trajectory_df']
                total_trials = trajectory_df['bout_length'].sum()

                for s in range(3):
                    state_trials = trajectory_df[trajectory_df['state'] == s]['bout_length'].sum()
                    occupancy = (state_trials / total_trials) * 100 if total_trials > 0 else 0
                    geno_states[g][s].append(occupancy)

            # Create grouped bar plot
            genotypes = [g for g in GENOTYPE_ORDER if g in geno_states]
            x = np.arange(len(genotypes))
            width = 0.25

            for i, state in enumerate([0, 1, 2]):
                state_name = get_state_label(state)
                means = [np.mean(geno_states[g][state]) for g in genotypes]
                sems = [stats.sem(geno_states[g][state]) for g in genotypes]

                ax.bar(x + i*width, means, width, yerr=sems,
                       label=state_name, color=get_state_color(state),
                       alpha=0.8, capsize=5)

            ax.set_xlabel('Genotype', fontsize=14, fontweight='bold')
            ax.set_ylabel('% Time in State', fontsize=14, fontweight='bold')
            ax.set_title(f'Cohort {cohort}', fontsize=16, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(genotypes, rotation=45, ha='right')
            ax.legend(fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 100])

        plt.suptitle('State Occupancy by Genotype (Phase 1)',
                     fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_file = self.output_dir / 'phase1' / 'fig1_state_occupancy_by_genotype.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

    def phase1_fig2_learning_trajectories(self, data_W, data_F, trials_W, trials_F):
        """Figure 2: Learning trajectories by genotype."""
        print("\nGenerating Phase 1 Figure 2: Learning Trajectories...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax, (cohort, data, trials) in zip(axes,
                                                [('W', data_W, trials_W),
                                                 ('F', data_F, trials_F)]):
            # Calculate performance by session for each animal
            for r in data:
                animal_id = r['animal_id']
                genotype = r['genotype']

                animal_trials = trials[trials['animal_id'] == animal_id].copy()
                if len(animal_trials) == 0:
                    continue

                # Group by session and calculate accuracy
                session_acc = animal_trials.groupby('session').apply(
                    lambda x: x['correct'].mean() * 100
                ).reset_index()
                session_acc.columns = ['session', 'accuracy']

                # Plot with genotype color
                ax.plot(session_acc['session'], session_acc['accuracy'],
                       alpha=0.3, linewidth=1,
                       color=get_genotype_color(genotype))

            # Plot genotype averages
            geno_session_data = {}
            for r in data:
                animal_id = r['animal_id']
                genotype = r['genotype']

                if genotype not in geno_session_data:
                    geno_session_data[genotype] = {}

                animal_trials = trials[trials['animal_id'] == animal_id].copy()
                session_acc = animal_trials.groupby('session')['correct'].mean() * 100

                for sess, acc in session_acc.items():
                    if sess not in geno_session_data[genotype]:
                        geno_session_data[genotype][sess] = []
                    geno_session_data[genotype][sess].append(acc)

            # Plot averages
            for genotype in [g for g in GENOTYPE_ORDER if g in geno_session_data]:
                sessions = sorted(geno_session_data[genotype].keys())
                means = [np.mean(geno_session_data[genotype][s]) for s in sessions]

                ax.plot(sessions, means, linewidth=3,
                       label=genotype, color=get_genotype_color(genotype))

            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
            ax.set_xlabel('Session Number', fontsize=14, fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
            ax.set_title(f'Cohort {cohort}', fontsize=16, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3)
            ax.set_ylim([40, 100])

        plt.suptitle('Learning Trajectories by Genotype (Phase 1)',
                     fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_file = self.output_dir / 'phase1' / 'fig2_learning_trajectories.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

    def phase1_fig3_glm_weights_by_genotype(self, data_W, data_F):
        """Figure 3: GLM weights comparison by genotype."""
        print("\nGenerating Phase 1 Figure 3: GLM Weights by Genotype...")

        fig, axes = plt.subplots(3, 2, figsize=(16, 18))

        # Feature names (excluding stimulus)
        feature_names = ['bias', 'prev_choice', 'wsls', 'session_prog',
                        'side_bias', 'task_stage', 'cum_exp']

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

                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
                ax.set_ylabel('GLM Weight', fontsize=12, fontweight='bold')
                ax.set_title(f'{get_state_label(state)} - Cohort {cohort}',
                            fontsize=14, fontweight='bold')
                ax.set_xticks(x + width * (len(genotypes)-1) / 2)
                ax.set_xticklabels(feature_names, rotation=45, ha='right')
                if state == 0:
                    ax.legend(fontsize=10, loc='upper right')
                ax.grid(axis='y', alpha=0.3)

        plt.suptitle('GLM Weights by State and Genotype (Phase 1)',
                     fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_file = self.output_dir / 'phase1' / 'fig3_glm_weights_by_genotype.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

    def phase1_fig4_transition_matrices_by_genotype(self, data_F):
        """Figure 4: Transition matrices for F cohort by genotype."""
        print("\nGenerating Phase 1 Figure 4: Transition Matrices by Genotype...")

        genotypes_F = sorted(set([r['genotype'] for r in data_F]))
        n_genotypes = len(genotypes_F)

        fig, axes = plt.subplots(1, n_genotypes, figsize=(5*n_genotypes, 5))
        if n_genotypes == 1:
            axes = [axes]

        for ax, genotype in zip(axes, genotypes_F):
            # Average transition matrix for this genotype
            trans_matrices = []
            for r in data_F:
                if r['genotype'] == genotype:
                    trans_matrices.append(r['model'].transition_matrix)

            if len(trans_matrices) == 0:
                continue

            avg_trans = np.mean(trans_matrices, axis=0)

            # Plot heatmap
            im = ax.imshow(avg_trans, cmap='Blues', vmin=0, vmax=1, aspect='auto')

            # Add text annotations
            for i in range(3):
                for j in range(3):
                    text = ax.text(j, i, f'{avg_trans[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=12)

            # Labels
            state_labels = [get_state_label(i) for i in range(3)]
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(state_labels, fontsize=11)
            ax.set_yticklabels(state_labels, fontsize=11)
            ax.set_xlabel('To State', fontsize=12, fontweight='bold')
            ax.set_ylabel('From State', fontsize=12, fontweight='bold')
            ax.set_title(f'{genotype}\n(n={len(trans_matrices)} animals)',
                        fontsize=14, fontweight='bold')

            # Colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle('State Transition Matrices by Genotype (F Cohort)',
                     fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_file = self.output_dir / 'phase1' / 'fig4_transition_matrices_F.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

    def phase2_fig1_reversal_performance(self, data_W, data_F):
        """Figure 1: Reversal learning performance by genotype."""
        print("\nGenerating Phase 2 Figure 1: Reversal Performance...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax, (cohort, data) in zip(axes, [('W', data_W), ('F', data_F)]):
            if len(data) == 0:
                ax.text(0.5, 0.5, f'No Phase 2 data for Cohort {cohort}',
                       ha='center', va='center', fontsize=14)
                ax.set_title(f'Cohort {cohort}', fontsize=16, fontweight='bold')
                continue

            # Calculate reversal metrics by genotype
            geno_metrics = {}

            for r in data:
                g = r['genotype']
                if g not in geno_metrics:
                    geno_metrics[g] = []

                # Calculate accuracy
                y_true = r['y']
                states = r['model'].viterbi(r['X'], r['y'])

                # Predict using state-specific GLMs
                correct_preds = 0
                for i, (x, y_true_i, state) in enumerate(zip(r['X'], y_true, states)):
                    from scipy.special import expit
                    logit = x @ r['model'].glm_weights[state] + r['model'].glm_intercepts[state]
                    y_pred = 1 if expit(logit) > 0.5 else 0
                    if y_pred == y_true_i:
                        correct_preds += 1

                accuracy = correct_preds / len(y_true) * 100
                geno_metrics[g].append(accuracy)

            # Plot
            genotypes = [g for g in GENOTYPE_ORDER if g in geno_metrics]
            means = [np.mean(geno_metrics[g]) for g in genotypes]
            sems = [stats.sem(geno_metrics[g]) for g in genotypes]

            x = np.arange(len(genotypes))
            bars = ax.bar(x, means, yerr=sems, capsize=8, alpha=0.7)

            # Color bars by genotype
            for bar, genotype in zip(bars, genotypes):
                bar.set_color(get_genotype_color(genotype))

            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
            ax.set_xlabel('Genotype', fontsize=14, fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
            ax.set_title(f'Cohort {cohort}', fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(genotypes, rotation=45, ha='right')
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([40, 100])

        plt.suptitle('Reversal Learning Performance by Genotype (Phase 2)',
                     fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_file = self.output_dir / 'phase2' / 'fig1_reversal_performance.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

    def phase2_fig2_state_dynamics_phase2(self, data_W, data_F):
        """Figure 2: State dynamics during reversal by genotype."""
        print("\nGenerating Phase 2 Figure 2: State Dynamics...")

        fig, axes = plt.subplots(3, 2, figsize=(16, 16))

        for col, (cohort, data) in enumerate([('W', data_W), ('F', data_F)]):
            if len(data) == 0:
                for row in range(3):
                    axes[row, col].text(0.5, 0.5, f'No data for Cohort {cohort}',
                                       ha='center', va='center')
                continue

            for state in range(3):
                ax = axes[state, col]

                # Calculate state occupancy over trials for each genotype
                geno_state_traj = {}

                for r in data:
                    g = r['genotype']
                    if g not in geno_state_traj:
                        geno_state_traj[g] = []

                    states = r['model'].viterbi(r['X'], r['y'])
                    # Binned state occupancy
                    n_bins = 20
                    bin_size = len(states) // n_bins
                    state_over_time = []

                    for i in range(n_bins):
                        start = i * bin_size
                        end = start + bin_size
                        if end > len(states):
                            end = len(states)
                        state_over_time.append(np.mean(states[start:end] == state) * 100)

                    geno_state_traj[g].append(state_over_time)

                # Plot
                for genotype in [g for g in GENOTYPE_ORDER if g in geno_state_traj]:
                    trajectories = np.array(geno_state_traj[genotype])
                    mean_traj = np.mean(trajectories, axis=0)
                    sem_traj = stats.sem(trajectories, axis=0)

                    x = np.arange(len(mean_traj))
                    ax.plot(x, mean_traj, linewidth=2.5, label=genotype,
                           color=get_genotype_color(genotype))
                    ax.fill_between(x, mean_traj - sem_traj, mean_traj + sem_traj,
                                   alpha=0.2, color=get_genotype_color(genotype))

                ax.set_xlabel('Trial Bin', fontsize=12, fontweight='bold')
                ax.set_ylabel('% in State', fontsize=12, fontweight='bold')
                ax.set_title(f'{get_state_label(state)} - Cohort {cohort}',
                            fontsize=14, fontweight='bold')
                if state == 0:
                    ax.legend(fontsize=10)
                ax.grid(alpha=0.3)
                ax.set_ylim([0, 100])

        plt.suptitle('State Dynamics During Reversal (Phase 2)',
                     fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_file = self.output_dir / 'phase2' / 'fig2_state_dynamics_reversal.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

    def combined_fig1_cross_cohort_comparison(self, data_W_p1, data_F_p1):
        """Combined figure: Cross-cohort genotype comparison."""
        print("\nGenerating Combined Figure 1: Cross-Cohort Comparison...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Panel A: B6 vs C3H x B6 comparison
        ax = axes[0]

        b6_W = [r for r in data_W_p1 if r['genotype'] == 'B6']
        c3h_W = [r for r in data_W_p1 if r['genotype'] == 'C3H x B6']

        b6_F = [r for r in data_F_p1 if r['genotype'] == 'B6']

        metrics = {}
        for label, group in [('B6 (W)', b6_W), ('C3H x B6 (W)', c3h_W), ('B6 (F)', b6_F)]:
            # Calculate engaged state occupancy
            engaged_pct = []
            for r in group:
                states = r['model'].viterbi(r['X'], r['y'])
                engaged_pct.append(np.mean(states == 0) * 100)
            metrics[label] = engaged_pct

        labels = list(metrics.keys())
        means = [np.mean(metrics[l]) for l in labels]
        sems = [stats.sem(metrics[l]) for l in labels]

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=sems, capsize=8, alpha=0.7)
        bars[0].set_color(get_genotype_color('B6'))
        bars[1].set_color(get_genotype_color('C3H x B6'))
        bars[2].set_color(get_genotype_color('B6'))

        ax.set_xlabel('Group', fontsize=14, fontweight='bold')
        ax.set_ylabel('% Time Engaged', fontsize=14, fontweight='bold')
        ax.set_title('Engaged State: Cross-Cohort Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])

        # Panel B: F cohort genotype comparison
        ax = axes[1]

        geno_engaged = {}
        for genotype in ['B6', 'A1D_Wt', 'A1D_Het', 'A1D_KO']:
            group = [r for r in data_F_p1 if r['genotype'] == genotype]
            if len(group) == 0:
                continue

            engaged_pct = []
            for r in group:
                states = r['model'].viterbi(r['X'], r['y'])
                engaged_pct.append(np.mean(states == 0) * 100)
            geno_engaged[genotype] = engaged_pct

        genotypes = list(geno_engaged.keys())
        means = [np.mean(geno_engaged[g]) for g in genotypes]
        sems = [stats.sem(geno_engaged[g]) for g in genotypes]

        x = np.arange(len(genotypes))
        bars = ax.bar(x, means, yerr=sems, capsize=8, alpha=0.7)

        for bar, genotype in zip(bars, genotypes):
            bar.set_color(get_genotype_color(genotype))

        ax.set_xlabel('Genotype', fontsize=14, fontweight='bold')
        ax.set_ylabel('% Time Engaged', fontsize=14, fontweight='bold')
        ax.set_title('Engaged State: F Cohort Genotypes', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(genotypes, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])

        plt.suptitle('Cross-Cohort and Genotype Comparisons',
                     fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_file = self.output_dir / 'combined' / 'fig1_cross_cohort_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

    def run_all(self):
        """Run all figure regeneration."""
        print("\n" + "="*80)
        print("REGENERATING ALL FIGURES WITH NEW LABELS")
        print("="*80)

        # Load data
        phase1_data, phase1_trials = self.load_phase1_data()
        phase2_data = self.load_phase2_data()

        # Phase 1 figures
        print("\n" + "="*80)
        print("Phase 1 Figures")
        print("="*80)
        self.phase1_fig1_state_occupancy_by_genotype(phase1_data['W'], phase1_data['F'])
        self.phase1_fig2_learning_trajectories(phase1_data['W'], phase1_data['F'],
                                               phase1_trials['W'], phase1_trials['F'])
        self.phase1_fig3_glm_weights_by_genotype(phase1_data['W'], phase1_data['F'])
        self.phase1_fig4_transition_matrices_by_genotype(phase1_data['F'])

        # Phase 2 figures
        print("\n" + "="*80)
        print("Phase 2 Figures")
        print("="*80)
        self.phase2_fig1_reversal_performance(phase2_data['W'], phase2_data['F'])
        self.phase2_fig2_state_dynamics_phase2(phase2_data['W'], phase2_data['F'])

        # Combined figures
        print("\n" + "="*80)
        print("Combined Figures")
        print("="*80)
        self.combined_fig1_cross_cohort_comparison(phase1_data['W'], phase1_data['F'])

        # Summary
        print("\n" + "="*80)
        print("✓ ALL FIGURES REGENERATED SUCCESSFULLY!")
        print("="*80)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nGenerated:")
        print("  Phase 1: 4 figures")
        print("  Phase 2: 2 figures")
        print("  Combined: 1 figure")
        print("\nAll figures saved as PNG and PDF")


def main():
    regenerator = FigureRegenerator()
    regenerator.run_all()


if __name__ == '__main__':
    main()
