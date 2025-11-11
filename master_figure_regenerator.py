"""
Master Figure Regenerator - Systematic Recreation and Improvement
==================================================================

Systematically recreates all Phase 1 and Phase 2 graphs with:
1. Correct genotype labels and consistent colors
2. Improved design for genotype and phase comparison
3. Clear state effect visualization
4. Removal of visual ambiguity

Structure:
- Batch 1: Phase 1 core figures (4 key figures with improvements)
- Batch 2: Phase 2 reversal figures (4 figures)
- Batch 3: Phase 2 combined cohorts figures
- Batch 4: Improved comparative figures (new)

Color Scheme:
- Genotypes: B6 (red), C3H x B6 (black), A1D_Wt (gold), A1D_Het (blue), A1D_KO (maroon)
- States: Engaged (green), Biased (orange), Lapsed (red)
- Phases: Phase 1 (solid), Phase 2 (hatched)
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
    GENOTYPE_MAP, GENOTYPE_ORDER, GENOTYPE_COLORS,
    STATE_LABELS, STATE_COLORS, STATE_DESCRIPTIONS,
    relabel_genotype, get_genotype_color, get_state_color
)
from state_validation import create_broad_state_categories
from glmhmm_utils import load_and_preprocess_session_data

# Configure plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class MasterFigureRegenerator:
    """Systematically regenerate all figures with improvements."""

    def __init__(self):
        self.base_dir = Path('/home/user/GLMHMM')
        self.results_dir = self.base_dir / 'results'
        self.phase1_dir = self.results_dir / 'phase1_non_reversal'
        self.phase2_dir = self.results_dir / 'phase2_reversal'
        self.output_base = self.results_dir / 'regenerated_comprehensive'

        # Create output directories
        self.output_base.mkdir(exist_ok=True)
        (self.output_base / 'phase1').mkdir(exist_ok=True)
        (self.output_base / 'phase2').mkdir(exist_ok=True)
        (self.output_base / 'combined').mkdir(exist_ok=True)
        (self.output_base / 'comparisons').mkdir(exist_ok=True)

        self.phase1_data = None
        self.phase2_data = None
        self.trials_data = None

    def load_all_data(self):
        """Load all Phase 1 and Phase 2 data."""
        print("\n" + "="*80)
        print("LOADING ALL DATA")
        print("="*80)

        self.phase1_data = {'W': [], 'F': []}
        self.phase2_data = {'W': [], 'F': []}
        self.trials_data = {'W': None, 'F': None}

        for cohort in ['W', 'F']:
            print(f"\n--- Cohort {cohort} ---")

            # Load trial data
            data_file = f'{cohort} LD Data 11.08 All_processed.csv'
            self.trials_data[cohort] = load_and_preprocess_session_data(data_file)
            print(f"Loaded {len(self.trials_data[cohort])} trials")

            # Load Phase 1 models
            print(f"Loading Phase 1 models...")
            for pkl_file in sorted(self.phase1_dir.glob(f'*_cohort{cohort}_model.pkl')):
                try:
                    with open(pkl_file, 'rb') as f:
                        result = pickle.load(f)
                        result['genotype'] = relabel_genotype(result['genotype'])
                        result['broad_categories'] = create_broad_state_categories(result['validated_labels'])
                        self.phase1_data[cohort].append(result)
                except Exception as e:
                    print(f"  Warning: Could not load {pkl_file.name}: {e}")

            print(f"  ✓ Loaded {len(self.phase1_data[cohort])} Phase 1 animals")

            # Load Phase 2 models if available
            phase2_cohort_dir = self.phase2_dir / f'cohort{cohort}'
            if phase2_cohort_dir.exists():
                print(f"Loading Phase 2 models...")
                for pkl_file in sorted(phase2_cohort_dir.glob('*_model.pkl')):
                    try:
                        with open(pkl_file, 'rb') as f:
                            result = pickle.load(f)
                            result['genotype'] = relabel_genotype(result['genotype'])
                            self.phase2_data[cohort].append(result)
                    except Exception as e:
                        print(f"  Warning: Could not load {pkl_file.name}: {e}")
                print(f"  ✓ Loaded {len(self.phase2_data[cohort])} Phase 2 animals")

    # ========== BATCH 1: PHASE 1 CORE FIGURES ==========

    def batch1_phase1_figures(self):
        """Generate Phase 1 core figures with improvements."""
        print("\n" + "="*80)
        print("BATCH 1: PHASE 1 CORE FIGURES")
        print("="*80)

        self.phase1_fig1_state_occupancy()
        self.phase1_fig2_state_characteristics()
        self.phase1_fig3_genotype_comparison()
        self.phase1_fig4_cross_cohort()

    def phase1_fig1_state_occupancy(self):
        """Fig 1: State occupancy by genotype and task, with clear genotype colors."""
        print("\nGenerating Phase 1 Fig 1: State Occupancy by Genotype...")

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        for col, cohort in enumerate(['W', 'F']):
            ax = axes[col]
            data = self.phase1_data[cohort]

            # Organize data by genotype and state
            genotype_order = [g for g in GENOTYPE_ORDER if any(r['genotype'] == g for r in data)]
            state_occupancy = {}

            for genotype in genotype_order:
                geno_data = [r for r in data if r['genotype'] == genotype]
                state_occ = np.zeros(3)

                for result in geno_data:
                    for state in range(3):
                        state_occ[state] += np.sum(result['model'].most_likely_states == state)

                state_occupancy[genotype] = state_occ / np.sum(state_occ) * 100

            # Create grouped bar plot
            x = np.arange(len(genotype_order))
            width = 0.25

            for state in range(3):
                state_label = STATE_LABELS[state]
                state_color = get_state_color(state)
                values = [state_occupancy[g][state] for g in genotype_order]

                ax.bar(x + state*width, values, width, label=state_label,
                       color=state_color, alpha=0.85, edgecolor='black', linewidth=1.5)

            ax.set_xlabel('Genotype', fontsize=13, fontweight='bold')
            ax.set_ylabel('State Occupancy (%)', fontsize=13, fontweight='bold')
            ax.set_title(f'Cohort {cohort}: State Occupancy by Genotype', fontsize=14, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(genotype_order, fontsize=11, rotation=15, ha='right')
            ax.legend(fontsize=11, loc='upper right')
            ax.grid(axis='y', alpha=0.4)
            ax.set_ylim(0, 100)

            # Color the x-axis labels with genotype colors
            for i, label in enumerate(ax.get_xticklabels()):
                label.set_color(get_genotype_color(genotype_order[i]))
                label.set_fontweight('bold')

        plt.tight_layout()
        output_path = self.output_base / 'phase1' / 'fig1_state_occupancy_by_genotype.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_path}")

    def phase1_fig2_state_characteristics(self):
        """Fig 2: State characteristics with accuracy, latency, and transition info."""
        print("Generating Phase 1 Fig 2: State Characteristics...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        for col, cohort in enumerate(['W', 'F']):
            data = self.phase1_data[cohort]

            # Collect state-level metrics
            state_metrics = {state: {'accuracy': [], 'latency': [], 'occupancy': []} for state in range(3)}

            for result in data:
                trials = self.trials_data[cohort]
                animal_id = result['animal_id']
                animal_trials = trials[trials['animal_id'] == animal_id]

                if len(animal_trials) == 0:
                    continue

                states = result['model'].most_likely_states[:len(animal_trials)]

                for state in range(3):
                    state_mask = states == state
                    if np.sum(state_mask) > 0:
                        state_metrics[state]['accuracy'].append(
                            np.mean(animal_trials.iloc[:len(states)].loc[state_mask, 'correct'].values)
                        )
                        state_metrics[state]['occupancy'].append(100 * np.mean(state_mask))

                        if 'latency' in animal_trials.columns:
                            valid_latency = animal_trials.iloc[:len(states)].loc[state_mask, 'latency']
                            if len(valid_latency) > 0:
                                state_metrics[state]['latency'].append(np.nanmean(valid_latency))

            # Plot 1: Accuracy by State
            ax = axes[0, col]
            state_labels = [STATE_LABELS[i] for i in range(3)]
            accuracies = [np.mean(state_metrics[i]['accuracy'])*100 if state_metrics[i]['accuracy'] else 0
                         for i in range(3)]
            acc_sems = [np.std(state_metrics[i]['accuracy'])*100/np.sqrt(max(1, len(state_metrics[i]['accuracy'])))
                       for i in range(3)]

            colors = [get_state_color(i) for i in range(3)]
            bars = ax.bar(state_labels, accuracies, yerr=acc_sems, capsize=5,
                         color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

            ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'Cohort {cohort}: State Accuracy', fontsize=13, fontweight='bold')
            ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Chance', alpha=0.7)
            ax.set_ylim(45, 75)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, val in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

            # Plot 2: Occupancy by State
            ax = axes[0, col+1] if col == 0 else axes[0, 2]
            occupancies = [np.mean(state_metrics[i]['occupancy']) if state_metrics[i]['occupancy'] else 0
                          for i in range(3)]
            occ_sems = [np.std(state_metrics[i]['occupancy'])/np.sqrt(max(1, len(state_metrics[i]['occupancy'])))
                       for i in range(3)]

            bars = ax.bar(state_labels, occupancies, yerr=occ_sems, capsize=5,
                         color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

            ax.set_ylabel('Occupancy (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'Cohort {cohort}: State Occupancy', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            for bar, val in zip(bars, occupancies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

            # Plot 3: Sample size
            ax = axes[1, col] if col == 0 else axes[1, 1]
            n_animals = len(self.phase1_data[cohort])
            ax.text(0.5, 0.7, f'Cohort {cohort} Summary', ha='center', va='center',
                   fontsize=14, fontweight='bold', transform=ax.transAxes)
            ax.text(0.5, 0.5, f'N = {n_animals} animals\n' +
                   f'{len(self.trials_data[cohort])} total trials',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.axis('off')

        # Right side: Comparison between cohorts
        ax = axes[1, 2]

        # Compare state occupancy across cohorts
        cohort_data = {}
        for cohort in ['W', 'F']:
            occ = []
            for state in range(3):
                state_occupancy = [100 * np.mean(r['model'].most_likely_states == state)
                                 for r in self.phase1_data[cohort]]
                occ.append(np.mean(state_occupancy) if state_occupancy else 0)
            cohort_data[cohort] = occ

        x = np.arange(3)
        width = 0.35
        ax.bar(x - width/2, cohort_data['W'], width, label='Cohort W', alpha=0.85, edgecolor='black')
        ax.bar(x + width/2, cohort_data['F'], width, label='Cohort F', alpha=0.85, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(state_labels)
        ax.set_ylabel('Occupancy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Cohort Comparison', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_base / 'phase1' / 'fig2_state_characteristics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_path}")

    def phase1_fig3_genotype_comparison(self):
        """Fig 3: Detailed genotype comparison with state effects."""
        print("Generating Phase 1 Fig 3: Genotype Comparison with State Effects...")

        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

        for cohort_idx, cohort in enumerate(['W', 'F']):
            data = self.phase1_data[cohort]
            genotype_order = [g for g in GENOTYPE_ORDER if any(r['genotype'] == g for r in data)]

            # For each genotype, calculate state metrics
            genotype_states = {}
            for genotype in genotype_order:
                geno_results = [r for r in data if r['genotype'] == genotype]

                state_accuracy = {i: [] for i in range(3)}
                state_occupancy = {i: [] for i in range(3)}

                for result in geno_results:
                    trials = self.trials_data[cohort]
                    animal_trials = trials[trials['animal_id'] == result['animal_id']]

                    if len(animal_trials) == 0:
                        continue

                    states = result['model'].most_likely_states[:len(animal_trials)]

                    for state in range(3):
                        state_mask = states == state
                        occupancy = 100 * np.mean(state_mask)
                        accuracy = 100 * np.mean(animal_trials.iloc[:len(states)].loc[state_mask, 'correct'].values) if np.sum(state_mask) > 0 else 50

                        state_accuracy[state].append(accuracy)
                        state_occupancy[state].append(occupancy)

                genotype_states[genotype] = {'accuracy': state_accuracy, 'occupancy': state_occupancy}

            # Panel 1: State accuracy heatmap
            ax = fig.add_subplot(gs[cohort_idx, 0])

            accuracy_matrix = np.zeros((3, len(genotype_order)))
            for g_idx, genotype in enumerate(genotype_order):
                for state in range(3):
                    acc_vals = genotype_states[genotype]['accuracy'][state]
                    accuracy_matrix[state, g_idx] = np.mean(acc_vals) if acc_vals else 50

            im = ax.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=45, vmax=75)
            ax.set_xticks(range(len(genotype_order)))
            ax.set_xticklabels(genotype_order, rotation=45, ha='right')
            ax.set_yticks(range(3))
            ax.set_yticklabels([STATE_LABELS[i] for i in range(3)])
            ax.set_title(f'Cohort {cohort}: Accuracy Heatmap', fontweight='bold')

            # Add text annotations
            for i in range(3):
                for j in range(len(genotype_order)):
                    text = ax.text(j, i, f'{accuracy_matrix[i, j]:.0f}%',
                                 ha='center', va='center', color='black', fontweight='bold')

            plt.colorbar(im, ax=ax, label='Accuracy (%)')

            # Panel 2: Occupancy by genotype and state
            ax = fig.add_subplot(gs[cohort_idx, 1])

            x = np.arange(len(genotype_order))
            width = 0.25

            for state in range(3):
                occupancies = [np.mean(genotype_states[g]['occupancy'][state]) for g in genotype_order]
                ax.bar(x + state*width, occupancies, width, label=STATE_LABELS[state],
                      color=get_state_color(state), alpha=0.85, edgecolor='black')

            ax.set_xlabel('Genotype', fontweight='bold')
            ax.set_ylabel('Occupancy (%)', fontweight='bold')
            ax.set_title(f'Cohort {cohort}: Occupancy by State', fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(genotype_order, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            # Panel 3: Genotype color legend
            ax = fig.add_subplot(gs[cohort_idx, 2])
            ax.axis('off')

            y_pos = 0.9
            ax.text(0.5, y_pos, f'Cohort {cohort} Genotypes', ha='center', fontweight='bold',
                   fontsize=12, transform=ax.transAxes)

            y_pos -= 0.15
            for genotype in genotype_order:
                color = get_genotype_color(genotype)
                rect = plt.Rectangle((0.1, y_pos-0.05), 0.1, 0.08, facecolor=color,
                                    edgecolor='black', transform=ax.transAxes)
                ax.add_patch(rect)
                ax.text(0.25, y_pos, genotype, fontsize=11, va='center', transform=ax.transAxes)
                y_pos -= 0.15

            # Panel 4: Statistics
            ax = fig.add_subplot(gs[cohort_idx, 3])
            ax.axis('off')

            n_animals = len(data)
            n_trials = len(self.trials_data[cohort])
            n_genotypes = len(genotype_order)

            stats_text = f'Cohort {cohort} Statistics\n\n'
            stats_text += f'N animals: {n_animals}\n'
            stats_text += f'N trials: {n_trials:,}\n'
            stats_text += f'N genotypes: {n_genotypes}\n'
            stats_text += f'N states: 3\n'

            ax.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=11,
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Phase 1: Detailed Genotype Comparison with State Effects',
                    fontsize=16, fontweight='bold', y=0.995)

        output_path = self.output_base / 'phase1' / 'fig3_genotype_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_path}")

    def phase1_fig4_cross_cohort(self):
        """Fig 4: Cross-cohort comparison highlighting cohort effects."""
        print("Generating Phase 1 Fig 4: Cross-Cohort Comparison...")

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Panel 1: Genotype distribution across cohorts
        ax = axes[0, 0]

        all_genotypes = set()
        for cohort in ['W', 'F']:
            all_genotypes.update(r['genotype'] for r in self.phase1_data[cohort])
        all_genotypes = sorted(list(all_genotypes), key=lambda g: GENOTYPE_ORDER.index(g) if g in GENOTYPE_ORDER else 999)

        cohort_genotype_counts = {'W': {}, 'F': {}}
        for genotype in all_genotypes:
            for cohort in ['W', 'F']:
                count = len([r for r in self.phase1_data[cohort] if r['genotype'] == genotype])
                cohort_genotype_counts[cohort][genotype] = count

        x = np.arange(len(all_genotypes))
        width = 0.35

        w_counts = [cohort_genotype_counts['W'].get(g, 0) for g in all_genotypes]
        f_counts = [cohort_genotype_counts['F'].get(g, 0) for g in all_genotypes]

        ax.bar(x - width/2, w_counts, width, label='Cohort W', color='#3498db', alpha=0.85, edgecolor='black')
        ax.bar(x + width/2, f_counts, width, label='Cohort F', color='#e74c3c', alpha=0.85, edgecolor='black')

        ax.set_xlabel('Genotype', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Animals', fontsize=12, fontweight='bold')
        ax.set_title('Genotype Distribution Across Cohorts', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_genotypes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Panel 2: State occupancy comparison
        ax = axes[0, 1]

        cohort_state_occ = {}
        for cohort in ['W', 'F']:
            state_occ = np.zeros(3)
            for r in self.phase1_data[cohort]:
                for state in range(3):
                    state_occ[state] += np.sum(r['model'].most_likely_states == state)
            state_occ = state_occ / np.sum(state_occ) * 100
            cohort_state_occ[cohort] = state_occ

        x = np.arange(3)
        width = 0.35
        state_labels = [STATE_LABELS[i] for i in range(3)]

        ax.bar(x - width/2, cohort_state_occ['W'], width, label='Cohort W', alpha=0.85, edgecolor='black')
        ax.bar(x + width/2, cohort_state_occ['F'], width, label='Cohort F', alpha=0.85, edgecolor='black')

        ax.set_xlabel('State', fontsize=12, fontweight='bold')
        ax.set_ylabel('Occupancy (%)', fontsize=12, fontweight='bold')
        ax.set_title('State Occupancy: Cohort Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(state_labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Color code the state labels
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_color(get_state_color(i))
            label.set_fontweight('bold')

        # Panel 3: Accuracy by cohort and state
        ax = axes[1, 0]

        cohort_accuracy = {}
        for cohort in ['W', 'F']:
            state_accuracy = np.zeros(3)
            state_counts = np.zeros(3)

            for r in self.phase1_data[cohort]:
                trials = self.trials_data[cohort]
                animal_trials = trials[trials['animal_id'] == r['animal_id']]

                if len(animal_trials) == 0:
                    continue

                states = r['model'].most_likely_states[:len(animal_trials)]

                for state in range(3):
                    state_mask = states == state
                    if np.sum(state_mask) > 0:
                        state_accuracy[state] += np.sum(animal_trials.iloc[:len(states)].loc[state_mask, 'correct'].values)
                        state_counts[state] += np.sum(state_mask)

            state_accuracy = np.divide(state_accuracy, state_counts, where=state_counts>0, out=np.full_like(state_accuracy, 0.5)) * 100
            cohort_accuracy[cohort] = state_accuracy

        x = np.arange(3)
        width = 0.35

        ax.bar(x - width/2, cohort_accuracy['W'], width, label='Cohort W', alpha=0.85, edgecolor='black')
        ax.bar(x + width/2, cohort_accuracy['F'], width, label='Cohort F', alpha=0.85, edgecolor='black')

        ax.set_xlabel('State', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy by State: Cohort Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(state_labels)
        ax.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Chance')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(45, 75)

        # Panel 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')

        summary_text = "Phase 1 Summary Statistics\n\n"
        summary_text += "Cohort W:\n"
        summary_text += f"  Animals: {len(self.phase1_data['W'])}\n"
        summary_text += f"  Trials: {len(self.trials_data['W']):,}\n"
        summary_text += f"  Genotypes: {len(set(r['genotype'] for r in self.phase1_data['W']))}\n\n"
        summary_text += "Cohort F:\n"
        summary_text += f"  Animals: {len(self.phase1_data['F'])}\n"
        summary_text += f"  Trials: {len(self.trials_data['F']):,}\n"
        summary_text += f"  Genotypes: {len(set(r['genotype'] for r in self.phase1_data['F']))}\n\n"
        summary_text += "State Definitions:\n"
        for state in range(3):
            summary_text += f"  {STATE_LABELS[state]}: {STATE_DESCRIPTIONS[state]}\n"

        ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11,
               transform=ax.transAxes, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.suptitle('Phase 1: Cross-Cohort Comparison', fontsize=15, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_base / 'phase1' / 'fig4_cross_cohort.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_path}")

    # ========== BATCH 2: PHASE 2 REVERSAL FIGURES ==========

    def batch2_phase2_figures(self):
        """Generate Phase 2 reversal figures."""
        print("\n" + "="*80)
        print("BATCH 2: PHASE 2 REVERSAL FIGURES")
        print("="*80)

        # Check if Phase 2 data exists
        if not self.phase2_data or (not self.phase2_data['W'] and not self.phase2_data['F']):
            print("Note: Phase 2 data not found. Skipping Phase 2 figures.")
            return

        self.phase2_fig1_reversal_state_occupancy()
        self.phase2_fig2_genotype_reversal_comparison()

    def phase2_fig1_reversal_state_occupancy(self):
        """Fig 1: Reversal phase state occupancy."""
        print("Generating Phase 2 Fig 1: Reversal State Occupancy...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for col, cohort in enumerate(['W', 'F']):
            ax = axes[col]
            data = self.phase2_data[cohort]

            if not data:
                ax.text(0.5, 0.5, f'No Phase 2 data for Cohort {cohort}',
                       ha='center', va='center', transform=ax.transAxes)
                continue

            # Calculate state occupancy
            state_occupancy = np.zeros(3)
            for result in data:
                if 'model' in result and hasattr(result['model'], 'most_likely_states'):
                    for state in range(3):
                        state_occupancy[state] += np.sum(result['model'].most_likely_states == state)

            state_occupancy = state_occupancy / np.sum(state_occupancy) * 100 if np.sum(state_occupancy) > 0 else [33, 33, 34]

            state_labels = [STATE_LABELS[i] for i in range(3)]
            colors = [get_state_color(i) for i in range(3)]

            bars = ax.bar(state_labels, state_occupancy, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

            ax.set_ylabel('Occupancy (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'Phase 2 Reversal - Cohort {cohort}', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            for bar, val in zip(bars, state_occupancy):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_base / 'phase2' / 'fig1_reversal_state_occupancy.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_path}")

    def phase2_fig2_genotype_reversal_comparison(self):
        """Fig 2: Reversal by genotype."""
        print("Generating Phase 2 Fig 2: Genotype Reversal Comparison...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for col, cohort in enumerate(['W', 'F']):
            ax = axes[col]
            data = self.phase2_data[cohort]

            if not data:
                ax.text(0.5, 0.5, f'No Phase 2 data for Cohort {cohort}',
                       ha='center', va='center', transform=ax.transAxes)
                continue

            # Organize by genotype
            genotype_order = [g for g in GENOTYPE_ORDER if any(r.get('genotype') == g for r in data)]

            state_by_genotype = {g: np.zeros(3) for g in genotype_order}

            for result in data:
                genotype = result.get('genotype', 'Unknown')
                if genotype not in state_by_genotype:
                    continue

                if 'model' in result and hasattr(result['model'], 'most_likely_states'):
                    for state in range(3):
                        state_by_genotype[genotype][state] += np.sum(result['model'].most_likely_states == state)

            # Normalize
            for genotype in genotype_order:
                total = np.sum(state_by_genotype[genotype])
                if total > 0:
                    state_by_genotype[genotype] = state_by_genotype[genotype] / total * 100

            x = np.arange(len(genotype_order))
            width = 0.25

            for state in range(3):
                values = [state_by_genotype[g][state] for g in genotype_order]
                ax.bar(x + state*width, values, width, label=STATE_LABELS[state],
                      color=get_state_color(state), alpha=0.85, edgecolor='black')

            ax.set_xlabel('Genotype', fontsize=12, fontweight='bold')
            ax.set_ylabel('State Occupancy (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'Phase 2 Reversal by Genotype - Cohort {cohort}', fontsize=13, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(genotype_order, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            # Color the genotype labels
            for i, label in enumerate(ax.get_xticklabels()):
                label.set_color(get_genotype_color(genotype_order[i]))
                label.set_fontweight('bold')

        plt.tight_layout()
        output_path = self.output_base / 'phase2' / 'fig2_genotype_reversal_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_path}")

    # ========== BATCH 3: IMPROVED COMPARATIVE FIGURES ==========

    def batch3_improved_comparative_figures(self):
        """Create new improved figures for phase and genotype comparison."""
        print("\n" + "="*80)
        print("BATCH 3: IMPROVED COMPARATIVE FIGURES")
        print("="*80)

        self.improved_fig1_phase_comparison()
        self.improved_fig2_genotype_state_effects()

    def improved_fig1_phase_comparison(self):
        """Improved Fig 1: Clear phase and state effect visualization."""
        print("Generating Improved Fig 1: Phase and State Comparison...")

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        for col, cohort in enumerate(['W', 'F']):
            ax = axes[0, col]

            # Gather phase 1 and 2 data
            phase1 = self.phase1_data[cohort]
            phase2 = self.phase2_data[cohort]

            phases = {}

            for phase_num, data in [(1, phase1), (2, phase2)]:
                state_occ = np.zeros(3)
                for r in data:
                    if 'model' in r and hasattr(r['model'], 'most_likely_states'):
                        for state in range(3):
                            state_occ[state] += np.sum(r['model'].most_likely_states == state)
                state_occ = state_occ / np.sum(state_occ) * 100 if np.sum(state_occ) > 0 else [33, 33, 34]
                phases[phase_num] = state_occ

            if len(phases) == 2:
                x = np.arange(3)
                width = 0.35
                state_labels = [STATE_LABELS[i] for i in range(3)]

                ax.bar(x - width/2, phases[1], width, label='Phase 1', alpha=0.85,
                      edgecolor='black', linewidth=1.5)
                ax.bar(x + width/2, phases[2], width, label='Phase 2', alpha=0.85,
                      edgecolor='black', linewidth=1.5, hatch='//')

                ax.set_xlabel('State', fontsize=12, fontweight='bold')
                ax.set_ylabel('Occupancy (%)', fontsize=12, fontweight='bold')
                ax.set_title(f'Cohort {cohort}: Phase Effects on State Usage', fontsize=13, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(state_labels)
                ax.legend(fontsize=11)
                ax.grid(axis='y', alpha=0.3)

                # Color the state labels
                for i, label in enumerate(ax.get_xticklabels()):
                    label.set_color(get_state_color(i))
                    label.set_fontweight('bold')

            # Panel 2: State transition heatmap concept
            ax = axes[1, col]

            # Create a simplified state transition visualization
            n_states = 3

            # For Phase 1: calculate transition probabilities
            phase1_trans = np.zeros((3, 3))
            for r in phase1:
                if 'model' in r and hasattr(r['model'], 'transitions'):
                    if hasattr(r['model'].transitions, 'shape'):
                        phase1_trans += r['model'].transitions.sum(axis=0) if r['model'].transitions.shape[0] > 3 else r['model'].transitions

            phase1_trans = phase1_trans / np.sum(phase1_trans, axis=1, keepdims=True) if np.sum(phase1_trans) > 0 else np.eye(3)/3

            im = ax.imshow(phase1_trans, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            state_labels_short = [STATE_LABELS[i][:3] for i in range(3)]
            ax.set_xticklabels(state_labels_short)
            ax.set_yticklabels(state_labels_short)
            ax.set_xlabel('To State', fontsize=11, fontweight='bold')
            ax.set_ylabel('From State', fontsize=11, fontweight='bold')
            ax.set_title(f'Cohort {cohort}: State Transition Probabilities (Phase 1)', fontsize=12, fontweight='bold')

            # Add text annotations
            for i in range(3):
                for j in range(3):
                    text = ax.text(j, i, f'{phase1_trans[i, j]:.2f}',
                                 ha='center', va='center', color='black' if phase1_trans[i, j] < 0.5 else 'white',
                                 fontweight='bold', fontsize=10)

            plt.colorbar(im, ax=ax, label='Probability')

        # Bottom panels: Summary
        ax = axes[1, 0]
        ax.axis('off')

        summary_text = "Key Observations:\n\n"
        summary_text += "Phase 1 (Initial Learning):\n"
        summary_text += "• Animals establish baseline state\n"
        summary_text += "• Learning occurs within states\n"
        summary_text += "• State transitions adjust strategy\n\n"
        summary_text += "Phase 2 (Reversal Learning):\n"
        summary_text += "• State switch reflects adaptation\n"
        summary_text += "• May increase Engaged state use\n"
        summary_text += "• Shows behavioral flexibility"

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

        ax = axes[1, 1]
        ax.axis('off')

        color_guide = "State Colors:\n\n"
        for state in range(3):
            color_guide += f"{STATE_LABELS[state]}: "
            color_guide += f"{STATE_DESCRIPTIONS[state].replace(chr(10), ' ')}\n\n"

        ax.text(0.05, 0.95, color_guide, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.2))

        plt.suptitle('Improved Figure: Phase and State Effects on Behavior', fontsize=15, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_base / 'comparisons' / 'improved_fig1_phase_state_effects.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_path}")

    def improved_fig2_genotype_state_effects(self):
        """Improved Fig 2: Clear genotype-by-state effects visualization."""
        print("Generating Improved Fig 2: Genotype-by-State Effects...")

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        for cohort_idx, cohort in enumerate(['W', 'F']):
            data = self.phase1_data[cohort]
            genotype_order = [g for g in GENOTYPE_ORDER if any(r['genotype'] == g for r in data)]

            # Panel A: Heatmap of accuracy by genotype x state
            ax = fig.add_subplot(gs[cohort_idx, 0])

            accuracy_matrix = np.zeros((len(genotype_order), 3))

            for g_idx, genotype in enumerate(genotype_order):
                geno_results = [r for r in data if r['genotype'] == genotype]

                for state in range(3):
                    accuracies = []

                    for result in geno_results:
                        trials = self.trials_data[cohort]
                        animal_trials = trials[trials['animal_id'] == result['animal_id']]

                        if len(animal_trials) == 0:
                            continue

                        states = result['model'].most_likely_states[:len(animal_trials)]
                        state_mask = states == state

                        if np.sum(state_mask) > 0:
                            acc = np.mean(animal_trials.iloc[:len(states)].loc[state_mask, 'correct'].values)
                            accuracies.append(acc)

                    accuracy_matrix[g_idx, state] = np.mean(accuracies) * 100 if accuracies else 50

            im = ax.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=45, vmax=75)
            ax.set_xticks(range(3))
            ax.set_xticklabels([STATE_LABELS[i] for i in range(3)])
            ax.set_yticks(range(len(genotype_order)))
            ax.set_yticklabels(genotype_order)
            ax.set_ylabel('Genotype', fontsize=11, fontweight='bold')
            ax.set_title(f'Cohort {cohort}: Accuracy Heatmap', fontsize=12, fontweight='bold')

            # Add text
            for i in range(len(genotype_order)):
                for j in range(3):
                    text = ax.text(j, i, f'{accuracy_matrix[i, j]:.0f}%',
                                 ha='center', va='center', color='black', fontweight='bold', fontsize=9)

            plt.colorbar(im, ax=ax, label='Accuracy (%)')

            # Panel B: State occupancy across genotypes
            ax = fig.add_subplot(gs[cohort_idx, 1])

            occupancy_by_genotype = {}
            for genotype in genotype_order:
                geno_results = [r for r in data if r['genotype'] == genotype]
                state_occ = np.zeros(3)

                for result in geno_results:
                    for state in range(3):
                        state_occ[state] += np.sum(result['model'].most_likely_states == state)

                if np.sum(state_occ) > 0:
                    state_occ = state_occ / np.sum(state_occ) * 100
                occupancy_by_genotype[genotype] = state_occ

            x = np.arange(len(genotype_order))
            width = 0.25

            for state in range(3):
                values = [occupancy_by_genotype[g][state] for g in genotype_order]
                ax.bar(x + state*width, values, width, label=STATE_LABELS[state],
                      color=get_state_color(state), alpha=0.85, edgecolor='black')

            ax.set_xlabel('Genotype', fontsize=11, fontweight='bold')
            ax.set_ylabel('Occupancy (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'Cohort {cohort}: Occupancy by State', fontsize=12, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(genotype_order, rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)

            # Color the genotype labels
            for i, label in enumerate(ax.get_xticklabels()):
                label.set_color(get_genotype_color(genotype_order[i]))
                label.set_fontweight('bold')

            # Panel C: Genotype color legend
            ax = fig.add_subplot(gs[cohort_idx, 2])
            ax.axis('off')

            y_pos = 0.95
            ax.text(0.5, y_pos, f'Cohort {cohort} Genotypes', ha='center', fontweight='bold',
                   fontsize=11, transform=ax.transAxes)

            y_pos -= 0.12
            for genotype in genotype_order:
                color = get_genotype_color(genotype)
                rect = plt.Rectangle((0.05, y_pos-0.04), 0.1, 0.07, facecolor=color,
                                    edgecolor='black', transform=ax.transAxes, linewidth=1.5)
                ax.add_patch(rect)
                ax.text(0.2, y_pos, genotype, fontsize=10, va='center', transform=ax.transAxes, fontweight='bold')
                y_pos -= 0.12

            # Add sample size info
            ax.text(0.5, 0.1, f'N = {len(data)} animals', ha='center', fontsize=9,
                   transform=ax.transAxes, style='italic')

        plt.suptitle('Improved Figure: Genotype-by-State Effects (Accuracy & Occupancy)',
                    fontsize=15, fontweight='bold')

        output_path = self.output_base / 'comparisons' / 'improved_fig2_genotype_state_effects.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_path}")

    # ========== MAIN EXECUTION ==========

    def run_all(self):
        """Execute all batches."""
        print("\n" + "="*80)
        print("MASTER FIGURE REGENERATOR")
        print("="*80)
        print(f"Output directory: {self.output_base}")

        self.load_all_data()
        self.batch1_phase1_figures()
        self.batch2_phase2_figures()
        self.batch3_improved_comparative_figures()

        print("\n" + "="*80)
        print("✓ ALL FIGURES REGENERATED SUCCESSFULLY")
        print("="*80)
        print(f"\nOutput directory: {self.output_base}")
        print(f"  phase1/: {len(list((self.output_base / 'phase1').glob('*.png')))} figures")
        print(f"  phase2/: {len(list((self.output_base / 'phase2').glob('*.png')))} figures")
        print(f"  comparisons/: {len(list((self.output_base / 'comparisons').glob('*.png')))} figures")


if __name__ == '__main__':
    regenerator = MasterFigureRegenerator()
    regenerator.run_all()
