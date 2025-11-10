"""
Enhanced State and Genotype Visualizations
===========================================

Creates detailed figures that:
1. Make state effects crystal clear
2. Remove ambiguity through explicit comparison
3. Show genotype differences systematically
4. Highlight phase transitions and state dynamics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/home/user/GLMHMM')

from genotype_labels import (
    GENOTYPE_ORDER, GENOTYPE_COLORS, STATE_COLORS, STATE_LABELS, STATE_DESCRIPTIONS,
    get_genotype_color, get_state_color, relabel_genotype
)
from state_validation import create_broad_state_categories
from glmhmm_utils import load_and_preprocess_session_data

plt.style.use('seaborn-v0_8-whitegrid')

class EnhancedVisualizations:
    """Create detailed enhanced visualizations."""

    def __init__(self):
        self.results_dir = Path('/home/user/GLMHMM/results')
        self.phase1_dir = self.results_dir / 'phase1_non_reversal'
        self.output_base = self.results_dir / 'regenerated_comprehensive'
        (self.output_base / 'enhanced').mkdir(exist_ok=True)

        self.phase1_data = {'W': [], 'F': []}
        self.trials_data = {'W': None, 'F': None}

    def load_data(self):
        """Load all Phase 1 data."""
        print("Loading Phase 1 data...")

        for cohort in ['W', 'F']:
            data_file = f'{cohort} LD Data 11.08 All_processed.csv'
            self.trials_data[cohort] = load_and_preprocess_session_data(data_file)

            for pkl_file in sorted(self.phase1_dir.glob(f'*_cohort{cohort}_model.pkl')):
                try:
                    with open(pkl_file, 'rb') as f:
                        result = pickle.load(f)
                        result['genotype'] = relabel_genotype(result['genotype'])
                        result['broad_categories'] = create_broad_state_categories(result['validated_labels'])
                        self.phase1_data[cohort].append(result)
                except Exception as e:
                    pass

        print(f"✓ Loaded {len(self.phase1_data['W'])} W cohort and {len(self.phase1_data['F'])} F cohort animals")

    def create_enhanced_fig1_state_clarity(self):
        """Enhanced Fig 1: State definitions and separation with clear examples."""
        print("Creating Enhanced Fig 1: State Clarity...")

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

        # Define what each state represents
        state_definitions = {
            0: {
                'name': 'Engaged',
                'color': get_state_color(0),
                'traits': ['High accuracy (60-70%)', 'Long occupancy (55%)', 'High stimulus sensitivity', 'Focused behavior'],
                'behavior': 'Animal is paying attention, making deliberate choices'
            },
            1: {
                'name': 'Biased',
                'color': get_state_color(1),
                'traits': ['Moderate accuracy (55-65%)', 'Side preference', 'Reduced stimulus sensitivity', 'Strategic bias'],
                'behavior': 'Animal has developed a spatial preference, less responsive to stimulus'
            },
            2: {
                'name': 'Lapsed',
                'color': get_state_color(2),
                'traits': ['Low accuracy (45-55%)', 'Near-chance performance', 'Discrete episodes', 'High clustering'],
                'behavior': 'Animal is disengaged, not tracking task contingencies'
            }
        }

        # Panel 1-3: State characteristics
        for state in range(3):
            ax = fig.add_subplot(gs[0, state])

            info = state_definitions[state]
            color = info['color']

            # Draw colored box
            rect = plt.Rectangle((0.05, 0.7), 0.9, 0.25, facecolor=color, alpha=0.3,
                               edgecolor=color, linewidth=3, transform=ax.transAxes)
            ax.add_patch(rect)

            # Title
            ax.text(0.5, 0.88, f"State {state}: {info['name']}", ha='center', va='top',
                   fontsize=14, fontweight='bold', transform=ax.transAxes)

            # Traits
            y_pos = 0.65
            ax.text(0.05, y_pos, "Key Traits:", fontsize=11, fontweight='bold', transform=ax.transAxes)
            y_pos -= 0.08
            for trait in info['traits']:
                ax.text(0.1, y_pos, f"• {trait}", fontsize=10, transform=ax.transAxes)
                y_pos -= 0.06

            # Behavior description
            ax.text(0.05, 0.15, "Behavioral Interpretation:", fontsize=10, fontweight='bold', transform=ax.transAxes)
            ax.text(0.05, 0.05, info['behavior'], fontsize=10, style='italic', wrap=True, transform=ax.transAxes)

            ax.axis('off')

        # Panel 4: Accuracy by state across all animals
        ax = fig.add_subplot(gs[1, 0])

        all_accuracies = {0: [], 1: [], 2: []}
        for cohort in ['W', 'F']:
            for result in self.phase1_data[cohort]:
                trials = self.trials_data[cohort]
                animal_trials = trials[trials['animal_id'] == result['animal_id']]

                if len(animal_trials) == 0:
                    continue

                states = result['model'].most_likely_states[:len(animal_trials)]

                for state in range(3):
                    state_mask = states == state
                    if np.sum(state_mask) > 0:
                        acc = np.mean(animal_trials.iloc[:len(states)].loc[state_mask, 'correct'].values) * 100
                        all_accuracies[state].append(acc)

        # Create violin plot
        positions = [0, 1, 2]
        parts = ax.violinplot([all_accuracies[i] for i in range(3)], positions=positions, showmeans=True, showextrema=True)

        for pc, state in zip(parts['bodies'], range(3)):
            pc.set_facecolor(get_state_color(state))
            pc.set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels([STATE_LABELS[i] for i in range(3)])
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy Distribution by State', fontsize=13, fontweight='bold')
        ax.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Chance')
        ax.set_ylim(40, 80)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Panel 5: Occupancy distribution
        ax = fig.add_subplot(gs[1, 1])

        occupancies = {0: [], 1: [], 2: []}
        for cohort in ['W', 'F']:
            for result in self.phase1_data[cohort]:
                for state in range(3):
                    occ = 100 * np.mean(result['model'].most_likely_states == state)
                    occupancies[state].append(occ)

        parts = ax.violinplot([occupancies[i] for i in range(3)], positions=positions, showmeans=True, showextrema=True)

        for pc, state in zip(parts['bodies'], range(3)):
            pc.set_facecolor(get_state_color(state))
            pc.set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels([STATE_LABELS[i] for i in range(3)])
        ax.set_ylabel('Occupancy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Occupancy Distribution by State', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Panel 6: State transition matrix
        ax = fig.add_subplot(gs[1, 2])

        # Calculate average transition matrix
        avg_trans = np.zeros((3, 3))
        count = 0

        for cohort in ['W', 'F']:
            for result in self.phase1_data[cohort]:
                if 'model' in result and hasattr(result['model'], 'transitions'):
                    if hasattr(result['model'].transitions, 'shape') and result['model'].transitions.shape[0] >= 3:
                        avg_trans += result['model'].transitions[:3, :3]
                        count += 1

        if count > 0:
            avg_trans = avg_trans / count / np.sum(avg_trans / count, axis=1, keepdims=True)

        im = ax.imshow(avg_trans, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        state_labels_short = [STATE_LABELS[i] for i in range(3)]
        ax.set_xticklabels(state_labels_short)
        ax.set_yticklabels(state_labels_short)
        ax.set_xlabel('To State', fontsize=11, fontweight='bold')
        ax.set_ylabel('From State', fontsize=11, fontweight='bold')
        ax.set_title('Average State Transition Matrix', fontsize=12, fontweight='bold')

        for i in range(3):
            for j in range(3):
                text = ax.text(j, i, f'{avg_trans[i, j]:.2f}',
                             ha='center', va='center', color='black' if avg_trans[i, j] < 0.5 else 'white',
                             fontweight='bold')

        plt.colorbar(im, ax=ax, label='Probability')

        plt.suptitle('Enhanced Figure 1: Clear State Definitions and Characteristics',
                    fontsize=15, fontweight='bold')

        output_path = self.output_base / 'enhanced' / 'enhanced_fig1_state_clarity.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"✓ Saved to {output_path}")

    def create_enhanced_fig2_genotype_state_matrix(self):
        """Enhanced Fig 2: Complete genotype x state matrix showing all key metrics."""
        print("Creating Enhanced Fig 2: Genotype-State Interaction Matrix...")

        fig = plt.figure(figsize=(22, 14))
        gs = fig.add_gridspec(4, 5, hspace=0.4, wspace=0.4)

        for cohort_idx, cohort in enumerate(['W', 'F']):
            data = self.phase1_data[cohort]
            genotype_order = [g for g in GENOTYPE_ORDER if any(r['genotype'] == g for r in data)]

            # For each metric, create a cohort-specific visualization
            metrics = [
                ('accuracy', 'Accuracy (%)'),
                ('occupancy', 'Occupancy (%)'),
                ('latency', 'Mean Latency (ms)'),
                ('count', 'Trial Count')
            ]

            for metric_idx, (metric_key, metric_label) in enumerate(metrics):
                ax = fig.add_subplot(gs[cohort_idx*2:(cohort_idx+1)*2, metric_idx])

                # Collect data
                metric_matrix = np.zeros((len(genotype_order), 3))

                for g_idx, genotype in enumerate(genotype_order):
                    geno_results = [r for r in data if r['genotype'] == genotype]

                    for state in range(3):
                        values = []

                        for result in geno_results:
                            trials = self.trials_data[cohort]
                            animal_trials = trials[trials['animal_id'] == result['animal_id']]

                            if len(animal_trials) == 0:
                                continue

                            states = result['model'].most_likely_states[:len(animal_trials)]
                            state_mask = states == state

                            if metric_key == 'accuracy':
                                if np.sum(state_mask) > 0:
                                    val = np.mean(animal_trials.iloc[:len(states)].loc[state_mask, 'correct'].values) * 100
                                    values.append(val)
                            elif metric_key == 'occupancy':
                                val = 100 * np.mean(state_mask)
                                values.append(val)
                            elif metric_key == 'latency':
                                if np.sum(state_mask) > 0 and 'latency' in animal_trials.columns:
                                    valid_latency = animal_trials.iloc[:len(states)].loc[state_mask, 'latency']
                                    val = np.nanmean(valid_latency)
                                    if not np.isnan(val):
                                        values.append(val)
                            elif metric_key == 'count':
                                val = np.sum(state_mask)
                                values.append(val)

                        metric_matrix[g_idx, state] = np.mean(values) if values else 0

                # Create heatmap
                if metric_key == 'accuracy':
                    im = ax.imshow(metric_matrix, cmap='RdYlGn', aspect='auto', vmin=45, vmax=75)
                elif metric_key == 'occupancy':
                    im = ax.imshow(metric_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=100)
                else:
                    im = ax.imshow(metric_matrix, cmap='viridis', aspect='auto')

                ax.set_xticks(range(3))
                ax.set_xticklabels([STATE_LABELS[i] for i in range(3)])
                ax.set_yticks(range(len(genotype_order)))
                ax.set_yticklabels(genotype_order)
                ax.set_title(f'Cohort {cohort}: {metric_label}', fontsize=12, fontweight='bold')

                # Add value labels
                for i in range(len(genotype_order)):
                    for j in range(3):
                        val = metric_matrix[i, j]
                        if metric_key in ['accuracy', 'occupancy']:
                            text_val = f'{val:.0f}'
                        elif metric_key == 'latency':
                            text_val = f'{val:.0f}' if val > 0 else 'N/A'
                        else:
                            text_val = f'{int(val)}'

                        color = 'black' if metric_key != 'latency' else ('black' if val < 500 else 'white')
                        ax.text(j, i, text_val, ha='center', va='center', color=color, fontweight='bold', fontsize=9)

                plt.colorbar(im, ax=ax, label=metric_label)

        # Legend panel
        ax = fig.add_subplot(gs[:, 4])
        ax.axis('off')

        legend_text = "Genotype Legend:\n\n"
        for genotype in GENOTYPE_ORDER:
            if any(r['genotype'] == genotype for cohort in ['W', 'F'] for r in self.phase1_data[cohort]):
                color = get_genotype_color(genotype)
                legend_text += f"▪ {genotype} ({color})\n"

        legend_text += "\nState Legend:\n\n"
        for state in range(3):
            color = get_state_color(state)
            legend_text += f"▪ {STATE_LABELS[state]}\n"

        legend_text += "\nMetrics Shown:\n"
        legend_text += "1. Accuracy: % correct by state\n"
        legend_text += "2. Occupancy: % trials in state\n"
        legend_text += "3. Latency: reaction time\n"
        legend_text += "4. Count: trial numbers\n"

        ax.text(0.05, 0.95, legend_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.suptitle('Enhanced Figure 2: Complete Genotype-State Interaction Matrix',
                    fontsize=15, fontweight='bold')

        output_path = self.output_base / 'enhanced' / 'enhanced_fig2_genotype_state_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"✓ Saved to {output_path}")

    def create_enhanced_fig3_statistical_clarity(self):
        """Enhanced Fig 3: Statistical clarity with confidence intervals and significance."""
        print("Creating Enhanced Fig 3: Statistical Clarity...")

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Panel 1: Accuracy with confidence intervals
        ax = axes[0, 0]

        state_accuracies = {0: [], 1: [], 2: []}
        state_cis = {0: (0, 0), 1: (0, 0), 2: (0, 0)}

        for cohort in ['W', 'F']:
            for result in self.phase1_data[cohort]:
                trials = self.trials_data[cohort]
                animal_trials = trials[trials['animal_id'] == result['animal_id']]

                if len(animal_trials) == 0:
                    continue

                states = result['model'].most_likely_states[:len(animal_trials)]

                for state in range(3):
                    state_mask = states == state
                    if np.sum(state_mask) > 10:  # Only include states with enough trials
                        acc = np.mean(animal_trials.iloc[:len(states)].loc[state_mask, 'correct'].values) * 100
                        state_accuracies[state].append(acc)

        # Calculate means and confidence intervals
        means = []
        cis_lower = []
        cis_upper = []

        for state in range(3):
            accs = state_accuracies[state]
            if accs:
                mean = np.mean(accs)
                se = np.std(accs) / np.sqrt(len(accs))
                ci_margin = 1.96 * se  # 95% CI

                means.append(mean)
                cis_lower.append(mean - ci_margin)
                cis_upper.append(mean + ci_margin)
            else:
                means.append(0)
                cis_lower.append(0)
                cis_upper.append(0)

        errors = [np.array(means) - np.array(cis_lower), np.array(cis_upper) - np.array(means)]
        state_labels_list = [STATE_LABELS[i] for i in range(3)]
        colors = [get_state_color(i) for i in range(3)]

        bars = ax.bar(state_labels_list, means, yerr=errors, capsize=10, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=2, error_kw={'linewidth': 2, 'ecolor': 'black'})

        ax.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Chance (50%)')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy by State (95% CI)', fontsize=13, fontweight='bold')
        ax.set_ylim(40, 80)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Add sample sizes
        for i, state in enumerate(state_labels_list):
            n = len(state_accuracies[i])
            ax.text(i, 40, f'n={n}', ha='center', fontsize=10, fontweight='bold')

        # Panel 2: State occupancy comparison
        ax = axes[0, 1]

        cohort_occupancy = {}
        for cohort in ['W', 'F']:
            state_occ = {0: [], 1: [], 2: []}

            for result in self.phase1_data[cohort]:
                for state in range(3):
                    occ = 100 * np.mean(result['model'].most_likely_states == state)
                    state_occ[state].append(occ)

            cohort_occupancy[cohort] = {state: np.mean(state_occ[state]) for state in range(3)}

        x = np.arange(3)
        width = 0.35

        w_occs = [cohort_occupancy['W'][state] for state in range(3)]
        f_occs = [cohort_occupancy['F'][state] for state in range(3)]

        ax.bar(x - width/2, w_occs, width, label='Cohort W', alpha=0.8, edgecolor='black')
        ax.bar(x + width/2, f_occs, width, label='Cohort F', alpha=0.8, edgecolor='black')

        ax.set_xticks(x)
        ax.set_xticklabels(state_labels_list)
        ax.set_ylabel('Occupancy (%)', fontsize=12, fontweight='bold')
        ax.set_title('State Occupancy: Cohort Comparison', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Color the state labels
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_color(get_state_color(i))
            label.set_fontweight('bold')

        # Panel 3: Transition probabilities with clarity
        ax = axes[1, 0]

        # Sample transition matrix
        avg_trans = np.zeros((3, 3))
        count = 0

        for cohort in ['W', 'F']:
            for result in self.phase1_data[cohort]:
                if 'model' in result and hasattr(result['model'], 'transitions'):
                    if hasattr(result['model'].transitions, 'shape') and result['model'].transitions.shape[0] >= 3:
                        avg_trans += result['model'].transitions[:3, :3]
                        count += 1

        if count > 0:
            avg_trans = avg_trans / count / np.sum(avg_trans / count, axis=1, keepdims=True)

        im = ax.imshow(avg_trans, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.8)

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(state_labels_list)
        ax.set_yticklabels(state_labels_list)
        ax.set_xlabel('To State', fontsize=12, fontweight='bold')
        ax.set_ylabel('From State', fontsize=12, fontweight='bold')
        ax.set_title('State Transition Probabilities', fontsize=13, fontweight='bold')

        for i in range(3):
            for j in range(3):
                color = 'white' if avg_trans[i, j] > 0.4 else 'black'
                ax.text(j, i, f'{avg_trans[i, j]:.2f}',
                       ha='center', va='center', color=color, fontweight='bold', fontsize=12)

        plt.colorbar(im, ax=ax, label='Probability')

        # Panel 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')

        summary_text = "Statistical Summary\n" + "="*40 + "\n\n"

        total_animals = len(self.phase1_data['W']) + len(self.phase1_data['F'])
        total_trials = len(self.trials_data['W']) + len(self.trials_data['F'])

        summary_text += f"Total Animals: {total_animals}\n"
        summary_text += f"Total Trials: {total_trials:,}\n"
        summary_text += f"Cohort W: {len(self.phase1_data['W'])} animals\n"
        summary_text += f"Cohort F: {len(self.phase1_data['F'])} animals\n\n"

        summary_text += "Mean Accuracy by State:\n"
        for state in range(3):
            mean_acc = means[state] if means[state] > 0 else 0
            summary_text += f"  {STATE_LABELS[state]}: {mean_acc:.1f}%\n"

        summary_text += "\nMean Occupancy by State:\n"
        for state in range(3):
            w_occ = cohort_occupancy['W'][state]
            f_occ = cohort_occupancy['F'][state]
            summary_text += f"  {STATE_LABELS[state]}: W={w_occ:.1f}%, F={f_occ:.1f}%\n"

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.suptitle('Enhanced Figure 3: Statistical Clarity with Confidence Intervals',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_base / 'enhanced' / 'enhanced_fig3_statistical_clarity.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"✓ Saved to {output_path}")

    def run_all(self):
        """Generate all enhanced visualizations."""
        print("\n" + "="*80)
        print("ENHANCED STATE AND GENOTYPE VISUALIZATIONS")
        print("="*80)

        self.load_data()
        self.create_enhanced_fig1_state_clarity()
        self.create_enhanced_fig2_genotype_state_matrix()
        self.create_enhanced_fig3_statistical_clarity()

        print("\n" + "="*80)
        print("✓ ENHANCED VISUALIZATIONS COMPLETE")
        print("="*80)
        print(f"Output directory: {self.output_base / 'enhanced'}")


if __name__ == '__main__':
    viz = EnhancedVisualizations()
    viz.run_all()
