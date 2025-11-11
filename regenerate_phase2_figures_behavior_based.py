"""
Phase 2 Figure Regenerator - Behavior-Based State Classification
=================================================================

Regenerates Phase 2 figures using BEHAVIOR-BASED state classification
(accuracy thresholds) rather than arbitrary state numbers.

Accuracy Thresholds:
- Engaged: ≥0.65 (65%)
- Mixed: 0.20-0.65
- Lapsed: ≤0.20 (20%)

Usage: python regenerate_phase2_figures_behavior_based.py
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
    GENOTYPE_ORDER, GENOTYPE_COLORS
)

plt.style.use('seaborn-v0_8-whitegrid')


def classify_state_by_behavior(accuracy):
    """
    Classify state based on accuracy threshold.

    Returns: ('Engaged'|'Mixed'|'Lapsed', color)
    """
    if accuracy >= 0.65:
        return 'Engaged', '#2ecc71'  # Green
    elif accuracy <= 0.20:
        return 'Lapsed', '#e74c3c'   # Red
    else:
        return 'Mixed', '#f39c12'    # Orange


class Phase2BehaviorBasedRegenerator:
    """Regenerate Phase 2 figures with behavior-based state classification."""

    def __init__(self):
        self.phase2_dir = Path('results/phase2_reversal/models')
        self.output_dir = Path('results/regenerated_figures/phase2_behavior_based')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_phase2_data(self):
        """Load Phase 2 data with behavior-based state classification."""
        print("\n" + "="*80)
        print("Loading Phase 2 Data with Behavior-Based State Classification...")
        print("="*80)

        data_W, data_F = [], []

        for pkl_file in self.phase2_dir.glob('*_reversal.pkl'):
            try:
                with open(pkl_file, 'rb') as f:
                    r = pickle.load(f)
                    r['genotype'] = relabel_genotype(r['genotype'])

                    # Add behavior-based state classification
                    traj = r['trajectory_df']
                    r['behavior_states'] = {}

                    for state in traj['state'].unique():
                        state_data = traj[traj['state'] == state]
                        acc = state_data['during_accuracy'].mean()
                        classification, color = classify_state_by_behavior(acc)

                        r['behavior_states'][state] = {
                            'classification': classification,
                            'accuracy': acc,
                            'color': color,
                            'occupancy': len(state_data) / len(traj)
                        }

                    if r['cohort'] == 'W':
                        data_W.append(r)
                    else:
                        data_F.append(r)
            except Exception as e:
                print(f"  Warning: Could not load {pkl_file.name}: {e}")

        print(f"  W Cohort: {len(data_W)} animals")
        print(f"  F Cohort: {len(data_F)} animals")

        return data_W, data_F

    def fig1_state_classification_summary(self, data_W, data_F):
        """Figure 1: Behavioral state classification by genotype."""
        print("\nGenerating Fig 1: Behavioral State Classification...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        for ax, (cohort, data) in zip(axes, [('W', data_W), ('F', data_F)]):
            if len(data) == 0:
                ax.text(0.5, 0.5, f'No Phase 2 data for Cohort {cohort}',
                       ha='center', va='center', fontsize=14)
                continue

            # Aggregate behavioral state time by genotype
            geno_behavior = {}

            for r in data:
                g = r['genotype']
                if g not in geno_behavior:
                    geno_behavior[g] = {'Engaged': [], 'Mixed': [], 'Lapsed': []}

                # Calculate total time in each behavioral state
                behavior_time = {'Engaged': 0, 'Mixed': 0, 'Lapsed': 0}

                for state, info in r['behavior_states'].items():
                    classification = info['classification']
                    occupancy = info['occupancy']
                    behavior_time[classification] += occupancy

                # Convert to percentages
                for behavior_type in ['Engaged', 'Mixed', 'Lapsed']:
                    geno_behavior[g][behavior_type].append(behavior_time[behavior_type] * 100)

            # Plot
            genotypes = [g for g in GENOTYPE_ORDER if g in geno_behavior]
            x = np.arange(len(genotypes))
            width = 0.25

            colors = {'Engaged': '#2ecc71', 'Mixed': '#f39c12', 'Lapsed': '#e74c3c'}

            for i, behavior_type in enumerate(['Engaged', 'Mixed', 'Lapsed']):
                means = [np.mean(geno_behavior[g][behavior_type]) for g in genotypes]
                sems = [stats.sem(geno_behavior[g][behavior_type]) if len(geno_behavior[g][behavior_type]) > 1 else 0
                       for g in genotypes]

                ax.bar(x + i*width, means, width, yerr=sems,
                      label=behavior_type, color=colors[behavior_type],
                      alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)

            ax.set_xlabel('Genotype', fontsize=14, fontweight='bold')
            ax.set_ylabel('% Time in Behavioral State', fontsize=14, fontweight='bold')
            ax.set_title(f'Cohort {cohort}', fontsize=16, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(genotypes, rotation=45, ha='right', fontsize=12)
            ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 100])

            for i, g in enumerate(genotypes):
                n = len(geno_behavior[g]['Engaged'])
                ax.text(i, -10, f'n={n}', ha='center', fontsize=11, fontweight='bold')

        plt.suptitle('Behavioral State Classification During Reversal (Phase 2)\nBased on Accuracy Thresholds',
                     fontsize=18, fontweight='bold')
        plt.tight_layout()

        self.save_figure('fig1_behavioral_state_classification')
        print(f"  ✓ Saved")

    def fig2_engaged_time_by_genotype(self, data_W, data_F):
        """Figure 2: Time spent in Engaged state by genotype."""
        print("\nGenerating Fig 2: Engaged Time by Genotype...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        for ax, (cohort, data) in zip(axes, [('W', data_W), ('F', data_F)]):
            if len(data) == 0:
                ax.text(0.5, 0.5, f'No Phase 2 data for Cohort {cohort}',
                       ha='center', va='center', fontsize=14)
                continue

            geno_engaged = {}

            for r in data:
                g = r['genotype']
                if g not in geno_engaged:
                    geno_engaged[g] = []

                # Total engaged time
                engaged_time = sum(info['occupancy'] for info in r['behavior_states'].values()
                                 if info['classification'] == 'Engaged')
                geno_engaged[g].append(engaged_time * 100)

            genotypes = [g for g in GENOTYPE_ORDER if g in geno_engaged]
            means = [np.mean(geno_engaged[g]) for g in genotypes]
            sems = [stats.sem(geno_engaged[g]) for g in genotypes]

            x = np.arange(len(genotypes))
            bars = ax.bar(x, means, yerr=sems, capsize=8, alpha=0.7, edgecolor='black', linewidth=2)

            for bar, genotype in zip(bars, genotypes):
                bar.set_color(get_genotype_color(genotype))

            ax.set_xlabel('Genotype', fontsize=14, fontweight='bold')
            ax.set_ylabel('% Time Engaged (≥65% accuracy)', fontsize=14, fontweight='bold')
            ax.set_title(f'Cohort {cohort}', fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(genotypes, rotation=45, ha='right', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 100])

            for i, g in enumerate(genotypes):
                n = len(geno_engaged[g])
                ax.text(i, -10, f'n={n}', ha='center', fontsize=11, fontweight='bold')

        plt.suptitle('Engaged Performance During Reversal by Genotype (Phase 2)',
                     fontsize=18, fontweight='bold')
        plt.tight_layout()

        self.save_figure('fig2_engaged_time_by_genotype')
        print(f"  ✓ Saved")

    def fig3_example_trajectories_colored(self, data_W, data_F):
        """Figure 3: Example behavioral trajectories colored by state."""
        print("\nGenerating Fig 3: Example Trajectories with Behavioral State Colors...")

        # Select 2 examples from each cohort
        examples = []
        for cohort, data in [('W', data_W), ('F', data_F)]:
            if len(data) >= 2:
                # Pick animals with diverse behavior
                data_sorted = sorted(data, key=lambda r: sum(info['occupancy']
                                    for info in r['behavior_states'].values()
                                    if info['classification'] == 'Engaged'), reverse=True)
                examples.append((cohort, data_sorted[0]))  # High performer
                examples.append((cohort, data_sorted[-1]))  # Low performer

        if len(examples) == 0:
            print("  ⚠ No data for example trajectories")
            return

        n_examples = len(examples)
        fig, axes = plt.subplots(n_examples, 1, figsize=(18, 4*n_examples))
        if n_examples == 1:
            axes = [axes]

        for ax, (cohort, r) in zip(axes, examples):
            traj = r['trajectory_df']

            # Create colored trajectory
            trial_nums = np.arange(len(traj))

            for state in traj['state'].unique():
                state_data = traj[traj['state'] == state]
                color = r['behavior_states'][state]['color']
                classification = r['behavior_states'][state]['classification']

                # Plot each bout
                for _, bout in state_data.iterrows():
                    bout_start = bout.name
                    bout_length = int(bout['bout_length'])
                    ax.axvspan(bout_start, bout_start + bout_length,
                             color=color, alpha=0.6, linewidth=0)

            # Overlay accuracy
            acc_window = 50  # Rolling window
            rolling_acc = traj['during_accuracy'].rolling(window=acc_window, min_periods=1).mean()
            ax2 = ax.twinx()
            ax2.plot(trial_nums, rolling_acc, 'k-', linewidth=2, alpha=0.7)
            ax2.set_ylabel('Rolling Accuracy (50 trials)', fontsize=12, fontweight='bold')
            ax2.set_ylim([0, 1])
            ax2.axhline(y=0.65, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
            ax2.axhline(y=0.20, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', alpha=0.6, label='Engaged (≥65%)'),
                Patch(facecolor='#f39c12', alpha=0.6, label='Mixed (20-65%)'),
                Patch(facecolor='#e74c3c', alpha=0.6, label='Lapsed (≤20%)')
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)

            ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
            ax.set_ylabel('State', fontsize=12, fontweight='bold')
            animal_id = r['animal_id']
            genotype = r['genotype']
            ax.set_title(f'Animal {animal_id} ({genotype}, Cohort {cohort})',
                        fontsize=14, fontweight='bold', color=get_genotype_color(genotype))
            ax.set_xlim([0, len(traj)])
            ax.set_yticks([])

        plt.suptitle('Behavioral State Trajectories During Reversal (Phase 2)\nColored by Accuracy-Based Classification',
                     fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()

        self.save_figure('fig3_example_trajectories_colored')
        print(f"  ✓ Saved")

    def fig4_individual_animal_summary(self, data_W, data_F, target_animals):
        """Figure 4: Summary for specific target animals."""
        print("\nGenerating Fig 4: Target Animal Summary...")

        # Find target animals
        all_data = {r['animal_id']: r for r in data_W + data_F}
        found_animals = [(aid, all_data[aid]) for aid in target_animals if aid in all_data]

        if len(found_animals) == 0:
            print("  ⚠ None of the target animals found")
            return

        n_animals = len(found_animals)
        fig, axes = plt.subplots(n_animals, 2, figsize=(16, 4*n_animals))
        if n_animals == 1:
            axes = axes.reshape(1, -1)

        for row, (animal_id, r) in enumerate(found_animals):
            # Panel A: State classification pie chart
            ax = axes[row, 0]

            behavior_pcts = {}
            colors = []
            labels = []

            for state, info in r['behavior_states'].items():
                classification = info['classification']
                occupancy = info['occupancy'] * 100
                accuracy = info['accuracy']

                label = f"{classification}\n{occupancy:.1f}% time\n{accuracy:.0%} acc"
                labels.append(label)

                if classification not in behavior_pcts:
                    behavior_pcts[classification] = 0
                behavior_pcts[classification] += occupancy

                colors.append(info['color'])

            values = [behavior_pcts.get(c, 0) for c in ['Engaged', 'Mixed', 'Lapsed']]
            pie_colors = ['#2ecc71', '#f39c12', '#e74c3c']

            ax.pie(values, labels=['Engaged', 'Mixed', 'Lapsed'], colors=pie_colors,
                  autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
            genotype = r['genotype']
            ax.set_title(f'{animal_id} ({genotype})\nBehavioral State Distribution',
                        fontsize=13, fontweight='bold', color=get_genotype_color(genotype))

            # Panel B: State details table
            ax = axes[row, 1]
            ax.axis('off')

            table_data = []
            table_data.append(['State #', 'Classification', 'Accuracy', '% Time'])

            for state in sorted(r['behavior_states'].keys()):
                info = r['behavior_states'][state]
                table_data.append([
                    f'{int(state)}',
                    info['classification'],
                    f"{info['accuracy']:.1%}",
                    f"{info['occupancy']*100:.1f}%"
                ])

            table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                           colWidths=[0.15, 0.35, 0.25, 0.25])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2)

            # Color header row
            for i in range(4):
                table[(0, i)].set_facecolor('#cccccc')
                table[(0, i)].set_text_props(weight='bold')

            # Color classification column
            for i, row_data in enumerate(table_data[1:], 1):
                classification = row_data[1]
                if classification == 'Engaged':
                    table[(i, 1)].set_facecolor('#d5f4e6')
                elif classification == 'Lapsed':
                    table[(i, 1)].set_facecolor('#fadbd8')
                else:
                    table[(i, 1)].set_facecolor('#fdebd0')

            ax.set_title(f'{animal_id} State Details',
                        fontsize=13, fontweight='bold')

        plt.suptitle('Target Animals: Behavioral State Summary (Phase 2)',
                     fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()

        self.save_figure('fig4_target_animals_summary')
        print(f"  ✓ Saved")

    def save_figure(self, filename):
        """Save figure as PNG and PDF."""
        plt.savefig(self.output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f"{filename}.pdf", bbox_inches='tight')
        plt.close()

    def run_all(self, target_animals=None):
        """Run all Phase 2 behavior-based figure generation."""
        print("\n" + "="*80)
        print("PHASE 2 BEHAVIOR-BASED FIGURE REGENERATION")
        print("="*80)
        print("\nAccuracy Thresholds:")
        print("  Engaged: ≥0.65 (65%)")
        print("  Mixed: 0.20-0.65")
        print("  Lapsed: ≤0.20 (20%)")
        print("="*80)

        # Load data
        data_W, data_F = self.load_phase2_data()

        if len(data_W) == 0 and len(data_F) == 0:
            print("\n⚠ No Phase 2 data available")
            return

        # Generate figures
        print("\n" + "="*80)
        print("GENERATING FIGURES")
        print("="*80)

        self.fig1_state_classification_summary(data_W, data_F)
        self.fig2_engaged_time_by_genotype(data_W, data_F)
        self.fig3_example_trajectories_colored(data_W, data_F)

        if target_animals:
            self.fig4_individual_animal_summary(data_W, data_F, target_animals)

        print("\n" + "="*80)
        print("✓ PHASE 2 BEHAVIOR-BASED FIGURES COMPLETE!")
        print("="*80)
        print(f"\nOutput: {self.output_dir}")
        print("\nGenerated:")
        print("  Fig 1: Behavioral state classification by genotype")
        print("  Fig 2: Engaged time by genotype")
        print("  Fig 3: Example trajectories colored by behavior")
        if target_animals:
            print("  Fig 4: Target animal summary")


def main():
    """Main execution."""
    # Target animals requested by user
    target_animals = ['c2m3', 'c4m1', 'c2m5', 91, 32, 61]

    regenerator = Phase2BehaviorBasedRegenerator()
    regenerator.run_all(target_animals=target_animals)


if __name__ == '__main__':
    main()
