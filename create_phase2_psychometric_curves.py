#!/usr/bin/env python3
"""
Create comprehensive psychometric learning curves for Phase 2 reversal learning.

Generates learning trajectory visualizations with multiple groupings:
- Individual animals by genotype
- Genotype-averaged trajectories
- Cohort-averaged trajectories
- High vs Low performer trajectories
- State-specific performance curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy.ndimage import uniform_filter1d

# Set style
plt.style.use('default')
sns.set_palette("husl")

class PsychometricCurveGenerator:
    """Generate comprehensive psychometric learning curves for Phase 2."""

    def __init__(self):
        self.results_dir = Path('results/phase2_reversal/psychometric_curves')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = Path('results/phase2_reversal/models')

        # Color schemes
        self.genotype_colors = {
            '-': '#1f77b4',      # Blue
            '+': '#ff7f0e',      # Orange
            '+/-': '#2ca02c',    # Green
            '+/+': '#d62728',    # Red
            '-/-': '#9467bd'     # Purple
        }

        self.cohort_colors = {
            'W': '#e74c3c',  # Red
            'F': '#3498db'   # Blue
        }

    def load_models(self):
        """Load Phase 2 fitted models."""
        models = []
        if self.models_dir.exists():
            for model_file in sorted(self.models_dir.glob('*_reversal.pkl')):
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                    models.append(model)
        return models

    def calculate_rolling_accuracy(self, choices, outcomes, window=30):
        """Calculate rolling window accuracy."""
        n_trials = len(choices)
        correct = (choices == outcomes).astype(float)

        # Use uniform filter for rolling average
        if n_trials < window:
            return np.full(n_trials, np.mean(correct))

        rolling_acc = uniform_filter1d(correct, size=window, mode='nearest')
        return rolling_acc

    def get_state_performance(self, model, window=30):
        """Get performance trajectory colored by state."""
        # Phase 2 models store data differently
        outcomes = model['y']
        correct = model['metadata']['correct']

        # Derive choices from correct and outcomes
        # If correct, choice = outcome; if incorrect, choice = 1 - outcome
        choices = np.where(correct == 1, outcomes, 1 - outcomes)

        states = model['states']

        # Calculate rolling accuracy
        rolling_acc = self.calculate_rolling_accuracy(choices, outcomes, window)

        return rolling_acc, states

    def create_figure1_individual_by_genotype(self, models):
        """Individual animal psychometric curves grouped by genotype."""
        print("Creating individual curves by genotype...")

        # Group by genotype
        genotype_groups = {}
        for model in models:
            geno = model['genotype']
            if geno not in genotype_groups:
                genotype_groups[geno] = []
            genotype_groups[geno].append(model)

        # Create figure with subplots for each genotype
        n_genotypes = len(genotype_groups)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (geno, geno_models) in enumerate(sorted(genotype_groups.items())):
            ax = axes[idx]

            for model in geno_models:
                rolling_acc, _ = self.get_state_performance(model, window=30)
                trials = np.arange(len(rolling_acc))

                ax.plot(trials, rolling_acc, alpha=0.3, linewidth=1,
                       color=self.genotype_colors[geno])

            # Add mean trajectory
            all_accs = []
            max_len = max(len(m['y']) for m in geno_models)

            for model in geno_models:
                rolling_acc, _ = self.get_state_performance(model, window=30)
                # Pad to max length
                padded = np.full(max_len, np.nan)
                padded[:len(rolling_acc)] = rolling_acc
                all_accs.append(padded)

            mean_acc = np.nanmean(all_accs, axis=0)
            sem_acc = np.nanstd(all_accs, axis=0) / np.sqrt(np.sum(~np.isnan(all_accs), axis=0))

            trials = np.arange(len(mean_acc))
            ax.plot(trials, mean_acc, color=self.genotype_colors[geno],
                   linewidth=3, label=f'Mean (n={len(geno_models)})')
            ax.fill_between(trials, mean_acc - sem_acc, mean_acc + sem_acc,
                          alpha=0.2, color=self.genotype_colors[geno])

            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Trial', fontsize=10)
            ax.set_ylabel('Accuracy (30-trial avg)', fontsize=10)
            ax.set_title(f'Genotype {geno} (n={len(geno_models)})', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.legend(frameon=False, fontsize=9)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_genotypes, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Phase 2 Reversal: Individual Psychometric Curves by Genotype',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save
        plt.savefig(self.results_dir / 'fig1_individual_by_genotype.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'fig1_individual_by_genotype.pdf', bbox_inches='tight')
        plt.close()

    def create_figure2_genotype_averaged(self, models):
        """Genotype-averaged psychometric curves."""
        print("Creating genotype-averaged curves...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Group by genotype
        genotype_groups = {}
        for model in models:
            geno = model['genotype']
            if geno not in genotype_groups:
                genotype_groups[geno] = []
            genotype_groups[geno].append(model)

        # Panel A: All genotypes
        ax = axes[0]
        for geno, geno_models in sorted(genotype_groups.items()):
            all_accs = []
            max_len = max(len(m['y']) for m in geno_models)

            for model in geno_models:
                rolling_acc, _ = self.get_state_performance(model, window=30)
                padded = np.full(max_len, np.nan)
                padded[:len(rolling_acc)] = rolling_acc
                all_accs.append(padded)

            mean_acc = np.nanmean(all_accs, axis=0)
            sem_acc = np.nanstd(all_accs, axis=0) / np.sqrt(np.sum(~np.isnan(all_accs), axis=0))

            trials = np.arange(len(mean_acc))
            ax.plot(trials, mean_acc, color=self.genotype_colors[geno],
                   linewidth=2.5, label=f'{geno} (n={len(geno_models)})')
            ax.fill_between(trials, mean_acc - sem_acc, mean_acc + sem_acc,
                          alpha=0.2, color=self.genotype_colors[geno])

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('Trial Number', fontsize=12)
        ax.set_ylabel('Accuracy (30-trial rolling avg)', fontsize=12)
        ax.set_title('A. Genotype Comparison', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Panel B: By cohort
        ax = axes[1]
        for cohort in ['W', 'F']:
            cohort_models = [m for m in models if m['cohort'] == cohort]

            if len(cohort_models) == 0:
                continue

            all_accs = []
            max_len = max(len(m['y']) for m in cohort_models)

            for model in cohort_models:
                rolling_acc, _ = self.get_state_performance(model, window=30)
                padded = np.full(max_len, np.nan)
                padded[:len(rolling_acc)] = rolling_acc
                all_accs.append(padded)

            mean_acc = np.nanmean(all_accs, axis=0)
            sem_acc = np.nanstd(all_accs, axis=0) / np.sqrt(np.sum(~np.isnan(all_accs), axis=0))

            trials = np.arange(len(mean_acc))
            ax.plot(trials, mean_acc, color=self.cohort_colors[cohort],
                   linewidth=2.5, label=f'Cohort {cohort} (n={len(cohort_models)})')
            ax.fill_between(trials, mean_acc - sem_acc, mean_acc + sem_acc,
                          alpha=0.2, color=self.cohort_colors[cohort])

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('Trial Number', fontsize=12)
        ax.set_ylabel('Accuracy (30-trial rolling avg)', fontsize=12)
        ax.set_title('B. Cohort Comparison', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.suptitle('Phase 2 Reversal: Averaged Psychometric Curves',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        # Save
        plt.savefig(self.results_dir / 'fig2_genotype_averaged.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'fig2_genotype_averaged.pdf', bbox_inches='tight')
        plt.close()

    def create_figure3_performance_groups(self, models):
        """Psychometric curves for high vs low performers."""
        print("Creating performance group curves...")

        # Calculate overall accuracy for each animal
        accuracies = []
        for model in models:
            acc = np.mean(model['metadata']['correct'])
            accuracies.append((model, acc))

        # Split at median
        median_acc = np.median([a[1] for a in accuracies])
        high_performers = [a[0] for a in accuracies if a[1] >= median_acc]
        low_performers = [a[0] for a in accuracies if a[1] < median_acc]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: High vs Low performers
        ax = axes[0]

        for group, label, color in [(high_performers, 'High Performers', '#2ecc71'),
                                     (low_performers, 'Low Performers', '#e74c3c')]:
            all_accs = []
            max_len = max(len(m['y']) for m in group)

            for model in group:
                rolling_acc, _ = self.get_state_performance(model, window=30)
                padded = np.full(max_len, np.nan)
                padded[:len(rolling_acc)] = rolling_acc
                all_accs.append(padded)

            mean_acc = np.nanmean(all_accs, axis=0)
            sem_acc = np.nanstd(all_accs, axis=0) / np.sqrt(np.sum(~np.isnan(all_accs), axis=0))

            trials = np.arange(len(mean_acc))
            ax.plot(trials, mean_acc, color=color, linewidth=2.5, label=f'{label} (n={len(group)})')
            ax.fill_between(trials, mean_acc - sem_acc, mean_acc + sem_acc, alpha=0.2, color=color)

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.axhline(median_acc, color='purple', linestyle=':', alpha=0.7,
                  label=f'Median (acc={median_acc:.3f})')
        ax.set_xlabel('Trial Number', fontsize=12)
        ax.set_ylabel('Accuracy (30-trial rolling avg)', fontsize=12)
        ax.set_title('A. High vs Low Performers', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Panel B: Performance groups by genotype
        ax = axes[1]

        for geno in sorted(set(m['genotype'] for m in models)):
            high_geno = [m for m in high_performers if m['genotype'] == geno]
            low_geno = [m for m in low_performers if m['genotype'] == geno]

            if len(high_geno) == 0 and len(low_geno) == 0:
                continue

            # Count
            n_high = len(high_geno)
            n_low = len(low_geno)
            n_total = n_high + n_low

            # Plot
            x_pos = list(sorted(set(m['genotype'] for m in models))).index(geno)
            ax.bar(x_pos - 0.2, n_high, width=0.4, color='#2ecc71', alpha=0.7, label='High' if x_pos == 0 else '')
            ax.bar(x_pos + 0.2, n_low, width=0.4, color='#e74c3c', alpha=0.7, label='Low' if x_pos == 0 else '')

            # Add percentage labels
            if n_high > 0:
                ax.text(x_pos - 0.2, n_high + 0.2, f'{n_high}', ha='center', va='bottom', fontsize=9)
            if n_low > 0:
                ax.text(x_pos + 0.2, n_low + 0.2, f'{n_low}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Genotype', fontsize=12)
        ax.set_ylabel('Number of Animals', fontsize=12)
        ax.set_title('B. Performance Distribution by Genotype', fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(set(m['genotype'] for m in models))))
        ax.set_xticklabels(sorted(set(m['genotype'] for m in models)))
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Phase 2 Reversal: Performance Groups (Median Split at {median_acc:.3f})',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        # Save
        plt.savefig(self.results_dir / 'fig3_performance_groups.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'fig3_performance_groups.pdf', bbox_inches='tight')
        plt.close()

    def create_figure4_state_colored_trajectories(self, models):
        """Example psychometric curves with state coloring."""
        print("Creating state-colored trajectory examples...")

        # Select 6 representative animals (2 per performance level)
        accuracies = [(m, np.mean(m['metadata']['correct'])) for m in models]
        accuracies.sort(key=lambda x: x[1], reverse=True)

        # Get high, medium, and low performers
        n = len(accuracies)
        examples = [
            accuracies[0][0],  # Highest
            accuracies[n//4][0],  # Upper quartile
            accuracies[n//2][0],  # Median
            accuracies[3*n//4][0],  # Lower quartile
            accuracies[-2][0],  # Second lowest
            accuracies[-1][0]  # Lowest
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        state_colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green for states 0,1,2

        for idx, model in enumerate(examples):
            ax = axes[idx]

            # Get performance and states
            rolling_acc, states = self.get_state_performance(model, window=30)
            trials = np.arange(len(rolling_acc))

            # Plot line
            ax.plot(trials, rolling_acc, color='black', linewidth=2, alpha=0.5, zorder=1)

            # Plot state-colored points
            for state in range(3):
                mask = (states == state)
                ax.scatter(trials[mask], rolling_acc[mask], c=state_colors[state],
                          s=10, alpha=0.6, label=f'State {state}', zorder=2)

            # Calculate metrics
            overall_acc = np.mean(model['metadata']['correct'])

            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Trial', fontsize=10)
            ax.set_ylabel('Accuracy (30-trial avg)', fontsize=10)
            ax.set_title(f"{model['animal_id']} ({model['genotype']}, {model['cohort']})\n"
                        f"Overall Acc: {overall_acc:.3f}",
                        fontsize=11, fontweight='bold')
            ax.set_ylim([0, 1])
            if idx == 0:
                ax.legend(frameon=False, fontsize=9, loc='lower right')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Phase 2 Reversal: Example Trajectories with State Coloring',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save
        plt.savefig(self.results_dir / 'fig4_state_colored_examples.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'fig4_state_colored_examples.pdf', bbox_inches='tight')
        plt.close()

    def run_all(self):
        """Run all psychometric curve analyses."""
        print("="*80)
        print("PHASE 2 PSYCHOMETRIC CURVES GENERATION")
        print("="*80)

        # Load models
        print("\nLoading Phase 2 models...")
        models = self.load_models()
        print(f"  Loaded {len(models)} animals")

        # Create all figures
        self.create_figure1_individual_by_genotype(models)
        print("  ✓ Figure 1: Individual curves by genotype")

        self.create_figure2_genotype_averaged(models)
        print("  ✓ Figure 2: Genotype-averaged curves")

        self.create_figure3_performance_groups(models)
        print("  ✓ Figure 3: Performance group curves")

        self.create_figure4_state_colored_trajectories(models)
        print("  ✓ Figure 4: State-colored trajectory examples")

        print("\n" + "="*80)
        print("PSYCHOMETRIC CURVES COMPLETE")
        print("="*80)
        print(f"\nAll results saved to: {self.results_dir}")
        print("\nGenerated files:")
        for f in sorted(self.results_dir.glob('*')):
            print(f"  - {f.name}")
        print()

if __name__ == '__main__':
    generator = PsychometricCurveGenerator()
    generator.run_all()
