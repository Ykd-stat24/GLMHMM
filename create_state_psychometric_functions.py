#!/usr/bin/env python3
"""
State-Specific Psychometric Functions for Phase 2 Reversal Learning
====================================================================

Creates Ashwood et al. style psychometric functions showing how choice
probability depends on stimulus/context in different behavioral states.

Analyzes:
1. P(correct) vs trial position in reversal - by state
2. P(stay) after reward vs no reward - by state
3. P(correct) vs recent performance - by state
4. State occupancy dynamics through reversal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy import stats
from scipy.ndimage import uniform_filter1d

# Set style
plt.style.use('default')
sns.set_palette("husl")

class StatePsychometricAnalyzer:
    """Analyze state-specific psychometric functions."""

    def __init__(self):
        self.results_dir = Path('results/phase2_reversal/state_psychometrics')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = Path('results/phase2_reversal/models')

        # State colors
        self.state_colors = {
            0: '#e74c3c',  # Red
            1: '#3498db',  # Blue
            2: '#2ecc71'   # Green
        }

        # State labels based on validation
        self.state_labels = {
            0: 'State 0',
            1: 'State 1',
            2: 'State 2'
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

    def extract_trial_data(self, model):
        """Extract trial-by-trial data with states."""
        outcomes = model['y']
        correct = model['metadata']['correct']
        states = model['states']

        # Derive choices from correct and outcomes
        choices = np.where(correct == 1, outcomes, 1 - outcomes)

        # Get features
        prev_choice = model['X'][:, 1]  # Previous choice

        # Calculate choice changes (stay vs switch)
        stays = np.zeros(len(choices))
        stays[1:] = (choices[1:] == choices[:-1]).astype(int)

        # Previous outcomes
        prev_outcomes = np.zeros(len(outcomes))
        prev_outcomes[1:] = outcomes[:-1]

        return {
            'choices': choices,
            'outcomes': outcomes,
            'correct': correct,
            'states': states,
            'stays': stays,
            'prev_outcomes': prev_outcomes,
            'trial_num': np.arange(len(choices))
        }

    def calculate_psychometric_by_state(self, models, bin_by, metric='correct', n_bins=10):
        """
        Calculate psychometric function by state.

        Parameters:
        -----------
        bin_by : str
            Variable to bin by: 'trial_num', 'recent_perf', 'trial_blocks'
        metric : str
            What to calculate: 'correct', 'stay_after_reward', 'stay_after_no_reward'
        """
        all_data = []

        for model in models:
            data = self.extract_trial_data(model)

            # Create bins
            if bin_by == 'trial_num':
                bins = np.linspace(0, len(data['trial_num']), n_bins + 1)
                bin_indices = np.digitize(data['trial_num'], bins) - 1
                bin_centers = (bins[:-1] + bins[1:]) / 2

            elif bin_by == 'trial_blocks':
                # Fixed size blocks
                block_size = 50
                bin_indices = data['trial_num'] // block_size
                bin_centers = np.unique(bin_indices) * block_size + block_size/2

            elif bin_by == 'recent_perf':
                # Recent performance (last 10 trials)
                window = 10
                recent_perf = uniform_filter1d(data['correct'].astype(float),
                                               size=window, mode='nearest')
                bins = np.linspace(0, 1, n_bins + 1)
                bin_indices = np.digitize(recent_perf, bins) - 1
                bin_centers = (bins[:-1] + bins[1:]) / 2

            # Calculate metric for each bin and state
            for state in range(3):
                state_mask = data['states'] == state

                for bin_idx in np.unique(bin_indices):
                    bin_mask = bin_indices == bin_idx
                    mask = state_mask & bin_mask

                    if np.sum(mask) < 5:  # Minimum trials
                        continue

                    if metric == 'correct':
                        value = np.mean(data['correct'][mask])
                    elif metric == 'stay_after_reward':
                        # Look at trials after reward
                        reward_mask = mask & (data['prev_outcomes'] == 1)
                        if np.sum(reward_mask[1:]) < 3:
                            continue
                        value = np.mean(data['stays'][1:][reward_mask[1:]])
                    elif metric == 'stay_after_no_reward':
                        # Look at trials after no reward
                        no_reward_mask = mask & (data['prev_outcomes'] == 0)
                        if np.sum(no_reward_mask[1:]) < 3:
                            continue
                        value = np.mean(data['stays'][1:][no_reward_mask[1:]])

                    all_data.append({
                        'animal': model['animal_id'],
                        'genotype': model['genotype'],
                        'cohort': model['cohort'],
                        'state': state,
                        'bin': bin_idx,
                        'bin_center': bin_centers[bin_idx] if bin_idx < len(bin_centers) else bin_centers[-1],
                        'value': value,
                        'n_trials': np.sum(mask)
                    })

        return pd.DataFrame(all_data)

    def create_figure1_psychometric_by_trial(self, models):
        """P(correct) as a function of trial position, by state."""
        print("Creating psychometric functions by trial position...")

        df = self.calculate_psychometric_by_state(models, bin_by='trial_blocks',
                                                   metric='correct', n_bins=10)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: All states
        ax = axes[0]
        for state in [0, 1, 2]:
            state_df = df[df['state'] == state]

            # Group by bin and calculate mean ± SEM
            grouped = state_df.groupby('bin_center')['value'].agg(['mean', 'sem', 'count'])

            ax.plot(grouped.index, grouped['mean'],
                   color=self.state_colors[state], linewidth=2.5,
                   marker='o', markersize=6, label=self.state_labels[state])
            ax.fill_between(grouped.index,
                           grouped['mean'] - grouped['sem'],
                           grouped['mean'] + grouped['sem'],
                           alpha=0.2, color=self.state_colors[state])

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('Trial Number in Reversal', fontsize=12)
        ax.set_ylabel('P(Correct Choice)', fontsize=12)
        ax.set_title('A. State-Specific Performance Through Reversal',
                    fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Panel B: By genotype (collapsed across states, weighted by occupancy)
        ax = axes[1]
        for geno in sorted(df['genotype'].unique()):
            geno_df = df[df['genotype'] == geno]

            # Weight by number of trials in each state
            grouped = geno_df.groupby('bin_center').apply(
                lambda x: np.average(x['value'], weights=x['n_trials'])
            )

            # Also get SEM
            sem = geno_df.groupby('bin_center')['value'].sem()

            ax.plot(grouped.index, grouped.values, linewidth=2,
                   marker='o', markersize=5, label=f'{geno}')
            ax.fill_between(grouped.index,
                           grouped.values - sem.values,
                           grouped.values + sem.values,
                           alpha=0.15)

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('Trial Number in Reversal', fontsize=12)
        ax.set_ylabel('P(Correct Choice)', fontsize=12)
        ax.set_title('B. Genotype Comparison', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.suptitle('Psychometric Functions: P(Correct) vs Trial Position',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        # Save
        plt.savefig(self.results_dir / 'fig1_psychometric_by_trial.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'fig1_psychometric_by_trial.pdf',
                   bbox_inches='tight')
        plt.close()

    def create_figure2_stay_switch_by_state(self, models):
        """P(stay) after reward vs no reward, by state."""
        print("Creating stay/switch psychometric functions...")

        # Calculate stay probabilities
        df_stay_reward = self.calculate_psychometric_by_state(
            models, bin_by='trial_blocks', metric='stay_after_reward', n_bins=10)
        df_stay_no_reward = self.calculate_psychometric_by_state(
            models, bin_by='trial_blocks', metric='stay_after_no_reward', n_bins=10)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for state_idx, state in enumerate([0, 1, 2]):
            ax = axes[state_idx]

            # Stay after reward
            reward_df = df_stay_reward[df_stay_reward['state'] == state]
            grouped_reward = reward_df.groupby('bin_center')['value'].agg(['mean', 'sem'])

            ax.plot(grouped_reward.index, grouped_reward['mean'],
                   color='#2ecc71', linewidth=2.5, marker='o', markersize=6,
                   label='After Reward')
            ax.fill_between(grouped_reward.index,
                           grouped_reward['mean'] - grouped_reward['sem'],
                           grouped_reward['mean'] + grouped_reward['sem'],
                           alpha=0.2, color='#2ecc71')

            # Stay after no reward
            no_reward_df = df_stay_no_reward[df_stay_no_reward['state'] == state]
            grouped_no_reward = no_reward_df.groupby('bin_center')['value'].agg(['mean', 'sem'])

            ax.plot(grouped_no_reward.index, grouped_no_reward['mean'],
                   color='#e74c3c', linewidth=2.5, marker='o', markersize=6,
                   label='After No Reward')
            ax.fill_between(grouped_no_reward.index,
                           grouped_no_reward['mean'] - grouped_no_reward['sem'],
                           grouped_no_reward['mean'] + grouped_no_reward['sem'],
                           alpha=0.2, color='#e74c3c')

            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Trial Number in Reversal', fontsize=11)
            ax.set_ylabel('P(Stay)', fontsize=11)
            ax.set_title(f'{self.state_labels[state]}', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.legend(frameon=False, fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Psychometric Functions: P(Stay) After Outcome, by State',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        # Save
        plt.savefig(self.results_dir / 'fig2_stay_switch_by_state.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'fig2_stay_switch_by_state.pdf',
                   bbox_inches='tight')
        plt.close()

    def create_figure3_state_occupancy_dynamics(self, models):
        """State occupancy through reversal learning."""
        print("Creating state occupancy dynamics...")

        # Collect state occupancies in bins
        all_data = []
        block_size = 50

        for model in models:
            data = self.extract_trial_data(model)
            n_trials = len(data['states'])
            n_blocks = int(np.ceil(n_trials / block_size))

            for block in range(n_blocks):
                start = block * block_size
                end = min((block + 1) * block_size, n_trials)
                block_states = data['states'][start:end]

                for state in range(3):
                    occupancy = np.mean(block_states == state)

                    all_data.append({
                        'animal': model['animal_id'],
                        'genotype': model['genotype'],
                        'block': block,
                        'trial_center': start + block_size/2,
                        'state': state,
                        'occupancy': occupancy
                    })

        df = pd.DataFrame(all_data)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: State occupancy over time
        ax = axes[0]
        for state in [0, 1, 2]:
            state_df = df[df['state'] == state]
            grouped = state_df.groupby('trial_center')['occupancy'].agg(['mean', 'sem'])

            ax.plot(grouped.index, grouped['mean'],
                   color=self.state_colors[state], linewidth=2.5,
                   label=self.state_labels[state])
            ax.fill_between(grouped.index,
                           grouped['mean'] - grouped['sem'],
                           grouped['mean'] + grouped['sem'],
                           alpha=0.2, color=self.state_colors[state])

        ax.axhline(1/3, color='gray', linestyle='--', alpha=0.5, label='Uniform')
        ax.set_xlabel('Trial Number in Reversal', fontsize=12)
        ax.set_ylabel('State Occupancy', fontsize=12)
        ax.set_title('A. State Dynamics Through Reversal', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Panel B: By genotype
        ax = axes[1]
        for geno in sorted(df['genotype'].unique()):
            geno_df = df[df['genotype'] == geno]

            # Calculate "engaged" occupancy (combine states with >50% accuracy)
            # For now, just show state 2 occupancy as proxy
            state2_df = geno_df[geno_df['state'] == 2]
            grouped = state2_df.groupby('trial_center')['occupancy'].agg(['mean', 'sem'])

            ax.plot(grouped.index, grouped['mean'], linewidth=2,
                   marker='o', markersize=4, label=f'{geno}')
            ax.fill_between(grouped.index,
                           grouped['mean'] - grouped['sem'],
                           grouped['mean'] + grouped['sem'],
                           alpha=0.15)

        ax.set_xlabel('Trial Number in Reversal', fontsize=12)
        ax.set_ylabel('State 2 Occupancy', fontsize=12)
        ax.set_title('B. State 2 Occupancy by Genotype', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 0.8])
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.suptitle('State Occupancy Dynamics Through Reversal',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        # Save
        plt.savefig(self.results_dir / 'fig3_state_occupancy_dynamics.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'fig3_state_occupancy_dynamics.pdf',
                   bbox_inches='tight')
        plt.close()

    def create_figure4_glm_weights_by_state(self, models):
        """GLM-HMM weights showing how states use stimulus information."""
        print("Creating GLM weight visualizations by state...")

        # Collect weights from all models
        all_weights = {state: [] for state in range(3)}
        feature_names = models[0]['feature_names']

        for model in models:
            glm_hmm = model['model']
            # Get GLM weights (shape: n_states x n_features)
            weights_matrix = glm_hmm.glm_weights

            for state in range(3):
                # Extract weights for this state
                weights = weights_matrix[state, :]
                all_weights[state].append(weights)

        # Convert to arrays and calculate mean ± SEM
        weight_stats = {}
        for state in range(3):
            if len(all_weights[state]) > 0:
                weights_array = np.array(all_weights[state])
                weight_stats[state] = {
                    'mean': np.mean(weights_array, axis=0),
                    'sem': stats.sem(weights_array, axis=0)
                }

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

        for state_idx, state in enumerate([0, 1, 2]):
            ax = axes[state_idx]

            if state in weight_stats:
                means = weight_stats[state]['mean']
                sems = weight_stats[state]['sem']

                x_pos = np.arange(len(feature_names))
                ax.bar(x_pos, means, yerr=sems, color=self.state_colors[state],
                      alpha=0.7, capsize=5)
                ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
                ax.set_ylabel('GLM Weight', fontsize=11)
                ax.set_title(f'{self.state_labels[state]}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('GLM Weights by State: How States Use Stimulus Information',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        # Save
        plt.savefig(self.results_dir / 'fig4_glm_weights_by_state.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'fig4_glm_weights_by_state.pdf',
                   bbox_inches='tight')
        plt.close()

    def run_all(self):
        """Run all psychometric analyses."""
        print("="*80)
        print("STATE-SPECIFIC PSYCHOMETRIC FUNCTIONS")
        print("="*80)

        # Load models
        print("\nLoading Phase 2 models...")
        models = self.load_models()
        print(f"  Loaded {len(models)} animals")

        # Create all figures
        self.create_figure1_psychometric_by_trial(models)
        print("  ✓ Figure 1: P(correct) vs trial position")

        self.create_figure2_stay_switch_by_state(models)
        print("  ✓ Figure 2: P(stay) after reward/no reward by state")

        self.create_figure3_state_occupancy_dynamics(models)
        print("  ✓ Figure 3: State occupancy dynamics")

        self.create_figure4_glm_weights_by_state(models)
        print("  ✓ Figure 4: GLM weights by state")

        print("\n" + "="*80)
        print("PSYCHOMETRIC ANALYSES COMPLETE")
        print("="*80)
        print(f"\nAll results saved to: {self.results_dir}")
        print("\nGenerated files:")
        for f in sorted(self.results_dir.glob('*')):
            print(f"  - {f.name}")
        print()

if __name__ == '__main__':
    analyzer = StatePsychometricAnalyzer()
    analyzer.run_all()
