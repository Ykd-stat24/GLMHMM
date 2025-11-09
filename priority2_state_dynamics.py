"""
Priority 2: P(State) vs Trial Number by Genotype
=================================================

Creates faceted plots showing the probability of each state over trials,
averaged by genotype with variability bands.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# Import utilities
import sys
sys.path.insert(0, '/home/user/GLMHMM')
from state_validation import create_broad_state_categories
from glmhmm_utils import load_and_preprocess_session_data

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


class StateDynamicsVisualizer:
    """Generate state probability over trial visualizations."""

    def __init__(self, results_dir='results/phase1_non_reversal'):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'priority2_dynamics'
        self.output_dir.mkdir(exist_ok=True)

    def load_data_with_states(self, cohort, animals):
        """Load model results with state sequences."""
        results = []
        all_trials = []

        # Determine data file
        if cohort == 'W':
            data_file = '/home/user/GLMHMM/W LD Data 11.08 All_processed.csv'
        else:
            data_file = '/home/user/GLMHMM/F LD Data 11.08 All_processed.csv'

        # Load trial data
        trial_df = load_and_preprocess_session_data(data_file)

        for animal in animals:
            pkl_file = self.results_dir / f'{animal}_cohort{cohort}_model.pkl'
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)

                    # Add broad categories
                    broad_categories = create_broad_state_categories(data['validated_labels'])
                    data['broad_categories'] = broad_categories

                    results.append(data)

                    # Get trial data
                    animal_trials = trial_df[trial_df['animal_id'] == animal].copy()
                    if len(animal_trials) > 0:
                        # Match state sequence length
                        n_states = len(data['model'].most_likely_states)
                        if n_states <= len(animal_trials):
                            animal_trials_matched = animal_trials.iloc[:n_states].copy()
                            animal_trials_matched['glmhmm_state'] = data['model'].most_likely_states
                            all_trials.append(animal_trials_matched)

        # Combine
        if len(all_trials) > 0:
            combined = pd.concat(all_trials, ignore_index=True)
        else:
            combined = pd.DataFrame()

        return results, combined

    def compute_state_probabilities(self, results, trials):
        """
        Compute P(state) over trials for each animal, then average by genotype.

        Returns: Dictionary mapping genotype -> state probabilities over time
        """
        genotype_data = {}

        for r in results:
            animal_id = r['animal_id']
            genotype = r['genotype']
            n_states = r['model'].n_states

            # Get this animal's trials
            animal_trials = trials[trials['animal_id'] == animal_id].copy()
            if len(animal_trials) == 0:
                continue

            # Get state sequence
            state_seq = animal_trials['glmhmm_state'].values
            n_trials = len(state_seq)

            # Compute probability of each state over trials using sliding window
            window_size = 50
            state_probs = np.zeros((n_trials, n_states))

            for t in range(n_trials):
                window_start = max(0, t - window_size // 2)
                window_end = min(n_trials, t + window_size // 2)
                window_states = state_seq[window_start:window_end]

                for s in range(n_states):
                    state_probs[t, s] = np.mean(window_states == s)

            # Store by genotype
            if genotype not in genotype_data:
                genotype_data[genotype] = {
                    'state_probs': [],
                    'broad_categories': r['broad_categories'],
                    'validated_labels': r['validated_labels']
                }

            genotype_data[genotype]['state_probs'].append(state_probs)

        # Average across animals within each genotype
        for genotype in genotype_data:
            probs_list = genotype_data[genotype]['state_probs']

            # Find max length
            max_len = max(len(p) for p in probs_list)

            # Pad shorter sequences with NaN
            padded_probs = []
            for probs in probs_list:
                if len(probs) < max_len:
                    padding = np.full((max_len - len(probs), probs.shape[1]), np.nan)
                    probs_padded = np.vstack([probs, padding])
                else:
                    probs_padded = probs
                padded_probs.append(probs_padded)

            # Stack and compute mean/sem
            probs_array = np.stack(padded_probs, axis=0)  # (n_animals, n_trials, n_states)

            genotype_data[genotype]['mean_probs'] = np.nanmean(probs_array, axis=0)
            genotype_data[genotype]['sem_probs'] = np.nanstd(probs_array, axis=0) / np.sqrt(
                np.sum(~np.isnan(probs_array), axis=0)
            )

        return genotype_data

    def plot_state_probabilities_by_genotype(self, results_W, trials_W,
                                             results_F, trials_F):
        """
        Create faceted plot showing P(state) over trials for each genotype.
        """
        # Compute state probabilities
        print("Computing state probabilities...")
        genotype_data_W = self.compute_state_probabilities(results_W, trials_W)
        genotype_data_F = self.compute_state_probabilities(results_F, trials_F)

        # Combine both cohorts
        all_genotype_data = {}
        for g, data in genotype_data_W.items():
            all_genotype_data[f'W-{g}'] = data
        for g, data in genotype_data_F.items():
            all_genotype_data[f'F-{g}'] = data

        # Sort genotypes
        genotypes_sorted = sorted(all_genotype_data.keys())
        n_genotypes = len(genotypes_sorted)

        # Create figure with subplots
        fig, axes = plt.subplots(n_genotypes, 1, figsize=(14, 4*n_genotypes),
                                sharex=True)
        if n_genotypes == 1:
            axes = [axes]

        # Define colors for states
        state_colors = ['#2ecc71', '#e74c3c', '#f39c12']

        for idx, genotype in enumerate(genotypes_sorted):
            ax = axes[idx]
            data = all_genotype_data[genotype]

            mean_probs = data['mean_probs']
            sem_probs = data['sem_probs']
            broad_categories = data['broad_categories']
            validated_labels = data['validated_labels']

            n_trials = mean_probs.shape[0]
            x = np.arange(n_trials)

            # Plot each state
            for state in range(mean_probs.shape[1]):
                # Get state label
                broad_cat, detailed_label, conf = broad_categories[state]
                label = f'State {state}: {broad_cat}'

                # Get color based on broad category
                if broad_cat == 'Engaged':
                    color = '#2ecc71'
                elif broad_cat == 'Lapsed':
                    color = '#e74c3c'
                else:  # Mixed
                    color = '#f39c12'

                # Plot mean
                y = mean_probs[:, state]
                sem = sem_probs[:, state]

                # Smooth for visualization
                y_smooth = gaussian_filter1d(y[~np.isnan(y)], sigma=2)
                sem_smooth = gaussian_filter1d(sem[~np.isnan(sem)], sigma=2)
                x_valid = x[~np.isnan(y)][:len(y_smooth)]

                ax.plot(x_valid, y_smooth, linewidth=2.5, label=label,
                       color=color, alpha=0.9)
                ax.fill_between(x_valid,
                               np.maximum(0, y_smooth - sem_smooth),
                               np.minimum(1, y_smooth + sem_smooth),
                               color=color, alpha=0.2)

            # Formatting
            ax.set_ylabel('P(State)', fontsize=12, fontweight='bold')
            ax.set_title(f'{genotype}', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=10, framealpha=0.9)
            ax.grid(alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

            # Add sample size
            n_animals = len(data['state_probs'])
            ax.text(0.98, 0.02, f'n={n_animals}', transform=ax.transAxes,
                   ha='right', va='bottom', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        axes[-1].set_xlabel('Trial Number', fontsize=13, fontweight='bold')

        fig.suptitle('State Dynamics: P(State) Over Trials by Genotype',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save
        plt.savefig(self.output_dir / 'pstate_vs_trial_by_genotype.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'pstate_vs_trial_by_genotype.pdf',
                   bbox_inches='tight')
        plt.close()

        print(f"✓ Created P(state) vs trial plot by genotype")
        print(f"  Output: {self.output_dir}")

    def plot_state_probabilities_by_session(self, results_W, trials_W,
                                           results_F, trials_F):
        """
        Alternative view: P(state) over session number instead of absolute trial.
        """
        # Compute state probabilities by session
        print("Computing state probabilities by session...")

        all_data = []

        for results, trials, cohort in [(results_W, trials_W, 'W'),
                                        (results_F, trials_F, 'F')]:
            if len(trials) == 0:
                continue

            # Add session number
            trials_copy = trials.copy()
            trials_copy = trials_copy.sort_values(['animal_id', 'session_date'])
            trials_copy['session_num'] = trials_copy.groupby('animal_id').cumcount() // 30

            for r in results:
                animal_id = r['animal_id']
                genotype = r['genotype']
                n_states = r['model'].n_states
                broad_categories = create_broad_state_categories(r['validated_labels'])

                animal_trials = trials_copy[trials_copy['animal_id'] == animal_id].copy()
                if len(animal_trials) == 0:
                    continue

                # Group by session
                for session_num, session_trials in animal_trials.groupby('session_num'):
                    state_seq = session_trials['glmhmm_state'].values

                    for s in range(n_states):
                        broad_cat, detailed_label, conf = broad_categories[s]

                        all_data.append({
                            'animal_id': animal_id,
                            'cohort': cohort,
                            'genotype': genotype,
                            'session_num': session_num,
                            'state': s,
                            'broad_category': broad_cat,
                            'p_state': np.mean(state_seq == s)
                        })

        df = pd.DataFrame(all_data)

        if len(df) == 0:
            print("  No data available for session-based plot")
            return

        # Create genotype labels
        df['genotype_label'] = df['cohort'] + '-' + df['genotype']

        # Plot
        genotypes = sorted(df['genotype_label'].unique())
        n_genotypes = len(genotypes)

        fig, axes = plt.subplots(n_genotypes, 1, figsize=(14, 4*n_genotypes),
                                sharex=True)
        if n_genotypes == 1:
            axes = [axes]

        for idx, genotype in enumerate(genotypes):
            ax = axes[idx]
            geno_data = df[df['genotype_label'] == genotype]

            # Plot each state
            for broad_cat, color in [('Engaged', '#2ecc71'),
                                     ('Lapsed', '#e74c3c'),
                                     ('Mixed', '#f39c12')]:
                cat_data = geno_data[geno_data['broad_category'] == broad_cat]

                if len(cat_data) == 0:
                    continue

                # Average across animals and states with same category
                session_stats = cat_data.groupby('session_num').agg({
                    'p_state': ['mean', 'sem']
                }).reset_index()

                x = session_stats['session_num']
                y = session_stats['p_state']['mean']
                sem = session_stats['p_state']['sem']

                ax.plot(x, y, linewidth=2.5, label=broad_cat,
                       color=color, alpha=0.9, marker='o', markersize=4)
                ax.fill_between(x,
                               np.maximum(0, y - sem),
                               np.minimum(1, y + sem),
                               color=color, alpha=0.2)

            ax.set_ylabel('P(State)', fontsize=12, fontweight='bold')
            ax.set_title(f'{genotype}', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

            # Add sample size
            n_animals = geno_data['animal_id'].nunique()
            ax.text(0.98, 0.02, f'n={n_animals}', transform=ax.transAxes,
                   ha='right', va='bottom', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        axes[-1].set_xlabel('Session Number', fontsize=13, fontweight='bold')

        fig.suptitle('State Dynamics: P(State) Over Sessions by Genotype',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save
        plt.savefig(self.output_dir / 'pstate_vs_session_by_genotype.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'pstate_vs_session_by_genotype.pdf',
                   bbox_inches='tight')
        plt.close()

        print(f"✓ Created P(state) vs session plot by genotype")


def main():
    """Run state dynamics visualizations."""
    print("="*80)
    print("PRIORITY 2: STATE DYNAMICS VISUALIZATIONS")
    print("="*80)

    viz = StateDynamicsVisualizer()

    # Define animals
    animals_W = [f'c{c}m{m}' for c in range(1, 5) for m in range(1, 6)
                 if not (c == 1 and m == 5)]
    # F cohort uses integer IDs
    animals_F = [11, 12, 13, 14, 21, 22, 23, 24, 25,
                 31, 32, 33, 34, 41, 42, 51, 52,
                 61, 62, 63, 64, 71, 72, 73,
                 81, 82, 83, 84, 91, 92, 93,
                 101, 102, 103, 104]

    # Load data
    print("\nLoading data...")
    results_W, trials_W = viz.load_data_with_states('W', animals_W)
    results_F, trials_F = viz.load_data_with_states('F', animals_F)
    print(f"  Cohort W: {len(results_W)} animals, {len(trials_W)} trials")
    print(f"  Cohort F: {len(results_F)} animals, {len(trials_F)} trials")

    # Generate visualizations
    print("\nGenerating visualizations...")
    viz.plot_state_probabilities_by_genotype(results_W, trials_W,
                                             results_F, trials_F)
    viz.plot_state_probabilities_by_session(results_W, trials_W,
                                            results_F, trials_F)

    print("\n" + "="*80)
    print("✓ STATE DYNAMICS VISUALIZATIONS COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
