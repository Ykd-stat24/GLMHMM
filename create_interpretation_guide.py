"""
Priority 2: GLM Weight Interpretation & Lapse Metrics
======================================================

Creates comprehensive interpretation guides including:
1. Annotated heatmaps of GLM weights
2. Lapse duration and frequency by genotype
3. Interpretation guides for behavioral implications
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import utilities
import sys
sys.path.insert(0, '/home/user/GLMHMM')
from state_validation import create_broad_state_categories
from glmhmm_utils import load_and_preprocess_session_data

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


class InterpretationGuide:
    """Create interpretation guides for GLM-HMM results."""

    def __init__(self, results_dir='results/phase1_non_reversal'):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'priority2_interpretation'
        self.output_dir.mkdir(exist_ok=True)

    def load_all_results(self, cohort, animals):
        """Load all model results."""
        results = []
        trials_list = []

        # Load trial data
        if cohort == 'W':
            data_file = '/home/user/GLMHMM/W LD Data 11.08 All_processed.csv'
        else:
            data_file = '/home/user/GLMHMM/F LD Data 11.08 All_processed.csv'

        trial_df = load_and_preprocess_session_data(data_file)

        for animal in animals:
            pkl_file = self.results_dir / f'{animal}_cohort{cohort}_model.pkl'
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    data['broad_categories'] = create_broad_state_categories(data['validated_labels'])
                    results.append(data)

                    # Get trial data
                    animal_trials = trial_df[trial_df['animal_id'] == animal].copy()
                    if len(animal_trials) > 0:
                        n_states = len(data['model'].most_likely_states)
                        if n_states <= len(animal_trials):
                            animal_trials_matched = animal_trials.iloc[:n_states].copy()
                            animal_trials_matched['glmhmm_state'] = data['model'].most_likely_states
                            animal_trials_matched['state_label'] = animal_trials_matched['glmhmm_state'].map(
                                lambda s: data['broad_categories'][s][0] if s in data['broad_categories'] else 'Unknown'
                            )
                            trials_list.append(animal_trials_matched)

        trials = pd.concat(trials_list, ignore_index=True) if trials_list else pd.DataFrame()
        return results, trials

    def create_annotated_weight_heatmap(self, results):
        """
        Create annotated heatmap showing average GLM weights across animals
        with detailed interpretation.
        """
        # Collect weights by state category
        weight_dict = {'Engaged': [], 'Lapsed': [], 'Mixed': []}
        feature_names = None

        for r in results:
            model = r['model']
            broad_cat = r['broad_categories']

            if feature_names is None:
                feature_names = model.feature_names if model.feature_names else \
                    ['bias', 'prev_choice', 'WSLS', 'session_prog', 'side_bias', 'task_stage', 'experience']

            weights = model.glm_weights  # (n_states, n_features)

            for state_id, (cat, _, _) in broad_cat.items():
                # Convert state_id to int (it may be np.float64)
                try:
                    state_idx = int(state_id)
                    if cat in weight_dict and 0 <= state_idx < weights.shape[0]:
                        weight_dict[cat].append(weights[state_idx, :])
                except (ValueError, TypeError):
                    continue

        # Average weights
        avg_weights = {}
        for cat in ['Engaged', 'Lapsed', 'Mixed']:
            if weight_dict[cat]:
                avg_weights[cat] = np.mean(weight_dict[cat], axis=0)

        if not avg_weights:
            print("  No weight data available")
            return

        # Create figure
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 2, width_ratios=[3, 1], hspace=0.4, wspace=0.3)

        # Main heatmap
        ax_heat = fig.add_subplot(gs[:, 0])

        # Prepare data for heatmap
        categories = ['Engaged', 'Lapsed', 'Mixed']
        weight_matrix = np.array([avg_weights[cat] for cat in categories if cat in avg_weights])

        # Plot heatmap
        im = ax_heat.imshow(weight_matrix, cmap='RdBu_r', aspect='auto',
                           vmin=-1, vmax=1)

        # Set ticks
        ax_heat.set_yticks(range(len(categories)))
        ax_heat.set_yticklabels([c for c in categories if c in avg_weights],
                                fontsize=12, fontweight='bold')

        ax_heat.set_xticks(range(len(feature_names)))
        ax_heat.set_xticklabels(feature_names, rotation=45, ha='right',
                               fontsize=11)

        # Add value annotations
        for i in range(len(categories)):
            for j in range(len(feature_names)):
                if i < weight_matrix.shape[0]:
                    text = ax_heat.text(j, i, f'{weight_matrix[i, j]:.2f}',
                                      ha="center", va="center",
                                      color="white" if abs(weight_matrix[i, j]) > 0.5 else "black",
                                      fontsize=9, fontweight='bold')

        ax_heat.set_title('GLM Weights by State Category\\n(Positive = Bias toward Right)',
                         fontsize=14, fontweight='bold', pad=15)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar.set_label('Weight', fontweight='bold', fontsize=12)

        # Interpretation panels
        interpretation_text = {
            'Engaged': """
ENGAGED STATE
•Interpretation guide
•High accuracy (>65%)
•Strong WSLS usage
•Consistent latencies
•Appropriate stimulus use

Genotype Implications:
→ WT/+ animals show
  stable engagement
→ -/- animals have
  shorter, less frequent
  engaged bouts
→ Reflects attentional
  capacity & motivation
            """,
            'Lapsed': """
LAPSED STATE
• Low accuracy (<60%)
• Weak WSLS strategy
• Variable latencies
• Side bias emerges
• Reduced stimulus weight

Genotype Implications:
→ -/- genotype shows
  more frequent lapses
→ Longer lapse duration
→ Difficulty re-engaging
→ Impaired sustained
  attention
            """,
            'Mixed': """
MIXED STATE
• Moderate accuracy
• Transitional dynamics
• Variable strategy use
• Context-dependent
• Learning-related

Genotype Implications:
→ Early training state
→ Strategy exploration
→ Task acquisition phase
→ Individual variability
  in stabilization
            """
        }

        # Add interpretation boxes
        for idx, (cat, text) in enumerate(interpretation_text.items()):
            ax_interp = fig.add_subplot(gs[idx, 1])
            ax_interp.text(0.05, 0.95, text.strip(),
                          transform=ax_interp.transAxes,
                          fontsize=9,
                          verticalalignment='top',
                          family='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightyellow',
                                  alpha=0.8, edgecolor='black', linewidth=2))
            ax_interp.axis('off')

        plt.suptitle('GLM Weight Interpretation Guide with Genotype Implications',
                    fontsize=16, fontweight='bold')

        # Save
        plt.savefig(self.output_dir / 'weight_interpretation.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'weight_interpretation.pdf',
                   bbox_inches='tight')
        plt.close()

        print("  ✓ Created annotated weight heatmap")

    def analyze_lapse_metrics(self, trials):
        """
        Analyze lapse duration and frequency by genotype.
        """
        if len(trials) == 0 or 'state_label' not in trials.columns:
            print("  No state data available for lapse analysis")
            return

        # Identify lapse bouts
        trials_sorted = trials.sort_values(['animal_id', 'trial_num']).copy()
        trials_sorted['is_lapsed'] = (trials_sorted['state_label'] == 'Lapsed').astype(int)

        # Find bout boundaries (where state changes)
        trials_sorted['state_change'] = trials_sorted.groupby('animal_id')['is_lapsed'].diff().fillna(0) != 0
        trials_sorted['bout_id'] = trials_sorted.groupby('animal_id')['state_change'].cumsum()

        # Calculate bout durations
        bout_stats = []

        for (animal, bout), group in trials_sorted.groupby(['animal_id', 'bout_id']):
            if group['is_lapsed'].iloc[0] == 1:  # Lapsed bout
                duration = len(group)
                genotype = group['genotype'].iloc[0]

                bout_stats.append({
                    'animal_id': animal,
                    'genotype': genotype,
                    'bout_duration': duration,
                    'bout_type': 'Lapsed'
                })

        if not bout_stats:
            print("  No lapse bouts found")
            return

        bout_df = pd.DataFrame(bout_stats)

        # Calculate frequency (bouts per 100 trials)
        animal_stats = []
        for (animal, genotype), group in trials_sorted.groupby(['animal_id', 'genotype']):
            n_trials = len(group)
            n_lapse_bouts = len(bout_df[bout_df['animal_id'] == animal])
            lapse_frequency = (n_lapse_bouts / n_trials) * 100

            animal_lapse_bouts = bout_df[bout_df['animal_id'] == animal]
            avg_duration = animal_lapse_bouts['bout_duration'].mean() if len(animal_lapse_bouts) > 0 else 0

            animal_stats.append({
                'animal_id': animal,
                'genotype': genotype,
                'lapse_frequency': lapse_frequency,
                'avg_lapse_duration': avg_duration,
                'total_trials': n_trials
            })

        stats_df = pd.DataFrame(animal_stats)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Panel A: Lapse duration by genotype
        ax = axes[0, 0]
        genotypes = sorted(stats_df['genotype'].unique())

        duration_data = [
            bout_df[bout_df['genotype'] == g]['bout_duration'].values
            for g in genotypes
        ]

        bp = ax.boxplot(duration_data, labels=genotypes, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#e74c3c')
            patch.set_alpha(0.6)

        ax.set_ylabel('Lapse Duration (trials)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Genotype', fontweight='bold', fontsize=12)
        ax.set_title('Lapse Bout Duration by Genotype', fontweight='bold', fontsize=13)
        ax.grid(axis='y', alpha=0.3)

        # Add sample sizes
        for i, g in enumerate(genotypes):
            n = len(bout_df[bout_df['genotype'] == g])
            ax.text(i+1, ax.get_ylim()[1]*0.95, f'n={n}',
                   ha='center', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Panel B: Lapse frequency by genotype
        ax = axes[0, 1]

        freq_means = stats_df.groupby('genotype')['lapse_frequency'].agg(['mean', 'sem'])

        bars = ax.bar(range(len(genotypes)),
                     freq_means['mean'].values,
                     yerr=freq_means['sem'].values,
                     color='#e74c3c', alpha=0.7, capsize=5)

        ax.set_xticks(range(len(genotypes)))
        ax.set_xticklabels(genotypes, rotation=45, ha='right')
        ax.set_ylabel('Lapse Bouts per 100 Trials', fontweight='bold', fontsize=12)
        ax.set_xlabel('Genotype', fontweight='bold', fontsize=12)
        ax.set_title('Lapse Frequency by Genotype', fontweight='bold', fontsize=13)
        ax.grid(axis='y', alpha=0.3)

        # Panel C: Duration vs Frequency scatter
        ax = axes[1, 0]

        for geno in genotypes:
            geno_data = stats_df[stats_df['genotype'] == geno]
            ax.scatter(geno_data['lapse_frequency'],
                      geno_data['avg_lapse_duration'],
                      s=100, alpha=0.6, label=geno)

        ax.set_xlabel('Lapse Frequency (per 100 trials)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Average Lapse Duration (trials)', fontweight='bold', fontsize=12)
        ax.set_title('Lapse Frequency vs Duration', fontweight='bold', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        # Panel D: Genotype implications summary
        ax = axes[1, 1]
        ax.axis('off')

        # Calculate key genotype differences
        summary_text = "GENOTYPE IMPLICATIONS:\\n\\n"

        # Find genotypes with highest/lowest metrics
        max_freq_geno = stats_df.groupby('genotype')['lapse_frequency'].mean().idxmax()
        min_freq_geno = stats_df.groupby('genotype')['lapse_frequency'].mean().idxmin()

        max_dur_geno = bout_df.groupby('genotype')['bout_duration'].mean().idxmax()
        min_dur_geno = bout_df.groupby('genotype')['bout_duration'].mean().idxmin()

        summary_text += f"LAPSE FREQUENCY:\\n"
        summary_text += f"  Highest: {max_freq_geno}\\n"
        summary_text += f"  Lowest: {min_freq_geno}\\n\\n"

        summary_text += f"LAPSE DURATION:\\n"
        summary_text += f"  Longest: {max_dur_geno}\\n"
        summary_text += f"  Shortest: {min_dur_geno}\\n\\n"

        summary_text += "INTERPRETATION:\\n"
        summary_text += "• Higher lapse frequency →\\n"
        summary_text += "  Difficulty maintaining attention\\n"
        summary_text += "• Longer lapse duration →\\n"
        summary_text += "  Impaired re-engagement\\n"
        summary_text += "• Both elevated →\\n"
        summary_text += "  Severe attentional deficits\\n"
        summary_text += "\\n"
        summary_text += "Expected pattern:\\n"
        summary_text += "-/- > +/- > +/+ ≈ +\\n"
        summary_text += "(dose-dependent effect)"

        ax.text(0.1, 0.9, summary_text,
               transform=ax.transAxes,
               fontsize=11,
               verticalalignment='top',
               family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow',
                        alpha=0.9, edgecolor='black', linewidth=2))

        plt.suptitle('Lapse Metrics: Duration & Frequency by Genotype',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save
        plt.savefig(self.output_dir / 'lapse_metrics.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'lapse_metrics.pdf',
                   bbox_inches='tight')
        plt.close()

        # Save statistics
        stats_df.to_csv(self.output_dir / 'lapse_statistics.csv', index=False)

        print("  ✓ Created lapse metrics analysis")
        print(f"    Found {len(bout_df)} lapse bouts across {stats_df['animal_id'].nunique()} animals")


def main():
    """Create interpretation guides."""
    print("="*80)
    print("PRIORITY 2: GLM INTERPRETATION & LAPSE METRICS")
    print("="*80)

    guide = InterpretationGuide()

    # Define animals
    animals_W = [f'c{c}m{m}' for c in range(1, 5) for m in range(1, 6)
                 if not (c == 1 and m == 5)]
    animals_F = [11, 12, 13, 14, 21, 22, 23, 24, 25,
                 31, 32, 33, 34, 41, 42, 51, 52,
                 61, 62, 63, 64, 71, 72, 73,
                 81, 82, 83, 84, 91, 92, 93,
                 101, 102, 103, 104]

    # Load data
    print("\\nLoading data...")
    results_W, trials_W = guide.load_all_results('W', animals_W)
    results_F, trials_F = guide.load_all_results('F', animals_F)
    print(f"  Cohort W: {len(results_W)} animals")
    print(f"  Cohort F: {len(results_F)} animals")

    all_results = results_W + results_F
    all_trials = pd.concat([trials_W, trials_F], ignore_index=True)

    # Generate visualizations
    print("\\nGenerating visualizations...")

    print("\\n[1/2] Creating annotated weight heatmap...")
    guide.create_annotated_weight_heatmap(all_results)

    print("\\n[2/2] Analyzing lapse metrics...")
    guide.analyze_lapse_metrics(all_trials)

    print("\\n" + "="*80)
    print("✓ INTERPRETATION GUIDES COMPLETE!")
    print("="*80)
    print(f"\\nOutput directory: {guide.output_dir}")


if __name__ == '__main__':
    main()
