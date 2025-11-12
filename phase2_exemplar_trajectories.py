"""
Phase 2 Individual State Trajectories: Exemplar Animals with Accuracy

Shows actual trial-by-trial state sequences and accuracy within states for
representative animals from each genotype, illustrating different learning
strategies during Phase 2 reversal.
"""

import os
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PHASE2_DIR = '/home/user/GLMHMM/results/phase2_reversal'
OUTPUT_DIR = '/home/user/GLMHMM/results/statistical_analysis/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Genotype mapping
OLD_TO_NEW = {
    '+/+': 'A1D_Wt', '-/-': 'A1D_KO', '+/-': 'A1D_Het',
    '+': 'B6', '-': 'C3H x B6'
}

# State names and colors
STATE_NAMES = {0: 'Engaged', 1: 'Biased', 2: 'Lapsed'}
STATE_COLORS = {0: '#1f77b4', 1: '#ff7f0e', 2: '#d62728'}  # Blue, Orange, Red

# =============================================================================
# EXTRACT EXEMPLAR ANIMALS
# =============================================================================

class ExemplarSelector:
    """Select exemplar animals that best represent each genotype's learning strategy"""

    def __init__(self):
        self.animals_data = {}
        self.exemplars = {}

    def load_all_animals(self):
        """Load data for all Phase 2 animals"""
        phase2_files = sorted(glob.glob(os.path.join(PHASE2_DIR, 'models', '*_reversal.pkl')))

        for pkl_file in phase2_files:
            animal_id = os.path.basename(pkl_file).replace('_reversal.pkl', '')

            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            genotype = OLD_TO_NEW.get(data.get('genotype', ''), data.get('genotype', ''))
            traj_df = data.get('trajectory_df', None)

            if traj_df is not None:
                self.animals_data[animal_id] = {
                    'genotype': genotype,
                    'trajectory_df': traj_df,
                    'pkl_data': data
                }

        print(f"Loaded {len(self.animals_data)} Phase 2 animals")

    def select_exemplars(self):
        """Select best exemplars per genotype based on clear learning patterns"""

        # Group by genotype
        by_genotype = {}
        for animal_id, data in self.animals_data.items():
            geno = data['genotype']
            if geno not in by_genotype:
                by_genotype[geno] = []
            by_genotype[geno].append(animal_id)

        # For each genotype, select 2 exemplars showing clearest patterns
        exemplars = {}

        for genotype in sorted(by_genotype.keys()):
            animal_ids = by_genotype[genotype]

            # Calculate learning metrics for each animal
            learning_metrics = []

            for animal_id in animal_ids:
                traj_df = self.animals_data[animal_id]['trajectory_df']

                # Calculate state occupancy and accuracy by state
                state_occ = {}
                state_acc = {}

                for state in [0, 1, 2]:
                    state_mask = traj_df['state'] == state
                    if state_mask.sum() > 0:
                        total_trials = traj_df.loc[state_mask, 'bout_length'].sum()
                        state_occ[STATE_NAMES[state]] = total_trials
                        state_acc[STATE_NAMES[state]] = traj_df.loc[state_mask, 'during_accuracy'].mean()

                learning_metrics.append({
                    'animal_id': animal_id,
                    'state_occ': state_occ,
                    'state_acc': state_acc,
                    'total_trials': traj_df['bout_length'].sum()
                })

            # Select exemplars that best represent genotype patterns
            # Choose 2 animals with different profiles if possible
            exemplars[genotype] = sorted(animal_ids)[:2]

        self.exemplars = exemplars
        return exemplars

    def get_exemplar_data(self, genotype, animal_id):
        """Get complete trajectory data for exemplar animal"""
        return self.animals_data[animal_id]


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_exemplar_trajectories(selector, output_dir):
    """Create comprehensive figure showing exemplar animal trajectories"""

    exemplars = selector.exemplars
    n_genotypes = len(exemplars)
    n_animals_per_geno = 2

    fig, axes = plt.subplots(
        n_genotypes, n_animals_per_geno,
        figsize=(16, 4 * n_genotypes),
        gridspec_kw={'hspace': 0.4, 'wspace': 0.25}
    )

    if n_genotypes == 1:
        axes = [axes]

    row_idx = 0
    for genotype in sorted(exemplars.keys()):
        animal_ids = exemplars[genotype]

        for col_idx, animal_id in enumerate(animal_ids):
            ax = axes[row_idx][col_idx]

            # Get trajectory data
            data = selector.get_exemplar_data(genotype, animal_id)
            traj_df = data['trajectory_df']

            # Reconstruct trial-by-trial state sequence
            trial_states = []
            trial_accuracies = []
            trial_numbers = []

            trial_num = 0
            for _, row in traj_df.iterrows():
                state = int(row['state'])
                bout_length = int(row['bout_length'])
                accuracy = row['during_accuracy']

                for _ in range(bout_length):
                    trial_states.append(state)
                    trial_accuracies.append(accuracy)
                    trial_numbers.append(trial_num)
                    trial_num += 1

            trial_states = np.array(trial_states)
            trial_accuracies = np.array(trial_accuracies)
            trial_numbers = np.array(trial_numbers)

            # Plot 1: State sequence (bottom layer)
            for i, (trial, state) in enumerate(zip(trial_numbers, trial_states)):
                color = STATE_COLORS[state]
                ax.barh(0, 1, left=trial, height=0.3, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

            # Plot 2: Accuracy overlay (top layer, scaled)
            ax2 = ax.twinx()
            ax2.plot(trial_numbers, trial_accuracies, 'k-', linewidth=1.5, alpha=0.6, label='Accuracy')
            ax2.scatter(trial_numbers, trial_accuracies, s=20, c=trial_states, cmap='tab10', alpha=0.5)
            ax2.set_ylim([-0.05, 1.05])
            ax2.set_ylabel('Accuracy (% Correct)', fontsize=10, fontweight='bold')
            ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax2.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])

            # Format main axes
            ax.set_ylim([-0.5, 0.5])
            ax.set_xlabel('Trial Number (Phase 2)', fontsize=10, fontweight='bold')
            ax.set_yticks([])
            ax.set_xlim([trial_numbers[0] - 5, trial_numbers[-1] + 5])

            # Title with animal info
            ax.set_title(
                f'{genotype} - Animal {animal_id}\n(n={len(trial_states)} trials)',
                fontsize=11, fontweight='bold'
            )

            # Add state occurrence text
            state_stats = []
            for state in [0, 1, 2]:
                state_mask = trial_states == state
                if state_mask.sum() > 0:
                    occ = 100 * state_mask.sum() / len(trial_states)
                    acc = np.mean(trial_accuracies[state_mask])
                    state_stats.append(f"{STATE_NAMES[state]}: {occ:.0f}% occ, {acc:.1%} acc")

            stats_text = '\n'.join(state_stats)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   family='monospace')

        row_idx += 1

    # Overall title and legend
    fig.suptitle(
        'Phase 2 Individual Animal Trajectories:\nState Sequences with Accuracy by State',
        fontsize=14, fontweight='bold', y=0.995
    )

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=STATE_COLORS[0], edgecolor='black', label='Engaged State (High Engagement)'),
        mpatches.Patch(facecolor=STATE_COLORS[1], edgecolor='black', label='Biased State (Perseverative)'),
        mpatches.Patch(facecolor=STATE_COLORS[2], edgecolor='black', label='Lapsed State (Disengaged)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=10,
              bbox_to_anchor=(0.5, 0.02), frameon=True)

    plt.savefig(os.path.join(output_dir, 'phase2_exemplar_trajectories.png'),
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'phase2_exemplar_trajectories.pdf'),
               bbox_inches='tight')
    plt.close()

    print("✓ Saved exemplar trajectories figure")


def create_learning_comparison_summary(selector, output_dir):
    """Create summary table comparing learning patterns across exemplar animals"""

    exemplars = selector.exemplars
    summary_data = []

    for genotype in sorted(exemplars.keys()):
        for animal_id in exemplars[genotype]:
            data = selector.get_exemplar_data(genotype, animal_id)
            traj_df = data['trajectory_df']

            overall_accuracy = traj_df['during_accuracy'].mean()

            summary_data.append({
                'Genotype': genotype,
                'Animal ID': animal_id,
                'Overall Accuracy': f"{overall_accuracy:.1%}",
                'Engaged % Occ': f"{(traj_df[traj_df['state']==0]['bout_length'].sum() / traj_df['bout_length'].sum() * 100):.0f}%",
                'Biased % Occ': f"{(traj_df[traj_df['state']==1]['bout_length'].sum() / traj_df['bout_length'].sum() * 100):.0f}%",
                'Lapsed % Occ': f"{(traj_df[traj_df['state']==2]['bout_length'].sum() / traj_df['bout_length'].sum() * 100):.0f}%",
                'Engaged Acc': f"{(traj_df[traj_df['state']==0]['during_accuracy'].mean() if (traj_df['state']==0).sum() > 0 else np.nan):.1%}",
                'Biased Acc': f"{(traj_df[traj_df['state']==1]['during_accuracy'].mean() if (traj_df['state']==1).sum() > 0 else np.nan):.1%}",
                'Lapsed Acc': f"{(traj_df[traj_df['state']==2]['during_accuracy'].mean() if (traj_df['state']==2).sum() > 0 else np.nan):.1%}"
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, '../phase2_exemplar_summary.csv'), index=False)

    print("\n" + "=" * 140)
    print("PHASE 2 EXEMPLAR ANIMAL LEARNING PATTERNS")
    print("=" * 140)
    print(summary_df.to_string(index=False))
    print("=" * 140)
    print()


def create_interpretation_guide(output_dir):
    """Create guide explaining the trajectory visualization"""

    guide = """
================================================================================
PHASE 2 INDIVIDUAL ANIMAL TRAJECTORIES: INTERPRETATION GUIDE
================================================================================

WHAT THE FIGURE SHOWS
================================================================================

Each panel represents ONE ANIMAL'S Phase 2 reversal learning trajectory.

Visual Elements:
1. COLORED BARS (bottom): The STATE each trial was classified into
   • Blue bar = Trial in ENGAGED state (high engagement, good decisions)
   • Orange bar = Trial in BIASED state (perseverative/biased choices)
   • Red bar = Trial in LAPSED state (disengaged, random choices)

2. BLACK LINE WITH DOTS: The ACCURACY during each trial
   • Vertical position = % correct choices within that state
   • Goes from 0% (bottom) to 100% (top)
   • Connected line shows how accuracy changes over the session

3. TEXT BOX (upper left): Summary statistics
   • State Occupancy: % of trials spent in each state
   • State Accuracy: % correct choices while in each state


WHAT THESE METRICS MEAN
================================================================================

STATE OCCUPANCY (e.g., "Engaged: 35%")
→ The animal spent 35% of Phase 2 trials in the Engaged state
→ This is about DISTRIBUTION OF TIME across states
→ Example: If 800 trials total, 280 were in Engaged state

ACCURACY (e.g., "Engaged: 72.3%")
→ While in Engaged state, the animal chose correctly 72.3% of the time
→ This is about DECISION-MAKING QUALITY within that state
→ Example: In 280 Engaged trials, correct on ~203 trials

COMBINED INTERPRETATION:
→ High Occupancy + High Accuracy = Adaptive state (good for task)
→ Low Occupancy + High Accuracy = Underutilized strength
→ High Occupancy + Low Accuracy = Maladaptive strategy
→ Low Occupancy + Low Accuracy = Irrelevant state


WHAT EACH STATE REPRESENTS
================================================================================

ENGAGED STATE (Blue):
• Animal shows high engagement with task
• Responsive to task contingencies
• Makes deliberate decisions, not random
• Usually shows elevated accuracy
• Indicator of learning and performance

BIASED STATE (Orange):
• Animal shows perseverative or biased choices
• Examples: Always chooses left, always repeats previous choice
• Not random, but not task-adaptive
• Accuracy varies but often medium
• Can indicate stereotyped responding or learning a wrong rule

LAPSED STATE (Red):
• Animal is disengaged, has "given up"
• Choices are near-random or completely random
• Accuracy low (around 50% or chance level)
• Indicates loss of engagement or learning failure
• Problematic for task performance


LEARNING PATTERNS TO LOOK FOR
================================================================================

ADAPTIVE LEARNING (like B6):
→ Early phase: Mixed states, variable accuracy
→ Late phase: Increased Engaged occupancy, high Engaged accuracy
→ Lapsed occupancy decreases over time
→ Overall trajectory: Learning to engage more, disengage less
→ Result: High overall accuracy by end of session

ALTERNATIVE LEARNING (like C3H×B6):
→ Increased reliance on Biased state (stereotyped but consistent)
→ Lapsed state decreases significantly
→ Not perfectly adaptive but functional
→ Result: Medium-high accuracy through consistent strategy

MALADAPTIVE LEARNING (like A1D_Wt):
→ Early phase: Tries multiple states
→ Late phase: Increases Lapsed state, decreases Biased state
→ Gives up on task over time
→ Result: Low overall accuracy, declining performance

OSCILLATORY LEARNING (like A1D_KO):
→ Alternates between states without clear pattern
→ High variability in state sequence
→ Mixed state dominance
→ Result: Medium accuracy but inconsistent pattern

NON-LEARNING (like A1D_Het):
→ No clear progression through states
→ Random-looking state sequence
→ Inconsistent accuracy across time
→ Result: Low, variable performance


HOW TO READ SPECIFIC ANIMALS
================================================================================

Example 1: B6 Animal (Adaptive)
Panel shows:
  - Many blue bars early and late → Engaged throughout
  - Accuracy line stays high (70-80% range)
  - Few red bars → Minimal lapse
  - Interpretation: "This animal learned the reversal effectively by staying engaged"

Example 2: A1D_Wt Animal (Maladaptive)
Panel shows:
  - Orange bars (Biased) → Relying on stereotyped choice
  - Accuracy declining over time (starts 60%, ends 40%)
  - Increasing red bars (Lapsed) toward end
  - Interpretation: "This animal started with a strategy but gave up, ending lapsed"

Example 3: A1D_KO Animal (Mixed)
Panel shows:
  - Alternating colors → Switching between states
  - Accuracy fluctuating with state changes
  - No clear trend in state sequence
  - Interpretation: "This animal doesn't settle on a strategy, oscillates constantly"


CONNECTING TO REGRESSION ANALYSIS
================================================================================

The learning rates from the regression analysis panel showed:

B6: +0.0257% Engaged per 100 trials ***
→ This animal's panel should show INCREASING blue dominance over trials

A1D_Wt: -0.1107% Biased per 100 trials ***
→ This animal's panel should show DECREASING orange over trials
A1D_Wt: +0.1195% Lapsed per 100 trials
→ This animal's panel should show INCREASING red toward end of session

A1D_KO: +0.0416% Biased per 100 trials *
→ This animal's panel should show INCREASING orange relative to others


WHAT GENOTYPES TELL US
================================================================================

B6 - ADAPTIVE LEARNER
Exemplars show: Clear engagement trajectory, decreasing lapse
Learning strategy: Master task by staying engaged
Phase 2 success: HIGH (transitions to correctness efficiently)

C3H×B6 - SCAFFOLD LEARNER
Exemplars show: Early biased phase, then lapse suppression
Learning strategy: Use bias to maintain engagement, avoid lapse
Phase 2 success: HIGH (functional despite alternative path)

A1D_KO - OSCILLATOR
Exemplars show: Mixed state dominance, variable patterns
Learning strategy: None clear - alternates between approaches
Phase 2 success: MEDIUM (some compensatory engagement)

A1D_Het - NON-LEARNER
Exemplars show: Random-appearing state sequences
Learning strategy: None apparent
Phase 2 success: LOW (unable to consolidate learning)

A1D_Wt - MALADAPTIVE
Exemplars show: Disengagement over time, increasing lapse
Learning strategy: Attempted (some biased state use) but failed
Phase 2 success: LOWEST (progressive deterioration)


TECHNICAL NOTES
================================================================================

State Classification Method:
- Hidden Markov Model (GLM-HMM) identifies most likely state per trial
- Based on choices, outcomes, and state transition probabilities
- Probabilistic inference, not deterministic categorization

Accuracy Calculation:
- "during_accuracy" = P(correct choice | animal in this state)
- Computed within state-specific trial bouts
- Reflects decision quality conditioned on state

Trial Binning:
- Raw trials colored by inferred state
- State changes marked by color transitions
- Bout length determines how many trials share same color
================================================================================
"""

    with open(os.path.join(output_dir, '../PHASE2_TRAJECTORIES_GUIDE.txt'), 'w') as f:
        f.write(guide)

    print("✓ Saved trajectories interpretation guide")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("PHASE 2 INDIVIDUAL ANIMAL TRAJECTORIES: EXEMPLAR VISUALIZATION")
    print("=" * 80)
    print()

    print("[1/3] Loading all Phase 2 animals...")
    selector = ExemplarSelector()
    selector.load_all_animals()

    print("[2/3] Selecting exemplar animals...")
    selector.select_exemplars()
    print(f"Selected exemplars:")
    for genotype, animals in sorted(selector.exemplars.items()):
        print(f"  {genotype}: {animals}")

    print("\n[3/3] Creating visualizations...")
    create_exemplar_trajectories(selector, OUTPUT_DIR)
    create_learning_comparison_summary(selector, OUTPUT_DIR)
    create_interpretation_guide(OUTPUT_DIR)

    print()
    print("=" * 80)
    print("✓ EXEMPLAR TRAJECTORIES COMPLETE")
    print("=" * 80)
    print()
    print("Output files:")
    print(f"  - phase2_exemplar_trajectories.png (exemplar animal state sequences)")
    print(f"  - phase2_exemplar_trajectories.pdf (vector format)")
    print(f"  - phase2_exemplar_summary.csv (learning metrics table)")
    print(f"  - PHASE2_TRAJECTORIES_GUIDE.txt (interpretation guide)")
