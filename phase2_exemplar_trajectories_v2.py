"""
Phase 2 Individual Animal Exemplar Trajectories - AESTHETIC REDESIGN

Improved visualization with better colors, layout, typography, and clarity
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

# Set better style defaults
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

PHASE2_DIR = '/home/user/GLMHMM/results/phase2_reversal'
OUTPUT_DIR = '/home/user/GLMHMM/results/statistical_analysis/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Improved color scheme - more professional and distinct
STATE_COLORS = {
    0: '#2E86AB',    # Deep blue for Engaged (confidence, focus)
    1: '#A23B72',    # Deep purple for Biased (distinct from others)
    2: '#F18F01'     # Orange for Lapsed (stands out, warmth indicates disengagement)
}

STATE_NAMES = {0: 'Engaged', 1: 'Biased', 2: 'Lapsed'}

# Genotype mapping
OLD_TO_NEW = {
    '+/+': 'A1D_Wt', '-/-': 'A1D_KO', '+/-': 'A1D_Het',
    '+': 'B6', '-': 'C3H x B6'
}

# Genotype colors - professional palette
GENO_COLORS = {
    'A1D_Wt': '#E63946',      # Red
    'A1D_KO': '#457B9D',      # Blue
    'A1D_Het': '#A8DADC',     # Light blue
    'B6': '#2D3142',          # Dark blue-grey
    'C3H x B6': '#8B5A8E'     # Purple
}

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

class TrajectoryDataLoader:
    def __init__(self):
        self.animals = {}
        self.exemplars = {}

    def load_all_animals(self):
        """Load Phase 2 animal data"""
        phase2_files = sorted(glob.glob(os.path.join(PHASE2_DIR, 'models', '*_reversal.pkl')))

        for pkl_file in phase2_files:
            animal_id = os.path.basename(pkl_file).replace('_reversal.pkl', '')

            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            genotype = OLD_TO_NEW.get(data.get('genotype', ''), data.get('genotype', ''))
            traj_df = data.get('trajectory_df', None)

            if traj_df is not None:
                self.animals[animal_id] = {
                    'genotype': genotype,
                    'trajectory_df': traj_df,
                }

        print(f"Loaded {len(self.animals)} animals")

    def select_exemplars(self):
        """Select best exemplars per genotype"""
        by_genotype = {}
        for animal_id, data in self.animals.items():
            geno = data['genotype']
            if geno not in by_genotype:
                by_genotype[geno] = []
            by_genotype[geno].append(animal_id)

        for genotype in sorted(by_genotype.keys()):
            self.exemplars[genotype] = sorted(by_genotype[genotype])[:2]

        return self.exemplars

    def get_trajectory_data(self, animal_id):
        """Build trial-by-trial arrays for visualization"""
        traj_df = self.animals[animal_id]['trajectory_df']

        trial_states = []
        trial_accuracies = []
        bout_starts = [0]
        bout_colors = []

        trial_num = 0
        for _, row in traj_df.iterrows():
            state = int(row['state'])
            bout_length = int(row['bout_length'])
            accuracy = row['during_accuracy']

            for _ in range(bout_length):
                trial_states.append(state)
                trial_accuracies.append(accuracy)
                trial_num += 1

            bout_colors.append(STATE_COLORS[state])
            bout_starts.append(trial_num)

        return {
            'states': np.array(trial_states),
            'accuracies': np.array(trial_accuracies),
            'trials': np.arange(len(trial_states)),
            'bout_starts': bout_starts,
            'bout_colors': bout_colors,
            'trajectory_df': traj_df,
            'genotype': self.animals[animal_id]['genotype']
        }

    def compute_stats(self, animal_id):
        """Compute summary statistics"""
        traj_df = self.animals[animal_id]['trajectory_df']
        stats = {}

        for state in [0, 1, 2]:
            state_mask = traj_df['state'] == state
            if state_mask.sum() > 0:
                total_trials = traj_df.loc[state_mask, 'bout_length'].sum()
                occ_pct = 100 * total_trials / traj_df['bout_length'].sum()
                acc = traj_df.loc[state_mask, 'during_accuracy'].mean()
                stats[STATE_NAMES[state]] = {
                    'occupancy': occ_pct,
                    'accuracy': acc,
                    'trials': int(total_trials)
                }

        overall_acc = traj_df['during_accuracy'].mean()
        stats['overall_accuracy'] = overall_acc

        return stats


# =============================================================================
# BEAUTIFUL VISUALIZATION
# =============================================================================

def create_beautiful_trajectories(loader, output_dir):
    """Create aesthetic individual trajectory visualization"""

    exemplars = loader.exemplars
    n_genotypes = len(exemplars)

    # Create figure with better proportions
    fig = plt.figure(figsize=(18, 2.5 * n_genotypes))
    fig.patch.set_facecolor('white')

    # Create gridspec for better control
    gs = fig.add_gridspec(n_genotypes, 2, hspace=0.45, wspace=0.3,
                         left=0.08, right=0.95, top=0.95, bottom=0.08)

    row_idx = 0
    for genotype in sorted(exemplars.keys()):
        animal_ids = exemplars[genotype]

        for col_idx, animal_id in enumerate(animal_ids):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            # Get data
            trajectory = loader.get_trajectory_data(animal_id)
            stats = loader.compute_stats(animal_id)

            states = trajectory['states']
            accuracies = trajectory['accuracies']
            trials = trajectory['trials']
            bout_starts = trajectory['bout_starts']
            bout_colors = trajectory['bout_colors']

            # ===== PLOT 1: State bars (background bands) =====
            # Use rectangles for each bout to create clean state visualization
            for i in range(len(bout_starts) - 1):
                start = bout_starts[i]
                end = bout_starts[i + 1]
                color = bout_colors[i]

                # State rectangle
                rect = Rectangle((start, -0.08), end - start, 0.16,
                               facecolor=color, edgecolor='none', alpha=0.85, zorder=1)
                ax.add_patch(rect)

            # ===== PLOT 2: Accuracy line (foreground) =====
            # Smooth the accuracy line for better aesthetics
            from scipy.ndimage import uniform_filter1d
            window = max(1, len(trials) // 50)  # Adaptive smoothing
            accuracies_smooth = uniform_filter1d(accuracies, size=window, mode='nearest')

            ax.plot(trials, accuracies_smooth, linewidth=2.5, color='#1A1A1A',
                   zorder=10, alpha=0.8, label='Accuracy Trajectory')

            # Add subtle scatter for original points
            sample_every = max(1, len(trials) // 30)
            ax.scatter(trials[::sample_every], accuracies[::sample_every],
                      c=states[::sample_every], cmap='tab10', s=35,
                      alpha=0.4, zorder=5, edgecolors='none')

            # ===== FORMATTING =====
            ax.set_ylim([-0.12, 1.08])
            ax.set_xlim([-5, len(trials) + 5])
            ax.set_facecolor('#FAFAFA')

            # Y-axis
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9)
            ax.set_ylabel('Accuracy (% Correct)', fontsize=10, fontweight='600', labelpad=8)

            # X-axis
            ax.set_xlabel('Trial Number', fontsize=10, fontweight='600', labelpad=8)
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))

            # Grid
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)

            # Spine styling
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color('#CCCCCC')
                ax.spines[spine].set_linewidth(0.8)

            # ===== TITLE & STATS BOX =====
            title = f'{genotype} — Animal {animal_id}'
            ax.set_title(title, fontsize=12, fontweight='700', pad=15, color='#1A1A1A')

            # Statistics box
            stat_lines = [
                f"Overall Accuracy: {stats['overall_accuracy']*100:.1f}%",
                "",
                f"Engaged:  {stats.get('Engaged', {}).get('occupancy', 0):.0f}% occ, {stats.get('Engaged', {}).get('accuracy', 0)*100:.0f}% acc",
                f"Biased:   {stats.get('Biased', {}).get('occupancy', 0):.0f}% occ, {stats.get('Biased', {}).get('accuracy', 0)*100:.0f}% acc",
                f"Lapsed:   {stats.get('Lapsed', {}).get('occupancy', 0):.0f}% occ, {stats.get('Lapsed', {}).get('accuracy', 0)*100:.0f}% acc",
            ]

            stat_text = '\n'.join(stat_lines)

            ax.text(0.98, 0.97, stat_text, transform=ax.transAxes,
                   fontsize=8.5, verticalalignment='top', horizontalalignment='right',
                   family='monospace', color='#2A2A2A',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='white',
                            edgecolor='#CCCCCC', linewidth=1, alpha=0.95))

            # Tick label styling
            ax.tick_params(labelsize=9, colors='#4A4A4A')

        row_idx += 1

    # ===== LEGEND =====
    # Create custom legend
    legend_elements = [
        mpatches.Patch(facecolor=STATE_COLORS[0], edgecolor='none',
                      label='Engaged — High engagement, responsive to task', alpha=0.85),
        mpatches.Patch(facecolor=STATE_COLORS[1], edgecolor='none',
                      label='Biased — Perseverative, stereotyped choices', alpha=0.85),
        mpatches.Patch(facecolor=STATE_COLORS[2], edgecolor='none',
                      label='Lapsed — Disengaged, random/giving up', alpha=0.85),
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
              bbox_to_anchor=(0.5, -0.01), frameon=True, fancybox=True, shadow=True)

    # ===== OVERALL TITLE =====
    fig.suptitle('Phase 2 Individual Animal Trajectories: State Sequences & Accuracy Dynamics',
                fontsize=14, fontweight='700', color='#1A1A1A', y=0.98)

    # Save
    plt.savefig(os.path.join(output_dir, 'phase2_exemplar_trajectories_v2.png'),
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(output_dir, 'phase2_exemplar_trajectories_v2.pdf'),
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print("✓ Saved beautiful exemplar trajectories figure")


def create_summary_table_visual(loader, output_dir):
    """Create beautiful summary table visualization"""

    exemplars = loader.exemplars
    data = []

    for genotype in sorted(exemplars.keys()):
        for animal_id in exemplars[genotype]:
            stats = loader.compute_stats(animal_id)
            data.append({
                'Genotype': genotype,
                'Animal': animal_id,
                'Overall Acc': f"{stats['overall_accuracy']*100:.1f}%",
                'Eng Occ': f"{stats.get('Engaged', {}).get('occupancy', 0):.0f}%",
                'Eng Acc': f"{stats.get('Engaged', {}).get('accuracy', 0)*100:.0f}%",
                'Bias Occ': f"{stats.get('Biased', {}).get('occupancy', 0):.0f}%",
                'Bias Acc': f"{stats.get('Biased', {}).get('accuracy', 0)*100:.0f}%",
                'Lapse Occ': f"{stats.get('Lapsed', {}).get('occupancy', 0):.0f}%",
                'Lapse Acc': f"{stats.get('Lapsed', {}).get('accuracy', 0)*100:.0f}%",
            })

    df = pd.DataFrame(data)

    # Create figure for table
    fig, ax = plt.subplots(figsize=(16, len(df) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows with genotype colors
    for i, row in enumerate(df.itertuples()):
        genotype = row.Genotype
        color = GENO_COLORS.get(genotype, '#FFFFFF')

        for j in range(len(df.columns)):
            if j == 0:  # Genotype column
                table[(i+1, j)].set_facecolor(color)
                table[(i+1, j)].set_text_props(weight='bold', color='white')
            else:
                # Alternate row shading
                if i % 2 == 0:
                    table[(i+1, j)].set_facecolor('#F5F5F5')
                else:
                    table[(i+1, j)].set_facecolor('#FFFFFF')

    plt.title('Phase 2 Exemplar Animals: Summary Statistics',
             fontsize=13, fontweight='bold', pad=20)

    plt.savefig(os.path.join(output_dir, 'phase2_exemplar_summary_table.png'),
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print("✓ Saved summary table visualization")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("PHASE 2 EXEMPLAR TRAJECTORIES - AESTHETIC REDESIGN")
    print("=" * 80)
    print()

    print("[1/2] Loading and preparing data...")
    loader = TrajectoryDataLoader()
    loader.load_all_animals()
    loader.select_exemplars()

    print("[2/2] Creating beautiful visualizations...")
    create_beautiful_trajectories(loader, OUTPUT_DIR)
    create_summary_table_visual(loader, OUTPUT_DIR)

    print()
    print("=" * 80)
    print("✓ AESTHETIC REDESIGN COMPLETE")
    print("=" * 80)
    print()
    print("New files created:")
    print("  - phase2_exemplar_trajectories_v2.png (improved aesthetics)")
    print("  - phase2_exemplar_trajectories_v2.pdf (vector format)")
    print("  - phase2_exemplar_summary_table.png (styled summary)")
