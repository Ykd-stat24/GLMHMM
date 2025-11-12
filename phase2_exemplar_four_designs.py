"""
Phase 2 Individual Animal Trajectories - 4 DESIGN OPTIONS
Inspired by Ashwood et al. GLM-HMM paper aesthetic style

Creates 4 different visualization approaches:
1. Raster Plot Style - Each trial = small colored square
2. Stacked Bar + Accuracy Heatmap - Separate state and accuracy information
3. Minimalist Geological - Clean colored regions only
4. Multi-row Compact - Show all animals for comparison
"""

import os
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PHASE2_DIR = '/home/user/GLMHMM/results/phase2_reversal'
OUTPUT_DIR = '/home/user/GLMHMM/results/statistical_analysis/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Professional minimal color palette - inspired by Ashwood paper
STATE_COLORS = {
    0: '#1F78B4',    # Engaged - Blue (professional)
    1: '#E31A1C',    # Biased - Red (distinct)
    2: '#6A3D2A'     # Lapsed - Brown (muted, disengagement)
}

STATE_NAMES = {0: 'Engaged', 1: 'Biased', 2: 'Lapsed'}

OLD_TO_NEW = {
    '+/+': 'A1D_Wt', '-/-': 'A1D_KO', '+/-': 'A1D_Het',
    '+': 'B6', '-': 'C3H x B6'
}

GENO_COLORS = {
    'A1D_Wt': '#E63946', 'A1D_KO': '#457B9D', 'A1D_Het': '#A8DADC',
    'B6': '#2D3142', 'C3H x B6': '#8B5A8E'
}

# =============================================================================
# DATA LOADING
# =============================================================================

class TrajectoryLoader:
    def __init__(self):
        self.animals = {}
        self.exemplars = {}

    def load_all(self):
        phase2_files = sorted(glob.glob(os.path.join(PHASE2_DIR, 'models', '*_reversal.pkl')))
        for pkl_file in phase2_files:
            animal_id = os.path.basename(pkl_file).replace('_reversal.pkl', '')
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            genotype = OLD_TO_NEW.get(data.get('genotype', ''), data.get('genotype', ''))
            traj_df = data.get('trajectory_df', None)
            if traj_df is not None:
                self.animals[animal_id] = {'genotype': genotype, 'trajectory_df': traj_df}

        by_genotype = {}
        for animal_id, data in self.animals.items():
            geno = data['genotype']
            if geno not in by_genotype:
                by_genotype[geno] = []
            by_genotype[geno].append(animal_id)

        for genotype in sorted(by_genotype.keys()):
            self.exemplars[genotype] = sorted(by_genotype[genotype])[:2]

        print(f"Loaded {len(self.animals)} animals, {len(self.exemplars)} genotypes")

    def get_trajectory(self, animal_id):
        traj_df = self.animals[animal_id]['trajectory_df']
        states, accuracies, trials = [], [], []
        trial_num = 0
        for _, row in traj_df.iterrows():
            state = int(row['state'])
            bout_length = int(row['bout_length'])
            accuracy = row['during_accuracy']
            for _ in range(bout_length):
                states.append(state)
                accuracies.append(accuracy)
                trials.append(trial_num)
                trial_num += 1

        stats = {}
        for state in [0, 1, 2]:
            state_mask = traj_df['state'] == state
            if state_mask.sum() > 0:
                total = traj_df.loc[state_mask, 'bout_length'].sum()
                occ = 100 * total / traj_df['bout_length'].sum()
                acc = traj_df.loc[state_mask, 'during_accuracy'].mean()
                stats[STATE_NAMES[state]] = {'occ': occ, 'acc': acc}
        stats['overall'] = traj_df['during_accuracy'].mean()

        return {
            'states': np.array(states), 'accuracies': np.array(accuracies),
            'trials': np.arange(len(states)), 'traj_df': traj_df,
            'genotype': self.animals[animal_id]['genotype'], 'stats': stats
        }


# =============================================================================
# OPTION 1: RASTER PLOT STYLE
# =============================================================================

def create_raster_style(loader):
    """Each trial = small colored square. Minimalist and discrete."""
    exemplars = loader.exemplars
    n_genotypes = len(exemplars)

    fig, axes = plt.subplots(n_genotypes, 2, figsize=(16, 2.5 * n_genotypes))
    if n_genotypes == 1:
        axes = [axes]

    for row_idx, genotype in enumerate(sorted(exemplars.keys())):
        for col_idx, animal_id in enumerate(exemplars[genotype]):
            ax = axes[row_idx][col_idx]
            traj = loader.get_trajectory(animal_id)
            states = traj['states']
            accuracies = traj['accuracies']

            # Calculate square size based on number of trials
            n_trials = len(states)
            squares_per_row = min(100, max(50, int(np.sqrt(n_trials))))
            square_size = 3

            # Draw raster
            for trial_idx, (state, accuracy) in enumerate(zip(states, accuracies)):
                row = trial_idx // squares_per_row
                col = trial_idx % squares_per_row

                color = STATE_COLORS[state]
                alpha = 0.4 + 0.6 * accuracy  # Darker if low accuracy

                rect = Rectangle((col * square_size, -row * square_size), square_size - 0.2,
                               square_size - 0.2, facecolor=color, edgecolor='none', alpha=alpha)
                ax.add_patch(rect)

            ax.set_xlim(-1, (squares_per_row + 1) * square_size)
            ax.set_ylim(-(((n_trials - 1) // squares_per_row + 2) * square_size), square_size)
            ax.set_facecolor('#FFFFFF')
            ax.axis('off')

            # Title with stats
            title = f'{genotype} — {animal_id}'
            stats_text = f"Overall: {traj['stats']['overall']*100:.0f}% | " + \
                        " | ".join([f"{s}: {d['occ']:.0f}% occ, {d['acc']*100:.0f}% acc"
                                   for s, d in traj['stats'].items() if s != 'overall'])
            ax.text(0.5, 1.05, title, transform=ax.transAxes, fontsize=11, fontweight='700',
                   ha='center', va='bottom')
            ax.text(0.5, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                   ha='center', va='top', family='monospace', color='#666666')

    # Legend
    legend_elements = [mpatches.Patch(facecolor=STATE_COLORS[i], label=STATE_NAMES[i], alpha=0.8)
                      for i in range(3)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle('OPTION 1: Raster Plot Style — Each trial as colored square',
                fontsize=13, fontweight='700', y=0.98)

    plt.savefig(os.path.join(OUTPUT_DIR, 'phase2_exemplars_option1_raster.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Option 1: Raster Plot Style")


# =============================================================================
# OPTION 2: STACKED BAR + ACCURACY HEATMAP
# =============================================================================

def create_stacked_heatmap(loader):
    """Top: state blocks. Bottom: accuracy heatmap. Information-rich."""
    exemplars = loader.exemplars
    n_genotypes = len(exemplars)

    fig = plt.figure(figsize=(16, 3.5 * n_genotypes))
    gs = fig.add_gridspec(n_genotypes * 2, 2, hspace=0.5, wspace=0.25)

    for row_idx, genotype in enumerate(sorted(exemplars.keys())):
        for col_idx, animal_id in enumerate(exemplars[genotype]):
            # State plot
            ax_state = fig.add_subplot(gs[row_idx * 2, col_idx])
            # Accuracy heatmap
            ax_heat = fig.add_subplot(gs[row_idx * 2 + 1, col_idx])

            traj = loader.get_trajectory(animal_id)
            states = traj['states']
            accuracies = traj['accuracies']
            trials = traj['trials']

            # Plot state blocks
            current_state = states[0]
            start_idx = 0
            for idx in range(1, len(states) + 1):
                if idx == len(states) or states[idx] != current_state:
                    color = STATE_COLORS[current_state]
                    ax_state.barh(0, idx - start_idx, left=start_idx, height=0.6,
                                 color=color, edgecolor='white', linewidth=0.5)
                    if idx < len(states):
                        current_state = states[idx]
                        start_idx = idx

            ax_state.set_ylim(-0.5, 0.5)
            ax_state.set_xlim(-5, len(states) + 5)
            ax_state.set_facecolor('#FAFAFA')
            ax_state.set_yticks([])
            ax_state.spines['left'].set_visible(False)
            ax_state.spines['top'].set_visible(False)
            ax_state.spines['right'].set_visible(False)
            ax_state.set_ylabel('State', fontsize=9, fontweight='600')

            # Plot accuracy heatmap
            accuracy_2d = accuracies.reshape(1, -1)
            im = ax_heat.imshow(accuracy_2d, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                               interpolation='nearest')
            ax_heat.set_yticks([])
            ax_heat.set_facecolor('#FAFAFA')
            ax_heat.spines['left'].set_visible(False)
            ax_heat.spines['top'].set_visible(False)
            ax_heat.spines['right'].set_visible(False)
            ax_heat.set_ylabel('Accuracy', fontsize=9, fontweight='600')
            ax_heat.set_xlabel('Trial Number', fontsize=9, fontweight='600')

            # Title and stats
            title = f'{genotype} — {animal_id}'
            stats_text = f"Overall: {traj['stats']['overall']*100:.0f}%"
            ax_state.text(0.5, 1.15, title, transform=ax_state.transAxes, fontsize=11,
                         fontweight='700', ha='center')
            ax_state.text(0.5, 1.05, stats_text, transform=ax_state.transAxes, fontsize=9,
                         ha='center', family='monospace')

    legend_elements = [mpatches.Patch(facecolor=STATE_COLORS[i], label=STATE_NAMES[i])
                      for i in range(3)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('OPTION 2: Stacked Bar + Accuracy Heatmap — Information-rich view',
                fontsize=13, fontweight='700', y=0.98)

    plt.savefig(os.path.join(OUTPUT_DIR, 'phase2_exemplars_option2_stacked.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Option 2: Stacked Bar + Accuracy Heatmap")


# =============================================================================
# OPTION 3: MINIMALIST GEOLOGICAL
# =============================================================================

def create_minimalist_geological(loader):
    """Just colored regions like geological strata. Very clean. RECOMMENDED."""
    exemplars = loader.exemplars
    n_genotypes = len(exemplars)

    fig, axes = plt.subplots(n_genotypes, 2, figsize=(16, 2.2 * n_genotypes))
    if n_genotypes == 1:
        axes = [axes]

    for row_idx, genotype in enumerate(sorted(exemplars.keys())):
        for col_idx, animal_id in enumerate(exemplars[genotype]):
            ax = axes[row_idx][col_idx]
            traj = loader.get_trajectory(animal_id)
            states = traj['states']
            traj_df = traj['traj_df']

            # Draw state regions as geological layers
            current_state = states[0]
            start_idx = 0
            for idx in range(1, len(states) + 1):
                if idx == len(states) or states[idx] != current_state:
                    color = STATE_COLORS[current_state]
                    rect = Rectangle((start_idx, 0), idx - start_idx, 1,
                                   facecolor=color, edgecolor='#333333', linewidth=0.5, alpha=0.85)
                    ax.add_patch(rect)

                    # Label if bout is large enough
                    if idx - start_idx > 50:
                        mid_x = (start_idx + idx) / 2
                        ax.text(mid_x, 0.5, STATE_NAMES[current_state],
                               ha='center', va='center', fontsize=8, fontweight='600',
                               color='white', alpha=0.7)

                    if idx < len(states):
                        current_state = states[idx]
                        start_idx = idx

            ax.set_xlim(-10, len(states) + 10)
            ax.set_ylim(-0.1, 1.2)
            ax.set_facecolor('#FAFAFA')
            ax.set_xlabel('Trial Number', fontsize=10, fontweight='600')
            ax.set_yticks([])

            # Professional spines
            for spine in ['top', 'right', 'left']:
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.8)
            ax.spines['bottom'].set_color('#333333')

            # Title with minimal stats
            title = f'{genotype} — {animal_id}'
            overall = traj['stats']['overall']

            # Compute bout counts per state
            bout_counts = {0: 0, 1: 0, 2: 0}
            for state in traj_df['state'].values:
                bout_counts[int(state)] += 1

            stats_text = f"Overall Acc: {overall*100:.0f}% | " + \
                        " | ".join([f"{STATE_NAMES[i]}: {bout_counts[i]} bouts"
                                   for i in range(3)])

            ax.text(0.5, 1.10, title, transform=ax.transAxes, fontsize=11,
                   fontweight='700', ha='center')
            ax.text(0.5, 1.03, stats_text, transform=ax.transAxes, fontsize=8,
                   ha='center', family='monospace', color='#444444')

    legend_elements = [mpatches.Patch(facecolor=STATE_COLORS[i], label=STATE_NAMES[i], alpha=0.85)
                      for i in range(3)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle('OPTION 3: Minimalist Geological Style — Clean, focused on state transitions ★',
                fontsize=13, fontweight='700', y=0.98)

    plt.savefig(os.path.join(OUTPUT_DIR, 'phase2_exemplars_option3_minimal.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Option 3: Minimalist Geological (RECOMMENDED)")


# =============================================================================
# OPTION 4: MULTI-ROW COMPACT
# =============================================================================

def create_multirow_compact(loader):
    """All animals at once. Very compact. Compare patterns across animals."""
    exemplars = loader.exemplars
    n_animals = sum(len(animals) for animals in exemplars.values())

    fig, axes = plt.subplots(n_animals, 1, figsize=(16, 0.8 * n_animals))
    if n_animals == 1:
        axes = [axes]

    ax_idx = 0
    for genotype in sorted(exemplars.keys()):
        for animal_id in exemplars[genotype]:
            ax = axes[ax_idx]
            traj = loader.get_trajectory(animal_id)
            states = traj['states']

            # Draw state blocks
            current_state = states[0]
            start_idx = 0
            for idx in range(1, len(states) + 1):
                if idx == len(states) or states[idx] != current_state:
                    color = STATE_COLORS[current_state]
                    rect = Rectangle((start_idx, 0), idx - start_idx, 1,
                                   facecolor=color, edgecolor='#999999', linewidth=0.3, alpha=0.85)
                    ax.add_patch(rect)
                    if idx < len(states):
                        current_state = states[idx]
                        start_idx = idx

            ax.set_xlim(-5, len(states) + 5)
            ax.set_ylim(-0.2, 1.2)
            ax.set_facecolor('#FAFAFA')
            ax.set_yticks([])

            # Left label
            overall = traj['stats']['overall']
            label_text = f"{genotype} {animal_id} ({overall*100:.0f}%)"
            ax.text(-len(states) * 0.02, 0.5, label_text, fontsize=8, fontweight='600',
                   ha='right', va='center', transform=ax.get_xaxis_transform())

            # Remove spines
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(False)

            ax_idx += 1

    axes[-1].set_xlabel('Trial Number', fontsize=10, fontweight='600')

    legend_elements = [mpatches.Patch(facecolor=STATE_COLORS[i], label=STATE_NAMES[i], alpha=0.85)
                      for i in range(3)]
    fig.legend(handles=legend_elements, loc='lower right', ncol=3, fontsize=9, bbox_to_anchor=(0.98, 0.01))

    fig.suptitle('OPTION 4: Multi-row Compact — Compare all exemplars at once',
                fontsize=13, fontweight='700', y=0.995)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'phase2_exemplars_option4_multirow.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Option 4: Multi-row Compact")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("PHASE 2 EXEMPLAR TRAJECTORIES - 4 DESIGN OPTIONS")
    print("=" * 80)
    print()

    print("[1/5] Loading data...")
    loader = TrajectoryLoader()
    loader.load_all()

    print("[2/5] Creating Option 1: Raster Plot Style...")
    create_raster_style(loader)

    print("[3/5] Creating Option 2: Stacked Bar + Accuracy Heatmap...")
    create_stacked_heatmap(loader)

    print("[4/5] Creating Option 3: Minimalist Geological...")
    create_minimalist_geological(loader)

    print("[5/5] Creating Option 4: Multi-row Compact...")
    create_multirow_compact(loader)

    print()
    print("=" * 80)
    print("✓ ALL 4 OPTIONS CREATED")
    print("=" * 80)
    print()
    print("Output files:")
    print("  1. phase2_exemplars_option1_raster.png (discrete trial squares)")
    print("  2. phase2_exemplars_option2_stacked.png (state bars + accuracy heatmap)")
    print("  3. phase2_exemplars_option3_minimal.png (clean geological layers) ★")
    print("  4. phase2_exemplars_option4_multirow.png (all animals in one view)")
    print()
    print("RECOMMENDATION: Option 3 (Minimalist Geological) is most professional")
    print("=" * 80)
