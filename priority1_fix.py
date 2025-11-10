# Quick fix for Priority 1 - just update fig3_side_bias function

def fix_chosen_side(df):
    """Convert chosen_side from text to numeric."""
    df = df.copy()
    df['chosen_side_num'] = (df['chosen_side'] == 'right').astype(int)
    return df

# Test the fixed version
import sys
sys.path.insert(0, '/home/user/GLMHMM')
from priority1_complete import *

# Patch the fig3 function
original_fig3 = CompletePriority1.fig3_side_bias

def fixed_fig3(self, rF, tF):
    """Fixed version with numeric conversion."""
    mm_res = [r for r in rF if r['genotype'] == '-/-']
    if not mm_res:
        print("  No -/- animals, skipping Fig 3")
        return

    mm_ids = [r['animal_id'] for r in mm_res]
    mm_tri = tF[tF['animal_id'].isin(mm_ids)].copy()
    
    # Convert chosen_side to numeric
    mm_tri['chosen_side_num'] = (mm_tri['chosen_side'] == 'right').astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Panel A: Position bias
    ax = axes[0, 0]
    if 'position' in mm_tri.columns:
        pos_stats = mm_tri.groupby('position').agg({
            'chosen_side_num': 'mean',  # Use numeric version
            'correct': 'mean'
        }).reset_index()

        ax.bar(pos_stats['position'], pos_stats['chosen_side_num'], alpha=0.6, color='#3498db')
        ax.axhline(y=0.5, color='gray', linestyle='--')
        ax.set_xlabel('Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('P(Right)', fontsize=12, fontweight='bold')
        ax.set_title('-/- Genotype: Choice Bias by Position', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)

    # Panel B: Bias over time
    ax = axes[0, 1]
    mm_tri_s = mm_tri.sort_values(['animal_id', 'session_date'])
    mm_tri_s['block'] = mm_tri_s.groupby('animal_id').cumcount() // 100

    block_stats = mm_tri_s.groupby(['animal_id', 'block']).agg({
        'chosen_side_num': lambda x: abs(x.mean() - 0.5)  # Use numeric
    }).reset_index()

    block_avg = block_stats.groupby('block').agg({'chosen_side_num': ['mean', 'sem']}).reset_index()
    x = block_avg['block']
    y = block_avg['chosen_side_num']['mean']
    sem = block_avg['chosen_side_num']['sem']

    ax.plot(x, y, linewidth=3, color='#e74c3c')
    ax.fill_between(x, y-sem, y+sem, color='#e74c3c', alpha=0.2)
    ax.set_xlabel('Block (×100 trials)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Side Bias', fontsize=12, fontweight='bold')
    ax.set_title('-/- Genotype: Side Bias Over Time', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # Panels C-D: Individual examples
    for i, (aid, ares) in enumerate(list(zip(mm_ids, mm_res))[:2]):
        ax = axes[1, i]
        a_tri = mm_tri[mm_tri['animal_id'] == aid].copy().sort_values('trial_num')
        a_tri['bias_roll'] = a_tri['chosen_side_num'].rolling(50, min_periods=1).apply(
            lambda x: abs(x.mean() - 0.5))  # Use numeric

        ax.plot(a_tri.index, a_tri['bias_roll'], linewidth=2, color='#9b59b6', alpha=0.8)
        ax.set_xlabel('Trial', fontsize=11, fontweight='bold')
        ax.set_ylabel('Side Bias', fontsize=11, fontweight='bold')

        states = [ares['broad_categories'][s][0] for s in range(ares['model'].n_states)]
        ax.set_title(f'{aid}\nStates: {", ".join(states)}', fontsize=12)
        ax.grid(alpha=0.3)

    fig.suptitle('-/- Genotype: Detailed Side Bias Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(self.output_dir / 'fig3_side_bias.png', dpi=300, bbox_inches='tight')
    plt.savefig(self.output_dir / 'fig3_side_bias.pdf', bbox_inches='tight')
    plt.close()

    print("✓ Figure 3: Side bias for -/-")

# Apply patch
CompletePriority1.fig3_side_bias = fixed_fig3

# Run
if __name__ == '__main__':
    main()
