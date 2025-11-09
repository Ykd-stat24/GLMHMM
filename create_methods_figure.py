"""
Priority 2: Methods Pipeline Figure
====================================

Creates a visual workflow showing the complete GLM-HMM analysis pipeline
from raw behavioral data to final state characterization.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def create_methods_pipeline():
    """
    Create comprehensive methods pipeline figure showing analysis workflow.
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Define colors
    color_data = '#3498db'  # Blue - data
    color_process = '#2ecc71'  # Green - processing
    color_model = '#e74c3c'  # Red - modeling
    color_validate = '#f39c12'  # Orange - validation
    color_analyze = '#9b59b6'  # Purple - analysis

    # Helper function to create boxes
    def add_box(x, y, width, height, text, color, fontsize=10, fontweight='bold'):
        """Add a fancy box with text."""
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=color,
            alpha=0.7,
            linewidth=2
        )
        ax.add_patch(box)

        # Add text
        ax.text(x + width/2, y + height/2, text,
               ha='center', va='center',
               fontsize=fontsize, fontweight=fontweight,
               wrap=True)

    def add_arrow(x1, y1, x2, y2, label='', style='->'):
        """Add arrow between boxes."""
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            mutation_scale=25,
            linewidth=2.5,
            color='black',
            alpha=0.7
        )
        ax.add_patch(arrow)

        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, label,
                   fontsize=9, style='italic',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # SECTION 1: RAW DATA (Top)
    add_box(0.5, 12, 9, 1.2,
            "RAW BEHAVIORAL DATA\nTrial-by-trial choices, outcomes, latencies, stimulus positions\nCohorts: W (n=19) & F (n=35) | Tasks: LD, PI",
            color_data, fontsize=11)

    # Arrow down
    add_arrow(5, 12, 5, 11.2)

    # SECTION 2: PREPROCESSING
    add_box(0.5, 9.5, 4, 1.5,
            "DATA PREPROCESSING\n• Filter non-reversal trials\n• Compute session metrics\n• Calculate WSLS ratios\n• Track side biases",
            color_process, fontsize=9)

    add_box(5.5, 9.5, 4, 1.5,
            "FEATURE ENGINEERING\n• Previous choice & outcome\n• Session progression\n• Task stage & experience\n• Cumulative trial count",
            color_process, fontsize=9)

    # Arrows to design matrix
    add_arrow(2.5, 9.5, 3.5, 8.5)
    add_arrow(7.5, 9.5, 6.5, 8.5)

    # Design matrix box
    add_box(2, 7.2, 6, 1.2,
            "DESIGN MATRIX (n_trials × 7 features)\nbias | prev_choice | WSLS | session_prog | side_bias | task_stage | experience",
            color_process, fontsize=9)

    # Arrow down
    add_arrow(5, 7.2, 5, 6.2)

    # SECTION 3: GLM-HMM MODEL
    add_box(1, 4.5, 3.5, 1.5,
            "GLM-HMM FITTING\n• EM algorithm (100 iter)\n• 3 hidden states\n• Bernoulli observations\n• L2 regularization",
            color_model, fontsize=9)

    add_box(5.5, 4.5, 3.5, 1.5,
            "MODEL COMPONENTS\n• Transition matrix (3×3)\n• GLM weights (3×7)\n• Initial state probs (3)\n• Viterbi decoding",
            color_model, fontsize=9)

    # Arrow down from GLM-HMM
    add_arrow(2.75, 4.5, 2.75, 3.5)

    # SECTION 4: STATE VALIDATION
    add_box(0.5, 2, 4, 1.3,
            "STATE VALIDATION\n• Accuracy > 65%: High-performance\n• Latency CV < 0.65: Procedural\n• WSLS ratio: Strategy use\n• Side bias: Perseveration",
            color_validate, fontsize=9)

    add_box(5.5, 2, 4, 1.3,
            "STATE LABELS\nBroad: Engaged / Lapsed / Mixed\nDetailed: Deliberative HP,\nProcedural HP, Disengaged,\nWSLS, Perseverative Bias",
            color_validate, fontsize=9)

    # Arrows to state validation
    add_arrow(7.25, 4.5, 7.25, 3.3)

    # Arrow down from validation
    add_arrow(5, 2, 5, 1)

    # SECTION 5: ANALYSIS & VISUALIZATION
    add_box(0.5, 0.2, 2.8, 0.7,
            "STATE DYNAMICS\n• Occupancy by genotype\n• P(state) over time\n• Transition patterns",
            color_analyze, fontsize=8)

    add_box(3.6, 0.2, 2.8, 0.7,
            "LEARNING CURVES\n• Accuracy trajectories\n• State overlays\n• Genotype comparisons",
            color_analyze, fontsize=8)

    add_box(6.7, 0.2, 2.8, 0.7,
            "STATISTICAL TESTS\n• Batch effects\n• Mixed models\n• Cross-validation",
            color_analyze, fontsize=8)

    # Side panel: Model validation
    add_box(10.2, 4.5, 2.3, 3,
            "MODEL\nVALIDATION\n\n3-fold CV\n2-5 states\n\nMetrics:\n• LL\n• AIC\n• BIC\n• Accuracy",
            '#95a5a6', fontsize=8)

    # Arrow from model to validation
    add_arrow(9, 5.2, 10.2, 5.2)

    # Add title
    ax.text(5, 13.5, 'GLM-HMM ANALYSIS PIPELINE',
           ha='center', fontsize=18, fontweight='bold')

    # Add phase labels
    ax.text(0.2, 11.8, '1', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
    ax.text(0.2, 10.2, '2', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
    ax.text(0.2, 5.2, '3', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
    ax.text(0.2, 2.6, '4', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
    ax.text(0.2, 0.5, '5', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

    plt.tight_layout()

    # Save figure
    output_dir = 'results/phase1_non_reversal/priority2_methods'
    import os
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(f'{output_dir}/methods_pipeline.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/methods_pipeline.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ Created methods pipeline figure")
    print(f"  Output: {output_dir}/")


def create_detailed_workflow():
    """
    Create detailed workflow with example data at each step.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Panel 1: Raw trial data (example)
    ax = axes[0, 0]
    ax.text(0.5, 0.9, 'Step 1: Raw Data', ha='center', fontsize=14, fontweight='bold',
           transform=ax.transAxes)

    example_data = [
        ['Trial', 'Stim', 'Choice', 'Correct'],
        ['1', 'L', 'L', '1'],
        ['2', 'R', 'R', '1'],
        ['3', 'L', 'R', '0'],
        ['4', 'R', 'L', '0'],
        ['5', 'L', 'L', '1'],
    ]

    table = ax.table(cellText=example_data, cellLoc='center', loc='center',
                    bbox=[0.1, 0.2, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.axis('off')

    # Panel 2: Features
    ax = axes[0, 1]
    ax.text(0.5, 0.9, 'Step 2: Feature Engineering', ha='center', fontsize=14, fontweight='bold',
           transform=ax.transAxes)

    features = ['bias', 'prev_choice', 'WSLS', 'session_prog', 'side_bias', 'task_stage', 'experience']
    y_pos = np.arange(len(features))
    example_values = [1.0, -1.0, -1.0, 0.2, 0.3, 1.0, 150]

    ax.barh(y_pos, example_values, color='#3498db', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Feature Value', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Panel 3: State sequence
    ax = axes[0, 2]
    ax.text(0.5, 0.95, 'Step 3: Hidden States', ha='center', fontsize=14, fontweight='bold',
           transform=ax.transAxes)

    trials = np.arange(50)
    states = np.random.choice([0, 1, 2], size=50, p=[0.5, 0.3, 0.2])
    colors = ['#2ecc71', '#e74c3c', '#f39c12']

    for s in range(3):
        mask = states == s
        ax.scatter(trials[mask], states[mask], s=100, alpha=0.7,
                  color=colors[s], label=f'State {s}')

    ax.set_xlabel('Trial Number', fontweight='bold')
    ax.set_ylabel('State', fontweight='bold')
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 4: GLM weights
    ax = axes[1, 0]
    ax.text(0.5, 0.95, 'Step 4: GLM Weights', ha='center', fontsize=14, fontweight='bold',
           transform=ax.transAxes)

    weights = np.array([
        [0.1, 0.8, 0.6, 0.3, -0.1, 0.2, 0.1],  # State 0: Engaged
        [0.05, -0.2, -0.3, -0.1, 0.7, 0.1, 0.0],  # State 1: Lapsed
        [0.08, 0.4, 0.2, 0.1, 0.3, 0.15, 0.05],  # State 2: Mixed
    ])

    im = ax.imshow(weights, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(7))
    ax.set_xticklabels(['bias', 'prev', 'WSLS', 'sess', 'side', 'task', 'exp'],
                       rotation=45, ha='right', fontsize=9)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Engaged', 'Lapsed', 'Mixed'])
    ax.set_ylabel('State', fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label('Weight', fontweight='bold')

    # Panel 5: State metrics
    ax = axes[1, 1]
    ax.text(0.5, 0.95, 'Step 5: State Validation', ha='center', fontsize=14, fontweight='bold',
           transform=ax.transAxes)

    metrics = ['Accuracy', 'WSLS Ratio', 'Latency CV', 'Side Bias']
    state_metrics = np.array([
        [0.85, 0.65, 0.45, 0.15],  # Engaged
        [0.52, 0.35, 0.75, 0.65],  # Lapsed
        [0.68, 0.50, 0.60, 0.35],  # Mixed
    ])

    x = np.arange(len(metrics))
    width = 0.25

    for i, (label, color) in enumerate([('Engaged', '#2ecc71'),
                                         ('Lapsed', '#e74c3c'),
                                         ('Mixed', '#f39c12')]):
        ax.bar(x + i*width, state_metrics[i], width, label=label,
              color=color, alpha=0.7)

    ax.set_ylabel('Value', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

    # Panel 6: Final output
    ax = axes[1, 2]
    ax.text(0.5, 0.95, 'Step 6: Analysis', ha='center', fontsize=14, fontweight='bold',
           transform=ax.transAxes)

    # Show genotype comparison
    genotypes = ['W+', 'W-', 'F+', 'F+/+', 'F+/-', 'F-/-']
    engaged_pct = [75, 55, 70, 65, 50, 35]

    bars = ax.bar(range(len(genotypes)), engaged_pct, color='#2ecc71', alpha=0.7)
    ax.set_ylabel('% Trials in Engaged State', fontweight='bold')
    ax.set_xticks(range(len(genotypes)))
    ax.set_xticklabels(genotypes, rotation=45, ha='right')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    # Overall title
    fig.suptitle('GLM-HMM Analysis: Step-by-Step Workflow with Examples',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save
    output_dir = 'results/phase1_non_reversal/priority2_methods'
    import os
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(f'{output_dir}/workflow_detailed.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/workflow_detailed.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ Created detailed workflow figure")


def main():
    """Create both methods figures."""
    print("="*80)
    print("CREATING METHODS PIPELINE FIGURES")
    print("="*80)

    print("\n[1/2] Creating pipeline flowchart...")
    create_methods_pipeline()

    print("\n[2/2] Creating detailed workflow...")
    create_detailed_workflow()

    print("\n" + "="*80)
    print("✓ METHODS FIGURES COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
