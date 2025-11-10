"""
Create Transition Matrices with State Labels
=============================================

Regenerates transition matrices with detailed state labels instead of just S0, S1, S2.
Also provides guidance on how to compare matrices between genotypes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import utilities
import sys
sys.path.insert(0, '/home/user/GLMHMM')
from state_validation import create_broad_state_categories

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def create_labeled_transition_matrices():
    """Create transition matrices with state labels for both cohorts."""

    results_dir = Path('results/phase1_non_reversal')
    output_dir = results_dir / 'priority2_fixes'
    output_dir.mkdir(exist_ok=True)

    # Define animals
    animals_W = [f'c{c}m{m}' for c in range(1, 5) for m in range(1, 6)
                 if not (c == 1 and m == 5)]
    animals_F = [11, 12, 13, 14, 21, 22, 23, 24, 25,
                 31, 32, 33, 34, 41, 42, 51, 52,
                 61, 62, 63, 64, 71, 72, 73,
                 81, 82, 83, 84, 91, 92, 93,
                 101, 102, 103, 104]

    for cohort, animals in [('W', animals_W), ('F', animals_F)]:
        print(f"\nProcessing Cohort {cohort}...")

        # Load all results
        results = []
        for animal in animals:
            pkl_file = results_dir / f'{animal}_cohort{cohort}_model.pkl'
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    # Add broad categories
                    broad_categories = create_broad_state_categories(data['validated_labels'])
                    data['broad_categories'] = broad_categories
                    results.append(data)

        if len(results) == 0:
            print(f"  No results found for cohort {cohort}")
            continue

        n_states = results[0]['model'].n_states

        # Group by genotype
        genotype_groups = {}
        for r in results:
            g = r['genotype']
            if g not in genotype_groups:
                genotype_groups[g] = []
            genotype_groups[g].append(r)

        unique_genotypes = sorted(genotype_groups.keys())
        n_genotypes = len(unique_genotypes)

        # Create multi-panel figure
        fig, axes = plt.subplots(1, n_genotypes, figsize=(7*n_genotypes, 6))
        if n_genotypes == 1:
            axes = [axes]

        for ax, genotype in zip(axes, unique_genotypes):
            # Collect transition matrices
            trans_matrices = []

            # Get most common state labels for this genotype
            state_label_counts = {i: defaultdict(int) for i in range(n_states)}

            for r in genotype_groups[genotype]:
                # Count state labels
                broad_cat = r['broad_categories']
                for state in range(n_states):
                    _, detailed_label, _ = broad_cat[state]
                    state_label_counts[state][detailed_label] += 1

                # Compute transition matrix
                state_seq = r['model'].most_likely_states
                trans_mat = np.zeros((n_states, n_states))

                for i in range(len(state_seq) - 1):
                    s1 = state_seq[i]
                    s2 = state_seq[i+1]
                    trans_mat[s1, s2] += 1

                # Normalize
                row_sums = trans_mat.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                trans_mat = trans_mat / row_sums

                trans_matrices.append(trans_mat)

            # Get most common label for each state
            state_labels = []
            for state in range(n_states):
                if len(state_label_counts[state]) > 0:
                    most_common = max(state_label_counts[state].items(),
                                    key=lambda x: x[1])[0]
                    # Abbreviate label
                    abbrev = {
                        'Deliberative High-Performance': 'Delib HP',
                        'Procedural High-Performance': 'Proc HP',
                        'Disengaged Lapse': 'Disengaged',
                        'WSLS Strategy': 'WSLS',
                        'Perseverative Left-Bias': 'Persever',
                        'Perseverative Right-Bias': 'Persever',
                        'Undefined State 0.0': 'Undef-0',
                        'Undefined State 1.0': 'Undef-1',
                        'Undefined State 2.0': 'Undef-2'
                    }
                    label = abbrev.get(most_common, most_common[:10])
                    state_labels.append(f'S{state}\n{label}')
                else:
                    state_labels.append(f'S{state}')

            # Average transition matrix
            avg_trans = np.mean(trans_matrices, axis=0)
            std_trans = np.std(trans_matrices, axis=0)

            # Plot heatmap
            im = ax.imshow(avg_trans, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')

            # Add text annotations
            for i in range(n_states):
                for j in range(n_states):
                    text_color = 'white' if avg_trans[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{avg_trans[i,j]:.2f}',
                           ha='center', va='center', color=text_color,
                           fontsize=11, fontweight='bold')

            # Set axis labels with state names
            ax.set_xticks(range(n_states))
            ax.set_yticks(range(n_states))
            ax.set_xticklabels(state_labels, fontsize=10)
            ax.set_yticklabels(state_labels, fontsize=10)
            ax.set_xlabel('To State', fontsize=12, fontweight='bold')
            ax.set_ylabel('From State', fontsize=12, fontweight='bold')
            ax.set_title(f'{genotype} (n={len(genotype_groups[genotype])})',
                        fontsize=14, fontweight='bold')

            # Colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Transition Probability', fontsize=11)

        fig.suptitle(f'Cohort {cohort}: Transition Matrices by Genotype\n' +
                    '(With State Labels)',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.savefig(output_dir / f'cohort_{cohort}_transition_matrices_LABELED.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'cohort_{cohort}_transition_matrices_LABELED.pdf',
                   bbox_inches='tight')
        plt.close()

        print(f"  ✓ Created labeled transition matrices for cohort {cohort}")

    # Create comparison guidance document
    comparison_guide = """
TRANSITION MATRIX COMPARISON GUIDE
===================================

WHAT IS A TRANSITION MATRIX?
-----------------------------
A transition matrix shows the probability of switching between behavioral states.

Each cell (i,j) = P(transitioning from state i to state j)

Example:
        To: Engaged  Lapsed  Mixed
  From:
  Engaged    0.85     0.10    0.05    → 85% stay engaged, 10% lapse, 5% mixed
  Lapsed     0.20     0.70    0.10    → 70% stay lapsed, 20% recover to engaged
  Mixed      0.40     0.20    0.40    → 40% stay mixed, 40% to engaged, 20% lapse

INTERPRETING THE MATRIX:
------------------------
1. **Diagonal values (self-transitions)**:
   - High values (>0.7): State is stable, animals tend to stay in state
   - Low values (<0.5): State is transient, animals frequently switch

2. **Off-diagonal values (state switches)**:
   - Large values indicate common transitions
   - Small values indicate rare transitions

3. **Row patterns**:
   - Each row sums to 1.0 (probabilities)
   - Compare rows to understand "exit patterns" from each state

HOW TO COMPARE BETWEEN GENOTYPES:
----------------------------------

1. **Visual Comparison**:
   - Look for differences in color patterns
   - Compare diagonal values (state stability)
   - Check which off-diagonal transitions are prominent

2. **Key Metrics to Compare**:

   a) **State Persistence** (diagonal values):
      - P(Engaged → Engaged): How stable is engaged state?
      - P(Lapsed → Lapsed): Do animals get "stuck" in lapsed state?
      - Higher persistence = more stable states

   b) **Recovery Rate** (Lapsed → Engaged):
      - P(Lapsed → Engaged): How quickly do animals recover?
      - Higher values = better recovery
      - Genotype differences here may indicate resilience

   c) **Lapse Entry** (Engaged → Lapsed):
      - P(Engaged → Lapsed): How often do animals lose focus?
      - Lower values = more stable performance
      - Higher values = more vulnerable to distraction

   d) **Mixed State Dynamics**:
      - P(Mixed → Engaged): Progressing toward expertise?
      - P(Mixed → Lapsed): Falling into disengagement?

3. **Statistical Comparison**:

   For each cell (i,j), compare genotypes:
   - Compute mean transition probability across animals
   - Compute standard deviation (shown as ± values)
   - Larger SD = more heterogeneity within genotype

   Example comparison:
   - Genotype +/+: P(Engaged → Engaged) = 0.85 ± 0.05
   - Genotype -/-: P(Engaged → Engaged) = 0.70 ± 0.12

   Interpretation:
   - +/+ animals have more stable engaged state
   - -/- animals have more variable performance

4. **Patterns to Look For**:

   a) **"Sticky" states**:
      - One state has very high self-transition (>0.9)
      - Suggests animals get "trapped" in that state

   b) **Cyclic patterns**:
      - Strong A→B and B→A transitions
      - Suggests oscillation between states

   c) **Unidirectional flow**:
      - Strong A→B but weak B→A
      - Suggests learning progression or deterioration

   d) **Random switching**:
      - All off-diagonal values similar (~1/3 for 3 states)
      - Suggests no structured state dynamics

EXAMPLE INTERPRETATIONS:
------------------------

**High-Performing Genotype:**
  - High P(Engaged → Engaged): 0.85+ (stable performance)
  - Low P(Engaged → Lapsed): <0.10 (resistant to distraction)
  - High P(Lapsed → Engaged): >0.30 (quick recovery)
  - P(Lapsed → Lapsed): <0.60 (don't get stuck)

**Impaired Genotype:**
  - Lower P(Engaged → Engaged): 0.60-0.70 (less stable)
  - Higher P(Engaged → Lapsed): 0.20-0.30 (more distractible)
  - Lower P(Lapsed → Engaged): <0.20 (slow recovery)
  - Higher P(Lapsed → Lapsed): >0.70 (stuck in lapse)

STATISTICAL TESTS:
------------------
To formally test genotype differences:

1. **Cell-by-cell t-tests**:
   - For each transition (i→j), compare genotypes
   - Correct for multiple comparisons (Bonferroni or FDR)

2. **Matrix distance metrics**:
   - Frobenius norm: ||Matrix1 - Matrix2||
   - Measures overall similarity
   - Can use permutation tests for significance

3. **Key transition comparisons**:
   - Focus on theoretically important transitions
   - Reduces multiple comparison burden
   - e.g., only test recovery and lapse entry rates

COHORT-SPECIFIC NOTES:
----------------------

Your data shows:
  - Cohort W: Generally well-defined states
  - Cohort F: Some genotypes have "Undefined" states
    → May have less clear state structure
    → Transitions harder to interpret

For genotypes with undefined states:
  - Transitions may not have clear behavioral meaning
  - Consider whether 3-state model is appropriate
  - May need to focus on defined states only

RECOMMENDATIONS:
----------------
1. Create a table comparing key transitions across genotypes
2. Focus on biologically meaningful transitions
3. Report both mean and variability (SD)
4. Consider effect sizes, not just statistical significance
5. Relate transition patterns to behavioral outcomes
"""

    with open(output_dir / 'transition_matrix_comparison_guide.txt', 'w') as f:
        f.write(comparison_guide)

    print(f"\n✓ Created transition matrix comparison guide")
    print(f"  Output: {output_dir / 'transition_matrix_comparison_guide.txt'}")


def main():
    print("="*80)
    print("CREATING LABELED TRANSITION MATRICES")
    print("="*80)

    create_labeled_transition_matrices()

    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
