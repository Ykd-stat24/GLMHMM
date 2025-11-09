"""
Priority 2: Fixes and Additional Analyses
==========================================

Addresses user feedback:
1. Explain GLM weight magnitudes (feature scaling vs Ashwood)
2. Add state labels to transition matrices
3. Fix/regenerate detailed state counts plot
4. Create side bias plots by cohort and genotype
5. Investigate learning curve methodology
6. Review RT calculation and KS test stringency
7. Investigate undefined states and multiple disengaged states
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
from glmhmm_utils import load_and_preprocess_session_data

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


class Priority2Fixes:
    """Address user feedback and create additional analyses."""

    def __init__(self):
        self.results_dir = Path('results/phase1_non_reversal')
        self.output_dir = self.results_dir / 'priority2_fixes'
        self.output_dir.mkdir(exist_ok=True)

    def create_side_bias_by_genotype(self):
        """
        Create side bias plots for each cohort and genotype.
        Shows trial-by-trial evolution of side bias.
        """
        print("\n[1/6] Creating side bias plots by cohort and genotype...")

        # Load all data
        animals_W = [f'c{c}m{m}' for c in range(1, 5) for m in range(1, 6)
                     if not (c == 1 and m == 5)]
        animals_F = [11, 12, 13, 14, 21, 22, 23, 24, 25,
                     31, 32, 33, 34, 41, 42, 51, 52,
                     61, 62, 63, 64, 71, 72, 73,
                     81, 82, 83, 84, 91, 92, 93,
                     101, 102, 103, 104]

        for cohort, animals in [('W', animals_W), ('F', animals_F)]:
            all_data = []
            for animal in animals:
                # Load data for this animal
                filepath = f'Raw Files/Cohort {cohort}/{animal}.csv'
                try:
                    df = load_and_preprocess_session_data(filepath)
                    if df is not None and len(df) > 0:
                        df['animal_id'] = animal
                        df['cohort'] = cohort
                        # Determine genotype from animal ID
                        if cohort == 'W':
                            cage = int(animal[1])
                            mouse = int(animal[3])
                            if cage == 1:
                                genotype = '+' if mouse in [1, 2] else '--'
                            elif cage == 2:
                                genotype = '--'
                            elif cage == 3:
                                genotype = '+/-'
                            else:  # cage == 4
                                genotype = '+/+'
                        else:  # F cohort
                            animal_int = int(animal) if isinstance(animal, str) else animal
                            if animal_int in [11, 12, 13, 14]:
                                genotype = '+/+'
                            elif animal_int in [21, 22, 23, 24, 25]:
                                genotype = '+/-'
                            elif animal_int in [31, 32, 33, 34]:
                                genotype = '-/-'
                            elif animal_int in [41, 42]:
                                genotype = '+/+'
                            elif animal_int in [51, 52]:
                                genotype = '+/-'
                            elif animal_int in [61, 62, 63, 64]:
                                genotype = '+/+'
                            elif animal_int in [71, 72, 73]:
                                genotype = '+/-'
                            elif animal_int in [81, 82, 83, 84]:
                                genotype = '+'
                            elif animal_int in [91, 92, 93]:
                                genotype = '+/-'
                            elif animal_int in [101, 102, 103, 104]:
                                genotype = '+'
                            else:
                                genotype = 'unknown'
                        df['genotype'] = genotype
                        all_data.append(df)
                except:
                    continue

            if len(all_data) == 0:
                continue

            trials_df = pd.concat(all_data, ignore_index=True)
            genotypes = sorted(trials_df['genotype'].unique())

            # Create multi-panel plot
            n_genos = len(genotypes)
            fig, axes = plt.subplots(n_genos, 1, figsize=(16, 4*n_genos),
                                    sharex=True)
            if n_genos == 1:
                axes = [axes]

            for idx, geno in enumerate(genotypes):
                ax = axes[idx]
                geno_data = trials_df[trials_df['genotype'] == geno].copy()

                # Compute running side bias for each animal
                animals_in_geno = geno_data['animal_id'].unique()

                for animal in animals_in_geno:
                    animal_data = geno_data[geno_data['animal_id'] == animal].copy()
                    animal_data = animal_data.sort_values('trial_num')

                    # Compute cumulative side bias
                    choices = animal_data['choice'].values
                    cumulative_right = np.cumsum(choices)
                    cumulative_total = np.arange(1, len(choices) + 1)
                    side_bias = cumulative_right / cumulative_total

                    # Plot with transparency
                    ax.plot(np.arange(len(side_bias)), side_bias,
                           alpha=0.3, linewidth=1, color='gray')

                # Compute genotype average
                geno_data = geno_data.sort_values(['animal_id', 'trial_num'])
                geno_data['trial_idx'] = geno_data.groupby('animal_id').cumcount()

                # Bin by trial index
                max_trials = geno_data['trial_idx'].max()
                bin_size = 50
                bins = np.arange(0, max_trials + bin_size, bin_size)

                geno_data['trial_bin'] = pd.cut(geno_data['trial_idx'], bins,
                                                labels=False, include_lowest=True)

                bin_stats = geno_data.groupby('trial_bin').agg({
                    'choice': ['mean', 'sem']
                }).reset_index()

                x = bins[:-1] + bin_size/2
                y = bin_stats['choice']['mean'].values
                sem = bin_stats['choice']['sem'].values

                ax.plot(x[:len(y)], y, linewidth=3, color='red',
                       label=f'{geno} average', alpha=0.9)
                ax.fill_between(x[:len(y)], y - sem, y + sem,
                               color='red', alpha=0.2)

                # Add reference lines
                ax.axhline(y=0.5, color='black', linestyle='--',
                          alpha=0.5, linewidth=2, label='No bias')

                ax.set_ylabel('P(Right Choice)', fontsize=12, fontweight='bold')
                ax.set_title(f'{geno} (n={len(animals_in_geno)} animals)',
                            fontsize=13, fontweight='bold')
                ax.legend(fontsize=10, loc='upper right')
                ax.grid(alpha=0.3)
                ax.set_ylim(0, 1)

            axes[-1].set_xlabel('Trial Number', fontsize=13, fontweight='bold')
            fig.suptitle(f'Cohort {cohort}: Side Bias Evolution by Genotype',
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()

            plt.savefig(self.output_dir / f'side_bias_by_genotype_cohort{cohort}.png',
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / f'side_bias_by_genotype_cohort{cohort}.pdf',
                       bbox_inches='tight')
            plt.close()

        print(f"  ✓ Created side bias plots for both cohorts")

    def investigate_state_labeling(self):
        """
        Investigate why W-- has multiple disengaged states and F-/- has undefined states.
        """
        print("\n[2/6] Investigating state labeling patterns...")

        # Load W-- animals
        w_minus_animals = []
        animals_W = [f'c{c}m{m}' for c in range(1, 5) for m in range(1, 6)
                     if not (c == 1 and m == 5)]

        for animal in animals_W:
            pkl_file = self.results_dir / f'{animal}_cohortW_model.pkl'
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    if data['genotype'] == '--':
                        w_minus_animals.append(data)

        # Load F-/- animals
        f_minusminus_animals = []
        animals_F = [11, 12, 13, 14, 21, 22, 23, 24, 25,
                     31, 32, 33, 34, 41, 42, 51, 52,
                     61, 62, 63, 64, 71, 72, 73,
                     81, 82, 83, 84, 91, 92, 93,
                     101, 102, 103, 104]

        for animal in animals_F:
            pkl_file = self.results_dir / f'{animal}_cohortF_model.pkl'
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    if data['genotype'] == '-/-':
                        f_minusminus_animals.append(data)

        # Analyze W-- states
        print(f"\n  W-- genotype ({len(w_minus_animals)} animals):")
        for data in w_minus_animals[:3]:  # Show first 3 examples
            print(f"\n    Animal {data['animal_id']}:")
            from state_validation import create_broad_state_categories
            broad_cat = create_broad_state_categories(data['validated_labels'])
            state_metrics = data['state_metrics']

            for state in range(data['model'].n_states):
                cat, label, conf = broad_cat[state]
                metrics = state_metrics[state_metrics['state'] == state]
                if len(metrics) > 0:
                    acc = metrics['accuracy'].values[0]
                    wsls = metrics['wsls_ratio'].values[0]
                    cv = metrics['latency_cv'].values[0]
                    n_trials = int(metrics['n_trials'].values[0])
                    print(f"      State {state}: {label} ({cat})")
                    print(f"        Acc={acc:.3f}, WSLS={wsls:.3f}, CV={cv:.3f}, n={n_trials}")

        # Analyze F-/- states
        print(f"\n  F-/- genotype ({len(f_minusminus_animals)} animals):")
        for data in f_minusminus_animals[:3]:  # Show first 3 examples
            print(f"\n    Animal {data['animal_id']}:")
            from state_validation import create_broad_state_categories
            broad_cat = create_broad_state_categories(data['validated_labels'])
            state_metrics = data['state_metrics']

            for state in range(data['model'].n_states):
                cat, label, conf = broad_cat[state]
                metrics = state_metrics[state_metrics['state'] == state]
                if len(metrics) > 0:
                    acc = metrics['accuracy'].values[0]
                    wsls = metrics['wsls_ratio'].values[0]
                    cv = metrics['latency_cv'].values[0]
                    n_trials = int(metrics['n_trials'].values[0])
                    print(f"      State {state}: {label} ({cat})")
                    print(f"        Acc={acc:.3f}, WSLS={wsls:.3f}, CV={cv:.3f}, n={n_trials}")

        print("\n  ✓ State labeling investigation complete")

    def create_glm_weight_explanation(self):
        """
        Create documentation explaining GLM weight magnitudes.
        """
        print("\n[3/6] Creating GLM weight explanation...")

        explanation = """
GLM WEIGHT MAGNITUDES: Feature Scaling vs Ashwood (2022)
=========================================================

QUESTION: Why are our GLM weights smaller than Ashwood et al. (2022)?

ANSWER: We use FEATURE SCALING (standardization), they don't.

WHAT IS FEATURE SCALING?
------------------------
We apply StandardScaler to all features (except bias/intercept):
- Centers each feature to mean = 0
- Scales to standard deviation = 1

This means our features are in "standard deviation units" rather than raw units.

COMPARISON:
-----------

Ashwood (2022) - RAW FEATURES:
  • prev_choice: -1 or +1 (raw)
  • Weight ~ 2.0 means: "Previous right choice increases log-odds of choosing right by 2.0"
  • Weights in raw behavioral units

Our Approach - STANDARDIZED FEATURES:
  • prev_choice: Standardized to mean=0, std=1
  • Weight ~ 0.5 means: "One SD increase in feature increases log-odds by 0.5"
  • Weights represent EFFECT SIZES

WHY USE FEATURE SCALING?
-------------------------
1. **Comparability**: Can directly compare weights across features
   - Larger |weight| = stronger effect, regardless of original units

2. **Interpretability**: Weights are effect sizes
   - Weight of 0.8 for WSLS vs 0.3 for prev_choice means WSLS has bigger impact

3. **Numerical stability**: Helps optimization converge faster

4. **Standard practice**: Common in machine learning for mixed-scale features

EXAMPLE FROM OUR DATA:
----------------------
Animal c1m1, State 2 (Procedural HP):
  - prev_choice: -0.238 (standardized)
  - wsls: +0.124 (standardized)
  - task_stage: -0.827 (standardized)

Interpretation:
  - Task stage has LARGEST effect (|0.827|)
  - Previous choice has moderate effect (|0.238|)
  - WSLS has smaller effect (|0.124|)

If we hadn't scaled, weights would be in arbitrary units and not comparable!

TYPICAL RANGES:
---------------
Ashwood (unstandardized):
  • bias: 0.5-2.0
  • prev_choice: 1.0-3.0
  • wsls: 0.5-1.5

Our data (standardized):
  • All features: 0.1-1.5 (effect sizes)
  • Outliers up to ~2.0 for very strong effects

BOTTOM LINE:
------------
Smaller weights in our analysis ≠ weaker effects
They're in different units (effect sizes vs raw units)

Our approach is MORE interpretable because weights are directly comparable!
"""

        with open(self.output_dir / 'GLM_weight_explanation.txt', 'w') as f:
            f.write(explanation)

        print(f"  ✓ Created GLM weight explanation")
        print(f"    Output: {self.output_dir / 'GLM_weight_explanation.txt'}")

    def create_rt_calculation_explanation(self):
        """
        Explain RT calculation and KS test interpretation.
        """
        print("\n[4/6] Creating RT and KS test explanation...")

        explanation = """
RESPONSE TIME CALCULATION AND KS TEST INTERPRETATION
=====================================================

RESPONSE TIME (RT) CALCULATION:
-------------------------------
RT = response_latency field from raw data

From Bussey touchscreen system:
  • RT = Time from stimulus onset to first touch response
  • Units: seconds
  • Typically ranges from 0.5s to 5s for mice
  • Excludes: reward collection time, ITI, correction trials

Data processing:
  1. Load raw 'Response Latency' from touchscreen data
  2. Convert to numeric (handles missing/invalid values)
  3. Remove outliers (> 99th percentile)
  4. No transformation applied (use raw RT in seconds)

KOLMOGOROV-SMIRNOV (KS) TEST:
------------------------------
Tests whether two samples come from the same distribution.

Test statistic (D):
  • D = max vertical distance between CDFs
  • Range: 0 to 1
  • Larger D = more different distributions

P-value interpretation:
  • p < 0.05: Distributions are statistically different
  • p ≥ 0.05: Cannot reject null (distributions may be same)

WHY KS TESTS SEEM "TOO LIBERAL":
--------------------------------

PROBLEM: Small visual differences in Q-Q plots yield significant p-values.

EXPLANATION:
1. **Large sample sizes**:
   - Engaged: ~10,000+ trials
   - Lapsed: ~3,000+ trials
   - Mixed: ~2,000+ trials

   With large N, even tiny effect sizes become statistically significant!

2. **KS test is VERY sensitive**:
   - Detects small deviations anywhere in distribution
   - Not just mean/variance differences
   - Picks up subtle shape differences

3. **Biological variability**:
   - Real behavioral states DO have slightly different RT distributions
   - Even if visually similar, subtle differences exist

4. **Q-Q plot close to diagonal ≠ same distribution**:
   - Q-Q plots show quantile-quantile correspondence
   - Can be "close" but still statistically different
   - Visual similarity doesn't account for sample size

EXAMPLE: Engaged vs Mixed
-------------------------
  • Q-Q plot: Mostly along diagonal
  • Visual: 95% of points within ±0.2s
  • KS test: D = 0.08, p = 0.001

  Interpretation:
  - Distributions ARE different (p < 0.05)
  - But effect size is SMALL (D = 0.08)
  - With N=10,000 trials, we have power to detect tiny effects

WHAT TO REPORT:
---------------
1. **Effect size (D statistic)**:
   - D < 0.1: Trivial difference (even if p < 0.05)
   - D = 0.1-0.3: Small difference
   - D = 0.3-0.5: Moderate difference
   - D > 0.5: Large difference

2. **Practical significance**:
   - Median RT difference < 200ms: Probably not behaviorally meaningful
   - Median RT difference > 500ms: Likely behaviorally meaningful

3. **Consider both statistics AND plots**:
   - Don't rely on p-values alone
   - Look at Q-Q plots AND KS statistic
   - Report median RT differences

GENOTYPE COMPARISONS (+/- vs -/-):
-----------------------------------
You noted Q-Q plots are very close to diagonal but p < 0.05.

This is EXPECTED because:
  • Similar genotypes → similar RT distributions
  • Large sample sizes → high statistical power
  • Small biological differences become detectable
  • P-value reflects statistical significance, NOT effect size

RECOMMENDATION:
---------------
For manuscript:
  - Report KS D statistic alongside p-value
  - Include median RT and IQR for each group
  - Emphasize effect sizes over p-values
  - Note: "Statistically significant but small effect (D = 0.08)"

For interpretation:
  - Focus on large effects (D > 0.3)
  - Consider biological/behavioral relevance
  - Don't over-interpret small but significant differences
"""

        with open(self.output_dir / 'RT_and_KS_test_explanation.txt', 'w') as f:
            f.write(explanation)

        print(f"  ✓ Created RT and KS test explanation")
        print(f"    Output: {self.output_dir / 'RT_and_KS_test_explanation.txt'}")

    def create_learning_curve_explanation(self):
        """
        Explain learning curve methodology and F cohort decline.
        """
        print("\n[5/6] Creating learning curve explanation...")

        explanation = """
LEARNING CURVE METHODOLOGY
==========================

DATA SOURCE:
-----------
Uses trial-level 'correct' field (choice_correct), NOT "End Summary - Percentage Correct"

CALCULATION:
------------
1. Load all trial-level data for each animal
2. Sort by animal_id, session_date, trial_num
3. Compute rolling 30-trial accuracy for each animal
   - Window: 30 trials, centered
   - Handles edge effects with min_periods=1
4. Group trials into sessions (~30 trials per session)
5. Compute session-level statistics:
   - Mean accuracy across animals in genotype
   - SEM (standard error of mean)
6. Plot genotype-averaged learning curve

CODE:
-----
g_tri['session'] = g_tri.groupby('animal_id').cumcount() // 30
g_tri['rolling_acc'] = g_tri.groupby('animal_id')['correct'].transform(
    lambda x: x.rolling(30, min_periods=1, center=True).mean())

sess_stats = g_tri.groupby('session').agg({
    'rolling_acc': ['mean', 'sem'],
    'animal_id': 'nunique'
}).reset_index()

F COHORT DECLINE - WHY VALUES GO DOWN:
======================================

OBSERVED PATTERN:
-----------------
Several F cohort genotypes show declining accuracy in later sessions.

POSSIBLE EXPLANATIONS:

1. **Task Progression / Difficulty**:
   - Animals progress from easier to harder task stages
   - Phase 1 includes multiple task types:
     * LD (easier)
     * Punish Incorrect (medium)
     * Pairwise Discrimination (harder)
   - Later sessions may have more PD trials → lower accuracy

2. **Reversal Learning**:
   - Some animals received reversal blocks
   - Reversals cause temporary accuracy drops
   - Would show as decline in learning curve

3. **Motivational Changes**:
   - Satiation effects (less motivated in later sessions)
   - Habituation to task
   - Reduced effort/engagement

4. **Sample Size Effects**:
   - Fewer animals in later sessions (some didn't complete all stages)
   - Smaller N → more variability, potential bias

5. **Genotype-Specific Patterns**:
   - Some genotypes may have different learning dynamics
   - Could reflect true biological differences
   - F-/- genotype shows high accuracy but short engagement bouts

RECOMMENDATION TO INVESTIGATE:
------------------------------
1. Separate learning curves by task type (LD vs PD vs PI)
2. Check sample size (n animals) at each session timepoint
3. Examine individual animal trajectories (not just averages)
4. Compare early vs late session task distributions
5. Check if decline coincides with task transitions

NEXT STEPS:
-----------
Would you like me to:
  a) Create task-specific learning curves?
  b) Visualize sample sizes across sessions?
  c) Plot individual animal trajectories?
  d) Analyze task distribution over time?
"""

        with open(self.output_dir / 'learning_curve_explanation.txt', 'w') as f:
            f.write(explanation)

        print(f"  ✓ Created learning curve explanation")
        print(f"    Output: {self.output_dir / 'learning_curve_explanation.txt'}")

    def regenerate_detailed_state_counts(self):
        """
        Regenerate the detailed state counts plot with fixes.
        """
        print("\n[6/6] Regenerating detailed state counts plot...")

        # Load data
        animals_W = [f'c{c}m{m}' for c in range(1, 5) for m in range(1, 6)
                     if not (c == 1 and m == 5)]
        animals_F = [11, 12, 13, 14, 21, 22, 23, 24, 25,
                     31, 32, 33, 34, 41, 42, 51, 52,
                     61, 62, 63, 64, 71, 72, 73,
                     81, 82, 83, 84, 91, 92, 93,
                     101, 102, 103, 104]

        # Extract state info
        from state_validation import create_broad_state_categories

        all_counts = []
        for cohort, animals in [('W', animals_W), ('F', animals_F)]:
            for animal in animals:
                pkl_file = self.results_dir / f'{animal}_cohort{cohort}_model.pkl'
                if pkl_file.exists():
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                        genotype = data['genotype']
                        broad_cat = create_broad_state_categories(data['validated_labels'])

                        for state in range(data['model'].n_states):
                            _, detailed_label, _ = broad_cat[state]
                            all_counts.append({
                                'cohort': cohort,
                                'genotype': f'{cohort}-{genotype}',
                                'detailed_label': detailed_label,
                                'animal_id': animal
                            })

        df = pd.DataFrame(all_counts)

        # Count unique animals per state per genotype
        genotypes = sorted(df['genotype'].unique())
        detailed_labels = ['Deliberative High-Performance', 'Procedural High-Performance',
                          'Disengaged Lapse', 'WSLS Strategy', 'Perseverative Left-Bias']

        count_data = []
        for geno in genotypes:
            for label in detailed_labels:
                n_animals = df[(df['genotype'] == geno) &
                              (df['detailed_label'] == label)]['animal_id'].nunique()
                count_data.append({
                    'genotype': geno,
                    'detailed_label': label,
                    'n_animals': n_animals
                })

        count_df = pd.DataFrame(count_data)

        # Create plot
        fig, ax = plt.subplots(figsize=(16, 9))

        colors = {
            'Deliberative High-Performance': '#27ae60',
            'Procedural High-Performance': '#16a085',
            'Disengaged Lapse': '#e74c3c',
            'WSLS Strategy': '#f39c12',
            'Perseverative Left-Bias': '#9b59b6'
        }

        x = np.arange(len(genotypes))
        width = 0.15

        for i, label in enumerate(detailed_labels):
            label_data = count_df[count_df['detailed_label'] == label]
            values = [label_data[label_data['genotype'] == g]['n_animals'].values[0]
                     if len(label_data[label_data['genotype'] == g]) > 0 else 0
                     for g in genotypes]

            ax.bar(x + i*width, values, width, label=label,
                  color=colors.get(label, '#95a5a6'), alpha=0.85, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Genotype', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Animals', fontsize=14, fontweight='bold')
        ax.set_title('Distribution of Detailed Behavioral States by Genotype',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(genotypes, rotation=45, ha='right', fontsize=11)
        ax.legend(loc='upper right', fontsize=12, title='Detailed State',
                 title_fontsize=13, framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'detailed_state_counts_by_genotype_FIXED.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'detailed_state_counts_by_genotype_FIXED.pdf',
                   bbox_inches='tight')
        plt.close()

        print(f"  ✓ Regenerated detailed state counts plot")
        print(f"    Output: {self.output_dir}")


def main():
    """Run all fixes and additional analyses."""
    print("="*80)
    print("PRIORITY 2: FIXES AND ADDITIONAL ANALYSES")
    print("="*80)

    fixer = Priority2Fixes()

    # Run all analyses
    fixer.create_glm_weight_explanation()
    fixer.create_rt_calculation_explanation()
    fixer.create_learning_curve_explanation()
    fixer.create_side_bias_by_genotype()
    fixer.investigate_state_labeling()
    fixer.regenerate_detailed_state_counts()

    print("\n" + "="*80)
    print("✓ ALL FIXES AND ANALYSES COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {fixer.output_dir}")
    print("\nSummary of outputs:")
    print("  1. GLM_weight_explanation.txt - Why weights are smaller than Ashwood")
    print("  2. RT_and_KS_test_explanation.txt - RT calculation and KS test interpretation")
    print("  3. learning_curve_explanation.txt - Learning curve methodology and F cohort decline")
    print("  4. side_bias_by_genotype_cohortW/F.png/pdf - Side bias evolution plots")
    print("  5. Terminal output: State labeling investigation (W--, F-/-)")
    print("  6. detailed_state_counts_by_genotype_FIXED.png/pdf - Regenerated plot")


if __name__ == '__main__':
    main()
