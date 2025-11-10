"""
Late Lapser Analysis and Poor Performer Investigation
======================================================

Addresses critical validity questions:
1. How is latency calculated? (Check F-/- CVs)
2. Review F-/- animals comprehensively
3. Learning curves using End Summary - Percentage Correct
4. Identify animals that lapse/stall later in Phase 1 (especially in PI)
5. Determine if poor performers are driving genotype differences
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

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


class LateLapserAnalysis:
    """Identify animals that deteriorate later in training."""

    def __init__(self):
        self.results_dir = Path('results/phase1_non_reversal')
        self.output_dir = self.results_dir / 'late_lapser_analysis'
        self.output_dir.mkdir(exist_ok=True)

    def explain_latency_calculation(self):
        """
        Document how latency is calculated in our analysis.
        """
        explanation = """
LATENCY CALCULATION IN OUR ANALYSIS
====================================

QUESTION: How is latency calculated? Are different latencies weighted differently?

ANSWER: We use IMAGE RESPONSE LATENCY only. No weighting.

DETAILED EXPLANATION:
--------------------

1. **Which Latency Field?**

   Raw data contains multiple latency fields:
   - "Correct touch latency" - Time to touch correct stimulus
   - "Blank Touch Latency" - Time to touch blank (incorrect) location
   - "Reward Collection Latency" - Time to collect reward after correct response
   - "Trial Analysis - Correct Image Response Latency (N)" - Per-trial correct
   - "Trial Analysis - Incorrect Image Latency (N)" - Per-trial incorrect

   **We use**: Trial Analysis - [Correct/Incorrect] Image Response Latency

   This is the time from stimulus presentation to first image touch.

2. **How It's Extracted** (from glmhmm_utils.py lines 100-114):

   ```python
   if correct:
       latency_col = f'Trial Analysis - Correct Image Response Latency ({trial_num})'
   else:
       latency_col = f'Trial Analysis - Incorrect Image Latency ({trial_num})'

   latency_val = row.get(latency_col, np.nan)
   ```

   - Extracts trial-by-trial response latency
   - Uses correct image latency for correct trials
   - Uses incorrect image latency for incorrect trials
   - Missing values coded as NaN

3. **No Weighting Applied**:

   - We do NOT weight by:
     * Correct vs incorrect trials
     * Blank touch latency
     * Reward collection latency

   - Each trial has one latency value (response time)
   - All trials weighted equally in CV calculation

4. **Latency CV Calculation**:

   For each behavioral state:
   ```python
   latencies = state_trials['latency'].values
   latency_mean = np.mean(latencies)
   latency_std = np.std(latencies)
   latency_cv = latency_std / latency_mean
   ```

   - CV = Coefficient of Variation = std / mean
   - Measures relative variability (dimensionless)
   - High CV = inconsistent responding
   - Low CV (<0.65) = habitual/procedural responding

5. **Special Cases**:

   - **Missing latencies** (coded as '-' or empty):
     * Converted to NaN
     * Excluded from CV calculation

   - **Zero mean latencies**:
     * Would cause division by zero
     * Handled by setting CV to NaN

   - **All NaN latencies**:
     * No valid response times recorded
     * Entire CV becomes NaN
     * This is why F-/- animals have NaN CVs!

6. **F-/- Latency CV Issue**:

   Analysis shows F-/- animals have **NaN latency CVs** because:

   Possible causes:
   a) **Data recording errors**: Latency fields not populated
   b) **Timeouts**: Animals not responding within time limit
   c) **Data processing errors**: Latency values coded as '-' or invalid
   d) **Very low trial counts**: Not enough trials to compute meaningful CV

   **Recommendation**:
   - Check raw data files for F-/- animals
   - Verify latency fields are populated
   - May need to exclude F-/- from latency-based analyses

WHAT LATENCY REPRESENTS:
------------------------

**Response Latency** = Decision time + motor execution time

- Short latency (~0.5-1s): Fast, possibly habitual responding
- Medium latency (~1-2s): Normal decision-making
- Long latency (>3s): Slow/deliberative, or disengaged

**Why CV matters**:
- Low CV = Consistent strategy (habitual or efficient)
- High CV = Variable strategy (exploring, switching, uncertain)

DOES NOT INCLUDE:
-----------------
- ✗ Reward collection time (measured separately)
- ✗ ITI touches (not part of trial)
- ✗ Correction trial latencies
- ✗ Initiation latency (nose poke to stimulus)

BOTTOM LINE:
-----------
- We use simple image response latency (stimulus → touch)
- No weighting by trial type or outcome
- F-/- animals have missing latency data → NaN CVs
- This prevents automated state classification for F-/-
"""

        with open(self.output_dir / 'latency_calculation_explained.txt', 'w') as f:
            f.write(explanation)

        print("✓ Created latency calculation explanation")

    def review_f_minus_minus_animals(self):
        """
        Comprehensive review of all F-/- animals.
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE F-/- ANIMAL REVIEW")
        print("="*80)

        # F-/- animals: 31, 32, 33, 34
        f_minus_animals = [31, 32, 33, 34]

        review_data = []

        for animal in f_minus_animals:
            pkl_file = self.results_dir / f'{animal}_cohortF_model.pkl'

            if not pkl_file.exists():
                print(f"\n✗ Animal {animal}: No model file found")
                continue

            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            print(f"\n{'='*80}")
            print(f"ANIMAL {animal} (F-/-)")
            print(f"{'='*80}")

            n_trials = data['n_trials']
            state_metrics = data['state_metrics']
            validated_labels = data['validated_labels']

            print(f"\nTotal trials: {n_trials}")
            print(f"Number of states: {data['model'].n_states}")

            # Check for latency data
            from state_validation import create_broad_state_categories
            broad_cat = create_broad_state_categories(validated_labels)

            print(f"\nState-by-state breakdown:")
            print(f"{'-'*80}")

            for state in range(data['model'].n_states):
                metrics = state_metrics[state_metrics['state'] == state]

                if len(metrics) > 0:
                    cat, label, conf = broad_cat[state]

                    acc = metrics['accuracy'].values[0]
                    wsls = metrics['wsls_ratio'].values[0]
                    cv = metrics['latency_cv'].values[0]
                    lat_mean = metrics['latency_mean'].values[0]
                    lat_std = metrics['latency_std'].values[0]
                    side_bias = metrics['side_bias'].values[0]
                    n_state_trials = int(metrics['n_trials'].values[0])
                    occupancy = metrics['occupancy'].values[0]

                    print(f"\nState {state}: {label} ({cat})")
                    print(f"  Trials: {n_state_trials} ({occupancy*100:.1f}% of total)")
                    print(f"  Accuracy: {acc:.3f}")
                    print(f"  WSLS ratio: {wsls:.3f}")
                    print(f"  Latency: mean={lat_mean:.3f}s, std={lat_std:.3f}s, CV={cv:.3f}")
                    print(f"  Side bias: {side_bias:.3f}")

                    # Store for summary
                    review_data.append({
                        'animal_id': animal,
                        'state': state,
                        'label': label,
                        'category': cat,
                        'n_trials': n_state_trials,
                        'occupancy': occupancy,
                        'accuracy': acc,
                        'wsls_ratio': wsls,
                        'latency_mean': lat_mean,
                        'latency_std': lat_std,
                        'latency_cv': cv,
                        'side_bias': side_bias
                    })

        # Create summary table
        if len(review_data) > 0:
            df = pd.DataFrame(review_data)
            df.to_csv(self.output_dir / 'F_minus_minus_animal_review.csv', index=False)

            print(f"\n{'='*80}")
            print("SUMMARY STATISTICS FOR F-/- GENOTYPE")
            print(f"{'='*80}")

            print(f"\nOverall metrics (averaged across all states and animals):")
            print(f"  Mean accuracy: {df['accuracy'].mean():.3f} ± {df['accuracy'].std():.3f}")
            print(f"  Mean WSLS: {df['wsls_ratio'].mean():.3f} ± {df['wsls_ratio'].std():.3f}")
            print(f"  Mean latency CV: {df['latency_cv'].mean():.3f} (many NaN)")
            print(f"  States with NaN CV: {df['latency_cv'].isna().sum()} / {len(df)}")

            print(f"\n✓ Saved detailed review to: {self.output_dir / 'F_minus_minus_animal_review.csv'}")

    def create_end_summary_learning_curves(self):
        """
        Create learning curves using 'End Summary - % Correct (1)' field.
        """
        print("\n" + "="*80)
        print("CREATING LEARNING CURVES FROM END SUMMARY DATA")
        print("="*80)

        # Load processed data files
        cohort_files = {
            'W': 'W LD Data 11.08 All_processed.csv',
            'F': 'F LD Data 11.08 All_processed.csv'
        }

        fig, axes = plt.subplots(2, 1, figsize=(18, 12))

        for ax, (cohort, filepath) in zip(axes, cohort_files.items()):
            if not Path(filepath).exists():
                print(f"  Warning: {filepath} not found, skipping cohort {cohort}")
                continue

            # Load data
            df = pd.read_csv(filepath)

            # Extract relevant columns
            if 'End Summary - Percentage Correct (1)' not in df.columns:
                print(f"  Warning: 'End Summary - Percentage Correct (1)' not found in {filepath}")
                continue

            # Clean data
            df['pct_correct'] = pd.to_numeric(df['End Summary - Percentage Correct (1)'], errors='coerce') / 100
            df['animal_id'] = df['Animal ID']
            df['session_date'] = pd.to_datetime(df['Schedule run date'], errors='coerce')

            # Add genotype
            if cohort == 'W':
                def get_genotype_W(animal_id):
                    if pd.isna(animal_id) or not isinstance(animal_id, str):
                        return 'unknown'
                    try:
                        cage = int(animal_id[1])
                        mouse = int(animal_id[3])
                        if cage == 1:
                            return '+' if mouse in [1, 2] else '--'
                        elif cage == 2:
                            return '--'
                        elif cage == 3:
                            return '+/-'
                        else:
                            return '+/+'
                    except:
                        return 'unknown'
                df['genotype'] = df['animal_id'].apply(get_genotype_W)
            else:  # F cohort
                def get_genotype_F(animal_id):
                    try:
                        aid = int(animal_id)
                        if aid in [11, 12, 13, 14, 41, 42, 61, 62, 63, 64]:
                            return '+/+'
                        elif aid in [21, 22, 23, 24, 25, 51, 52, 71, 72, 73, 91, 92, 93]:
                            return '+/-'
                        elif aid in [31, 32, 33, 34]:
                            return '-/-'
                        elif aid in [81, 82, 83, 84, 101, 102, 103, 104]:
                            return '+'
                        else:
                            return 'unknown'
                    except:
                        return 'unknown'
                df['genotype'] = df['animal_id'].apply(get_genotype_F)

            # Remove unknown genotypes
            df = df[df['genotype'] != 'unknown'].copy()

            # Sort by date
            df = df.sort_values(['animal_id', 'session_date'])

            # Assign session numbers
            df['session_num'] = df.groupby('animal_id').cumcount()

            # Group by genotype and session
            genotypes = sorted(df['genotype'].unique())
            colors = sns.color_palette("husl", len(genotypes))

            for g_idx, geno in enumerate(genotypes):
                geno_data = df[df['genotype'] == geno]

                # Compute session statistics
                session_stats = geno_data.groupby('session_num').agg({
                    'pct_correct': ['mean', 'sem'],
                    'animal_id': 'nunique'
                }).reset_index()

                x = session_stats['session_num']
                y = session_stats['pct_correct']['mean']
                sem = session_stats['pct_correct']['sem']
                n_animals = session_stats['animal_id']['nunique'].iloc[0]

                # Plot
                ax.plot(x, y, linewidth=3, color=colors[g_idx],
                       label=f'{geno} (n={n_animals})', alpha=0.9, marker='o', markersize=4)
                ax.fill_between(x, y - sem, y + sem, color=colors[g_idx], alpha=0.2)

            # Add reference lines
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.6,
                      linewidth=2, label='Chance')
            ax.axhline(y=0.8, color='green', linestyle=':', alpha=0.6,
                      linewidth=2, label='80% Criterion')

            ax.set_ylabel('Accuracy ± SEM', fontsize=14, fontweight='bold')
            ax.set_title(f'Cohort {cohort}: Learning Curves (End Summary Data)',
                        fontsize=15, fontweight='bold')
            ax.legend(fontsize=11, loc='lower right')
            ax.grid(alpha=0.3)
            ax.set_ylim(0.3, 1.0)

        axes[-1].set_xlabel('Session Number', fontsize=14, fontweight='bold')
        fig.suptitle('Learning Curves by Genotype\n(Using End Summary - % Correct)',
                    fontsize=18, fontweight='bold')
        plt.tight_layout()

        plt.savefig(self.output_dir / 'learning_curves_end_summary.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'learning_curves_end_summary.pdf',
                   bbox_inches='tight')
        plt.close()

        print(f"✓ Created learning curves from End Summary data")

    def identify_late_lapsers(self):
        """
        Identify animals that start well but deteriorate/stall during PI phase.
        """
        print("\n" + "="*80)
        print("IDENTIFYING LATE LAPSERS (Animals that stall in PI)")
        print("="*80)

        # Define animals
        animals_W = [f'c{c}m{m}' for c in range(1, 5) for m in range(1, 6)
                     if not (c == 1 and m == 5)]
        animals_F = [11, 12, 13, 14, 21, 22, 23, 24, 25,
                     31, 32, 33, 34, 41, 42, 51, 52,
                     61, 62, 63, 64, 71, 72, 73,
                     81, 82, 83, 84, 91, 92, 93,
                     101, 102, 103, 104]

        late_lapser_data = []

        for cohort, animals in [('W', animals_W), ('F', animals_F)]:
            for animal in animals:
                pkl_file = self.results_dir / f'{animal}_cohort{cohort}_model.pkl'

                if not pkl_file.exists():
                    continue

                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)

                # Load trial-level data from processed files
                if cohort == 'W':
                    trial_file = 'W LD Data 11.08 All_processed.csv'
                else:
                    trial_file = 'F LD Data 11.08 All_processed.csv'

                if not Path(trial_file).exists():
                    continue

                trial_df = pd.read_csv(trial_file)
                trial_df = trial_df[trial_df['Animal ID'] == animal]

                if len(trial_df) == 0:
                    continue

                # Extract trial-level performance
                trial_df = trial_df.sort_values('Schedule run date')
                n_trials = len(trial_df)

                # Compute rolling accuracy from End Summary
                if 'End Summary - Percentage Correct (1)' in trial_df.columns:
                    trial_df['acc'] = pd.to_numeric(trial_df['End Summary - Percentage Correct (1)'],
                                                     errors='coerce') / 100
                else:
                    continue

                # Early vs late performance
                early_cutoff = n_trials // 3
                late_cutoff = 2 * n_trials // 3

                early_trials = trial_df.iloc[:early_cutoff]
                late_trials = trial_df.iloc[late_cutoff:]

                early_acc = early_trials['acc'].mean()
                late_acc = late_trials['acc'].mean()
                acc_change = late_acc - early_acc

                # Check for PI-specific decline
                # Identify PI tasks by schedule name
                trial_df['is_pi'] = trial_df['Schedule name'].str.contains('Punish', case=False, na=False)
                pi_trials = trial_df[trial_df['is_pi']]

                if len(pi_trials) > 10:
                    n_pi = len(pi_trials)
                    pi_early = pi_trials.iloc[:n_pi//2]
                    pi_late = pi_trials.iloc[n_pi//2:]

                    pi_early_acc = pi_early['acc'].mean() if len(pi_early) > 0 else np.nan
                    pi_late_acc = pi_late['acc'].mean() if len(pi_late) > 0 else np.nan
                    pi_acc_change = pi_late_acc - pi_early_acc
                    has_pi = True
                else:
                    pi_early_acc = np.nan
                    pi_late_acc = np.nan
                    pi_acc_change = np.nan
                    has_pi = False

                # Check if late lapser
                is_late_lapser = (acc_change < -0.1) and (late_acc < 0.7)
                is_pi_lapser = has_pi and (pi_acc_change < -0.1) and (pi_late_acc < 0.7)

                # Get dominant state in late trials (from model)
                state_seq = data['model'].most_likely_states
                if len(state_seq) > late_cutoff:
                    late_states = state_seq[late_cutoff:]
                    dominant_late_state = int(np.bincount(late_states).argmax())
                else:
                    dominant_late_state = np.nan

                late_lapser_data.append({
                    'animal_id': animal,
                    'cohort': cohort,
                    'genotype': data['genotype'],
                    'n_trials': n_trials,
                    'early_accuracy': early_acc,
                    'late_accuracy': late_acc,
                    'accuracy_change': acc_change,
                    'pi_early_accuracy': pi_early_acc,
                    'pi_late_accuracy': pi_late_acc,
                    'pi_accuracy_change': pi_acc_change,
                    'has_pi': has_pi,
                    'is_late_lapser': is_late_lapser,
                    'is_pi_lapser': is_pi_lapser,
                    'dominant_late_state': dominant_late_state
                })

        # Create DataFrame
        df = pd.DataFrame(late_lapser_data)

        # Save
        df.to_csv(self.output_dir / 'late_lapser_identification.csv', index=False)

        # Report findings
        print(f"\nTotal animals analyzed: {len(df)}")
        print(f"Animals with late lapsing (overall): {df['is_late_lapser'].sum()}")
        print(f"Animals with PI-specific lapsing: {df['is_pi_lapser'].sum()}")

        print(f"\n{'-'*80}")
        print("LATE LAPSERS BY GENOTYPE:")
        print(f"{'-'*80}")

        for cohort in ['W', 'F']:
            cohort_data = df[df['cohort'] == cohort]
            print(f"\nCohort {cohort}:")

            genotype_summary = cohort_data.groupby('genotype').agg({
                'is_late_lapser': 'sum',
                'is_pi_lapser': 'sum',
                'animal_id': 'count',
                'accuracy_change': 'mean'
            }).round(3)

            genotype_summary.columns = ['N_Late_Lapsers', 'N_PI_Lapsers',
                                       'Total_Animals', 'Mean_Acc_Change']
            print(genotype_summary)

        # Visualize
        self._plot_late_lapsers(df)

        print(f"\n✓ Late lapser analysis complete")
        print(f"  Saved: {self.output_dir / 'late_lapser_identification.csv'}")

        return df

    def _plot_late_lapsers(self, df):
        """Create visualization of late lapsers."""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Accuracy change by genotype
        ax = axes[0, 0]
        for cohort, marker in [('W', 'o'), ('F', 's')]:
            cohort_data = df[df['cohort'] == cohort]
            genotypes = sorted(cohort_data['genotype'].unique())

            for geno in genotypes:
                geno_data = cohort_data[cohort_data['genotype'] == geno]
                x = [f'{cohort}-{geno}'] * len(geno_data)
                y = geno_data['accuracy_change'].values

                ax.scatter([f'{cohort}-{geno}'], [np.mean(y)],
                          s=200, marker=marker, alpha=0.7,
                          label=f'{cohort}-{geno}')
                ax.scatter(x, y, s=50, marker=marker, alpha=0.3, color='gray')

        ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax.axhline(y=-0.1, color='red', linestyle=':', linewidth=2,
                  label='Late lapser threshold')
        ax.set_ylabel('Accuracy Change (Late - Early)', fontsize=12, fontweight='bold')
        ax.set_title('Overall Accuracy Change by Genotype', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=0.3)

        # 2. Late lapser counts
        ax = axes[0, 1]
        genotype_labels = []
        late_lapser_counts = []

        for cohort in ['W', 'F']:
            cohort_data = df[df['cohort'] == cohort]
            for geno in sorted(cohort_data['genotype'].unique()):
                geno_data = cohort_data[cohort_data['genotype'] == geno]
                genotype_labels.append(f'{cohort}-{geno}')
                late_lapser_counts.append(geno_data['is_late_lapser'].sum())

        colors_bar = ['#e74c3c' if count > 0 else '#95a5a6'
                     for count in late_lapser_counts]
        ax.bar(range(len(genotype_labels)), late_lapser_counts,
              color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(genotype_labels)))
        ax.set_xticklabels(genotype_labels, rotation=45, ha='right')
        ax.set_ylabel('Number of Late Lapsers', fontsize=12, fontweight='bold')
        ax.set_title('Late Lapser Counts by Genotype', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # 3. PI-specific accuracy change
        ax = axes[1, 0]
        pi_animals = df[df['has_pi']]

        for cohort, marker in [('W', 'o'), ('F', 's')]:
            cohort_data = pi_animals[pi_animals['cohort'] == cohort]

            for geno in sorted(cohort_data['genotype'].unique()):
                geno_data = cohort_data[cohort_data['genotype'] == geno]
                if len(geno_data) > 0:
                    x = [f'{cohort}-{geno}'] * len(geno_data)
                    y = geno_data['pi_accuracy_change'].values

                    ax.scatter([f'{cohort}-{geno}'], [np.nanmean(y)],
                              s=200, marker=marker, alpha=0.7)
                    ax.scatter(x, y, s=50, marker=marker, alpha=0.3, color='gray')

        ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax.axhline(y=-0.1, color='red', linestyle=':', linewidth=2)
        ax.set_ylabel('PI Accuracy Change (Late - Early)', fontsize=12, fontweight='bold')
        ax.set_title('PI-Specific Accuracy Change', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=0.3)

        # 4. Early vs Late accuracy scatter
        ax = axes[1, 1]

        colors_scatter = {'W': '#3498db', 'F': '#e67e22'}
        markers_scatter = {'W': 'o', 'F': 's'}

        for cohort in ['W', 'F']:
            cohort_data = df[df['cohort'] == cohort]

            # Late lapsers
            lapsers = cohort_data[cohort_data['is_late_lapser']]
            ax.scatter(lapsers['early_accuracy'], lapsers['late_accuracy'],
                      s=150, marker=markers_scatter[cohort],
                      color=colors_scatter[cohort], edgecolor='red',
                      linewidth=3, alpha=0.8,
                      label=f'{cohort} Late Lapsers')

            # Non-lapsers
            non_lapsers = cohort_data[~cohort_data['is_late_lapser']]
            ax.scatter(non_lapsers['early_accuracy'], non_lapsers['late_accuracy'],
                      s=100, marker=markers_scatter[cohort],
                      color=colors_scatter[cohort], alpha=0.4,
                      label=f'{cohort} Normal')

        # Add diagonal line
        ax.plot([0.3, 1.0], [0.3, 1.0], 'k--', linewidth=2, alpha=0.5,
               label='No change')

        ax.set_xlabel('Early Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Late Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Early vs Late Performance', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0.3, 1.0)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'late_lapser_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'late_lapser_analysis.pdf',
                   bbox_inches='tight')
        plt.close()

    def analyze_poor_performer_impact(self, late_lapser_df):
        """
        Determine if poor performers are driving genotype differences.
        """
        print("\n" + "="*80)
        print("POOR PERFORMER IMPACT ANALYSIS")
        print("="*80)

        # Define poor performers as:
        # 1. Overall accuracy < 60%
        # 2. Late lapsers (accuracy drops >10%)

        animals_W = [f'c{c}m{m}' for c in range(1, 5) for m in range(1, 6)
                     if not (c == 1 and m == 5)]
        animals_F = [11, 12, 13, 14, 21, 22, 23, 24, 25,
                     31, 32, 33, 34, 41, 42, 51, 52,
                     61, 62, 63, 64, 71, 72, 73,
                     81, 82, 83, 84, 91, 92, 93,
                     101, 102, 103, 104]

        performance_data = []

        for cohort, animals in [('W', animals_W), ('F', animals_F)]:
            for animal in animals:
                pkl_file = self.results_dir / f'{animal}_cohort{cohort}_model.pkl'

                if not pkl_file.exists():
                    continue

                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)

                # Get overall metrics from state metrics
                # Weight by occupancy
                state_metrics = data['state_metrics']
                overall_acc = np.average(
                    state_metrics['accuracy'].values,
                    weights=state_metrics['occupancy'].values
                )

                # Check if late lapser
                late_lapser_row = late_lapser_df[
                    (late_lapser_df['animal_id'] == animal) &
                    (late_lapser_df['cohort'] == cohort)
                ]

                is_late_lapser = late_lapser_row['is_late_lapser'].iloc[0] if len(late_lapser_row) > 0 else False

                # Classify as poor performer
                is_poor_performer = (overall_acc < 0.6) or is_late_lapser

                performance_data.append({
                    'animal_id': animal,
                    'cohort': cohort,
                    'genotype': data['genotype'],
                    'overall_accuracy': overall_acc,
                    'is_late_lapser': is_late_lapser,
                    'is_poor_performer': is_poor_performer
                })

        perf_df = pd.DataFrame(performance_data)
        perf_df.to_csv(self.output_dir / 'poor_performer_classification.csv', index=False)

        # Analyze impact
        print(f"\nTotal animals: {len(perf_df)}")
        print(f"Poor performers: {perf_df['is_poor_performer'].sum()} ({perf_df['is_poor_performer'].sum()/len(perf_df)*100:.1f}%)")

        print(f"\n{'-'*80}")
        print("POOR PERFORMERS BY GENOTYPE:")
        print(f"{'-'*80}")

        for cohort in ['W', 'F']:
            cohort_data = perf_df[perf_df['cohort'] == cohort]
            print(f"\nCohort {cohort}:")

            geno_summary = cohort_data.groupby('genotype').agg({
                'is_poor_performer': ['sum', 'mean'],
                'animal_id': 'count',
                'overall_accuracy': 'mean'
            }).round(3)

            print(geno_summary)

        # Test if removing poor performers changes genotype differences
        print(f"\n{'-'*80}")
        print("GENOTYPE COMPARISONS: ALL vs GOOD PERFORMERS ONLY")
        print(f"{'-'*80}")

        for cohort in ['W', 'F']:
            cohort_data = perf_df[perf_df['cohort'] == cohort]

            print(f"\nCohort {cohort}:")
            print(f"{'Genotype':<10} {'All Animals':<20} {'Good Performers Only':<25}")
            print(f"{'-'*65}")

            for geno in sorted(cohort_data['genotype'].unique()):
                geno_all = cohort_data[cohort_data['genotype'] == geno]
                geno_good = geno_all[~geno_all['is_poor_performer']]

                acc_all = geno_all['overall_accuracy'].mean()
                sem_all = geno_all['overall_accuracy'].sem()
                n_all = len(geno_all)

                acc_good = geno_good['overall_accuracy'].mean() if len(geno_good) > 0 else np.nan
                sem_good = geno_good['overall_accuracy'].sem() if len(geno_good) > 0 else np.nan
                n_good = len(geno_good)

                print(f"{geno:<10} {acc_all:.3f}±{sem_all:.3f} (n={n_all:<2})    "
                     f"{acc_good:.3f}±{sem_good:.3f} (n={n_good:<2})")

        print(f"\n✓ Poor performer impact analysis complete")
        print(f"  Saved: {self.output_dir / 'poor_performer_classification.csv'}")

        # Create recommendation
        recommendation = """
RECOMMENDATIONS FOR HANDLING POOR PERFORMERS
=============================================

FINDINGS:
---------
See poor_performer_classification.csv for full details.

QUESTIONS TO ADDRESS:
---------------------
1. Are poor performers evenly distributed across genotypes?
   → If YES: Safe to include all animals
   → If NO: Poor performers may bias genotype comparisons

2. Do genotype differences persist after excluding poor performers?
   → If YES: Robust effect, not driven by poor performers
   → If NO: Effect may be driven by a few bad animals

3. What causes poor performance?
   - Data quality issues (missing latencies, recording errors)
   - True behavioral impairment
   - Motivational issues
   - Task difficulty (some animals never learn)

ANALYSIS OPTIONS:
-----------------

Option 1: INCLUDE ALL ANIMALS
  Pros:
  - Maximum sample size
  - No selection bias
  - Represents full genotype range

  Cons:
  - Poor performers may obscure real effects
  - Data quality issues may confound results

Option 2: EXCLUDE POOR PERFORMERS
  Pros:
  - Cleaner signal
  - Focus on animals that learned task
  - Reduces noise from data quality issues

  Cons:
  - Reduced sample size
  - Selection bias
  - May miss important phenotypes

Option 3: STRATIFIED ANALYSIS
  - Report results both ways (all animals + good performers only)
  - Test if genotype effects are consistent across both
  - Most transparent approach

  RECOMMENDED for your analyses

Option 4: CONTROL FOR PERFORMANCE
  - Include overall accuracy as covariate in statistical models
  - Tests if genotype effects persist after accounting for ability
  - Separates "learning impairment" from "strategy differences"

SPECIFIC RECOMMENDATIONS FOR YOUR DATA:
---------------------------------------

1. For GLM-HMM state analyses:
   → Keep all animals (states may capture poor performance)
   → But mark poor performers in visualizations
   → Test if state distributions differ for good vs poor performers

2. For genotype comparisons:
   → Use stratified analysis (Option 3)
   → Report both including and excluding poor performers
   → Check if conclusions change

3. For manuscript:
   → Clearly define "poor performer" criteria
   → Report n excluded and reasons
   → Show data with and without poor performers in supplement

4. For F-/- genotype specifically:
   → Many have data quality issues (NaN latencies)
   → Consider excluding animals with >50% missing data
   → May need manual review of raw data files

NEXT STEPS:
-----------
1. Review poor_performer_classification.csv
2. Decide on exclusion criteria (if any)
3. Re-run key analyses with/without poor performers
4. Compare results to assess robustness
5. Document decisions and rationale for manuscript
"""

        with open(self.output_dir / 'poor_performer_recommendations.txt', 'w') as f:
            f.write(recommendation)

        print(f"\n✓ Created recommendations: {self.output_dir / 'poor_performer_recommendations.txt'}")


def main():
    """Run complete late lapser and poor performer analysis."""

    print("="*80)
    print("LATE LAPSER AND POOR PERFORMER ANALYSIS")
    print("="*80)

    analyzer = LateLapserAnalysis()

    # 1. Explain latency calculation
    print("\n[1/5] Explaining latency calculation...")
    analyzer.explain_latency_calculation()

    # 2. Review F-/- animals
    print("\n[2/5] Reviewing F-/- animals...")
    analyzer.review_f_minus_minus_animals()

    # 3. Create End Summary learning curves
    print("\n[3/5] Creating End Summary learning curves...")
    analyzer.create_end_summary_learning_curves()

    # 4. Identify late lapsers
    print("\n[4/5] Identifying late lapsers...")
    late_lapser_df = analyzer.identify_late_lapsers()

    # 5. Analyze poor performer impact
    print("\n[5/5] Analyzing poor performer impact...")
    analyzer.analyze_poor_performer_impact(late_lapser_df)

    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {analyzer.output_dir}")
    print("\nKey outputs:")
    print("  1. latency_calculation_explained.txt - How latency is calculated")
    print("  2. F_minus_minus_animal_review.csv - Detailed F-/- metrics")
    print("  3. learning_curves_end_summary.png/pdf - Learning curves from End Summary")
    print("  4. late_lapser_identification.csv - Animals that deteriorate")
    print("  5. late_lapser_analysis.png/pdf - Visualization of lapsers")
    print("  6. poor_performer_classification.csv - Performance classification")
    print("  7. poor_performer_recommendations.txt - Analysis recommendations")


if __name__ == '__main__':
    main()
