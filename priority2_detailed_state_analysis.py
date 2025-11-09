"""
Priority 2: Detailed State Analysis
====================================

Creates comprehensive visualizations and explanations for the 5 detailed
behavioral states and how they map to the 3 broad categories:

BROAD CATEGORIES (3):
--------------------
1. Engaged: High-performance, task-focused behavior
2. Lapsed: Disengaged, inattentive behavior
3. Mixed: Intermediate or strategic behavior

DETAILED STATES (5):
-------------------
1. Deliberative High-Performance (HP):
   - Accuracy > 65% (high performance)
   - Moderate-to-high latency CV (thoughtful, variable responding)
   - Strong WSLS ratio (cognitive strategy use)
   - Maps to: ENGAGED

2. Procedural High-Performance (HP):
   - Accuracy > 65% (high performance)
   - LOW latency CV < 0.65 (habitual, consistent responding)
   - May have moderate WSLS
   - Maps to: ENGAGED

3. Disengaged Lapse:
   - Accuracy < 65% (poor performance)
   - High latency CV (erratic responding)
   - Low WSLS ratio
   - Maps to: LAPSED

4. WSLS Strategy:
   - Variable accuracy
   - Strong WSLS ratio (> 0.6)
   - Moderate latency
   - Maps to: MIXED (strategic but not consistently high-performing)

5. Perseverative Bias:
   - High side bias (> 0.5)
   - Low accuracy
   - Low WSLS
   - Maps to: LAPSED or MIXED (depending on severity)

KEY DISTINCTION: Deliberative vs Procedural HP
----------------------------------------------
Both achieve high accuracy (>65%), but differ in HOW:

DELIBERATIVE HP:
- Thoughtful, effortful responding
- Variable response times (moderate-to-high CV)
- Strong win-stay/lose-shift strategy (cognitive flexibility)
- Actively tracking contingencies
- "Thinking through" each trial

PROCEDURAL HP:
- Automatic, habitual responding
- Consistent response times (low CV < 0.65)
- Less reliance on trial-by-trial strategy
- Well-learned stimulus-response associations
- "On autopilot" but accurate

This distinction is critical for understanding learning:
- Early training: More Deliberative HP (effortful learning)
- Late training: More Procedural HP (automatized performance)
- Genotype differences may affect transition from Deliberative → Procedural
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
sns.set_palette("Set2")


class DetailedStateAnalyzer:
    """Analyze and visualize detailed state patterns by genotype."""

    def __init__(self, results_dir='results/phase1_non_reversal'):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'priority2_detailed_states'
        self.output_dir.mkdir(exist_ok=True)

    def load_all_results(self, cohort, animals):
        """Load all model results for analysis."""
        results = []

        for animal in animals:
            pkl_file = self.results_dir / f'{animal}_cohort{cohort}_model.pkl'
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    # Add broad categories
                    broad_categories = create_broad_state_categories(data['validated_labels'])
                    data['broad_categories'] = broad_categories
                    results.append(data)

        return results

    def extract_detailed_state_info(self, results):
        """
        Extract detailed state information for each animal.

        Returns DataFrame with columns:
        - animal_id, cohort, genotype
        - state (0, 1, 2)
        - broad_category (Engaged, Lapsed, Mixed)
        - detailed_label (Deliberative HP, Procedural HP, etc.)
        - accuracy, wsls_ratio, latency_cv, side_bias
        - n_trials (number of trials in this state)
        - pct_trials (percentage of trials in this state)
        """
        data_rows = []

        for r in results:
            animal_id = r['animal_id']
            cohort = r['cohort']
            genotype = r['genotype']
            n_states = r['model'].n_states
            state_seq = r['model'].most_likely_states
            total_trials = len(state_seq)

            broad_categories = r['broad_categories']

            # Get state metrics from stored DataFrame
            state_metrics_df = r['state_metrics']

            for state in range(n_states):
                # Get state labels
                broad_cat, detailed_label, confidence = broad_categories[state]

                # Get metrics for this state
                state_row = state_metrics_df[state_metrics_df['state'] == state]

                if len(state_row) > 0:
                    accuracy = state_row['accuracy'].values[0]
                    wsls_ratio = state_row['wsls_ratio'].values[0]
                    latency_cv = state_row['latency_cv'].values[0]
                    side_bias = state_row['side_bias'].values[0]
                    n_trials = int(state_row['n_trials'].values[0])
                else:
                    # No data for this state
                    accuracy = 0
                    wsls_ratio = 0
                    latency_cv = 0
                    side_bias = 0
                    n_trials = 0

                # Calculate percentage
                pct_trials = 100 * n_trials / total_trials if total_trials > 0 else 0

                data_rows.append({
                    'animal_id': animal_id,
                    'cohort': cohort,
                    'genotype': genotype,
                    'state': state,
                    'broad_category': broad_cat,
                    'detailed_label': detailed_label,
                    'accuracy': accuracy,
                    'wsls_ratio': wsls_ratio,
                    'latency_cv': latency_cv,
                    'side_bias': side_bias,
                    'n_trials': n_trials,
                    'pct_trials': pct_trials,
                    'confidence': confidence
                })

        return pd.DataFrame(data_rows)

    def plot_detailed_state_distribution(self, df):
        """
        Create bar plots showing distribution of detailed states by genotype.
        """
        # Combine cohort and genotype
        df['genotype_label'] = df['cohort'] + '-' + df['genotype']

        # Get unique genotypes and detailed labels
        genotypes = sorted(df['genotype_label'].unique())
        detailed_labels = ['Deliberative HP', 'Procedural HP', 'Disengaged',
                          'WSLS', 'Perseverative']

        # Count animals in each detailed state by genotype
        state_counts = []
        for geno in genotypes:
            geno_data = df[df['genotype_label'] == geno]
            counts = {}
            for label in detailed_labels:
                # Count unique animals with at least one state of this type
                n_animals = geno_data[geno_data['detailed_label'] == label]['animal_id'].nunique()
                counts[label] = n_animals
            state_counts.append(counts)

        # Create DataFrame
        count_df = pd.DataFrame(state_counts, index=genotypes)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Define colors for each detailed state
        colors = {
            'Deliberative HP': '#27ae60',  # Green
            'Procedural HP': '#16a085',    # Teal
            'Disengaged': '#e74c3c',       # Red
            'WSLS': '#f39c12',             # Orange
            'Perseverative': '#9b59b6'     # Purple
        }

        x = np.arange(len(genotypes))
        width = 0.15

        for i, label in enumerate(detailed_labels):
            if label in count_df.columns:
                values = count_df[label].values
                ax.bar(x + i*width, values, width, label=label,
                      color=colors.get(label, '#95a5a6'), alpha=0.8)

        ax.set_xlabel('Genotype', fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Animals', fontsize=13, fontweight='bold')
        ax.set_title('Distribution of Detailed Behavioral States by Genotype',
                    fontsize=15, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(genotypes, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=11, title='Detailed State')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'detailed_state_counts_by_genotype.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'detailed_state_counts_by_genotype.pdf',
                   bbox_inches='tight')
        plt.close()

        print(f"✓ Created detailed state count plot")

        # Also save the counts table
        count_df.to_csv(self.output_dir / 'detailed_state_counts.csv')
        print(f"  Saved counts table: {self.output_dir / 'detailed_state_counts.csv'}")

    def plot_deliberative_vs_procedural(self, df):
        """
        Create focused comparison of Deliberative vs Procedural HP states.
        """
        # Filter to only HP states (flexible matching for full names)
        hp_data = df[df['detailed_label'].str.contains('Deliberative|Procedural', case=False, na=False)].copy()

        if len(hp_data) == 0:
            print("  No HP states found for comparison")
            return

        # Simplify labels for grouping
        hp_data['hp_type'] = hp_data['detailed_label'].apply(
            lambda x: 'Deliberative HP' if 'Deliberative' in x else 'Procedural HP'
        )

        hp_data['genotype_label'] = hp_data['cohort'] + '-' + hp_data['genotype']
        genotypes = sorted(hp_data['genotype_label'].unique())

        # Create 2x2 subplot comparing key metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        metrics = [
            ('latency_cv', 'Latency CV', 'Response Consistency'),
            ('wsls_ratio', 'WSLS Ratio', 'Strategy Use'),
            ('accuracy', 'Accuracy', 'Performance Level'),
            ('pct_trials', '% Trials', 'State Occupancy')
        ]

        for idx, (metric, ylabel, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            # Prepare data for plotting
            delib_data = []
            proc_data = []
            labels = []

            for geno in genotypes:
                geno_hp = hp_data[hp_data['genotype_label'] == geno]

                delib = geno_hp[geno_hp['hp_type'] == 'Deliberative HP'][metric].values
                proc = geno_hp[geno_hp['hp_type'] == 'Procedural HP'][metric].values

                if len(delib) > 0 or len(proc) > 0:
                    delib_data.append(delib if len(delib) > 0 else [np.nan])
                    proc_data.append(proc if len(proc) > 0 else [np.nan])
                    labels.append(geno)

            # Create grouped box plots
            x = np.arange(len(labels))
            width = 0.35

            # Compute means and SEMs
            delib_means = [np.nanmean(d) for d in delib_data]
            delib_sems = []
            for d in delib_data:
                d_arr = np.array(d)
                valid = d_arr[~np.isnan(d_arr)]
                if len(valid) > 0:
                    delib_sems.append(np.nanstd(d_arr) / np.sqrt(len(valid)))
                else:
                    delib_sems.append(0)

            proc_means = [np.nanmean(p) for p in proc_data]
            proc_sems = []
            for p in proc_data:
                p_arr = np.array(p)
                valid = p_arr[~np.isnan(p_arr)]
                if len(valid) > 0:
                    proc_sems.append(np.nanstd(p_arr) / np.sqrt(len(valid)))
                else:
                    proc_sems.append(0)

            # Plot bars
            ax.bar(x - width/2, delib_means, width, yerr=delib_sems,
                  label='Deliberative HP', color='#27ae60', alpha=0.8,
                  capsize=5, error_kw={'linewidth': 2})
            ax.bar(x + width/2, proc_means, width, yerr=proc_sems,
                  label='Procedural HP', color='#16a085', alpha=0.8,
                  capsize=5, error_kw={'linewidth': 2})

            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)

            # Add horizontal line for CV threshold on latency_cv plot
            if metric == 'latency_cv':
                ax.axhline(y=0.65, color='red', linestyle='--', linewidth=2,
                          alpha=0.7, label='Procedural threshold')
                ax.legend(fontsize=10)

        fig.suptitle('Deliberative vs Procedural High-Performance States:\nKey Metric Comparison by Genotype',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        plt.savefig(self.output_dir / 'deliberative_vs_procedural_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'deliberative_vs_procedural_comparison.pdf',
                   bbox_inches='tight')
        plt.close()

        print(f"✓ Created Deliberative vs Procedural comparison plot")

    def plot_state_mapping_diagram(self):
        """
        Create visual diagram showing how 5 detailed states map to 3 broad categories.
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')

        # Define colors
        color_engaged = '#2ecc71'
        color_lapsed = '#e74c3c'
        color_mixed = '#f39c12'

        # Helper function for boxes
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

        def add_box(x, y, w, h, text, color, fontsize=11):
            box = FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.15",
                                edgecolor='black',
                                facecolor=color,
                                alpha=0.7,
                                linewidth=2.5)
            ax.add_patch(box)
            ax.text(x + w/2, y + h/2, text,
                   ha='center', va='center',
                   fontsize=fontsize, fontweight='bold',
                   wrap=True)

        def add_arrow(x1, y1, x2, y2):
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                   arrowstyle='->',
                                   mutation_scale=30,
                                   linewidth=3,
                                   color='black',
                                   alpha=0.6)
            ax.add_patch(arrow)

        # Title
        ax.text(5, 11.2, 'STATE CATEGORIZATION HIERARCHY',
               ha='center', fontsize=18, fontweight='bold')

        # BROAD CATEGORIES (Top level)
        ax.text(5, 10.2, 'BROAD CATEGORIES (3)',
               ha='center', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        add_box(0.5, 8.5, 2.5, 1.2, 'ENGAGED\nHigh Performance\nTask-Focused',
               color_engaged, fontsize=12)
        add_box(3.7, 8.5, 2.5, 1.2, 'MIXED\nStrategic/Variable\nIntermediate',
               color_mixed, fontsize=12)
        add_box(7, 8.5, 2.5, 1.2, 'LAPSED\nPoor Performance\nDisengaged',
               color_lapsed, fontsize=12)

        # DETAILED STATES (Bottom level)
        ax.text(5, 7.2, 'DETAILED STATES (5)',
               ha='center', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        # Deliberative HP
        add_box(0.2, 5, 1.8, 1.8,
               'Deliberative\nHP\n\n• Acc > 65%\n• Variable RT\n• Strong WSLS',
               '#27ae60', fontsize=9)
        add_arrow(1.1, 8.5, 1.1, 6.8)

        # Procedural HP
        add_box(2.3, 5, 1.8, 1.8,
               'Procedural\nHP\n\n• Acc > 65%\n• CV < 0.65\n• Habitual',
               '#16a085', fontsize=9)
        add_arrow(2.0, 8.5, 3.2, 6.8)

        # WSLS Strategy
        add_box(4.4, 5, 1.8, 1.8,
               'WSLS\nStrategy\n\n• Strong WSLS\n• Variable Acc\n• Moderate RT',
               '#f39c12', fontsize=9)
        add_arrow(4.95, 8.5, 5.3, 6.8)

        # Disengaged
        add_box(6.5, 5, 1.8, 1.8,
               'Disengaged\nLapse\n\n• Acc < 65%\n• High CV\n• Low WSLS',
               '#c0392b', fontsize=9)
        add_arrow(8.25, 8.5, 7.4, 6.8)

        # Perseverative
        add_box(8.6, 5, 1.8, 1.8,
               'Perseverative\nBias\n\n• High Side Bias\n• Low Acc\n• Repetitive',
               '#8e44ad', fontsize=9)
        add_arrow(8.7, 8.5, 9.5, 6.8)

        # Add KEY DISTINCTION box
        add_box(0.5, 2.5, 4, 2,
               'KEY DISTINCTION: Deliberative vs Procedural HP\n\n' +
               'Both achieve high accuracy, but differ in HOW:\n\n' +
               'DELIBERATIVE: Thoughtful, effortful, variable RT\n' +
               'PROCEDURAL: Automatic, habitual, consistent RT\n\n' +
               'Early learning → Deliberative | Late learning → Procedural',
               '#ecf0f1', fontsize=9)

        # Add validation criteria
        add_box(5.5, 2.5, 4, 2,
               'VALIDATION CRITERIA\n\n' +
               '• Accuracy: Choice correctness\n' +
               '• Latency CV: Response consistency\n' +
               '• WSLS Ratio: Strategy use\n' +
               '• Side Bias: Perseveration\n\n' +
               'Automated classification with\nconfidence scoring',
               '#ecf0f1', fontsize=9)

        # Add legend for colors
        ax.text(1, 0.8, '■ Engaged States', fontsize=11, color='#27ae60', fontweight='bold')
        ax.text(3.5, 0.8, '■ Mixed States', fontsize=11, color='#f39c12', fontweight='bold')
        ax.text(6, 0.8, '■ Lapsed States', fontsize=11, color='#e74c3c', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'state_categorization_hierarchy.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'state_categorization_hierarchy.pdf',
                   bbox_inches='tight')
        plt.close()

        print(f"✓ Created state categorization hierarchy diagram")

    def create_summary_statistics(self, df):
        """
        Create summary statistics table for detailed states.
        """
        df['genotype_label'] = df['cohort'] + '-' + df['genotype']

        # Overall statistics
        print("\n" + "="*80)
        print("DETAILED STATE SUMMARY STATISTICS")
        print("="*80)

        # Count by detailed label
        print("\nOverall state distribution:")
        print(df.groupby('detailed_label').agg({
            'animal_id': 'nunique',
            'n_trials': 'sum',
            'pct_trials': 'mean'
        }).round(2))

        # Count by genotype
        print("\nState distribution by genotype:")
        geno_summary = df.groupby(['genotype_label', 'detailed_label']).agg({
            'animal_id': 'nunique',
            'n_trials': 'sum'
        }).reset_index()

        pivot_table = geno_summary.pivot_table(
            index='genotype_label',
            columns='detailed_label',
            values='animal_id',
            fill_value=0
        )
        print(pivot_table)

        # Save to CSV
        pivot_table.to_csv(self.output_dir / 'state_counts_by_genotype.csv')
        print(f"\n✓ Saved summary table: {self.output_dir / 'state_counts_by_genotype.csv'}")

        # Metrics by detailed state
        print("\nAverage metrics by detailed state:")
        metrics_summary = df.groupby('detailed_label').agg({
            'accuracy': ['mean', 'std'],
            'wsls_ratio': ['mean', 'std'],
            'latency_cv': ['mean', 'std'],
            'side_bias': ['mean', 'std']
        }).round(3)
        print(metrics_summary)

        metrics_summary.to_csv(self.output_dir / 'metrics_by_detailed_state.csv')
        print(f"✓ Saved metrics table: {self.output_dir / 'metrics_by_detailed_state.csv'}")


def main():
    """Run detailed state analysis."""
    print("="*80)
    print("PRIORITY 2: DETAILED STATE ANALYSIS")
    print("="*80)

    analyzer = DetailedStateAnalyzer()

    # Define animals
    animals_W = [f'c{c}m{m}' for c in range(1, 5) for m in range(1, 6)
                 if not (c == 1 and m == 5)]
    animals_F = [11, 12, 13, 14, 21, 22, 23, 24, 25,
                 31, 32, 33, 34, 41, 42, 51, 52,
                 61, 62, 63, 64, 71, 72, 73,
                 81, 82, 83, 84, 91, 92, 93,
                 101, 102, 103, 104]

    # Load results
    print("\nLoading model results...")
    results_W = analyzer.load_all_results('W', animals_W)
    results_F = analyzer.load_all_results('F', animals_F)
    print(f"  Cohort W: {len(results_W)} animals")
    print(f"  Cohort F: {len(results_F)} animals")

    # Extract detailed state information
    print("\nExtracting detailed state information...")
    df_W = analyzer.extract_detailed_state_info(results_W)
    df_F = analyzer.extract_detailed_state_info(results_F)
    df_all = pd.concat([df_W, df_F], ignore_index=True)
    print(f"  Total states analyzed: {len(df_all)}")

    # Create visualizations
    print("\nCreating visualizations...")

    print("\n[1/4] State categorization hierarchy...")
    analyzer.plot_state_mapping_diagram()

    print("\n[2/4] Detailed state counts by genotype...")
    analyzer.plot_detailed_state_distribution(df_all)

    print("\n[3/4] Deliberative vs Procedural comparison...")
    analyzer.plot_deliberative_vs_procedural(df_all)

    print("\n[4/4] Summary statistics...")
    analyzer.create_summary_statistics(df_all)

    print("\n" + "="*80)
    print("✓ DETAILED STATE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {analyzer.output_dir}")


if __name__ == '__main__':
    main()
