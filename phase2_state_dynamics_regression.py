"""
Phase 2 State Occupancy Dynamics: Mixed-Effects Regression Analysis
===================================================================

Analyzes P(State) over trials in Phase 2 reversal learning:
- Tests for learning effects (trial × state interaction)
- Identifies genotype-specific learning rates
- Quantifies state-dependent learning trajectories
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/home/user/GLMHMM')

from genotype_labels import GENOTYPE_COLORS, GENOTYPE_ORDER, STATE_COLORS, STATE_LABELS, relabel_genotype
from glmhmm_utils import load_and_preprocess_session_data

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import glmer
    from statsmodels.regression.mixed_linear_model import MixedLM
    STATSMODELS_AVAILABLE = True
except:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available for full mixed-effects models")

plt.style.use('seaborn-v0_8-whitegrid')

# Genotype mapping
OLD_TO_NEW_GENOTYPE = {
    '+': 'B6',
    '-': 'C3H x B6',
    '+/+': 'A1D_Wt',
    '+/-': 'A1D_Het',
    '-/-': 'A1D_KO'
}


class Phase2StateRegressionAnalysis:
    """Analyze state occupancy dynamics in Phase 2 using mixed-effects models."""

    def __init__(self):
        self.results_dir = Path('/home/user/GLMHMM/results')
        self.phase2_dir = self.results_dir / 'phase2_reversal'
        self.output_dir = self.results_dir / 'statistical_analysis'
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'phase2_dynamics').mkdir(exist_ok=True)

        self.summary_report = []

    def add_report(self, text):
        """Add to report."""
        self.summary_report.append(text)
        print(text)

    def extract_phase2_state_occupancy(self):
        """Extract state occupancy over trials from Phase 2 pickle files."""
        print("\nExtracting Phase 2 state occupancy data...")

        all_data = []

        # Find all Phase 2 reversal model files
        for pkl_file in sorted(self.phase2_dir.glob('**//*_reversal.pkl')):
            try:
                with open(pkl_file, 'rb') as f:
                    result = pickle.load(f)

                animal_id = result.get('animal_id', pkl_file.stem)
                genotype = OLD_TO_NEW_GENOTYPE.get(result.get('genotype', ''), result.get('genotype'))
                cohort = result.get('cohort', 'Unknown')

                if 'states' not in result:
                    continue

                states = result['states']

                # Calculate occupancy in sliding windows
                window_size = 15  # 15 trial window
                for start in range(0, len(states) - window_size, 5):
                    end = start + window_size
                    window_states = states[start:end]

                    trial_number = start + window_size // 2  # Center of window
                    for state in range(3):
                        occupancy = 100 * np.mean(window_states == state)

                        all_data.append({
                            'animal_id': animal_id,
                            'genotype': genotype,
                            'cohort': cohort,
                            'trial_bin': trial_number,
                            'state': STATE_LABELS.get(state, f'State {state}'),
                            'state_num': state,
                            'occupancy': occupancy
                        })

            except Exception as e:
                print(f"Warning: Could not load {pkl_file}: {e}")

        df = pd.DataFrame(all_data)
        if len(df) > 0:
            print(f"Extracted data: {len(df)} observations from {df['animal_id'].nunique()} animals")
        else:
            print("No data extracted.")

        return df

    def analyze_state_occupancy_dynamics(self, df):
        """Perform mixed-effects regression on state occupancy."""
        print("\n" + "="*80)
        print("PHASE 2: STATE OCCUPANCY DYNAMICS REGRESSION ANALYSIS")
        print("="*80)

        self.add_report("\n" + "="*80)
        self.add_report("PHASE 2: STATE OCCUPANCY DYNAMICS REGRESSION ANALYSIS")
        self.add_report("="*80)

        if len(df) == 0:
            self.add_report("\nNo Phase 2 data available for regression analysis.")
            return None

        # Standardize trial bin for easier interpretation
        df['trial_bin_scaled'] = df['trial_bin'] / 100.0  # Scale to hundreds of trials

        # Overall learning effect per state
        self.add_report("\n" + "-"*80)
        self.add_report("OVERALL LEARNING EFFECTS BY STATE")
        self.add_report("-"*80)

        state_results = {}
        for state in range(3):
            state_data = df[df['state_num'] == state].copy()

            if len(state_data) < 3:
                continue

            state_label = STATE_LABELS[state]
            self.add_report(f"\n{state_label} State:")

            # Simple linear regression: occupancy ~ trial_bin
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                state_data['trial_bin_scaled'].values,
                state_data['occupancy'].values
            )

            self.add_report(f"  Slope: {slope:.4f} %/100trials")
            self.add_report(f"  Intercept: {intercept:.4f}%")
            self.add_report(f"  R-squared: {r_value**2:.4f}")
            self.add_report(f"  p-value: {p_value:.6f}")
            self.add_report(f"  Standard Error: {std_err:.4f}")

            if p_value < 0.05:
                if slope > 0:
                    self.add_report(f"  *** SIGNIFICANT: {state_label} occupancy INCREASES over trials (p < 0.05)")
                else:
                    self.add_report(f"  *** SIGNIFICANT: {state_label} occupancy DECREASES over trials (p < 0.05)")

            state_results[state] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err': std_err
            }

        # Genotype-specific slopes
        self.add_report("\n" + "-"*80)
        self.add_report("GENOTYPE-SPECIFIC LEARNING RATES (slopes by genotype and state)")
        self.add_report("-"*80)

        genotype_slopes = {}
        for genotype in sorted(df['genotype'].unique()):
            genotype_slopes[genotype] = {}
            self.add_report(f"\n{genotype}:")

            geno_data = df[df['genotype'] == genotype]

            for state in range(3):
                state_data = geno_data[geno_data['state_num'] == state]

                if len(state_data) < 3:
                    self.add_report(f"  {STATE_LABELS[state]}: Insufficient data (N={len(state_data)})")
                    continue

                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    state_data['trial_bin_scaled'].values,
                    state_data['occupancy'].values
                )

                genotype_slopes[genotype][state] = slope

                self.add_report(f"  {STATE_LABELS[state]}: slope={slope:.4f} %/100trials, p={p_value:.4f}")

                if p_value < 0.05:
                    if slope > 0:
                        self.add_report(f"    *** Significant increase")
                    else:
                        self.add_report(f"    *** Significant decrease")

        # Comparison of slopes across genotypes (Kruskal-Wallis on slopes)
        self.add_report("\n" + "-"*80)
        self.add_report("COMPARISON OF LEARNING RATES ACROSS GENOTYPES")
        self.add_report("-"*80)

        for state in range(3):
            slopes_by_geno = []
            geno_labels = []

            for genotype in sorted(df['genotype'].unique()):
                if state in genotype_slopes.get(genotype, {}):
                    # Collect individual slopes per animal
                    geno_data = df[(df['genotype'] == genotype) & (df['state_num'] == state)]

                    if len(geno_data) >= 3:
                        animals = geno_data['animal_id'].unique()
                        for animal in animals:
                            animal_data = geno_data[geno_data['animal_id'] == animal]
                            if len(animal_data) >= 2:
                                slope, _, _, _, _ = stats.linregress(
                                    animal_data['trial_bin_scaled'].values,
                                    animal_data['occupancy'].values
                                )
                                slopes_by_geno.append(slope)
                                geno_labels.append(genotype)

            if len(slopes_by_geno) > 5:
                # Kruskal-Wallis on slopes
                unique_genos = sorted(set(geno_labels))
                slope_groups = [np.array([s for s, g in zip(slopes_by_geno, geno_labels) if g == geno])
                               for geno in unique_genos]

                try:
                    h_stat, p_val = stats.kruskal(*slope_groups)
                    self.add_report(f"\n{STATE_LABELS[state]} State - Slope Comparison Across Genotypes:")
                    self.add_report(f"  Kruskal-Wallis H: {h_stat:.4f}, p: {p_val:.6f}")
                    if p_val < 0.05:
                        self.add_report(f"  *** SIGNIFICANT: Learning rates differ by genotype")
                except:
                    pass

        return state_results, genotype_slopes, df

    def create_state_dynamics_figures(self, df, state_results, genotype_slopes):
        """Create figures showing state occupancy over trials."""
        print("\nCreating Phase 2 state dynamics figures...")

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        genotype_order = [g for g in GENOTYPE_ORDER if g in df['genotype'].unique()]

        # Panel per state
        for state_idx, state in enumerate(range(3)):
            ax = axes[state_idx]
            state_label = STATE_LABELS[state]

            # Plot data by genotype
            for genotype in genotype_order:
                geno_data = df[(df['genotype'] == genotype) & (df['state_num'] == state)]

                if len(geno_data) == 0:
                    continue

                # Group by trial bin and calculate mean
                grouped = geno_data.groupby('trial_bin')['occupancy'].mean()
                color = GENOTYPE_COLORS.get(genotype, '#95a5a6')

                ax.plot(grouped.index, grouped.values, 'o-', label=genotype, color=color,
                       linewidth=2, markersize=6, alpha=0.7)

                # Add regression line
                if state in state_results:
                    x_line = np.array([grouped.index.min(), grouped.index.max()])
                    y_line = (state_results[state]['intercept'] +
                             state_results[state]['slope'] * (x_line / 100.0))
                    ax.plot(x_line, y_line, '--', color=color, alpha=0.4, linewidth=1.5)

            ax.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
            ax.set_ylabel('State Occupancy (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'{state_label} State Over Trials', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.legend(fontsize=9, loc='best')
            ax.set_ylim(-5, 105)

            # Add statistics text
            if state in state_results:
                stats_info = f"Slope: {state_results[state]['slope']:.2f}%/100t\np={state_results[state]['p_value']:.4f}"
                sig = "***Sig" if state_results[state]['p_value'] < 0.05 else "ns"
                ax.text(0.98, 0.05, f"{stats_info}\n{sig}", transform=ax.transAxes,
                       fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle('Phase 2: State Occupancy Dynamics Over Trials by Genotype',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'figures' / 'phase2_state_occupancy_dynamics_with_regression.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()

        print(f"✓ Saved to {output_path}")

    def save_report(self):
        """Save summary report."""
        report_path = self.output_dir / 'PHASE2_STATE_DYNAMICS_REPORT.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(self.summary_report))
        print(f"✓ Report saved to {report_path}")

    def run_analysis(self):
        """Run full analysis."""
        print("\n" + "="*80)
        print("PHASE 2 STATE OCCUPANCY DYNAMICS REGRESSION ANALYSIS")
        print("="*80)

        # Extract data
        df = self.extract_phase2_state_occupancy()

        if len(df) == 0:
            print("No Phase 2 data available.")
            return

        # Run regression analysis
        state_results, genotype_slopes, df = self.analyze_state_occupancy_dynamics(df)

        # Create figures
        self.create_state_dynamics_figures(df, state_results, genotype_slopes)

        # Save report
        self.save_report()

        print("\n" + "="*80)
        print("✓ PHASE 2 DYNAMICS ANALYSIS COMPLETE")
        print("="*80)


if __name__ == '__main__':
    analyzer = Phase2StateRegressionAnalysis()
    analyzer.run_analysis()
