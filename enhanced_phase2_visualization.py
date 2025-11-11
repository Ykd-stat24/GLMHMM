"""
Enhanced Phase 2 State Occupancy Dynamics Visualization

This script creates a comprehensive multi-panel figure showing:
1. Individual animal trajectories with genotype-specific regression lines
2. Confidence intervals around slope estimates
3. Statistical annotations (p-values, slope ± CI)
4. Learning rate comparison across genotypes and states
5. Effect size visualization
"""

import os
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# Setup
PHASE2_DIR = '/home/user/GLMHMM/results/phase2_reversal'
OUTPUT_DIR = '/home/user/GLMHMM/results/statistical_analysis/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Genotype mapping
OLD_TO_NEW = {
    '+/+': 'A1D_Wt', '-/-': 'A1D_KO', '+/-': 'A1D_Het',
    '+': 'B6', '-': 'C3H x B6'
}

# Colors by genotype
GENO_COLORS = {
    'A1D_Wt': '#FF6B6B',    # Red
    'A1D_KO': '#4ECDC4',    # Teal
    'A1D_Het': '#95E1D3',   # Light teal
    'B6': '#2E86AB',        # Blue
    'C3H x B6': '#A23B72'   # Purple
}

STATE_COLORS = {
    'Engaged': '#1f77b4',   # Blue
    'Biased': '#ff7f0e',    # Orange
    'Lapsed': '#d62728'     # Red
}

STATE_NAMES = {0: 'Engaged', 1: 'Biased', 2: 'Lapsed'}

# =============================================================================
# EXTRACT PHASE 2 STATE OCCUPANCY
# =============================================================================

class Phase2StateAnalyzer:
    """Extract and analyze state occupancy trajectories from Phase 2 data"""

    def __init__(self):
        self.all_data = []
        self.genotype_slopes = {}
        self.confidence_intervals = {}

    def extract_phase2_states(self):
        """Extract state sequences and occupancy from Phase 2 pickle files"""
        phase2_files = sorted(glob.glob(os.path.join(PHASE2_DIR, 'models', '*_reversal.pkl')))

        print(f"Found {len(phase2_files)} Phase 2 reversal models")

        for pkl_file in phase2_files:
            try:
                animal_id = os.path.basename(pkl_file).replace('_reversal.pkl', '')

                with open(pkl_file, 'rb') as f:
                    model_dict = pickle.load(f)

                # Extract trajectory if available
                if 'trajectory_df' not in model_dict:
                    print(f"Skipping {animal_id}: no trajectory_df")
                    continue

                traj_df = model_dict['trajectory_df']
                if 'state' not in traj_df.columns:
                    continue

                # Get genotype
                genotype = model_dict.get('genotype', '')
                genotype_new = OLD_TO_NEW.get(genotype, genotype)

                # Extract state sequence
                states = traj_df['state'].values
                bout_lengths = traj_df['bout_length'].values

                # Build trial-by-trial state occupancy using sliding windows
                window_size = 15
                window_stride = 5

                trial_idx = 0
                for bout_state, bout_len in zip(states, bout_lengths):
                    for _ in range(int(bout_len)):
                        trial_idx += 1

                # Recompute with sliding windows for occupancy
                window_centers = []
                occupancy_engaged = []
                occupancy_biased = []
                occupancy_lapsed = []

                # Create trial-level state array
                trial_states = []
                for state, length in zip(states, bout_lengths):
                    trial_states.extend([state] * int(length))
                trial_states = np.array(trial_states)

                # Compute sliding window occupancy
                for start in range(0, len(trial_states) - window_size + 1, window_stride):
                    end = start + window_size
                    window = trial_states[start:end]

                    window_centers.append((start + end) / 2)
                    occupancy_engaged.append(np.mean(window == 0))
                    occupancy_biased.append(np.mean(window == 1))
                    occupancy_lapsed.append(np.mean(window == 2))

                if len(window_centers) > 0:
                    for trial_bin, occ_eng, occ_bias, occ_lap in zip(
                        window_centers, occupancy_engaged, occupancy_biased, occupancy_lapsed
                    ):
                        self.all_data.append({
                            'animal_id': animal_id,
                            'genotype': genotype_new,
                            'trial_bin': trial_bin,
                            'occupancy_engaged': occ_eng,
                            'occupancy_biased': occ_bias,
                            'occupancy_lapsed': occ_lap
                        })

            except Exception as e:
                print(f"Error processing {pkl_file}: {e}")
                continue

        self.data_df = pd.DataFrame(self.all_data)
        print(f"Extracted {len(self.data_df)} trial bins from {len(self.data_df['animal_id'].unique())} animals")
        return self.data_df

    def fit_regressions(self):
        """Fit linear regressions for each genotype and state"""
        results = {}

        for state_col, state_name in [
            ('occupancy_engaged', 'Engaged'),
            ('occupancy_biased', 'Biased'),
            ('occupancy_lapsed', 'Lapsed')
        ]:
            results[state_name] = {}

            # Overall regression
            slope, intercept, r_value, p_value, std_err = linregress(
                self.data_df['trial_bin'], self.data_df[state_col]
            )
            results[state_name]['Overall'] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_err': std_err,
                'ci_lower': slope - 1.96 * std_err,
                'ci_upper': slope + 1.96 * std_err
            }

            # Per-genotype regressions
            for genotype in sorted(self.data_df['genotype'].unique()):
                geno_data = self.data_df[self.data_df['genotype'] == genotype]

                if len(geno_data) < 5:
                    continue

                slope, intercept, r_value, p_value, std_err = linregress(
                    geno_data['trial_bin'], geno_data[state_col]
                )

                results[state_name][genotype] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'std_err': std_err,
                    'ci_lower': slope - 1.96 * std_err,
                    'ci_upper': slope + 1.96 * std_err,
                    'n': len(geno_data)
                }

        self.regression_results = results
        return results

    def fit_animal_regressions(self):
        """Fit regressions per animal to assess individual variation"""
        animal_slopes = {}

        for state_col, state_name in [
            ('occupancy_engaged', 'Engaged'),
            ('occupancy_biased', 'Biased'),
            ('occupancy_lapsed', 'Lapsed')
        ]:
            animal_slopes[state_name] = {}

            for animal_id in self.data_df['animal_id'].unique():
                animal_data = self.data_df[self.data_df['animal_id'] == animal_id]

                if len(animal_data) < 3:
                    continue

                slope, intercept, r_value, p_value, std_err = linregress(
                    animal_data['trial_bin'], animal_data[state_col]
                )

                animal_slopes[state_name][animal_id] = {
                    'slope': slope,
                    'genotype': animal_data['genotype'].values[0]
                }

        self.animal_slopes = animal_slopes
        return animal_slopes


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_enhanced_phase2_figure(analyzer, output_dir):
    """Create comprehensive multi-panel enhanced Phase 2 visualization"""

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # =================================================================
    # PANEL 1-3: STATE TRAJECTORIES WITH INDIVIDUAL CURVES
    # =================================================================

    state_names = ['Engaged', 'Biased', 'Lapsed']
    state_cols = ['occupancy_engaged', 'occupancy_biased', 'occupancy_lapsed']

    for state_idx, (state_name, state_col) in enumerate(zip(state_names, state_cols)):
        ax = fig.add_subplot(gs[0, state_idx])

        # Plot individual animal curves (light, transparent)
        for animal_id in analyzer.data_df['animal_id'].unique():
            animal_data = analyzer.data_df[analyzer.data_df['animal_id'] == animal_id]
            genotype = animal_data['genotype'].values[0]
            color = GENO_COLORS[genotype]

            ax.plot(animal_data['trial_bin'], animal_data[state_col],
                   alpha=0.15, color=color, linewidth=0.8)

        # Plot genotype-specific regression lines with confidence intervals
        for genotype in sorted(analyzer.data_df['genotype'].unique()):
            geno_data = analyzer.data_df[analyzer.data_df['genotype'] == genotype]
            color = GENO_COLORS[genotype]

            # Get regression results
            if genotype in analyzer.regression_results[state_name]:
                result = analyzer.regression_results[state_name][genotype]

                # Create regression line
                x_line = np.array([geno_data['trial_bin'].min(), geno_data['trial_bin'].max()])
                y_line = result['intercept'] + result['slope'] * x_line

                ax.plot(x_line, y_line, color=color, linewidth=2.5, label=genotype)

                # Plot confidence interval as shaded region
                ci_lower = result['intercept'] + result['ci_lower'] * x_line
                ci_upper = result['intercept'] + result['ci_upper'] * x_line
                ax.fill_between(x_line, ci_lower, ci_upper, alpha=0.15, color=color)

        ax.set_xlabel('Trial Number (Phase 2)')
        ax.set_ylabel(f'P({state_name})')
        ax.set_title(f'{state_name} State Occupancy Over Trials\n(Individual curves + genotype-specific regression lines)')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])

    # =================================================================
    # PANEL 4: LEARNING RATE COMPARISON (Slopes)
    # =================================================================

    ax = fig.add_subplot(gs[1, 0])

    # Prepare data for slopes plot
    slope_data = []
    for state_name in state_names:
        for genotype in sorted(analyzer.data_df['genotype'].unique()):
            if genotype in analyzer.regression_results[state_name]:
                result = analyzer.regression_results[state_name][genotype]
                slope_data.append({
                    'genotype': genotype,
                    'state': state_name,
                    'slope': result['slope'] * 100,  # Convert to %/100trials
                    'ci_lower': result['ci_lower'] * 100,
                    'ci_upper': result['ci_upper'] * 100,
                    'p_value': result['p_value']
                })

    slope_df = pd.DataFrame(slope_data)

    # Group by state for visualization
    x_pos = 0
    for state_idx, state_name in enumerate(state_names):
        state_data = slope_df[slope_df['state'] == state_name]

        for gen_idx, (_, row) in enumerate(state_data.iterrows()):
            color = GENO_COLORS[row['genotype']]
            ci_width = (row['ci_upper'] - row['ci_lower'])

            # Bar plot with error bars (CI)
            ax.bar(x_pos, row['slope'], color=color, alpha=0.7, width=0.8)
            ax.errorbar(x_pos, row['slope'],
                       yerr=[[row['slope'] - row['ci_lower']], [row['ci_upper'] - row['slope']]],
                       fmt='none', ecolor='black', capsize=3, linewidth=1.5)

            # Add significance asterisks
            if row['p_value'] < 0.001:
                sig_marker = '***'
            elif row['p_value'] < 0.01:
                sig_marker = '**'
            elif row['p_value'] < 0.05:
                sig_marker = '*'
            else:
                sig_marker = 'n.s.'

            if sig_marker != 'n.s.':
                y_pos = max(row['ci_upper'], 0) + 0.5
                ax.text(x_pos, y_pos, sig_marker, ha='center', fontsize=10, fontweight='bold')

            x_pos += 1

        # Add spacing between states
        x_pos += 1

    ax.set_ylabel('Learning Rate (% change per 100 trials)')
    ax.set_title('Learning Rates by Genotype and State\n(Error bars = 95% CI)')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    # Create legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=GENO_COLORS[g], alpha=0.7)
                      for g in sorted(analyzer.data_df['genotype'].unique())]
    ax.legend(legend_elements, sorted(analyzer.data_df['genotype'].unique()),
             title='Genotype', fontsize=8, loc='best')

    # =================================================================
    # PANEL 5: STATISTICAL SUMMARY TABLE
    # =================================================================

    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')

    # Create summary text
    summary_text = "STATISTICAL SUMMARY: LEARNING RATES\n" + "=" * 50 + "\n\n"

    for state_name in state_names:
        summary_text += f"{state_name} State:\n"
        for genotype in sorted(analyzer.data_df['genotype'].unique()):
            if genotype in analyzer.regression_results[state_name]:
                result = analyzer.regression_results[state_name][genotype]
                slope = result['slope'] * 100
                p_val = result['p_value']
                ci_lower = result['ci_lower'] * 100
                ci_upper = result['ci_upper'] * 100

                # Significance marker
                sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'n.s.'))

                summary_text += f"  {genotype:12s}: {slope:+6.2f} [{ci_lower:+6.2f}, {ci_upper:+6.2f}] {sig}\n"

        summary_text += "\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # =================================================================
    # PANEL 6: EFFECT SIZE (R-squared) BY STATE AND GENOTYPE
    # =================================================================

    ax = fig.add_subplot(gs[1, 2])

    r_squared_data = []
    for state_name in state_names:
        for genotype in sorted(analyzer.data_df['genotype'].unique()):
            if genotype in analyzer.regression_results[state_name]:
                result = analyzer.regression_results[state_name][genotype]
                r_squared_data.append({
                    'genotype': genotype,
                    'state': state_name,
                    'r_squared': result['r_squared']
                })

    r2_df = pd.DataFrame(r_squared_data)
    r2_pivot = r2_df.pivot(index='genotype', columns='state', values='r_squared')
    r2_pivot = r2_pivot[state_names]  # Reorder columns

    sns.heatmap(r2_pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
               cbar_kws={'label': 'R² (Variance Explained)'})
    ax.set_title('Effect Size: R² by Genotype and State\n(Higher = Better model fit)')
    ax.set_xlabel('State')
    ax.set_ylabel('Genotype')

    # =================================================================
    # PANEL 7-9: WITHIN-GENOTYPE VARIATION (Individual slopes)
    # =================================================================

    for state_idx, (state_name, state_col) in enumerate(zip(state_names, state_cols)):
        ax = fig.add_subplot(gs[2, state_idx])

        # Extract individual slopes for this state
        individual_slopes = []
        for genotype in sorted(analyzer.data_df['genotype'].unique()):
            if state_name in analyzer.animal_slopes:
                for animal_id, data in analyzer.animal_slopes[state_name].items():
                    if data['genotype'] == genotype:
                        individual_slopes.append({
                            'genotype': genotype,
                            'slope': data['slope'] * 100,
                            'animal_id': animal_id
                        })

        if individual_slopes:
            ind_slope_df = pd.DataFrame(individual_slopes)

            # Violin plot with individual points
            genotypes_list = sorted(analyzer.data_df['genotype'].unique())
            positions = range(len(genotypes_list))
            data_by_geno = [ind_slope_df[ind_slope_df['genotype'] == g]['slope'].values
                           for g in genotypes_list]

            parts = ax.violinplot(data_by_geno, positions=positions, showmeans=True, showmedians=True)

            # Customize violin colors
            for pc, genotype in zip(parts['bodies'], genotypes_list):
                pc.set_facecolor(GENO_COLORS[genotype])
                pc.set_alpha(0.6)

            # Overlay individual points
            for pos, genotype in enumerate(genotypes_list):
                geno_slopes = ind_slope_df[ind_slope_df['genotype'] == genotype]['slope'].values
                x_jitter = np.random.normal(pos, 0.04, size=len(geno_slopes))
                ax.scatter(x_jitter, geno_slopes, alpha=0.5, s=50, color=GENO_COLORS[genotype])

            ax.set_xticks(positions)
            ax.set_xticklabels(genotypes_list, rotation=45, ha='right')
            ax.set_ylabel('Individual Animal Learning Rate (% per 100 trials)')
            ax.set_title(f'{state_name}: Within-Genotype Variation in Learning Rates')
            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')

    # Overall title
    fig.suptitle('Enhanced Phase 2 State Occupancy Dynamics: Learning Effects by Genotype',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(os.path.join(output_dir, 'phase2_state_occupancy_dynamics_enhanced.png'),
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'phase2_state_occupancy_dynamics_enhanced.pdf'),
               bbox_inches='tight')
    plt.close()

    print("✓ Saved enhanced Phase 2 state dynamics figure")


def create_learning_rate_summary_table(analyzer, output_dir):
    """Create detailed summary table of learning rates with statistics"""

    summary_data = []

    for state_name in ['Engaged', 'Biased', 'Lapsed']:
        for genotype in sorted(analyzer.data_df['genotype'].unique()):
            if genotype in analyzer.regression_results[state_name]:
                result = analyzer.regression_results[state_name][genotype]

                summary_data.append({
                    'State': state_name,
                    'Genotype': genotype,
                    'Slope (% per 100 trials)': f"{result['slope']*100:.4f}",
                    '95% CI Lower': f"{result['ci_lower']*100:.4f}",
                    '95% CI Upper': f"{result['ci_upper']*100:.4f}",
                    'P-value': f"{result['p_value']:.6f}",
                    'Significance': ('***' if result['p_value'] < 0.001
                                    else '**' if result['p_value'] < 0.01
                                    else '*' if result['p_value'] < 0.05
                                    else 'n.s.'),
                    'R²': f"{result['r_squared']:.4f}",
                    'N Bins': result['n']
                })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, '../phase2_learning_rates_enhanced.csv'), index=False)

    print("\n" + "="*120)
    print("ENHANCED PHASE 2 LEARNING RATES - DETAILED SUMMARY")
    print("="*120)
    print(summary_df.to_string(index=False))
    print("="*120 + "\n")

    return summary_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("="*80)
    print("ENHANCED PHASE 2 STATE OCCUPANCY DYNAMICS VISUALIZATION")
    print("="*80)
    print()

    # Step 1: Extract data
    print("[1/3] Extracting Phase 2 state occupancy data...")
    analyzer = Phase2StateAnalyzer()
    analyzer.extract_phase2_states()

    # Step 2: Fit regressions
    print("[2/3] Fitting regression models...")
    analyzer.fit_regressions()
    analyzer.fit_animal_regressions()

    # Step 3: Create visualizations
    print("[3/3] Creating enhanced visualizations...")
    create_enhanced_phase2_figure(analyzer, OUTPUT_DIR)
    create_learning_rate_summary_table(analyzer, OUTPUT_DIR)

    print()
    print("="*80)
    print("✓ ENHANCED VISUALIZATION COMPLETE")
    print("="*80)
    print()
    print("Output files:")
    print(f"  - phase2_state_occupancy_dynamics_enhanced.png")
    print(f"  - phase2_state_occupancy_dynamics_enhanced.pdf")
    print(f"  - phase2_learning_rates_enhanced.csv")
