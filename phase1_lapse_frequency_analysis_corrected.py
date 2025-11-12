"""
Phase 1 Lapse Frequency Analysis - Non-parametric Statistical Testing (CORRECTED)

Uses existing pre-calculated lapse_frequency data and performs Kruskal-Wallis test
to compare lapse frequency across genotypes with post-hoc pairwise Mann-Whitney U tests.

Data source: /home/user/GLMHMM/results/phase1_non_reversal/priority2_interpretation/lapse_statistics.csv
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

LAPSE_DATA_PATH = '/home/user/GLMHMM/results/phase1_non_reversal/priority2_interpretation/lapse_statistics.csv'
OUTPUT_DIR = '/home/user/GLMHMM/results/statistical_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Genotype mapping
OLD_TO_NEW = {
    '+/+': 'A1D_Wt', '-/-': 'A1D_KO', '+/-': 'A1D_Het',
    '+': 'B6', '-': 'C3H x B6'
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_lapse_data():
    """Load pre-calculated lapse frequency data."""
    df = pd.read_csv(LAPSE_DATA_PATH)

    # Map genotypes to new labels
    df['genotype_new'] = df['genotype'].map(OLD_TO_NEW)
    df = df.dropna(subset=['genotype_new'])

    return df[['animal_id', 'genotype', 'genotype_new', 'lapse_frequency', 'avg_lapse_duration', 'total_trials']]

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def perform_kruskal_wallis(df):
    """Perform Kruskal-Wallis test across genotypes."""
    genotypes = sorted(df['genotype_new'].unique())
    groups = [df[df['genotype_new'] == g]['lapse_frequency'].values for g in genotypes]

    # Kruskal-Wallis H-test
    h_statistic, p_value = stats.kruskal(*groups)

    return {
        'H_statistic': h_statistic,
        'p_value': p_value,
        'df': len(genotypes) - 1,
        'test_name': 'Kruskal-Wallis H-test'
    }

def perform_mann_whitney_u(group1, group2, alternative='two-sided'):
    """Perform Mann-Whitney U test between two groups."""
    u_statistic, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
    return u_statistic, p_value

def calculate_effect_size_r(u_statistic, n1, n2):
    """Calculate effect size r from Mann-Whitney U statistic."""
    n_total = n1 + n2
    z = (u_statistic - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n_total + 1) / 12)
    r = abs(z) / np.sqrt(n_total)
    return r

def interpret_effect_size(r):
    """Interpret effect size r (Cohen's guidelines)."""
    if r < 0.1:
        return 'Negligible'
    elif r < 0.3:
        return 'Small'
    elif r < 0.5:
        return 'Medium'
    else:
        return 'Large'

def perform_pairwise_comparisons(df):
    """Perform pairwise Mann-Whitney U tests with Bonferroni correction."""
    genotypes = sorted(df['genotype_new'].unique())
    n_comparisons = len(genotypes) * (len(genotypes) - 1) // 2
    bonferroni_alpha = 0.05 / n_comparisons

    pairwise_results = []

    for i, g1 in enumerate(genotypes):
        for g2 in genotypes[i+1:]:
            group1 = df[df['genotype_new'] == g1]['lapse_frequency'].values
            group2 = df[df['genotype_new'] == g2]['lapse_frequency'].values

            u_stat, p_val = perform_mann_whitney_u(group1, group2)
            r = calculate_effect_size_r(u_stat, len(group1), len(group2))

            # Calculate descriptive stats
            median1 = np.median(group1)
            median2 = np.median(group2)
            iqr1 = np.percentile(group1, 75) - np.percentile(group1, 25)
            iqr2 = np.percentile(group2, 75) - np.percentile(group2, 25)

            pairwise_results.append({
                'Comparison': f'{g1} vs {g2}',
                'Group1': g1,
                'Group2': g2,
                'n1': len(group1),
                'n2': len(group2),
                'Median1': median1,
                'IQR1': iqr1,
                'Median2': median2,
                'IQR2': iqr2,
                'U_statistic': u_stat,
                'p_value': p_val,
                'p_bonferroni': p_val * n_comparisons,
                'Significant': 'Yes' if p_val < bonferroni_alpha else 'No',
                'Effect_size_r': r,
                'Effect_interpretation': interpret_effect_size(r)
            })

    return pd.DataFrame(pairwise_results), bonferroni_alpha

def compute_descriptive_stats(df):
    """Compute descriptive statistics by genotype."""
    stats_by_geno = []

    for genotype in sorted(df['genotype_new'].unique()):
        group = df[df['genotype_new'] == genotype]['lapse_frequency'].values

        stats_by_geno.append({
            'Genotype': genotype,
            'N': len(group),
            'Mean': np.mean(group),
            'Median': np.median(group),
            'SD': np.std(group, ddof=1),
            'Min': np.min(group),
            'Q1': np.percentile(group, 25),
            'Q3': np.percentile(group, 75),
            'Max': np.max(group),
            'IQR': np.percentile(group, 75) - np.percentile(group, 25)
        })

    return pd.DataFrame(stats_by_geno)

# =============================================================================
# VISUALIZATION AND REPORTING
# =============================================================================

def create_summary_report(df, kw_results, pairwise_df, descriptive_df, bonferroni_alpha):
    """Create comprehensive statistical summary report."""

    report = []
    report.append("=" * 100)
    report.append("PHASE 1 LAPSE FREQUENCY ANALYSIS - NON-PARAMETRIC STATISTICS")
    report.append("=" * 100)
    report.append("")

    report.append("BACKGROUND:")
    report.append("-" * 100)
    report.append("Lapse Frequency = Percentage of trials in which animal was lapsed (disengaged)")
    report.append("Data source: Pre-calculated lapse statistics from Phase 1 models")
    report.append("Total animals: {} across {} genotypes".format(len(df), df['genotype_new'].nunique()))
    report.append("")
    report.append("RATIONALE FOR NON-PARAMETRIC TESTING:")
    report.append("  • Small sample sizes per group (n=7-22) may violate normality assumptions")
    report.append("  • Lapse frequency data may be skewed (many zeros in B6 group)")
    report.append("  • Kruskal-Wallis H-test: robust alternative to one-way ANOVA")
    report.append("  • Mann-Whitney U test: robust pairwise comparisons without normality")
    report.append("")

    report.append("=" * 100)
    report.append("1. DESCRIPTIVE STATISTICS BY GENOTYPE (Lapse Frequency %)")
    report.append("=" * 100)
    report.append("")
    report.append(descriptive_df.to_string(index=False))
    report.append("")

    report.append("=" * 100)
    report.append("2. KRUSKAL-WALLIS H-TEST (OVERALL COMPARISON)")
    report.append("=" * 100)
    report.append("")
    report.append(f"Test: {kw_results['test_name']}")
    report.append(f"H-statistic: {kw_results['H_statistic']:.4f}")
    report.append(f"Degrees of freedom: {kw_results['df']}")
    report.append(f"P-value: {kw_results['p_value']:.6f}")
    report.append("")
    if kw_results['p_value'] < 0.05:
        report.append("★ SIGNIFICANT (p < 0.05)")
        report.append("  → Lapse frequency differs significantly among genotypes")
    else:
        report.append("✗ NOT SIGNIFICANT (p ≥ 0.05)")
        report.append("  → No significant overall differences in lapse frequency among genotypes")
    report.append("")

    report.append("=" * 100)
    report.append("3. POST-HOC PAIRWISE COMPARISONS (MANN-WHITNEY U TESTS)")
    report.append("=" * 100)
    report.append("")
    report.append(f"Bonferroni-corrected α: {bonferroni_alpha:.6f}")
    report.append("(Bonferroni correction applied for multiple comparisons)")
    report.append("")

    # Format pairwise results for display
    pairwise_display = pairwise_df.copy()
    pairwise_display['Median1'] = pairwise_display['Median1'].round(2)
    pairwise_display['Median2'] = pairwise_display['Median2'].round(2)
    pairwise_display['IQR1'] = pairwise_display['IQR1'].round(2)
    pairwise_display['IQR2'] = pairwise_display['IQR2'].round(2)
    pairwise_display['U_statistic'] = pairwise_display['U_statistic'].round(1)
    pairwise_display['p_value'] = pairwise_display['p_value'].apply(lambda x: f'{x:.6f}')
    pairwise_display['p_bonferroni'] = pairwise_display['p_bonferroni'].apply(lambda x: f'{x:.6f}')
    pairwise_display['Effect_size_r'] = pairwise_display['Effect_size_r'].round(3)

    report.append(pairwise_display[['Comparison', 'n1', 'n2', 'Median1', 'Median2', 'U_statistic',
                                     'p_value', 'Significant', 'Effect_size_r',
                                     'Effect_interpretation']].to_string(index=False))
    report.append("")

    report.append("=" * 100)
    report.append("4. EFFECT SIZE INTERPRETATION (Cohen's Guidelines for r)")
    report.append("=" * 100)
    report.append("")
    report.append("r < 0.1      : Negligible effect")
    report.append("0.1 ≤ r < 0.3: Small effect")
    report.append("0.3 ≤ r < 0.5: Medium effect")
    report.append("r ≥ 0.5      : Large effect")
    report.append("")

    report.append("=" * 100)
    report.append("5. SUMMARY AND INTERPRETATION")
    report.append("=" * 100)
    report.append("")

    sig_comparisons = pairwise_df[pairwise_df['Significant'] == 'Yes']
    if len(sig_comparisons) > 0:
        report.append(f"Found {len(sig_comparisons)} significant pairwise difference(s):")
        for _, row in sig_comparisons.iterrows():
            report.append(f"  ★ {row['Comparison']}")
            report.append(f"    Median: {row['Median1']:.2f}% vs {row['Median2']:.2f}%")
            report.append(f"    p = {row['p_value']:.6f}, r = {row['Effect_size_r']:.3f} ({row['Effect_interpretation']} effect)")
    else:
        report.append("No significant pairwise differences found after Bonferroni correction.")

    report.append("")
    report.append("=" * 100)
    report.append("STATISTICAL METHODS REFERENCE:")
    report.append("=" * 100)
    report.append("")
    report.append("Kruskal-Wallis H-test:")
    report.append("  Non-parametric alternative to one-way ANOVA testing whether k independent")
    report.append("  samples come from the same distribution. Null hypothesis: all groups have")
    report.append("  identical distributions.")
    report.append("")
    report.append("Mann-Whitney U test:")
    report.append("  Non-parametric test comparing two independent samples. Tests whether one")
    report.append("  distribution is stochastically larger than the other.")
    report.append("")
    report.append("Effect size (r):")
    report.append("  Standardized measure of difference magnitude independent of sample size.")
    report.append("  Calculated as: r = |Z| / sqrt(N), where Z is standardized U statistic.")
    report.append("")

    return "\n".join(report)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("PHASE 1 LAPSE FREQUENCY ANALYSIS (CORRECTED)")
    print("=" * 80)
    print()

    print("[1/4] Loading Phase 1 lapse frequency data...")
    df = load_lapse_data()
    print(f"Loaded {len(df)} animals across {df['genotype_new'].nunique()} genotypes")
    print()

    print("[2/4] Computing descriptive statistics...")
    descriptive_df = compute_descriptive_stats(df)
    print(descriptive_df.to_string(index=False))
    print()

    print("[3/4] Performing Kruskal-Wallis H-test...")
    kw_results = perform_kruskal_wallis(df)
    print(f"H-statistic = {kw_results['H_statistic']:.4f}, p-value = {kw_results['p_value']:.6f}")
    if kw_results['p_value'] < 0.05:
        print("★ SIGNIFICANT")
    else:
        print("✗ NOT SIGNIFICANT")
    print()

    print("[4/4] Performing pairwise Mann-Whitney U tests...")
    pairwise_df, bonferroni_alpha = perform_pairwise_comparisons(df)
    print(f"Bonferroni-corrected α = {bonferroni_alpha:.6f}")
    print()

    # Create report
    report = create_summary_report(df, kw_results, pairwise_df, descriptive_df, bonferroni_alpha)
    print(report)

    # Save report
    report_file = os.path.join(OUTPUT_DIR, 'phase1_lapse_frequency_kruskal_wallis.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    print()
    print(f"✓ Report saved to: {report_file}")

    # Save detailed data tables
    descriptive_df.to_csv(os.path.join(OUTPUT_DIR, 'phase1_lapse_frequency_descriptive_kw.csv'), index=False)
    pairwise_df.to_csv(os.path.join(OUTPUT_DIR, 'phase1_lapse_frequency_pairwise_kw.csv'), index=False)
    print(f"✓ Data tables saved to: {OUTPUT_DIR}/phase1_lapse_frequency_*_kw.csv")

    print()
    print("=" * 80)
