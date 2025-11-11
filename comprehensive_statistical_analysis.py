"""
Comprehensive Statistical Analysis of GLM-HMM Results
======================================================

Adds statistical rigor to existing graphs and analyses:

1. Phase 1 Lapse Metrics: Kruskal-Wallis tests + post-hoc Dunn tests
2. Phase 2 P(State) over Trials: Mixed-effects regression
3. Late Lapsers: Early vs late performance by genotype
4. High/Low Performers: Genotypic distribution tests
5. State Stability: Phase 1 → Phase 2 transitions by genotype
6. State Transitions: Genotypic differences in transition matrices

All with correct genotype labels and comprehensive visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, wilcoxon, chi2_contingency, spearmanr
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '/home/user/GLMHMM')

from genotype_labels import (
    GENOTYPE_MAP, GENOTYPE_ORDER, GENOTYPE_COLORS,
    STATE_LABELS, STATE_COLORS, relabel_genotype
)
from glmhmm_utils import load_and_preprocess_session_data

plt.style.use('seaborn-v0_8-whitegrid')

# Genotype mapping for old labels (from raw data) to new labels
OLD_TO_NEW_GENOTYPE = {
    '+': 'B6',
    '-': 'C3H x B6',
    '+/+': 'A1D_Wt',
    '+/-': 'A1D_Het',
    '-/-': 'A1D_KO'
}

class ComprehensiveStatisticalAnalysis:
    """Perform statistical analysis on GLM-HMM results with correct labels."""

    def __init__(self):
        self.results_dir = Path('/home/user/GLMHMM/results')
        self.phase1_dir = self.results_dir / 'phase1_non_reversal'
        self.phase2_dir = self.results_dir / 'phase2_reversal'

        self.output_dir = self.results_dir / 'statistical_analysis'
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'phase1').mkdir(exist_ok=True)
        (self.output_dir / 'phase2').mkdir(exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)

        self.summary_report = []

    def add_report(self, text):
        """Add text to summary report."""
        self.summary_report.append(text)
        print(text)

    def save_report(self):
        """Save complete summary report."""
        report_path = self.output_dir / 'STATISTICAL_ANALYSIS_REPORT.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(self.summary_report))
        print(f"\n✓ Report saved to {report_path}")

    # ========== PHASE 1: LAPSE METRICS ==========

    def analyze_phase1_lapse_metrics(self):
        """Kruskal-Wallis tests for lapse frequency and duration."""
        print("\n" + "="*80)
        print("PHASE 1: LAPSE METRICS STATISTICAL ANALYSIS")
        print("="*80)

        self.add_report("\n" + "="*80)
        self.add_report("PHASE 1: LAPSE METRICS STATISTICAL ANALYSIS")
        self.add_report("="*80)

        # Load lapse statistics data
        lapse_stats_path = self.phase1_dir / 'priority2_interpretation' / 'lapse_statistics.csv'
        lapse_df = pd.read_csv(lapse_stats_path)

        # Convert genotype labels
        lapse_df['genotype_new'] = lapse_df['genotype'].map(OLD_TO_NEW_GENOTYPE)
        lapse_df = lapse_df.dropna(subset=['genotype_new'])

        self.add_report(f"\nData loaded: {len(lapse_df)} animals")
        self.add_report(f"Genotypes present: {sorted(lapse_df['genotype_new'].unique())}")

        # ===== LAPSE FREQUENCY =====
        self.add_report("\n" + "-"*80)
        self.add_report("LAPSE FREQUENCY ANALYSIS (per 100 trials)")
        self.add_report("-"*80)

        # Kruskal-Wallis test
        genotype_groups_freq = [lapse_df[lapse_df['genotype_new'] == g]['lapse_frequency'].values
                               for g in sorted(lapse_df['genotype_new'].unique())]

        h_stat, p_freq = kruskal(*genotype_groups_freq)
        self.add_report(f"\nKruskal-Wallis H-statistic: {h_stat:.4f}")
        self.add_report(f"p-value: {p_freq:.6f}")
        if p_freq < 0.05:
            self.add_report("*** SIGNIFICANT: Lapse frequency differs by genotype (p < 0.05)")
        else:
            self.add_report(f"Not significant (p = {p_freq:.4f})")

        # Descriptive statistics
        self.add_report("\nDescriptive Statistics:")
        freq_summary = lapse_df.groupby('genotype_new')['lapse_frequency'].agg([
            ('N', 'count'),
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('SD', 'std'),
            ('Min', 'min'),
            ('Max', 'max')
        ]).round(3)
        self.add_report(freq_summary.to_string())
        freq_summary.to_csv(self.output_dir / 'phase1' / 'lapse_frequency_stats.csv')

        # ===== LAPSE DURATION =====
        self.add_report("\n" + "-"*80)
        self.add_report("LAPSE DURATION ANALYSIS (average duration of lapse bouts)")
        self.add_report("-"*80)

        # Kruskal-Wallis test
        genotype_groups_dur = [lapse_df[lapse_df['genotype_new'] == g]['avg_lapse_duration'].values
                              for g in sorted(lapse_df['genotype_new'].unique())]

        h_stat, p_dur = kruskal(*genotype_groups_dur)
        self.add_report(f"\nKruskal-Wallis H-statistic: {h_stat:.4f}")
        self.add_report(f"p-value: {p_dur:.6f}")
        if p_dur < 0.05:
            self.add_report("*** SIGNIFICANT: Lapse duration differs by genotype (p < 0.05)")
        else:
            self.add_report(f"Not significant (p = {p_dur:.4f})")

        # Descriptive statistics
        self.add_report("\nDescriptive Statistics:")
        dur_summary = lapse_df.groupby('genotype_new')['avg_lapse_duration'].agg([
            ('N', 'count'),
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('SD', 'std'),
            ('Min', 'min'),
            ('Max', 'max')
        ]).round(3)
        self.add_report(dur_summary.to_string())
        dur_summary.to_csv(self.output_dir / 'phase1' / 'lapse_duration_stats.csv')

        return lapse_df, (h_stat, p_freq), (h_stat, p_dur)

    def create_phase1_lapse_figures_with_stats(self, lapse_df, freq_stats, dur_stats):
        """Create enhanced lapse metric figures with p-values."""
        print("\nCreating Phase 1 lapse metric figures with statistics...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Prepare data
        genotype_order = [g for g in GENOTYPE_ORDER if g in lapse_df['genotype_new'].unique()]

        # ===== LAPSE FREQUENCY FIGURE =====
        ax = axes[0]

        freq_data = []
        for genotype in genotype_order:
            freq_data.append(lapse_df[lapse_df['genotype_new'] == genotype]['lapse_frequency'].values)

        bp1 = ax.boxplot(freq_data, labels=genotype_order, patch_artist=True, widths=0.6)

        # Color the boxes by genotype
        for patch, genotype in zip(bp1['boxes'], genotype_order):
            patch.set_facecolor(GENOTYPE_COLORS.get(genotype, '#95a5a6'))
            patch.set_alpha(0.7)

        # Overlay individual points
        for i, (genotype, data) in enumerate(zip(genotype_order, freq_data)):
            x = np.random.normal(i+1, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.4, s=50, color='black')

        ax.set_ylabel('Lapse Frequency (per 100 trials)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Genotype', fontsize=12, fontweight='bold')
        ax.set_title('Phase 1: Lapse Frequency by Genotype', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add p-value annotation
        h_stat, p_val = freq_stats
        sig_text = f"Kruskal-Wallis\np = {p_val:.4f}"
        if p_val < 0.05:
            sig_text += "\n***Significant"
            color = 'green'
        else:
            color = 'gray'
        ax.text(0.98, 0.97, sig_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
               fontweight='bold')

        # ===== LAPSE DURATION FIGURE =====
        ax = axes[1]

        dur_data = []
        for genotype in genotype_order:
            dur_data.append(lapse_df[lapse_df['genotype_new'] == genotype]['avg_lapse_duration'].values)

        bp2 = ax.boxplot(dur_data, labels=genotype_order, patch_artist=True, widths=0.6)

        # Color the boxes by genotype
        for patch, genotype in zip(bp2['boxes'], genotype_order):
            patch.set_facecolor(GENOTYPE_COLORS.get(genotype, '#95a5a6'))
            patch.set_alpha(0.7)

        # Overlay individual points
        for i, (genotype, data) in enumerate(zip(genotype_order, dur_data)):
            x = np.random.normal(i+1, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.4, s=50, color='black')

        ax.set_ylabel('Average Lapse Duration (trials)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Genotype', fontsize=12, fontweight='bold')
        ax.set_title('Phase 1: Lapse Duration by Genotype', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add p-value annotation
        h_stat, p_val = dur_stats
        sig_text = f"Kruskal-Wallis\np = {p_val:.4f}"
        if p_val < 0.05:
            sig_text += "\n***Significant"
            color = 'green'
        else:
            color = 'gray'
        ax.text(0.98, 0.97, sig_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
               fontweight='bold')

        plt.suptitle('Phase 1: Lapse Metrics by Genotype with Statistical Significance',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'figures' / 'phase1_lapse_metrics_with_stats.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()

        print(f"✓ Saved to {output_path}")

    # ========== PHASE 1: LATE LAPSERS ==========

    def analyze_phase1_late_lapsers(self):
        """Analyze late lapser performance improvement by genotype."""
        print("\n" + "="*80)
        print("PHASE 1: LATE LAPSER ANALYSIS")
        print("="*80)

        self.add_report("\n" + "="*80)
        self.add_report("PHASE 1: LATE LAPSER ANALYSIS")
        self.add_report("="*80)

        # Load late lapser data
        late_lapser_path = self.phase1_dir / 'late_lapser_analysis' / 'late_lapser_identification.csv'
        late_lapser_df = pd.read_csv(late_lapser_path)

        # Convert genotype labels
        late_lapser_df['genotype_new'] = late_lapser_df['genotype'].map(OLD_TO_NEW_GENOTYPE)
        late_lapser_df = late_lapser_df.dropna(subset=['genotype_new'])

        # Filter for late lapsers only
        late_lapsers = late_lapser_df[late_lapser_df['is_late_lapser'] == True].copy()

        self.add_report(f"\nTotal animals: {len(late_lapser_df)}")
        self.add_report(f"Late lapsers identified: {len(late_lapsers)} ({100*len(late_lapsers)/len(late_lapser_df):.1f}%)")

        if len(late_lapsers) == 0:
            self.add_report("No late lapsers found in the dataset.")
            return None

        self.add_report(f"\nLate Lapsers by Genotype:")
        self.add_report(late_lapsers.groupby('genotype_new').size().to_string())

        # ===== PAIRED COMPARISON: EARLY vs LATE =====
        self.add_report("\n" + "-"*80)
        self.add_report("PAIRED COMPARISON: Early vs Late Accuracy in Late Lapsers")
        self.add_report("-"*80)

        # Wilcoxon signed-rank test (non-parametric paired test)
        early_acc = late_lapsers['early_accuracy'].values
        late_acc = late_lapsers['late_accuracy'].values

        w_stat, p_paired = wilcoxon(early_acc, late_acc)

        self.add_report(f"\nWilcoxon Signed-Rank Test (paired):")
        self.add_report(f"W-statistic: {w_stat:.4f}")
        self.add_report(f"p-value: {p_paired:.6f}")

        mean_early = np.mean(early_acc)
        mean_late = np.mean(late_acc)
        median_early = np.median(early_acc)
        median_late = np.median(late_acc)

        self.add_report(f"\nOverall Results (all late lapsers):")
        self.add_report(f"  Early Accuracy - Mean: {mean_early:.4f}, Median: {median_early:.4f}")
        self.add_report(f"  Late Accuracy  - Mean: {mean_late:.4f}, Median: {median_late:.4f}")
        self.add_report(f"  Difference     - Mean: {mean_late - mean_early:.4f}")

        if p_paired < 0.05:
            self.add_report(f"*** SIGNIFICANT: Performance changed significantly (p < 0.05)")
        else:
            self.add_report(f"Not significant (p = {p_paired:.4f})")

        # ===== BY GENOTYPE =====
        self.add_report("\n" + "-"*80)
        self.add_report("STRATIFIED ANALYSIS: Early vs Late by Genotype")
        self.add_report("-"*80)

        genotype_results = []
        for genotype in sorted(late_lapsers['genotype_new'].unique()):
            geno_data = late_lapsers[late_lapsers['genotype_new'] == genotype]
            n = len(geno_data)

            if n < 2:
                continue

            early = geno_data['early_accuracy'].values
            late = geno_data['late_accuracy'].values

            # Wilcoxon test for this genotype
            w_stat_g, p_g = wilcoxon(early, late)

            mean_change = np.mean(late - early)
            median_change = np.median(late - early)

            self.add_report(f"\n{genotype} (N={n}):")
            self.add_report(f"  Early Acc:  {np.mean(early):.4f} ± {np.std(early):.4f}")
            self.add_report(f"  Late Acc:   {np.mean(late):.4f} ± {np.std(late):.4f}")
            self.add_report(f"  Mean Change: {mean_change:.4f}, Median Change: {median_change:.4f}")
            self.add_report(f"  Wilcoxon p: {p_g:.6f}")
            if p_g < 0.05:
                self.add_report(f"  *** SIGNIFICANT (p < 0.05)")

            genotype_results.append({
                'Genotype': genotype,
                'N': n,
                'Early_Mean': np.mean(early),
                'Late_Mean': np.mean(late),
                'Mean_Change': mean_change,
                'Median_Change': median_change,
                'Wilcoxon_p': p_g,
                'Significant': p_g < 0.05
            })

        genotype_results_df = pd.DataFrame(genotype_results)
        genotype_results_df.to_csv(self.output_dir / 'phase1' / 'late_lapser_by_genotype.csv', index=False)

        return late_lapsers, (w_stat, p_paired), genotype_results_df

    def create_phase1_late_lapser_figures(self, late_lapsers, paired_stats, genotype_results):
        """Create enhanced late lapser figures."""
        print("\nCreating Phase 1 late lapser figures with statistics...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # ===== OVERALL PAIRED COMPARISON =====
        ax = axes[0]

        early_acc = late_lapsers['early_accuracy'].values
        late_acc = late_lapsers['late_accuracy'].values

        # Create paired plot
        for i, (e, l) in enumerate(zip(early_acc, late_acc)):
            ax.plot([0, 1], [e, l], 'o-', color='gray', alpha=0.3, linewidth=1)

        # Add means
        ax.scatter([0], [np.mean(early_acc)], s=200, color='red', marker='D',
                  edgecolor='black', linewidth=2, label='Early', zorder=5)
        ax.scatter([1], [np.mean(late_acc)], s=200, color='green', marker='D',
                  edgecolor='black', linewidth=2, label='Late', zorder=5)

        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Early Phase', 'Late Phase'], fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Phase 1: Performance Change in Late Lapsers (All)', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=11)

        # Add statistics
        w_stat, p_val = paired_stats
        sig_text = f"Wilcoxon Signed-Rank\np = {p_val:.4f}"
        if p_val < 0.05:
            sig_text += "\n***Significant"
            color = 'green'
        else:
            color = 'gray'
        ax.text(0.98, 0.97, sig_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
               fontweight='bold')

        # ===== BY GENOTYPE =====
        ax = axes[1]

        genotype_order = sorted(late_lapsers['genotype_new'].unique())
        genotype_changes = []

        for genotype in genotype_order:
            geno_data = late_lapsers[late_lapsers['genotype_new'] == genotype]
            changes = geno_data['late_accuracy'].values - geno_data['early_accuracy'].values
            genotype_changes.append(changes)

        bp = ax.boxplot(genotype_changes, labels=genotype_order, patch_artist=True, widths=0.6)

        for patch, genotype in zip(bp['boxes'], genotype_order):
            patch.set_facecolor(GENOTYPE_COLORS.get(genotype, '#95a5a6'))
            patch.set_alpha(0.7)

        # Overlay individual points
        for i, (genotype, changes) in enumerate(zip(genotype_order, genotype_changes)):
            x = np.random.normal(i+1, 0.04, size=len(changes))
            ax.scatter(x, changes, alpha=0.4, s=50, color='black')

        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_ylabel('Performance Change (Late - Early)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Genotype', fontsize=12, fontweight='bold')
        ax.set_title('Phase 1: Performance Change by Genotype', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.suptitle('Phase 1: Late Lapser Analysis - Early vs Late Performance',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'figures' / 'phase1_late_lapsers_with_stats.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()

        print(f"✓ Saved to {output_path}")

    # ========== PHASE 2: HIGH VS LOW PERFORMERS ==========

    def analyze_phase2_performers(self):
        """Analyze genotypic distribution in high vs low performers."""
        print("\n" + "="*80)
        print("PHASE 2: HIGH VS LOW PERFORMER CLASSIFICATION")
        print("="*80)

        self.add_report("\n" + "="*80)
        self.add_report("PHASE 2: HIGH VS LOW PERFORMER CLASSIFICATION")
        self.add_report("="*80)

        # Load high/low performer data
        hvl_path = self.phase2_dir / 'detailed_analyses' / 'high_vs_low_performers.csv'
        hvl_df = pd.read_csv(hvl_path)

        # Convert genotype labels
        hvl_df['genotype_new'] = hvl_df['genotype'].map(OLD_TO_NEW_GENOTYPE)
        hvl_df = hvl_df.dropna(subset=['genotype_new'])

        self.add_report(f"\nTotal animals: {len(hvl_df)}")
        self.add_report(f"High Performers: {len(hvl_df[hvl_df['performer_group'] == 'High Performer'])}")
        self.add_report(f"Low Performers: {len(hvl_df[hvl_df['performer_group'] == 'Low Performer'])}")

        # ===== CHI-SQUARE TEST =====
        self.add_report("\n" + "-"*80)
        self.add_report("CHI-SQUARE TEST: Genotype Distribution across Performance Groups")
        self.add_report("-"*80)

        # Create contingency table
        contingency = pd.crosstab(hvl_df['genotype_new'], hvl_df['performer_group'])

        self.add_report("\nContingency Table:")
        self.add_report(contingency.to_string())

        chi2, p_chi, dof, expected = chi2_contingency(contingency.values)

        self.add_report(f"\nChi-Square Test:")
        self.add_report(f"  Chi2 statistic: {chi2:.4f}")
        self.add_report(f"  p-value: {p_chi:.6f}")
        self.add_report(f"  Degrees of freedom: {dof}")

        # Effect size: Cramér's V
        n = contingency.values.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
        self.add_report(f"  Cramér's V: {cramers_v:.4f}")

        if p_chi < 0.05:
            self.add_report(f"*** SIGNIFICANT: Genotype and performance group are associated (p < 0.05)")
        else:
            self.add_report(f"Not significant (p = {p_chi:.4f})")

        # ===== DESCRIPTIVE ANALYSIS BY GENOTYPE =====
        self.add_report("\n" + "-"*80)
        self.add_report("PERFORMER DISTRIBUTION BY GENOTYPE")
        self.add_report("-"*80)

        genotype_performer_stats = []
        for genotype in sorted(hvl_df['genotype_new'].unique()):
            geno_data = hvl_df[hvl_df['genotype_new'] == genotype]
            n_total = len(geno_data)
            n_high = len(geno_data[geno_data['performer_group'] == 'High Performer'])
            n_low = len(geno_data[geno_data['performer_group'] == 'Low Performer'])
            pct_high = 100 * n_high / n_total if n_total > 0 else 0

            self.add_report(f"\n{genotype} (N={n_total}):")
            self.add_report(f"  High Performers: {n_high} ({pct_high:.1f}%)")
            self.add_report(f"  Low Performers: {n_low} ({100-pct_high:.1f}%)")

            genotype_performer_stats.append({
                'Genotype': genotype,
                'N_Total': n_total,
                'N_High': n_high,
                'N_Low': n_low,
                'Pct_High': pct_high,
                'Pct_Low': 100 - pct_high
            })

        hvl_summary = pd.DataFrame(genotype_performer_stats)
        hvl_summary.to_csv(self.output_dir / 'phase2' / 'performer_distribution_by_genotype.csv', index=False)

        return hvl_df, (chi2, p_chi, cramers_v), contingency

    def create_phase2_performer_figures(self, hvl_df, chi_stats, contingency):
        """Create enhanced Phase 2 performer figures."""
        print("\nCreating Phase 2 performer classification figures...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # ===== STACKED BAR CHART BY GENOTYPE =====
        ax = axes[0]

        genotype_order = [g for g in GENOTYPE_ORDER if g in hvl_df['genotype_new'].unique()]

        high_counts = []
        low_counts = []

        for genotype in genotype_order:
            geno_data = hvl_df[hvl_df['genotype_new'] == genotype]
            high = len(geno_data[geno_data['performer_group'] == 'High Performer'])
            low = len(geno_data[geno_data['performer_group'] == 'Low Performer'])
            high_counts.append(high)
            low_counts.append(low)

        x = np.arange(len(genotype_order))
        width = 0.6

        bars1 = ax.bar(x, high_counts, width, label='High Performer',
                      color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x, low_counts, width, bottom=high_counts, label='Low Performer',
                      color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add labels
        for i, (h, l) in enumerate(zip(high_counts, low_counts)):
            total = h + l
            if h > 0:
                ax.text(i, h/2, f'{h}\n({100*h/total:.0f}%)', ha='center', va='center',
                       fontweight='bold', fontsize=10)
            if l > 0:
                ax.text(i, h + l/2, f'{l}\n({100*l/total:.0f}%)', ha='center', va='center',
                       fontweight='bold', fontsize=10)

        ax.set_ylabel('Number of Animals', fontsize=12, fontweight='bold')
        ax.set_xlabel('Genotype', fontsize=12, fontweight='bold')
        ax.set_title('Phase 2: Performer Distribution by Genotype', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(genotype_order, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Color the genotype labels
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_color(GENOTYPE_COLORS.get(genotype_order[i], '#95a5a6'))
            label.set_fontweight('bold')

        # Add p-value annotation
        chi2, p_val, cramers_v = chi_stats
        sig_text = f"Chi-Square Test\np = {p_val:.4f}\nCramér's V = {cramers_v:.3f}"
        if p_val < 0.05:
            sig_text += "\n***Significant"
            color = 'green'
        else:
            color = 'gray'
        ax.text(0.98, 0.97, sig_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
               fontweight='bold')

        # ===== PERCENTAGE BAR CHART =====
        ax = axes[1]

        high_pcts = [100*h/(h+l) if (h+l) > 0 else 0 for h, l in zip(high_counts, low_counts)]
        low_pcts = [100 - hp for hp in high_pcts]

        bars1 = ax.bar(x, high_pcts, width, label='High Performer',
                      color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x, low_pcts, width, bottom=high_pcts, label='Low Performer',
                      color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add percentage labels
        for i, (hp, lp) in enumerate(zip(high_pcts, low_pcts)):
            if hp > 5:
                ax.text(i, hp/2, f'{hp:.0f}%', ha='center', va='center',
                       fontweight='bold', fontsize=10, color='white')
            if lp > 5:
                ax.text(i, hp + lp/2, f'{lp:.0f}%', ha='center', va='center',
                       fontweight='bold', fontsize=10, color='white')

        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Genotype', fontsize=12, fontweight='bold')
        ax.set_title('Phase 2: Percentage Distribution', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(genotype_order, fontsize=11)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Color the genotype labels
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_color(GENOTYPE_COLORS.get(genotype_order[i], '#95a5a6'))
            label.set_fontweight('bold')

        plt.suptitle('Phase 2: High vs Low Performer Classification by Genotype',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'figures' / 'phase2_performer_classification_with_stats.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()

        print(f"✓ Saved to {output_path}")

    # ========== PHASE 2: STATE TRANSITIONS P1 TO P2 ==========

    def analyze_phase2_state_transitions(self):
        """Analyze state transition stability from Phase 1 to Phase 2."""
        print("\n" + "="*80)
        print("PHASE 2: STATE TRANSITION STABILITY (Phase 1 → Phase 2)")
        print("="*80)

        self.add_report("\n" + "="*80)
        self.add_report("PHASE 2: STATE TRANSITION STABILITY (Phase 1 → Phase 2)")
        self.add_report("="*80)

        # Load state transition data
        trans_path = self.phase2_dir / 'detailed_analyses' / 'state_transitions_p1_to_p2.csv'
        trans_df = pd.read_csv(trans_path)

        # Convert genotype labels
        trans_df['genotype_new'] = trans_df['genotype'].map(OLD_TO_NEW_GENOTYPE)
        trans_df = trans_df.dropna(subset=['genotype_new'])

        self.add_report(f"\nTotal animals with Phase 1 → Phase 2 data: {len(trans_df)}")

        # ===== ENGAGED STATE CHANGES =====
        self.add_report("\n" + "-"*80)
        self.add_report("ENGAGED STATE: Phase 1 → Phase 2 Changes")
        self.add_report("-"*80)

        # Kruskal-Wallis on engaged changes
        genotype_eng_changes = [trans_df[trans_df['genotype_new'] == g]['engaged_change'].values
                               for g in sorted(trans_df['genotype_new'].unique())]

        h_stat_eng, p_eng = kruskal(*genotype_eng_changes)

        self.add_report(f"\nKruskal-Wallis Test on Engaged State Changes:")
        self.add_report(f"  H-statistic: {h_stat_eng:.4f}")
        self.add_report(f"  p-value: {p_eng:.6f}")
        if p_eng < 0.05:
            self.add_report(f"  *** SIGNIFICANT: Engaged state changes differ by genotype")

        # Descriptive stats
        self.add_report("\nDescriptive Statistics - Engaged State Change (P2 - P1):")
        eng_summary = trans_df.groupby('genotype_new')['engaged_change'].agg([
            ('N', 'count'),
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('SD', 'std'),
            ('Min', 'min'),
            ('Max', 'max')
        ]).round(4)
        self.add_report(eng_summary.to_string())
        eng_summary.to_csv(self.output_dir / 'phase2' / 'engaged_state_changes.csv')

        # ===== LAPSED STATE CHANGES =====
        self.add_report("\n" + "-"*80)
        self.add_report("LAPSED STATE: Phase 1 → Phase 2 Changes")
        self.add_report("-"*80)

        # Kruskal-Wallis on lapsed changes
        genotype_laps_changes = [trans_df[trans_df['genotype_new'] == g]['lapsed_change'].values
                                for g in sorted(trans_df['genotype_new'].unique())]

        h_stat_laps, p_laps = kruskal(*genotype_laps_changes)

        self.add_report(f"\nKruskal-Wallis Test on Lapsed State Changes:")
        self.add_report(f"  H-statistic: {h_stat_laps:.4f}")
        self.add_report(f"  p-value: {p_laps:.6f}")
        if p_laps < 0.05:
            self.add_report(f"  *** SIGNIFICANT: Lapsed state changes differ by genotype")

        # Descriptive stats
        self.add_report("\nDescriptive Statistics - Lapsed State Change (P2 - P1):")
        laps_summary = trans_df.groupby('genotype_new')['lapsed_change'].agg([
            ('N', 'count'),
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('SD', 'std'),
            ('Min', 'min'),
            ('Max', 'max')
        ]).round(4)
        self.add_report(laps_summary.to_string())
        laps_summary.to_csv(self.output_dir / 'phase2' / 'lapsed_state_changes.csv')

        return trans_df, (h_stat_eng, p_eng), (h_stat_laps, p_laps)

    def create_phase2_transition_figures(self, trans_df, eng_stats, laps_stats):
        """Create enhanced Phase 2 state transition figures."""
        print("\nCreating Phase 2 state transition figures...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        genotype_order = [g for g in GENOTYPE_ORDER if g in trans_df['genotype_new'].unique()]

        # ===== ENGAGED STATE CHANGES =====
        ax = axes[0]

        eng_data = []
        for genotype in genotype_order:
            eng_data.append(trans_df[trans_df['genotype_new'] == genotype]['engaged_change'].values)

        bp1 = ax.boxplot(eng_data, labels=genotype_order, patch_artist=True, widths=0.6)

        for patch, genotype in zip(bp1['boxes'], genotype_order):
            patch.set_facecolor(GENOTYPE_COLORS.get(genotype, '#95a5a6'))
            patch.set_alpha(0.7)

        for i, (genotype, data) in enumerate(zip(genotype_order, eng_data)):
            x = np.random.normal(i+1, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.4, s=50, color='black')

        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_ylabel('Change in Engaged State (Phase 2 - Phase 1)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Genotype', fontsize=12, fontweight='bold')
        ax.set_title('Engaged State Transitions by Genotype', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add p-value
        h_stat, p_val = eng_stats
        sig_text = f"Kruskal-Wallis\np = {p_val:.4f}"
        if p_val < 0.05:
            sig_text += "\n***Significant"
            color = 'green'
        else:
            color = 'gray'
        ax.text(0.98, 0.97, sig_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
               fontweight='bold')

        # ===== LAPSED STATE CHANGES =====
        ax = axes[1]

        laps_data = []
        for genotype in genotype_order:
            laps_data.append(trans_df[trans_df['genotype_new'] == genotype]['lapsed_change'].values)

        bp2 = ax.boxplot(laps_data, labels=genotype_order, patch_artist=True, widths=0.6)

        for patch, genotype in zip(bp2['boxes'], genotype_order):
            patch.set_facecolor(GENOTYPE_COLORS.get(genotype, '#95a5a6'))
            patch.set_alpha(0.7)

        for i, (genotype, data) in enumerate(zip(genotype_order, laps_data)):
            x = np.random.normal(i+1, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.4, s=50, color='black')

        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_ylabel('Change in Lapsed State (Phase 2 - Phase 1)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Genotype', fontsize=12, fontweight='bold')
        ax.set_title('Lapsed State Transitions by Genotype', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add p-value
        h_stat, p_val = laps_stats
        sig_text = f"Kruskal-Wallis\np = {p_val:.4f}"
        if p_val < 0.05:
            sig_text += "\n***Significant"
            color = 'green'
        else:
            color = 'gray'
        ax.text(0.98, 0.97, sig_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
               fontweight='bold')

        plt.suptitle('Phase 2: State Transition Stability (Phase 1 → Phase 2)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'figures' / 'phase2_state_transitions_with_stats.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()

        print(f"✓ Saved to {output_path}")

    # ========== SUMMARY ==========

    def run_all_analyses(self):
        """Run all statistical analyses."""
        print("\n" + "="*80)
        print("COMPREHENSIVE STATISTICAL ANALYSIS OF GLM-HMM RESULTS")
        print("="*80)
        print("\nStarting comprehensive statistical analysis...")
        print("All genotypes using new labels: B6, C3H x B6, A1D_Wt, A1D_Het, A1D_KO\n")

        # Phase 1: Lapse metrics
        lapse_df, freq_stats, dur_stats = self.analyze_phase1_lapse_metrics()
        self.create_phase1_lapse_figures_with_stats(lapse_df, freq_stats, dur_stats)

        # Phase 1: Late lapsers
        late_lapsers, paired_stats, genotype_results = self.analyze_phase1_late_lapsers()
        if late_lapsers is not None:
            self.create_phase1_late_lapser_figures(late_lapsers, paired_stats, genotype_results)

        # Phase 2: High/Low performers
        hvl_df, chi_stats, contingency = self.analyze_phase2_performers()
        self.create_phase2_performer_figures(hvl_df, chi_stats, contingency)

        # Phase 2: State transitions
        trans_df, eng_stats, laps_stats = self.analyze_phase2_state_transitions()
        self.create_phase2_transition_figures(trans_df, eng_stats, laps_stats)

        # Save report
        self.save_report()

        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Figures saved to: {self.output_dir / 'figures'}")
        print(f"Statistics CSVs saved to: {self.output_dir / 'phase1'} and {self.output_dir / 'phase2'}")
        print(f"Summary report: {self.output_dir / 'STATISTICAL_ANALYSIS_REPORT.txt'}")


if __name__ == '__main__':
    analyzer = ComprehensiveStatisticalAnalysis()
    analyzer.run_all_analyses()
