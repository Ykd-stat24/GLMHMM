"""
Create poster-friendly summary visualizations for GLM-HMM analysis
showing both population-level patterns and individual heterogeneity
for within-cohort genotype comparisons and cross-cohort comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CohortSummaryVisualizer:
    """Create summary visualizations for GLM-HMM cohort analyses."""

    def __init__(self, results_dir='results/phase1_non_reversal'):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'summary_figures'
        self.output_dir.mkdir(exist_ok=True)

        # Define feature names
        self.feature_names = ['bias', 'prev_choice', 'wsls', 'session_prog',
                             'side_bias', 'task_stage', 'cum_exp']

    def load_cohort_results(self, cohort, animals):
        """Load all animal results for a cohort."""
        results = []

        for animal in animals:
            model_file = self.results_dir / f'{animal}_cohort{cohort}_model.pkl'
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                    results.append({
                        'animal_id': data['animal_id'],
                        'cohort': cohort,
                        'model': data['model'],
                        'state_metrics': data['state_metrics'],
                        'genotype': data.get('genotype', 'Unknown'),
                        'sex': data.get('sex', 'Unknown')
                    })

        return results

    def extract_weights_matrix(self, results):
        """Extract GLM weights into a matrix for visualization."""
        n_animals = len(results)
        n_states = results[0]['model'].n_states
        n_features = len(self.feature_names)

        # Shape: (animals, states, features)
        weights = np.zeros((n_animals, n_states, n_features))
        animal_ids = []
        genotypes = []

        for i, r in enumerate(results):
            weights[i] = r['model'].glm_weights
            animal_ids.append(r['animal_id'])
            genotypes.append(r['genotype'])

        return weights, animal_ids, genotypes

    def plot_weight_distributions_by_genotype(self, results, cohort):
        """
        Box/violin plots showing distribution of GLM weights by genotype.
        Shows heterogeneity and central tendency.
        """
        weights, animal_ids, genotypes = self.extract_weights_matrix(results)
        n_states = weights.shape[1]
        n_features = weights.shape[2]

        # Get unique genotypes
        unique_genotypes = sorted(set(genotypes))
        n_genotypes = len(unique_genotypes)

        # Create figure with subplots for each state
        fig, axes = plt.subplots(n_states, 1, figsize=(14, 4*n_states))
        if n_states == 1:
            axes = [axes]

        for state in range(n_states):
            ax = axes[state]

            # Prepare data for plotting
            plot_data = []
            for i, animal in enumerate(animal_ids):
                for j, feature in enumerate(self.feature_names):
                    plot_data.append({
                        'Animal': animal,
                        'Genotype': genotypes[i],
                        'Feature': feature,
                        'Weight': weights[i, state, j]
                    })

            df = pd.DataFrame(plot_data)

            # Create violin plot with individual points
            positions = []
            for g_idx, genotype in enumerate(unique_genotypes):
                for f_idx in range(n_features):
                    positions.append(g_idx * (n_features + 0.5) + f_idx)

            # Plot by genotype
            for g_idx, genotype in enumerate(unique_genotypes):
                genotype_data = df[df['Genotype'] == genotype]

                for f_idx, feature in enumerate(self.feature_names):
                    feature_data = genotype_data[genotype_data['Feature'] == feature]['Weight']

                    pos = g_idx * (n_features + 0.5) + f_idx

                    # Violin plot
                    parts = ax.violinplot([feature_data], positions=[pos],
                                         widths=0.7, showmeans=True, showmedians=True)

                    # Color by genotype
                    color = sns.color_palette("husl", n_genotypes)[g_idx]
                    for pc in parts['bodies']:
                        pc.set_facecolor(color)
                        pc.set_alpha(0.6)

                    # Overlay individual points
                    ax.scatter([pos]*len(feature_data), feature_data,
                             alpha=0.4, s=30, color=color, zorder=3)

            # Format axis
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylabel('GLM Weight', fontsize=12)
            ax.set_title(f'State {state} GLM Weights by Genotype (Cohort {cohort})',
                        fontsize=14, fontweight='bold')

            # Set x-ticks
            tick_positions = []
            tick_labels = []
            for g_idx, genotype in enumerate(unique_genotypes):
                for f_idx, feature in enumerate(self.feature_names):
                    tick_positions.append(g_idx * (n_features + 0.5) + f_idx)
                    if g_idx == 0:
                        tick_labels.append(feature)
                    else:
                        tick_labels.append('')

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')

            # Add genotype labels
            for g_idx, genotype in enumerate(unique_genotypes):
                center_pos = g_idx * (n_features + 0.5) + (n_features - 1) / 2
                ax.text(center_pos, ax.get_ylim()[0], genotype,
                       ha='center', va='top', fontsize=12, fontweight='bold',
                       transform=ax.get_xaxis_transform())

            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'cohort_{cohort}_weight_distributions.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f'cohort_{cohort}_weight_distributions.pdf',
                   bbox_inches='tight')
        plt.close()

        print(f"✓ Created weight distribution plot for cohort {cohort}")

    def plot_state_metrics_heatmap(self, results, cohort):
        """
        Heatmap showing all animals × all state metrics.
        Reveals patterns and clusters in behavioral strategies.
        """
        # Collect all metrics (using column names from DataFrame)
        metric_names = ['accuracy', 'wsls_ratio', 'side_bias', 'latency_cv',
                       'occupancy', 'dwell_mean']

        n_animals = len(results)
        n_states = results[0]['model'].n_states

        # Create data matrix: rows = animals, cols = state × metric
        data_matrix = []
        row_labels = []
        genotypes = []

        for r in results:
            row = []
            metrics_df = r['state_metrics']  # This is a DataFrame
            for state in range(n_states):
                state_data = metrics_df[metrics_df['state'] == state]
                for metric in metric_names:
                    if len(state_data) > 0 and metric in state_data.columns:
                        row.append(state_data[metric].values[0])
                    else:
                        row.append(0)
            data_matrix.append(row)
            row_labels.append(r['animal_id'])
            genotypes.append(r['genotype'])

        data_matrix = np.array(data_matrix)

        # Create column labels
        col_labels = []
        for state in range(n_states):
            for metric in metric_names:
                col_labels.append(f'S{state}_{metric}')

        # Handle NaN values - replace with column mean
        data_matrix_clean = np.copy(data_matrix)
        for j in range(data_matrix.shape[1]):
            col = data_matrix[:, j]
            col_mean = np.nanmean(col)
            if np.isnan(col_mean):
                col_mean = 0
            data_matrix_clean[np.isnan(data_matrix_clean[:, j]), j] = col_mean

        # Standardize for visualization
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data_matrix_clean)

        # Hierarchical clustering
        linkage = hierarchy.linkage(data_standardized, method='ward')
        dendro_order = hierarchy.dendrogram(linkage, no_plot=True)['leaves']

        # Reorder by clustering
        data_ordered = data_standardized[dendro_order, :]
        labels_ordered = [row_labels[i] for i in dendro_order]
        genotypes_ordered = [genotypes[i] for i in dendro_order]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10),
                                       gridspec_kw={'width_ratios': [20, 1]})

        # Heatmap
        im = ax1.imshow(data_ordered, aspect='auto', cmap='RdBu_r',
                       vmin=-2, vmax=2)

        ax1.set_xticks(range(len(col_labels)))
        ax1.set_xticklabels(col_labels, rotation=90, ha='right', fontsize=8)
        ax1.set_yticks(range(len(labels_ordered)))
        ax1.set_yticklabels(labels_ordered, fontsize=8)

        # Color code by genotype
        for i, (label, genotype) in enumerate(zip(labels_ordered, genotypes_ordered)):
            ax1.get_yticklabels()[i].set_color(
                'blue' if genotype == '+' else
                'red' if genotype == '-' else
                'green' if genotype == '+/+' else
                'orange' if genotype == '+/-' else
                'purple' if genotype == '-/-' else
                'black'
            )

        ax1.set_title(f'Cohort {cohort}: Hierarchically Clustered State Metrics\n' +
                     '(Animals × State Metrics, standardized)',
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('State Metrics', fontsize=12)
        ax1.set_ylabel('Animal ID (colored by genotype)', fontsize=12)

        # Add gridlines between states
        for i in range(1, n_states):
            ax1.axvline(x=i*len(metric_names)-0.5, color='white', linewidth=2)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Standardized Value', fontsize=12)

        # Genotype legend
        unique_genotypes = sorted(set(genotypes))
        legend_elements = []
        colors = {'+': 'blue', '-': 'red', '+/+': 'green',
                 '+/-': 'orange', '-/-': 'purple', 'Unknown': 'black'}
        for g in unique_genotypes:
            if g in colors:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                  markerfacecolor=colors[g],
                                                  markersize=8, label=g))
        ax1.legend(handles=legend_elements, loc='upper left',
                  bbox_to_anchor=(1.15, 1), fontsize=10)

        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'cohort_{cohort}_metrics_heatmap.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f'cohort_{cohort}_metrics_heatmap.pdf',
                   bbox_inches='tight')
        plt.close()

        print(f"✓ Created state metrics heatmap for cohort {cohort}")

    def plot_behavioral_space_pca(self, results, cohort):
        """
        PCA visualization showing animals in behavioral space.
        Points colored by genotype, shows clustering patterns.
        """
        # Collect metrics for PCA (using DataFrame column names)
        metric_names = ['accuracy', 'wsls_ratio', 'side_bias', 'latency_cv',
                       'occupancy', 'dwell_mean']

        n_states = results[0]['model'].n_states

        data_matrix = []
        animal_ids = []
        genotypes = []

        for r in results:
            row = []
            metrics_df = r['state_metrics']
            for state in range(n_states):
                state_data = metrics_df[metrics_df['state'] == state]
                for metric in metric_names:
                    if len(state_data) > 0 and metric in state_data.columns:
                        row.append(state_data[metric].values[0])
                    else:
                        row.append(0)

            # Also add summary of GLM weights (mean absolute value per feature)
            for j in range(r['model'].glm_weights.shape[1]):
                row.append(np.mean(np.abs(r['model'].glm_weights[:, j])))

            data_matrix.append(row)
            animal_ids.append(r['animal_id'])
            genotypes.append(r['genotype'])

        data_matrix = np.array(data_matrix)

        # Handle NaN values
        data_matrix_clean = np.copy(data_matrix)
        for j in range(data_matrix.shape[1]):
            col = data_matrix[:, j]
            col_mean = np.nanmean(col)
            if np.isnan(col_mean):
                col_mean = 0
            data_matrix_clean[np.isnan(data_matrix_clean[:, j]), j] = col_mean

        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_matrix_clean)

        # PCA
        pca = PCA()
        pca_coords = pca.fit_transform(data_scaled)

        # Plot first 3 PCs
        fig = plt.figure(figsize=(18, 6))

        # PC1 vs PC2
        ax1 = fig.add_subplot(131)
        unique_genotypes = sorted(set(genotypes))
        colors = sns.color_palette("husl", len(unique_genotypes))

        for i, genotype in enumerate(unique_genotypes):
            mask = np.array(genotypes) == genotype
            ax1.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                       c=[colors[i]], label=genotype, s=100, alpha=0.7,
                       edgecolors='black', linewidth=1)

            # Add animal labels
            for j, animal in enumerate(np.array(animal_ids)[mask]):
                ax1.annotate(animal, (pca_coords[mask, 0][j], pca_coords[mask, 1][j]),
                           fontsize=7, alpha=0.7)

        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax1.set_title('Behavioral Space: PC1 vs PC2', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # PC1 vs PC3
        ax2 = fig.add_subplot(132)
        for i, genotype in enumerate(unique_genotypes):
            mask = np.array(genotypes) == genotype
            ax2.scatter(pca_coords[mask, 0], pca_coords[mask, 2],
                       c=[colors[i]], label=genotype, s=100, alpha=0.7,
                       edgecolors='black', linewidth=1)

        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax2.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=12)
        ax2.set_title('Behavioral Space: PC1 vs PC3', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Scree plot
        ax3 = fig.add_subplot(133)
        variance_explained = pca.explained_variance_ratio_ * 100
        cumulative_variance = np.cumsum(variance_explained)

        ax3.bar(range(1, min(11, len(variance_explained)+1)),
               variance_explained[:10], alpha=0.6, label='Individual')
        ax3.plot(range(1, min(11, len(variance_explained)+1)),
                cumulative_variance[:10], 'ro-', linewidth=2, label='Cumulative')
        ax3.set_xlabel('Principal Component', fontsize=12)
        ax3.set_ylabel('Variance Explained (%)', fontsize=12)
        ax3.set_title('Scree Plot', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        fig.suptitle(f'Cohort {cohort}: Behavioral Space Analysis',
                    fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'cohort_{cohort}_behavioral_pca.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f'cohort_{cohort}_behavioral_pca.pdf',
                   bbox_inches='tight')
        plt.close()

        print(f"✓ Created behavioral space PCA for cohort {cohort}")

        # Return PCA for cross-cohort analysis
        return pca, scaler, data_matrix

    def plot_transition_matrices_by_genotype(self, results, cohort):
        """
        Average transition matrices by genotype with variability.
        Shows population-level patterns.
        """
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

        fig, axes = plt.subplots(1, n_genotypes, figsize=(6*n_genotypes, 5))
        if n_genotypes == 1:
            axes = [axes]

        for ax, genotype in zip(axes, unique_genotypes):
            # Collect transition matrices
            trans_matrices = []
            for r in genotype_groups[genotype]:
                # Estimate transition matrix from state sequence
                state_seq = r['model'].most_likely_states
                trans_mat = np.zeros((n_states, n_states))

                for i in range(len(state_seq) - 1):
                    s1 = state_seq[i]
                    s2 = state_seq[i+1]
                    trans_mat[s1, s2] += 1

                # Normalize
                row_sums = trans_mat.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                trans_mat = trans_mat / row_sums

                trans_matrices.append(trans_mat)

            # Average transition matrix
            avg_trans = np.mean(trans_matrices, axis=0)
            std_trans = np.std(trans_matrices, axis=0)

            # Plot heatmap
            im = ax.imshow(avg_trans, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')

            # Add text annotations with mean ± std
            for i in range(n_states):
                for j in range(n_states):
                    text_color = 'white' if avg_trans[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{avg_trans[i,j]:.2f}\n±{std_trans[i,j]:.2f}',
                           ha='center', va='center', color=text_color, fontsize=10)

            ax.set_xticks(range(n_states))
            ax.set_yticks(range(n_states))
            ax.set_xticklabels([f'S{i}' for i in range(n_states)])
            ax.set_yticklabels([f'S{i}' for i in range(n_states)])
            ax.set_xlabel('To State', fontsize=12)
            ax.set_ylabel('From State', fontsize=12)
            ax.set_title(f'{genotype} (n={len(genotype_groups[genotype])})',
                        fontsize=14, fontweight='bold')

            # Colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f'Cohort {cohort}: Population Transition Matrices by Genotype\n' +
                    '(Mean ± SD across animals)',
                    fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'cohort_{cohort}_transition_matrices.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f'cohort_{cohort}_transition_matrices.pdf',
                   bbox_inches='tight')
        plt.close()

        print(f"✓ Created transition matrix comparison for cohort {cohort}")

    def plot_state_occupancy_distributions(self, results, cohort):
        """
        Distribution of state occupancy by genotype.
        Shows heterogeneity in state usage.
        """
        n_states = results[0]['model'].n_states

        # Group by genotype
        genotype_groups = {}
        for r in results:
            g = r['genotype']
            if g not in genotype_groups:
                genotype_groups[g] = {s: [] for s in range(n_states)}

            metrics_df = r['state_metrics']
            for s in range(n_states):
                state_data = metrics_df[metrics_df['state'] == s]
                if len(state_data) > 0 and 'occupancy' in state_data.columns:
                    genotype_groups[g][s].append(state_data['occupancy'].values[0])

        unique_genotypes = sorted(genotype_groups.keys())

        # Create figure
        fig, axes = plt.subplots(1, n_states, figsize=(6*n_states, 5))
        if n_states == 1:
            axes = [axes]

        for state, ax in enumerate(axes):
            # Collect data for this state
            plot_data = []
            for genotype in unique_genotypes:
                for occupancy in genotype_groups[genotype][state]:
                    plot_data.append({
                        'Genotype': genotype,
                        'Occupancy': occupancy
                    })

            df = pd.DataFrame(plot_data)

            # Violin plot with swarm overlay
            sns.violinplot(data=df, x='Genotype', y='Occupancy', ax=ax,
                          palette='husl', inner=None, alpha=0.6)
            sns.swarmplot(data=df, x='Genotype', y='Occupancy', ax=ax,
                         color='black', alpha=0.5, size=6)

            ax.set_title(f'State {state} Occupancy', fontsize=14, fontweight='bold')
            ax.set_ylabel('Proportion of Trials', fontsize=12)
            ax.set_xlabel('Genotype', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

            # Add n per group
            for i, genotype in enumerate(unique_genotypes):
                n = len(genotype_groups[genotype][state])
                y_pos = ax.get_ylim()[1] * 0.95
                ax.text(i, y_pos, f'n={n}', ha='center', va='top',
                       fontsize=10, fontweight='bold')

        fig.suptitle(f'Cohort {cohort}: State Occupancy Distributions by Genotype',
                    fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'cohort_{cohort}_occupancy_distributions.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f'cohort_{cohort}_occupancy_distributions.pdf',
                   bbox_inches='tight')
        plt.close()

        print(f"✓ Created state occupancy distributions for cohort {cohort}")

    def plot_cross_cohort_comparison(self, results_W, results_F):
        """
        Cross-cohort comparison for WT animals (+ from W, +/+ from F).
        """
        # Filter for WT animals
        wt_W = [r for r in results_W if r['genotype'] == '+']
        wt_F = [r for r in results_F if r['genotype'] == '+/+']

        print(f"\nCross-cohort WT comparison:")
        print(f"  Cohort W (+): n={len(wt_W)}")
        print(f"  Cohort F (+/+): n={len(wt_F)}")

        if len(wt_W) == 0 or len(wt_F) == 0:
            print("  ⚠ Insufficient data for cross-cohort comparison")
            return

        # Combine for comparison
        combined_results = wt_W + wt_F

        # Add cohort label
        for r in wt_W:
            r['cohort_label'] = 'W (+)'
        for r in wt_F:
            r['cohort_label'] = 'F (+/+)'

        # Collect metrics
        metric_names = ['accuracy', 'wsls_ratio', 'side_bias', 'latency_cv']
        n_states = combined_results[0]['model'].n_states

        fig, axes = plt.subplots(n_states, len(metric_names),
                                figsize=(5*len(metric_names), 4*n_states))
        if n_states == 1:
            axes = axes.reshape(1, -1)

        for state in range(n_states):
            for m_idx, metric in enumerate(metric_names):
                ax = axes[state, m_idx]

                # Collect data
                plot_data = []
                for r in combined_results:
                    metrics_df = r['state_metrics']
                    state_data = metrics_df[metrics_df['state'] == state]
                    if len(state_data) > 0 and metric in state_data.columns:
                        plot_data.append({
                            'Cohort': r['cohort_label'],
                            'Value': state_data[metric].values[0],
                            'Animal': r['animal_id']
                        })

                df = pd.DataFrame(plot_data)

                # Box plot with points
                sns.boxplot(data=df, x='Cohort', y='Value', ax=ax, palette='Set2')
                sns.swarmplot(data=df, x='Cohort', y='Value', ax=ax,
                            color='black', alpha=0.5, size=5)

                # Statistical test
                w_vals = df[df['Cohort'] == 'W (+)']['Value']
                f_vals = df[df['Cohort'] == 'F (+/+)']['Value']

                if len(w_vals) > 0 and len(f_vals) > 0:
                    stat, pval = stats.mannwhitneyu(w_vals, f_vals, alternative='two-sided')

                    # Add p-value
                    y_max = ax.get_ylim()[1]
                    sig_str = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
                    ax.text(0.5, y_max * 0.95, f'p={pval:.3f} {sig_str}',
                           ha='center', va='top', fontsize=10,
                           transform=ax.get_xaxis_transform())

                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
                ax.set_xlabel('')
                if state == 0:
                    ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

                # Add state label on left
                if m_idx == 0:
                    ax.set_ylabel(f'State {state}\n' + ax.get_ylabel(), fontsize=11)

        fig.suptitle('Cross-Cohort Comparison: WT Animals (W+ vs F+/+)',
                    fontsize=16, fontweight='bold', y=1.01)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_cohort_WT_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'cross_cohort_WT_comparison.pdf',
                   bbox_inches='tight')
        plt.close()

        print("✓ Created cross-cohort WT comparison")

    def create_summary_statistics_table(self, results, cohort):
        """Create a summary table of key metrics by genotype."""
        # Group by genotype
        genotype_groups = {}
        for r in results:
            g = r['genotype']
            if g not in genotype_groups:
                genotype_groups[g] = []
            genotype_groups[g].append(r)

        # Compute summary statistics
        summary_rows = []

        for genotype in sorted(genotype_groups.keys()):
            group = genotype_groups[genotype]
            n = len(group)

            n_states = group[0]['model'].n_states

            for state in range(n_states):
                # Collect metrics from DataFrame
                accuracies = []
                wsls_ratios = []
                side_biases = []
                occupancies = []
                dwell_times = []

                for r in group:
                    metrics_df = r['state_metrics']
                    state_data = metrics_df[metrics_df['state'] == state]
                    if len(state_data) > 0:
                        accuracies.append(state_data['accuracy'].values[0] if 'accuracy' in state_data.columns else np.nan)
                        wsls_ratios.append(state_data['wsls_ratio'].values[0] if 'wsls_ratio' in state_data.columns else np.nan)
                        side_biases.append(state_data['side_bias'].values[0] if 'side_bias' in state_data.columns else np.nan)
                        occupancies.append(state_data['occupancy'].values[0] if 'occupancy' in state_data.columns else np.nan)
                        dwell_times.append(state_data['dwell_mean'].values[0] if 'dwell_mean' in state_data.columns else np.nan)

                summary_rows.append({
                    'Cohort': cohort,
                    'Genotype': genotype,
                    'N': n,
                    'State': state,
                    'Accuracy_Mean': np.nanmean(accuracies),
                    'Accuracy_SD': np.nanstd(accuracies),
                    'WSLS_Mean': np.nanmean(wsls_ratios),
                    'WSLS_SD': np.nanstd(wsls_ratios),
                    'SideBias_Mean': np.nanmean(side_biases),
                    'SideBias_SD': np.nanstd(side_biases),
                    'Occupancy_Mean': np.nanmean(occupancies),
                    'Occupancy_SD': np.nanstd(occupancies),
                    'DwellTime_Mean': np.nanmean(dwell_times),
                    'DwellTime_SD': np.nanstd(dwell_times)
                })

        df = pd.DataFrame(summary_rows)

        # Save to CSV
        csv_path = self.output_dir / f'cohort_{cohort}_summary_statistics.csv'
        df.to_csv(csv_path, index=False, float_format='%.3f')

        print(f"✓ Created summary statistics table: {csv_path}")

        return df


def main():
    """Run all summary visualizations."""
    print("="*80)
    print("Creating Cohort Summary Visualizations")
    print("="*80)

    viz = CohortSummaryVisualizer()

    # Define cohorts
    animals_W = [f'c{c}m{m}' for c in range(1, 5) for m in range(1, 6)
                 if not (c == 1 and m == 5)]  # 19 animals
    animals_F = [str(i) for i in [11, 12, 13, 14,  # c1
                                   21, 22, 23, 24, 25,  # c2
                                   31, 32, 33, 34,  # c3
                                   41, 42,  # c4
                                   51, 52,  # c5
                                   61, 62, 63, 64,  # c6
                                   71, 72, 73,  # c7
                                   81, 82, 83, 84,  # c8
                                   91, 92, 93,  # c9
                                   101, 102, 103, 104]]  # c10 (35 animals, 105 excluded)

    print(f"\nCohort W: {len(animals_W)} animals")
    print(f"Cohort F: {len(animals_F)} animals")

    # Load results
    print("\n" + "="*80)
    print("Loading Cohort W Results...")
    print("="*80)
    results_W = viz.load_cohort_results('W', animals_W)
    print(f"✓ Loaded {len(results_W)} animals")

    print("\n" + "="*80)
    print("Loading Cohort F Results...")
    print("="*80)
    results_F = viz.load_cohort_results('F', animals_F)
    print(f"✓ Loaded {len(results_F)} animals")

    if len(results_W) == 0 and len(results_F) == 0:
        print("\n❌ No results found! Please run the analysis first.")
        return

    # Create visualizations for each cohort
    if len(results_W) > 0:
        print("\n" + "="*80)
        print("Creating Cohort W Visualizations...")
        print("="*80)

        viz.plot_weight_distributions_by_genotype(results_W, 'W')
        viz.plot_state_metrics_heatmap(results_W, 'W')
        viz.plot_behavioral_space_pca(results_W, 'W')
        viz.plot_transition_matrices_by_genotype(results_W, 'W')
        viz.plot_state_occupancy_distributions(results_W, 'W')
        viz.create_summary_statistics_table(results_W, 'W')

    if len(results_F) > 0:
        print("\n" + "="*80)
        print("Creating Cohort F Visualizations...")
        print("="*80)

        viz.plot_weight_distributions_by_genotype(results_F, 'F')
        viz.plot_state_metrics_heatmap(results_F, 'F')
        viz.plot_behavioral_space_pca(results_F, 'F')
        viz.plot_transition_matrices_by_genotype(results_F, 'F')
        viz.plot_state_occupancy_distributions(results_F, 'F')
        viz.create_summary_statistics_table(results_F, 'F')

    # Cross-cohort comparison
    if len(results_W) > 0 and len(results_F) > 0:
        print("\n" + "="*80)
        print("Creating Cross-Cohort Comparison...")
        print("="*80)
        viz.plot_cross_cohort_comparison(results_W, results_F)

    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {viz.output_dir}")
    print("\nGenerated files:")
    for f in sorted(viz.output_dir.glob('*')):
        print(f"  - {f.name}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
