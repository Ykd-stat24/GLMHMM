"""
Comprehensive Individual-Level Analysis of GLM-HMM Phase 1 and Phase 1→Phase 2 Predictability

This script performs multi-layered analysis:
1. State transition matrix extraction and statistical comparison by genotype
2. Individual behavioral profiles linked to existing visualizations
3. Phase 1 → Phase 2 predictability with performance reversals
4. A1D_KO mixed state deep characterization
"""

import os
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import xlogy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Setup
PHASE1_DIR = '/home/user/GLMHMM/results/phase1_non_reversal'
PHASE2_DIR = '/home/user/GLMHMM/results/phase2_reversal'
OUTPUT_DIR = '/home/user/GLMHMM/results/comprehensive_individual_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Genotype mapping
OLD_TO_NEW = {
    '+/+': 'A1D_Wt', '-/-': 'A1D_KO', '+/-': 'A1D_Het',
    '+': 'B6', '-': 'C3H x B6'
}

# =============================================================================
# PART 1: EXTRACT TRANSITION MATRICES AND STATE CHARACTERISTICS
# =============================================================================

class TransitionAnalyzer:
    """Extract and analyze state transitions from Phase 1 GLM-HMM models"""

    def __init__(self):
        self.transition_data = []
        self.state_characteristics = {}

    def extract_all_transitions(self):
        """Extract transition matrices from all Phase 1 model pickle files"""
        phase1_models = sorted(glob.glob(os.path.join(PHASE1_DIR, '*_model.pkl')))

        print(f"Found {len(phase1_models)} Phase 1 models")

        for pkl_path in phase1_models:
            try:
                # Extract animal ID from filename
                filename = os.path.basename(pkl_path)
                animal_id = filename.replace('_model.pkl', '')

                with open(pkl_path, 'rb') as f:
                    model_dict = pickle.load(f)

                # Extract transition probabilities (already computed in pickle)
                trans_probs = model_dict['transition_metrics']['transition_probs']

                # Extract genotype
                genotype = model_dict.get('genotype', '')
                genotype_new = OLD_TO_NEW.get(genotype, genotype)

                # Extract state occupancy from trajectory_df (bout-level data)
                traj_df = model_dict['trajectory_df']
                if 'state' in traj_df.columns:
                    states_in_bout = traj_df['state'].values
                    bout_lengths = traj_df['bout_length'].values

                    # Calculate occupancy weighted by bout length
                    total_trials = np.sum(bout_lengths)
                    occupancy = np.zeros(3)
                    for state in range(3):
                        state_mask = states_in_bout == state
                        occupancy[state] = np.sum(bout_lengths[state_mask]) / total_trials
                else:
                    occupancy = None

                # Extract behavioral metrics
                metrics = {}
                if 'during_accuracy' in traj_df.columns:
                    for state in range(3):
                        state_mask = traj_df['state'] == state
                        if state_mask.sum() > 0:
                            metrics[f'accuracy_state{state}'] = traj_df.loc[state_mask, 'during_accuracy'].mean()

                if 'during_latency_mean' in traj_df.columns:
                    for state in range(3):
                        state_mask = traj_df['state'] == state
                        if state_mask.sum() > 0:
                            metrics[f'latency_state{state}'] = traj_df.loc[state_mask, 'during_latency_mean'].mean()

                # Store transition matrix for later analysis
                self.transition_data.append({
                    'animal_id': animal_id,
                    'genotype': genotype_new,
                    'trans_matrix': trans_probs,
                    'occupancy': occupancy,
                    'n_trials': model_dict.get('n_trials', total_trials),
                    'n_bouts': len(traj_df),
                    'trajectory_df': traj_df,
                    'metrics': metrics,
                    'model_dict': model_dict
                })

            except Exception as e:
                print(f"Error processing {pkl_path}: {e}")
                continue

        print(f"Successfully extracted {len(self.transition_data)} transition matrices")
        return self.transition_data

    def compute_transition_entropy(self):
        """Compute transition entropy: measure of how committed states are"""
        entropy_data = []

        for trans in self.transition_data:
            animal_id = trans['animal_id']
            trans_mat = trans['trans_matrix']

            # Entropy per state: -sum(p_ij * log(p_ij))
            entropies = []
            for row in trans_mat:
                ent = -np.sum(xlogy(row, row))
                entropies.append(ent)

            # Mean entropy (higher = less committed, more exploratory)
            mean_entropy = np.mean(entropies)
            max_entropy = np.max(entropies)

            # Diagonal dominance (self-transition strength)
            diag_mean = np.mean(np.diag(trans_mat))

            entropy_data.append({
                'animal_id': animal_id,
                'genotype': trans['genotype'],
                'mean_entropy': mean_entropy,
                'max_entropy': max_entropy,
                'diag_dominance': diag_mean,
                'state_entropies': entropies
            })

        self.entropy_df = pd.DataFrame(entropy_data)
        return self.entropy_df

    def compute_transition_stability(self):
        """Compute stability: persistence of transitioning to same state"""
        stability_data = []

        for trans in self.transition_data:
            animal_id = trans['animal_id']
            trans_mat = trans['trans_matrix']

            # Self-transition probabilities (stability = staying in same state)
            self_trans = np.diag(trans_mat)

            # Off-diagonal transitions (switching behavior)
            off_diag = trans_mat.copy()
            np.fill_diagonal(off_diag, 0)
            switching_prob = np.nanmean(off_diag, axis=1)

            stability_data.append({
                'animal_id': animal_id,
                'genotype': trans['genotype'],
                'self_transition_engaged': self_trans[0],
                'self_transition_biased': self_trans[1],
                'self_transition_lapsed': self_trans[2],
                'switching_prob_engaged': switching_prob[0],
                'switching_prob_biased': switching_prob[1],
                'switching_prob_lapsed': switching_prob[2]
            })

        self.stability_df = pd.DataFrame(stability_data)
        return self.stability_df

    def analyze_transition_differences_by_genotype(self):
        """Statistical tests for transition differences across genotypes"""
        results = {}
        state_names = ['Engaged', 'Biased', 'Lapsed']

        # Test each transition probability across genotypes
        for state_from in range(3):
            for state_to in range(3):
                trans_probs_by_genotype = {}
                for trans in self.transition_data:
                    geno = trans['genotype']
                    prob = trans['trans_matrix'][state_from, state_to]

                    if geno not in trans_probs_by_genotype:
                        trans_probs_by_genotype[geno] = []
                    trans_probs_by_genotype[geno].append(prob)

                # Only test if multiple genotypes have data
                if len(trans_probs_by_genotype) > 1:
                    groups = [np.array(probs) for probs in trans_probs_by_genotype.values()]
                    if all(len(g) > 0 for g in groups):
                        h_stat, p_val = stats.kruskal(*groups)

                        results[f'trans_{state_names[state_from]}_to_{state_names[state_to]}'] = {
                            'h_stat': h_stat,
                            'p_value': p_val,
                            'by_genotype': {g: np.mean(probs) for g, probs in trans_probs_by_genotype.items()}
                        }

        self.transition_test_results = results
        return results


# =============================================================================
# PART 2: BUILD COMPREHENSIVE BEHAVIORAL PROFILES
# =============================================================================

class BehavioralProfiler:
    """Extract and analyze behavioral characteristics per animal per state"""

    def __init__(self, transition_analyzer):
        self.trans_analyzer = transition_analyzer
        self.behavioral_profiles = []
        self.phase1_data = self._load_phase1_data()
        self.phase2_data = self._load_phase2_data()

    def _load_phase1_data(self):
        """Load Phase 1 lapse statistics"""
        try:
            lapse_file = os.path.join(PHASE1_DIR, 'priority2_interpretation', 'lapse_statistics.csv')
            df = pd.read_csv(lapse_file)
            # Map genotypes
            df['genotype'] = df['genotype'].map(OLD_TO_NEW).fillna(df['genotype'])
            return df
        except:
            print("Warning: Could not load Phase 1 lapse data")
            return pd.DataFrame()

    def _load_phase2_data(self):
        """Load Phase 2 performer classification"""
        try:
            perf_file = os.path.join(PHASE2_DIR, 'detailed_analyses', 'high_vs_low_performers.csv')
            df = pd.read_csv(perf_file)
            return df
        except:
            print("Warning: Could not load Phase 2 performer data")
            return pd.DataFrame()

    def build_profiles(self):
        """Build comprehensive behavioral profile per animal"""
        for trans in self.trans_analyzer.transition_data:
            animal_id = trans['animal_id']
            genotype = trans['genotype']

            profile = {
                'animal_id': animal_id,
                'genotype': genotype,
                'n_trials': trans['n_trials'],
                'n_bouts': trans['n_bouts']
            }

            # Extract state occupancy
            if trans['occupancy'] is not None:
                occupancy = trans['occupancy']
                profile['occupancy_engaged'] = occupancy[0]
                profile['occupancy_biased'] = occupancy[1]
                profile['occupancy_lapsed'] = occupancy[2]

            # Extract behavioral metrics from trajectory_df
            traj_df = trans['trajectory_df']

            # Accuracy by state
            for state in range(3):
                state_mask = traj_df['state'] == state
                if state_mask.sum() > 0:
                    if 'during_accuracy' in traj_df.columns:
                        profile[f'accuracy_state{state}'] = traj_df.loc[state_mask, 'during_accuracy'].mean()
                    if 'during_latency_mean' in traj_df.columns:
                        profile[f'latency_state{state}'] = traj_df.loc[state_mask, 'during_latency_mean'].mean()

            # Overall metrics
            if 'during_accuracy' in traj_df.columns:
                profile['overall_accuracy'] = traj_df['during_accuracy'].mean()
            if 'during_latency_mean' in traj_df.columns:
                profile['overall_latency'] = traj_df['during_latency_mean'].mean()

            # Bout duration statistics
            for state in range(3):
                state_bouts = traj_df[traj_df['state'] == state]['bout_length']
                if len(state_bouts) > 0:
                    profile[f'mean_bout_length_state{state}'] = state_bouts.mean()
                    profile[f'std_bout_length_state{state}'] = state_bouts.std()

            # Link to lapse statistics from Phase 1
            if not self.phase1_data.empty:
                phase1_match = self.phase1_data[self.phase1_data['animal_id'].astype(str) == str(animal_id)]
                if not phase1_match.empty:
                    profile['lapse_frequency'] = phase1_match['lapse_frequency'].values[0]
                    profile['avg_lapse_duration'] = phase1_match['avg_lapse_duration'].values[0]

            # Link to Phase 2 performer status
            if not self.phase2_data.empty:
                phase2_match = self.phase2_data[self.phase2_data['animal_id'].astype(str) == str(animal_id)]
                if not phase2_match.empty:
                    profile['phase2_performer'] = phase2_match['performer_group'].values[0]
                    profile['phase2_accuracy'] = phase2_match['accuracy'].values[0]
                    if 'engaged_occ' in phase2_match.columns:
                        profile['phase2_engaged_occ'] = phase2_match['engaged_occ'].values[0]
                    if 'lapsed_occ' in phase2_match.columns:
                        profile['phase2_lapsed_occ'] = phase2_match['lapsed_occ'].values[0]

            self.behavioral_profiles.append(profile)

        self.profiles_df = pd.DataFrame(self.behavioral_profiles)
        return self.profiles_df


# =============================================================================
# PART 3: PHASE 1 → PHASE 2 PREDICTABILITY ANALYSIS
# =============================================================================

class PredictabilityAnalyzer:
    """Analyze whether Phase 1 metrics predict Phase 2 performance"""

    def __init__(self, behavioral_profiler):
        self.profiler = behavioral_profiler
        self.prediction_data = None
        self.auc = None
        self.reversal_summary = None

    def prepare_prediction_data(self):
        """Combine Phase 1 and Phase 2 data for predictive modeling"""
        df = self.profiler.profiles_df.copy()

        # Only keep animals with both Phase 1 and Phase 2 data
        if 'phase2_performer' in df.columns:
            df = df.dropna(subset=['phase2_performer'])
        else:
            print("Warning: No phase2_performer column found")
            self.prediction_data = df.iloc[:0]  # Return empty dataframe
            return df.iloc[:0]

        # Create binary target: High=1, Low=0
        df['phase2_high_performer'] = (df['phase2_performer'] == 'High Performer').astype(int)

        # Identify reversals
        # Phase 1 performance proxy: accuracy or engagement
        if 'overall_accuracy' in df.columns:
            p1_median = df['overall_accuracy'].median()
            df['phase1_high'] = (df['overall_accuracy'] >= p1_median).astype(int)
            df['is_reversal'] = (df['phase1_high'] != df['phase2_high_performer']).astype(int)

        self.prediction_data = df
        return df

    def fit_predictive_model(self):
        """Fit logistic regression to predict Phase 2 performance from Phase 1"""
        if self.prediction_data is None:
            self.prepare_prediction_data()

        df = self.prediction_data.copy()

        if df.empty or 'phase2_high_performer' not in df.columns:
            print("Warning: No prediction data available (Phase 1-2 overlap insufficient)")
            self.auc = None
            return None

        # Select Phase 1 features
        feature_cols = [
            'occupancy_engaged', 'occupancy_biased', 'occupancy_lapsed',
            'overall_accuracy', 'lapse_frequency'
        ]

        # Filter to available features
        feature_cols = [c for c in feature_cols if c in df.columns]
        df = df[feature_cols + ['phase2_high_performer']].dropna()

        if len(df) < 5:
            print("Warning: Not enough animals with complete data for prediction model")
            self.auc = None
            return None

        X = df[feature_cols].values
        y = df['phase2_high_performer'].values

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit logistic regression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)

        # Get predictions and probabilities
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]

        # Calculate metrics
        auc = roc_auc_score(y, y_pred_proba)
        fpr, tpr, _ = roc_curve(y, y_pred_proba)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)

        self.model = model
        self.scaler = scaler
        self.auc = auc
        self.fpr = fpr
        self.tpr = tpr
        self.feature_importance = feature_importance
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.X = X
        self.y = y

        return {
            'auc': auc,
            'feature_importance': feature_importance,
            'confusion_matrix': confusion_matrix(y, y_pred)
        }

    def analyze_reversals(self):
        """Identify and characterize Phase 1→Phase 2 reversals"""
        if self.prediction_data is None:
            self.prepare_prediction_data()

        df = self.prediction_data.copy()

        if 'is_reversal' in df.columns:
            reversals = df[df['is_reversal'] == 1]
            non_reversals = df[df['is_reversal'] == 0]

            reversal_summary = {
                'total_reversals': len(reversals),
                'total_consistent': len(non_reversals),
                'reversal_rate': len(reversals) / len(df) if len(df) > 0 else 0,
                'reversals_by_genotype': reversals['genotype'].value_counts().to_dict() if len(reversals) > 0 else {}
            }

            self.reversals = reversals
            self.reversal_summary = reversal_summary
            return reversal_summary

        return None


# =============================================================================
# PART 4: A1D_KO MIXED STATE CHARACTERIZATION
# =============================================================================

class A1DKOAnalyzer:
    """Deep characterization of A1D_KO mixed state behavior"""

    def __init__(self, behavioral_profiler):
        self.profiler = behavioral_profiler
        self.ko_animals = None
        self.mixed_summary = {'n_ko_animals': 0, 'n_mixed': 0, 'pct_mixed': 0, 'median_state_balance': 0}
        self.bias_analysis = pd.DataFrame()
        self.genotype_comparison = pd.DataFrame()

    def characterize_mixed_state(self):
        """Quantify occupancy of mixed vs pure states in A1D_KO"""
        df = self.profiler.profiles_df.copy()
        self.ko_animals = df[df['genotype'] == 'A1D_KO'].copy()

        if len(self.ko_animals) == 0:
            print("No A1D_KO animals found")
            return None

        # Define "mixed state" as relatively balanced occupancy
        # Calculate state balance: 1 - max_occupancy gives measure of balance
        occupancy_cols = ['occupancy_engaged', 'occupancy_biased', 'occupancy_lapsed']

        self.ko_animals['max_occupancy'] = self.ko_animals[occupancy_cols].max(axis=1)
        self.ko_animals['state_balance'] = 1 - self.ko_animals['max_occupancy']

        # Classify as "mixed" if state balance > median
        median_balance = self.ko_animals['state_balance'].median()
        self.ko_animals['is_mixed'] = (self.ko_animals['state_balance'] > median_balance).astype(int)

        mixed_summary = {
            'n_ko_animals': len(self.ko_animals),
            'n_mixed': self.ko_animals['is_mixed'].sum(),
            'pct_mixed': 100 * self.ko_animals['is_mixed'].mean() if len(self.ko_animals) > 0 else 0,
            'median_state_balance': median_balance,
        }

        if self.ko_animals['is_mixed'].sum() > 0:
            mixed_animals = self.ko_animals[self.ko_animals['is_mixed'] == 1]
            pure_animals = self.ko_animals[self.ko_animals['is_mixed'] == 0]
            if len(mixed_animals) > 0:
                mixed_summary['mixed_occupancy_profile'] = mixed_animals[occupancy_cols].describe().to_dict()
            if len(pure_animals) > 0:
                mixed_summary['pure_occupancy_profile'] = pure_animals[occupancy_cols].describe().to_dict()

        self.mixed_summary = mixed_summary
        return mixed_summary

    def detect_bias_types(self):
        """Characterize if 'mixed' state is actually a side/response bias"""
        analysis = {}

        for animal_id in self.ko_animals['animal_id'].unique():
            trans = next((t for t in self.profiler.trans_analyzer.transition_data
                         if t['animal_id'] == animal_id), None)
            if trans is None:
                continue

            trans_mat = trans['trans_matrix']

            # Analyze transition patterns for bias signatures
            is_mixed = self.ko_animals[self.ko_animals['animal_id'] == animal_id]['is_mixed'].values
            is_mixed = is_mixed[0] if len(is_mixed) > 0 else 0

            analysis[animal_id] = {
                'is_mixed': is_mixed,
                'biased_self_transition': trans_mat[1, 1],
                'engaged_to_biased': trans_mat[0, 1],
                'biased_to_lapsed': trans_mat[1, 2],
                'lapsed_to_biased': trans_mat[2, 1],
            }

        self.bias_analysis = pd.DataFrame(analysis).T
        return self.bias_analysis

    def compare_with_other_genotypes(self):
        """Compare A1D_KO behavioral features with other genotypes"""
        df = self.profiler.profiles_df.copy()

        comparison = {}
        occupancy_cols = ['occupancy_engaged', 'occupancy_biased', 'occupancy_lapsed']

        for genotype in sorted(df['genotype'].unique()):
            geno_data = df[df['genotype'] == genotype][occupancy_cols]
            if len(geno_data) > 0:
                comparison[genotype] = {
                    'n_animals': len(geno_data),
                    'mean_engaged': geno_data['occupancy_engaged'].mean(),
                    'mean_biased': geno_data['occupancy_biased'].mean(),
                    'mean_lapsed': geno_data['occupancy_lapsed'].mean(),
                    'std_engaged': geno_data['occupancy_engaged'].std(),
                    'std_biased': geno_data['occupancy_biased'].std(),
                    'std_lapsed': geno_data['occupancy_lapsed'].std()
                }

        self.genotype_comparison = pd.DataFrame(comparison).T
        return self.genotype_comparison


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_transition_heatmaps(trans_analyzer, output_dir):
    """Create transition probability heatmaps by genotype"""
    genotypes = sorted(set([t['genotype'] for t in trans_analyzer.transition_data]))
    state_names = ['Engaged', 'Biased', 'Lapsed']

    ncols = 3
    nrows = (len(genotypes) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    axes = axes.flatten() if len(genotypes) > 1 else [axes]

    for idx, genotype in enumerate(genotypes):
        trans_matrices = [t['trans_matrix'] for t in trans_analyzer.transition_data
                         if t['genotype'] == genotype]

        if len(trans_matrices) == 0:
            continue

        # Average transition matrix for genotype
        avg_trans = np.mean(trans_matrices, axis=0)

        # Standard deviation
        std_trans = np.std(trans_matrices, axis=0)

        ax = axes[idx]
        im = ax.imshow(avg_trans, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = ax.text(j, i, f'{avg_trans[i, j]:.2f}\n±{std_trans[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)

        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(state_names)
        ax.set_yticklabels(state_names)
        ax.set_title(f'{genotype} (n={len(trans_matrices)})')
        ax.set_xlabel('To State')
        ax.set_ylabel('From State')
        plt.colorbar(im, ax=ax, label='Transition Prob')

    # Remove empty subplots
    for idx in range(len(genotypes), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transition_matrices_by_genotype.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'transition_matrices_by_genotype.pdf'), bbox_inches='tight')
    plt.close()

    print("✓ Saved transition heatmaps")


def create_entropy_stability_plots(trans_analyzer, output_dir):
    """Visualize transition entropy and stability by genotype"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    entropy_df = trans_analyzer.entropy_df
    stability_df = trans_analyzer.stability_df

    # Plot 1: Entropy
    sns.boxplot(data=entropy_df, x='genotype', y='mean_entropy', ax=axes[0], palette='Set2')
    axes[0].set_title('Transition Entropy by Genotype\n(Higher = Less Committed to Single State)')
    axes[0].set_ylabel('Mean Entropy')
    axes[0].set_xlabel('Genotype')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Diagonal dominance
    sns.boxplot(data=entropy_df, x='genotype', y='diag_dominance', ax=axes[1], palette='Set2')
    axes[1].set_title('Self-Transition Strength by Genotype\n(Higher = More Stable States)')
    axes[1].set_ylabel('Self-Transition Probability')
    axes[1].set_xlabel('Genotype')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Occupancy distribution
    occupancy_cols = ['occupancy_engaged', 'occupancy_biased', 'occupancy_lapsed']
    profiles_df = trans_analyzer.transition_data
    profiles_list = []
    for trans in profiles_df:
        if trans['occupancy'] is not None:
            profiles_list.append({
                'genotype': trans['genotype'],
                'engaged': trans['occupancy'][0],
                'biased': trans['occupancy'][1],
                'lapsed': trans['occupancy'][2]
            })

    if profiles_list:
        profiles_plot_df = pd.DataFrame(profiles_list)
        sns.boxplot(data=profiles_plot_df, x='genotype', y='biased', ax=axes[2], palette='Set2')
        axes[2].set_title('Biased State Occupancy by Genotype')
        axes[2].set_ylabel('Occupancy')
        axes[2].set_xlabel('Genotype')
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_stability_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'entropy_stability_analysis.pdf'), bbox_inches='tight')
    plt.close()

    print("✓ Saved entropy and stability plots")


def create_behavioral_profile_heatmap(profiler, output_dir):
    """Heatmap of behavioral characteristics across animals and genotypes"""
    df = profiler.profiles_df.copy()

    # Select key behavioral metrics
    metrics = ['occupancy_engaged', 'occupancy_biased', 'occupancy_lapsed',
              'overall_accuracy', 'lapse_frequency']

    metrics = [m for m in metrics if m in df.columns]

    if len(metrics) == 0:
        print("No metrics available for behavioral profile heatmap")
        return

    # Create heatmap per genotype
    genotypes = sorted(df['genotype'].unique())

    fig, axes = plt.subplots(1, len(genotypes), figsize=(5*len(genotypes), 8))
    if len(genotypes) == 1:
        axes = [axes]

    for idx, genotype in enumerate(genotypes):
        geno_data = df[df['genotype'] == genotype][['animal_id'] + metrics].set_index('animal_id')
        geno_data = geno_data.dropna(how='all')

        if len(geno_data) == 0:
            continue

        # Standardize for better visualization
        geno_data_std = (geno_data - geno_data.mean()) / (geno_data.std() + 1e-6)

        sns.heatmap(geno_data_std.T, cmap='RdBu_r', center=0, ax=axes[idx],
                   cbar_kws={'label': 'Standardized Value'}, vmin=-2, vmax=2)
        axes[idx].set_title(f'{genotype}\n(n={len(geno_data)} animals)')
        axes[idx].set_xlabel('Animal ID')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'behavioral_profile_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'behavioral_profile_heatmap.pdf'), bbox_inches='tight')
    plt.close()

    print("✓ Saved behavioral profile heatmap")


def create_predictability_plots(analyzer, output_dir):
    """ROC curve and feature importance for Phase 1→Phase 2 prediction"""
    if analyzer.auc is None:
        print("Skipping predictability plots - model not fitted")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    axes[0].plot(analyzer.fpr, analyzer.tpr, linewidth=2.5, label=f'ROC Curve (AUC={analyzer.auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')
    axes[0].fill_between(analyzer.fpr, analyzer.tpr, alpha=0.3)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Phase 1 → Phase 2 Performance Prediction\nROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Feature Importance
    feat_imp = analyzer.feature_importance.sort_values('coefficient')
    colors = ['red' if x < 0 else 'blue' for x in feat_imp['coefficient']]
    axes[1].barh(feat_imp['feature'], feat_imp['coefficient'], color=colors)
    axes[1].set_xlabel('Logistic Regression Coefficient')
    axes[1].set_title('Feature Importance for Phase 2 Prediction')
    axes[1].axvline(0, color='black', linestyle='-', linewidth=0.8)
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase1_to_phase2_predictability.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'phase1_to_phase2_predictability.pdf'), bbox_inches='tight')
    plt.close()

    print("✓ Saved predictability plots")


def create_reversal_analysis_plot(analyzer, output_dir):
    """Visualize Phase 1→Phase 2 reversals"""
    if analyzer.prediction_data is None:
        print("Skipping reversal plots - no prediction data")
        return

    df = analyzer.prediction_data.copy()

    if 'is_reversal' not in df.columns:
        print("No reversal data available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Reversal rate by genotype
    reversal_by_geno = df.groupby('genotype')['is_reversal'].agg(['sum', 'count'])
    reversal_by_geno['rate'] = reversal_by_geno['sum'] / reversal_by_geno['count']

    ax = axes[0, 0]
    reversal_by_geno['rate'].plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Phase 1→Phase 2 Reversal Rate by Genotype')
    ax.set_ylabel('Reversal Rate')
    ax.set_xlabel('Genotype')
    ax.set_ylim([0, 1])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Phase 1 vs Phase 2 performance scatter
    ax = axes[0, 1]
    reversals = df[df['is_reversal'] == 1]
    consistent = df[df['is_reversal'] == 0]

    if len(consistent) > 0:
        ax.scatter(consistent['overall_accuracy'], consistent['phase2_accuracy'],
                  alpha=0.6, s=100, label='Consistent', color='green')
    if len(reversals) > 0:
        ax.scatter(reversals['overall_accuracy'], reversals['phase2_accuracy'],
                  alpha=0.6, s=100, label='Reversals', color='red', marker='X')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Phase 1 Accuracy')
    ax.set_ylabel('Phase 2 Accuracy')
    ax.set_title('Phase 1 vs Phase 2 Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Engagement changes
    ax = axes[1, 0]
    if 'occupancy_engaged' in df.columns and 'phase2_engaged_occ' in df.columns:
        df['engagement_change'] = df['phase2_engaged_occ'] - df['occupancy_engaged']
        reversals_eng = reversals['engagement_change'].dropna()
        consistent_eng = consistent['engagement_change'].dropna()

        bp = ax.boxplot([consistent_eng, reversals_eng], labels=['Consistent', 'Reversals'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['green', 'red']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('Engagement Change (Phase 2 - Phase 1)')
        ax.set_title('State Engagement Change by Reversal Status')
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')

    # Reversal counts
    ax = axes[1, 1]
    reversal_counts = df['is_reversal'].value_counts()
    colors_pie = ['green', 'red']
    ax.pie(reversal_counts, labels=['Consistent', 'Reversals'], autopct='%1.1f%%',
          colors=colors_pie, startangle=90)
    ax.set_title(f'Overall Reversal Rate\n(n={len(df)} animals)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase1_phase2_reversals.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'phase1_phase2_reversals.pdf'), bbox_inches='tight')
    plt.close()

    print("✓ Saved reversal analysis plots")


def create_ko_analysis_plots(ko_analyzer, output_dir):
    """Visualize A1D_KO mixed state characterization"""
    if len(ko_analyzer.ko_animals) == 0:
        print("Skipping A1D_KO plots - no KO animals found")
        return

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: State occupancy distribution
    ax1 = fig.add_subplot(gs[0, :2])
    occupancy_cols = ['occupancy_engaged', 'occupancy_biased', 'occupancy_lapsed']
    occupancy_data = ko_analyzer.ko_animals[occupancy_cols].values

    box_data = [occupancy_data[:, 0], occupancy_data[:, 1], occupancy_data[:, 2]]
    bp = ax1.boxplot(box_data, labels=['Engaged', 'Biased', 'Lapsed'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#d62728']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.set_ylabel('State Occupancy')
    ax1.set_title('A1D_KO: State Occupancy Distribution')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Mixed vs Pure classification
    ax2 = fig.add_subplot(gs[0, 2])
    mixed_counts = ko_analyzer.ko_animals['is_mixed'].value_counts()
    if len(mixed_counts) > 0:
        colors_mix = ['lightblue', 'salmon']
        labels_mix = ['Pure', 'Mixed'] if 0 in mixed_counts.index else ['Mixed']
        ax2.pie(mixed_counts, labels=labels_mix[:len(mixed_counts)], autopct='%1.1f%%',
               colors=colors_mix[:len(mixed_counts)], startangle=90)
    ax2.set_title(f'A1D_KO State Classification\n(n={len(ko_analyzer.ko_animals)} animals)')

    # Plot 3: State balance distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(ko_analyzer.ko_animals['state_balance'], bins=10, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(ko_analyzer.ko_animals['state_balance'].median(), color='red', linestyle='--', linewidth=2, label='Median')
    ax3.set_xlabel('State Balance Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('A1D_KO: Distribution of State Balance')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Bias analysis - biased state transitions
    ax4 = fig.add_subplot(gs[1, 1])
    if not ko_analyzer.bias_analysis.empty:
        bias_data = ko_analyzer.bias_analysis
        if 'is_mixed' in bias_data.columns and len(bias_data) > 0:
            mixed_mask = bias_data['is_mixed'] == 1
            pure_mask = bias_data['is_mixed'] == 0

            data_to_plot = []
            if pure_mask.sum() > 0:
                data_to_plot.append(bias_data[pure_mask]['biased_self_transition'].values)
            if mixed_mask.sum() > 0:
                data_to_plot.append(bias_data[mixed_mask]['biased_self_transition'].values)

            if len(data_to_plot) > 0:
                labels_bias = []
                if pure_mask.sum() > 0:
                    labels_bias.append('Pure')
                if mixed_mask.sum() > 0:
                    labels_bias.append('Mixed')

                bp = ax4.boxplot(data_to_plot, labels=labels_bias, patch_artist=True)
                for patch, color in zip(bp['boxes'], ['lightblue', 'salmon']):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                ax4.set_ylabel('Biased → Biased Transition')
                ax4.set_title('A1D_KO: Biased State Self-Transitions')
                ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Comparison with other genotypes
    ax5 = fig.add_subplot(gs[1, 2])
    if not ko_analyzer.genotype_comparison.empty:
        comp = ko_analyzer.genotype_comparison[['mean_engaged', 'mean_biased', 'mean_lapsed']]
        comp.plot(kind='bar', ax=ax5)
        ax5.set_title('State Occupancy: All Genotypes')
        ax5.set_ylabel('Mean Occupancy')
        ax5.set_xlabel('Genotype')
        ax5.legend(['Engaged', 'Biased', 'Lapsed'], loc='best', fontsize=8)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax5.grid(True, alpha=0.3, axis='y')

    # Plot 6: Bias type distribution
    ax6 = fig.add_subplot(gs[2, :])
    if not ko_analyzer.bias_analysis.empty:
        bias_cols = ['biased_self_transition', 'engaged_to_biased', 'biased_to_lapsed', 'lapsed_to_biased']
        bias_data_plot = ko_analyzer.bias_analysis[bias_cols].mean()

        ax6.barh(bias_cols, bias_data_plot, color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax6.set_xlabel('Mean Transition Probability')
        ax6.set_title('A1D_KO: Average Transition Patterns (Bias Signatures)')
        ax6.grid(True, alpha=0.3, axis='x')

    plt.savefig(os.path.join(output_dir, 'a1d_ko_mixed_state_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'a1d_ko_mixed_state_analysis.pdf'), bbox_inches='tight')
    plt.close()

    print("✓ Saved A1D_KO analysis plots")


# =============================================================================
# GENERATE COMPREHENSIVE REPORT
# =============================================================================

def generate_comprehensive_report(trans_analyzer, profiler, pred_analyzer, ko_analyzer, output_dir):
    """Generate comprehensive markdown report of all analyses"""

    report = []
    report.append("# COMPREHENSIVE INDIVIDUAL-LEVEL GLM-HMM ANALYSIS")
    report.append("")
    report.append("## Executive Summary")
    report.append(f"- **Total animals analyzed:** {len(profiler.profiles_df)}")
    report.append(f"- **Genotypes:** {', '.join(sorted(profiler.profiles_df['genotype'].unique()))}")
    report.append(f"- **Phase 1→Phase 2 prediction AUC:** {pred_analyzer.auc if hasattr(pred_analyzer, 'auc') and pred_analyzer.auc else 'N/A'}")
    if pred_analyzer.reversal_summary:
        report.append(f"- **Phase 1→Phase 2 reversals:** {pred_analyzer.reversal_summary['reversal_rate']:.1%}")
    report.append(f"- **A1D_KO mixed state prevalence:** {ko_analyzer.mixed_summary['pct_mixed']:.1f}%")
    report.append("")

    # SECTION 1: Transition Analysis
    report.append("## SECTION 1: STATE TRANSITION ANALYSIS")
    report.append("")
    report.append("### Key Findings:")
    report.append("- Transition matrices extracted from all Phase 1 GLM-HMM models")
    report.append("- Entropy metrics quantify state commitment (higher=more exploratory)")
    report.append("- Diagonal dominance indicates stability of behavioral states")
    report.append("")

    report.append("### Statistical Tests (Kruskal-Wallis across genotypes):")
    if trans_analyzer.transition_test_results:
        for trans_type, results in trans_analyzer.transition_test_results.items():
            report.append(f"- **{trans_type}:** H={results['h_stat']:.3f}, p={results['p_value']:.4f}")
            for geno, mean_prob in results['by_genotype'].items():
                report.append(f"  - {geno}: {mean_prob:.3f}")
    else:
        report.append("- No significant transition differences detected across genotypes")
    report.append("")

    # SECTION 2: Behavioral Profiles
    report.append("## SECTION 2: INDIVIDUAL-LEVEL BEHAVIORAL PROFILES")
    report.append("")
    report.append("### Occupancy Patterns by Genotype:")

    for genotype in sorted(profiler.profiles_df['genotype'].unique()):
        geno_data = profiler.profiles_df[profiler.profiles_df['genotype'] == genotype]
        report.append(f"\n**{genotype}** (n={len(geno_data)}):")
        if 'occupancy_engaged' in geno_data.columns:
            report.append(f"  - Engaged: {geno_data['occupancy_engaged'].mean():.3f} ± {geno_data['occupancy_engaged'].std():.3f}")
            report.append(f"  - Biased: {geno_data['occupancy_biased'].mean():.3f} ± {geno_data['occupancy_biased'].std():.3f}")
            report.append(f"  - Lapsed: {geno_data['occupancy_lapsed'].mean():.3f} ± {geno_data['occupancy_lapsed'].std():.3f}")

    report.append("")
    report.append("### Transition Entropy by Genotype:")
    entropy_summary = trans_analyzer.entropy_df.groupby('genotype')[['mean_entropy', 'diag_dominance']].describe()
    for genotype in sorted(trans_analyzer.entropy_df['genotype'].unique()):
        geno_entropy = trans_analyzer.entropy_df[trans_analyzer.entropy_df['genotype'] == genotype]
        report.append(f"\n**{genotype}** (n={len(geno_entropy)}):")
        report.append(f"  - Mean entropy: {geno_entropy['mean_entropy'].mean():.3f} ± {geno_entropy['mean_entropy'].std():.3f}")
        report.append(f"  - Self-transition strength: {geno_entropy['diag_dominance'].mean():.3f} ± {geno_entropy['diag_dominance'].std():.3f}")

    report.append("")

    # SECTION 3: Predictability
    report.append("## SECTION 3: PHASE 1 → PHASE 2 PREDICTABILITY")
    report.append("")
    if hasattr(pred_analyzer, 'auc') and pred_analyzer.auc:
        report.append(f"### Prediction Model Performance:")
        report.append(f"- **ROC-AUC Score:** {pred_analyzer.auc:.3f}")
        report.append(f"- **Feature Importance (Top 5):**")
        for idx, row in pred_analyzer.feature_importance.head(5).iterrows():
            report.append(f"  - {row['feature']}: {row['coefficient']:.4f}")
        report.append("")

    if pred_analyzer.reversal_summary:
        report.append("### Phase 1→Phase 2 Reversals:")
        report.append(f"- **Total reversals:** {pred_analyzer.reversal_summary['total_reversals']}/{pred_analyzer.reversal_summary['total_reversals'] + pred_analyzer.reversal_summary['total_consistent']}")
        report.append(f"- **Reversal rate:** {pred_analyzer.reversal_summary['reversal_rate']:.1%}")
        if pred_analyzer.reversal_summary['reversals_by_genotype']:
            report.append("- **By genotype:**")
            for geno, count in sorted(pred_analyzer.reversal_summary['reversals_by_genotype'].items()):
                report.append(f"  - {geno}: {count} reversals")
    report.append("")

    # SECTION 4: A1D_KO Analysis
    report.append("## SECTION 4: A1D_KO MIXED STATE CHARACTERIZATION")
    report.append("")
    report.append(f"### Prevalence of Mixed State:")
    report.append(f"- **Total A1D_KO animals:** {ko_analyzer.mixed_summary['n_ko_animals']}")
    report.append(f"- **Mixed state animals:** {ko_analyzer.mixed_summary['n_mixed']} ({ko_analyzer.mixed_summary['pct_mixed']:.1f}%)")
    report.append(f"- **State balance (mixed threshold):** {ko_analyzer.mixed_summary['median_state_balance']:.3f}")
    report.append("")

    if ko_analyzer.mixed_summary['n_ko_animals'] > 0:
        report.append("### Bias Type Analysis:")
        report.append("- Analyzing transition patterns to characterize bias signatures")
        if not ko_analyzer.bias_analysis.empty:
            report.append(f"- Mean biased self-transition: {ko_analyzer.bias_analysis['biased_self_transition'].mean():.3f}")
            report.append(f"- Mean engaged→biased transition: {ko_analyzer.bias_analysis['engaged_to_biased'].mean():.3f}")
        report.append("")

        report.append("### Comparison with Other Genotypes:")
        if not ko_analyzer.genotype_comparison.empty:
            for geno in ko_analyzer.genotype_comparison.index:
                row = ko_analyzer.genotype_comparison.loc[geno]
                report.append(f"\n**{geno}** (n={int(row['n_animals'])}):")
                report.append(f"  - Mean engaged: {row['mean_engaged']:.3f} ± {row['std_engaged']:.3f}")
                report.append(f"  - Mean biased: {row['mean_biased']:.3f} ± {row['std_biased']:.3f}")
                report.append(f"  - Mean lapsed: {row['mean_lapsed']:.3f} ± {row['std_lapsed']:.3f}")

    report.append("")
    report.append("## Visualization Outputs")
    report.append("- `transition_matrices_by_genotype.png` - Heatmaps of state transitions")
    report.append("- `entropy_stability_analysis.png` - Entropy and self-transition metrics")
    report.append("- `behavioral_profile_heatmap.png` - Individual animal behavioral features")
    report.append("- `phase1_to_phase2_predictability.png` - ROC curve and feature importance")
    report.append("- `phase1_phase2_reversals.png` - Reversal analysis across genotypes")
    report.append("- `a1d_ko_mixed_state_analysis.png` - A1D_KO state characterization")

    # Save report
    report_text = "\n".join(report)
    with open(os.path.join(output_dir, 'COMPREHENSIVE_INDIVIDUAL_ANALYSIS_REPORT.txt'), 'w') as f:
        f.write(report_text)

    print("✓ Saved comprehensive report")
    return report_text


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("COMPREHENSIVE INDIVIDUAL-LEVEL GLM-HMM ANALYSIS")
    print("=" * 80)
    print()

    # Phase 1: Transition Analysis
    print("[1/5] Extracting state transition matrices...")
    trans_analyzer = TransitionAnalyzer()
    trans_analyzer.extract_all_transitions()
    trans_analyzer.compute_transition_entropy()
    trans_analyzer.compute_transition_stability()
    trans_analyzer.analyze_transition_differences_by_genotype()

    # Phase 2: Behavioral Profiles
    print("[2/5] Building comprehensive behavioral profiles...")
    profiler = BehavioralProfiler(trans_analyzer)
    profiler.build_profiles()

    # Phase 3: Predictability Analysis
    print("[3/5] Performing Phase 1→Phase 2 predictability analysis...")
    pred_analyzer = PredictabilityAnalyzer(profiler)
    pred_analyzer.prepare_prediction_data()
    pred_results = pred_analyzer.fit_predictive_model()
    pred_analyzer.analyze_reversals()

    # Phase 4: A1D_KO Analysis
    print("[4/5] Analyzing A1D_KO mixed state characterization...")
    ko_analyzer = A1DKOAnalyzer(profiler)
    ko_mixed = ko_analyzer.characterize_mixed_state()
    ko_bias = ko_analyzer.detect_bias_types()
    ko_comparison = ko_analyzer.compare_with_other_genotypes()

    # Phase 5: Visualizations and Report
    print("[5/5] Creating visualizations and report...")
    create_transition_heatmaps(trans_analyzer, OUTPUT_DIR)
    create_entropy_stability_plots(trans_analyzer, OUTPUT_DIR)
    create_behavioral_profile_heatmap(profiler, OUTPUT_DIR)
    create_predictability_plots(pred_analyzer, OUTPUT_DIR)
    create_reversal_analysis_plot(pred_analyzer, OUTPUT_DIR)
    create_ko_analysis_plots(ko_analyzer, OUTPUT_DIR)

    # Save data files
    profiler.profiles_df.to_csv(os.path.join(OUTPUT_DIR, 'behavioral_profiles_all_animals.csv'), index=False)
    trans_analyzer.entropy_df.to_csv(os.path.join(OUTPUT_DIR, 'transition_entropy_metrics.csv'), index=False)
    trans_analyzer.stability_df.to_csv(os.path.join(OUTPUT_DIR, 'transition_stability_metrics.csv'), index=False)
    pred_analyzer.prediction_data.to_csv(os.path.join(OUTPUT_DIR, 'phase1_phase2_prediction_data.csv'), index=False)
    if hasattr(pred_analyzer, 'feature_importance'):
        pred_analyzer.feature_importance.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance_phase2_prediction.csv'), index=False)
    ko_analyzer.ko_animals.to_csv(os.path.join(OUTPUT_DIR, 'a1d_ko_animals_characterization.csv'), index=False)
    if not ko_analyzer.bias_analysis.empty:
        ko_analyzer.bias_analysis.to_csv(os.path.join(OUTPUT_DIR, 'a1d_ko_bias_analysis.csv'))
    if not ko_analyzer.genotype_comparison.empty:
        ko_analyzer.genotype_comparison.to_csv(os.path.join(OUTPUT_DIR, 'genotype_comparison_occupancy.csv'))

    # Generate report
    report = generate_comprehensive_report(trans_analyzer, profiler, pred_analyzer, ko_analyzer, OUTPUT_DIR)

    print()
    print("=" * 80)
    print(f"✓ ANALYSIS COMPLETE - All results saved to {OUTPUT_DIR}")
    print("=" * 80)
    print()
    print("Key Statistics:")
    print(f"  - Animals analyzed: {len(profiler.profiles_df)}")
    if hasattr(pred_analyzer, 'auc') and pred_analyzer.auc:
        print(f"  - Phase 1→Phase 2 prediction AUC: {pred_analyzer.auc:.3f}")
    if pred_analyzer.reversal_summary:
        print(f"  - Reversal rate: {pred_analyzer.reversal_summary['reversal_rate']:.1%}")
    print(f"  - A1D_KO mixed state prevalence: {ko_analyzer.mixed_summary['pct_mixed']:.1f}%")
