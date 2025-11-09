"""
Priority 2: Model Cross-Validation (2-5 States)
================================================

Tests GLM-HMM models with 2, 3, 4, and 5 states using cross-validation
to validate the choice of 3 states.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Import utilities
import sys
sys.path.insert(0, '/home/user/GLMHMM')
from glmhmm_utils import load_and_preprocess_session_data, create_design_matrix
from glmhmm_ashwood import GLMHMM

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


class ModelValidator:
    """Cross-validate GLM-HMM models with different numbers of states."""

    def __init__(self, results_dir='results/phase1_non_reversal'):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'priority2_validation'
        self.output_dir.mkdir(exist_ok=True)

    def fit_and_evaluate_model(self, X, y, n_states, train_idx, test_idx):
        """
        Fit model on training data and evaluate on test data.

        Returns: dict with metrics
        """
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Fit model
        model = GLMHMM(
            n_states=n_states,
            observation_model='bernoulli',
            feature_names=None,
            normalize_features=True,
            regularization_strength=1.0,
            random_state=42
        )

        try:
            model.fit(X_train, y_train, n_iter=200, tolerance=1e-4, verbose=False)

            # Get training log-likelihood
            train_ll = model.log_likelihood_history[-1] if len(model.log_likelihood_history) > 0 else np.nan

            # Evaluate accuracy on test set
            test_states = model.viterbi(X_test, y_test)
            test_probs = model.predict_proba(X_test)  # Marginal probabilities
            test_acc = np.mean((test_probs > 0.5).astype(int) == y_test)

            # Compute information criteria based on training data
            n_params = n_states * (n_states - 1) + \
                      n_states * (X_train.shape[1] + 1)  # Transitions + weights
            aic = -2 * train_ll + 2 * n_params
            bic = -2 * train_ll + n_params * np.log(len(y_train))

            return {
                'log_likelihood': train_ll,
                'accuracy': test_acc,
                'aic': aic,
                'bic': bic,
                'n_params': n_params,
                'success': True
            }

        except Exception as e:
            print(f"  Failed: {str(e)[:40]}")
            return {
                'log_likelihood': np.nan,
                'accuracy': np.nan,
                'aic': np.nan,
                'bic': np.nan,
                'n_params': np.nan,
                'success': False
            }

    def cross_validate_animal(self, animal_id, cohort, n_states_list=[2, 3, 4, 5],
                              n_folds=3):
        """
        Perform k-fold cross-validation for one animal across different state numbers.

        Returns: DataFrame with results
        """
        print(f"\nCross-validating {animal_id} (Cohort {cohort})...")

        # Load data
        if cohort == 'W':
            data_file = '/home/user/GLMHMM/W LD Data 11.08 All_processed.csv'
        else:
            data_file = '/home/user/GLMHMM/F LD Data 11.08 All_processed.csv'

        trial_df = load_and_preprocess_session_data(data_file)

        # Check if animal exists
        if animal_id not in trial_df['animal_id'].values:
            print(f"  No data found for {animal_id}")
            return pd.DataFrame()

        # Create design matrix with standard features
        X, y, feature_names, metadata, animal_data = create_design_matrix(
            trial_df,
            animal_id=animal_id,
            include_session_progression=True
        )

        # Remove stimulus column (first column) as in Phase 1 analysis
        if 'stimulus_correct_side' in feature_names:
            stimulus_idx = feature_names.index('stimulus_correct_side')
            feature_indices = [i for i in range(len(feature_names)) if i != stimulus_idx]
            X = X[:, feature_indices]
            feature_names = [feature_names[i] for i in feature_indices]

        if len(X) < n_folds * 20:  # Need enough trials for k-fold
            print(f"  Insufficient trials ({len(X)}) for {n_folds}-fold CV")
            return pd.DataFrame()

        # Cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        results = []

        for n_states in n_states_list:
            print(f"  Testing {n_states} states...", end=' ')

            fold_metrics = []
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
                metrics = self.fit_and_evaluate_model(X, y, n_states,
                                                      train_idx, test_idx)
                if metrics['success']:
                    fold_metrics.append(metrics)

            if len(fold_metrics) > 0:
                # Average across folds
                avg_metrics = {
                    'animal_id': animal_id,
                    'cohort': cohort,
                    'n_states': n_states,
                    'n_trials': len(X),
                    'log_likelihood': np.mean([m['log_likelihood'] for m in fold_metrics]),
                    'log_likelihood_sem': np.std([m['log_likelihood'] for m in fold_metrics]) / np.sqrt(len(fold_metrics)),
                    'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
                    'accuracy_sem': np.std([m['accuracy'] for m in fold_metrics]) / np.sqrt(len(fold_metrics)),
                    'aic': np.mean([m['aic'] for m in fold_metrics]),
                    'bic': np.mean([m['bic'] for m in fold_metrics]),
                    'n_params': fold_metrics[0]['n_params'],
                    'n_folds': len(fold_metrics)
                }
                results.append(avg_metrics)
                print(f"✓ (LL={avg_metrics['log_likelihood']:.1f}, Acc={avg_metrics['accuracy']:.3f})")
            else:
                print("✗ All folds failed")

        return pd.DataFrame(results)

    def run_validation_study(self, animals_subset):
        """
        Run cross-validation study on subset of animals.

        Args:
            animals_subset: List of (animal_id, cohort) tuples

        Returns: DataFrame with all results
        """
        print("="*80)
        print("MODEL CROSS-VALIDATION STUDY")
        print("="*80)

        all_results = []

        for animal_id, cohort in animals_subset:
            animal_results = self.cross_validate_animal(animal_id, cohort)
            if len(animal_results) > 0:
                all_results.append(animal_results)

        if len(all_results) == 0:
            print("\nNo results obtained!")
            return pd.DataFrame()

        # Combine results
        df = pd.concat(all_results, ignore_index=True)

        # Save
        df.to_csv(self.output_dir / 'cross_validation_results.csv', index=False)
        print(f"\n✓ Saved results to {self.output_dir / 'cross_validation_results.csv'}")

        return df

    def plot_validation_results(self, df):
        """
        Create comprehensive visualization of cross-validation results.
        """
        if len(df) == 0:
            print("No data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Panel A: Log-likelihood by number of states
        ax = axes[0, 0]
        ll_stats = df.groupby('n_states').agg({
            'log_likelihood': ['mean', 'sem']
        }).reset_index()

        x = ll_stats['n_states']
        y = ll_stats['log_likelihood']['mean']
        sem = ll_stats['log_likelihood']['sem']

        ax.plot(x, y, 'o-', linewidth=2.5, markersize=10, color='#3498db')
        ax.fill_between(x, y - sem, y + sem, alpha=0.3, color='#3498db')

        # Mark best model
        best_idx = np.argmax(y)
        ax.plot(x.iloc[best_idx], y.iloc[best_idx], 'r*', markersize=20,
               label=f'Best: {x.iloc[best_idx]:.0f} states')

        ax.set_xlabel('Number of States', fontsize=13, fontweight='bold')
        ax.set_ylabel('Test Log-Likelihood', fontsize=13, fontweight='bold')
        ax.set_title('Model Fit: Log-Likelihood', fontsize=14, fontweight='bold')
        ax.set_xticks([2, 3, 4, 5])
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        # Panel B: Test accuracy by number of states
        ax = axes[0, 1]
        acc_stats = df.groupby('n_states').agg({
            'accuracy': ['mean', 'sem']
        }).reset_index()

        x = acc_stats['n_states']
        y = acc_stats['accuracy']['mean']
        sem = acc_stats['accuracy']['sem']

        ax.plot(x, y, 'o-', linewidth=2.5, markersize=10, color='#2ecc71')
        ax.fill_between(x, y - sem, y + sem, alpha=0.3, color='#2ecc71')

        # Mark best model
        best_idx = np.argmax(y)
        ax.plot(x.iloc[best_idx], y.iloc[best_idx], 'r*', markersize=20,
               label=f'Best: {x.iloc[best_idx]:.0f} states')

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('Number of States', fontsize=13, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
        ax.set_title('Model Performance: Prediction Accuracy', fontsize=14, fontweight='bold')
        ax.set_xticks([2, 3, 4, 5])
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        # Panel C: AIC comparison
        ax = axes[1, 0]
        aic_stats = df.groupby('n_states').agg({
            'aic': 'mean'
        }).reset_index()

        x = aic_stats['n_states']
        y = aic_stats['aic']

        bars = ax.bar(x, y, color='#e74c3c', alpha=0.7)

        # Mark best (lowest AIC)
        best_idx = np.argmin(y)
        bars[best_idx].set_color('#c0392b')
        bars[best_idx].set_alpha(1.0)

        ax.set_xlabel('Number of States', fontsize=13, fontweight='bold')
        ax.set_ylabel('AIC (lower is better)', fontsize=13, fontweight='bold')
        ax.set_title('Model Selection: Akaike Information Criterion',
                    fontsize=14, fontweight='bold')
        ax.set_xticks([2, 3, 4, 5])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=10)

        # Panel D: BIC comparison
        ax = axes[1, 1]
        bic_stats = df.groupby('n_states').agg({
            'bic': 'mean'
        }).reset_index()

        x = bic_stats['n_states']
        y = bic_stats['bic']

        bars = ax.bar(x, y, color='#9b59b6', alpha=0.7)

        # Mark best (lowest BIC)
        best_idx = np.argmin(y)
        bars[best_idx].set_color('#8e44ad')
        bars[best_idx].set_alpha(1.0)

        ax.set_xlabel('Number of States', fontsize=13, fontweight='bold')
        ax.set_ylabel('BIC (lower is better)', fontsize=13, fontweight='bold')
        ax.set_title('Model Selection: Bayesian Information Criterion',
                    fontsize=14, fontweight='bold')
        ax.set_xticks([2, 3, 4, 5])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=10)

        # Overall title
        n_animals = df['animal_id'].nunique()
        fig.suptitle(f'GLM-HMM Model Cross-Validation (n={n_animals} animals)',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save
        plt.savefig(self.output_dir / 'model_cross_validation.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'model_cross_validation.pdf',
                   bbox_inches='tight')
        plt.close()

        print(f"✓ Created model validation visualization")

        # Print summary
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)

        for metric, criterion in [('log_likelihood', 'max'),
                                  ('accuracy', 'max'),
                                  ('aic', 'min'),
                                  ('bic', 'min')]:
            summary = df.groupby('n_states')[metric].mean()
            if criterion == 'max':
                best_n = summary.idxmax()
                best_val = summary.max()
            else:
                best_n = summary.idxmin()
                best_val = summary.min()

            print(f"\n{metric.upper()}:")
            print(f"  Best model: {best_n} states ({metric}={best_val:.2f})")
            for n_states in sorted(summary.index):
                print(f"    {n_states} states: {summary[n_states]:.2f}")


def main():
    """Run model validation study."""
    print("="*80)
    print("PRIORITY 2: MODEL CROSS-VALIDATION")
    print("="*80)

    validator = ModelValidator()

    # Select representative subset of animals for computational efficiency
    # Use 1-2 animals from each major genotype group, 3-fold CV
    animals_subset = [
        # Cohort W
        ('c1m1', 'W'),  # W+
        ('c3m1', 'W'),  # W-
        # Cohort F
        (11, 'F'),  # F+
        (21, 'F'),  # F+/+
        (31, 'F'),  # F+/-
        (61, 'F'),  # F-/-
    ]

    print(f"\nValidating {len(animals_subset)} animals across 2-5 states with 3-fold CV...")
    print("This will take approximately 20-30 minutes...\n")

    # Run validation
    results_df = validator.run_validation_study(animals_subset)

    if len(results_df) > 0:
        # Plot results
        validator.plot_validation_results(results_df)

        print("\n" + "="*80)
        print("✓ MODEL VALIDATION COMPLETE!")
        print("="*80)
        print(f"\nOutput directory: {validator.output_dir}")
    else:
        print("\n✗ Validation failed - no results obtained")


if __name__ == '__main__':
    main()
