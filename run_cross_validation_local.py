"""
Cross-Validation Script for Local Execution
============================================

This script runs cross-validation on GLM-HMM models with 2, 3, 4, and 5 states
to validate the optimal number of states for your behavioral data.

INSTRUCTIONS:
1. Update the file paths in the CONFIG section below to match your data location
2. (Optional) Modify the list of animals to test
3. Run the script: python run_cross_validation_local.py
4. Results will be saved in: results/phase1_non_reversal/priority2_validation/
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
sys.path.insert(0, str(Path(__file__).parent))
from glmhmm_utils import load_and_preprocess_session_data, create_design_matrix
from glmhmm_ashwood import GLMHMM

# ==============================================================================
# CONFIG SECTION - UPDATE THESE PATHS TO MATCH YOUR DATA LOCATION
# ==============================================================================

# UPDATE THESE PATHS TO YOUR LOCAL FILES:
DATA_FILES = {
    'W': r"C:/Users/yashodad/OneDrive - Michigan Medicine/Documents/GitHub/GLMHMM/W LD Data 11.08 All_processed.csv",
    'F': r"C:/Users/yashodad/OneDrive - Michigan Medicine/Documents/GitHub/GLMHMM/F LD Data 11.08 All_processed.csv"
}

# Select animals to test (smaller subset = faster)
# Format: (animal_id, cohort)
# For quick test (10-15 min), use 3-4 animals
# For comprehensive test (1-2 hours), use 10-15 animals
ANIMALS_TO_TEST = [
    # W Cohort examples
    ('c1m1', 'W'),
    ('c3m1', 'W'),

    # F Cohort examples
    (11, 'F'),
    (21, 'F'),
    (31, 'F'),
    (61, 'F'),
]

# Cross-validation settings
N_STATES_TO_TEST = [2, 3, 4, 5]  # Which model sizes to compare
N_FOLDS = 3  # Number of cross-validation folds

# Output directory
OUTPUT_DIR = Path('results/phase1_non_reversal/priority2_validation')

# ==============================================================================
# END CONFIG - No need to modify below this line
# ==============================================================================

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


class ModelValidator:
    """Cross-validate GLM-HMM models with different numbers of states."""

    def __init__(self, data_files, output_dir):
        self.data_files = data_files
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fit_and_evaluate_model(self, X, y, n_states, train_idx, test_idx):
        """Fit model on training data and evaluate on test data."""
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

            # Evaluate on test set using Viterbi algorithm
            test_states = model.viterbi(X_test, y_test)

            # Compute accuracy for each trial
            test_acc_list = []
            for i in range(len(X_test)):
                state = test_states[i]
                x_test_i = X_test[i:i+1]

                # Apply feature scaling
                if model.feature_scaler is not None:
                    x_scaled = x_test_i.copy()
                    non_constant_cols = np.where(np.std(X_train, axis=0) > 1e-6)[0]
                    if len(non_constant_cols) > 0:
                        x_scaled[:, non_constant_cols] = model.feature_scaler.transform(
                            x_test_i[:, non_constant_cols]
                        )
                else:
                    x_scaled = x_test_i

                # Predict with this state's GLM
                from scipy.special import expit
                logit = x_scaled @ model.glm_weights[state] + model.glm_intercepts[state]
                prob_1 = expit(logit)[0]
                pred = 1 if prob_1 > 0.5 else 0
                test_acc_list.append(pred == y_test[i])

            test_acc = np.mean(test_acc_list)

            # Compute information criteria
            n_params = n_states * (n_states - 1) + n_states * (X_train.shape[1] + 1)
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
            print(f"  Failed: {str(e)[:50]}")
            return {
                'log_likelihood': np.nan,
                'accuracy': np.nan,
                'aic': np.nan,
                'bic': np.nan,
                'n_params': np.nan,
                'success': False
            }

    def cross_validate_animal(self, animal_id, cohort, n_states_list, n_folds):
        """Perform k-fold cross-validation for one animal."""
        print(f"\nCross-validating {animal_id} (Cohort {cohort})...")

        # Load data
        data_file = self.data_files[cohort]

        if not Path(data_file).exists():
            print(f"  ERROR: Data file not found: {data_file}")
            return pd.DataFrame()

        trial_df = load_and_preprocess_session_data(data_file)

        # Check if animal exists
        if animal_id not in trial_df['animal_id'].values:
            print(f"  WARNING: Animal {animal_id} not found in data")
            return pd.DataFrame()

        # Create design matrix
        X, y, feature_names, metadata, animal_data = create_design_matrix(
            trial_df,
            animal_id=animal_id,
            include_session_progression=True
        )

        # Remove stimulus column (as in Phase 1 analysis)
        if 'stimulus_correct_side' in feature_names:
            stimulus_idx = feature_names.index('stimulus_correct_side')
            feature_indices = [i for i in range(len(feature_names)) if i != stimulus_idx]
            X = X[:, feature_indices]
            feature_names = [feature_names[i] for i in feature_indices]

        if len(X) < n_folds * 20:
            print(f"  WARNING: Insufficient trials ({len(X)}) for {n_folds}-fold CV")
            return pd.DataFrame()

        # Cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        results = []

        for n_states in n_states_list:
            print(f"  Testing {n_states} states...", end=' ')

            fold_metrics = []
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
                metrics = self.fit_and_evaluate_model(X, y, n_states, train_idx, test_idx)
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

    def run_validation_study(self, animals_subset, n_states_list, n_folds):
        """Run cross-validation study on subset of animals."""
        print("="*80)
        print("MODEL CROSS-VALIDATION STUDY")
        print("="*80)
        print(f"Testing {len(animals_subset)} animals")
        print(f"States to test: {n_states_list}")
        print(f"Cross-validation folds: {n_folds}")
        print("="*80)

        all_results = []

        for animal_id, cohort in animals_subset:
            animal_results = self.cross_validate_animal(
                animal_id, cohort, n_states_list, n_folds
            )
            if len(animal_results) > 0:
                all_results.append(animal_results)

        if len(all_results) == 0:
            print("\n✗ No results obtained!")
            return pd.DataFrame()

        # Combine results
        df = pd.concat(all_results, ignore_index=True)

        # Save
        output_file = self.output_dir / 'cross_validation_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved results to {output_file}")

        return df

    def plot_validation_results(self, df):
        """Create comprehensive visualization of cross-validation results."""
        if len(df) == 0:
            print("No data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Panel A: Log-likelihood
        ax = axes[0, 0]
        ll_stats = df.groupby('n_states').agg({
            'log_likelihood': ['mean', 'sem']
        }).reset_index()

        x = ll_stats['n_states']
        y = ll_stats['log_likelihood']['mean']
        sem = ll_stats['log_likelihood']['sem']

        ax.plot(x, y, 'o-', linewidth=2.5, markersize=10, color='#3498db')
        ax.fill_between(x, y - sem, y + sem, alpha=0.3, color='#3498db')

        best_idx = np.argmax(y)
        ax.plot(x.iloc[best_idx], y.iloc[best_idx], 'r*', markersize=20,
               label=f'Best: {x.iloc[best_idx]:.0f} states')

        ax.set_xlabel('Number of States', fontsize=13, fontweight='bold')
        ax.set_ylabel('Test Log-Likelihood', fontsize=13, fontweight='bold')
        ax.set_title('Model Fit: Log-Likelihood', fontsize=14, fontweight='bold')
        ax.set_xticks(N_STATES_TO_TEST)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        # Panel B: Accuracy
        ax = axes[0, 1]
        acc_stats = df.groupby('n_states').agg({
            'accuracy': ['mean', 'sem']
        }).reset_index()

        x = acc_stats['n_states']
        y = acc_stats['accuracy']['mean']
        sem = acc_stats['accuracy']['sem']

        ax.plot(x, y, 'o-', linewidth=2.5, markersize=10, color='#2ecc71')
        ax.fill_between(x, y - sem, y + sem, alpha=0.3, color='#2ecc71')

        best_idx = np.argmax(y)
        ax.plot(x.iloc[best_idx], y.iloc[best_idx], 'r*', markersize=20,
               label=f'Best: {x.iloc[best_idx]:.0f} states')

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('Number of States', fontsize=13, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
        ax.set_title('Model Performance: Prediction Accuracy', fontsize=14, fontweight='bold')
        ax.set_xticks(N_STATES_TO_TEST)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        # Panel C: AIC
        ax = axes[1, 0]
        aic_stats = df.groupby('n_states')['aic'].mean().reset_index()
        x = aic_stats['n_states']
        y = aic_stats['aic']

        bars = ax.bar(x, y, color='#e74c3c', alpha=0.7)
        best_idx = np.argmin(y)
        bars[best_idx].set_color('#c0392b')
        bars[best_idx].set_alpha(1.0)

        ax.set_xlabel('Number of States', fontsize=13, fontweight='bold')
        ax.set_ylabel('AIC (lower is better)', fontsize=13, fontweight='bold')
        ax.set_title('Model Selection: Akaike Information Criterion',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(N_STATES_TO_TEST)
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=10)

        # Panel D: BIC
        ax = axes[1, 1]
        bic_stats = df.groupby('n_states')['bic'].mean().reset_index()
        x = bic_stats['n_states']
        y = bic_stats['bic']

        bars = ax.bar(x, y, color='#9b59b6', alpha=0.7)
        best_idx = np.argmin(y)
        bars[best_idx].set_color('#8e44ad')
        bars[best_idx].set_alpha(1.0)

        ax.set_xlabel('Number of States', fontsize=13, fontweight='bold')
        ax.set_ylabel('BIC (lower is better)', fontsize=13, fontweight='bold')
        ax.set_title('Model Selection: Bayesian Information Criterion',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(N_STATES_TO_TEST)
        ax.grid(axis='y', alpha=0.3)

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
        png_file = self.output_dir / 'model_cross_validation.png'
        pdf_file = self.output_dir / 'model_cross_validation.pdf'
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_file, bbox_inches='tight')
        plt.close()

        print(f"✓ Created visualizations: {png_file}")

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
                indicator = "  ←" if n_states == best_n else ""
                print(f"    {n_states} states: {summary[n_states]:.2f}{indicator}")


def main():
    """Run model validation study."""
    print("="*80)
    print("GLM-HMM CROSS-VALIDATION")
    print("="*80)
    print("\nThis script will:")
    print("1. Load your behavioral data")
    print("2. Test models with 2, 3, 4, and 5 hidden states")
    print("3. Use 3-fold cross-validation to assess performance")
    print("4. Generate comparison plots and summary statistics")
    print("\nEstimated runtime: ~5 minutes per animal")
    print("="*80)

    # Verify data files exist
    print("\nVerifying data files...")
    for cohort, filepath in DATA_FILES.items():
        if Path(filepath).exists():
            print(f"  ✓ {cohort} cohort: {filepath}")
        else:
            print(f"  ✗ {cohort} cohort: FILE NOT FOUND - {filepath}")
            print("\n  ERROR: Please update DATA_FILES paths at the top of this script")
            return

    # Run validation
    validator = ModelValidator(DATA_FILES, OUTPUT_DIR)
    results_df = validator.run_validation_study(
        ANIMALS_TO_TEST,
        N_STATES_TO_TEST,
        N_FOLDS
    )

    if len(results_df) > 0:
        # Plot results
        validator.plot_validation_results(results_df)

        print("\n" + "="*80)
        print("✓ CROSS-VALIDATION COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {OUTPUT_DIR}")
        print("\nFiles created:")
        print(f"  - cross_validation_results.csv")
        print(f"  - model_cross_validation.png")
        print(f"  - model_cross_validation.pdf")
    else:
        print("\n✗ Validation failed - no results obtained")


if __name__ == '__main__':
    main()
