"""
Phase 2: Reversal Task Analysis
================================

Analyzes behavioral states during reversal learning tasks.

Goals:
1. Identify behavioral states specific to reversal learning
2. Compare Phase 1 (non-reversal) vs Phase 2 (reversal) states
3. Analyze genotype differences in reversal adaptation
4. Quantify cognitive flexibility metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import sys
sys.path.insert(0, '/home/user/GLMHMM')

from glmhmm_utils import load_and_preprocess_session_data, create_design_matrix
from glmhmm_ashwood import GLMHMM
from state_validation import (
    compute_comprehensive_state_metrics,
    compute_performance_trajectory,
    validate_state_labels,
    create_broad_state_categories
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


class Phase2ReversalAnalysis:
    """Analyze reversal learning with GLM-HMM."""

    def __init__(self):
        self.results_dir = Path('results/phase2_reversal')
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Create subdirectories
        (self.results_dir / 'figures').mkdir(exist_ok=True)
        (self.results_dir / 'models').mkdir(exist_ok=True)
        (self.results_dir / 'summary').mkdir(exist_ok=True)

    def load_reversal_data(self, cohort):
        """Load and filter reversal task data."""
        print(f"\n{'='*80}")
        print(f"LOADING {cohort} COHORT REVERSAL DATA")
        print(f"{'='*80}")

        if cohort == 'W':
            data_file = 'W LD Data 11.08 All_processed.csv'
        else:
            data_file = 'F LD Data 11.08 All_processed.csv'

        # Load all data
        df_all = load_and_preprocess_session_data(data_file)

        # Filter to reversal tasks only
        df_reversal = df_all[df_all['task_type'].str.contains('reversal', case=False, na=False)].copy()

        print(f"\nTotal trials: {len(df_all)}")
        print(f"Reversal trials: {len(df_reversal)} ({len(df_reversal)/len(df_all)*100:.1f}%)")

        print(f"\nReversal task types:")
        print(df_reversal['task_type'].value_counts())

        print(f"\nAnimals with reversal data: {df_reversal['animal_id'].nunique()}")

        # Get genotype distribution
        animals_by_genotype = df_reversal.groupby('genotype')['animal_id'].nunique()
        print(f"\nAnimals per genotype:")
        for genotype, count in animals_by_genotype.items():
            print(f"  {genotype}: {count} animals")

        return df_reversal

    def run_animal_model(self, animal_id, cohort, reversal_df, n_states=3):
        """Fit GLM-HMM to a single animal's reversal data."""

        # Get animal data
        animal_data = reversal_df[reversal_df['animal_id'] == animal_id].copy()

        if len(animal_data) < 100:  # Need minimum trials
            print(f"  {animal_id}: Insufficient trials ({len(animal_data)})")
            return None

        # Get genotype
        genotype = animal_data['genotype'].iloc[0]

        print(f"\n{animal_id} ({genotype}): {len(animal_data)} trials")

        # Create design matrix
        X, y, feature_names, metadata, _ = create_design_matrix(
            reversal_df,
            animal_id=animal_id,
            include_session_progression=True
        )

        # Remove stimulus column (as in Phase 1)
        if 'stimulus_correct_side' in feature_names:
            stimulus_idx = feature_names.index('stimulus_correct_side')
            feature_indices = [i for i in range(len(feature_names)) if i != stimulus_idx]
            X = X[:, feature_indices]
            feature_names = [feature_names[i] for i in feature_indices]

        # Fit GLM-HMM
        try:
            model = GLMHMM(
                n_states=n_states,
                feature_names=feature_names,
                normalize_features=True,
                random_state=42
            )

            model.fit(X, y)

            # Get state sequence
            states = model.predict(X)

            # Validate states using comprehensive metrics
            state_metrics = compute_comprehensive_state_metrics(animal_data, model, metadata)
            trajectory_df = compute_performance_trajectory(animal_data, model)
            validated_labels = validate_state_labels(state_metrics, trajectory_df)

            # Create broad categories
            broad_categories = create_broad_state_categories(validated_labels)

            # Package results
            results = {
                'animal_id': animal_id,
                'cohort': cohort,
                'genotype': genotype,
                'n_trials': len(animal_data),
                'n_states': n_states,
                'model': model,
                'states': states,
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'metadata': metadata,
                'state_metrics': state_metrics,
                'trajectory_df': trajectory_df,
                'validated_labels': validated_labels,
                'broad_categories': broad_categories,
                'log_likelihood': model.log_likelihood_,
                'n_iterations': model.n_iter_
            }

            # Save
            output_file = self.results_dir / 'models' / f'{animal_id}_cohort{cohort}_reversal.pkl'
            with open(output_file, 'wb') as f:
                pickle.dump(results, f)

            print(f"  ✓ Completed (LL={model.log_likelihood_:.1f}, {model.n_iter_} iterations)")

            return results

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return None

    def run_cohort_analysis(self, cohort):
        """Run GLM-HMM on all animals in cohort."""
        print(f"\n{'='*80}")
        print(f"PHASE 2 REVERSAL ANALYSIS - {cohort} COHORT")
        print(f"{'='*80}")

        # Load reversal data
        reversal_df = self.load_reversal_data(cohort)

        # Get unique animals
        animals = reversal_df['animal_id'].unique()

        print(f"\n{'='*80}")
        print(f"FITTING GLM-HMM MODELS")
        print(f"{'='*80}")

        results = []
        for animal_id in animals:
            result = self.run_animal_model(animal_id, cohort, reversal_df)
            if result is not None:
                results.append(result)

        print(f"\n{'='*80}")
        print(f"COHORT {cohort} SUMMARY")
        print(f"{'='*80}")
        print(f"Models fitted: {len(results)} / {len(animals)}")

        return results

    def compare_phase1_phase2(self, cohort):
        """Compare Phase 1 (non-reversal) vs Phase 2 (reversal) states."""
        print(f"\n{'='*80}")
        print(f"COMPARING PHASE 1 vs PHASE 2 - {cohort} COHORT")
        print(f"{'='*80}")

        # Load Phase 1 models
        phase1_dir = Path('results/phase1_non_reversal')
        phase1_models = {}

        for pkl_file in phase1_dir.glob(f'*_cohort{cohort}_model.pkl'):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    animal_id = data['animal_id']
                    phase1_models[animal_id] = data
            except:
                continue

        # Load Phase 2 models
        phase2_dir = self.results_dir / 'models'
        phase2_models = {}

        for pkl_file in phase2_dir.glob(f'*_cohort{cohort}_reversal.pkl'):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    animal_id = data['animal_id']
                    phase2_models[animal_id] = data
            except:
                continue

        # Find animals with both Phase 1 and Phase 2 data
        common_animals = set(phase1_models.keys()) & set(phase2_models.keys())

        print(f"\nPhase 1 animals: {len(phase1_models)}")
        print(f"Phase 2 animals: {len(phase2_models)}")
        print(f"Animals with both: {len(common_animals)}")

        if len(common_animals) == 0:
            print("\nNo animals with both Phase 1 and Phase 2 data!")
            return

        # Compare state characteristics
        comparisons = []

        for animal_id in common_animals:
            p1 = phase1_models[animal_id]
            p2 = phase2_models[animal_id]

            # Aggregate state metrics
            p1_metrics = p1.get('state_metrics', {})
            p2_metrics = p2.get('state_metrics', {})

            # Compute averages across states
            p1_avg_acc = np.mean([m['accuracy'] for m in p1_metrics.values()])
            p2_avg_acc = np.mean([m['accuracy'] for m in p2_metrics.values()])

            p1_avg_wsls = np.mean([m['wsls_ratio'] for m in p1_metrics.values()])
            p2_avg_wsls = np.mean([m['wsls_ratio'] for m in p2_metrics.values()])

            comparisons.append({
                'animal_id': animal_id,
                'genotype': p1['genotype'],
                'p1_accuracy': p1_avg_acc,
                'p2_accuracy': p2_avg_acc,
                'p1_wsls': p1_avg_wsls,
                'p2_wsls': p2_avg_wsls,
                'accuracy_change': p2_avg_acc - p1_avg_acc,
                'wsls_change': p2_avg_wsls - p1_avg_wsls
            })

        comp_df = pd.DataFrame(comparisons)

        # Save comparison
        comp_df.to_csv(self.results_dir / 'summary' / f'{cohort}_phase1_vs_phase2_comparison.csv', index=False)

        print(f"\nPhase 1 vs Phase 2 Comparison:")
        print(f"  Mean accuracy change: {comp_df['accuracy_change'].mean():.3f}")
        print(f"  Mean WSLS change: {comp_df['wsls_change'].mean():.3f}")

        return comp_df


def main():
    """Run Phase 2 reversal analysis for both cohorts."""

    analyzer = Phase2ReversalAnalysis()

    # Analyze both cohorts
    for cohort in ['W', 'F']:
        print(f"\n\n{'#'*80}")
        print(f"# COHORT {cohort}")
        print(f"{'#'*80}")

        # Run analysis
        results = analyzer.run_cohort_analysis(cohort)

        # Compare with Phase 1
        if len(results) > 0:
            analyzer.compare_phase1_phase2(cohort)

    print(f"\n\n{'='*80}")
    print("PHASE 2 REVERSAL ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {analyzer.results_dir}")


if __name__ == '__main__':
    main()
