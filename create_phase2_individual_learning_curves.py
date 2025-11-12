"""
Phase 2 Individual Learning Curves with State Occupancy
========================================================

Creates publication-quality learning curves for individual mice in Phase 2
(reversal learning), matching the style from Phase 1 non-reversal figures.

Features:
- Top panel: Rolling average accuracy curve (50-trial window)
- Bottom panel: State occupancy over time (stacked area plot)
- Publication-quality formatting (300 DPI, PDF + PNG)
- Focus on specific sample mice: 82 and c1m2

Style matches Phase 1 individual animal learning curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3

# Directories
PHASE2_DIR = Path('/home/user/GLMHMM/results/phase2_reversal')
MODELS_DIR = PHASE2_DIR / 'models'
OUTPUT_DIR = PHASE2_DIR / 'individual_learning_curves'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target mice
TARGET_MICE = {
    '82': {'cohort': 'F', 'label': 'Mouse 82 (Cohort F)'},
    'c1m2': {'cohort': 'W', 'label': 'Mouse c1m2 (Cohort W)'}
}

# State colors (matching Phase 1 style)
STATE_COLORS = ['#5790C4', '#F5A86B', '#73C17C']  # Blue, Orange, Green


def load_phase2_model_data(animal_id, cohort):
    """
    Load Phase 2 reversal model data for a specific animal.

    Args:
        animal_id: Animal identifier (e.g., '82', 'c1m2')
        cohort: Cohort letter ('W' or 'F')

    Returns:
        Dictionary with model data or None if file not found
    """
    model_file = MODELS_DIR / f'{animal_id}_cohort{cohort}_reversal.pkl'

    if not model_file.exists():
        print(f"ERROR: Model file not found: {model_file}")
        return None

    try:
        with open(model_file, 'rb') as f:
            data = pickle.load(f)

        print(f"\n✓ Loaded Phase 2 data for {animal_id}:")
        print(f"  Genotype: {data.get('genotype', 'Unknown')}")
        print(f"  Total trials: {len(data.get('y', []))}")
        print(f"  Cohort: {cohort}")

        return data

    except Exception as e:
        print(f"ERROR loading {model_file}: {e}")
        return None


def compute_rolling_accuracy(correct_trials, window=50, min_periods=10):
    """
    Compute rolling average accuracy.

    Args:
        correct_trials: Binary array of correct/incorrect trials
        window: Rolling window size (default: 50 trials)
        min_periods: Minimum periods for computation (default: 10)

    Returns:
        Pandas Series with rolling accuracy
    """
    return pd.Series(correct_trials).rolling(
        window=window,
        min_periods=min_periods,
        center=True
    ).mean()


def plot_phase2_learning_curve(animal_id, cohort_letter, data, save_dir):
    """
    Create Phase 2 learning curve with state occupancy.
    Matches Phase 1 style exactly.

    Args:
        animal_id: Animal identifier
        cohort_letter: Cohort ('W' or 'F')
        data: Model data dictionary
        save_dir: Directory to save figures
    """
    # Extract data
    y = data.get('y')  # Binary correct/incorrect
    model = data.get('model')
    genotype = data.get('genotype', 'Unknown')
    state_metrics = data.get('state_metrics')
    broad_categories = data.get('broad_categories', {})

    if y is None or model is None:
        print(f"ERROR: Missing required data for {animal_id}")
        return

    n_trials = len(y)

    # Get state sequence
    if hasattr(model, 'most_likely_states'):
        states = model.most_likely_states
    else:
        print(f"ERROR: Model has no state sequence for {animal_id}")
        return

    # Get state probabilities (for smooth visualization)
    if hasattr(model, 'state_probabilities'):
        state_probs = model.state_probabilities
    else:
        # Fallback: create one-hot encoding from most likely states
        n_states = model.n_states
        state_probs = np.zeros((n_trials, n_states))
        for i, s in enumerate(states):
            state_probs[i, s] = 1.0

    # Extract state labels with accuracy information
    state_labels = {}
    for state in range(model.n_states):
        # Get broad category label
        category = 'Unknown'
        for key, value in broad_categories.items():
            if int(key) == state:
                category = value[0]  # First element is the category
                break

        # Get accuracy for this state
        accuracy = np.nan
        if state_metrics is not None and isinstance(state_metrics, pd.DataFrame):
            state_row = state_metrics[state_metrics['state'] == state]
            if len(state_row) > 0:
                accuracy = state_row['accuracy'].values[0]

        # Create informative label
        if not np.isnan(accuracy):
            state_labels[state] = f"State {state}: {category}\n(Acc: {accuracy:.2f})"
        else:
            state_labels[state] = f"State {state}: {category}"

    # Compute rolling accuracy
    rolling_acc = compute_rolling_accuracy(y, window=50, min_periods=10)

    # Create figure with 2 subplots (matching Phase 1 style)
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )

    trials = np.arange(n_trials)

    # ========================================
    # Top Panel: Learning Curve
    # ========================================
    ax1.plot(trials, rolling_acc, linewidth=2, color='black', label='Accuracy')
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.set_title(
        f'Phase 2 Learning Curve: {animal_id} - Cohort {cohort_letter}\n' +
        f'Genotype: {genotype} | Reversal Learning',
        fontsize=14,
        fontweight='bold'
    )
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # ========================================
    # Bottom Panel: State Occupancy
    # ========================================
    # Stack states from bottom to top
    cumulative = np.zeros(n_trials)

    for state in range(model.n_states):
        ax2.fill_between(
            trials,
            cumulative,
            cumulative + state_probs[:, state],
            color=STATE_COLORS[state % len(STATE_COLORS)],
            alpha=0.7,
            label=state_labels.get(state, f'State {state}'),
            edgecolor='none'
        )
        cumulative += state_probs[:, state]

    ax2.set_xlabel('Trial Number', fontsize=12)
    ax2.set_ylabel('State Probability', fontsize=12)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # ========================================
    # Save Figure
    # ========================================
    plt.tight_layout()

    # Save as PNG (high resolution)
    png_path = save_dir / f'{animal_id}_phase2_learning_curve.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')

    # Save as PDF (publication quality)
    pdf_path = save_dir / f'{animal_id}_phase2_learning_curve.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

    plt.close()

    print(f"  ✓ Saved learning curves:")
    print(f"    - {png_path.name}")
    print(f"    - {pdf_path.name}")

    return png_path, pdf_path


def create_combined_comparison_figure(data_dict, save_dir):
    """
    Create a combined figure showing both mice side by side.

    Args:
        data_dict: Dictionary with animal_id as keys, model data as values
        save_dir: Directory to save figures
    """
    n_animals = len(data_dict)

    if n_animals == 0:
        print("No data to create combined figure")
        return

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, n_animals, hspace=0.3, wspace=0.3, height_ratios=[2, 1])

    for col, (animal_id, data) in enumerate(data_dict.items()):
        cohort = TARGET_MICE[animal_id]['cohort']
        genotype = data.get('genotype', 'Unknown')
        y = data.get('y')
        model = data.get('model')
        state_metrics = data.get('state_metrics')
        broad_categories = data.get('broad_categories', {})

        if y is None or model is None:
            continue

        n_trials = len(y)
        trials = np.arange(n_trials)

        # Get state sequence and probabilities
        if hasattr(model, 'most_likely_states'):
            states = model.most_likely_states
        else:
            continue

        if hasattr(model, 'state_probabilities'):
            state_probs = model.state_probabilities
        else:
            n_states = model.n_states
            state_probs = np.zeros((n_trials, n_states))
            for i, s in enumerate(states):
                state_probs[i, s] = 1.0

        # Extract state labels with accuracy information
        state_labels = {}
        for state in range(model.n_states):
            # Get broad category label
            category = 'Unknown'
            for key, value in broad_categories.items():
                if int(key) == state:
                    category = value[0]  # First element is the category
                    break

            # Get accuracy for this state
            accuracy = np.nan
            if state_metrics is not None and isinstance(state_metrics, pd.DataFrame):
                state_row = state_metrics[state_metrics['state'] == state]
                if len(state_row) > 0:
                    accuracy = state_row['accuracy'].values[0]

            # Create informative label
            if not np.isnan(accuracy):
                state_labels[state] = f"S{state}: {category}\n(Acc: {accuracy:.2f})"
            else:
                state_labels[state] = f"S{state}: {category}"

        # Compute rolling accuracy
        rolling_acc = compute_rolling_accuracy(y, window=50, min_periods=10)

        # Top panel: Learning curve
        ax1 = fig.add_subplot(gs[0, col])
        ax1.plot(trials, rolling_acc, linewidth=2.5, color='black')
        ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1.0)
        ax1.set_title(
            f'{animal_id} (Cohort {cohort})\n{genotype}',
            fontsize=13,
            fontweight='bold'
        )
        ax1.grid(True, alpha=0.3)

        # Bottom panel: State occupancy
        ax2 = fig.add_subplot(gs[1, col])
        cumulative = np.zeros(n_trials)

        for state in range(model.n_states):
            ax2.fill_between(
                trials,
                cumulative,
                cumulative + state_probs[:, state],
                color=STATE_COLORS[state % len(STATE_COLORS)],
                alpha=0.7,
                label=state_labels.get(state, f'State {state}'),
                edgecolor='none'
            )
            cumulative += state_probs[:, state]

        ax2.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('State Probability', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1.0)
        if col == n_animals - 1:  # Only show legend on rightmost plot
            ax2.legend(loc='upper right', framealpha=0.9, fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        'Phase 2 Reversal Learning: Individual Sample Mice\nLearning Curves with State Occupancy',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    # Save combined figure
    png_path = save_dir / 'phase2_sample_mice_combined.png'
    pdf_path = save_dir / 'phase2_sample_mice_combined.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

    plt.close()

    print(f"\n✓ Saved combined figure:")
    print(f"  - {png_path.name}")
    print(f"  - {pdf_path.name}")

    return png_path, pdf_path


def generate_summary_statistics(data_dict, save_dir):
    """
    Generate summary statistics for the sample mice.

    Args:
        data_dict: Dictionary with animal_id as keys, model data as values
        save_dir: Directory to save summary
    """
    summary_data = []

    for animal_id, data in data_dict.items():
        cohort = TARGET_MICE[animal_id]['cohort']
        genotype = data.get('genotype', 'Unknown')
        y = data.get('y', [])
        model = data.get('model')
        state_metrics = data.get('state_metrics')
        broad_categories = data.get('broad_categories', {})

        if len(y) == 0 or model is None:
            continue

        # Overall accuracy
        overall_acc = np.mean(y)

        # State occupancy and characterization
        state_info = {}
        if hasattr(model, 'most_likely_states'):
            states = model.most_likely_states
            state_counts = np.bincount(states, minlength=model.n_states)
            state_occupancy = state_counts / len(states)

            for state in range(model.n_states):
                # Get category
                category = 'Unknown'
                for key, value in broad_categories.items():
                    if int(key) == state:
                        category = value[0]
                        break

                # Get accuracy
                state_acc = np.nan
                if state_metrics is not None and isinstance(state_metrics, pd.DataFrame):
                    state_row = state_metrics[state_metrics['state'] == state]
                    if len(state_row) > 0:
                        state_acc = state_row['accuracy'].values[0]

                state_info[f'State_{state}_Label'] = category
                state_info[f'State_{state}_Occupancy'] = state_occupancy[state]
                state_info[f'State_{state}_Accuracy'] = state_acc
        else:
            for state in range(3):
                state_info[f'State_{state}_Label'] = 'Unknown'
                state_info[f'State_{state}_Occupancy'] = np.nan
                state_info[f'State_{state}_Accuracy'] = np.nan

        # Early vs late performance
        n_trials = len(y)
        early_trials = n_trials // 4
        late_trials = n_trials - early_trials

        early_acc = np.mean(y[:early_trials])
        late_acc = np.mean(y[-early_trials:])

        row_data = {
            'Animal_ID': animal_id,
            'Cohort': cohort,
            'Genotype': genotype,
            'Total_Trials': n_trials,
            'Overall_Accuracy': overall_acc,
            'Early_Accuracy': early_acc,
            'Late_Accuracy': late_acc,
            'Accuracy_Change': late_acc - early_acc,
        }
        row_data.update(state_info)
        summary_data.append(row_data)

    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    csv_path = save_dir / 'phase2_sample_mice_summary.csv'
    summary_df.to_csv(csv_path, index=False, float_format='%.4f')

    print(f"\n✓ Saved summary statistics: {csv_path.name}")

    # Print summary table
    print("\n" + "="*80)
    print("PHASE 2 SAMPLE MICE SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

    return summary_df


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("PHASE 2 INDIVIDUAL LEARNING CURVES")
    print("Creating publication-quality figures for sample mice")
    print("="*80)

    # Load data for target mice
    loaded_data = {}

    for animal_id, info in TARGET_MICE.items():
        cohort = info['cohort']
        print(f"\n[{animal_id}] Loading Phase 2 data...")

        data = load_phase2_model_data(animal_id, cohort)

        if data is not None:
            loaded_data[animal_id] = data
        else:
            print(f"  ✗ Failed to load data for {animal_id}")

    if len(loaded_data) == 0:
        print("\nERROR: No data could be loaded. Exiting.")
        return

    print(f"\n✓ Successfully loaded {len(loaded_data)}/{len(TARGET_MICE)} mice")

    # Create individual figures
    print("\n" + "-"*80)
    print("Creating individual learning curves...")
    print("-"*80)

    for animal_id, data in loaded_data.items():
        cohort = TARGET_MICE[animal_id]['cohort']
        print(f"\n[{animal_id}] Generating figure...")

        plot_phase2_learning_curve(animal_id, cohort, data, OUTPUT_DIR)

    # Create combined comparison figure
    print("\n" + "-"*80)
    print("Creating combined comparison figure...")
    print("-"*80)

    create_combined_comparison_figure(loaded_data, OUTPUT_DIR)

    # Generate summary statistics
    print("\n" + "-"*80)
    print("Generating summary statistics...")
    print("-"*80)

    generate_summary_statistics(loaded_data, OUTPUT_DIR)

    # Final summary
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob('*')):
        print(f"  ✓ {f.name}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
