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
    '82': {'cohort': 'F', 'cohort_display': 'Cohort 1', 'label': 'Mouse 82 (Cohort 1)'},
    'c1m2': {'cohort': 'W', 'cohort_display': 'Cohort 2', 'label': 'Mouse c1m2 (Cohort 2)'}
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
    Create Phase 2 learning curve colored by state with info table.
    NEW STYLE: Learning curve line colored by state + stats table.

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
    cohort_display = TARGET_MICE[animal_id]['cohort_display']

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

    # Extract state info
    state_info = []
    for state in range(model.n_states):
        # Get broad category label
        category = 'Unknown'
        for key, value in broad_categories.items():
            if int(key) == state:
                category = value[0]
                break

        # Get accuracy for this state
        accuracy = np.nan
        occupancy = 0.0
        if state_metrics is not None and isinstance(state_metrics, pd.DataFrame):
            state_row = state_metrics[state_metrics['state'] == state]
            if len(state_row) > 0:
                accuracy = state_row['accuracy'].values[0]
                occupancy = state_row['occupancy'].values[0]

        state_info.append({
            'state': state,
            'category': category,
            'accuracy': accuracy,
            'occupancy': occupancy
        })

    # Compute rolling accuracy
    rolling_acc = compute_rolling_accuracy(y, window=50, min_periods=10)

    # Create figure with gridspec for learning curve + table
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[3, 0.15, 1.2], wspace=0.4)

    ax_curve = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 2])

    trials = np.arange(n_trials)

    # ========================================
    # Left Panel: Learning Curve Colored by State
    # ========================================

    # Plot learning curve with segments colored by state
    for i in range(len(trials) - 1):
        if not np.isnan(rolling_acc.iloc[i]) and not np.isnan(rolling_acc.iloc[i+1]):
            current_state = states[i]
            color = STATE_COLORS[current_state % len(STATE_COLORS)]
            ax_curve.plot(
                [trials[i], trials[i+1]],
                [rolling_acc.iloc[i], rolling_acc.iloc[i+1]],
                color=color,
                linewidth=2.5,
                alpha=0.8
            )

    # Add data points to show individual trials
    # Sample points to avoid overcrowding (every 5th trial)
    sample_interval = max(1, n_trials // 100)  # Show ~100 points max
    for i in range(0, len(trials), sample_interval):
        if not np.isnan(rolling_acc.iloc[i]):
            current_state = states[i]
            color = STATE_COLORS[current_state % len(STATE_COLORS)]
            ax_curve.scatter(
                trials[i],
                rolling_acc.iloc[i],
                color=color,
                s=20,
                alpha=0.6,
                edgecolors='white',
                linewidth=0.5,
                zorder=3
            )

    # Add chance line
    ax_curve.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Chance')

    ax_curve.set_xlabel('Trial Number', fontsize=13, fontweight='bold')
    ax_curve.set_ylabel('Accuracy (50-trial rolling avg)', fontsize=13, fontweight='bold')
    ax_curve.set_ylim(0, 1.0)
    ax_curve.set_title(
        f'Phase 2 Learning Curve: {animal_id} ({cohort_display})\n' +
        f'Genotype: {genotype} | Reversal Learning',
        fontsize=15,
        fontweight='bold',
        pad=15
    )
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc='best', framealpha=0.9)

    # ========================================
    # Right Panel: State Info Table
    # ========================================
    ax_table.axis('off')

    # Build table data
    table_data = []
    table_data.append(['State', 'Label', 'Occupancy', 'Accuracy'])

    for info in state_info:
        state_num = info['state']
        category = info['category']
        occ = info['occupancy']
        acc = info['accuracy']

        occ_str = f"{occ*100:.1f}%" if not np.isnan(occ) else "N/A"
        acc_str = f"{acc*100:.1f}%" if not np.isnan(acc) else "N/A"

        table_data.append([
            f"{state_num}",
            category,
            occ_str,
            acc_str
        ])

    # Create table with better spacing
    table = ax_table.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.18, 0.32, 0.25, 0.25]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3.0)  # Increased vertical spacing

    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#40466e')
        cell.set_text_props(weight='bold', color='white', fontsize=13)
        cell.set_height(0.12)

    # Style data rows with state colors
    for row in range(1, len(table_data)):
        state_idx = row - 1
        color = STATE_COLORS[state_idx % len(STATE_COLORS)]

        # State number cell gets the state color
        table[(row, 0)].set_facecolor(color)
        table[(row, 0)].set_text_props(weight='bold', fontsize=13)
        table[(row, 0)].set_height(0.1)

        # Other cells get light gray
        for col in range(1, 4):
            table[(row, col)].set_facecolor('#f0f0f0')
            table[(row, col)].set_text_props(fontsize=12)
            table[(row, col)].set_height(0.1)

    ax_table.set_title('State Statistics', fontsize=14, fontweight='bold', pad=25)

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
    NEW STYLE: Each mouse has colored learning curve + info table.

    Args:
        data_dict: Dictionary with animal_id as keys, model data as values
        save_dir: Directory to save figures
    """
    n_animals = len(data_dict)

    if n_animals == 0:
        print("No data to create combined figure")
        return

    # New layout: each animal gets 2 columns (curve + table)
    fig = plt.figure(figsize=(22, 8))
    gs = fig.add_gridspec(1, n_animals * 3, wspace=0.5, width_ratios=[3, 0.15, 1.3] * n_animals)

    for idx, (animal_id, data) in enumerate(data_dict.items()):
        cohort_display = TARGET_MICE[animal_id]['cohort_display']
        genotype = data.get('genotype', 'Unknown')
        y = data.get('y')
        model = data.get('model')
        state_metrics = data.get('state_metrics')
        broad_categories = data.get('broad_categories', {})

        if y is None or model is None:
            continue

        n_trials = len(y)
        trials = np.arange(n_trials)

        # Get state sequence
        if hasattr(model, 'most_likely_states'):
            states = model.most_likely_states
        else:
            continue

        # Extract state info
        state_info = []
        for state in range(model.n_states):
            category = 'Unknown'
            for key, value in broad_categories.items():
                if int(key) == state:
                    category = value[0]
                    break

            accuracy = np.nan
            occupancy = 0.0
            if state_metrics is not None and isinstance(state_metrics, pd.DataFrame):
                state_row = state_metrics[state_metrics['state'] == state]
                if len(state_row) > 0:
                    accuracy = state_row['accuracy'].values[0]
                    occupancy = state_row['occupancy'].values[0]

            state_info.append({
                'state': state,
                'category': category,
                'accuracy': accuracy,
                'occupancy': occupancy
            })

        # Compute rolling accuracy
        rolling_acc = compute_rolling_accuracy(y, window=50, min_periods=10)

        # Calculate subplot positions
        col_start = idx * 3

        # Learning curve (colored by state)
        ax_curve = fig.add_subplot(gs[0, col_start])

        # Plot learning curve with segments colored by state
        for i in range(len(trials) - 1):
            if not np.isnan(rolling_acc.iloc[i]) and not np.isnan(rolling_acc.iloc[i+1]):
                current_state = states[i]
                color = STATE_COLORS[current_state % len(STATE_COLORS)]
                ax_curve.plot(
                    [trials[i], trials[i+1]],
                    [rolling_acc.iloc[i], rolling_acc.iloc[i+1]],
                    color=color,
                    linewidth=2.5,
                    alpha=0.8
                )

        # Add data points
        sample_interval = max(1, n_trials // 100)
        for i in range(0, len(trials), sample_interval):
            if not np.isnan(rolling_acc.iloc[i]):
                current_state = states[i]
                color = STATE_COLORS[current_state % len(STATE_COLORS)]
                ax_curve.scatter(
                    trials[i],
                    rolling_acc.iloc[i],
                    color=color,
                    s=18,
                    alpha=0.6,
                    edgecolors='white',
                    linewidth=0.5,
                    zorder=3
                )

        ax_curve.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax_curve.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax_curve.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        ax_curve.set_ylim(0, 1.0)
        ax_curve.set_title(
            f'{animal_id} ({cohort_display})\n{genotype}',
            fontsize=13,
            fontweight='bold'
        )
        ax_curve.grid(True, alpha=0.3)

        # Info table
        ax_table = fig.add_subplot(gs[0, col_start + 2])
        ax_table.axis('off')

        # Build table
        table_data = [['State', 'Label', 'Occ.', 'Acc.']]
        for info in state_info:
            state_num = info['state']
            category = info['category']
            occ = info['occupancy']
            acc = info['accuracy']

            occ_str = f"{occ*100:.0f}%" if not np.isnan(occ) else "N/A"
            acc_str = f"{acc*100:.0f}%" if not np.isnan(acc) else "N/A"

            table_data.append([f"{state_num}", category, occ_str, acc_str])

        table = ax_table.table(
            cellText=table_data,
            cellLoc='center',
            loc='center',
            colWidths=[0.15, 0.35, 0.25, 0.25]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.8)

        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
            table[(0, i)].set_height(0.12)

        # Style data rows
        for row in range(1, len(table_data)):
            state_idx = row - 1
            color = STATE_COLORS[state_idx % len(STATE_COLORS)]
            table[(row, 0)].set_facecolor(color)
            table[(row, 0)].set_text_props(weight='bold', fontsize=12)
            table[(row, 0)].set_height(0.1)
            for col in range(1, 4):
                table[(row, col)].set_facecolor('#f0f0f0')
                table[(row, col)].set_text_props(fontsize=11)
                table[(row, col)].set_height(0.1)

        ax_table.set_title('State Stats', fontsize=12, fontweight='bold', pad=15)

    fig.suptitle(
        'Phase 2 Reversal Learning: Individual Sample Mice',
        fontsize=16,
        fontweight='bold',
        y=0.98
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
