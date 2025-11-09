"""
Reversal Detection Module

Detects reversal points in LD reversal tasks using criterion achievement data.

W Cohort: A_Mouse LD 1 choice reversal v3
  - First reversal: 7/8 consecutive correct
  - Second reversal: 5/6 correct in 6-trial blocks
  - Some animals achieve 2 reversals in one session

F Cohort: A_Mouse LD 1 Reversal 9
  - Single reversal: first criterion achievement
  - Usually only one reversal per session

Author: Claude (Anthropic)
Date: November 9, 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_reversal_criterion_data(ldr_filepath):
    """
    Load LDR file with criterion achievement data.

    Parameters:
    -----------
    ldr_filepath : str or Path
        Path to LDR criterion file

    Returns:
    --------
    ldr_df : DataFrame
        Reversal sessions with criterion columns
    """
    ldr_df = pd.read_csv(ldr_filepath)

    # Convert date to datetime for matching
    ldr_df['Schedule.run.date'] = pd.to_datetime(ldr_df['Schedule.run.date'])

    return ldr_df


def detect_reversals_single_session(session_data, ldr_session_data, cohort='W'):
    """
    Detect reversal points for a single session.

    Parameters:
    -----------
    session_data : DataFrame
        Trial-by-trial data for this session
    ldr_session_data : Series or dict
        LDR row for this session with criterion columns
    cohort : str
        'W' or 'F' (determines reversal rules)

    Returns:
    --------
    reversal_info : dict
        {
            'reversal_trials': list of trial numbers where reversals occurred,
            'stimulus_values': array of stimulus values (-1 or +1) for each trial,
            'n_reversals': int
        }
    """
    n_trials = len(session_data)

    # Get criterion trial numbers from LDR data
    first_reversal_trial = ldr_session_data.get('No trials to criterion - Generic Evaluation (1)', np.nan)
    second_reversal_trial = ldr_session_data.get('No.trials.to.criterion...Generic.Evaluation..2.', np.nan)

    # Handle missing or invalid values
    if pd.isna(first_reversal_trial):
        first_reversal_trial = None
    else:
        first_reversal_trial = int(first_reversal_trial)

    if pd.isna(second_reversal_trial):
        second_reversal_trial = None
    else:
        second_reversal_trial = int(second_reversal_trial)

    # Initialize stimulus values
    # Both cohorts start with position 8 (left) correct = stimulus -1
    stimulus_values = np.full(n_trials, -1.0)
    reversal_trials = []

    current_correct_side = -1  # Start with left correct

    # Apply reversals
    for trial_idx in range(n_trials):
        trial_num = trial_idx + 1  # Trial numbers start at 1

        # Check if first reversal occurs at this trial
        if first_reversal_trial is not None and trial_num == first_reversal_trial:
            current_correct_side = +1  # Flip to right correct
            reversal_trials.append(trial_num)

        # Check if second reversal occurs (W cohort only, usually)
        if second_reversal_trial is not None and trial_num == second_reversal_trial:
            current_correct_side = -1  # Flip back to left correct
            reversal_trials.append(trial_num)

        stimulus_values[trial_idx] = current_correct_side

    reversal_info = {
        'reversal_trials': reversal_trials,
        'stimulus_values': stimulus_values,
        'n_reversals': len(reversal_trials),
        'first_reversal_trial': first_reversal_trial,
        'second_reversal_trial': second_reversal_trial
    }

    return reversal_info


def add_reversal_info_to_trials(trial_df, ldr_filepath, cohort='W'):
    """
    Add reversal information to trial dataframe.

    Updates stimulus values for reversal sessions based on LDR criterion data.

    Parameters:
    -----------
    trial_df : DataFrame
        Trial-by-trial data (all animals, all sessions)
    ldr_filepath : str or Path
        Path to LDR criterion file
    cohort : str
        'W' or 'F'

    Returns:
    --------
    trial_df_updated : DataFrame
        Trial data with added columns:
        - 'reversal_session': bool, is this a reversal session?
        - 'reversal_trial': int or NaN, trial number of reversal (if in reversal session)
        - 'trials_since_reversal': int or NaN, trials since last reversal
        - 'pre_reversal': bool, before first reversal
        - 'post_reversal': bool, after first reversal
        - 'stimulus_corrected': float, corrected stimulus value accounting for reversals
    """
    # Load LDR data
    ldr_df = load_reversal_criterion_data(ldr_filepath)

    # Convert trial_df dates to datetime
    trial_df = trial_df.copy()
    trial_df['session_date'] = pd.to_datetime(trial_df['session_date'])

    # Initialize new columns
    trial_df['reversal_session'] = False
    trial_df['reversal_trial'] = np.nan
    trial_df['trials_since_reversal'] = np.nan
    trial_df['pre_reversal'] = False
    trial_df['post_reversal'] = False
    trial_df['stimulus_corrected'] = trial_df.get('position', np.nan).copy()  # Will update for reversals
    trial_df['n_reversals_in_session'] = 0

    # Process each reversal session
    for idx, ldr_row in ldr_df.iterrows():
        animal_id = ldr_row['Animal ID']
        session_date = pd.to_datetime(ldr_row['Schedule.run.date'])

        # Find matching session in trial_df
        session_mask = (
            (trial_df['animal_id'] == animal_id) &
            (trial_df['session_date'] == session_date) &
            (trial_df['task_type'].isin(['LD_reversal', 'LD Reversal']))
        )

        if not session_mask.any():
            # Try matching by animal and approximate date (within 1 day)
            date_mask = (trial_df['session_date'] >= session_date - pd.Timedelta(days=1)) & \
                       (trial_df['session_date'] <= session_date + pd.Timedelta(days=1))
            session_mask = (
                (trial_df['animal_id'] == animal_id) &
                date_mask &
                (trial_df['task_type'].isin(['LD_reversal', 'LD Reversal']))
            )

        if not session_mask.any():
            continue  # No matching session found

        session_data = trial_df[session_mask].copy()

        # Detect reversals for this session
        reversal_info = detect_reversals_single_session(session_data, ldr_row, cohort=cohort)

        # Update trial_df with reversal information
        session_indices = trial_df[session_mask].index

        trial_df.loc[session_indices, 'reversal_session'] = True
        trial_df.loc[session_indices, 'n_reversals_in_session'] = reversal_info['n_reversals']

        # Assign reversal trial numbers and compute trials_since_reversal
        first_rev = reversal_info['first_reversal_trial']
        second_rev = reversal_info['second_reversal_trial']

        for i, idx in enumerate(session_indices):
            trial_num_in_session = i + 1

            # Mark reversal trials
            if first_rev is not None and trial_num_in_session == first_rev:
                trial_df.loc[idx, 'reversal_trial'] = first_rev
            elif second_rev is not None and trial_num_in_session == second_rev:
                trial_df.loc[idx, 'reversal_trial'] = second_rev

            # Mark pre/post reversal
            if first_rev is not None:
                if trial_num_in_session < first_rev:
                    trial_df.loc[idx, 'pre_reversal'] = True
                    trial_df.loc[idx, 'trials_since_reversal'] = first_rev - trial_num_in_session  # Negative = before
                else:
                    trial_df.loc[idx, 'post_reversal'] = True

                    # Trials since most recent reversal
                    if second_rev is not None and trial_num_in_session >= second_rev:
                        trial_df.loc[idx, 'trials_since_reversal'] = trial_num_in_session - second_rev
                    else:
                        trial_df.loc[idx, 'trials_since_reversal'] = trial_num_in_session - first_rev

            # Update stimulus value (corrected for reversals)
            trial_df.loc[idx, 'stimulus_corrected'] = reversal_info['stimulus_values'][i]

    return trial_df


def validate_reversal_detection(trial_df, ldr_filepath):
    """
    Validate that reversal detection matches LDR file's Second_Criterion_Count.

    Parameters:
    -----------
    trial_df : DataFrame
        Trial data with reversal info added
    ldr_filepath : str or Path
        Path to LDR file

    Returns:
    --------
    validation_df : DataFrame
        Comparison of detected vs expected reversals
    """
    ldr_df = load_reversal_criterion_data(ldr_filepath)
    ldr_df['session_date'] = pd.to_datetime(ldr_df['Schedule.run.date'])

    validation_results = []

    for idx, ldr_row in ldr_df.iterrows():
        animal_id = ldr_row['Animal ID']
        session_date = ldr_row['session_date']
        expected_second_crit = ldr_row.get('Second_Criterion_Count', 0)

        # Find matching session
        session_mask = (
            (trial_df['animal_id'] == animal_id) &
            (trial_df['session_date'] == session_date) &
            (trial_df['reversal_session'] == True)
        )

        if session_mask.any():
            session_data = trial_df[session_mask]
            detected_reversals = session_data['n_reversals_in_session'].iloc[0]

            # Second_Criterion_Count counts 5/6 blocks after first reversal
            # Our n_reversals counts total reversal events
            # So: n_reversals = 0 → Second_Criterion_Count should be 0
            #     n_reversals = 1 → Second_Criterion_Count should be 0 (only first reversal)
            #     n_reversals = 2 → Second_Criterion_Count should be 1 (second reversal occurred)

            expected_total_reversals = 1 + expected_second_crit
            match = detected_reversals == expected_total_reversals

            validation_results.append({
                'animal_id': animal_id,
                'session_date': session_date,
                'detected_reversals': detected_reversals,
                'expected_second_criterion': expected_second_crit,
                'expected_total_reversals': expected_total_reversals,
                'match': match
            })

    validation_df = pd.DataFrame(validation_results)

    # Summary
    if len(validation_df) > 0:
        n_matches = validation_df['match'].sum()
        n_total = len(validation_df)
        print(f"Reversal Detection Validation:")
        print(f"  Matches: {n_matches}/{n_total} ({n_matches/n_total*100:.1f}%)")

        if n_matches < n_total:
            print(f"  Mismatches:")
            mismatches = validation_df[~validation_df['match']]
            for idx, row in mismatches.iterrows():
                print(f"    {row['animal_id']} {row['session_date'].date()}: "
                      f"detected={row['detected_reversals']}, expected={row['expected_total_reversals']}")

    return validation_df


def compute_reversal_adaptation_metrics(trial_df):
    """
    Compute adaptation metrics for reversal sessions.

    Measures:
    - Trials to criterion after reversal
    - Accuracy before vs after reversal
    - State transitions around reversal points

    Parameters:
    -----------
    trial_df : DataFrame
        Trial data with reversal info

    Returns:
    --------
    adaptation_metrics : DataFrame
        Per-session adaptation metrics
    """
    reversal_sessions = trial_df[trial_df['reversal_session'] == True].copy()

    metrics = []

    for (animal_id, session_date), session_data in reversal_sessions.groupby(['animal_id', 'session_date']):
        session_data = session_data.sort_values('trial_num')

        # Get first reversal trial
        first_reversal = session_data['reversal_trial'].dropna()
        if len(first_reversal) == 0:
            continue

        first_rev_trial = first_reversal.iloc[0]

        # Pre-reversal accuracy (trials before reversal)
        pre_rev = session_data[session_data['trial_num'] < first_rev_trial]
        pre_accuracy = pre_rev['correct'].mean() if len(pre_rev) > 0 else np.nan

        # Post-reversal accuracy (first 10 trials after reversal)
        post_rev = session_data[session_data['trial_num'] >= first_rev_trial]
        post_rev_10 = post_rev.iloc[:10] if len(post_rev) >= 10 else post_rev
        post_accuracy_10 = post_rev_10['correct'].mean() if len(post_rev_10) > 0 else np.nan

        # Full post-reversal accuracy
        post_accuracy_all = post_rev['correct'].mean() if len(post_rev) > 0 else np.nan

        # Trials to reach 70% after reversal (using 10-trial rolling window)
        if len(post_rev) >= 10:
            rolling_acc = post_rev['correct'].rolling(window=10, min_periods=5).mean()
            criterion_met = rolling_acc >= 0.7
            if criterion_met.any():
                trials_to_criterion = criterion_met.idxmax() - post_rev.index[0] + 1
            else:
                trials_to_criterion = np.nan
        else:
            trials_to_criterion = np.nan

        # Number of reversals in this session
        n_reversals = session_data['n_reversals_in_session'].iloc[0]

        metrics.append({
            'animal_id': animal_id,
            'session_date': session_date,
            'first_reversal_trial': first_rev_trial,
            'n_reversals': n_reversals,
            'pre_reversal_accuracy': pre_accuracy,
            'post_reversal_accuracy_first10': post_accuracy_10,
            'post_reversal_accuracy_all': post_accuracy_all,
            'trials_to_70pct_criterion': trials_to_criterion,
            'adaptation_drop': pre_accuracy - post_accuracy_10 if not np.isnan(pre_accuracy) and not np.isnan(post_accuracy_10) else np.nan
        })

    return pd.DataFrame(metrics)


def create_reversal_design_matrix(trial_df, animal_id, ldr_filepath, cohort='W',
                                  include_position=True, include_session_progression=True):
    """
    Create design matrix specifically for reversal task analysis.

    This function:
    1. Filters to reversal sessions only
    2. Uses CORRECTED stimulus values (accounting for reversals)
    3. Adds trials_since_reversal as a feature

    Parameters:
    -----------
    trial_df : DataFrame
        Trial data (should already have reversal info added via add_reversal_info_to_trials)
    animal_id : str
        Animal to analyze
    ldr_filepath : str
        Path to LDR file (for validation)
    cohort : str
        'W' or 'F'
    include_position : bool
        Include position features
    include_session_progression : bool
        Include session progression

    Returns:
    --------
    X : ndarray
        Design matrix with corrected stimulus
    y : ndarray
        Binary choices
    feature_names : list
        Feature names
    metadata : dict
        Trial metadata
    trial_data : DataFrame
        Subset of trials used
    """
    # Import here to avoid circular dependency
    from glmhmm_utils import create_design_matrix

    # Filter to this animal and reversal sessions only
    animal_data = trial_df[
        (trial_df['animal_id'] == animal_id) &
        (trial_df['reversal_session'] == True)
    ].copy()

    if len(animal_data) == 0:
        raise ValueError(f"No reversal sessions found for animal {animal_id}")

    # Reset index
    animal_data = animal_data.sort_values(['session_date', 'trial_num']).reset_index(drop=True)

    # Create design matrix using standard function
    # But we'll replace stimulus with corrected values
    X, y, feature_names, metadata, data = create_design_matrix(
        trial_df,
        animal_id=animal_id,
        include_position=include_position,
        include_session_progression=include_session_progression
    )

    # Replace stimulus with corrected stimulus (only for reversal sessions)
    # Find reversal session trials in the full dataset
    reversal_mask = data.index.isin(animal_data.index)

    if 'stimulus_correct_side' in feature_names:
        stim_idx = feature_names.index('stimulus_correct_side')
        # Update stimulus values for reversal sessions
        X[reversal_mask, stim_idx] = animal_data['stimulus_corrected'].values

    # Add trials_since_reversal as a feature (optional)
    # Normalize to 0-1 range
    if reversal_mask.any():
        trials_since_rev = animal_data['trials_since_reversal'].values
        # Normalize: negative values (pre-reversal) map to 0, positive map to 0-1
        max_trials = max(abs(trials_since_rev.min()), trials_since_rev.max())
        if max_trials > 0:
            trials_since_rev_norm = (trials_since_rev + max_trials) / (2 * max_trials)
        else:
            trials_since_rev_norm = np.zeros_like(trials_since_rev)

        # Add as feature only for reversal trials
        trials_since_feature = np.zeros(len(X))
        trials_since_feature[reversal_mask] = trials_since_rev_norm

        X = np.column_stack([X, trials_since_feature])
        feature_names.append('trials_since_reversal')

    # Add reversal metadata
    metadata['reversal_session'] = data['reversal_session'].values if 'reversal_session' in data.columns else np.zeros(len(data), dtype=bool)
    metadata['trials_since_reversal'] = data['trials_since_reversal'].values if 'trials_since_reversal' in data.columns else np.full(len(data), np.nan)
    metadata['pre_reversal'] = data['pre_reversal'].values if 'pre_reversal' in data.columns else np.zeros(len(data), dtype=bool)
    metadata['post_reversal'] = data['post_reversal'].values if 'post_reversal' in data.columns else np.zeros(len(data), dtype=bool)

    return X, y, feature_names, metadata, data
