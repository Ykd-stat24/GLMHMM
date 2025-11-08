"""
Utility functions for GLM-HMM analysis of touchscreen behavioral data.

This module provides functions for:
- Data preprocessing and design matrix creation
- Psychometric curve computation
- Visualization of GLM-HMM results

Author: Claude (Anthropic)
Date: 2025-11-08
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic
from sklearn.model_selection import KFold


def load_and_preprocess_session_data(filepath, tasks_to_include=None):
    """
    Load session-level touchscreen data and convert to trial-by-trial format.

    Parameters:
    -----------
    filepath : str
        Path to CSV file with session-level data
    tasks_to_include : list of str, optional
        Which schedule names to include (e.g., ['PI', 'LD', 'LD_reversal'])

    Returns:
    --------
    trial_df : DataFrame
        Trial-by-trial data with columns:
        - animal_id, session_date, session_index
        - task_type, is_reversal
        - trial_num, correct, chosen_side
        - position (grid position of stimulus)
        - latency, genotype, sex, etc.
    """
    df = pd.read_csv(filepath)

    # Default task filtering
    if tasks_to_include is None:
        # Include LD-related tasks
        ld_schedules = [s for s in df['Schedule name'].unique()
                       if 'LD' in s or 'Punish Incorrect' in s]
    else:
        ld_schedules = tasks_to_include

    df_filtered = df[df['Schedule name'].isin(ld_schedules)].copy()

    trials = []

    for idx, row in df_filtered.iterrows():
        animal_id = row['Animal ID']
        session_date = row['Schedule run date']
        schedule_name = row['Schedule name']

        # Determine task type
        if 'reversal' in schedule_name.lower():
            is_reversal = True
            task_type = 'LD_reversal'
        elif 'LD 1 choice' in schedule_name or 'LD Must Touch' in schedule_name or 'LD Initial Touch' in schedule_name:
            is_reversal = False
            task_type = 'LD'
        elif 'Punish Incorrect' in schedule_name:
            is_reversal = False
            if 'LD' in schedule_name:
                task_type = 'PI'
            else:
                task_type = 'PD_PI'  # Pairwise discrimination PI
        elif 'Pairwise Discrimination' in schedule_name:
            is_reversal = 'Reversal' in schedule_name
            task_type = 'PD' + ('_reversal' if is_reversal else '')
        else:
            is_reversal = False
            task_type = 'Unknown'

        # Extract trial-by-trial data (up to 30 trials per session)
        for trial_num in range(1, 31):
            # Trial result - NEW FORMAT: Use "Trial Analysis - No. Correct (X)"
            no_correct_col = f'Trial Analysis - No. Correct ({trial_num})'
            if no_correct_col not in df.columns:
                continue

            trial_result = row.get(no_correct_col, np.nan)
            if pd.isna(trial_result):
                continue

            correct = int(trial_result)  # Already 1 or 0

            # Position - NEW FORMAT
            position_col = f'Trial Analysis - Correct Position ({trial_num})'
            position = row.get(position_col, np.nan)

            # Latency - NEW FORMAT
            if correct:
                latency_col = f'Trial Analysis - Correct Image Response Latency ({trial_num})'
            else:
                latency_col = f'Trial Analysis - Incorrect Image Latency ({trial_num})'

            latency_val = row.get(latency_col, np.nan)
            # Handle '-' as missing data
            if latency_val == '-' or pd.isna(latency_val):
                latency = np.nan
            else:
                try:
                    latency = float(latency_val)
                except (ValueError, TypeError):
                    latency = np.nan

            # Determine correct side and chosen side based on POSITION
            # Position 1 or 8 = LEFT correct
            # Position 2 or 11 = RIGHT correct
            if pd.notna(position):
                if position in [1, 8]:
                    correct_side = 'left'
                    chosen_side = 'left' if correct else 'right'
                elif position in [2, 11]:
                    correct_side = 'right'
                    chosen_side = 'right' if correct else 'left'
                else:
                    # Training phases with other positions - assume left-biased
                    correct_side = 'left'
                    chosen_side = 'left' if correct else 'right'
            else:
                # Fallback if position missing (rare)
                correct_side = 'unknown'
                chosen_side = 'left' if correct else 'right'

            trial_data = {
                'animal_id': animal_id,
                'session_date': session_date,
                'schedule_name': schedule_name,
                'task_type': task_type,
                'is_reversal': is_reversal,
                'trial_num': trial_num,
                'correct': correct,
                'chosen_side': chosen_side,
                'correct_side': correct_side,
                'position': position,
                'latency': latency,
                'genotype': row.get('Genotype', np.nan),
                'sex': row.get('Sex', np.nan),
            }

            # Add session-level summaries
            trial_data['session_accuracy'] = row.get('End Summary - % Correct (1)', np.nan) / 100.0
            trial_data['left_iti_touches'] = row.get('End Summary - Left ITI Touches (1)', 0)
            trial_data['right_iti_touches'] = row.get('End Summary - Right ITI Touches (1)', 0)

            trials.append(trial_data)

    trial_df = pd.DataFrame(trials)

    # Add session index (chronological order per animal)
    trial_df['session_date'] = pd.to_datetime(trial_df['session_date'])
    trial_df = trial_df.sort_values(['animal_id', 'session_date', 'trial_num'])
    trial_df['session_index'] = trial_df.groupby('animal_id')['session_date'].transform(
        lambda x: pd.factorize(x)[0]
    )

    # Add day index (cumulative)
    trial_df['day'] = trial_df.groupby('animal_id').cumcount()

    return trial_df.reset_index(drop=True)


def create_design_matrix(trial_df, animal_id=None, include_position=False,
                        include_session_progression=False):
    """
    Create GLM design matrix following Ashwood et al. methodology.

    Features:
    1. Stimulus (task-dependent coding)
    2. Bias (constant term)
    3. Previous choice (-1 for left, +1 for right)
    4. Win-stay/lose-switch (previous choice × previous reward)
    5. [Optional] Position features
    6. [Optional] Session progression

    Parameters:
    -----------
    trial_df : DataFrame
        Trial-by-trial data
    animal_id : str, optional
        Filter to specific animal. If None, uses all data.
    include_position : bool
        Whether to include position as stimulus feature
    include_session_progression : bool
        Whether to include session index as feature

    Returns:
    --------
    X : ndarray (n_trials, n_features)
        Design matrix
    y : ndarray (n_trials,)
        Binary choices (0=left, 1=right)
    feature_names : list of str
        Names of features
    metadata : dict
        Additional trial information for analysis
    trial_data : DataFrame
        Subset of trial_df used
    """
    # Filter to specific animal if requested
    if animal_id is not None:
        data = trial_df[trial_df['animal_id'] == animal_id].copy()
    else:
        data = trial_df.copy()

    data = data.sort_values(['animal_id', 'session_date', 'trial_num']).reset_index(drop=True)
    n_trials = len(data)

    # Initialize feature list
    features = []
    feature_names = []

    # Feature 1: Stimulus - which side is CORRECT
    # CORRECTED: Stimulus represents the correct choice side
    # Position 8 or 1 = LEFT correct (stimulus = -1)
    # Position 11 or 2 = RIGHT correct (stimulus = +1)
    # This matches the user's task design:
    # - LD: position 8 (left), position 11 (right)
    # - PD: position 1 (left), position 2 (right)

    stimulus = np.zeros(n_trials)
    position = data['position'].values

    for i in range(n_trials):
        pos = position[i]
        if pd.isna(pos):
            # Default based on task type if position missing
            stimulus[i] = 1 if data.iloc[i]['is_reversal'] else -1
        elif pos in [1, 8]:
            # Positions 1 (PD) or 8 (LD) = LEFT correct
            stimulus[i] = -1
        elif pos in [2, 11]:
            # Positions 2 (PD) or 11 (LD) = RIGHT correct
            stimulus[i] = +1
        else:
            # Other positions (training phases) - infer from task
            # For LD training, usually left-biased (position 8)
            stimulus[i] = -1

    feature_names.append('stimulus_correct_side')

    features.append(stimulus)

    # Feature 2: Bias (constant term)
    bias = np.ones(n_trials)
    features.append(bias)
    feature_names.append('bias')

    # Feature 3: Previous choice
    prev_choice = np.zeros(n_trials)
    # Encode: left = -1, right = +1
    choice_encoding = np.where(data['chosen_side'] == 'right', 1, -1)
    prev_choice[1:] = choice_encoding[:-1]
    features.append(prev_choice)
    feature_names.append('prev_choice')

    # Feature 4: Win-stay/lose-switch
    # = previous_choice × (2 × previous_reward - 1)
    # Positive weight means win-stay/lose-shift
    prev_reward = np.zeros(n_trials)
    prev_reward[1:] = data['correct'].iloc[:-1].values
    wsls = prev_choice * (2 * prev_reward - 1)
    features.append(wsls)
    feature_names.append('wsls')

    # Feature 5: Session progression (optional)
    if include_session_progression:
        session_prog = data['session_index'].values / data['session_index'].max()
        features.append(session_prog)
        feature_names.append('session_progression')

    # Stack features
    X = np.column_stack(features)

    # Output: binary choice (0=left, 1=right)
    y = np.where(data['chosen_side'] == 'right', 1, 0)

    # Metadata for analysis
    metadata = {
        'correct': data['correct'].values,
        'latency': data['latency'].values,
        'task_type': data['task_type'].values,
        'is_reversal': data['is_reversal'].values,
        'session_index': data['session_index'].values,
        'session_date': data['session_date'].values,
        'animal_id': data['animal_id'].values,
        'genotype': data['genotype'].values,
    }

    return X, y, feature_names, metadata, data


def compute_psychometric_curves(model, X, y, metadata, stimulus_values=None, n_bins=7):
    """
    Compute psychometric curves for each state.

    A psychometric curve shows P(choose right) as a function of stimulus strength.

    Parameters:
    -----------
    model : GLMHMM
        Fitted GLM-HMM model
    X : ndarray
        Design matrix
    y : ndarray
        Choices
    metadata : dict
        Trial metadata
    stimulus_values : ndarray, optional
        Stimulus values for each trial (default: use first column of X)
    n_bins : int
        Number of bins for stimulus

    Returns:
    --------
    curves : dict
        Dictionary with keys 'state_0', 'state_1', etc.
        Each value is a DataFrame with columns: stimulus_bin, p_right, n_trials, sem
    """
    if stimulus_values is None:
        stimulus_values = X[:, 0]  # Assume first feature is stimulus

    states = model.most_likely_states
    curves = {}

    for state in range(model.n_states):
        state_mask = (states == state)

        if state_mask.sum() < 10:
            # Not enough data
            curves[f'state_{state}'] = pd.DataFrame({
                'stimulus_bin': [],
                'p_right': [],
                'n_trials': [],
                'sem': []
            })
            continue

        # Get stimulus and choices for this state
        stim_state = stimulus_values[state_mask]
        choice_state = y[state_mask]

        # Bin stimulus values
        if len(np.unique(stim_state)) <= n_bins:
            # Use unique values if few
            bins = np.unique(stim_state)
            bin_indices = np.digitize(stim_state, bins) - 1
        else:
            # Create bins
            bins = np.linspace(stim_state.min(), stim_state.max(), n_bins + 1)
            bin_indices = np.digitize(stim_state, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            bins = bins[:-1] + np.diff(bins) / 2  # Bin centers

        # Compute P(right) for each bin
        p_right = []
        n_trials_bin = []
        sem_bin = []

        for i in range(len(bins)):
            mask = bin_indices == i
            if mask.sum() > 0:
                p = choice_state[mask].mean()
                n = mask.sum()
                # Standard error
                se = np.sqrt(p * (1 - p) / n)
                p_right.append(p)
                n_trials_bin.append(n)
                sem_bin.append(se)
            else:
                p_right.append(np.nan)
                n_trials_bin.append(0)
                sem_bin.append(np.nan)

        curves[f'state_{state}'] = pd.DataFrame({
            'stimulus_bin': bins,
            'p_right': p_right,
            'n_trials': n_trials_bin,
            'sem': sem_bin
        })

    return curves


def plot_psychometric_curves(curves, model, ax=None, title="Psychometric Curves by State"):
    """
    Plot psychometric curves for all states.

    Parameters:
    -----------
    curves : dict
        Output from compute_psychometric_curves
    model : GLMHMM
        Fitted model (for state colors)
    ax : matplotlib axis, optional
        Axis to plot on
    title : str
        Plot title
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, model.n_states))

    for state in range(model.n_states):
        curve = curves[f'state_{state}']

        if len(curve) == 0:
            continue

        ax.errorbar(curve['stimulus_bin'], curve['p_right'],
                   yerr=curve['sem'], marker='o', markersize=8,
                   linewidth=2, capsize=5, alpha=0.7,
                   label=f'State {state+1}', color=colors[state])

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Stimulus Strength (negative=left, positive=right)', fontsize=12)
    ax.set_ylabel('P(choose right)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    return ax


def plot_glmhmm_summary(model, X, y, metadata, feature_names=None, figsize=(16, 12)):
    """
    Create comprehensive summary figure of GLM-HMM results.

    Includes:
    - GLM weights by state
    - State occupancy
    - Performance by state
    - State sequence over time
    - Psychometric curves
    - State transitions at task boundaries

    Parameters:
    -----------
    model : GLMHMM
        Fitted model
    X : ndarray
        Design matrix
    y : ndarray
        Choices
    metadata : dict
        Trial metadata
    feature_names : list of str, optional
        Feature names for weight plot
    figsize : tuple
        Figure size

    Returns:
    --------
    fig : Figure
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # 1. GLM weights by state
    ax1 = fig.add_subplot(gs[0, 0])
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]

    x_pos = np.arange(len(feature_names))
    width = 0.8 / model.n_states
    colors = plt.cm.Set2(np.linspace(0, 1, model.n_states))

    for state in range(model.n_states):
        weights = model.glm_weights[state]
        offset = (state - model.n_states/2) * width
        ax1.bar(x_pos + offset, weights, width, label=f'State {state+1}',
               alpha=0.7, color=colors[state])

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(feature_names, rotation=45, ha='right')
    ax1.set_ylabel('GLM Weight')
    ax1.set_title('GLM Weights by State')
    ax1.legend(fontsize=8)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Intercepts
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(model.n_states), model.glm_intercepts, color=colors, alpha=0.7)
    ax2.set_xlabel('State')
    ax2.set_ylabel('Intercept')
    ax2.set_title('Intercepts by State')
    ax2.set_xticks(range(model.n_states))
    ax2.set_xticklabels([f'State {i+1}' for i in range(model.n_states)])
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.grid(axis='y', alpha=0.3)

    # 3. State occupancy
    ax3 = fig.add_subplot(gs[0, 2])
    state_counts = np.bincount(model.most_likely_states, minlength=model.n_states)
    ax3.bar(range(model.n_states), state_counts / len(model.most_likely_states),
           color=colors, alpha=0.7)
    ax3.set_xlabel('State')
    ax3.set_ylabel('Proportion of Trials')
    ax3.set_title('State Occupancy')
    ax3.set_xticks(range(model.n_states))
    ax3.set_xticklabels([f'State {i+1}' for i in range(model.n_states)])
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3)

    # 4. Performance by state
    ax4 = fig.add_subplot(gs[1, 0])
    correct = metadata['correct']
    for state in range(model.n_states):
        mask = model.most_likely_states == state
        if mask.sum() > 0:
            acc = correct[mask].mean()
            ax4.bar(state, acc, color=colors[state], alpha=0.7,
                   label=f'n={mask.sum()}')

    ax4.set_xlabel('State')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Performance by State')
    ax4.set_xticks(range(model.n_states))
    ax4.set_xticklabels([f'State {i+1}' for i in range(model.n_states)])
    ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(0.8, color='green', linestyle='--', alpha=0.3, label='Criterion')
    ax4.set_ylim(0, 1)
    ax4.legend(fontsize=8)
    ax4.grid(axis='y', alpha=0.3)

    # 5. Latency by state (if available)
    ax5 = fig.add_subplot(gs[1, 1])
    latencies = metadata.get('latency')
    if latencies is not None:
        latency_by_state = []
        labels = []
        for state in range(model.n_states):
            mask = model.most_likely_states == state
            state_lat = latencies[mask]
            state_lat = state_lat[~np.isnan(state_lat)]
            if len(state_lat) > 0:
                latency_by_state.append(state_lat)
                labels.append(f'S{state+1}')

        if latency_by_state:
            bp = ax5.boxplot(latency_by_state, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

    ax5.set_xlabel('State')
    ax5.set_ylabel('Response Latency (s)')
    ax5.set_title('Latency by State')
    ax5.grid(axis='y', alpha=0.3)

    # 6. State sequence over trials
    ax6 = fig.add_subplot(gs[1, 2])
    task_colors = {'PI': 'gray', 'LD': 'blue', 'LD_reversal': 'red', 'PD': 'green'}
    task_types = metadata.get('task_type', None)
    if task_types is not None:
        trial_colors = [task_colors.get(t, 'black') for t in task_types]
        ax6.scatter(range(len(model.most_likely_states)), model.most_likely_states,
                   c=trial_colors, alpha=0.5, s=5)
    else:
        ax6.plot(model.most_likely_states, 'o', markersize=2, alpha=0.5)

    ax6.set_xlabel('Trial')
    ax6.set_ylabel('State')
    ax6.set_title('State Sequence (colored by task)')
    ax6.set_yticks(range(model.n_states))
    ax6.set_yticklabels([f'State {i+1}' for i in range(model.n_states)])

    # 7. Psychometric curves
    ax7 = fig.add_subplot(gs[2, :])
    curves = compute_psychometric_curves(model, X, y, metadata)
    plot_psychometric_curves(curves, model, ax=ax7,
                            title="State-Specific Psychometric Curves")

    # 8. State occupancy over sessions
    ax8 = fig.add_subplot(gs[3, 0])
    session_idx = metadata.get('session_index')
    if session_idx is not None:
        unique_sessions = np.unique(session_idx)
        session_states = np.zeros((len(unique_sessions), model.n_states))

        for i, session in enumerate(unique_sessions):
            mask = session_idx == session
            states_in_session = model.most_likely_states[mask]
            for state in range(model.n_states):
                session_states[i, state] = (states_in_session == state).mean()

        # Stacked area plot
        ax8.stackplot(unique_sessions, *session_states.T, labels=[f'State {i+1}' for i in range(model.n_states)],
                     colors=colors, alpha=0.7)
        ax8.set_xlabel('Session')
        ax8.set_ylabel('State Proportion')
        ax8.set_title('State Usage Across Sessions')
        ax8.legend(loc='upper left', fontsize=8)
        ax8.set_ylim(0, 1)

    # 9. Transition matrix
    ax9 = fig.add_subplot(gs[3, 1])
    im = ax9.imshow(model.transition_matrix, cmap='Blues', vmin=0, vmax=1)
    ax9.set_xlabel('To State')
    ax9.set_ylabel('From State')
    ax9.set_title('Transition Matrix')
    ax9.set_xticks(range(model.n_states))
    ax9.set_yticks(range(model.n_states))
    ax9.set_xticklabels([f'S{i+1}' for i in range(model.n_states)])
    ax9.set_yticklabels([f'S{i+1}' for i in range(model.n_states)])

    # Add text annotations
    for i in range(model.n_states):
        for j in range(model.n_states):
            text = ax9.text(j, i, f'{model.transition_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black" if model.transition_matrix[i, j] < 0.5 else "white",
                           fontsize=10)
    plt.colorbar(im, ax=ax9)

    # 10. Log-likelihood convergence
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.plot(model.log_likelihood_history, 'o-', markersize=4)
    ax10.set_xlabel('EM Iteration')
    ax10.set_ylabel('Log Likelihood')
    ax10.set_title('Model Convergence')
    ax10.grid(alpha=0.3)

    return fig


def cross_validate_n_states(trial_df, animal_id, n_states_list=[2, 3, 4],
                           n_folds=5, **design_matrix_kwargs):
    """
    Perform cross-validation to select optimal number of states.

    Uses session-based cross-validation (not trial-based) to avoid leakage.

    Parameters:
    -----------
    trial_df : DataFrame
        Trial data
    animal_id : str
        Animal to analyze
    n_states_list : list of int
        Number of states to test
    n_folds : int
        Number of cross-validation folds
    **design_matrix_kwargs
        Additional arguments for create_design_matrix

    Returns:
    --------
    results_df : DataFrame
        Cross-validation results with columns:
        n_states, fold, train_ll, test_ll, test_accuracy
    """
    from glmhmm_ashwood import GLMHMM

    X, y, feature_names, metadata, data = create_design_matrix(
        trial_df, animal_id=animal_id, **design_matrix_kwargs
    )

    # Create session-based folds
    unique_sessions = np.unique(metadata['session_index'])
    np.random.shuffle(unique_sessions)
    session_folds = np.array_split(unique_sessions, n_folds)

    results = []

    for n_states in n_states_list:
        print(f"\nTesting {n_states} states...")

        for fold in range(n_folds):
            # Create train/test split
            test_sessions = session_folds[fold]
            train_mask = ~np.isin(metadata['session_index'], test_sessions)
            test_mask = np.isin(metadata['session_index'], test_sessions)

            if test_mask.sum() < 10 or train_mask.sum() < 50:
                continue

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            # Fit model
            model = GLMHMM(n_states=n_states, feature_names=feature_names,
                          normalize_features=True, regularization_strength=1.0)
            model.fit(X_train, y_train, n_iter=50, verbose=False)

            # Evaluate on test set
            # Recompute posteriors on test set
            gamma_test, _, test_ll = model._e_step(X_test, y_test)

            # Predictive accuracy
            test_states = np.argmax(gamma_test, axis=1)
            test_probs = []
            for i, state in enumerate(test_states):
                prob = model.predict_proba(X_test[i:i+1], state=state)
                test_probs.append(prob[0])
            test_probs = np.array(test_probs)
            test_pred = np.argmax(test_probs, axis=1)
            test_acc = (test_pred == y_test).mean()

            results.append({
                'n_states': n_states,
                'fold': fold,
                'train_ll': model.log_likelihood_history[-1],
                'test_ll': test_ll / len(y_test),  # Per-trial log-likelihood
                'test_accuracy': test_acc
            })

            print(f"  Fold {fold+1}: Test LL={test_ll/len(y_test):.3f}, Acc={test_acc:.1%}")

    return pd.DataFrame(results)
