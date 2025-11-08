"""
Advanced GLM-HMM Analysis for Dual-Process and VTE Hypotheses

This module implements sophisticated analyses beyond the standard Ashwood et al. framework
to test two novel hypotheses:

HYPOTHESIS 1: Discrete Lapse States (Following Ashwood)
- Are lapses discrete behavioral states or random noise?
- Test temporal clustering of lapse trials
- Analyze state stability and transitions

HYPOTHESIS 2: Dual-Process Decision Making (VTE Framework)
- Deliberative state (early learning): High latency variability, VTE, exploration
- Procedural state (late learning): Fast, consistent, automatic
- Test if early deliberation predicts better learning outcomes

Author: Claude (Anthropic)
Date: 2025-11-08
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings


def compute_latency_variability_metrics(trial_df, window_size=20):
    """
    Compute latency variability metrics for VTE analysis.

    Key insight: Vicarious Trial and Error (VTE) manifests as:
    - High latency variability (deliberation)
    - More blank touches (exploring alternatives)
    - Better eventual learning

    Parameters:
    -----------
    trial_df : DataFrame
        Trial-by-trial data with 'latency', 'animal_id', 'session_index'
    window_size : int
        Sliding window for computing variability

    Returns:
    --------
    metrics_df : DataFrame
        For each window: mean_latency, cv_latency, n_blank_touches, deliberation_index
    """
    metrics = []

    for animal in trial_df['animal_id'].unique():
        animal_data = trial_df[trial_df['animal_id'] == animal].copy()
        animal_data = animal_data.sort_values(['session_index', 'trial_num'])

        latencies = animal_data['latency'].values
        blank_touches = (animal_data['left_iti_touches'] +
                        animal_data['right_iti_touches']).values

        # Sliding window analysis
        for i in range(len(latencies) - window_size + 1):
            window_latencies = latencies[i:i+window_size]
            window_blanks = blank_touches[i:i+window_size]

            # Remove NaNs
            valid_latencies = window_latencies[~np.isnan(window_latencies)]

            if len(valid_latencies) < 5:
                continue

            mean_lat = np.mean(valid_latencies)
            std_lat = np.std(valid_latencies)
            cv_lat = std_lat / mean_lat if mean_lat > 0 else 0  # Coefficient of variation

            # Deliberation index: combines variability and exploration
            # High CV + High blank touches = deliberative state
            deliberation_index = cv_lat * (1 + np.mean(window_blanks) / 10.0)

            session_idx = animal_data.iloc[i]['session_index']
            trial_idx = i

            metrics.append({
                'animal_id': animal,
                'trial_index': trial_idx,
                'session_index': session_idx,
                'mean_latency': mean_lat,
                'std_latency': std_lat,
                'cv_latency': cv_lat,
                'mean_blank_touches': np.mean(window_blanks),
                'deliberation_index': deliberation_index,
                'genotype': animal_data.iloc[i]['genotype'],
                'task_type': animal_data.iloc[i]['task_type']
            })

    return pd.DataFrame(metrics)


def identify_vte_states(model, metadata, latency_metrics, cv_threshold=0.5):
    """
    Identify which GLM-HMM states correspond to deliberative vs procedural modes.

    State Classification:
    - Deliberative/VTE: High latency CV, moderate accuracy, exploratory
    - Procedural: Low latency CV, stable performance, automatic
    - Engaged: Fast AND accurate, stimulus-driven
    - Perseverative: Fast BUT poor accuracy, history-driven

    Parameters:
    -----------
    model : GLMHMM
        Fitted model
    metadata : dict
        Trial metadata with 'correct', 'latency'
    latency_metrics : DataFrame
        Output from compute_latency_variability_metrics
    cv_threshold : float
        Threshold for classifying high vs low variability

    Returns:
    --------
    state_classification : DataFrame
        Classification of each state with behavioral signature
    """
    classifications = []

    for state in range(model.n_states):
        state_mask = model.most_likely_states == state

        if state_mask.sum() < 10:
            continue

        # Get state characteristics
        accuracy = metadata['correct'][state_mask].mean()
        latencies = metadata['latency'][state_mask]
        valid_lats = latencies[~np.isnan(latencies)]

        if len(valid_lats) < 5:
            continue

        mean_lat = np.mean(valid_lats)
        cv_lat = np.std(valid_lats) / mean_lat if mean_lat > 0 else 0

        # GLM weights
        stim_weight = model.glm_weights[state, 0]  # Stimulus
        prev_choice_weight = model.glm_weights[state, 2] if model.glm_weights.shape[1] > 2 else 0
        wsls_weight = model.glm_weights[state, 3] if model.glm_weights.shape[1] > 3 else 0

        # Classify state based on multiple dimensions
        if cv_lat > cv_threshold and 0.5 < accuracy < 0.8:
            # High variability, moderate accuracy = DELIBERATIVE
            state_type = "Deliberative/VTE"
            process_type = "Dual-process: Deliberative"
        elif cv_lat < cv_threshold and accuracy > 0.75:
            # Low variability, high accuracy = PROCEDURAL (mastered)
            state_type = "Procedural/Automatic"
            process_type = "Dual-process: Procedural"
        elif accuracy > 0.75 and abs(stim_weight) > 1.0:
            # High accuracy, stimulus-driven = ENGAGED
            state_type = "Engaged"
            process_type = "Ashwood: Engaged"
        elif accuracy < 0.55 and abs(prev_choice_weight) > 1.0:
            # Poor accuracy, history-driven = PERSEVERATIVE
            state_type = "Perseverative"
            process_type = "Ashwood: Biased"
        elif 0.45 <= accuracy <= 0.55:
            # Chance performance = LAPSE
            state_type = "Lapse/Random"
            process_type = "Ashwood: Lapse"
        else:
            state_type = "Mixed"
            process_type = "Unclassified"

        classifications.append({
            'state': state,
            'state_type': state_type,
            'process_type': process_type,
            'accuracy': accuracy,
            'mean_latency': mean_lat,
            'cv_latency': cv_lat,
            'stimulus_weight': stim_weight,
            'prev_choice_weight': prev_choice_weight,
            'wsls_weight': wsls_weight,
            'occupancy': state_mask.sum() / len(state_mask),
            'n_trials': state_mask.sum()
        })

    return pd.DataFrame(classifications)


def analyze_lapse_discreteness(model, metadata, min_lapse_run=3):
    """
    Test Hypothesis 1: Are lapses discrete states or random noise?

    Following Ashwood et al., test if:
    1. Lapse trials cluster together (not randomly interspersed)
    2. Lapse state has stable GLM weights
    3. Animals transition INTO and OUT OF lapse states discretely

    Parameters:
    -----------
    model : GLMHMM
        Fitted GLM-HMM
    metadata : dict
        Trial metadata
    min_lapse_run : int
        Minimum consecutive trials to count as "lapse run"

    Returns:
    --------
    results : dict
        Statistical tests for lapse discreteness
    """
    # Identify lapse state (accuracy ~ 0.5)
    state_accs = []
    for state in range(model.n_states):
        mask = model.most_likely_states == state
        if mask.sum() > 0:
            acc = metadata['correct'][mask].mean()
            state_accs.append((state, acc))

    # Lapse state = closest to 0.5 accuracy
    lapse_state = min(state_accs, key=lambda x: abs(x[1] - 0.5))[0]
    lapse_mask = model.most_likely_states == lapse_state

    # Test 1: Temporal clustering (run length distribution)
    runs = []
    current_run = 0

    for is_lapse in lapse_mask:
        if is_lapse:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0

    if current_run > 0:
        runs.append(current_run)

    # Compare to random (Poisson distribution)
    lapse_prob = lapse_mask.mean()
    n_trials = len(lapse_mask)

    # Expected run lengths under random model
    expected_mean_run = 1 / (1 - lapse_prob) if lapse_prob < 1 else n_trials

    # Observed mean run length
    observed_mean_run = np.mean(runs) if runs else 0

    # Test 2: Proportion of long runs (evidence for discrete states)
    long_runs = np.sum(np.array(runs) >= min_lapse_run)
    total_runs = len(runs)
    proportion_long = long_runs / total_runs if total_runs > 0 else 0

    # Test 3: Entropy of state transitions (discrete = low entropy)
    # Compute transition probabilities
    transitions = []
    for i in range(len(model.most_likely_states) - 1):
        transitions.append((model.most_likely_states[i], model.most_likely_states[i+1]))

    transition_counts = {}
    for t in transitions:
        transition_counts[t] = transition_counts.get(t, 0) + 1

    # Entropy of transitions FROM lapse state
    lapse_transitions = {k: v for k, v in transition_counts.items() if k[0] == lapse_state}
    total_lapse_trans = sum(lapse_transitions.values())

    if total_lapse_trans > 0:
        probs = np.array(list(lapse_transitions.values())) / total_lapse_trans
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
    else:
        entropy = 0

    # Test 4: Autocorrelation of lapse indicator
    lapse_indicator = lapse_mask.astype(int)
    autocorr_lag1 = np.corrcoef(lapse_indicator[:-1], lapse_indicator[1:])[0, 1]

    results = {
        'lapse_state': lapse_state,
        'lapse_probability': lapse_prob,
        'lapse_accuracy': state_accs[lapse_state][1],
        'n_lapse_runs': len(runs),
        'mean_run_length_observed': observed_mean_run,
        'mean_run_length_expected_random': expected_mean_run,
        'run_length_ratio': observed_mean_run / expected_mean_run if expected_mean_run > 0 else 0,
        'proportion_long_runs': proportion_long,
        'transition_entropy': entropy,
        'autocorrelation_lag1': autocorr_lag1,
        'interpretation': None
    }

    # Interpretation
    if results['run_length_ratio'] > 1.5 and autocorr_lag1 > 0.3:
        results['interpretation'] = "✅ DISCRETE LAPSE STATE - Lapses cluster temporally"
    elif results['run_length_ratio'] < 1.2:
        results['interpretation'] = "❌ RANDOM LAPSES - No temporal clustering"
    else:
        results['interpretation'] = "⚠️  MIXED - Some clustering but not strong"

    return results


def test_deliberation_learning_hypothesis(trial_df, model, metadata, early_phase_sessions=5):
    """
    Test Hypothesis 2: Does early deliberation predict better learning?

    Core prediction: Animals with higher latency variability (VTE) in early training
    should show better eventual performance and faster reversal learning.

    Parameters:
    -----------
    trial_df : DataFrame
        Trial data
    model : GLMHMM
        Fitted model
    metadata : dict
        Trial metadata
    early_phase_sessions : int
        Number of initial sessions to define "early learning"

    Returns:
    --------
    results : DataFrame
        Per-animal: early_CV, final_performance, reversal_speed, correlation
    """
    animal_results = []

    for animal in trial_df['animal_id'].unique():
        animal_data = trial_df[trial_df['animal_id'] == animal].copy()
        animal_data = animal_data.sort_values(['session_index', 'trial_num'])

        # Early phase: First N sessions
        early_mask = animal_data['session_index'] < early_phase_sessions
        early_data = animal_data[early_mask]

        if len(early_data) < 20:
            continue

        # Compute early deliberation metrics
        early_latencies = early_data['latency'].dropna().values
        if len(early_latencies) < 10:
            continue

        early_cv = np.std(early_latencies) / np.mean(early_latencies) if np.mean(early_latencies) > 0 else 0
        early_blank_touches = (early_data['left_iti_touches'] + early_data['right_iti_touches']).mean()
        early_deliberation = early_cv * (1 + early_blank_touches / 10.0)

        # Final performance: Last 20% of trials
        n_trials = len(animal_data)
        final_mask = animal_data.index[-int(n_trials * 0.2):]
        final_accuracy = animal_data.loc[final_mask, 'correct'].mean()

        # Reversal learning (if animal has reversal data)
        reversal_data = animal_data[animal_data['task_type'] == 'LD_reversal']
        if len(reversal_data) > 0:
            # Trials to criterion after reversal (80% over 20 trials)
            reversal_accuracies = reversal_data['correct'].rolling(window=20, min_periods=10).mean()
            trials_to_criterion = np.where(reversal_accuracies >= 0.8)[0]
            reversal_speed = trials_to_criterion[0] if len(trials_to_criterion) > 0 else len(reversal_data)
        else:
            reversal_speed = np.nan

        # Learning efficiency: sessions to criterion / engaged state proportion
        animal_trials_idx = trial_df[trial_df['animal_id'] == animal].index
        animal_states = model.most_likely_states[animal_trials_idx]

        # Find "engaged" state (highest accuracy)
        state_accs = {}
        for state in range(model.n_states):
            state_mask = animal_states == state
            if state_mask.sum() > 0:
                state_accs[state] = metadata['correct'][animal_trials_idx[state_mask]].mean()

        if state_accs:
            engaged_state = max(state_accs, key=state_accs.get)
            engaged_proportion = (animal_states == engaged_state).mean()
        else:
            engaged_proportion = 0

        # Sessions to 80% criterion
        session_accs = animal_data.groupby('session_index')['correct'].mean()
        criterion_sessions = session_accs[session_accs >= 0.8]
        sessions_to_criterion = criterion_sessions.index[0] if len(criterion_sessions) > 0 else len(session_accs)

        learning_efficiency = sessions_to_criterion / (engaged_proportion + 0.1)  # Avoid division by zero

        animal_results.append({
            'animal_id': animal,
            'genotype': animal_data['genotype'].iloc[0],
            'early_cv_latency': early_cv,
            'early_blank_touches': early_blank_touches,
            'early_deliberation_index': early_deliberation,
            'final_accuracy': final_accuracy,
            'sessions_to_criterion': sessions_to_criterion,
            'reversal_speed': reversal_speed,
            'engaged_proportion': engaged_proportion,
            'learning_efficiency': learning_efficiency
        })

    results_df = pd.DataFrame(animal_results)

    # Correlations
    if len(results_df) > 3:
        # Test: early deliberation → better final performance
        corr_performance, p_performance = stats.pearsonr(
            results_df['early_deliberation_index'],
            results_df['final_accuracy']
        )

        # Test: early deliberation → faster learning
        valid_reversal = results_df['reversal_speed'].notna()
        if valid_reversal.sum() > 3:
            corr_reversal, p_reversal = stats.pearsonr(
                results_df.loc[valid_reversal, 'early_deliberation_index'],
                results_df.loc[valid_reversal, 'reversal_speed']
            )
        else:
            corr_reversal, p_reversal = 0, 1

        results_df.attrs['correlation_performance'] = corr_performance
        results_df.attrs['p_value_performance'] = p_performance
        results_df.attrs['correlation_reversal'] = corr_reversal
        results_df.attrs['p_value_reversal'] = p_reversal

    return results_df


def analyze_state_transitions_at_reversals(trial_df, model, metadata, window_before=20, window_after=20):
    """
    Analyze how states change when task contingencies reverse.

    Key questions:
    - Do animals switch to exploratory/deliberative states immediately after reversal?
    - How long does it take to return to procedural/automatic states?
    - Are there genotype differences in flexibility?

    Parameters:
    -----------
    trial_df : DataFrame
        Trial data
    model : GLMHMM
        Fitted model
    metadata : dict
        Trial metadata
    window_before : int
        Trials to include before reversal
    window_after : int
        Trials to include after reversal

    Returns:
    --------
    reversal_analysis : DataFrame
        State occupancy and accuracy around each reversal point
    """
    analyses = []

    for animal in trial_df['animal_id'].unique():
        animal_data = trial_df[trial_df['animal_id'] == animal].copy()
        animal_data = animal_data.sort_values(['session_index', 'trial_num']).reset_index(drop=True)

        # Find reversal points (transitions from LD to LD_reversal)
        task_changes = animal_data['task_type'].ne(animal_data['task_type'].shift())
        reversal_points = animal_data[task_changes & (animal_data['task_type'] == 'LD_reversal')].index.tolist()

        for rev_idx in reversal_points:
            if rev_idx < window_before or rev_idx + window_after >= len(animal_data):
                continue

            # Get trial indices
            trials_before = list(range(rev_idx - window_before, rev_idx))
            trials_after = list(range(rev_idx, rev_idx + window_after))

            # Get states
            animal_trials_idx = trial_df[trial_df['animal_id'] == animal].index
            animal_states = model.most_likely_states[animal_trials_idx]

            states_before = animal_states[trials_before]
            states_after = animal_states[trials_after]

            # State occupancy
            for state in range(model.n_states):
                prop_before = (states_before == state).mean()
                prop_after = (states_after == state).mean()
                change = prop_after - prop_before

                # Performance in this state
                acc_before = metadata['correct'][animal_trials_idx[trials_before]].mean()
                acc_after = metadata['correct'][animal_trials_idx[trials_after]].mean()

                analyses.append({
                    'animal_id': animal,
                    'genotype': animal_data.iloc[0]['genotype'],
                    'reversal_index': rev_idx,
                    'state': state,
                    'proportion_before': prop_before,
                    'proportion_after': prop_after,
                    'change_in_proportion': change,
                    'accuracy_before': acc_before,
                    'accuracy_after': acc_after,
                    'performance_drop': acc_before - acc_after
                })

    return pd.DataFrame(analyses)


def compute_state_dwell_times(model):
    """
    Compute how long animals persist in each behavioral state.

    Dwell time = consecutive trials in same state.

    Parameters:
    -----------
    model : GLMHMM
        Fitted model

    Returns:
    --------
    dwell_df : DataFrame
        Distribution of dwell times for each state
    """
    states = model.most_likely_states
    dwell_times = {state: [] for state in range(model.n_states)}

    current_state = states[0]
    current_dwell = 1

    for i in range(1, len(states)):
        if states[i] == current_state:
            current_dwell += 1
        else:
            dwell_times[current_state].append(current_dwell)
            current_state = states[i]
            current_dwell = 1

    # Add final dwell
    dwell_times[current_state].append(current_dwell)

    # Create summary
    summaries = []
    for state, dwells in dwell_times.items():
        if dwells:
            summaries.append({
                'state': state,
                'mean_dwell': np.mean(dwells),
                'median_dwell': np.median(dwells),
                'max_dwell': np.max(dwells),
                'n_bouts': len(dwells),
                'total_trials': np.sum(dwells)
            })

    return pd.DataFrame(summaries), dwell_times


def create_learning_efficiency_score(trial_df, model, metadata):
    """
    Create composite learning efficiency metric.

    Combines:
    - Sessions to criterion
    - Engaged state proportion
    - Reversal learning speed
    - Final performance

    Parameters:
    -----------
    trial_df : DataFrame
        Trial data
    model : GLMHMM
        Fitted model
    metadata : dict
        Trial metadata

    Returns:
    --------
    efficiency_df : DataFrame
        Learning efficiency scores for each animal
    """
    scores = []

    for animal in trial_df['animal_id'].unique():
        animal_data = trial_df[trial_df['animal_id'] == animal].copy()
        animal_trials_idx = trial_df[trial_df['animal_id'] == animal].index
        animal_states = model.most_likely_states[animal_trials_idx]

        # Find engaged state
        state_accs = {}
        for state in range(model.n_states):
            mask = animal_states == state
            if mask.sum() > 0:
                state_accs[state] = metadata['correct'][animal_trials_idx[mask]].mean()

        if not state_accs:
            continue

        engaged_state = max(state_accs, key=state_accs.get)
        engaged_prop = (animal_states == engaged_state).mean()

        # Sessions to criterion
        session_accs = animal_data.groupby('session_index')['correct'].mean()
        criterion_sessions = session_accs[session_accs >= 0.8]
        sessions_to_crit = criterion_sessions.index[0] if len(criterion_sessions) > 0 else len(session_accs)

        # Final performance
        final_acc = animal_data.iloc[-100:]['correct'].mean() if len(animal_data) > 100 else animal_data['correct'].mean()

        # Learning efficiency score (higher = better)
        # Normalize components
        norm_sessions = 1 / (sessions_to_crit + 1)  # Fewer sessions = higher score
        norm_engaged = engaged_prop
        norm_final = final_acc

        efficiency_score = (norm_sessions + norm_engaged + norm_final) / 3

        scores.append({
            'animal_id': animal,
            'genotype': animal_data['genotype'].iloc[0],
            'sessions_to_criterion': sessions_to_crit,
            'engaged_proportion': engaged_prop,
            'final_accuracy': final_acc,
            'learning_efficiency_score': efficiency_score
        })

    return pd.DataFrame(scores)


def create_flexibility_index(reversal_analysis_df):
    """
    Quantify behavioral flexibility during reversals.

    Flexibility = Speed of state adaptation after contingency change

    Parameters:
    -----------
    reversal_analysis_df : DataFrame
        Output from analyze_state_transitions_at_reversals

    Returns:
    --------
    flexibility_df : DataFrame
        Flexibility index for each animal
    """
    if len(reversal_analysis_df) == 0:
        return pd.DataFrame()

    flexibility_scores = []

    for animal in reversal_analysis_df['animal_id'].unique():
        animal_revs = reversal_analysis_df[reversal_analysis_df['animal_id'] == animal]

        # Find state with largest increase after reversal (adaptive state)
        state_changes = animal_revs.groupby('state')['change_in_proportion'].mean()
        most_adaptive_state = state_changes.idxmax()

        # Flexibility = magnitude of adaptive state increase
        flexibility = state_changes[most_adaptive_state]

        # Performance recovery speed
        perf_drop = animal_revs['performance_drop'].mean()

        flexibility_scores.append({
            'animal_id': animal,
            'genotype': animal_revs.iloc[0]['genotype'],
            'flexibility_index': flexibility,
            'performance_drop_reversal': perf_drop,
            'adaptive_state': most_adaptive_state
        })

    return pd.DataFrame(flexibility_scores)
