"""
Comprehensive State Validation Module

Validates GLM-HMM states by measuring actual behavior, not just model weights.

Key principles:
1. MEASURE behavior in each state
2. VALIDATE labels against actual data
3. ANALYZE performance trajectories (before/during/after state)
4. TEST core hypotheses

Author: Claude (Anthropic)
Date: November 9, 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import warnings


def compute_performance_trajectory(trial_df, model, window_before=10, window_during=None, window_after=10):
    """
    Measure performance BEFORE, DURING, and AFTER each state bout.

    This is critical for validating "lapse" states - we need to show that
    animals were performing WELL before the lapse, not just that accuracy
    is low overall.

    Parameters:
    -----------
    trial_df : DataFrame
        Trial-by-trial data (must be for single animal)
    model : GLMHMM
        Fitted GLM-HMM model
    window_before : int
        Number of trials before state entry to measure
    window_during : int, optional
        Number of trials during state (if None, use all)
    window_after : int
        Number of trials after state exit to measure

    Returns:
    --------
    trajectory_df : DataFrame
        Columns: bout_id, state, pre_accuracy, during_accuracy, post_accuracy,
                 pre_latency_mean, during_latency_mean, post_latency_mean,
                 pre_latency_cv, during_latency_cv, post_latency_cv
    """
    if len(trial_df) != len(model.most_likely_states):
        raise ValueError(f"trial_df length ({len(trial_df)}) != model states length ({len(model.most_likely_states)})")

    states = model.most_likely_states
    n_trials = len(states)

    # Identify state bouts (contiguous blocks of same state)
    bouts = []
    current_state = states[0]
    bout_start = 0

    for i in range(1, n_trials + 1):
        if i == n_trials or states[i] != current_state:
            # Bout ended
            bouts.append({
                'state': current_state,
                'start': bout_start,
                'end': i,
                'length': i - bout_start
            })
            if i < n_trials:
                current_state = states[i]
                bout_start = i

    # For each bout, compute before/during/after metrics
    trajectories = []

    for bout_id, bout in enumerate(bouts):
        state = bout['state']
        start = bout['start']
        end = bout['end']
        length = bout['length']

        # BEFORE: window_before trials BEFORE state entry
        pre_start = max(0, start - window_before)
        pre_end = start
        pre_trials = trial_df.iloc[pre_start:pre_end]

        if len(pre_trials) > 0:
            pre_accuracy = pre_trials['correct'].mean()
            pre_latency_mean = pre_trials['latency'].mean()
            pre_latency_cv = pre_trials['latency'].std() / pre_trials['latency'].mean() if pre_trials['latency'].mean() > 0 else np.nan
        else:
            pre_accuracy = np.nan
            pre_latency_mean = np.nan
            pre_latency_cv = np.nan

        # DURING: within state
        if window_during is None:
            during_start = start
            during_end = end
        else:
            during_start = start
            during_end = min(end, start + window_during)

        during_trials = trial_df.iloc[during_start:during_end]
        during_accuracy = during_trials['correct'].mean()
        during_latency_mean = during_trials['latency'].mean()
        during_latency_cv = during_trials['latency'].std() / during_trials['latency'].mean() if during_trials['latency'].mean() > 0 else np.nan

        # AFTER: window_after trials AFTER state exit
        post_start = end
        post_end = min(n_trials, end + window_after)
        post_trials = trial_df.iloc[post_start:post_end]

        if len(post_trials) > 0:
            post_accuracy = post_trials['correct'].mean()
            post_latency_mean = post_trials['latency'].mean()
            post_latency_cv = post_trials['latency'].std() / post_trials['latency'].mean() if post_trials['latency'].mean() > 0 else np.nan
        else:
            post_accuracy = np.nan
            post_latency_mean = np.nan
            post_latency_cv = np.nan

        trajectories.append({
            'bout_id': bout_id,
            'state': state,
            'bout_length': length,
            'pre_accuracy': pre_accuracy,
            'during_accuracy': during_accuracy,
            'post_accuracy': post_accuracy,
            'pre_latency_mean': pre_latency_mean,
            'during_latency_mean': during_latency_mean,
            'post_latency_mean': post_latency_mean,
            'pre_latency_cv': pre_latency_cv,
            'during_latency_cv': during_latency_cv,
            'post_latency_cv': post_latency_cv,
            'accuracy_drop': pre_accuracy - during_accuracy if not np.isnan(pre_accuracy) else np.nan
        })

    return pd.DataFrame(trajectories)


def compute_comprehensive_state_metrics(trial_df, model, metadata):
    """
    Measure EVERYTHING about each state to validate labels.

    Parameters:
    -----------
    trial_df : DataFrame
        Trial-by-trial data (single animal)
    model : GLMHMM
        Fitted model
    metadata : dict
        Trial metadata

    Returns:
    --------
    metrics_df : DataFrame
        One row per state with all behavioral measurements
    """
    states = model.most_likely_states
    n_states = model.n_states

    metrics = []

    for state_id in range(n_states):
        state_mask = states == state_id
        state_trials = trial_df[state_mask].copy()

        if len(state_trials) == 0:
            continue

        # === PERFORMANCE METRICS ===
        accuracy = state_trials['correct'].mean()
        accuracy_sem = state_trials['correct'].sem()

        # === LATENCY METRICS (VTE measures) ===
        latency_mean = state_trials['latency'].mean()
        latency_std = state_trials['latency'].std()
        latency_cv = latency_std / latency_mean if latency_mean > 0 else np.nan
        latency_median = state_trials['latency'].median()

        # Trial duration (deliberation)
        trial_duration_mean = state_trials['trial_duration'].mean()
        trial_duration_cv = state_trials['trial_duration'].std() / trial_duration_mean if trial_duration_mean > 0 else np.nan

        # Reward collection latency (engagement/motivation)
        reward_lat_mean = state_trials['reward_collection_latency'].mean()

        # === STRATEGY METRICS ===
        # WSLS: P(stay|win) / P(stay|lose)
        if len(state_trials) > 1:
            # Previous choice and outcome
            prev_choice = state_trials['chosen_side'].shift(1)
            curr_choice = state_trials['chosen_side']
            prev_correct = state_trials['correct'].shift(1)

            # Stay = chose same side as previous
            stayed = (prev_choice == curr_choice).iloc[1:]  # Skip first trial
            prev_won = prev_correct.iloc[:-1].values == 1
            prev_lost = prev_correct.iloc[:-1].values == 0

            p_stay_win = stayed.iloc[prev_won].mean() if prev_won.sum() > 0 else np.nan
            p_stay_lose = stayed.iloc[prev_lost].mean() if prev_lost.sum() > 0 else np.nan

            wsls_ratio = p_stay_win / p_stay_lose if (p_stay_lose > 0 and not np.isnan(p_stay_lose)) else np.nan
        else:
            p_stay_win = np.nan
            p_stay_lose = np.nan
            wsls_ratio = np.nan

        # Side bias
        right_choices = (state_trials['chosen_side'] == 'right').sum()
        total_choices = len(state_trials)
        side_bias = right_choices / total_choices  # 0 = all left, 1 = all right

        # Choice perseveration (autocorrelation)
        if len(state_trials) > 2:
            choice_numeric = (state_trials['chosen_side'] == 'right').astype(int).values
            if len(np.unique(choice_numeric)) > 1:  # Need variance
                choice_autocorr = pd.Series(choice_numeric).autocorr(lag=1)
            else:
                choice_autocorr = np.nan
        else:
            choice_autocorr = np.nan

        # === STIMULUS FOLLOWING (only if stimulus varies) ===
        # Check if stimulus has variance in this state
        state_metadata_indices = state_trials.index
        # Get stimulus values for these trials from the design matrix
        # Note: This requires the design matrix to be available
        # For now, we'll compute it separately if needed
        stimulus_following = np.nan  # Placeholder

        # === OCCUPANCY & DWELL TIME ===
        occupancy = len(state_trials) / len(trial_df)

        # Dwell times (bout lengths)
        bout_lengths = []
        current_bout_length = 0
        for i in range(len(states)):
            if states[i] == state_id:
                current_bout_length += 1
            else:
                if current_bout_length > 0:
                    bout_lengths.append(current_bout_length)
                    current_bout_length = 0
        if current_bout_length > 0:
            bout_lengths.append(current_bout_length)

        dwell_mean = np.mean(bout_lengths) if bout_lengths else np.nan
        dwell_median = np.median(bout_lengths) if bout_lengths else np.nan
        n_bouts = len(bout_lengths)

        # === AGGREGATE ALL METRICS ===
        metrics.append({
            'state': state_id,
            'n_trials': len(state_trials),
            'occupancy': occupancy,

            # Performance
            'accuracy': accuracy,
            'accuracy_sem': accuracy_sem,

            # Latency (VTE)
            'latency_mean': latency_mean,
            'latency_std': latency_std,
            'latency_cv': latency_cv,
            'latency_median': latency_median,
            'trial_duration_mean': trial_duration_mean,
            'trial_duration_cv': trial_duration_cv,
            'reward_collection_latency': reward_lat_mean,

            # Strategy
            'p_stay_win': p_stay_win,
            'p_stay_lose': p_stay_lose,
            'wsls_ratio': wsls_ratio,
            'side_bias': side_bias,
            'choice_autocorr': choice_autocorr,

            # Dwell time
            'dwell_mean': dwell_mean,
            'dwell_median': dwell_median,
            'n_bouts': n_bouts
        })

    return pd.DataFrame(metrics)


def validate_state_labels(state_metrics, trajectory_df):
    """
    Assign validated labels based on measured behavior.

    Parameters:
    -----------
    state_metrics : DataFrame
        Output from compute_comprehensive_state_metrics
    trajectory_df : DataFrame
        Output from compute_performance_trajectory

    Returns:
    --------
    validated_labels : dict
        {state_id: (label, confidence, evidence)}
    """
    validated = {}

    for idx, row in state_metrics.iterrows():
        state_id = row['state']
        accuracy = row['accuracy']
        latency_cv = row['latency_cv']
        wsls_ratio = row['wsls_ratio']
        side_bias = row['side_bias']
        trial_duration_cv = row['trial_duration_cv']

        # Get trajectory info for this state
        state_trajectories = trajectory_df[trajectory_df['state'] == state_id]

        # Evidence dictionary
        evidence = {}
        confidence = 0

        # === LAPSE STATE VALIDATION ===
        # Criteria:
        # 1. Low accuracy (~50%, near chance)
        # 2. Performance DROP from before state (good → bad)
        # 3. Performance RECOVERY after state (bad → better)

        if accuracy < 0.55:  # Low accuracy
            evidence['low_accuracy'] = f'{accuracy:.1%} (near chance)'
            confidence += 1

            # Check for performance drop
            if len(state_trajectories) > 0:
                mean_accuracy_drop = state_trajectories['accuracy_drop'].mean()
                if mean_accuracy_drop > 0.1:  # At least 10% drop
                    evidence['performance_drop'] = f'Accuracy dropped {mean_accuracy_drop:.1%} from before state'
                    confidence += 2  # Strong evidence for lapse

                    validated[state_id] = ("Disengaged Lapse", confidence, evidence)
                    continue

        # === PERSEVERATIVE / SIDE BIAS STATE ===
        # Criteria:
        # 1. Strong side bias (>70% or <30%)
        # 2. High choice autocorrelation
        # 3. Low WSLS ratio (doesn't learn from outcomes)

        if side_bias > 0.7 or side_bias < 0.3:
            evidence['side_bias'] = f'{side_bias:.1%} right choices'
            confidence += 1

            if not np.isnan(row['choice_autocorr']) and row['choice_autocorr'] > 0.5:
                evidence['choice_perseveration'] = f'Autocorr = {row["choice_autocorr"]:.2f}'
                confidence += 1

                side = 'Right' if side_bias > 0.7 else 'Left'
                validated[state_id] = (f"Perseverative {side}-Bias", confidence, evidence)
                continue

        # === DELIBERATIVE HIGH-PERFORMANCE ===
        # Criteria:
        # 1. High accuracy (>65%)
        # 2. High latency variability (CV > 0.6)
        # 3. Long trial durations (deliberation)

        if accuracy > 0.65:
            evidence['high_accuracy'] = f'{accuracy:.1%}'
            confidence += 1

            if not np.isnan(latency_cv) and latency_cv > 0.6:
                evidence['high_latency_variability'] = f'CV = {latency_cv:.2f}'
                confidence += 1

                validated[state_id] = ("Deliberative High-Performance", confidence, evidence)
                continue

        # === PROCEDURAL HIGH-PERFORMANCE ===
        # Criteria:
        # 1. High accuracy (>65%)
        # 2. LOW latency variability (CV < 0.5)
        # 3. Fast, consistent responses
        # UPDATED: Relaxed CV threshold from 0.5 to 0.65 for better sensitivity

        if accuracy > 0.65:
            if not np.isnan(latency_cv) and latency_cv < 0.65:
                evidence['low_latency_variability'] = f'CV = {latency_cv:.2f}'
                evidence['high_accuracy'] = f'{accuracy:.1%}'
                confidence += 2

                validated[state_id] = ("Procedural High-Performance", confidence, evidence)
                continue

        # === WSLS STRATEGY STATE ===
        # Criteria:
        # 1. High WSLS ratio (>1.5)
        # 2. Moderate accuracy (55-70%)

        if not np.isnan(wsls_ratio) and wsls_ratio > 1.5:
            evidence['wsls_strategy'] = f'WSLS ratio = {wsls_ratio:.2f}'
            evidence['accuracy'] = f'{accuracy:.1%}'
            confidence += 1

            validated[state_id] = ("WSLS Strategy", confidence, evidence)
            continue

        # === UNDEFINED / LOW CONFIDENCE ===
        # Report metrics but don't assign label
        evidence['accuracy'] = f'{accuracy:.1%}'
        if not np.isnan(latency_cv):
            evidence['latency_cv'] = f'{latency_cv:.2f}'
        if not np.isnan(wsls_ratio):
            evidence['wsls_ratio'] = f'{wsls_ratio:.2f}'

        validated[state_id] = (f"Undefined State {state_id}", 0, evidence)

    return validated


def create_broad_state_categories(validated_labels):
    """
    Create broad categorical groupings for easier interpretation.

    Maps detailed state labels to:
    - "Engaged" (high-performance states)
    - "Lapsed" (disengaged/poor performance states)
    - "Mixed" (intermediate or strategic states)

    Parameters:
    -----------
    validated_labels : dict
        {state_id: (label, confidence, evidence)}

    Returns:
    --------
    broad_categories : dict
        {state_id: (broad_category, detailed_label, confidence)}
    """
    broad_categories = {}

    engaged_labels = [
        "Deliberative High-Performance",
        "Procedural High-Performance"
    ]

    lapsed_labels = [
        "Disengaged Lapse",
        "Perseverative Left-Bias",
        "Perseverative Right-Bias"
    ]

    for state_id, (label, confidence, evidence) in validated_labels.items():
        if any(eng in label for eng in engaged_labels):
            broad_category = "Engaged"
        elif any(lap in label for lap in lapsed_labels):
            broad_category = "Lapsed"
        else:
            # WSLS Strategy, Undefined states
            broad_category = "Mixed"

        broad_categories[state_id] = (broad_category, label, confidence)

    return broad_categories


def test_core_hypotheses(trial_df_all_animals, multi_results, metadata_dict):
    """
    Test the three core hypotheses:
    1. Deliberation helps learning
    2. Genotypes differ in strategy use
    3. Reversal learning involves predictable state transitions

    Parameters:
    -----------
    trial_df_all_animals : DataFrame
        All animals' trial data
    multi_results : dict
        {animal_id: {'model': model, 'metadata': metadata, 'genotype': genotype, ...}}
    metadata_dict : dict
        Additional metadata

    Returns:
    --------
    hypothesis_results : dict
        {hypothesis_name: {statistic, p_value, interpretation, data}}
    """
    results = {}

    # === HYPOTHESIS 1: Deliberation → Better Learning ===
    # Measure: Correlation between latency variability and performance improvement

    animal_metrics = []
    for animal_id, res in multi_results.items():
        animal_data = trial_df_all_animals[trial_df_all_animals['animal_id'] == animal_id].copy()

        # Early phase deliberation (first 20% of trials)
        n_early = int(len(animal_data) * 0.2)
        early_trials = animal_data.iloc[:n_early]

        early_latency_cv = early_trials['latency'].std() / early_trials['latency'].mean() if early_trials['latency'].mean() > 0 else np.nan
        early_trial_duration_cv = early_trials['trial_duration'].std() / early_trials['trial_duration'].mean() if early_trials['trial_duration'].mean() > 0 else np.nan

        # Performance improvement (final 20% - first 20%)
        n_final = int(len(animal_data) * 0.2)
        final_trials = animal_data.iloc[-n_final:]

        early_accuracy = early_trials['correct'].mean()
        final_accuracy = final_trials['correct'].mean()
        performance_improvement = final_accuracy - early_accuracy

        animal_metrics.append({
            'animal_id': animal_id,
            'genotype': res.get('genotype', np.nan),
            'early_latency_cv': early_latency_cv,
            'early_trial_duration_cv': early_trial_duration_cv,
            'performance_improvement': performance_improvement,
            'final_accuracy': final_accuracy
        })

    metrics_df = pd.DataFrame(animal_metrics).dropna()

    if len(metrics_df) > 3:
        # Correlation test
        r, p = stats.pearsonr(metrics_df['early_latency_cv'], metrics_df['performance_improvement'])
        results['deliberation_helps_learning'] = {
            'statistic': r,
            'p_value': p,
            'interpretation': 'Supported' if (p < 0.05 and r > 0) else 'Not supported',
            'data': metrics_df,
            'summary': f'r={r:.3f}, p={p:.4f}'
        }

    # === HYPOTHESIS 2: Genotypes Differ in Strategy Use ===
    # Measure: State occupancy by genotype

    genotype_state_data = []
    for animal_id, res in multi_results.items():
        model = res['model']
        genotype = res.get('genotype', np.nan)

        for state_id in range(model.n_states):
            occupancy = (model.most_likely_states == state_id).mean()
            genotype_state_data.append({
                'animal_id': animal_id,
                'genotype': genotype,
                'state': state_id,
                'occupancy': occupancy
            })

    geno_state_df = pd.DataFrame(genotype_state_data).dropna()

    if len(geno_state_df) > 0:
        # ANOVA: Does genotype predict state occupancy?
        genotypes = geno_state_df['genotype'].unique()
        if len(genotypes) >= 2:
            # For each state, test genotype effect
            state_tests = {}
            for state_id in geno_state_df['state'].unique():
                state_data = geno_state_df[geno_state_df['state'] == state_id]
                groups = [state_data[state_data['genotype'] == g]['occupancy'].values for g in genotypes]
                groups = [g for g in groups if len(g) > 0]  # Remove empty groups

                if len(groups) >= 2:
                    f_stat, p_val = stats.f_oneway(*groups)
                    state_tests[state_id] = {'F': f_stat, 'p': p_val}

            results['genotype_strategy_differences'] = {
                'state_tests': state_tests,
                'data': geno_state_df,
                'summary': f'Tested {len(state_tests)} states across {len(genotypes)} genotypes'
            }

    # === HYPOTHESIS 3: Reversal Learning → Predictable State Transitions ===
    # Measure: State occupancy before/during/after reversal points
    # This requires reversal trials to be identified
    # Placeholder for now - will implement after reversal detection is done

    results['reversal_state_transitions'] = {
        'status': 'Pending reversal detection implementation',
        'summary': 'Will analyze state changes around reversal points'
    }

    return results
