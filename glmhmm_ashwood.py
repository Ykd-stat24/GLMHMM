"""
GLM-HMM Implementation Following Ashwood et al. (2022)

This module implements a Generalized Linear Model - Hidden Markov Model (GLM-HMM)
for analyzing behavioral choice data, following the methodology of:

Ashwood, Z. C., Roy, N. A., Stone, I. R., Urai, A. E., Churchland, A. K., Pouget, A., & Pillow, J. W. (2022).
Mice alternate between discrete strategies during perceptual decision-making.
Nature Neuroscience, 25(2), 201-212.

Author: Claude (Anthropic)
Date: 2025-11-08
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp, expit
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
from tqdm import tqdm


class GLMHMM:
    """
    Generalized Linear Model - Hidden Markov Model for binary choice data.

    Each hidden state has its own GLM that predicts choices based on task features.
    Follows the Ashwood et al. (2022) implementation.

    Parameters:
    -----------
    n_states : int
        Number of hidden states
    observation_model : str
        Type of observation model ('bernoulli' for binary choices)
    feature_names : list of str
        Names of features in design matrix
    normalize_features : bool
        Whether to z-score features (excluding bias term)
    regularization_strength : float
        L2 regularization strength (C parameter in sklearn)
    random_state : int
        Random seed for reproducibility
    """

    def __init__(self, n_states=3, observation_model='bernoulli',
                 feature_names=None, normalize_features=True,
                 regularization_strength=1.0, random_state=42):
        self.n_states = n_states
        self.observation_model = observation_model
        self.feature_names = feature_names
        self.normalize_features = normalize_features
        self.regularization_strength = regularization_strength
        self.random_state = random_state
        np.random.seed(random_state)

        # Model parameters (to be learned)
        self.transition_matrix = None  # (n_states, n_states)
        self.initial_state_probs = None  # (n_states,)
        self.glm_weights = None  # (n_states, n_features)
        self.glm_intercepts = None  # (n_states,)
        self.feature_scaler = None  # StandardScaler for features

        # Inference results
        self.state_probabilities = None  # Posterior state probabilities
        self.most_likely_states = None  # Viterbi path
        self.log_likelihood_history = []

    def _initialize_parameters(self, X, y):
        """
        Initialize model parameters using k-means clustering on features.

        This is a key difference from your old code - proper initialization
        prevents degenerate solutions.
        """
        n_trials, n_features = X.shape

        # Initialize transition matrix with sticky diagonal
        # States prefer to stay in themselves (sticky HMM)
        self.transition_matrix = np.ones((self.n_states, self.n_states)) * 0.1
        np.fill_diagonal(self.transition_matrix, 0.7)
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)

        # Uniform initial state probabilities
        self.initial_state_probs = np.ones(self.n_states) / self.n_states

        # Initialize GLM weights using k-means on features
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_states, random_state=self.random_state, n_init=20)

        # Use scaled features for clustering
        if self.normalize_features:
            X_for_clustering = X.copy()
            # Don't scale bias column (assumed to be column 0 or constant)
            non_constant_cols = np.where(np.std(X, axis=0) > 1e-6)[0]
            if len(non_constant_cols) > 0:
                self.feature_scaler = StandardScaler()
                X_for_clustering[:, non_constant_cols] = self.feature_scaler.fit_transform(X[:, non_constant_cols])
        else:
            X_for_clustering = X
            self.feature_scaler = None

        initial_states = kmeans.fit_predict(X_for_clustering)

        # Fit GLM for each initial state cluster
        self.glm_weights = np.zeros((self.n_states, n_features))
        self.glm_intercepts = np.zeros(self.n_states)

        for state in range(self.n_states):
            state_mask = initial_states == state

            if np.sum(state_mask) < 10 or len(np.unique(y[state_mask])) < 2:
                # Not enough data or only one class - use all data with weak weights
                state_mask = np.ones(len(y), dtype=bool)

            X_state = X_for_clustering[state_mask]
            y_state = y[state_mask]

            # Fit regularized logistic regression
            # KEY FIX: Use regularization to prevent dominant intercepts!
            glm = LogisticRegression(
                penalty='l2',
                C=self.regularization_strength,
                fit_intercept=True,
                max_iter=1000,
                solver='lbfgs',
                random_state=self.random_state
            )

            try:
                glm.fit(X_state, y_state)
                self.glm_weights[state] = glm.coef_[0]
                self.glm_intercepts[state] = glm.intercept_[0]
            except:
                # If fitting fails, use small random weights
                self.glm_weights[state] = np.random.randn(n_features) * 0.1
                self.glm_intercepts[state] = 0.0

        return X_for_clustering

    def _get_observation_log_likelihoods(self, X, y):
        """
        Compute log P(observation | state) for each state.

        For Bernoulli observations (binary choices):
        log P(y=1 | X, state) = log sigmoid(X @ weights + intercept)
        log P(y=0 | X, state) = log (1 - sigmoid(X @ weights + intercept))
        """
        n_trials = len(y)
        log_likes = np.zeros((n_trials, self.n_states))

        for state in range(self.n_states):
            # Compute logits
            logits = X @ self.glm_weights[state] + self.glm_intercepts[state]

            # Compute probabilities (with numerical stability)
            probs = expit(logits)  # sigmoid function
            probs = np.clip(probs, 1e-10, 1 - 1e-10)

            # Log likelihood for each trial
            log_likes[:, state] = y * np.log(probs) + (1 - y) * np.log(1 - probs)

        return log_likes

    def _forward_pass(self, log_observation_likes):
        """
        Forward algorithm to compute P(state_t | observations_1:t).

        Returns:
        --------
        log_alpha : array (n_trials, n_states)
            Log forward probabilities
        """
        n_trials, n_states = log_observation_likes.shape
        log_alpha = np.zeros((n_trials, n_states))

        # Initialize with initial state probabilities
        log_alpha[0] = np.log(self.initial_state_probs + 1e-10) + log_observation_likes[0]

        # Forward recursion
        for t in range(1, n_trials):
            for j in range(n_states):
                # Sum over previous states
                log_alpha[t, j] = logsumexp(
                    log_alpha[t-1] + np.log(self.transition_matrix[:, j] + 1e-10)
                ) + log_observation_likes[t, j]

        return log_alpha

    def _backward_pass(self, log_observation_likes):
        """
        Backward algorithm to compute P(observations_t+1:T | state_t).

        Returns:
        --------
        log_beta : array (n_trials, n_states)
            Log backward probabilities
        """
        n_trials, n_states = log_observation_likes.shape
        log_beta = np.zeros((n_trials, n_states))

        # Initialize (beta_T = 1 for all states)
        log_beta[-1] = 0.0

        # Backward recursion
        for t in range(n_trials - 2, -1, -1):
            for i in range(n_states):
                log_beta[t, i] = logsumexp(
                    np.log(self.transition_matrix[i, :] + 1e-10) +
                    log_observation_likes[t+1] +
                    log_beta[t+1]
                )

        return log_beta

    def _e_step(self, X, y):
        """
        E-step: Compute expected state occupancies using forward-backward algorithm.

        Returns:
        --------
        gamma : array (n_trials, n_states)
            Posterior state probabilities P(state_t | all observations)
        xi : array (n_trials-1, n_states, n_states)
            Posterior transition probabilities P(state_t, state_t+1 | all observations)
        log_likelihood : float
            Log likelihood of data
        """
        n_trials = len(y)

        # Get observation likelihoods
        log_obs_likes = self._get_observation_log_likelihoods(X, y)

        # Forward and backward passes
        log_alpha = self._forward_pass(log_obs_likes)
        log_beta = self._backward_pass(log_obs_likes)

        # Compute gamma: P(state_t | observations)
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Compute xi: P(state_t, state_t+1 | observations)
        xi = np.zeros((n_trials - 1, self.n_states, self.n_states))
        for t in range(n_trials - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (
                        log_alpha[t, i] +
                        np.log(self.transition_matrix[i, j] + 1e-10) +
                        log_obs_likes[t+1, j] +
                        log_beta[t+1, j]
                    )
            # Normalize
            xi[t] = np.exp(xi[t] - logsumexp(xi[t]))

        # Total log likelihood
        log_likelihood = logsumexp(log_alpha[-1])

        return gamma, xi, log_likelihood

    def _m_step(self, X, y, gamma, xi):
        """
        M-step: Update parameters to maximize expected log likelihood.

        Parameters:
        -----------
        gamma : array (n_trials, n_states)
            Posterior state probabilities
        xi : array (n_trials-1, n_states, n_states)
            Posterior transition probabilities
        """
        n_trials, n_features = X.shape

        # Update initial state probabilities
        self.initial_state_probs = gamma[0] / gamma[0].sum()

        # Update transition matrix
        for i in range(self.n_states):
            denom = np.sum(gamma[:-1, i]) + 1e-10
            for j in range(self.n_states):
                self.transition_matrix[i, j] = np.sum(xi[:, i, j]) / denom

        # Update GLM parameters for each state
        for state in range(self.n_states):
            # Weight each trial by posterior probability of being in this state
            weights = gamma[:, state]

            # Skip if no weight in this state
            if weights.sum() < 1.0:
                continue

            # Fit weighted logistic regression
            # KEY FIX: Regularization prevents intercept dominance!
            glm = LogisticRegression(
                penalty='l2',
                C=self.regularization_strength,
                fit_intercept=True,
                max_iter=1000,
                solver='lbfgs',
                random_state=self.random_state
            )

            try:
                # Use sample weights to weight trials
                glm.fit(X, y, sample_weight=weights)
                self.glm_weights[state] = glm.coef_[0]
                self.glm_intercepts[state] = glm.intercept_[0]
            except Exception as e:
                warnings.warn(f"GLM fitting failed for state {state}: {e}")
                # Keep previous weights if fitting fails
                pass

    def fit(self, X, y, n_iter=100, tolerance=1e-4, verbose=True):
        """
        Fit GLM-HMM using Expectation-Maximization (EM) algorithm.

        Parameters:
        -----------
        X : array (n_trials, n_features)
            Design matrix
        y : array (n_trials,)
            Binary choices (0 or 1)
        n_iter : int
            Maximum number of EM iterations
        tolerance : float
            Convergence tolerance (change in log likelihood)
        verbose : bool
            Whether to print progress

        Returns:
        --------
        self : GLMHMM
            Fitted model
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Initialize parameters
        X_scaled = self._initialize_parameters(X, y)

        # EM iterations
        self.log_likelihood_history = []
        iterator = tqdm(range(n_iter), desc="EM iterations") if verbose else range(n_iter)

        for iteration in iterator:
            # E-step
            gamma, xi, log_likelihood = self._e_step(X_scaled, y)
            self.log_likelihood_history.append(log_likelihood)

            # M-step
            self._m_step(X_scaled, y, gamma, xi)

            # Check convergence
            if iteration > 0:
                improvement = log_likelihood - self.log_likelihood_history[-2]
                if verbose:
                    iterator.set_postfix({'LL': f'{log_likelihood:.2f}', 'Δ': f'{improvement:.4f}'})

                if abs(improvement) < tolerance:
                    if verbose:
                        print(f"\nConverged after {iteration + 1} iterations")
                    break

        # Store final state probabilities
        self.state_probabilities = gamma
        self.most_likely_states = np.argmax(gamma, axis=1)

        return self

    def predict_proba(self, X, state=None):
        """
        Predict choice probabilities.

        Parameters:
        -----------
        X : array (n_trials, n_features)
            Design matrix
        state : int or None
            If int, predict using GLM for that specific state.
            If None, marginalize over all states using posterior probabilities.

        Returns:
        --------
        probs : array (n_trials, 2)
            Probability of choosing [0, 1] for each trial
        """
        X = np.asarray(X)

        # Scale features if needed
        if self.feature_scaler is not None:
            X_scaled = X.copy()
            non_constant_cols = np.where(np.std(X, axis=0) > 1e-6)[0]
            if len(non_constant_cols) > 0:
                X_scaled[:, non_constant_cols] = self.feature_scaler.transform(X[:, non_constant_cols])
        else:
            X_scaled = X

        if state is not None:
            # Predict using specific state
            logits = X_scaled @ self.glm_weights[state] + self.glm_intercepts[state]
            prob_1 = expit(logits)
            return np.column_stack([1 - prob_1, prob_1])
        else:
            # Marginalize over states
            if self.state_probabilities is None:
                raise ValueError("Must fit model first or specify state")

            probs = np.zeros((len(X), 2))
            for s in range(self.n_states):
                logits = X_scaled @ self.glm_weights[s] + self.glm_intercepts[s]
                prob_1 = expit(logits)
                state_probs = np.column_stack([1 - prob_1, prob_1])
                probs += state_probs * self.state_probabilities[:, s:s+1]

            return probs

    def get_state_summary(self, y=None, metadata=None):
        """
        Generate a summary dataframe of state characteristics.

        Parameters:
        -----------
        y : array, optional
            True choices to compute accuracy
        metadata : dict, optional
            Additional metadata (e.g., task type, latency)

        Returns:
        --------
        summary : DataFrame
            Summary of each state's characteristics
        """
        if self.most_likely_states is None:
            raise ValueError("Must fit model first")

        summaries = []
        for state in range(self.n_states):
            mask = self.most_likely_states == state
            n_trials = mask.sum()

            summary = {
                'state': state,
                'n_trials': n_trials,
                'proportion': n_trials / len(self.most_likely_states),
            }

            # Add GLM weights
            if self.feature_names is not None:
                for i, name in enumerate(self.feature_names):
                    summary[f'weight_{name}'] = self.glm_weights[state, i]
            else:
                for i in range(self.glm_weights.shape[1]):
                    summary[f'weight_{i}'] = self.glm_weights[state, i]

            summary['intercept'] = self.glm_intercepts[state]

            # Add accuracy if y provided
            if y is not None:
                if n_trials > 0:
                    summary['accuracy'] = y[mask].mean()
                else:
                    summary['accuracy'] = np.nan

            # Add metadata statistics if provided
            if metadata is not None:
                for key, values in metadata.items():
                    if n_trials > 0:
                        if np.issubdtype(np.array(values).dtype, np.number):
                            summary[f'mean_{key}'] = np.nanmean(values[mask])
                        else:
                            # For categorical, show most common
                            unique, counts = np.unique(values[mask], return_counts=True)
                            summary[f'mode_{key}'] = unique[counts.argmax()]
                    else:
                        summary[f'mean_{key}'] = np.nan

            summaries.append(summary)

        return pd.DataFrame(summaries)

    def viterbi(self, X, y):
        """
        Find most likely state sequence using Viterbi algorithm.

        This is more principled than just taking argmax of posterior probabilities.

        Returns:
        --------
        states : array (n_trials,)
            Most likely state sequence
        """
        n_trials = len(y)
        log_obs_likes = self._get_observation_log_likelihoods(X, y)

        # Initialize
        log_delta = np.zeros((n_trials, self.n_states))
        psi = np.zeros((n_trials, self.n_states), dtype=int)

        log_delta[0] = np.log(self.initial_state_probs + 1e-10) + log_obs_likes[0]

        # Forward pass
        for t in range(1, n_trials):
            for j in range(self.n_states):
                values = log_delta[t-1] + np.log(self.transition_matrix[:, j] + 1e-10)
                psi[t, j] = np.argmax(values)
                log_delta[t, j] = values[psi[t, j]] + log_obs_likes[t, j]

        # Backward pass
        states = np.zeros(n_trials, dtype=int)
        states[-1] = np.argmax(log_delta[-1])

        for t in range(n_trials - 2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states
