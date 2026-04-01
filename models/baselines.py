"""
Baseline models for autoscaling comparison.

Each baseline implements predict(features) interface:
    Input: features, array of shape (10,) with [lag_1, lag_2, ..., lag_10]
    Output: predicted container count (numeric)

Four baselines represent the design space:
    - ReactiveScaler: Respond to most recent demand (reactive)
    - StaticScaler: Fixed provisioning for P90 demand (static capacity planning)
    - ForecastOnlyScaler: Simple heuristic forecast from lag values (naive predictive)
    - MLPForecastScaler: Trained ML model for demand forecasting (strong baseline)
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class ReactiveScaler:
    """
    Reactive baseline: Use most recent observed demand.
    
    Logic:
        containers(t) = lag_1[t]
    
    Represents: System that reacts to previous timestep's load.
    Trade-off: Low idle cost but reactive lag causes cold starts.
    """
    
    def __init__(self):
        """Initialize reactive scaler (stateless)."""
        pass
    
    def predict(self, features):
        """
        Predict containers as most recent observation.
        
        INPUT: features, array of shape (10,) with [lag_1, lag_2, ..., lag_10]
        OUTPUT: container count = lag_1 (previous timestep demand)
        """
        return features[0]  # lag_1


class StaticScaler:
    """
    Static baseline: Fixed provisioning for high-percentile demand.
    
    Logic:
        containers(t) = P90(demand_train)
    
    Represents: Conservative capacity planning (provision for worst-case normal load).
    Trade-off: High idle cost but handles most spikes.
    
    Requires: Training data to compute P90 capacity.
    """
    
    def __init__(self, p90_capacity):
        """
        Initialize with pre-computed P90 capacity from TRAINING DATA.
        
        param p90_capacity: scalar, P90(train_data['concurrency'])
        """
        self.p90 = float(p90_capacity)
    
    def predict(self, features):
        """
        Predict container count (constant).
        
        INPUT: features (ignored)
        OUTPUT: constant capacity = P90(train)
        """
        return self.p90


class ForecastOnlyScaler:
    """
    Forecast-only baseline: Simple averaging forecast from historical lags.
    
    Logic:
        containers(t) = mean(lag_1, lag_2, ..., lag_10)
    
    Represents: Standard forecasting baseline without risk awareness.
    Neutral approach: Averages past values (no spike-specific optimization).
    Trade-off: Balances reactive and static provisioning strategies.
    
    IMPORTANT: This is intentionally simple and neutral.
    It does NOT explicitly optimize for extreme events, ensuring fair comparison.
    """
    
    def __init__(self):
        """Initialize forecast scaler (stateless, deterministic)."""
        pass
    
    def predict(self, features):
        """
        Predict containers as simple mean of lagged values.
        
        Logic: Average of the past 10 timesteps' demands.
        This is a neutral forecasting heuristic with no risk-awareness.
        
        INPUT: features, array of shape (10,) with [lag_1, lag_2, ..., lag_10]
        OUTPUT: container count = mean(lags)
        """
        if len(features) == 0:
            return 0
        
        return np.mean(features)


class MLPForecastScaler:
    """
    ML-based forecasting baseline: Trained neural network (no risk-awareness).
    
    Logic:
        containers(t) = model.predict(lag_features)
    
    Represents: Strong baseline using standard ML forecasting.
    Approach: Lightweight neural network trained on demand history.
    Trade-off: Balanced performance, computational overhead minimal.
    
    IMPORTANT: This is a standard ML baseline WITHOUT risk-awareness.
    It represents "what a good ML model can do" without domain-specific optimization.
    """
    
    def __init__(self, model=None):
        """
        Initialize with trained model.
        
        param model: Fitted ML model with predict() method.
                    Usually trained via train_mlp_scaler(train_data).
                    This model is a feedforward MLP, not a Temporal Convolutional Network.
        """
        self.model = model
    
    def predict(self, features):
        """
        Predict containers using trained ML model.
        
        INPUT: features, array of shape (10,) with [lag_1, lag_2, ..., lag_10]
        OUTPUT: predicted container count from trained model
        """
        if self.model is None:
            raise ValueError("MLPForecastScaler requires a trained model. "
                           "Use train_mlp_scaler(train_data) to create one.")
        
        # Ensure features are in correct shape for model
        features_2d = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_2d)[0]
        
        return max(0, prediction)  # Ensure non-negative


def train_mlp_scaler(train_data, random_state=42, verbose=False):
    """
    Train a neural network forecaster on training data.
    
    TRAINING PROTOCOL:
    - Uses ONLY training data (no test set leakage)
    - Predicts next-step demand from lag features
    - Simple architecture, fast training
    - No hyperparameter tuning (fixed, sensible defaults)
    
    INPUT:
    ------
    train_data : pd.DataFrame
        Training set with ['concurrency', 'lag_1', ..., 'lag_10'] columns.
        Must NOT contain test set data.
    
    random_state : int
        For reproducibility of neural network initialization.
    
    verbose : bool
        If True, print training progress.
    
    OUTPUT:
    -------
    scaler : MLPForecastScaler
        Trained model ready for evaluation (implements predict(features))
    
    Example:
        mlp_scaler = train_mlp_scaler(train_data)
        mlp_scaler.predict(features)  # Returns predicted demand
    """
    
    # Extract features and targets from training data
    feature_cols = [f'lag_{k}' for k in range(1, 11)]
    X_train = train_data[feature_cols].values  # Shape: (n_samples, 10)
    y_train = train_data['concurrency'].values  # Shape: (n_samples,)
    
    # Validate data
    assert X_train.shape[1] == 10, "Expected 10 lag features"
    assert len(X_train) == len(y_train), "Mismatch in feature/target lengths"
    assert not np.any(np.isnan(X_train)), "NaN values in features"
    assert not np.any(np.isnan(y_train)), "NaN values in targets"
    
    # Train lightweight neural network
    # Architecture: 10 -> 32 -> 16 -> 1 (simple, no risk-awareness)
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        max_iter=200,
        learning_rate_init=0.001,
        early_stopping=False,
        random_state=random_state,
        verbose=verbose,
    )
    
    # Fit model on training data only
    model.fit(X_train, y_train)
    
    if verbose:
        # Report training performance
        train_mse = np.mean((model.predict(X_train) - y_train) ** 2)
        train_mae = np.mean(np.abs(model.predict(X_train) - y_train))
        print(f"  MLP Forecast Training:")
        print(f"    MSE: {train_mse:,.0f}")
        print(f"    MAE: {train_mae:,.0f}")
    
    return MLPForecastScaler(model=model)


def create_baselines(train_data):
    """
    Create three unbiased baseline models for fair comparison.
    
    DESIGN PHILOSOPHY:
    All baselines are intentionally neutral and do NOT explicitly optimize for
    extreme events. This ensures fair comparison with forecasting models.
    
    INPUT:
    ------
    train_data : pd.DataFrame
        Training set with 'concurrency' column.
        Used ONLY for StaticScaler initialization.
    
    OUTPUT:
    -------
    baselines : dict
        Keys: baseline names
        Values: initialized model objects with predict(features) method
    
    BASELINES:
    - Reactive: Uses lag_1 only (reactive, no forecasting)
    - Static (P90): Fixed capacity (P90 of training, no adaptation)
    - Forecast-Only: Mean of lags (neutral averaging, no risk-awareness)
    
    Example:
        baselines = create_baselines(train_data)
        baselines['Reactive'].predict(features)      # lag_1
        baselines['Static (P90)'].predict(features)  # constant P90(train)
        baselines['Forecast-Only'].predict(features) # mean(lags)
    """
    
    demand = train_data['concurrency']
    p90_capacity = demand.quantile(0.90)
    
    baselines = {
        'Reactive': ReactiveScaler(),
        'Static (P90)': StaticScaler(p90_capacity),
        'Forecast-Only': ForecastOnlyScaler(),
    }
    
    return baselines
