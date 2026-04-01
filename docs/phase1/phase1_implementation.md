# Phase 1: Implementation & Technical Details

**Date**: April 1, 2026  
**Status**: Complete and Verified

---

## 1. File Structure

```
ColdStartOptimization2/
│
├── models/
│   └── baselines.py                # 4 baseline implementations
│
├── evaluation/
│   ├── __init__.py                 # Package exports
│   ├── config.py                   # Cost parameters
│   ├── core.py                     # Evaluation loop + metrics
│   ├── extreme.py                  # Extreme event analysis
│   ├── pipeline.py                 # Orchestration
│   └── compare.py                  # Fair comparison framework
│
├── scripts/
│   ├── run_comparison.py           # Baseline comparison demo
│   └── finalize_phase1.py          # Finalization pipeline
│
├── data/
│   └── processed/
│       └── azure/
│           ├── train.csv           # Training set (12,090 samples)
│           ├── val.csv             # Validation set (4,030 samples)
│           └── test.csv            # Test set (4,030 samples)
│
├── results/
│   └── phase1/
│       ├── tables/
│       │   └── phase1_comparison_azure.csv
│       ├── logs/
│       └── intermediate/
│           ├── detailed_results.csv
│           └── timeseries_predictions.csv
│
├── graphs/
│   └── phase1/
│       ├── comparison/
│       │   ├── cost_vs_models.png
│       │   ├── sla_vs_models.png
│       │   ├── extreme_sla.png
│       │   └── cost_vs_sla_scatter.png
│       ├── distribution/
│       │   ├── demand_distribution.png
│       │   └── extreme_events_plot.png
│       └── timeseries/
│           └── baseline_vs_actual.png
│
└── docs/
    └── phase1/
        ├── phase1_architecture.md
        ├── phase1_implementation.md
        ├── phase1_results.md
        └── phase1_summary.md
```

---

## 2. Evaluation Pipeline

### 2.1 Core Evaluation Loop (`evaluation/core.py`)

**Function**: `evaluate_model(model, test_data, cost_cold, cost_idle)`

```python
def evaluate_model(model, test_data, cost_cold=10.0, cost_idle=1.0):
    """
    Simulate autoscaling decisions for each timestep in test data.
    
    Per-timestep:
        1. Extract features (lags): x[t] = [lag_1, ..., lag_10]
        2. Predict: containers[t] = model.predict(x[t])
        3. Compute metrics:
           - cold[t] = max(0, demand[t] - containers[t])
           - idle[t] = max(0, containers[t] - demand[t])
           - cost[t] = idle[t] * C_idle + cold[t] * C_cold
        4. Store per-timestep results
    
    Returns:
        results_df: DataFrame with columns [timestamp, demand, containers, cold, idle, cost]
        summary: Dict with aggregated metrics
    """
```

**Execution Time**: ~1 second per 4,000 timesteps on modern hardware.

**Memory**: O(T) where T = number of test timesteps (minimal).

### 2.2 Metrics Aggregation (`evaluation/core.py`)

**Function**: `aggregate_results(results_df)`

```python
def aggregate_results(results_df):
    """
    Aggregate per-timestep metrics to summary statistics.
    
    Returns dict:
        total_cold: Sum of all cold starts
        total_idle: Sum of all idle time
        total_cost: Sum of all costs
        sla_compliance: Fraction of timesteps with cold[t] = 0
        percentile_distribution: P50, P95, P99 of cold starts
    """
```

### 2.3 Distribution Analysis (`evaluation/core.py`)

**Function**: `distribution_stats(results_df, percentiles=[50, 95, 99])`

Computes percentiles of per-timestep metrics for statistical validation.

---

## 3. Extreme Event Analysis (`evaluation/extreme.py`)

### 3.1 Event Identification

**Function**: `identify_extreme_events(results_df, percentile=99)`

Flags timesteps where demand exceeds the given percentile threshold:

```python
p_threshold = np.percentile(results_df['demand'], percentile)
extreme_mask = results_df['demand'] >= p_threshold
```

For Azure test set: P99 = 775,012.7 containers (41 items, 1.0% of 4,030 timesteps).

### 3.2 Extreme Metrics

**Function**: `analyze_extreme_events(results_df)`

For extreme timesteps only, computes:
- **SLA on Extreme**: Fraction of extreme timesteps with cold[t] = 0
- **Cold Mean on Extreme**: Average cold starts per extreme timestep
- **Cost on Extreme**: Average cost per extreme timestep

**Example**:
- Static (P90): 0% extreme SLA (complete failure)
- Reactive: 24.39% extreme SLA (partial success)
- MLP: 7.32% extreme SLA (moderate failure)

---

## 4. Baseline Implementations (`models/baselines.py`)

### 4.1 ReactiveScaler

```python
class ReactiveScaler:
    def predict(self, features):
        # features = [lag_1, lag_2, ..., lag_10]
        return features[0]  # lag_1
```

**Training**: None (stateless, deterministic)

**Complexity**: O(1) per prediction

### 4.2 StaticScaler

```python
class StaticScaler:
    def __init__(self, p90_capacity):
        self.p90 = float(p90_capacity)
    
    def predict(self, features):
        return self.p90  # constant
```

**Training**: Single pass on training data to compute P90:
```python
p90 = train_data['concurrency'].quantile(0.90)  # = 750,314.9 for Azure
```

**Complexity**: O(1) per prediction

### 4.3 ForecastOnlyScaler

```python
class ForecastOnlyScaler:
    def predict(self, features):
        # features = [lag_1, lag_2, ..., lag_10]
        return np.mean(features)  # average of all lags
```

**Training**: None (stateless, deterministic)

**Complexity**: O(10) = O(1) per prediction

### 4.4 MLPForecastScaler

```python
class MLPForecastScaler:
    def __init__(self, model):
        self.model = model  # sklearn MLPRegressor
    
    def predict(self, features):
        features_2d = np.array(features).reshape(1, -1)
        return self.model.predict(features_2d)[0]
```

**Training** (function `train_mlp_scaler`):

```python
def train_mlp_scaler(train_data):
    # Extract features and targets
    X_train = train_data[['lag_1', ..., 'lag_10']].values  # (12090, 10)
    y_train = train_data['concurrency'].values             # (12090,)
    
    # Train neural network
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),   # Two hidden layers
        activation='relu',              # ReLU activation
        solver='adam',                  # Stochastic gradient descent
        max_iter=200,                   # Max iterations (early stopping at ~81)
        random_state=42,                # Reproducibility
    )
    model.fit(X_train, y_train)
    
    return MLPForecastScaler(model=model)
```

**Training Time**: ~30 seconds (81 iterations until convergence)

**Training MSE**: 751,927,112 (RMSE = 27,421 containers)

**Training MAE**: 18,482 containers

**Complexity**: O(1) per prediction (forward pass)

---

## 5. Fair Comparison Framework (`evaluation/compare.py`)

### 5.1 Simultaneous Evaluation

**Function**: `compare_models(models_dict, test_data, ...)`

```python
def compare_models(models_dict, test_data, cost_cold=10.0, cost_idle=1.0):
    """
    Evaluate all models on IDENTICAL test set with IDENTICAL cost parameters.
    
    Ensures fair comparison by:
        1. Same test data (no cherry-picking)
        2. Same cost parameters (no tuning)
        3. Same evaluation loop (no algorithmic bias)
        4. Deterministic results (no randomness)
    
    Returns:
        comparison_df: DataFrame with summary metrics (one row per model)
        detailed_results: Dict with full results from each model evaluation
    """
```

### 5.2 Comparison Table Format

**Output: DataFrame with columns**:
- `Total Cold`: Total unmet requests
- `Total Idle`: Total over-provisioning
- `Total Cost`: Combined objective
- `SLA Compliance`: Overall SLA %
- `SLA on Extreme`: Extreme event SLA %
- `Cold Mean (Extreme)`: Avg cold on spikes
- `Avg Cost (Extreme)`: Avg cost on spikes

**Example**:
```
                 Total Cold  Total Idle   Total Cost  SLA Compliance  SLA on Extreme
Model                                                                            
Reactive         39974213    39942535  439684665.0        0.548883        0.243902
Static (P90)     17518652   229788318  404974838.0        0.864020        0.000000
Forecast-Only    47078449    46890868  517675358.0        0.568734        0.024390
MLP Forecast     43387885    28330910  462209760.0        0.491811        0.073171
```

---

## 6. Data Preparation & Splits

### 6.1 Input Data Structure

**Azure dataset**: 16,120 total samples (one per minute, 14 days + preprocessing)

**Columns**:
- `timestamp`: Datetime (one per minute)
- `concurrency`: Container count (int, range ~526k to ~866k)
- `lag_1`, `lag_2`, ..., `lag_10`: Historical lags (computed via feature engineering)

### 6.2 Chronological Split

**Protocol**: No shuffling, maintain temporal order

```
Training:   Samples 0-12089     (12,090 = 60% of 20,150)
Validation: Samples 12090-16119 (4,030 = 20%)
Test:       Samples 16120+      (4,030 = 20%)
```

**Why No Shuffling**: Shuffling would create artificial temporal structure and leak future information.

### 6.3 Lag Feature Computation

For each sample at timestep $t$:

```python
lag_k[t] = concurrency[t-k]  for k in [1, 2, ..., 10]
```

**Verification No Leakage**:
- All lags are at times $t-k$ where $k \geq 1$
- No lag looks ahead to $t+1, t+2, \ldots$ (future)
- Training set lags can look back to before training data (OK, external knowledge)

---

## 7. Execution & Reproducibility

### 7.1 Running Phase 1 Finalization

```bash
# Runs full comparison, saves results, generates all 7 graphs
python scripts/finalize_phase1.py
```

**Output**:
- Results CSV: `results/phase1/tables/phase1_comparison_azure.csv`
- Intermediate data: `results/phase1/intermediate/`
- 7 graphs: `graphs/phase1/`

### 7.2 Reproducibility Guarantees

✅ **Deterministic**:
- Fixed random seed (42) for MLP training
- No stochastic dropout or regularization
- Fixed hyperparameters (no tuning)

✅ **Verifiable**:
- All code is simple, direct, no hidden complexity
- Per-timestep metrics can be manually spot-checked
- Intermediate data saved for post-hoc analysis

✅ **Portable**:
- Pure Python + pandas + scikit-learn
- No GPU required
- Runs in <5 minutes on standard CPU

### 7.3 Training-Test Separation

**MLP Training**:
- Only uses training data (12,090 samples)
- Never sees validation or test data
- Hyperparameters are fixed (not tuned on any data)

**Result**:
- Zero data leakage
- Fair performance estimates
- Generalizable to new data

---

## 8. Verification Checklist

- [x] No data leakage (features historical only, test unseen during training)
- [x] Fair comparison (identical test set, cost params, evaluation loop)
- [x] Metrics computed correctly (spot-checked against manual calculation)
- [x] Extreme events identified correctly (41 p99+ timesteps)
- [x] Reproducible results (fixed random seed, deterministic pipeline)
- [x] Results match Phase 1 outputs (verified against earlier runs)
- [x] All graphs generated successfully (7 publication-ready images)
- [x] Results archived properly (CSVs, intermediate data, logs)

---

## References

**Scikit-learn Documentation**:
- MLPRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html

**Pandas Documentation**:
- Time series operations: https://pandas.pydata.org/docs/user_guide/timeseries.html

**Matplotlib Documentation**:
- Publication-ready figures: https://matplotlib.org/stable/users/explain/colors/colormaps.html
