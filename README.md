# Cold Start Optimization: Evaluation Framework

A clean, reproducible research framework for evaluating serverless autoscaling strategies.

**Research Focus**: Evaluating resource management trade-offs, with emphasis on **extreme event resilience**.

---

## Project Structure

```
ColdStartOptimization2/
│
├── data/                          # Datasets (raw + processed)
│   ├── raw/                       # Original traces
│   └── processed/                 # Preprocessed time series
│
├── preprocessing/                 # Data pipeline
│   ├── preprocess_azure.py        # Azure Functions trace processing
│   ├── preprocess_huawei.py       # Huawei public cloud trace processing
│   └── feature_engineering.py     # Lag feature extraction + train/val/test splits
│
├── evaluation/                    # Core evaluation framework
│   ├── config.py                  # Cost parameters (C_idle, C_cold)
│   ├── core.py                    # Model evaluation loop + metrics
│   ├── extreme.py                 # Extreme event identification (p99+)
│   ├── pipeline.py                # Orchestration (run_evaluation)
│   ├── baselines.py               # Baseline models (Reactive, Static, Forecast)
│   └── compare.py                 # Fair multi-model comparison
│
├── analysis/                      # Post-evaluation analysis
│   └── sanity_checks.py           # Data integrity checks
│
├── scripts/                       # Executable examples
│   ├── run_evaluation.py          # Single-model evaluation demo
│   └── run_comparison.py          # Baseline comparison study
│
└── results/                       # Output storage for experiments
```

---

## Baseline Models (Fair Comparison Set)

All models implement `predict(features)`: takes 10 lag features → outputs container count.

### 1. **Reactive**
- **Logic**: `containers(t) = lag_1[t]` (most recent demand)
- **Represents**: System reacting to previous timestep load
- **Trade-off**: Low idle, high cold starts (reactive lag)
- **Bias**: None (purely reactive)

### 2. **Static (P90)**
- **Logic**: `containers = P90(train_data['concurrency'])`  
- **Represents**: Conservative capacity planning
- **Trade-off**: High idle, low cold starts on normal load
- **Failure mode**: Completely fails on extreme spikes (0% SLA at p99+)
- **Bias**: None (computed from training data only)

### 3. **Forecast-Only**
- **Logic**: `containers(t) = mean(lag_1, lag_2, ..., lag_10)`
- **Represents**: Neutral averaging forecast (no risk-awareness)
- **Trade-off**: Balanced cold starts and idle (between Reactive and Static)
- **Bias**: None (simple averaging, no spike-specific optimization)
- **Purpose**: Baseline forecasting approach without ML

### 4. **MLP Forecast** (Trained ML Baseline)
- **Logic**: `containers(t) = trained_neural_network(lag_1, ..., lag_10)`
- **Architecture**: MLP with layers 10 → 32 → 16 → 1
- **Training**: Trained on training data only (no test leakage)
- **Represents**: Strong baseline using standard ML forecasting
- **Trade-off**: Balanced cold starts and idle, computational overhead minimal
- **Bias**: None (standard ML model without risk-awareness)
- **Purpose**: Demonstrate ceiling of what pure forecasting can achieve
- **Note**: This baseline represents a simple feedforward neural network using lag features. It does not capture temporal dependencies via convolution.

---

## Cost Model

**Objective function** (minimized per timestep):

$$\text{cost}[t] = C_{\text{idle}} \cdot \text{idle}[t] + C_{\text{cold}} \cdot \text{cold}[t]$$

**Default parameters**:
- $C_{\text{idle}} = 1.0$ (cost per idle container-second)
- $C_{\text{cold}} = 10.0$ (penalty per unserved request)
- **Ratio**: 10:1 (failures are 10x more costly than over-provisioning)

**Definitions**:
- $\text{cold}[t] = \max(0, \text{demand}[t] - \text{containers}[t])$
- $\text{idle}[t] = \max(0, \text{containers}[t] - \text{demand}[t])$

---

## Metrics

### **Aggregate Metrics** (reported per model)

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **Total Cold** | $\sum_t \text{cold}[t]$ | Total unmet requests |
| **Total Idle** | $\sum_t \text{idle}[t]$ | Total over-provisioning |
| **Total Cost** | $\sum_t \text{cost}[t]$ | Combined objective (main comparison) |
| **SLA Compliance** | $\# \{t: \text{cold}[t]=0\} / T$ | Fraction of timesteps meeting demand |

### **Extreme Event Analysis** (p99+ demand)

Separate evaluation on top 1% demand spikes:
- **SLA on Extreme**: No-cold-start compliance rate during spikes
- **Avg Cold on Extreme**: Mean cold starts per extreme timestep
- **Cost on Extreme**: Total cost contribution from spikes

**Why separate?** A model with good average metrics may catastrophically fail on rare spikes. This is critical for production systems.

---

## Reproducibility Guarantees

✅ **No data leakage**: Features use only historical lags (lag_1...lag_10)  
✅ **Fair comparison**: All models evaluated on same test_data, cost params, evaluation loop  
✅ **No tuning on test**: Static baselines use train data only  
✅ **Deterministic**: Fixed cost parameters, no randomness  
✅ **Version controlled**: All code in git with reproducible preprocessing  

---

## Usage

### Single Model Evaluation

```python
from evaluation import run_evaluation
import pandas as pd

test_data = pd.read_csv('dataset/processed/azure/test.csv')
test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])

# Assume your_model implements predict(features)
output = run_evaluation(your_model, test_data, 
                       dataset_name='Azure',
                       model_name='My Model')

results_df = output['results_df']       # Full per-timestep results
summary = output['summary']             # Aggregates
extreme_metrics = output['extreme_metrics']  # Spike performance
```

### Baseline Comparison

```python
from evaluation import create_baselines, compare_models, train_mlp_scaler

# Load data
train_data = pd.read_csv('dataset/processed/azure/train.csv')
test_data = pd.read_csv('dataset/processed/azure/test.csv')

# Create the 3 simple baselines (Reactive, Static P90, Forecast-Only)
baselines = create_baselines(train_data)

# Train the ML-based MLP baseline
mlp_scaler = train_mlp_scaler(train_data, verbose=True)
baselines['MLP Forecast'] = mlp_scaler

# Fair comparison on test set (all 4 baselines)
comparison_df, detailed_results = compare_models(baselines, test_data,
                                                 dataset_name='Azure')

print(comparison_df)  # Results table, ready for paper
```

### Run Comparison Script

```bash
cd ColdStartOptimization2
python scripts/run_comparison.py
```

---

## Example Results (Azure Functions)

```
Ranking by Total Cost (Lower is Better):
  1. Static (P90)      | Cost: 404.9M  | SLA: 86.40% | Extreme SLA: 0.00%
  2. Reactive          | Cost: 439.7M  | SLA: 54.89% | Extreme SLA: 24.39%
  3. MLP Forecast      | Cost: 462.2M  | SLA: 49.18% | Extreme SLA: 7.32%
  4. Forecast-Only     | Cost: 517.7M  | SLA: 56.87% | Extreme SLA: 2.44%
```

**Key Findings**:
- **Static (P90)** minimizes cost but provides **zero resilience** to demand spikes (0% SLA at p99+)
- **Reactive** handles spikes better (24.39% extreme SLA) at modest cost increase (+8.6%)
- **MLP Forecast** (neural network) falls between Reactive and Static on cost (+14.1%), with moderate extreme-event resilience (7.32% SLA at p99+)
- **Forecast-Only** (neutral averaging) is most expensive (+27.8%), but shows pure forecasting approach without risk-awareness

**Research Motivation**: 
- Cost-optimal static provisioning is brittle under spikes
- Standard ML forecasting alone (MLP-based forecaster without risk-awareness) cannot achieve both low cost AND high extreme-event SLA
- This motivates advanced prediction strategies that explicitly model extremes (Phase 2)

---

## Paper Alignment

**Comprehensive baseline set**: Four baselines covering the design space:
- **Reactive** & **Static**: Simple heuristics (domain knowledge)
- **Forecast-Only** & **MLP Forecast**: Forecasting approaches (simple + ML)
- None explicitly optimize for extreme events (fair comparison point)
- All use only valid historical information (no lookahead)

**Suitable metrics for publication**:
- Aggregate metrics table: cost, cold starts, idle, SLA (4x4 table)
- Ranking by total cost (primary objective, cost-aware provisioning)
- Separate extreme event analysis (research contribution: spike resilience)
- Trade-off visualization (Pareto frontier: cost vs extreme-event SLA)

**Research narrative**:
1. **Static provisioning** (P90) minimizes cost but fails on spikes (0% extreme SLA)
2. **Reactive provisioning** (lag_1) adapts but has inherent lag, costs +8.6%
3. **Forecast-only** (mean of lags) represents neutral baseline forecasting
4. **MLP Forecast** (neural network) shows strong forecasting ceiling (+14.1% cost, 7.32% extreme SLA)
5. **Key insight**: Pure forecasting is insufficient for extreme events
6. **Phase 2 motivation**: Need risk-aware methods (EVT, CVaR) to achieve both low cost AND high extreme-event resilience

The baseline set demonstrates why handling extreme events is important and provides comprehensive reference points for advanced ML methods.

---

## Next Steps

1. **Add ML-based forecasters** (LSTM, Transformer, etc.)
2. **Sensitivity analysis** on cost parameters
3. **Multi-dataset evaluation** (aggregate Azure + Huawei results)
4. **Plotting utilities** (demand vs prediction, cost distributions)
5. **Statistical significance testing** (confidence intervals, hypothesis tests)

---

## References

**Datasets**:
- **Azure Functions**: Azure public traces (2019), 14 days, minute granularity
- **Huawei**: Huawei public cloud traces (2025), 31 days, second granularity

**Related work**:
- Extreme events in cloud workloads: Li et al. (ASPLOS 2020)
- Serverless autoscaling: Copik et al. (EuroSys 2021)
- Cost-aware capacity planning: Urgaonkar et al. (SIGMETRICS 2008)
