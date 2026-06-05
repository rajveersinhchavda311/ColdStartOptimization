# Repository Structure

This document outlines the final directory structure of the Cold Start Optimization project after the separation of Phase 1 and Preprocessing artifacts.

## Root Directories

```text
ColdStartOptimization/
├── data/                  # Frozen datasets
├── docs/                  # Documentation
├── evaluation/            # Phase 1 simulation and metrics engine
├── graphs/                # Generated visualizations
├── models/                # Phase 1 forecasting baselines
├── preprocessing/         # Scripts for raw data processing and feature engineering
├── results/               # File-based outputs (CSVs, JSONs) from experiments
└── scripts/               # Entry points for running experiments, audits, and graph generation
```

---

## Detailed Breakdown

### `data/`
Contains the raw traces and the final engineered datasets.
* **`raw/`**: The original downloaded trace files (e.g., Azure 2019, Huawei).
* **`processed/`**: The clean, feature-engineered datasets split chronologically into `train.csv`, `val.csv`, and `test.csv`. This data is frozen for Phase 1 and Phase 2.

### `docs/`
Contains all architectural and implementation documentation, separated by phase.
* **`preprocessing/`**: Documentation on how raw data was aggregated and enriched.
* **`phase1/`**: Documentation for the Baseline Ecosystem, covering architecture, implementation details, and automated verification rules.

### `graphs/`
Contains all generated visualizations.
* **`preprocessing/`**: Visualizations generated during data processing (e.g., Huawei region decomposition, time series plots, histograms).
* **`phase1/`**: Visualizations of Phase 1 baseline results (cost comparisons, SLA comparisons, demand distributions, cold start timelines).

### `results/`
Contains the quantitative outputs of all experiments.
* **`preprocessing/`**: Directory for any quantitative processing outputs beyond the core datasets.
* **`phase1/`**: Outputs of the `run_baselines.py` and `run_audit.py` scripts, including `comparison.csv`, per-model simulation traces, extreme event thresholds, and the audit results JSON.

### `models/`
Contains the Phase 1 forecasting strategies:
* `base.py`: The abstract interface all models must implement.
* `reactive.py`: Provisions based on the immediately preceding timestep (`lag_1`).
* `static_p90.py`: Provisions a constant value equal to the P90 of the training set.
* `forecast_only.py`: Provisions the moving average of the last 10 timesteps.
* `tcn.py`: A genuine causal Temporal Convolutional Network.

### `evaluation/`
The simulation engine used to evaluate the models:
* `simulator.py`: Timestep-by-timestep simulator comparing predictions to actual demand.
* `metrics.py`: Computes Cost (Cold vs Idle) and Request SLA.
* `extreme.py`: Identifies extreme demand events based on the P99 of the training set and computes Extreme SLA.

### `preprocessing/`
Scripts responsible for data wrangling:
* `preprocess_azure.py`: Aggregates Azure trace functions.
* `preprocess_huawei.py`: Aggregates Huawei traces across regions.
* `feature_engineering.py`: Applies historical lag features and train/val/test chronological splitting.

### `scripts/`
Execution entry points:
* `run_baselines.py`: Trains and evaluates all Phase 1 models.
* `run_audit.py`: Runs automated constraint checks (e.g., causality, data leakage) against Phase 1 results.
* `generate_graphs.py`: Reads the `results/phase1/` outputs and plots them to `graphs/phase1/`.
