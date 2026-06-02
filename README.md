# Cold Start Optimization: Preprocessing Framework

A clean, reproducible data preprocessing pipeline for preparing serverless workload traces for analysis and modeling.

## Project Structure

```
ColdStartOptimization/
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
└── docs/                          # Documentation
    └── preprocessing/             # Preprocessing documentation
```

## Overview

This repository focuses exclusively on the data processing phase for serverless workload traces. It contains robust scripts to aggregate raw traces into platform-wide concurrency metrics and generates autoregressive lag features for time-series forecasting models.

For detailed information regarding the preprocessing steps, feature engineering, and the train/validation/test splitting methodology, refer to the documentation in `docs/preprocessing/preprocessing_guide.md`.

## Datasets

- **Azure Functions**: Azure public traces (2019), 14 days, minute granularity
- **Huawei**: Huawei public cloud traces (2025), multi-region (R1-R5) and combined

## Setup and Usage

```bash
# Process Azure dataset
python preprocessing/preprocess_azure.py

# Process Huawei dataset
python preprocessing/preprocess_huawei.py

# Generate lag features and splits
python preprocessing/feature_engineering.py
```
