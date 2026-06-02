# Preprocessing Guide

This document outlines the preprocessing pipeline for the Serverless Workload traces (Azure Functions and Huawei).

## 1. Raw Data Processing
The pipeline processes large-scale serverless traces to extract platform-wide concurrency per minute.

- **Azure Functions Dataset (2019)**: The raw data contains invocation counts per function per minute over 14 days. `preprocess_azure.py` aggregates these across all functions to create a total platform-wide concurrency metric for each minute (1440 minutes/day). 
- **Huawei Dataset**: `preprocess_huawei.py` performs a similar aggregation tailored to Huawei's public cloud traces, supporting both combined and multi-region (R1-R5) datasets.

The output for both scripts is a single `full_series.csv` time series consisting of two columns: `timestamp` and `concurrency`.

## 2. Feature Engineering
`feature_engineering.py` enriches the raw time series with autoregressive features:
- It generates 10 historical lag variables (`lag_1` to `lag_10`) for each timestep $t$. For example, `lag_1` corresponds to `concurrency[t-1]`.
- These features map historical state to the current target demand, enabling models to forecast based on the previous 10 minutes of workload intensity.

## 3. Train/Validation/Test Splits
To prevent data leakage, chronological splitting is strictly enforced. No shuffling is performed. The data is partitioned as follows:
- **Train**: First 60% of the timeline
- **Validation**: Next 20%
- **Test**: Final 20%

All splits guarantee no overlapping timestamps and are saved as `train.csv`, `val.csv`, and `test.csv` in their respective `dataset/processed` directories.
