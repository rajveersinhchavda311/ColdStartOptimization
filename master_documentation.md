# Cold Start Optimization: Master Documentation

## 1. System Goal
The core objective of this project is to provide a clean, robust data preprocessing pipeline for handling large-scale Serverless computing (function-as-a-service) workload traces. This repository prepares the raw data for downstream modeling tasks aimed at mitigating cold starts during demand spikes.

---

## 2. Dataset & Preprocessing

The preprocessing pipeline evaluates and transforms massive real-world serverless traces into time series structures suitable for machine learning algorithms.

### Supported Datasets:
- **Azure Functions Dataset**: Contains ~16,000 distinct minute-by-minute invocation traces covering variable load patterns over multiple weeks.
- **Huawei Cloud Dataset**: Multi-region traces with varying workload characteristics.

### Processing Steps:
-   **Aggregation**: Raw data is structurally cleaned, timestamped, and combined to represent the total platform-wide concurrency for each minute.
-   **Feature Engineering**: The resulting time series is extended with an autoregressive feature set. Specifically, 10 historical lag variables map $t_{-1}$ to $t_{-10}$ onto target $t$.
-   **Train/Validation/Test Split**: Enforced rigidly chronologically (60% Train / 20% Validation / 20% Test) to categorically avoid forward-looking data leakage.

For further details, consult the `docs/preprocessing` directory.
