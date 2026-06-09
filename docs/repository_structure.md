# Repository Structure

Complete directory and file reference for the Cold Start Optimization project. Covers all five research phases and supporting infrastructure.

---

## Root Layout

```
ColdStartOptimization/
├── data/
├── docs/
├── evaluation/
├── graphs/
├── models/
├── preprocessing/
├── results/
└── scripts/
```

---

## `data/`

Raw traces and feature-engineered datasets. All processed splits are frozen — do not regenerate unless intentionally redoing an experiment.

```
data/
├── raw/
│   ├── azure/                         # Original Azure 2019 invocation CSVs
│   └── huawei/                        # Original Huawei region CSVs
└── processed/
    ├── azure/
    │   ├── full_series.csv            # Aggregated time series (pre-split)
    │   ├── train.csv                  # 11,232 rows (60%), Jan–Jan+8.5d
    │   ├── val.csv                    # 3,744 rows (20%)
    │   └── test.csv                   # 3,744 rows (20%)
    └── huawei/
        ├── combined/                  # R1+R2+R3+R4+R5 aggregate
        │   ├── train.csv              # 25,920 rows (60%), Jan 2–19
        │   ├── val.csv                # 8,640 rows (20%), Jan 20–25
        │   └── test.csv              # 8,640 rows (20%), Jan 26–31
        ├── R1/ R2/ R3/ R4/ R5/       # Individual region splits (same format)
```

All CSV files have columns: `timestamp`, `concurrency`, `lag_1`..`lag_10`, `lag_1440`.
`concurrency` = total invocations/minute (demand proxy).

---

## `preprocessing/`

Data wrangling scripts. Run once to produce `data/processed/`.

| File | Purpose |
|------|---------|
| `preprocess_azure.py` | Aggregate Azure raw invocation traces → per-minute `full_series.csv` |
| `preprocess_huawei.py` | Aggregate Huawei traces per region + combined; applies burn-in (1,440 rows) and 60/20/20 split |
| `feature_engineering.py` | Add lag_1..lag_10 + lag_1440 to any time series; enforces chronological split |

---

## `models/`

All forecasting models implement the `BaseModel` interface from `base.py`.

| File | Class | Prediction |
|------|-------|------------|
| `base.py` | `BaseModel` | Abstract interface: `fit(train_df)` + `predict(row)` |
| `reactive.py` | `ReactiveModel` | `lag_1` — last minute's demand |
| `static_p90.py` | `StaticP90Model` | `P90(train concurrency)` — constant |
| `forecast_only.py` | `ForecastOnlyModel` | `mean(lag_1..lag_10)` — moving average |
| `seasonal_naive.py` | `SeasonalNaiveModel` | `lag_1440` — same clock minute yesterday |
| `linear_seasonal.py` | `LinearSeasonalModel` | OLS on `[lag_1, lag_1440]` |
| `tcn.py` | `TCNModel` | Causal dilated 1D conv, depth=6, kernel=2, ~22K params. Target normalization. lag_1440 as scalar side-channel. seed=42. |
| `risk_aware.py` | `RiskAwareModel` | Wraps any BaseModel. Adds `sigma_t × CVaR_z` buffer. EVT fitting via `evaluation/evt.py`. |

`RiskAwareModel` constructor takes `(base_model, alpha, window, threshold_pct)`. Default anchor: α=0.99, W=30, threshold=P90.

---

## `evaluation/`

Simulation engine and metrics. Used by all phases.

| File | Purpose |
|------|---------|
| `simulator.py` | Timestep loop: call `model.predict(row)`, compute provisioned=ceil(pred), cold_starts, idle, cost |
| `metrics.py` | `request_sla`, `extreme_sla`, total cost, cold start count. `extreme_sla = 1 − cold_extreme / demand_extreme` |
| `extreme.py` | `compute_extreme_threshold(train, percentile=99)` → P99 threshold. `EXTREME_PERCENTILE = 99`. |
| `evt.py` | `fit_gpd_pot(residuals, threshold_pct, alpha)` → `{xi, beta, u, cvar_z}`. Standardizes residuals before POT. |

---

## `scripts/`

One script per experiment or audit. All are self-contained entry points.

### Phase 1 (Azure baselines)

| Script | What it does |
|--------|-------------|
| `regenerate_features.py` | One-time: regenerate Azure processed splits with lag_1440 (if missing) |
| `run_baselines.py` | Trains + evaluates all 6 Phase 1 models on Azure. Saves to `results/phase1/azure/` |
| `run_audit.py` | Phase 1 Azure audit: 74 checks. Saves `audit_results.json` |
| `generate_phase1_graphs.py` | 5 figures → `graphs/phase1/azure/` |

### Phase 2 (Azure risk-aware)

| Script | What it does |
|--------|-------------|
| `run_phase2.py` | Wraps all 5 eligible Phase 1 models with RiskAwareModel (α=0.99, W=30, P90). Saves to `results/phase2/azure/` |
| `run_phase2_audit.py` | Phase 2 Azure audit: 106 checks, 2 expected failures. Saves `audit_results.json` |
| `generate_phase2_graphs.py` | 5 figures → `graphs/phase2/azure/` |

### Phase 3 (Sensitivity analysis)

| Script | What it does |
|--------|-------------|
| `run_phase3.py` | 30 configs: 9 one-at-a-time + 8 factorial + anchor (deduped). Saves per-config results to `results/phase3/azure/` |
| `audit_phase3.py` | 148 checks across all configs. Saves `audit_results.json` |
| `generate_phase3_graphs.py` | 5 figures → `graphs/phase3/azure/` |

### Phase 4 (Ablation study)

| Script | What it does |
|--------|-------------|
| `run_phase4.py` | Runs C1 (Static+Gaussian), C2 (Dynamic+Gaussian), C3 (Static+EVT). Loads C0 from Phase 1, C4 from Phase 2. Saves to `results/phase4/azure/` |
| `audit_phase4.py` | 63 checks. Verifies C0/C4 identity, static σ flatness, accounting. Saves `audit_results.json` |
| `generate_phase4_graphs.py` | 5 figures → `graphs/phase4/azure/` |

### Phase 5 (Huawei generalization)

| Script | What it does |
|--------|-------------|
| `run_phase1_huawei.py` | Phase 1 baselines on Huawei combined. Saves to `results/phase1/huawei/combined/` |
| `audit_phase1_huawei.py` | Huawei Phase 1 audit. Saves `audit_results.json` with `"overall": "PASS"/"FAIL"` |
| `run_phase2_huawei.py` | Phase 2 risk-aware on Huawei combined. Parameters unchanged from Azure. |
| `audit_phase2_huawei.py` | Huawei Phase 2 audit. Saves `audit_results.json` with `"overall": "PASS"/"FAIL"` |
| `generate_phase5_graphs.py` | All Phase 5 figures → `graphs/phase5/` |

---

## `results/`

All quantitative outputs. Never edit these manually — they are regenerated by running the scripts.

```
results/
├── phase1/azure/
│   ├── {Model}_diagnostics.csv       # Per-timestep: demand, predicted, provisioned, cold_starts, idle, cost
│   ├── metrics.json                  # Summary: sla, extreme_sla, cost, cold_starts per model
│   ├── extreme_threshold.txt         # P99(train) value used for extreme event detection
│   └── audit_results.json            # 74 checks, "overall": "PASS"
│
├── phase2/azure/
│   ├── {RiskAware(Model)}_diagnostics.csv
│   ├── metrics.json
│   ├── evt_parameters.json           # {xi, beta, u, cvar_z, cvar_z_over_k_gaussian} per model
│   └── audit_results.json            # 106 checks, 2 XFAIL, "overall": "PASS"
│
├── phase3/azure/
│   ├── configs/
│   │   └── config_{id}_{model}_metrics.json
│   ├── summary.csv                   # All 30 configs: alpha, W, threshold, sla, cost
│   └── audit_results.json            # 148 checks, "overall": "PASS"
│
├── phase4/azure/
│   ├── conditions/
│   │   ├── {C1/C2/C3}_{model}_metrics.json
│   │   └── {C1/C2/C3}_{model}_diagnostics.csv
│   ├── summary.csv                   # All 10 rows (5 conditions × 2 models)
│   ├── all_metrics.json
│   └── audit_results.json            # 63 checks, "overall": "PASS"
│
└── phase5/
    ├── evt_xi_summary.csv             # Cross-dataset ξ and ratio table (Reactive + TCN, 7 datasets)
    ├── phase1/huawei/
    │   └── combined/
    │       ├── {Model}_diagnostics.csv
    │       ├── metrics.json
    │       ├── extreme_threshold.txt
    │       └── audit_results.json     # "overall": "PASS"/"FAIL"
    └── phase2/huawei/
        └── combined/
            ├── {RiskAware(Model)}_diagnostics.csv
            ├── metrics.json
            ├── evt_parameters.json
            └── audit_results.json     # "overall": "PASS"/"FAIL"
```

---

## `graphs/`

Pre-generated figures. See `docs/paper_context.md` for a complete inventory mapping each figure to its paper claim.

```
graphs/
├── preprocessing/
│   ├── azure/                         # timeseries_plot.png, histogram_plot.png
│   └── huawei/
│       ├── region_decomposition.png   # R1–R5 + combined time series
│       └── {combined,R1-R5}/          # histogram_plot.png per region
├── phase1/azure/                      # sla_comparison, cost_comparison, extreme_event_analysis,
│                                      # prediction_error_distribution, cold_start_timeline
├── phase2/azure/                      # sla_comparison, cost_comparison, dynamic_buffer,
│                                      # buffer_distribution, prediction_overlay
├── phase3/azure/                      # sensitivity_curves, sensitivity_extreme_sla,
│                                      # interaction_plots, robustness_overview, buffer_sensitivity
├── phase4/azure/                      # ablation_sla, ablation_cost, ablation_incremental,
│                                      # ablation_buffer_profiles, ablation_2x2_heatmap
└── phase5/                            # tail_heaviness_comparison, evt_multiplier_comparison,
                                       # cold_start_reduction, cross_dataset_phase2_sla,
                                       # regional_evt_heatmap, evt_xi_summary
```

---

## `docs/`

All documentation. Start with `paper_context.md` for paper writing.

| File/Directory | Content |
|----------------|---------|
| `paper_context.md` | **Master paper-writing reference.** Complete methodology, all key numbers, figure inventory, narrative arc, limitations, terminology. |
| `repository_structure.md` | This file. |
| `preprocessing/preprocessing_guide.md` | Dataset description, aggregation logic, feature engineering, burn-in explanation, split methodology |
| `phase1/architecture.md` | Phase 1 model descriptions, simulator design, evaluation framework |
| `phase1/verification.md` | Phase 1 audit checks and results (74/74) |
| `phase2/architecture.md` | RiskAwareModel: EVT-CVaR math, buffer formula, leakage-free sequential construction |
| `phase2/verification.md` | Phase 2 audit results (104/106), expected failure analysis |
| `phase3/sensitivity_analysis.md` | Sensitivity study design, all 30 configs, key findings (α primary lever, W near-inert) |
| `phase4/ablation_study.md` | Ablation 2×2 design, K_GAUSSIAN derivation, full results table, paper narrative |
| `phase5/generalization_study.md` | Huawei methodology, EVT parameters across all 7 datasets, cross-dataset ξ table, extreme SLA gap explanation |
| `pre_phase3_changes.md` | Change log for pre-Phase-3 fixes (lag_1440 boundary correction, Huawei preprocessing redo) |
