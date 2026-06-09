"""
Phase 1 Baseline Runner — Huawei Combined Dataset
===================================================

Runs all 6 Phase 1 baseline models on the Huawei combined dataset.
Uses the IDENTICAL model classes, simulator, and cost function as Azure.
No modifications to methodology.

The target sentence: "The methodology was developed entirely on Azure
and applied to Huawei without modification."

Models:
    1. Reactive         — lag_1
    2. Static_P90       — P90(train) constant
    3. Forecast_Only    — mean(lag_1..lag_10)
    4. Seasonal_Naive   — lag_1440
    5. Linear_Seasonal  — OLS on [lag_1, lag_1440]
    6. TCN              — causal dilated TCN with lag_1440 scalar input

Outputs:
    results/phase1/huawei/combined/metrics.json
    results/phase1/huawei/combined/comparison.csv
    results/phase1/huawei/combined/{ModelName}_diagnostics.csv  (6 files)
    results/phase1/huawei/combined/comparison_summary.txt
    results/phase1/huawei/combined/extreme_threshold.txt
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.reactive import ReactiveModel
from models.static_p90 import StaticP90Model
from models.forecast_only import ForecastOnlyModel
from models.seasonal_naive import SeasonalNaiveModel
from models.linear_seasonal import LinearSeasonalModel
from models.tcn import TCNModel
from evaluation.simulator import simulate
from evaluation.metrics import compute_metrics, format_metrics_table, C_COLD, C_IDLE
from evaluation.extreme import (
    compute_extreme_threshold, summarize_extreme_events, EXTREME_PERCENTILE
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "huawei", "combined")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase1", "huawei", "combined")


def load_data():
    print("\n[1/5] Loading Huawei combined data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["timestamp"])
    val   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"),   parse_dates=["timestamp"])
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),  parse_dates=["timestamp"])

    print(f"  Train: {len(train):,} rows ({str(train.timestamp.iloc[0])[:16]} -> "
          f"{str(train.timestamp.iloc[-1])[:16]})")
    print(f"  Val:   {len(val):,} rows ({str(val.timestamp.iloc[0])[:16]} -> "
          f"{str(val.timestamp.iloc[-1])[:16]})")
    print(f"  Test:  {len(test):,} rows ({str(test.timestamp.iloc[0])[:16]} -> "
          f"{str(test.timestamp.iloc[-1])[:16]})")
    print(f"  Columns: {list(train.columns)}")

    assert "lag_1440" in train.columns, "lag_1440 missing — run preprocess_huawei.py first"
    assert train["lag_1440"].notna().all(), "lag_1440 has NaN in train"
    assert train["timestamp"].iloc[-1] < val["timestamp"].iloc[0], "Train/val overlap"
    assert val["timestamp"].iloc[-1] < test["timestamp"].iloc[0], "Val/test overlap"

    return train, val, test


def compute_threshold(train):
    print(f"\n[2/5] Computing extreme threshold (P{EXTREME_PERCENTILE} of training)...")
    threshold = compute_extreme_threshold(train, percentile=EXTREME_PERCENTILE)
    print(f"  Extreme threshold: {threshold:,.0f}")
    print(f"  Training concurrency range: [{train.concurrency.min():,} — "
          f"{train.concurrency.max():,}]")
    summarize_extreme_events(train["concurrency"].values, threshold, "train")
    return threshold


def run_all_models(train, val, test, threshold):
    print("\n[3/5] Running all baselines...")

    models = [
        ReactiveModel(),
        StaticP90Model(),
        ForecastOnlyModel(),
        SeasonalNaiveModel(),
        LinearSeasonalModel(),
        TCNModel(),
    ]

    all_results  = {}
    all_metrics  = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"  Model: {model.name}")
        print(f"{'='*60}")

        t_start = time.time()
        if isinstance(model, TCNModel):
            model.fit(train, val_df=val)
        else:
            model.fit(train)
        fit_time = time.time() - t_start
        print(f"  Fit time: {fit_time:.2f}s")

        t_start = time.time()
        results_df = simulate(model, test, threshold, c_cold=C_COLD, c_idle=C_IDLE)
        sim_time = time.time() - t_start
        print(f"  Simulation time: {sim_time:.2f}s")

        metrics = compute_metrics(results_df)
        metrics["fit_time_seconds"] = fit_time
        metrics["sim_time_seconds"] = sim_time

        print(f"\n  Results:")
        print(f"    Total cost:      {metrics['total_cost']:>14,.0f}")
        print(f"    Cold starts:     {metrics['total_cold_starts']:>14,}")
        print(f"    Request SLA:     {metrics['request_sla']:>14.6f}")
        print(f"    Extreme SLA:     {metrics['extreme_sla']:>14.6f}")

        all_results[model.name] = results_df
        all_metrics[model.name] = metrics

    return all_results, all_metrics


def save_results(all_results, all_metrics, threshold):
    print(f"\n[4/5] Saving results to {RESULTS_DIR}...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for model_name, results_df in all_results.items():
        # Save diagnostics CSV (demand, provisioned, cold_starts, idle per timestep)
        diag_path = os.path.join(RESULTS_DIR, f"{model_name}_diagnostics.csv")
        results_df.to_csv(diag_path, index=False)
        print(f"  Saved: {model_name}_diagnostics.csv")

    # Comparison CSV
    comparison_rows = []
    for model_name, m in all_metrics.items():
        comparison_rows.append({
            "model": model_name,
            "total_cost": m["total_cost"],
            "cold_cost": m["cold_cost"],
            "idle_cost": m["idle_cost"],
            "total_cold_starts": m["total_cold_starts"],
            "total_idle_capacity": m["total_idle_capacity"],
            "total_demand": m["total_demand"],
            "request_sla": m["request_sla"],
            "extreme_sla": m["extreme_sla"],
            "cold_start_rate": m["cold_start_rate"],
            "avg_cold_per_step": m["avg_cold_per_step"],
            "n_extreme_timesteps": m["n_extreme_timesteps"],
            "fit_time_seconds": m.get("fit_time_seconds", 0),
        })
    pd.DataFrame(comparison_rows).to_csv(
        os.path.join(RESULTS_DIR, "comparison.csv"), index=False
    )
    print("  Saved: comparison.csv")

    # Summary text
    summary = format_metrics_table(all_metrics)
    with open(os.path.join(RESULTS_DIR, "comparison_summary.txt"), "w") as f:
        f.write("Phase 1 Baseline Comparison — Huawei Combined Dataset\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Cost model: c_cold={C_COLD}, c_idle={C_IDLE} (experimental assumption)\n")
        f.write(f"Extreme threshold: P{EXTREME_PERCENTILE}(train) = {threshold:,.0f}\n\n")
        f.write(summary)
    print("  Saved: comparison_summary.txt")

    with open(os.path.join(RESULTS_DIR, "extreme_threshold.txt"), "w") as f:
        f.write(f"Extreme Event Threshold\nPercentile: P{EXTREME_PERCENTILE}\n"
                f"Source: TRAINING data ONLY\nValue: {threshold:,.0f}\n")
    print("  Saved: extreme_threshold.txt")

    # metrics.json — same format as Azure
    metrics_json = {k: v for k, v in all_metrics.items()}
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2)
    print("  Saved: metrics.json")


def main():
    print("=" * 60)
    print("Phase 1: Baseline Runner — Huawei Combined Dataset")
    print("=" * 60)

    train, val, test = load_data()
    threshold = compute_threshold(train)
    all_results, all_metrics = run_all_models(train, val, test, threshold)
    save_results(all_results, all_metrics, threshold)

    print(f"\n[5/5] Final Summary")
    print("=" * 80)
    print(format_metrics_table(all_metrics))
    print("\n[DONE] Phase 1 Huawei baselines complete")
    return all_results, all_metrics, threshold


if __name__ == "__main__":
    main()
