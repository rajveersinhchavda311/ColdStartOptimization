"""
Phase 1 Baseline Runner — Azure Dataset
==========================================

This script runs all Phase 1 baselines on the Azure dataset:
    1. Reactive (lag_1)
    2. Static P90
    3. Forecast-Only (mean of lags)
    4. TCN (causal temporal convolutional network)

For each model:
    - Fits on training data
    - Simulates provisioning on test data
    - Computes all metrics
    - Saves per-model results

Outputs:
    results/phase1/azure/{model_name}_results.csv     — per-timestep simulation results
    results/phase1/azure/comparison.csv               — consolidated metric comparison
    results/phase1/azure/comparison_summary.txt       — human-readable summary table
    results/phase1/azure/extreme_threshold.txt        — extreme threshold documentation
    results/phase1/azure/tcn_causality_verification.txt — TCN architecture verification
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.reactive import ReactiveModel
from models.static_p90 import StaticP90Model
from models.forecast_only import ForecastOnlyModel
from models.tcn import TCNModel
from evaluation.simulator import simulate
from evaluation.metrics import compute_metrics, format_metrics_table, C_COLD, C_IDLE
from evaluation.extreme import (
    compute_extreme_threshold, summarize_extreme_events, EXTREME_PERCENTILE
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "azure")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase1", "azure")


def load_data():
    """Load Azure train/val/test splits."""
    print("\n[1/5] Loading Azure data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["timestamp"])
    val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"), parse_dates=["timestamp"])
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), parse_dates=["timestamp"])

    print(f"  Train: {len(train):,} rows ({train['timestamp'].iloc[0]} -> {train['timestamp'].iloc[-1]})")
    print(f"  Val:   {len(val):,} rows ({val['timestamp'].iloc[0]} -> {val['timestamp'].iloc[-1]})")
    print(f"  Test:  {len(test):,} rows ({test['timestamp'].iloc[0]} -> {test['timestamp'].iloc[-1]})")

    # Verify chronological ordering
    assert train["timestamp"].iloc[-1] < val["timestamp"].iloc[0], "Train/val overlap!"
    assert val["timestamp"].iloc[-1] < test["timestamp"].iloc[0], "Val/test overlap!"

    return train, val, test


def compute_threshold(train):
    """Compute extreme event threshold from TRAINING data ONLY."""
    print(f"\n[2/5] Computing extreme threshold (P{EXTREME_PERCENTILE} of TRAINING data)...")
    threshold = compute_extreme_threshold(train, percentile=EXTREME_PERCENTILE)
    print(f"  Extreme threshold: {threshold:,.0f}")
    print(f"  Source: P{EXTREME_PERCENTILE} of training concurrency")
    print(f"  Training concurrency range: [{train['concurrency'].min():,} - {train['concurrency'].max():,}]")

    # Summarize extreme events in each split
    summarize_extreme_events(train["concurrency"].values, threshold, "train")

    return threshold


def run_all_models(train, val, test, threshold):
    """Fit all models and run simulation on test data."""
    print("\n[3/5] Running all baselines...")

    models = [
        ReactiveModel(),
        StaticP90Model(),
        ForecastOnlyModel(),
        TCNModel(),
    ]

    all_results = {}
    all_metrics = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"  Model: {model.name}")
        print(f"  Strategy: {model.description}")
        print(f"{'='*60}")

        # Fit
        t_start = time.time()
        if isinstance(model, TCNModel):
            # TCN needs validation data for early stopping
            model.fit(train, val_df=val)
        else:
            model.fit(train)
        fit_time = time.time() - t_start
        print(f"  Fit time: {fit_time:.2f}s")

        # Simulate on test data
        t_start = time.time()
        results_df = simulate(model, test, threshold, c_cold=C_COLD, c_idle=C_IDLE)
        sim_time = time.time() - t_start
        print(f"  Simulation time: {sim_time:.2f}s")

        # Compute metrics
        metrics = compute_metrics(results_df)
        metrics["fit_time_seconds"] = fit_time
        metrics["sim_time_seconds"] = sim_time

        # Print key metrics
        print(f"\n  Results:")
        print(f"    Total cost:      {metrics['total_cost']:>14,.0f}")
        print(f"    Cold cost:       {metrics['cold_cost']:>14,.0f}")
        print(f"    Idle cost:       {metrics['idle_cost']:>14,.0f}")
        print(f"    Cold starts:     {metrics['total_cold_starts']:>14,}")
        print(f"    Request SLA:     {metrics['request_sla']:>14.6f}")
        print(f"    Extreme SLA:     {metrics['extreme_sla']:>14.6f}")

        all_results[model.name] = results_df
        all_metrics[model.name] = metrics

        # TCN-specific: verify causality
        if model.name == "TCN":
            print(f"\n  --- TCN Causality Verification ---")
            causality = model.verify_causality()
            for check in causality["checks"]:
                status = "[PASS]" if check["pass"] else "[FAIL]"
                print(f"    {status} {check.get('type', 'unknown')}: {check.get('description', '')}")
            print(f"    Receptive field: {causality['receptive_field']} (input length: 10)")
            print(f"    Overall causal: {'YES' if causality['is_causal'] else 'NO'}")
            all_metrics[model.name]["causality_verification"] = causality

    return all_results, all_metrics


def save_results(all_results, all_metrics, threshold):
    """Save all results to disk."""
    print(f"\n[4/5] Saving results to {RESULTS_DIR}...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save per-model results
    for model_name, results_df in all_results.items():
        path = os.path.join(RESULTS_DIR, f"{model_name}_results.csv")
        results_df.to_csv(path, index=False)
        print(f"  Saved: {path}")

    # Save comparison table
    comparison_rows = []
    for model_name, m in all_metrics.items():
        row = {
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
        }
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"), index=False)
    print(f"  Saved: comparison.csv")

    # Save human-readable summary
    summary = format_metrics_table(all_metrics)
    summary_path = os.path.join(RESULTS_DIR, "comparison_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Phase 1 Baseline Comparison - Azure Dataset\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Cost model: c_cold={C_COLD}, c_idle={C_IDLE} (experimental assumption)\n")
        f.write(f"Extreme threshold: P{EXTREME_PERCENTILE}(train) = {threshold:,.0f}\n\n")
        f.write(summary)
    print(f"  Saved: {summary_path}")

    # Save extreme threshold documentation
    threshold_path = os.path.join(RESULTS_DIR, "extreme_threshold.txt")
    with open(threshold_path, "w") as f:
        f.write(f"Extreme Event Threshold Documentation\n")
        f.write(f"{'='*50}\n")
        f.write(f"Percentile: P{EXTREME_PERCENTILE}\n")
        f.write(f"Source: TRAINING data ONLY\n")
        f.write(f"Value: {threshold:,.0f}\n")
        f.write(f"NOT derived from: validation data, test data, or evaluation outputs\n")
    print(f"  Saved: {threshold_path}")

    # Save full metrics as JSON
    metrics_json = {}
    for model_name, m in all_metrics.items():
        # Filter out non-serializable items
        clean = {k: v for k, v in m.items() if k != "causality_verification"}
        metrics_json[model_name] = clean
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Saved: metrics.json")

    return comparison_df


def print_final_summary(all_metrics, threshold):
    """Print final summary table."""
    print(f"\n[5/5] Final Summary")
    print(f"{'='*80}")
    print(f"Phase 1 Baseline Comparison - Azure Dataset")
    print(f"Cost model: c_cold={C_COLD}, c_idle={C_IDLE} (experimental assumption)")
    print(f"Extreme threshold: P{EXTREME_PERCENTILE}(train) = {threshold:,.0f}")
    print()
    print(format_metrics_table(all_metrics))
    print()


def main():
    print("=" * 60)
    print("Phase 1: Baseline Runner - Azure Dataset")
    print("=" * 60)

    # Load data
    train, val, test = load_data()

    # Compute extreme threshold from TRAINING data ONLY
    threshold = compute_threshold(train)

    # Run all models
    all_results, all_metrics = run_all_models(train, val, test, threshold)

    # Save results
    save_results(all_results, all_metrics, threshold)

    # Final summary
    print_final_summary(all_metrics, threshold)

    print("[DONE] Phase 1 baselines complete")
    return all_results, all_metrics, threshold


if __name__ == "__main__":
    main()
