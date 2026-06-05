"""
Phase 2 Runner — Risk-Aware EVT+CVaR Provisioning on Azure Dataset
=====================================================================

This script wraps Phase 1 forecasting models with the RiskAwareModel
and evaluates them using the IDENTICAL Phase 1 evaluation framework
(same simulator, metrics, extreme threshold, cost model).

Wrapped models:
    1. RiskAware(Reactive)
    2. RiskAware(Forecast_Only)
    3. RiskAware(TCN)

    Static_P90 is excluded: it produces constant predictions with
    zero residual variance, making volatility-scaled EVT meaningless.

Output:
    results/phase2/azure/{model_name}_results.csv     -- per-timestep results
    results/phase2/azure/{model_name}_diagnostics.csv  -- enriched diagnostics
    results/phase2/azure/comparison.csv               -- metric comparison
    results/phase2/azure/comparison_summary.txt        -- human-readable table
    results/phase2/azure/evt_parameters.json          -- fitted EVT parameters
    results/phase2/azure/metrics.json                 -- all metrics

Phase 1 preservation:
    This script reads Phase 1 metrics for comparison but NEVER writes
    to results/phase1/. Phase 1 artifacts remain byte-identical.
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
from models.forecast_only import ForecastOnlyModel
from models.tcn import TCNModel
from models.risk_aware import RiskAwareModel
from evaluation.simulator import simulate
from evaluation.metrics import compute_metrics, format_metrics_table, C_COLD, C_IDLE
from evaluation.extreme import (
    compute_extreme_threshold, summarize_extreme_events, EXTREME_PERCENTILE
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "azure")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase2", "azure")
PHASE1_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase1", "azure")

# Risk-aware parameters
ALPHA = 0.99
VOLATILITY_WINDOW = 30
EVT_THRESHOLD_PERCENTILE = 90


def load_data():
    """Load Azure train/val/test splits (identical to Phase 1)."""
    print("\n[1/6] Loading Azure data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["timestamp"])
    val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"), parse_dates=["timestamp"])
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), parse_dates=["timestamp"])

    print(f"  Train: {len(train):,} rows "
          f"({train['timestamp'].iloc[0]} -> {train['timestamp'].iloc[-1]})")
    print(f"  Val:   {len(val):,} rows "
          f"({val['timestamp'].iloc[0]} -> {val['timestamp'].iloc[-1]})")
    print(f"  Test:  {len(test):,} rows "
          f"({test['timestamp'].iloc[0]} -> {test['timestamp'].iloc[-1]})")

    # Verify chronological ordering
    assert train["timestamp"].iloc[-1] < val["timestamp"].iloc[0], "Train/val overlap!"
    assert val["timestamp"].iloc[-1] < test["timestamp"].iloc[0], "Val/test overlap!"

    return train, val, test


def compute_threshold(train):
    """Compute extreme event threshold from TRAINING data ONLY (same as Phase 1)."""
    print(f"\n[2/6] Computing extreme threshold "
          f"(P{EXTREME_PERCENTILE} of TRAINING data)...")
    threshold = compute_extreme_threshold(train, percentile=EXTREME_PERCENTILE)
    print(f"  Extreme threshold: {threshold:,.0f}")
    print(f"  Source: P{EXTREME_PERCENTILE} of training concurrency")
    return threshold


def create_risk_aware_models():
    """Create risk-aware wrapped models."""
    base_models = [
        ReactiveModel(),
        ForecastOnlyModel(),
        TCNModel(),
    ]

    wrapped = []
    for base in base_models:
        ra = RiskAwareModel(
            base_model=base,
            alpha=ALPHA,
            volatility_window=VOLATILITY_WINDOW,
            evt_threshold_percentile=EVT_THRESHOLD_PERCENTILE,
        )
        wrapped.append(ra)

    return wrapped


def run_all_models(models, train, val, test, threshold):
    """Fit all risk-aware models and run simulation on test data."""
    print("\n[3/6] Running all risk-aware models...")

    all_results = {}
    all_metrics = {}
    all_diagnostics = {}
    all_evt_params = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"  Model: {model.name}")
        print(f"  Strategy: {model.description}")
        print(f"{'='*60}")

        # Fit
        t_start = time.time()
        if isinstance(model.base_model, TCNModel):
            model.fit(train, val_df=val)
        else:
            model.fit(train)
        fit_time = time.time() - t_start
        print(f"  Fit time: {fit_time:.2f}s")

        # Simulate on test data using EXISTING, UNMODIFIED simulator
        t_start = time.time()
        results_df = simulate(model, test, threshold, c_cold=C_COLD, c_idle=C_IDLE)
        sim_time = time.time() - t_start
        print(f"  Simulation time: {sim_time:.2f}s")

        # Compute metrics using EXISTING, UNMODIFIED metrics module
        metrics = compute_metrics(results_df)
        metrics["fit_time_seconds"] = fit_time
        metrics["sim_time_seconds"] = sim_time
        metrics["alpha"] = model.alpha
        metrics["volatility_window"] = model.volatility_window
        metrics["evt_threshold_percentile"] = model.evt_threshold_percentile

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
        all_diagnostics[model.name] = model._diagnostics
        all_evt_params[model.name] = model.evt_params

        # Print buffer statistics
        diag = model._diagnostics
        print(f"\n  Buffer statistics:")
        print(f"    sigma_train:     {model.sigma_train:>14,.2f}")
        print(f"    CVaR_z:          {model.cvar_z:>14.4f}")
        print(f"    Buffer mean:     {np.mean(diag['buffer_t']):>14,.2f}")
        print(f"    Buffer std:      {np.std(diag['buffer_t']):>14,.2f}")
        print(f"    Buffer min:      {np.min(diag['buffer_t']):>14,.2f}")
        print(f"    Buffer max:      {np.max(diag['buffer_t']):>14,.2f}")

    return all_results, all_metrics, all_diagnostics, all_evt_params


def save_results(all_results, all_metrics, all_diagnostics, all_evt_params,
                 threshold):
    """Save all Phase 2 results to disk."""
    print(f"\n[4/6] Saving results to {RESULTS_DIR}...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save per-model results (standard simulator output)
    for model_name, results_df in all_results.items():
        path = os.path.join(RESULTS_DIR, f"{model_name}_results.csv")
        results_df.to_csv(path, index=False)
        print(f"  Saved: {path}")

    # Save enriched diagnostics (with base_prediction, sigma_t, buffer_t)
    for model_name, results_df in all_results.items():
        diag = all_diagnostics[model_name]
        enriched = results_df.copy()
        enriched["base_prediction"] = diag["base_prediction"]
        enriched["sigma_t"] = diag["sigma_t"]
        enriched["buffer_t"] = diag["buffer_t"]
        path = os.path.join(RESULTS_DIR, f"{model_name}_diagnostics.csv")
        enriched.to_csv(path, index=False)
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
        f.write("Phase 2 Risk-Aware Comparison - Azure Dataset\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Cost model: c_cold={C_COLD}, c_idle={C_IDLE} "
                f"(experimental assumption)\n")
        f.write(f"Extreme threshold: P{EXTREME_PERCENTILE}(train) = "
                f"{threshold:,.0f}\n")
        f.write(f"Risk parameters: alpha={ALPHA}, W={VOLATILITY_WINDOW}, "
                f"EVT_threshold=P{EVT_THRESHOLD_PERCENTILE}\n\n")
        f.write(summary)
    print(f"  Saved: {summary_path}")

    # Save EVT parameters
    evt_path = os.path.join(RESULTS_DIR, "evt_parameters.json")
    with open(evt_path, "w") as f:
        json.dump(all_evt_params, f, indent=2)
    print(f"  Saved: {evt_path}")

    # Save full metrics as JSON
    metrics_json = {}
    for model_name, m in all_metrics.items():
        clean = {k: v for k, v in m.items()}
        metrics_json[model_name] = clean
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Saved: metrics.json")

    return comparison_df


def load_phase1_metrics():
    """Load Phase 1 metrics for comparison (read-only)."""
    phase1_path = os.path.join(PHASE1_RESULTS_DIR, "metrics.json")
    if os.path.exists(phase1_path):
        with open(phase1_path, "r") as f:
            return json.load(f)
    else:
        print("  WARNING: Phase 1 metrics not found. Skipping comparison.")
        return None


def print_combined_summary(phase2_metrics, threshold):
    """Print combined Phase 1 vs Phase 2 comparison."""
    print(f"\n[5/6] Combined Phase 1 vs Phase 2 Comparison")
    print(f"{'='*100}")

    phase1_metrics = load_phase1_metrics()
    if phase1_metrics is None:
        return

    # Merge Phase 1 and Phase 2 metrics
    combined = {}
    for name, m in phase1_metrics.items():
        combined[f"[P1] {name}"] = m
    for name, m in phase2_metrics.items():
        combined[f"[P2] {name}"] = m

    print(f"Cost model: c_cold={C_COLD}, c_idle={C_IDLE} "
          f"(experimental assumption)")
    print(f"Extreme threshold: P{EXTREME_PERCENTILE}(train) = {threshold:,.0f}")
    print(f"Risk parameters: alpha={ALPHA}, W={VOLATILITY_WINDOW}")
    print()
    print(format_metrics_table(combined))
    print()


def print_improvement_analysis(phase2_metrics):
    """Print per-model improvement analysis vs Phase 1."""
    print(f"\n[6/6] Improvement Analysis (Phase 2 vs Phase 1)")
    print(f"{'='*80}")

    phase1_metrics = load_phase1_metrics()
    if phase1_metrics is None:
        return

    # Map Phase 2 wrapper names back to Phase 1 names
    name_map = {
        "RiskAware(Reactive)": "Reactive",
        "RiskAware(Forecast_Only)": "Forecast_Only",
        "RiskAware(TCN)": "TCN",
    }

    for p2_name, p2_m in phase2_metrics.items():
        p1_name = name_map.get(p2_name)
        if p1_name and p1_name in phase1_metrics:
            p1_m = phase1_metrics[p1_name]
            print(f"\n  {p1_name} -> {p2_name}:")
            print(f"    Request SLA:  {p1_m['request_sla']:.6f} -> "
                  f"{p2_m['request_sla']:.6f}  "
                  f"({'+'if p2_m['request_sla']>=p1_m['request_sla'] else ''}"
                  f"{(p2_m['request_sla']-p1_m['request_sla'])*100:.4f}pp)")
            print(f"    Extreme SLA:  {p1_m['extreme_sla']:.6f} -> "
                  f"{p2_m['extreme_sla']:.6f}  "
                  f"({'+'if p2_m['extreme_sla']>=p1_m['extreme_sla'] else ''}"
                  f"{(p2_m['extreme_sla']-p1_m['extreme_sla'])*100:.4f}pp)")
            print(f"    Total cost:   {p1_m['total_cost']:>14,.0f} -> "
                  f"{p2_m['total_cost']:>14,.0f}  "
                  f"({(p2_m['total_cost']/p1_m['total_cost']-1)*100:+.1f}%)")
            print(f"    Cold starts:  {p1_m['total_cold_starts']:>14,} -> "
                  f"{p2_m['total_cold_starts']:>14,}  "
                  f"({(p2_m['total_cold_starts']/p1_m['total_cold_starts']-1)*100:+.1f}%)")
            print(f"    Idle cost:    {p1_m['idle_cost']:>14,.0f} -> "
                  f"{p2_m['idle_cost']:>14,.0f}  "
                  f"({(p2_m['idle_cost']/max(p1_m['idle_cost'],1)-1)*100:+.1f}%)")


def main():
    print("=" * 60)
    print("Phase 2: Risk-Aware EVT+CVaR Runner - Azure Dataset")
    print("=" * 60)
    print(f"  alpha = {ALPHA}")
    print(f"  volatility_window = {VOLATILITY_WINDOW}")
    print(f"  evt_threshold_percentile = P{EVT_THRESHOLD_PERCENTILE}")

    # Load data (identical to Phase 1)
    train, val, test = load_data()

    # Compute extreme threshold (identical to Phase 1)
    threshold = compute_threshold(train)

    # Create risk-aware models
    models = create_risk_aware_models()

    # Run all models
    all_results, all_metrics, all_diagnostics, all_evt_params = \
        run_all_models(models, train, val, test, threshold)

    # Save results
    save_results(all_results, all_metrics, all_diagnostics, all_evt_params,
                 threshold)

    # Combined comparison
    print_combined_summary(all_metrics, threshold)

    # Improvement analysis
    print_improvement_analysis(all_metrics)

    print("\n[DONE] Phase 2 risk-aware experiments complete")
    return all_results, all_metrics


if __name__ == "__main__":
    main()
