"""
Phase 2 Runner — Risk-Aware EVT+CVaR Provisioning on Azure Dataset
=====================================================================

Wraps Phase 1 forecasting models with RiskAwareModel and evaluates
using the IDENTICAL Phase 1 framework (same simulator, metrics, threshold).

Wrapped models (all non-constant Phase 1 models):
    1. RiskAware(Reactive)
    2. RiskAware(Forecast_Only)
    3. RiskAware(Seasonal_Naive)   [NEW — added Pre-Phase-3]
    4. RiskAware(Linear_Seasonal)  [NEW — added Pre-Phase-3]
    5. RiskAware(TCN)

Static_P90 is excluded: constant predictions produce residuals that equal
(demand - constant), which reflect demand variance rather than forecast error.
Applying a volatility-scaled EVT buffer on top of an already-conservative
static provisioner is semantically incoherent — the buffer would overload
an already-safe strategy without a principled motivation.

Outputs:
    results/phase2/azure/{model_name}_results.csv
    results/phase2/azure/{model_name}_diagnostics.csv
    results/phase2/azure/comparison.csv
    results/phase2/azure/comparison_summary.txt
    results/phase2/azure/evt_parameters.json
    results/phase2/azure/metrics.json

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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.reactive import ReactiveModel
from models.forecast_only import ForecastOnlyModel
from models.seasonal_naive import SeasonalNaiveModel
from models.linear_seasonal import LinearSeasonalModel
from models.tcn import TCNModel
from models.risk_aware import RiskAwareModel
from evaluation.simulator import simulate
from evaluation.metrics import compute_metrics, format_metrics_table, C_COLD, C_IDLE
from evaluation.extreme import (
    compute_extreme_threshold, summarize_extreme_events, EXTREME_PERCENTILE
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "azure")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase2", "azure")
PHASE1_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase1", "azure")

ALPHA = 0.99
VOLATILITY_WINDOW = 30
EVT_THRESHOLD_PERCENTILE = 90


def load_data():
    print("\n[1/6] Loading Azure data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["timestamp"])
    val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"), parse_dates=["timestamp"])
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), parse_dates=["timestamp"])

    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    assert train["timestamp"].iloc[-1] < val["timestamp"].iloc[0], "Train/val overlap!"
    assert val["timestamp"].iloc[-1] < test["timestamp"].iloc[0], "Val/test overlap!"
    return train, val, test


def compute_threshold(train):
    print(f"\n[2/6] Computing extreme threshold (P{EXTREME_PERCENTILE} of training)...")
    threshold = compute_extreme_threshold(train, percentile=EXTREME_PERCENTILE)
    print(f"  Extreme threshold: {threshold:,.0f}")
    return threshold


def create_risk_aware_models():
    base_models = [
        ReactiveModel(),
        ForecastOnlyModel(),
        SeasonalNaiveModel(),
        LinearSeasonalModel(),
        TCNModel(),
    ]
    return [
        RiskAwareModel(
            base_model=b,
            alpha=ALPHA,
            volatility_window=VOLATILITY_WINDOW,
            evt_threshold_percentile=EVT_THRESHOLD_PERCENTILE,
        )
        for b in base_models
    ]


def run_all_models(models, train, val, test, threshold):
    print("\n[3/6] Running all risk-aware models...")

    all_results, all_metrics, all_diagnostics, all_evt_params = {}, {}, {}, {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"  Model: {model.name}")
        print(f"{'='*60}")

        t_start = time.time()
        if isinstance(model.base_model, TCNModel):
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
        metrics["alpha"] = model.alpha
        metrics["volatility_window"] = model.volatility_window
        metrics["evt_threshold_percentile"] = model.evt_threshold_percentile

        print(f"\n  Results:")
        print(f"    Total cost:   {metrics['total_cost']:>14,.0f}")
        print(f"    Cold starts:  {metrics['total_cold_starts']:>14,}")
        print(f"    Request SLA:  {metrics['request_sla']:>14.6f}")
        print(f"    Extreme SLA:  {metrics['extreme_sla']:>14.6f}")

        diag = model._diagnostics
        print(f"\n  Buffer stats: "
              f"sigma_train={model.sigma_train:,.0f}  "
              f"CVaR_z={model.cvar_z:.4f}  "
              f"buffer_mean={np.mean(diag['buffer_t']):,.0f}")

        all_results[model.name] = results_df
        all_metrics[model.name] = metrics
        all_diagnostics[model.name] = diag
        all_evt_params[model.name] = model.evt_params

    return all_results, all_metrics, all_diagnostics, all_evt_params


def save_results(all_results, all_metrics, all_diagnostics, all_evt_params, threshold):
    print(f"\n[4/6] Saving results to {RESULTS_DIR}...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for model_name, results_df in all_results.items():
        results_df.to_csv(
            os.path.join(RESULTS_DIR, f"{model_name}_results.csv"), index=False
        )
        enriched = results_df.copy()
        diag = all_diagnostics[model_name]
        enriched["base_prediction"] = diag["base_prediction"]
        enriched["sigma_t"] = diag["sigma_t"]
        enriched["buffer_t"] = diag["buffer_t"]
        enriched.to_csv(
            os.path.join(RESULTS_DIR, f"{model_name}_diagnostics.csv"), index=False
        )
        print(f"  Saved: {model_name}_results.csv + _diagnostics.csv")

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

    summary = format_metrics_table(all_metrics)
    with open(os.path.join(RESULTS_DIR, "comparison_summary.txt"), "w") as f:
        f.write("Phase 2 Risk-Aware Comparison - Azure Dataset\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Cost model: c_cold={C_COLD}, c_idle={C_IDLE} (experimental assumption)\n")
        f.write(f"Extreme threshold: P{EXTREME_PERCENTILE}(train) = {threshold:,.0f}\n")
        f.write(f"Risk params: alpha={ALPHA}, W={VOLATILITY_WINDOW}, "
                f"EVT_threshold=P{EVT_THRESHOLD_PERCENTILE}\n\n")
        f.write(summary)
    print("  Saved: comparison_summary.txt")

    with open(os.path.join(RESULTS_DIR, "evt_parameters.json"), "w") as f:
        json.dump(all_evt_params, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("  Saved: evt_parameters.json, metrics.json")


def load_phase1_metrics():
    path = os.path.join(PHASE1_RESULTS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print("  WARNING: Phase 1 metrics not found.")
    return None


def print_combined_summary(phase2_metrics, threshold):
    print(f"\n[5/6] Phase 1 vs Phase 2 Comparison")
    print("=" * 100)
    phase1 = load_phase1_metrics()
    if phase1 is None:
        return
    combined = {}
    for n, m in phase1.items():
        combined[f"[P1] {n}"] = m
    for n, m in phase2_metrics.items():
        combined[f"[P2] {n}"] = m
    print(format_metrics_table(combined))


def print_improvement_analysis(phase2_metrics):
    print(f"\n[6/6] Improvement Analysis (Phase 2 vs Phase 1 counterpart)")
    print("=" * 80)
    phase1 = load_phase1_metrics()
    if phase1 is None:
        return
    name_map = {
        "RiskAware(Reactive)": "Reactive",
        "RiskAware(Forecast_Only)": "Forecast_Only",
        "RiskAware(Seasonal_Naive)": "Seasonal_Naive",
        "RiskAware(Linear_Seasonal)": "Linear_Seasonal",
        "RiskAware(TCN)": "TCN",
    }
    for p2_name, p2_m in phase2_metrics.items():
        p1_name = name_map.get(p2_name)
        if p1_name and p1_name in phase1:
            p1_m = phase1[p1_name]
            print(f"\n  {p1_name} -> {p2_name}:")
            sla_delta = (p2_m["request_sla"] - p1_m["request_sla"]) * 100
            ext_delta = (p2_m["extreme_sla"] - p1_m["extreme_sla"]) * 100
            cs_pct = (p2_m["total_cold_starts"] / p1_m["total_cold_starts"] - 1) * 100
            cost_pct = (p2_m["total_cost"] / p1_m["total_cost"] - 1) * 100
            print(f"    Request SLA: {p1_m['request_sla']:.6f} -> {p2_m['request_sla']:.6f}  ({sla_delta:+.2f}pp)")
            print(f"    Extreme SLA: {p1_m['extreme_sla']:.6f} -> {p2_m['extreme_sla']:.6f}  ({ext_delta:+.2f}pp)")
            print(f"    Cold starts: {p1_m['total_cold_starts']:,} -> {p2_m['total_cold_starts']:,}  ({cs_pct:+.1f}%)")
            print(f"    Total cost:  {p1_m['total_cost']:,.0f} -> {p2_m['total_cost']:,.0f}  ({cost_pct:+.1f}%)")


def main():
    print("=" * 60)
    print("Phase 2: Risk-Aware EVT+CVaR Runner - Azure Dataset")
    print("=" * 60)
    print(f"  alpha={ALPHA}, W={VOLATILITY_WINDOW}, EVT_threshold=P{EVT_THRESHOLD_PERCENTILE}")

    train, val, test = load_data()
    threshold = compute_threshold(train)
    models = create_risk_aware_models()
    all_results, all_metrics, all_diagnostics, all_evt_params = \
        run_all_models(models, train, val, test, threshold)
    save_results(all_results, all_metrics, all_diagnostics, all_evt_params, threshold)
    print_combined_summary(all_metrics, threshold)
    print_improvement_analysis(all_metrics)
    print("\n[DONE] Phase 2 risk-aware experiments complete")
    return all_results, all_metrics


if __name__ == "__main__":
    main()
