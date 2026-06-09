"""
Phase 2 Runner — Risk-Aware EVT+CVaR on Huawei Combined Dataset
================================================================

Wraps Phase 1 forecasting models with RiskAwareModel on Huawei combined.
Uses IDENTICAL parameters to Azure Phase 2 — no modification.

Anchor parameters (frozen from Azure Phase 2):
    alpha = 0.99
    volatility_window (W) = 30
    evt_threshold_percentile = 90

Wrapped models (excludes Static_P90):
    1. RiskAware(Reactive)
    2. RiskAware(Forecast_Only)
    3. RiskAware(Seasonal_Naive)
    4. RiskAware(Linear_Seasonal)
    5. RiskAware(TCN)

Critical output — evt_parameters.json:
    For each model: xi (GPD shape), beta (GPD scale), u (POT threshold),
    cvar_z, and the ratio cvar_z / K_GAUSSIAN (where K_GAUSSIAN ≈ 2.6652).

Outputs:
    results/phase2/huawei/combined/metrics.json
    results/phase2/huawei/combined/comparison.csv
    results/phase2/huawei/combined/evt_parameters.json  (CRITICAL)
    results/phase2/huawei/combined/RiskAware({ModelName})_diagnostics.csv
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import norm

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

DATA_DIR    = os.path.join(PROJECT_ROOT, "data",    "processed", "huawei", "combined")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase2",   "huawei", "combined")
PHASE1_DIR  = os.path.join(PROJECT_ROOT, "results", "phase1",   "huawei", "combined")

ALPHA                  = 0.99
VOLATILITY_WINDOW      = 30
EVT_THRESHOLD_PERCENTILE = 90

# Gaussian CVaR multiplier at alpha=0.99 (scipy-derived)
K_GAUSSIAN = norm.pdf(norm.ppf(ALPHA)) / (1 - ALPHA)


def load_data():
    print("\n[1/6] Loading Huawei combined data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["timestamp"])
    val   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"),   parse_dates=["timestamp"])
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),  parse_dates=["timestamp"])

    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    assert "lag_1440" in train.columns, "lag_1440 missing — run preprocess_huawei.py first"
    assert train["timestamp"].iloc[-1] < val["timestamp"].iloc[0], "Train/val overlap"
    assert val["timestamp"].iloc[-1] < test["timestamp"].iloc[0],  "Val/test overlap"
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
        cvar_z = model.cvar_z
        ratio  = cvar_z / K_GAUSSIAN
        print(f"\n  EVT:  CVaR_z={cvar_z:.4f}  K_Gaussian={K_GAUSSIAN:.4f}  "
              f"ratio={ratio:.3f}x  xi={model.evt_params['xi']:.4f}")

        all_results[model.name]    = results_df
        all_metrics[model.name]    = metrics
        all_diagnostics[model.name] = diag

        # Augment evt_params with ratio
        ep = dict(model.evt_params)
        ep["cvar_z_over_k_gaussian"] = float(ratio)
        ep["k_gaussian"] = float(K_GAUSSIAN)
        all_evt_params[model.name] = ep

    return all_results, all_metrics, all_diagnostics, all_evt_params


def save_results(all_results, all_metrics, all_diagnostics, all_evt_params, threshold):
    print(f"\n[4/6] Saving results to {RESULTS_DIR}...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for model_name, results_df in all_results.items():
        # Enriched diagnostics CSV
        enriched = results_df.copy()
        diag = all_diagnostics[model_name]
        enriched["base_prediction"] = diag["base_prediction"]
        enriched["sigma_t"]         = diag["sigma_t"]
        enriched["buffer_t"]        = diag["buffer_t"]
        diag_path = os.path.join(RESULTS_DIR, f"{model_name}_diagnostics.csv")
        enriched.to_csv(diag_path, index=False)
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

    # Summary text
    with open(os.path.join(RESULTS_DIR, "comparison_summary.txt"), "w") as f:
        f.write("Phase 2 Risk-Aware Comparison — Huawei Combined Dataset\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Cost model: c_cold={C_COLD}, c_idle={C_IDLE}\n")
        f.write(f"Extreme threshold: P{EXTREME_PERCENTILE}(train) = {threshold:,.0f}\n")
        f.write(f"Risk params: alpha={ALPHA}, W={VOLATILITY_WINDOW}, "
                f"EVT_threshold=P{EVT_THRESHOLD_PERCENTILE}\n")
        f.write(f"K_GAUSSIAN = {K_GAUSSIAN:.6f}\n\n")
        f.write(format_metrics_table(all_metrics))

    # EVT parameters JSON (critical deliverable)
    with open(os.path.join(RESULTS_DIR, "evt_parameters.json"), "w") as f:
        json.dump(all_evt_params, f, indent=2)
    print("  Saved: evt_parameters.json")

    # Metrics JSON
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("  Saved: metrics.json")


def load_phase1_metrics():
    path = os.path.join(PHASE1_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def print_improvement_analysis(phase2_metrics):
    print(f"\n[5/6] Cold-start reduction (Phase 1 -> Phase 2)")
    print("=" * 70)
    phase1 = load_phase1_metrics()
    if phase1 is None:
        print("  Phase 1 metrics not found; skipping comparison.")
        return
    name_map = {
        "RiskAware(Reactive)":       "Reactive",
        "RiskAware(Forecast_Only)":  "Forecast_Only",
        "RiskAware(Seasonal_Naive)": "Seasonal_Naive",
        "RiskAware(Linear_Seasonal)":"Linear_Seasonal",
        "RiskAware(TCN)":            "TCN",
    }
    for p2_name, p2_m in phase2_metrics.items():
        p1_name = name_map.get(p2_name)
        if p1_name and p1_name in phase1:
            p1_m = phase1[p1_name]
            cs_pct = (p2_m["total_cold_starts"] / max(p1_m["total_cold_starts"], 1) - 1) * 100
            ext_delta = (p2_m["extreme_sla"] - p1_m["extreme_sla"]) * 100
            print(f"  {p1_name:16s}: cold_starts {p1_m['total_cold_starts']:>12,} -> "
                  f"{p2_m['total_cold_starts']:>12,}  ({cs_pct:+.1f}%)  "
                  f"extreme_sla d{ext_delta:+.3f}pp")


def print_evt_summary(all_evt_params):
    print(f"\n[6/6] EVT Parameter Summary (K_GAUSSIAN = {K_GAUSSIAN:.4f})")
    print("=" * 75)
    print(f"  {'Model':<28} {'xi':>8} {'beta':>8} {'CVaR_z':>8} {'ratio':>8}")
    print("  " + "-" * 65)
    for name, ep in all_evt_params.items():
        print(f"  {name:<28} {ep['xi']:>8.4f} {ep['beta']:>8.4f} "
              f"{ep['cvar_z']:>8.4f} {ep['cvar_z_over_k_gaussian']:>7.3f}x")


def main():
    print("=" * 60)
    print("Phase 2: Risk-Aware EVT+CVaR — Huawei Combined")
    print("=" * 60)
    print(f"  alpha={ALPHA}, W={VOLATILITY_WINDOW}, "
          f"EVT_threshold=P{EVT_THRESHOLD_PERCENTILE}")
    print(f"  K_GAUSSIAN = {K_GAUSSIAN:.6f}")

    train, val, test = load_data()
    threshold = compute_threshold(train)
    models = create_risk_aware_models()
    all_results, all_metrics, all_diagnostics, all_evt_params = \
        run_all_models(models, train, val, test, threshold)
    save_results(all_results, all_metrics, all_diagnostics, all_evt_params, threshold)
    print_improvement_analysis(all_metrics)
    print_evt_summary(all_evt_params)
    print("\n[DONE] Phase 2 Huawei complete")
    return all_results, all_metrics


if __name__ == "__main__":
    main()
