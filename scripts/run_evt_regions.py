"""
EVT Parameters for Individual Huawei Regions (R1–R5) — Phase 5
===============================================================

For each region R1–R5, for each base model (Reactive, Forecast_Only,
Seasonal_Naive, Linear_Seasonal, TCN):
  1. Train the base model on the region's training data
  2. Compute training residuals → sigma_train → standardized z
  3. Fit EVT (same pipeline: evaluation/evt.py, α=0.99, threshold=P90)
  4. Save ξ, β, u, CVaR_z, ratio to results/phase2/huawei/{region}/evt_parameters.json

No full simulation — EVT parameters only. The purpose is the cross-dataset
ξ / CVaR_z/K_GAUSSIAN table for the generalization paper section.

Outputs:
    results/phase2/huawei/R1/evt_parameters.json
    results/phase2/huawei/R2/evt_parameters.json
    results/phase2/huawei/R3/evt_parameters.json
    results/phase2/huawei/R4/evt_parameters.json
    results/phase2/huawei/R5/evt_parameters.json
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
from evaluation.evt import fit_evt_pipeline

REGIONS   = ["R1", "R2", "R3", "R4", "R5"]
HUAWEI_DIR = os.path.join(PROJECT_ROOT, "data",    "processed", "huawei")
RESULTS_BASE = os.path.join(PROJECT_ROOT, "results", "phase2",  "huawei")

ALPHA                    = 0.99
EVT_THRESHOLD_PERCENTILE = 90

K_GAUSSIAN = norm.pdf(norm.ppf(ALPHA)) / (1 - ALPHA)


def make_base_models():
    return [
        ReactiveModel(),
        ForecastOnlyModel(),
        SeasonalNaiveModel(),
        LinearSeasonalModel(),
        TCNModel(),
    ]


def compute_evt_for_model(model, train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """Fit model, compute training residuals, run EVT pipeline."""
    t0 = time.time()
    if isinstance(model, TCNModel):
        model.fit(train, val_df=val)
    else:
        model.fit(train)
    fit_time = time.time() - t0

    # Training residuals (allowed during fit)
    base_preds = model.predict(train)
    actual     = train["concurrency"].values.astype(np.float64)
    residuals  = actual - base_preds

    sigma_train = float(np.std(residuals))
    if sigma_train < 1e-8:
        return {
            "error": f"sigma_train near zero ({sigma_train:.2e})",
            "fit_time_seconds": fit_time,
        }

    z = residuals / sigma_train

    print(f"    sigma_train={sigma_train:,.2f}  "
          f"z_mean={np.mean(z):.3f}  z_std={np.std(z):.3f}")

    evt_params = fit_evt_pipeline(
        z,
        threshold_percentile=EVT_THRESHOLD_PERCENTILE,
        alpha=ALPHA,
    )

    ratio = evt_params["cvar_z"] / K_GAUSSIAN
    result = dict(evt_params)
    result["sigma_train"]           = sigma_train
    result["cvar_z_over_k_gaussian"] = float(ratio)
    result["k_gaussian"]             = float(K_GAUSSIAN)
    result["fit_time_seconds"]       = fit_time
    return result


def process_region(region: str) -> None:
    data_dir   = os.path.join(HUAWEI_DIR, region)
    result_dir = os.path.join(RESULTS_BASE, region)
    os.makedirs(result_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Region: {region}")
    print(f"{'='*60}")

    train = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=["timestamp"])
    val   = pd.read_csv(os.path.join(data_dir, "val.csv"),   parse_dates=["timestamp"])

    print(f"  Train: {len(train):,} rows  |  Val: {len(val):,} rows")
    assert "lag_1440" in train.columns, \
        f"lag_1440 missing in {region} train — run preprocess_huawei.py first"

    region_evt = {}

    for model in make_base_models():
        model_name = f"RiskAware({model.name})"
        print(f"\n  --- {model_name} ---")
        try:
            params = compute_evt_for_model(model, train, val)
            region_evt[model_name] = params
            if "error" not in params:
                print(f"    xi={params['xi']:.4f}  CVaR_z={params['cvar_z']:.4f}  "
                      f"ratio={params['cvar_z_over_k_gaussian']:.3f}x")
        except Exception as e:
            print(f"    ERROR: {e}")
            region_evt[model_name] = {"error": str(e)}

    out_path = os.path.join(result_dir, "evt_parameters.json")
    with open(out_path, "w") as f:
        json.dump(region_evt, f, indent=2)
    print(f"\n  Saved: {out_path}")


def print_cross_region_summary():
    print(f"\n{'='*75}")
    print("Cross-Region EVT Summary")
    print(f"K_GAUSSIAN = {K_GAUSSIAN:.4f}")
    print(f"{'='*75}")

    model_names = ["RiskAware(Reactive)", "RiskAware(Forecast_Only)",
                   "RiskAware(Seasonal_Naive)", "RiskAware(Linear_Seasonal)",
                   "RiskAware(TCN)"]
    header = f"  {'Model':<28} " + "  ".join(f"{r:>6}" for r in REGIONS)
    print(header + "  (xi values)")
    print("  " + "-" * (28 + 10 * len(REGIONS)))

    for mname in model_names:
        row = f"  {mname:<28}"
        for region in REGIONS:
            path = os.path.join(RESULTS_BASE, region, "evt_parameters.json")
            xi_str = "  N/A"
            if os.path.exists(path):
                with open(path) as f:
                    params = json.load(f)
                if mname in params and "xi" in params[mname]:
                    xi_str = f"{params[mname]['xi']:>6.3f}"
            row += f"  {xi_str}"
        print(row)


def main():
    print("=" * 60)
    print("Phase 5: EVT Parameters for Huawei Regions R1–R5")
    print("=" * 60)
    print(f"  alpha={ALPHA}, EVT_threshold=P{EVT_THRESHOLD_PERCENTILE}")
    print(f"  K_GAUSSIAN = {K_GAUSSIAN:.6f}")

    for region in REGIONS:
        process_region(region)

    print_cross_region_summary()
    print("\n[DONE] EVT parameters for all regions saved.")


if __name__ == "__main__":
    main()
