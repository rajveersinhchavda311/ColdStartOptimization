"""
Cache TCN Standardized Training Residuals (for figure generation)
==================================================================

Purpose:
    graphs/phase5/tail_heaviness_comparison.png needs the TCN's
    standardized training residuals (z = epsilon / sigma_train). They were
    never persisted during the Phase 2 runs, and the graph script must not
    plot synthetic stand-in data.

    This script retrains the TCN through the exact Phase 2 code path
    (TCNModel.fit sets seed=42 and cudnn-deterministic at entry), computes
    standardized training residuals, and saves them as NEW files. It never
    modifies any existing file in results/.

Verification gate (refuses to save unfaithful residuals):
    The recomputed sigma_train and the EVT fit (xi, CVaR_z) on the
    recomputed z must match the values stored by the original Phase 2 runs:
        sigma_train within 0.5% relative
        xi          within +/-0.01 absolute
        CVaR_z      within +/-0.05 absolute
    If any check fails, nothing is written and the script exits non-zero —
    in that case the figure must fall back to dropping the TCN row.

Outputs (additive only):
    results/phase2/azure/TCN_training_residuals_z.npy
    results/phase2/azure/TCN_training_residuals_provenance.json
    results/phase2/huawei/combined/TCN_training_residuals_z.npy
    results/phase2/huawei/combined/TCN_training_residuals_provenance.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.tcn import TCNModel
from evaluation.evt import fit_evt_pipeline

DATASETS = [
    {
        "name": "azure",
        "data_dir": os.path.join(PROJECT_ROOT, "data", "processed", "azure"),
        "results_dir": os.path.join(PROJECT_ROOT, "results", "phase2", "azure"),
        "diagnostics_csv": "RiskAware(TCN)_diagnostics.csv",
    },
    {
        "name": "huawei_combined",
        "data_dir": os.path.join(PROJECT_ROOT, "data", "processed", "huawei", "combined"),
        "results_dir": os.path.join(PROJECT_ROOT, "results", "phase2", "huawei", "combined"),
        "diagnostics_csv": "RiskAware(TCN)_diagnostics.csv",
    },
]

SIGMA_RTOL = 0.005   # 0.5% relative
XI_ATOL = 0.01
CVAR_ATOL = 0.05


def load_stored_fingerprints(results_dir: str, diagnostics_csv: str) -> dict:
    """Stored sigma_train (= sigma_t during warm-up) and EVT parameters."""
    diag = pd.read_csv(os.path.join(results_dir, diagnostics_csv), nrows=1)
    with open(os.path.join(results_dir, "evt_parameters.json")) as f:
        evt = json.load(f)["RiskAware(TCN)"]
    return {
        "sigma_train": float(diag["sigma_t"].iloc[0]),
        "xi": float(evt["xi"]),
        "cvar_z": float(evt["cvar_z"]),
    }


def process_dataset(ds: dict) -> bool:
    print(f"\n{'=' * 60}\nDataset: {ds['name']}\n{'=' * 60}")

    train = pd.read_csv(os.path.join(ds["data_dir"], "train.csv"), parse_dates=["timestamp"])
    val = pd.read_csv(os.path.join(ds["data_dir"], "val.csv"), parse_dates=["timestamp"])

    stored = load_stored_fingerprints(ds["results_dir"], ds["diagnostics_csv"])
    print(f"  Stored fingerprints: sigma_train={stored['sigma_train']:,.4f}, "
          f"xi={stored['xi']:.4f}, cvar_z={stored['cvar_z']:.4f}")

    # Exact Phase 2 code path: seed=42 is set at the top of fit()
    model = TCNModel()
    model.fit(train, val_df=val)

    preds = model.predict(train)
    actual = train["concurrency"].values.astype(np.float64)
    residuals = actual - preds
    sigma_train = float(np.std(residuals))
    z = residuals / sigma_train

    evt = fit_evt_pipeline(z, threshold_percentile=90, alpha=0.99)

    d_sigma = abs(sigma_train - stored["sigma_train"]) / stored["sigma_train"]
    d_xi = abs(evt["xi"] - stored["xi"])
    d_cvar = abs(evt["cvar_z"] - stored["cvar_z"])

    checks = [
        ("sigma_train rel. delta", d_sigma, SIGMA_RTOL),
        ("xi abs. delta", d_xi, XI_ATOL),
        ("cvar_z abs. delta", d_cvar, CVAR_ATOL),
    ]
    ok = True
    for label, delta, tol in checks:
        status = "PASS" if delta <= tol else "FAIL"
        if delta > tol:
            ok = False
        print(f"  [GATE {status}] {label}: {delta:.6f} (tol {tol})")

    if not ok:
        print(f"  GATE FAILED for {ds['name']} — nothing written. "
              f"The retrained TCN does not reproduce the original Phase 2 model "
              f"on this machine; fall back to dropping the TCN row from the figure.")
        return False

    z_path = os.path.join(ds["results_dir"], "TCN_training_residuals_z.npy")
    np.save(z_path, z)
    provenance = {
        "description": "Standardized TCN training residuals (z = eps / sigma_train), "
                       "recomputed via the exact Phase 2 code path (seed=42) for "
                       "tail_heaviness_comparison.png. Additive artifact; no original "
                       "Phase 2 file was modified.",
        "n": int(len(z)),
        "recomputed": {"sigma_train": sigma_train, "xi": evt["xi"], "cvar_z": evt["cvar_z"]},
        "stored_phase2": stored,
        "gate_deltas": {"sigma_rel": d_sigma, "xi_abs": d_xi, "cvar_abs": d_cvar},
        "gate_tolerances": {"sigma_rtol": SIGMA_RTOL, "xi_atol": XI_ATOL, "cvar_atol": CVAR_ATOL},
    }
    with open(os.path.join(ds["results_dir"], "TCN_training_residuals_provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"  Saved {z_path} ({len(z):,} residuals) + provenance sidecar.")
    return True


def main():
    results = {ds["name"]: process_dataset(ds) for ds in DATASETS}
    print(f"\n{'=' * 60}\nSummary: {results}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
