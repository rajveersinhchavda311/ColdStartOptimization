"""
Phase 3 Runner — EVT-CVaR Sensitivity Analysis (Azure Only)
============================================================

Phase 3A: One-at-a-time sweep (9 configurations)
  Anchor: α=0.99, W=30, threshold=P90

  α sweep    (W=30, threshold=P90):  α ∈ {0.95, 0.975, 0.99}
  W sweep    (α=0.99, threshold=P90): W ∈ {10, 30, 60}
  P sweep    (α=0.99, W=30):         P ∈ {P85, P90, P95}

Phase 3B: Interaction check (8 configurations)
  α ∈ {0.95, 0.99} × W ∈ {10, 60} × threshold ∈ {P85, P95}

Models: RiskAware(TCN) and RiskAware(Reactive) only.
  These bracket the quality spectrum — best (TCN) and simplest (Reactive).

Optimization: the base model (Reactive, TCN) is trained ONCE per model type.
  EVT parameters are varied cheaply on pre-computed residuals, avoiding
  ~75 minutes of redundant TCN re-training.

Hard constraints (enforced by design):
  - results/phase1/ and results/phase2/ are NEVER written to
  - Same test set, simulator, extreme threshold, and cost model as Phases 1–2
  - Extreme threshold computed from training data only

Outputs:
  results/phase3/azure/runs/{config_key}_{model_name}_metrics.json
  results/phase3/azure/runs/{config_key}_{model_name}_diagnostics.csv
  results/phase3/azure/summary_3a.csv
  results/phase3/azure/summary_3b.csv
  results/phase3/azure/all_metrics.json
  results/phase3/azure/evt_parameters.json
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
from models.tcn import TCNModel
from models.risk_aware import RiskAwareModel
from evaluation.evt import fit_evt_pipeline
from evaluation.simulator import simulate
from evaluation.metrics import compute_metrics, C_COLD, C_IDLE
from evaluation.extreme import compute_extreme_threshold, EXTREME_PERCENTILE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = os.path.join(PROJECT_ROOT, "data",    "processed", "azure")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase3",   "azure")
RUNS_DIR    = os.path.join(RESULTS_DIR, "runs")

# ---------------------------------------------------------------------------
# Anchor — matches Phase 2 exactly
# ---------------------------------------------------------------------------
ANCHOR_ALPHA     = 0.99
ANCHOR_W         = 30
ANCHOR_THRESHOLD = 90

# ---------------------------------------------------------------------------
# Phase 3A: one-at-a-time configurations
# ---------------------------------------------------------------------------
PHASE3A_ALPHA_SWEEP = [
    {"sweep": "alpha", "alpha": 0.950, "W": 30, "threshold": 90},
    {"sweep": "alpha", "alpha": 0.975, "W": 30, "threshold": 90},
    {"sweep": "alpha", "alpha": 0.990, "W": 30, "threshold": 90},  # anchor
]

PHASE3A_W_SWEEP = [
    {"sweep": "W", "alpha": 0.99, "W": 10, "threshold": 90},
    {"sweep": "W", "alpha": 0.99, "W": 30, "threshold": 90},  # anchor
    {"sweep": "W", "alpha": 0.99, "W": 60, "threshold": 90},
]

PHASE3A_P_SWEEP = [
    {"sweep": "threshold", "alpha": 0.99, "W": 30, "threshold": 85},
    {"sweep": "threshold", "alpha": 0.99, "W": 30, "threshold": 90},  # anchor
    {"sweep": "threshold", "alpha": 0.99, "W": 30, "threshold": 95},
]

PHASE3A_CONFIGS = PHASE3A_ALPHA_SWEEP + PHASE3A_W_SWEEP + PHASE3A_P_SWEEP

# ---------------------------------------------------------------------------
# Phase 3B: full factorial on boundary values
# ---------------------------------------------------------------------------
PHASE3B_CONFIGS = []
for _alpha in [0.95, 0.99]:
    for _W in [10, 60]:
        for _P in [85, 95]:
            PHASE3B_CONFIGS.append({
                "alpha": _alpha, "W": _W, "threshold": _P
            })


def config_key(cfg: dict) -> str:
    """Canonical string key for a configuration dict."""
    return f"a{cfg['alpha']:.3f}_W{cfg['W']:03d}_P{cfg['threshold']:02d}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    print("\n[1/5] Loading Azure data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["timestamp"])
    val   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"),   parse_dates=["timestamp"])
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),  parse_dates=["timestamp"])
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    assert train["timestamp"].iloc[-1] < val["timestamp"].iloc[0],  "Train/val overlap!"
    assert val["timestamp"].iloc[-1]   < test["timestamp"].iloc[0], "Val/test overlap!"
    return train, val, test


# ---------------------------------------------------------------------------
# Base model training — called ONCE per model type
# ---------------------------------------------------------------------------
class BaseModelCache:
    """
    Holds a fitted base model and its pre-computed training residuals.
    Avoids re-training the base model for every EVT config variation.
    """
    def __init__(self, model_name: str, base_model, residuals: np.ndarray):
        self.model_name   = model_name
        self.base_model   = base_model
        self.residuals    = residuals
        self.sigma_train  = float(np.std(residuals))
        self.z            = residuals / self.sigma_train  # standardized


def train_base_models(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Train Reactive and TCN once each, return BaseModelCache per model."""
    print("\n[2/5] Training base models (once each)...")
    caches = {}

    for name, ModelClass, fit_kwargs in [
        ("Reactive", ReactiveModel, {}),
        ("TCN",      TCNModel,      {"val_df": val_df}),
    ]:
        print(f"\n  Training {name}...")
        t0 = time.time()
        base = ModelClass()
        base.fit(train_df, **fit_kwargs)
        fit_time = time.time() - t0

        preds = base.predict(train_df)
        actual = train_df["concurrency"].values.astype(np.float64)
        residuals = actual - preds
        sigma = float(np.std(residuals))

        print(f"  [{name}] fit_time={fit_time:.1f}s  "
              f"sigma_train={sigma:,.0f}  "
              f"residuals: mean={np.mean(residuals):,.0f}  "
              f"std={sigma:,.0f}")

        caches[name] = BaseModelCache(name, base, residuals)

    return caches


# ---------------------------------------------------------------------------
# Single-config runner
# ---------------------------------------------------------------------------
def run_single_config(cache: BaseModelCache, cfg: dict,
                      test_df: pd.DataFrame, threshold: float) -> dict:
    """
    Run one (model, EVT-config) combination without re-training the base model.

    Steps:
      1. Fit EVT on pre-computed standardized residuals (z = eps / sigma_train)
         using this config's alpha and threshold percentile.
      2. Manually assign fitted parameters to a new RiskAwareModel that wraps
         the already-trained base model.
      3. Call predict() — sequential rolling-window logic uses config's W.
      4. Simulate, compute metrics.
    """
    key = config_key(cfg)

    # Step 1: EVT fit on pre-computed z
    evt_params = fit_evt_pipeline(
        cache.z,
        threshold_percentile=cfg["threshold"],
        alpha=cfg["alpha"],
    )

    # Step 2: Build wrapper; bypass fit() to avoid base-model re-training
    wrapper = RiskAwareModel(
        base_model=cache.base_model,
        alpha=cfg["alpha"],
        volatility_window=cfg["W"],
        evt_threshold_percentile=cfg["threshold"],
    )
    wrapper.sigma_train = cache.sigma_train
    wrapper.cvar_z      = evt_params["cvar_z"]
    wrapper.evt_params  = evt_params

    # Step 3: Predict and simulate
    t0 = time.time()
    results_df = simulate(wrapper, test_df, threshold, c_cold=C_COLD, c_idle=C_IDLE)
    sim_time = time.time() - t0

    metrics = compute_metrics(results_df)
    metrics["alpha"]               = cfg["alpha"]
    metrics["volatility_window"]   = cfg["W"]
    metrics["evt_threshold_percentile"] = cfg["threshold"]
    metrics["config_key"]          = key
    metrics["base_model"]          = cache.model_name
    metrics["sim_time_seconds"]    = sim_time

    diag = wrapper._diagnostics

    return {
        "key":        key,
        "cfg":        cfg,
        "metrics":    metrics,
        "results_df": results_df,
        "diag":       diag,
        "evt_params": evt_params,
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def save_run(run: dict) -> None:
    """Save per-run metrics JSON and diagnostics CSV."""
    key        = run["key"]
    model_name = run["metrics"]["base_model"]
    metrics    = run["metrics"]
    results_df = run["results_df"]
    diag       = run["diag"]

    # Diagnostics CSV (per-timestep)
    enriched = results_df.copy()
    enriched["base_prediction"] = diag["base_prediction"]
    enriched["sigma_t"]         = diag["sigma_t"]
    enriched["buffer_t"]        = diag["buffer_t"]
    enriched.to_csv(
        os.path.join(RUNS_DIR, f"{key}_{model_name}_diagnostics.csv"),
        index=False,
    )

    # Metrics JSON (summary)
    safe_metrics = {k: (v if not isinstance(v, dict) else v)
                    for k, v in metrics.items()}
    with open(os.path.join(RUNS_DIR, f"{key}_{model_name}_metrics.json"), "w") as f:
        json.dump(safe_metrics, f, indent=2)


def build_summary_row(run: dict, sweep: str = None) -> dict:
    m   = run["metrics"]
    cfg = run["cfg"]
    row = {
        "alpha":              cfg["alpha"],
        "W":                  cfg["W"],
        "threshold":          cfg["threshold"],
        "config_key":         run["key"],
        "model":              m["base_model"],
        "request_sla":        m["request_sla"],
        "extreme_sla":        m["extreme_sla"],
        "total_cost":         m["total_cost"],
        "cold_cost":          m["cold_cost"],
        "idle_cost":          m["idle_cost"],
        "total_cold_starts":  m["total_cold_starts"],
        "cold_start_rate":    m["cold_start_rate"],
    }
    if sweep is not None:
        row["sweep"] = sweep
    return row


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------
def run_all(caches: dict, test_df: pd.DataFrame, threshold: float):
    """Run all Phase 3A and 3B configurations for both model types."""
    print("\n[3/5] Running all configurations...")
    os.makedirs(RUNS_DIR, exist_ok=True)

    all_results    = {}   # key: (config_key, model_name) → run dict
    all_evt_params = {}

    def run_and_record(cfg, label):
        key = config_key(cfg)
        for model_name, cache in caches.items():
            rk = (key, model_name)
            print(f"\n  [{label}] {key}  model={model_name}")
            run = run_single_config(cache, cfg, test_df, threshold)
            m   = run["metrics"]
            print(f"    Req SLA={m['request_sla']:.6f}  "
                  f"Ext SLA={m['extreme_sla']:.6f}  "
                  f"Cost={m['total_cost']/1e6:.1f}M  "
                  f"Cold={m['total_cold_starts']:,}")
            save_run(run)
            all_results[rk]    = run
            all_evt_params[rk] = run["evt_params"]
        return run  # last model; used externally if needed

    # Phase 3A
    print("\n" + "="*60)
    print("  PHASE 3A: One-at-a-time sweep")
    print("="*60)
    seen_keys = set()
    for cfg in PHASE3A_CONFIGS:
        k = config_key(cfg)
        if k in seen_keys:
            # Anchor appears in multiple sweeps; run it once, reuse results
            print(f"\n  [3A-{cfg['sweep']}] {k}  (already computed — reusing)")
            continue
        seen_keys.add(k)
        run_and_record(cfg, f"3A-{cfg['sweep']}")

    # Phase 3B
    print("\n" + "="*60)
    print("  PHASE 3B: Interaction check (2×2×2 factorial)")
    print("="*60)
    for cfg in PHASE3B_CONFIGS:
        run_and_record(cfg, "3B")

    return all_results, all_evt_params


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
def save_summaries(all_results: dict, all_evt_params: dict):
    print(f"\n[4/5] Saving summaries to {RESULTS_DIR}...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Phase 3A summary ---
    rows_3a = []
    seen_keys_3a = set()
    for cfg in PHASE3A_CONFIGS:
        k = config_key(cfg)
        sweep = cfg["sweep"]
        for model_name in ["Reactive", "TCN"]:
            rk = (k, model_name)
            if rk not in all_results:
                continue
            run = all_results[rk]
            row = build_summary_row(run, sweep=sweep)
            row_key = (k, model_name, sweep)
            if row_key not in seen_keys_3a:
                rows_3a.append(row)
                seen_keys_3a.add(row_key)
    df_3a = pd.DataFrame(rows_3a)
    df_3a.to_csv(os.path.join(RESULTS_DIR, "summary_3a.csv"), index=False)
    print("  Saved: summary_3a.csv")

    # --- Phase 3B summary ---
    rows_3b = []
    for cfg in PHASE3B_CONFIGS:
        k = config_key(cfg)
        for model_name in ["Reactive", "TCN"]:
            rk = (k, model_name)
            if rk not in all_results:
                continue
            rows_3b.append(build_summary_row(all_results[rk]))
    df_3b = pd.DataFrame(rows_3b)
    df_3b.to_csv(os.path.join(RESULTS_DIR, "summary_3b.csv"), index=False)
    print("  Saved: summary_3b.csv")

    # --- all_metrics.json (complete record) ---
    all_metrics_serializable = {}
    for (k, m), run in all_results.items():
        all_metrics_serializable[f"{k}__{m}"] = run["metrics"]
    with open(os.path.join(RESULTS_DIR, "all_metrics.json"), "w") as f:
        json.dump(all_metrics_serializable, f, indent=2)
    print("  Saved: all_metrics.json")

    # --- evt_parameters.json ---
    evt_serializable = {}
    for (k, m), params in all_evt_params.items():
        evt_serializable[f"{k}__{m}"] = params
    with open(os.path.join(RESULTS_DIR, "evt_parameters.json"), "w") as f:
        json.dump(evt_serializable, f, indent=2)
    print("  Saved: evt_parameters.json")

    return df_3a, df_3b


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary(df_3a: pd.DataFrame, df_3b: pd.DataFrame):
    print(f"\n[5/5] Results summary")
    print("="*80)

    print("\nPHASE 3A -- alpha sweep (W=30, P=90)")
    sub = df_3a[df_3a["sweep"] == "alpha"].sort_values(["alpha", "model"])
    print(sub[["alpha", "W", "threshold", "model",
               "request_sla", "extreme_sla", "total_cost"]].to_string(index=False))

    print("\nPHASE 3A -- W sweep (alpha=0.99, P=90)")
    sub = df_3a[df_3a["sweep"] == "W"].sort_values(["W", "model"])
    print(sub[["alpha", "W", "threshold", "model",
               "request_sla", "extreme_sla", "total_cost"]].to_string(index=False))

    print("\nPHASE 3A -- threshold sweep (alpha=0.99, W=30)")
    sub = df_3a[df_3a["sweep"] == "threshold"].sort_values(["threshold", "model"])
    print(sub[["alpha", "W", "threshold", "model",
               "request_sla", "extreme_sla", "total_cost"]].to_string(index=False))

    print("\nPHASE 3B — all 8 interaction configs")
    print(df_3b[["alpha", "W", "threshold", "model",
                 "request_sla", "extreme_sla", "total_cost"]].sort_values(
        ["alpha", "W", "threshold", "model"]).to_string(index=False))

    # Robustness report
    all_sla = pd.concat([df_3a["request_sla"], df_3b["request_sla"]])
    print(f"\n  Request SLA across all configs: "
          f"min={all_sla.min():.6f}  max={all_sla.max():.6f}  "
          f"range={all_sla.max()-all_sla.min():.6f}")

    anchor_3a = df_3a[
        (df_3a["alpha"] == ANCHOR_ALPHA) &
        (df_3a["W"] == ANCHOR_W) &
        (df_3a["threshold"] == ANCHOR_THRESHOLD)
    ]
    if len(anchor_3a) > 0:
        print("\n  Anchor config results (should match Phase 2 closely):")
        print(anchor_3a[["model", "request_sla", "extreme_sla", "total_cost"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("="*70)
    print("Phase 3: EVT-CVaR Sensitivity Analysis - Azure Dataset")
    print("="*70)
    print(f"  Anchor: alpha={ANCHOR_ALPHA}, W={ANCHOR_W}, threshold=P{ANCHOR_THRESHOLD}")
    print(f"  Phase 3A configs: {len(PHASE3A_CONFIGS)} (alpha x3 + W x3 + P x3)")
    print(f"  Phase 3B configs: {len(PHASE3B_CONFIGS)} (2x2x2 factorial)")
    print(f"  Models: Reactive, TCN")
    print(f"  WARNING: Phase 1/2 results are never modified.\n")

    train, val, test = load_data()

    threshold = compute_extreme_threshold(train, percentile=EXTREME_PERCENTILE)
    print(f"\n  Extreme threshold (P{EXTREME_PERCENTILE} of train): {threshold:,.0f}")

    caches = train_base_models(train, val)

    all_results, all_evt_params = run_all(caches, test, threshold)

    df_3a, df_3b = save_summaries(all_results, all_evt_params)

    print_summary(df_3a, df_3b)

    print("\n[DONE] Phase 3 sensitivity analysis complete.")
    print(f"  Results: {RESULTS_DIR}")
    return df_3a, df_3b


if __name__ == "__main__":
    main()
