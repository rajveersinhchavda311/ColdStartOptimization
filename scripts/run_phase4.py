"""
Phase 4 Runner -- EVT-CVaR Component Ablation Study (Azure Only)
================================================================

2x2 factorial ablation of the buffer formula: buffer_t = sigma_t * CVaR_z

  sigma component:    static (sigma_train, constant) vs dynamic (sigma_t, rolling)
  multiplier:         Gaussian (K_GAUSSIAN) vs EVT (CVaR_z from GPD fit)

Conditions:
  C0 -- No buffer        -- Phase 1 result (loaded from disk, not re-run)
  C1 -- Static+Gaussian  -- buffer = sigma_train * K_GAUSSIAN  (constant buffer)
  C2 -- Dynamic+Gaussian -- buffer = sigma_t * K_GAUSSIAN      (rolling sigma)
  C3 -- Static+EVT       -- buffer = sigma_train * CVaR_z      (constant buffer)
  C4 -- Dynamic+EVT      -- Phase 2 result (loaded from disk, not re-run)

K_GAUSSIAN derivation:
  For X ~ N(0,1), CVaR at alpha is: E[X | X > z_alpha] = phi(z_alpha)/(1-alpha)
  where phi is the standard normal PDF and z_alpha = Phi^{-1}(alpha).
  At alpha=0.99: z_0.99 ~= 2.3263, phi(2.3263) ~= 0.02665
  K_GAUSSIAN = 0.02665 / 0.01 ~= 2.6652
  This is the Gaussian answer to the same question EVT answers (alpha=0.99).
  Using k=3.0 would correspond to alpha~=0.9987, not 0.99 -- wrong confidence level.

Models: RiskAware(Reactive) and RiskAware(TCN) only.
  These bracket the quality spectrum from Phase 2.

Hard constraints:
  - results/phase1/, results/phase2/, results/phase3/ are NEVER written to
  - C0 and C4 are loaded from Phase 1/2 disk results, never re-run
  - Same test set, simulator, extreme threshold, cost model as all prior phases
  - C1 and C3: static sigma -- the rolling-window loop does NOT execute
  - K_GAUSSIAN computed via scipy, not hardcoded

Outputs:
  results/phase4/azure/conditions/{condition_id}_{model}_metrics.json
  results/phase4/azure/conditions/{condition_id}_{model}_diagnostics.csv
  results/phase4/azure/summary.csv
  results/phase4/azure/all_metrics.json
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
from models.tcn import TCNModel
from evaluation.evt import fit_evt_pipeline
from evaluation.simulator import simulate
from evaluation.metrics import compute_metrics, C_COLD, C_IDLE
from evaluation.extreme import compute_extreme_threshold, EXTREME_PERCENTILE

# ---------------------------------------------------------------------------
# Gaussian CVaR at alpha=0.99 -- the scientifically correct comparison point
# CVaR_alpha[N(0,1)] = phi(Phi^{-1}(alpha)) / (1 - alpha)
# ---------------------------------------------------------------------------
ALPHA = 0.99
_z_alpha = norm.ppf(ALPHA)                        # ~2.3263
K_GAUSSIAN = norm.pdf(_z_alpha) / (1 - ALPHA)     # ~2.6652

# EVT anchor params -- same as Phase 2
EVT_THRESHOLD_PERCENTILE = 90
VOLATILITY_WINDOW = 30

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR       = os.path.join(PROJECT_ROOT, "data",    "processed", "azure")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results", "phase4",   "azure")
CONDITIONS_DIR = os.path.join(RESULTS_DIR, "conditions")
PHASE1_DIR     = os.path.join(PROJECT_ROOT, "results", "phase1",   "azure")
PHASE2_DIR     = os.path.join(PROJECT_ROOT, "results", "phase2",   "azure")

# ---------------------------------------------------------------------------
# Condition metadata
# ---------------------------------------------------------------------------
CONDITION_ORDER = [
    "C0_no_buffer",
    "C1_static_gaussian",
    "C2_dynamic_gaussian",
    "C3_static_evt",
    "C4_dynamic_evt",
]

CONDITION_META = {
    "C0_no_buffer":        {"label": "No Buffer",       "sigma": "none",    "mult": "none"},
    "C1_static_gaussian":  {"label": "Static+Gaussian", "sigma": "static",  "mult": "gaussian"},
    "C2_dynamic_gaussian": {"label": "Dynamic+Gaussian","sigma": "dynamic", "mult": "gaussian"},
    "C3_static_evt":       {"label": "Static+EVT",      "sigma": "static",  "mult": "evt"},
    "C4_dynamic_evt":      {"label": "Dynamic+EVT",     "sigma": "dynamic", "mult": "evt"},
}

# New conditions to actually run (C0 and C4 come from disk)
NEW_CONDITIONS = ["C1_static_gaussian", "C2_dynamic_gaussian", "C3_static_evt"]


# ---------------------------------------------------------------------------
# Minimal pass-through to reuse simulate() with pre-computed predictions
# ---------------------------------------------------------------------------
class _FixedPredictor:
    """Wraps a pre-computed prediction array so simulate() can call .predict()."""
    def __init__(self, preds: np.ndarray):
        self._preds = preds

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._preds


# ---------------------------------------------------------------------------
# Base model cache (identical pattern to Phase 3)
# ---------------------------------------------------------------------------
class BaseModelCache:
    """Holds a fitted base model and pre-computed training residuals."""
    def __init__(self, model_name: str, base_model, residuals: np.ndarray):
        self.model_name  = model_name
        self.base_model  = base_model
        self.residuals   = residuals
        self.sigma_train = float(np.std(residuals))
        self.z           = residuals / self.sigma_train


def train_base_models(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Train Reactive and TCN ONCE each. Returns {model_name: BaseModelCache}."""
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
        print(f"  [{name}] fit_time={time.time()-t0:.1f}s")
        preds = base.predict(train_df)
        residuals = train_df["concurrency"].values.astype(np.float64) - preds
        sigma = float(np.std(residuals))
        print(f"  [{name}] sigma_train={sigma:,.0f}")
        caches[name] = BaseModelCache(name, base, residuals)
    return caches


# ---------------------------------------------------------------------------
# Rolling sigma (C2 only -- same sequential logic as RiskAwareModel.predict())
# ---------------------------------------------------------------------------
def _rolling_sigma(base_preds: np.ndarray, lag_1: np.ndarray,
                   sigma_train: float, window: int) -> np.ndarray:
    """
    Rolling volatility estimate -- identical to RiskAwareModel.predict() inner loop.
    First `window` timesteps use sigma_train (warm-up from training data).
    """
    n = len(base_preds)
    sigma = np.full(n, sigma_train, dtype=np.float64)
    residual_history = []
    for t in range(n):
        if t > 0:
            past_residual = lag_1[t] - base_preds[t - 1]
            residual_history.append(past_residual)
            if len(residual_history) >= window:
                sigma[t] = float(np.std(residual_history[-window:])) + 1e-8
    return sigma


# ---------------------------------------------------------------------------
# Single-condition runner
# ---------------------------------------------------------------------------
def run_condition(condition_id: str, cache: BaseModelCache,
                  evt_params: dict, test_df: pd.DataFrame,
                  extreme_threshold: float) -> dict:
    """
    Compute one ablation condition for one base model.

    C1_static_gaussian:   sigma_t = sigma_train (const), mult = K_GAUSSIAN
                          Rolling-window loop does NOT execute.
    C2_dynamic_gaussian:  sigma_t = rolling(W=30),        mult = K_GAUSSIAN
    C3_static_evt:        sigma_t = sigma_train (const), mult = EVT CVaR_z
                          Rolling-window loop does NOT execute.
    """
    base_preds = cache.base_model.predict(test_df)
    n          = len(test_df)
    lag_1      = test_df["lag_1"].values.astype(np.float64)

    if condition_id == "C1_static_gaussian":
        sigma_t    = np.full(n, cache.sigma_train, dtype=np.float64)  # constant
        multiplier = K_GAUSSIAN

    elif condition_id == "C2_dynamic_gaussian":
        sigma_t    = _rolling_sigma(base_preds, lag_1, cache.sigma_train, VOLATILITY_WINDOW)
        multiplier = K_GAUSSIAN

    elif condition_id == "C3_static_evt":
        sigma_t    = np.full(n, cache.sigma_train, dtype=np.float64)  # constant
        multiplier = evt_params["cvar_z"]

    else:
        raise ValueError(f"Unknown condition: {condition_id}")

    buffer_t    = sigma_t * multiplier
    final_preds = base_preds + buffer_t

    results_df = simulate(
        _FixedPredictor(final_preds), test_df, extreme_threshold,
        c_cold=C_COLD, c_idle=C_IDLE,
    )
    metrics = compute_metrics(results_df)
    metrics["condition_id"] = condition_id
    metrics["model"]        = cache.model_name

    diag = {
        "base_prediction": base_preds,
        "sigma_t":         sigma_t,
        "buffer_t":        buffer_t,
        "cvar_col":        np.full(n, multiplier, dtype=np.float64),
    }

    return {"metrics": metrics, "results_df": results_df, "diag": diag,
            "evt_params": evt_params if condition_id == "C3_static_evt" else None}


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_condition(condition_id: str, model_name: str, run: dict) -> None:
    """Save metrics JSON + diagnostics CSV for one (condition, model) pair."""
    stem = f"{condition_id}_{model_name}"
    diag = run["diag"]

    enriched = run["results_df"].copy()
    enriched["base_prediction"] = diag["base_prediction"]
    enriched["sigma_t"]         = diag["sigma_t"]
    enriched["buffer_t"]        = diag["buffer_t"]
    enriched["cvar_col"]        = diag["cvar_col"]
    enriched.to_csv(
        os.path.join(CONDITIONS_DIR, f"{stem}_diagnostics.csv"), index=False
    )

    with open(os.path.join(CONDITIONS_DIR, f"{stem}_metrics.json"), "w") as f:
        json.dump(run["metrics"], f, indent=2)


def build_summary_row(condition_id: str, model_name: str, metrics: dict) -> dict:
    meta = CONDITION_META[condition_id]
    return {
        "condition_id":      condition_id,
        "condition_label":   meta["label"],
        "model":             model_name,
        "sigma_type":        meta["sigma"],
        "multiplier_type":   meta["mult"],
        "request_sla":       metrics["request_sla"],
        "extreme_sla":       metrics["extreme_sla"],
        "total_cost":        metrics["total_cost"],
        "cold_cost":         metrics["cold_cost"],
        "idle_cost":         metrics["idle_cost"],
        "total_cold_starts": metrics["total_cold_starts"],
        "cold_start_rate":   metrics["cold_start_rate"],
    }


# ---------------------------------------------------------------------------
# Load Phase 1 / Phase 2 results (C0 and C4 come from disk)
# ---------------------------------------------------------------------------
def load_phase1_metrics() -> dict:
    """Load Reactive and TCN metrics from Phase 1 results."""
    with open(os.path.join(PHASE1_DIR, "metrics.json")) as f:
        data = json.load(f)
    return {"Reactive": data["Reactive"], "TCN": data["TCN"]}


def load_phase2_metrics() -> dict:
    """Load RiskAware(Reactive) and RiskAware(TCN) metrics from Phase 2 results."""
    with open(os.path.join(PHASE2_DIR, "metrics.json")) as f:
        data = json.load(f)
    return {
        "Reactive": data["RiskAware(Reactive)"],
        "TCN":      data["RiskAware(TCN)"],
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    print("\n[1/5] Loading Azure data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["timestamp"])
    val   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"),   parse_dates=["timestamp"])
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),  parse_dates=["timestamp"])
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "="*80)
    print("PHASE 4 ABLATION RESULTS")
    print("="*80)

    # Full table
    for model_name in ["Reactive", "TCN"]:
        print(f"\n  Model: {model_name}")
        sub = df[df["model"] == model_name].copy()
        sub = sub.set_index("condition_id").loc[CONDITION_ORDER].reset_index()
        print(sub[["condition_id", "condition_label",
                   "request_sla", "extreme_sla",
                   "total_cost", "cold_cost", "idle_cost",
                   "total_cold_starts"]].to_string(index=False))

    # 2x2 heatmap summary
    for model_name in ["Reactive", "TCN"]:
        print(f"\n  2x2 Request SLA heatmap ({model_name}):")
        print(f"    {'':22s} Gaussian       EVT")
        for sigma_type in ["static", "dynamic"]:
            row = df[(df["model"] == model_name) & (df["sigma_type"] == sigma_type)]
            gauss_sla = row[row["multiplier_type"] == "gaussian"]["request_sla"]
            evt_sla   = row[row["multiplier_type"] == "evt"]["request_sla"]
            g = f"{gauss_sla.values[0]:.6f}" if len(gauss_sla) else "  N/A   "
            e = f"{evt_sla.values[0]:.6f}"   if len(evt_sla)   else "  N/A   "
            print(f"    {sigma_type:22s} {g}      {e}")

    # SLA gain from each step
    print("\n  SLA gain summary:")
    for model_name in ["Reactive", "TCN"]:
        sub = df[df["model"] == model_name].set_index("condition_id")["request_sla"]
        c0 = sub["C0_no_buffer"]
        c1 = sub["C1_static_gaussian"]
        c2 = sub["C2_dynamic_gaussian"]
        c3 = sub["C3_static_evt"]
        c4 = sub["C4_dynamic_evt"]
        print(f"  [{model_name}]"
              f"  C0->C1: {(c1-c0)*100:+.3f}pp"
              f"  C1->C2: {(c2-c1)*100:+.3f}pp"
              f"  C1->C3: {(c3-c1)*100:+.3f}pp"
              f"  C2->C4: {(c4-c2)*100:+.3f}pp"
              f"  C3->C4: {(c4-c3)*100:+.3f}pp")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("="*70)
    print("Phase 4: EVT-CVaR Component Ablation Study - Azure Dataset")
    print("="*70)
    print(f"  K_GAUSSIAN = {K_GAUSSIAN:.6f}  (Gaussian CVaR at alpha={ALPHA})")
    print(f"  EVT params: alpha={ALPHA}, W={VOLATILITY_WINDOW}, P{EVT_THRESHOLD_PERCENTILE}")
    print(f"  New conditions: {', '.join(NEW_CONDITIONS)}")
    print(f"  C0 loaded from: {PHASE1_DIR}")
    print(f"  C4 loaded from: {PHASE2_DIR}")
    print(f"  WARNING: Phase 1/2/3 results are NEVER modified.\n")

    train, val, test = load_data()

    extreme_threshold = compute_extreme_threshold(train, percentile=EXTREME_PERCENTILE)
    print(f"\n  Extreme threshold (P{EXTREME_PERCENTILE} of train): {extreme_threshold:,.0f}")

    caches = train_base_models(train, val)

    # Fit EVT ONCE per model on training residuals (anchor params)
    print("\n[3/5] Fitting EVT (anchor params: alpha=0.99, W=30, P90)...")
    evt_params = {}
    for model_name, cache in caches.items():
        print(f"\n  [{model_name}] EVT fit...")
        evt_params[model_name] = fit_evt_pipeline(
            cache.z,
            threshold_percentile=EVT_THRESHOLD_PERCENTILE,
            alpha=ALPHA,
        )
        print(f"  [{model_name}] CVaR_z = {evt_params[model_name]['cvar_z']:.4f}"
              f"  xi = {evt_params[model_name]['xi']:.4f}"
              f"  (ratio vs K_GAUSSIAN: {evt_params[model_name]['cvar_z']/K_GAUSSIAN:.2f}x)")

    os.makedirs(CONDITIONS_DIR, exist_ok=True)

    print("\n[4/5] Running new ablation conditions (C1, C2, C3)...")
    new_runs = {}  # (condition_id, model_name) -> run dict

    for condition_id in NEW_CONDITIONS:
        print(f"\n  --- {condition_id} ---")
        for model_name, cache in caches.items():
            t0  = time.time()
            run = run_condition(condition_id, cache, evt_params[model_name],
                                test, extreme_threshold)
            m   = run["metrics"]
            print(f"  [{model_name}]  Req SLA={m['request_sla']:.6f}  "
                  f"Ext SLA={m['extreme_sla']:.6f}  "
                  f"Cost={m['total_cost']/1e6:.1f}M  "
                  f"Cold={m['total_cold_starts']:,}  "
                  f"({time.time()-t0:.1f}s)")
            save_condition(condition_id, model_name, run)
            new_runs[(condition_id, model_name)] = run

    print("\n[5/5] Building summary tables...")
    c0_metrics = load_phase1_metrics()
    c4_metrics = load_phase2_metrics()

    all_rows = []
    all_metrics_dict = {}

    for cid, metrics_map in [("C0_no_buffer", c0_metrics), ("C4_dynamic_evt", c4_metrics)]:
        for model_name in ["Reactive", "TCN"]:
            all_rows.append(build_summary_row(cid, model_name, metrics_map[model_name]))
            all_metrics_dict[f"{cid}_{model_name}"] = metrics_map[model_name]

    for condition_id in NEW_CONDITIONS:
        for model_name in ["Reactive", "TCN"]:
            run = new_runs[(condition_id, model_name)]
            all_rows.append(build_summary_row(condition_id, model_name, run["metrics"]))
            all_metrics_dict[f"{condition_id}_{model_name}"] = run["metrics"]

    # Sort by defined condition order
    df = pd.DataFrame(all_rows)
    df["_order"] = df["condition_id"].map({c: i for i, c in enumerate(CONDITION_ORDER)})
    df = df.sort_values(["_order", "model"]).drop(columns="_order").reset_index(drop=True)
    df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    print(f"  Saved: summary.csv ({len(df)} rows)")

    with open(os.path.join(RESULTS_DIR, "all_metrics.json"), "w") as f:
        json.dump(all_metrics_dict, f, indent=2)
    print("  Saved: all_metrics.json")

    print_summary(df)

    print("\n[DONE] Phase 4 ablation complete.")
    print(f"  Results: {RESULTS_DIR}")
    return df


if __name__ == "__main__":
    main()
