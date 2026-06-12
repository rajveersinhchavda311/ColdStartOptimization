"""
Parametric Bootstrap Confidence Intervals for GPD Parameters
=============================================================

Question answered (reviewer-facing): "Your xi point estimates — especially
the xi > 0.5 fits on small-traffic regions — carry how much sampling
uncertainty? Does the headline claim (CVaR_z > K_GAUSSIAN everywhere)
survive that uncertainty?"

Method (parametric bootstrap, conditional on the threshold):
    For each stored fit (xi_hat, beta_hat, n_exceedances) in
    evt_parameters.json, repeat B times:
        1. Sample n_exceedances points from GPD(xi_hat, beta_hat).
        2. Refit by MLE (genpareto.fit, floc=0) — same estimator as the
           original pipeline.
        3. Recompute VaR/CVaR with the ORIGINAL threshold u and exceedance
           probability (the bootstrap is conditional on threshold choice;
           threshold and sigma_train uncertainty are not modeled).
    Report percentile CIs (2.5%, 97.5%) for xi and CVaR_z, and whether
    CVaR_z > K_GAUSSIAN across the entire CI.

This script is read-only over results/: it reads stored parameters and
writes one NEW file. Seeded for reproducibility.

Output (additive only):
    results/analysis/evt_bootstrap_ci.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import genpareto, norm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.evt import compute_var, compute_cvar

B = 2000
SEED = 42
ALPHA = 0.99
K_GAUSSIAN = norm.pdf(norm.ppf(ALPHA)) / (1 - ALPHA)

SOURCES = [
    ("Azure", "results/phase2/azure/evt_parameters.json"),
    ("Huawei Combined", "results/phase2/huawei/combined/evt_parameters.json"),
    ("Huawei R1", "results/phase2/huawei/R1/evt_parameters.json"),
    ("Huawei R2", "results/phase2/huawei/R2/evt_parameters.json"),
    ("Huawei R3", "results/phase2/huawei/R3/evt_parameters.json"),
    ("Huawei R4", "results/phase2/huawei/R4/evt_parameters.json"),
    ("Huawei R5", "results/phase2/huawei/R5/evt_parameters.json"),
]


def bootstrap_fit(params: dict, rng: np.random.Generator) -> dict:
    xi_hat, beta_hat = params["xi"], params["beta"]
    u, exc_prob = params["threshold_u"], params["exceedance_prob"]
    n = params["n_exceedances"]

    xis = np.empty(B)
    cvars = np.full(B, np.nan)
    for b in range(B):
        sample = genpareto.rvs(c=xi_hat, scale=beta_hat, size=n, random_state=rng)
        c, _loc, scale = genpareto.fit(sample, floc=0)
        xis[b] = c
        if c < 1.0:  # CVaR finite only for xi < 1
            var_b = compute_var(c, scale, u, exc_prob, ALPHA)
            cvars[b] = compute_cvar(c, scale, var_b, u)

    cvars_ok = cvars[np.isfinite(cvars)]
    return {
        "xi_hat": xi_hat,
        "xi_lo": float(np.percentile(xis, 2.5)),
        "xi_hi": float(np.percentile(xis, 97.5)),
        "cvar_hat": params["cvar_z"],
        "cvar_lo": float(np.percentile(cvars_ok, 2.5)),
        "cvar_hi": float(np.percentile(cvars_ok, 97.5)),
        "n_exceedances": n,
        "n_infinite_cvar_resamples": int(B - len(cvars_ok)),
        "cvar_gt_gaussian_full_ci": bool(np.percentile(cvars_ok, 2.5) > K_GAUSSIAN),
    }


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for ds_name, rel_path in SOURCES:
        with open(os.path.join(PROJECT_ROOT, rel_path)) as f:
            all_params = json.load(f)
        for model, params in all_params.items():
            r = bootstrap_fit(params, rng)
            r.update({"dataset": ds_name, "model": model})
            rows.append(r)
            print(f"{ds_name:16s} {model:28s} "
                  f"xi={r['xi_hat']:+.4f} [{r['xi_lo']:+.4f}, {r['xi_hi']:+.4f}]  "
                  f"CVaR={r['cvar_hat']:.3f} [{r['cvar_lo']:.3f}, {r['cvar_hi']:.3f}]  "
                  f">K_G across CI: {r['cvar_gt_gaussian_full_ci']}")

    df = pd.DataFrame(rows)[[
        "dataset", "model", "n_exceedances",
        "xi_hat", "xi_lo", "xi_hi",
        "cvar_hat", "cvar_lo", "cvar_hi",
        "n_infinite_cvar_resamples", "cvar_gt_gaussian_full_ci",
    ]]
    out_dir = os.path.join(PROJECT_ROOT, "results", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "evt_bootstrap_ci.csv")
    df.to_csv(out, index=False)

    n_robust = int(df["cvar_gt_gaussian_full_ci"].sum())
    print(f"\nSaved {out}")
    print(f"CVaR_z > K_GAUSSIAN ({K_GAUSSIAN:.4f}) across the FULL 95% CI: "
          f"{n_robust}/{len(df)} fits")
    if n_robust < len(df):
        weak = df[~df["cvar_gt_gaussian_full_ci"]][["dataset", "model", "cvar_lo"]]
        print("Fits where the CI lower bound dips below K_GAUSSIAN "
              "(claim holds at point estimate only):")
        print(weak.to_string(index=False))


if __name__ == "__main__":
    main()
