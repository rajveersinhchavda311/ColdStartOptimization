"""
Documentation Number Regression Checker
========================================

Re-verifies every load-bearing number in the documentation against the
frozen source JSON/CSV files in results/. Run after ANY documentation edit.

Strategy:
    For each documented table row, compute the canonical value from the
    source file, format it at the documented precision, and assert the
    formatted string appears in the doc. Plus a stale-string sweep for
    known-bad values that must never reappear.

Exit code 0 = all checks pass; non-zero = at least one failure (printed).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K_GAUSSIAN = norm.pdf(norm.ppf(0.99)) / 0.01

FAILURES = []
CHECKS = [0]


def read(relpath):
    with open(os.path.join(PROJECT_ROOT, relpath), encoding="utf-8") as f:
        return f.read()


def check(doc_text, needle, label):
    CHECKS[0] += 1
    if needle not in doc_text:
        FAILURES.append(f"[MISSING] {label}: expected substring {needle!r}")


def check_absent(doc_relpath, needle, label):
    CHECKS[0] += 1
    text = read(doc_relpath)
    if needle in text:
        FAILURES.append(f"[STALE] {label}: forbidden substring {needle!r} "
                        f"still present in {doc_relpath}")


def jload(relpath):
    with open(os.path.join(PROJECT_ROOT, relpath)) as f:
        return json.load(f)


def fmt_m(x):
    """Cost in millions, 1 decimal, with thousands separators (doc style)."""
    return f"{x / 1e6:,.1f}M"


def main():
    paper = read("docs/paper_context.md")
    readme = read("README.md")
    gen = read("docs/phase5/generalization_study.md")
    sens = read("docs/phase3/sensitivity_analysis.md")
    abl = read("docs/phase4/ablation_study.md")

    # ---- Phase 1 Azure table (paper_context, 6 dp SLA, 0.1M cost, exact colds)
    m1 = jload("results/phase1/azure/metrics.json")
    for model, mm in m1.items():
        row = (f"| {mm['request_sla']:.6f} | {mm['extreme_sla']:.6f} "
               f"| {fmt_m(mm['total_cost'])} | {mm['total_cold_starts']:,} |")
        check(paper, f"{mm['request_sla']:.6f}", f"P1 {model} request_sla")
        check(paper, f"{mm['extreme_sla']:.6f}", f"P1 {model} extreme_sla")
        check(paper, fmt_m(mm["total_cost"]), f"P1 {model} cost")
        check(paper, f"{mm['total_cold_starts']:,}", f"P1 {model} cold starts")

    # ---- Phase 2 Azure table
    m2 = jload("results/phase2/azure/metrics.json")
    for model, mm in m2.items():
        check(paper, f"{mm['request_sla']:.6f}", f"P2 {model} request_sla")
        check(paper, f"{mm['extreme_sla']:.6f}", f"P2 {model} extreme_sla")
        check(paper, fmt_m(mm["total_cost"]), f"P2 {model} cost")
        check(paper, f"{mm['total_cold_starts']:,}", f"P2 {model} cold starts")

    # ---- Phase 2 reductions (1 dp percentage)
    for base in ["Reactive", "Forecast_Only", "Seasonal_Naive",
                 "Linear_Seasonal", "TCN"]:
        red = (1 - m2[f"RiskAware({base})"]["total_cold_starts"]
               / m1[base]["total_cold_starts"]) * 100
        check(paper, f"−{red:.1f}%", f"P2 {base} reduction −{red:.1f}%")

    # ---- EVT parameters Azure (xi 4dp, CVaR 3dp, ratio 2dp)
    evt_az = jload("results/phase2/azure/evt_parameters.json")
    for model, p in evt_az.items():
        sign = "+" if p["xi"] >= 0 else "−"
        check(paper, f"{sign}{abs(p['xi']):.4f}", f"EVT Azure {model} xi")
        check(paper, f"{p['cvar_z']:.3f}", f"EVT Azure {model} cvar")
        check(paper, f"{p['cvar_z'] / K_GAUSSIAN:.2f}×", f"EVT Azure {model} ratio")

    # ---- Cross-dataset xi table (paper_context + generalization_study)
    xi = pd.read_csv(os.path.join(PROJECT_ROOT, "results/phase5/evt_xi_summary.csv"))
    for _, r in xi.iterrows():
        for doc, name in [(paper, "paper_context"), (gen, "generalization_study")]:
            for col, prec, suffix in [("Reactive_xi", 4, ""), ("TCN_xi", 4, ""),
                                      ("Reactive_ratio", 2, "×"), ("TCN_ratio", 2, "×")]:
                v = r[col]
                if suffix == "×":
                    s = f"{v:.2f}×"
                else:
                    s = ("+" if v >= 0 else "−") + f"{abs(v):.4f}"
                check(doc, s, f"xi-table {name} {r['Dataset']} {col}")

    # ---- Phase 4 table + 2x2 heatmap (6 dp)
    p4 = jload("results/phase4/azure/all_metrics.json")
    for cond, mm in p4.items():
        check(paper, f"{mm['request_sla']:.6f}", f"P4 {cond} request_sla (heatmap)")
    # incremental deltas (pp, 2dp where doc uses 2; doc uses e.g. +1.44 pp)
    for model in ["Reactive", "TCN"]:
        c0 = p4[f"C0_no_buffer_{model}"]["request_sla"]
        c1 = p4[f"C1_static_gaussian_{model}"]["request_sla"]
        check(paper, f"+{(c1 - c0) * 100:.2f} pp", f"P4 C0→C1 delta {model}")

    # ---- Phase 5 Huawei tables (4 dp SLA in docs)
    m1h = jload("results/phase1/huawei/combined/metrics.json")
    m2h = jload("results/phase2/huawei/combined/metrics.json")
    for model, mm in m1h.items():
        check(gen, f"{mm['request_sla']:.4f}", f"P1 HW {model} request_sla")
        check(gen, f"{mm['extreme_sla']:.4f}", f"P1 HW {model} extreme_sla")
        check(gen, f"{mm['total_cold_starts']:,}", f"P1 HW {model} colds")
    for model, mm in m2h.items():
        check(gen, f"{mm['request_sla']:.4f}", f"P2 HW {model} request_sla")
        check(gen, f"{mm['extreme_sla']:.4f}", f"P2 HW {model} extreme_sla")
        check(gen, f"{mm['total_cold_starts']:,}", f"P2 HW {model} colds")

    # ---- EVT parameters Huawei combined (gen study Table 1: 4dp xi, 3dp cvar)
    evt_hw = jload("results/phase2/huawei/combined/evt_parameters.json")
    for model, p in evt_hw.items():
        sign = "+" if p["xi"] >= 0 else "−"
        check(gen, f"{sign}{abs(p['xi']):.4f}", f"EVT HW {model} xi")
        check(gen, f"{p['cvar_z']:.3f}", f"EVT HW {model} cvar")

    # ---- Audit count claims
    audits = {
        "results/phase1/azure/audit_results.json": ("74/74", paper),
        "results/phase3/azure/audit_results.json": ("148/148", paper),
        "results/phase4/azure/audit_results.json": ("63/63", paper),
        "results/phase1/huawei/combined/audit_results.json": ("126/126", paper),
        "results/phase2/huawei/combined/audit_results.json": ("121/121", paper),
    }
    for path, (claim, doc) in audits.items():
        a = jload(path)
        total = a.get("total_checks", a.get("total"))
        passed = a["passed"]
        CHECKS[0] += 1
        if f"{passed}/{total}" != claim:
            FAILURES.append(f"[AUDIT] {path}: JSON says {passed}/{total}, "
                            f"docs claim {claim}")
        check(doc, claim, f"audit claim {claim}")
    a2 = jload("results/phase2/azure/audit_results.json")
    CHECKS[0] += 1
    if not (a2["passed"] == 106 and a2["failed"] == 0
            and a2["overall"] == "PASS"
            and a2.get("expected_failures_noted") == 2):
        FAILURES.append("[AUDIT] phase2 azure audit JSON does not match "
                        "106/106 PASS (2 XFAIL) convention")

    # ---- Key scalar facts
    test = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/azure/test.csv"))
    train = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/azure/train.csv"))
    p99 = np.percentile(train["concurrency"], 99)
    CHECKS[0] += 1
    if int(test["concurrency"].max()) != 866343:
        FAILURES.append("[DATA] Azure test max changed?!")
    check(paper, "866,343", "Azure test max in paper_context")
    check(paper, "1.10×", "Azure test/P99 ratio in paper_context")
    check(gen, "1.10×", "Azure test/P99 ratio in generalization_study")

    # ---- Stale-string sweep (must never reappear)
    stale = {
        "docs/paper_context.md": ["104/106", "1.6× the training P99", "depth 6",
                                  "kernel 2", "−0.2289", "1.99×",
                                  "over-provisions during demand troughs",
                                  "strictly weakly dominates"],
        "README.md": ["104/106", "1.6× on Azure", "generate_graphs.py",
                      "Best result: TCN"],
        "docs/phase5/generalization_study.md": ["1.6×", "−0.2289", "1.99×",
                                                "0.21 for Azure", "+0.028",
                                                "pass_rate", "6/6"],
        "docs/phase3/sensitivity_analysis.md": ["strictly dominates"],
        "docs/phase4/ablation_study.md": ["Phase 6"],
        "docs/repository_structure.md": ["104/106", "depth=6"],
        "docs/preprocessing/preprocessing_guide.md": ["1.6×", "1,258,768 (~1.6"],
        "scripts/generate_phase5_graphs.py": ["standard_normal(1000)",
                                              "Generate representative data"],
    }
    for path, needles in stale.items():
        for n in needles:
            check_absent(path, n, f"stale sweep {path}")

    # ---- Report
    print(f"Checks run: {CHECKS[0]}")
    if FAILURES:
        print(f"FAILURES: {len(FAILURES)}")
        for f in FAILURES:
            print("  " + f)
        sys.exit(1)
    print("ALL DOC-NUMBER CHECKS PASS")


if __name__ == "__main__":
    main()
