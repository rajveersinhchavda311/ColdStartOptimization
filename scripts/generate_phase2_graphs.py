"""
Phase 2 Graph Generation — Risk-Aware EVT+CVaR Visualizations
================================================================

Generates publication-quality graphs comparing Phase 1 baselines
against Phase 2 risk-aware variants for all 5 wrapped models.

Graphs:
    1. Cost comparison — Phase 1 vs Phase 2 (grouped bar chart, 5 pairs)
    2. SLA comparison — Request SLA and Extreme SLA side by side (5 pairs)
    3. Dynamic buffer visualization — sigma_t and buffer_t over time (TCN)
    4. Buffer distribution — 2-row grid showing non-constant buffer per model
    5. Phase 1 vs Phase 2 overlay — predictions vs actual demand (TCN)

Output: graphs/phase2/azure/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase2", "azure")
PHASE1_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase1", "azure")
GRAPHS_DIR = os.path.join(PROJECT_ROOT, "graphs", "phase2", "azure")

# Colors: Phase 1 base models (warm), Phase 2 risk-aware (cool variants)
P1_COLORS = {
    "Reactive":        "#e15759",
    "Forecast_Only":   "#f28e2b",
    "Seasonal_Naive":  "#edc948",
    "Linear_Seasonal": "#76b7b2",
    "TCN":             "#ff9da7",
}
P2_COLORS = {
    "RiskAware(Reactive)":        "#4e79a7",
    "RiskAware(Forecast_Only)":   "#59a14f",
    "RiskAware(Seasonal_Naive)":  "#b07aa1",
    "RiskAware(Linear_Seasonal)": "#499894",
    "RiskAware(TCN)":             "#9c755f",
}

# Ordered pairs: (phase1_name, phase2_name) — must be kept in sync
PAIRS = [
    ("Reactive",        "RiskAware(Reactive)"),
    ("Forecast_Only",   "RiskAware(Forecast_Only)"),
    ("Seasonal_Naive",  "RiskAware(Seasonal_Naive)"),
    ("Linear_Seasonal", "RiskAware(Linear_Seasonal)"),
    ("TCN",             "RiskAware(TCN)"),
]

P2_MODELS = [r for _, r in PAIRS]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_data():
    """Load Phase 1 and Phase 2 metrics and all available diagnostics."""
    with open(os.path.join(PHASE1_RESULTS_DIR, "metrics.json"), "r") as f:
        p1_metrics = json.load(f)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "r") as f:
        p2_metrics = json.load(f)

    p2_diagnostics = {}
    for name in P2_MODELS:
        diag_path = os.path.join(RESULTS_DIR, f"{name}_diagnostics.csv")
        if os.path.exists(diag_path):
            p2_diagnostics[name] = pd.read_csv(diag_path,
                                                parse_dates=["timestamp"])
        else:
            print(f"  WARNING: diagnostics not found for {name}")

    return p1_metrics, p2_metrics, p2_diagnostics


def plot_cost_comparison(p1_metrics, p2_metrics):
    """Grouped stacked bar: Phase 1 vs Phase 2 cost decomposition for all 5 pairs."""
    fig, ax = plt.subplots(figsize=(16, 7))

    x = np.arange(len(PAIRS))
    width = 0.35
    labels = [p for p, _ in PAIRS]

    cold_p1 = [p1_metrics[p]["cold_cost"] for p, _ in PAIRS]
    idle_p1 = [p1_metrics[p]["idle_cost"] for p, _ in PAIRS]
    cold_p2 = [p2_metrics[r]["cold_cost"] for _, r in PAIRS]
    idle_p2 = [p2_metrics[r]["idle_cost"] for _, r in PAIRS]

    # Phase 1 bars (left)
    ax.bar(x - width/2, cold_p1, width, label="P1 Cold Cost",
           color="#e15759", alpha=0.85)
    ax.bar(x - width/2, idle_p1, width, bottom=cold_p1,
           label="P1 Idle Cost", color="#e15759", alpha=0.3)

    # Phase 2 bars (right)
    ax.bar(x + width/2, cold_p2, width, label="P2 Cold Cost",
           color="#4e79a7", alpha=0.85)
    ax.bar(x + width/2, idle_p2, width, bottom=cold_p2,
           label="P2 Idle Cost", color="#4e79a7", alpha=0.3)

    # Total cost labels
    for i, (p, r) in enumerate(PAIRS):
        t1 = p1_metrics[p]["total_cost"]
        t2 = p2_metrics[r]["total_cost"]
        ax.text(i - width/2, t1 * 1.01, f"{t1/1e6:.0f}M",
                ha="center", va="bottom", fontsize=7.5, color="#c0392b",
                fontweight="bold")
        ax.text(i + width/2, t2 * 1.01, f"{t2/1e6:.0f}M",
                ha="center", va="bottom", fontsize=7.5, color="#2c6fad",
                fontweight="bold")

    ax.set_xlabel("Base Model")
    ax.set_ylabel("Cost")
    ax.set_title("Phase 1 vs Phase 2: Cost Decomposition\n"
                 "(c_cold=10, c_idle=1 — experimental assumption)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.legend()
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "cost_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_sla_comparison(p1_metrics, p2_metrics):
    """Side-by-side panels: Request SLA and Extreme SLA for all 5 pairs."""
    labels = [p for p, _ in PAIRS]
    x = np.arange(len(PAIRS))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric_key, title in [
        (axes[0], "request_sla", "Request SLA"),
        (axes[1], "extreme_sla", "Extreme Event SLA"),
    ]:
        sla_p1 = [p1_metrics[p][metric_key] for p, _ in PAIRS]
        sla_p2 = [p2_metrics[r][metric_key] for _, r in PAIRS]

        bars1 = ax.bar(x - width/2, sla_p1, width, label="Phase 1",
                       color="#e15759", alpha=0.85)
        bars2 = ax.bar(x + width/2, sla_p2, width, label="Phase 2",
                       color="#4e79a7", alpha=0.85)

        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10, ha="right")
        ax.legend()
        floor = min(min(sla_p1), min(sla_p2))
        ax.set_ylim(max(0, floor - 0.03), 1.003)

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.001,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("Phase 1 vs Phase 2: SLA Comparison\n"
                 "(Extreme = demand > P99 of training data)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(GRAPHS_DIR, "sla_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_dynamic_buffer(p2_diagnostics):
    """3-panel time series for RiskAware(TCN): demand/predictions, sigma_t, buffer_t."""
    name = "RiskAware(TCN)"
    if name not in p2_diagnostics:
        print(f"  Skipping dynamic buffer plot: {name} diagnostics not found")
        return

    diag = p2_diagnostics[name]
    window = min(2000, len(diag))
    t = np.arange(window)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax = axes[0]
    ax.plot(t, diag["actual_demand"].values[:window],
            color="black", alpha=0.5, linewidth=0.5, label="Actual Demand")
    ax.plot(t, diag["base_prediction"].values[:window],
            color="#e15759", alpha=0.75, linewidth=0.5, label="Base (TCN)")
    ax.plot(t, diag["predicted"].values[:window],
            color="#4e79a7", alpha=0.75, linewidth=0.5,
            label="Risk-Aware (TCN)")
    ax.set_ylabel("Concurrency")
    ax.set_title(f"Dynamic Risk Buffer — {name}")
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K"))

    ax = axes[1]
    ax.plot(t, diag["sigma_t"].values[:window], color="#59a14f", linewidth=0.8)
    ax.axhline(y=diag["sigma_t"].values[0], color="gray",
               linestyle="--", alpha=0.5, label="sigma_train (warm-up value)")
    ax.set_ylabel("sigma_t")
    ax.set_title("Rolling Volatility (std of last 30 residuals)")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[2]
    ax.fill_between(t, 0, diag["buffer_t"].values[:window],
                    color="#f28e2b", alpha=0.45)
    ax.plot(t, diag["buffer_t"].values[:window],
            color="#f28e2b", linewidth=0.6)
    ax.set_ylabel("Buffer")
    ax.set_xlabel("Timestep")
    ax.set_title("Dynamic CVaR Buffer  (buffer_t = sigma_t × CVaR_z)")

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "dynamic_buffer.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_buffer_distribution(p2_diagnostics):
    """2-row grid of buffer histograms — one panel per Phase 2 model."""
    available = [n for n in P2_MODELS if n in p2_diagnostics]
    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    colors = [P2_COLORS.get(nm, "#999999") for nm in available]

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows))
    axes_flat = axes.flatten() if nrows > 1 else list(axes)

    for i, (name, color) in enumerate(zip(available, colors)):
        ax = axes_flat[i]
        buf = p2_diagnostics[name]["buffer_t"].values
        ax.hist(buf, bins=50, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(x=np.mean(buf), color="red", linestyle="--", linewidth=1.3,
                   label=f"Mean: {np.mean(buf):,.0f}")
        # Short model label (strip "RiskAware(" prefix)
        short = name.replace("RiskAware(", "RA(")
        ax.set_title(short, fontsize=10, fontweight="bold")
        ax.set_xlabel("Buffer Value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Distribution of Dynamic Buffer Values\n"
                 "(non-constant distribution confirms adaptive behaviour)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(GRAPHS_DIR, "buffer_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_prediction_overlay(p2_diagnostics):
    """Overlay base vs risk-aware prediction vs actual for RiskAware(TCN)."""
    name = "RiskAware(TCN)"
    if name not in p2_diagnostics:
        print(f"  Skipping overlay plot: {name} diagnostics not found")
        return

    diag = p2_diagnostics[name]
    n = len(diag)
    chunk = 500
    # Find the window with highest buffer variance (most informative)
    best_start, best_var = 0, 0
    for s in range(0, n - chunk, chunk // 2):
        v = np.var(diag["buffer_t"].values[s:s + chunk])
        if v > best_var:
            best_var = v
            best_start = s
    end = best_start + chunk
    sl = slice(best_start, end)

    fig, ax = plt.subplots(figsize=(14, 6))
    t = np.arange(chunk)

    ax.plot(t, diag["actual_demand"].values[sl],
            color="black", alpha=0.6, linewidth=0.8, label="Actual Demand")
    ax.plot(t, diag["base_prediction"].values[sl],
            color="#e15759", alpha=0.75, linewidth=0.8, label="Base (TCN)")
    ax.plot(t, diag["predicted"].values[sl],
            color="#4e79a7", alpha=0.75, linewidth=0.8,
            label="Risk-Aware (TCN)")
    ax.fill_between(t,
                    diag["base_prediction"].values[sl],
                    diag["predicted"].values[sl],
                    alpha=0.2, color="#4e79a7", label="EVT-CVaR Buffer")

    ax.set_xlabel(f"Timestep (test set offset {best_start}–{end})")
    ax.set_ylabel("Concurrency")
    ax.set_title("Phase 1 TCN vs Phase 2 RiskAware(TCN) — "
                 "Most Volatile Test Window")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K"))

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "prediction_overlay.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("Phase 2: Graph Generation -- Azure Dataset")
    print("=" * 60)

    os.makedirs(GRAPHS_DIR, exist_ok=True)

    print("\nLoading results...")
    p1_metrics, p2_metrics, p2_diagnostics = load_data()
    print(f"  Phase 1 models: {list(p1_metrics.keys())}")
    print(f"  Phase 2 models: {list(p2_metrics.keys())}")
    print(f"  Diagnostics loaded: {list(p2_diagnostics.keys())}")

    print("\nGenerating graphs...")
    plot_cost_comparison(p1_metrics, p2_metrics)
    plot_sla_comparison(p1_metrics, p2_metrics)
    plot_dynamic_buffer(p2_diagnostics)
    plot_buffer_distribution(p2_diagnostics)
    plot_prediction_overlay(p2_diagnostics)

    print(f"\nAll graphs saved to: {GRAPHS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
