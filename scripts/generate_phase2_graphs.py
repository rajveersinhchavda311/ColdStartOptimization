"""
Phase 2 Graph Generation — Risk-Aware EVT+CVaR Visualizations
================================================================

Generates publication-quality graphs comparing Phase 1 baselines
against Phase 2 risk-aware variants.

Graphs:
    1. Cost comparison — Phase 1 vs Phase 2 (grouped bar chart)
    2. SLA comparison — Request SLA and Extreme SLA side by side
    3. Dynamic buffer visualization — sigma_t and buffer_t over time
    4. Buffer distribution — histogram showing buffer is non-constant
    5. Phase 1 vs Phase 2 overlay — predictions vs actual demand

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

# Publication style
COLORS = {
    "Reactive": "#4e79a7",
    "Forecast_Only": "#f28e2b",
    "TCN": "#e15759",
    "RiskAware(Reactive)": "#76b7b2",
    "RiskAware(Forecast_Only)": "#59a14f",
    "RiskAware(TCN)": "#9c755f",
    "Static_P90": "#bab0ac",
}

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_data():
    """Load Phase 1 and Phase 2 metrics and diagnostics."""
    # Phase 1 metrics
    p1_path = os.path.join(PHASE1_RESULTS_DIR, "metrics.json")
    with open(p1_path, "r") as f:
        p1_metrics = json.load(f)

    # Phase 2 metrics
    p2_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(p2_path, "r") as f:
        p2_metrics = json.load(f)

    # Phase 2 diagnostics
    p2_diagnostics = {}
    for name in ["RiskAware(Reactive)", "RiskAware(Forecast_Only)",
                  "RiskAware(TCN)"]:
        diag_path = os.path.join(RESULTS_DIR, f"{name}_diagnostics.csv")
        if os.path.exists(diag_path):
            p2_diagnostics[name] = pd.read_csv(diag_path,
                                                parse_dates=["timestamp"])

    return p1_metrics, p2_metrics, p2_diagnostics


def plot_cost_comparison(p1_metrics, p2_metrics):
    """Plot Phase 1 vs Phase 2 cost comparison (grouped bar chart)."""
    fig, ax = plt.subplots(figsize=(14, 7))

    pairs = [
        ("Reactive", "RiskAware(Reactive)"),
        ("Forecast_Only", "RiskAware(Forecast_Only)"),
        ("TCN", "RiskAware(TCN)"),
    ]

    x = np.arange(len(pairs))
    width = 0.35

    cold_p1 = [p1_metrics[p][  "cold_cost"] for p, _ in pairs]
    idle_p1 = [p1_metrics[p]["idle_cost"] for p, _ in pairs]
    cold_p2 = [p2_metrics[r]["cold_cost"] for _, r in pairs]
    idle_p2 = [p2_metrics[r]["idle_cost"] for _, r in pairs]

    # Phase 1 bars
    ax.bar(x - width/2, cold_p1, width, label="P1 Cold Cost",
           color="#e15759", alpha=0.8)
    ax.bar(x - width/2, idle_p1, width, bottom=cold_p1,
           label="P1 Idle Cost", color="#e15759", alpha=0.3)

    # Phase 2 bars
    ax.bar(x + width/2, cold_p2, width, label="P2 Cold Cost",
           color="#4e79a7", alpha=0.8)
    ax.bar(x + width/2, idle_p2, width, bottom=cold_p2,
           label="P2 Idle Cost", color="#4e79a7", alpha=0.3)

    ax.set_xlabel("Base Model")
    ax.set_ylabel("Cost")
    ax.set_title("Phase 1 vs Phase 2: Cost Decomposition")
    ax.set_xticks(x)
    ax.set_xticklabels([p for p, _ in pairs])
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v/1e6:.0f}M"))

    path = os.path.join(GRAPHS_DIR, "cost_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_sla_comparison(p1_metrics, p2_metrics):
    """Plot Phase 1 vs Phase 2 SLA comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pairs = [
        ("Reactive", "RiskAware(Reactive)"),
        ("Forecast_Only", "RiskAware(Forecast_Only)"),
        ("TCN", "RiskAware(TCN)"),
    ]
    labels = [p for p, _ in pairs]
    x = np.arange(len(pairs))
    width = 0.35

    # Request SLA
    ax = axes[0]
    sla_p1 = [p1_metrics[p]["request_sla"] for p, _ in pairs]
    sla_p2 = [p2_metrics[r]["request_sla"] for _, r in pairs]
    ax.bar(x - width/2, sla_p1, width, label="Phase 1",
           color="#e15759", alpha=0.8)
    ax.bar(x + width/2, sla_p2, width, label="Phase 2",
           color="#4e79a7", alpha=0.8)
    ax.set_ylabel("Request SLA")
    ax.set_title("Request SLA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(min(min(sla_p1), min(sla_p2)) - 0.005, 1.0)

    # Extreme SLA
    ax = axes[1]
    esla_p1 = [p1_metrics[p]["extreme_sla"] for p, _ in pairs]
    esla_p2 = [p2_metrics[r]["extreme_sla"] for _, r in pairs]
    ax.bar(x - width/2, esla_p1, width, label="Phase 1",
           color="#e15759", alpha=0.8)
    ax.bar(x + width/2, esla_p2, width, label="Phase 2",
           color="#4e79a7", alpha=0.8)
    ax.set_ylabel("Extreme SLA")
    ax.set_title("Extreme Event SLA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(min(min(esla_p1), min(esla_p2)) - 0.01, 1.0)

    plt.suptitle("Phase 1 vs Phase 2: SLA Comparison", fontsize=14, y=1.02)
    plt.tight_layout()

    path = os.path.join(GRAPHS_DIR, "sla_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_dynamic_buffer(p2_diagnostics):
    """Plot sigma_t and buffer_t over time for RiskAware(TCN)."""
    name = "RiskAware(TCN)"
    if name not in p2_diagnostics:
        print(f"  Skipping dynamic buffer plot: {name} not found")
        return

    diag = p2_diagnostics[name]
    n = len(diag)

    # Show a representative window (first 2000 timesteps)
    window = min(2000, n)
    t = np.arange(window)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Actual demand and predictions
    ax = axes[0]
    ax.plot(t, diag["actual_demand"].values[:window],
            color="black", alpha=0.5, linewidth=0.5, label="Actual Demand")
    ax.plot(t, diag["base_prediction"].values[:window],
            color="#e15759", alpha=0.7, linewidth=0.5, label="Base Prediction")
    ax.plot(t, diag["predicted"].values[:window],
            color="#4e79a7", alpha=0.7, linewidth=0.5,
            label="Risk-Aware Prediction")
    ax.set_ylabel("Concurrency")
    ax.set_title(f"Dynamic Risk Buffer -- {name}")
    ax.legend(loc="upper right", fontsize=9)

    # Panel 2: Rolling volatility
    ax = axes[1]
    ax.plot(t, diag["sigma_t"].values[:window],
            color="#59a14f", linewidth=0.8)
    ax.axhline(y=diag["sigma_t"].values[0], color="gray",
               linestyle="--", alpha=0.5, label="sigma_train")
    ax.set_ylabel("sigma_t")
    ax.set_title("Rolling Volatility (sigma_t)")
    ax.legend(loc="upper right", fontsize=9)

    # Panel 3: Dynamic buffer
    ax = axes[2]
    ax.fill_between(t, 0, diag["buffer_t"].values[:window],
                    color="#f28e2b", alpha=0.5)
    ax.plot(t, diag["buffer_t"].values[:window],
            color="#f28e2b", linewidth=0.5)
    ax.set_ylabel("Buffer")
    ax.set_xlabel("Timestep")
    ax.set_title("Dynamic CVaR Buffer (buffer_t = sigma_t * CVaR_z)")

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "dynamic_buffer.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_buffer_distribution(p2_diagnostics):
    """Plot histogram of buffer values showing non-constant behavior."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = ["RiskAware(Reactive)", "RiskAware(Forecast_Only)",
             "RiskAware(TCN)"]
    colors = ["#76b7b2", "#59a14f", "#9c755f"]

    for ax, name, color in zip(axes, names, colors):
        if name not in p2_diagnostics:
            continue
        buffer = p2_diagnostics[name]["buffer_t"].values
        ax.hist(buffer, bins=50, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(x=np.mean(buffer), color="red", linestyle="--",
                   label=f"Mean: {np.mean(buffer):,.0f}")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Buffer Value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.suptitle("Distribution of Dynamic Buffer Values", fontsize=14, y=1.02)
    plt.tight_layout()

    path = os.path.join(GRAPHS_DIR, "buffer_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_prediction_overlay(p2_diagnostics):
    """Overlay base vs risk-aware prediction vs actual for TCN."""
    name = "RiskAware(TCN)"
    if name not in p2_diagnostics:
        print(f"  Skipping overlay plot: {name} not found")
        return

    diag = p2_diagnostics[name]
    # Show a 500-timestep window centered on a volatile region
    n = len(diag)
    # Find the region with highest buffer variance (most interesting)
    chunk_size = 500
    best_start = 0
    best_var = 0
    for start in range(0, n - chunk_size, chunk_size // 2):
        v = np.var(diag["buffer_t"].values[start:start + chunk_size])
        if v > best_var:
            best_var = v
            best_start = start

    end = best_start + chunk_size
    sl = slice(best_start, end)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(diag["actual_demand"].values[sl],
            color="black", alpha=0.6, linewidth=0.8, label="Actual Demand")
    ax.plot(diag["base_prediction"].values[sl],
            color="#e15759", alpha=0.7, linewidth=0.8, label="Base (TCN)")
    ax.plot(diag["predicted"].values[sl],
            color="#4e79a7", alpha=0.7, linewidth=0.8,
            label="Risk-Aware (TCN)")

    # Shade the buffer region
    ax.fill_between(
        range(chunk_size),
        diag["base_prediction"].values[sl],
        diag["predicted"].values[sl],
        alpha=0.2, color="#4e79a7", label="EVT-CVaR Buffer"
    )

    ax.set_xlabel("Timestep (relative)")
    ax.set_ylabel("Concurrency")
    ax.set_title(f"Phase 1 TCN vs Phase 2 RiskAware(TCN) -- "
                 f"Timesteps {best_start}-{end}")
    ax.legend(loc="upper right")

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
