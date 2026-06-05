"""
Phase 1 Graph Generation -- Azure Dataset
============================================

Generates publication-quality graphs for Phase 1 results:
    1. Cost comparison (bar chart with cold/idle decomposition)
    2. SLA comparison (grouped bar: Request SLA and Extreme SLA)
    3. Extreme event analysis (test demand with threshold + cold start markers)
    4. Demand distribution (histogram with P90, P99 lines)
    5. Baseline vs actual time series (multi-panel, representative window)
    6. Cold start heatmap (where cold starts occur for each model)

All graphs use:
    - Publication-quality styling
    - Consistent color palette
    - Saved as PNG (300 DPI)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase1", "azure")
GRAPHS_DIR = os.path.join(PROJECT_ROOT, "graphs", "phase1", "azure")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "azure")

# Publication color palette
COLORS = {
    "Reactive": "#4e79a7",
    "Static_P90": "#f28e2b",
    "Forecast_Only": "#e15759",
    "TCN": "#59a14f",
}
COLD_COLOR = "#d62728"
IDLE_COLOR = "#aec7e8"

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_data():
    """Load all results and metrics."""
    # Load metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "r") as f:
        all_metrics = json.load(f)

    # Load per-model results
    all_results = {}
    for model_name in all_metrics.keys():
        path = os.path.join(RESULTS_DIR, f"{model_name}_results.csv")
        all_results[model_name] = pd.read_csv(path, parse_dates=["timestamp"])

    # Load training data for distribution plots
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["timestamp"])

    return all_metrics, all_results, train_df


def plot_cost_comparison(all_metrics):
    """Bar chart: total cost decomposed into cold cost + idle cost."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(all_metrics.keys())
    x = np.arange(len(models))
    width = 0.6

    cold_costs = [all_metrics[m]["cold_cost"] for m in models]
    idle_costs = [all_metrics[m]["idle_cost"] for m in models]

    bars1 = ax.bar(x, cold_costs, width, label="Cold Start Cost", color=COLD_COLOR, alpha=0.85)
    bars2 = ax.bar(x, idle_costs, width, bottom=cold_costs, label="Idle Capacity Cost",
                   color=IDLE_COLOR, alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Total Cost")
    ax.set_title("Cost Comparison: Cold Start vs Idle Capacity\n(c_cold=10, c_idle=1 -- experimental assumption)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x/1e6:.0f}M"))

    # Add total cost labels
    for i, m in enumerate(models):
        total = all_metrics[m]["total_cost"]
        ax.text(i, total + total * 0.01, f"{total/1e6:.1f}M",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "cost_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_sla_comparison(all_metrics):
    """Grouped bar chart: Request SLA and Extreme SLA."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(all_metrics.keys())
    x = np.arange(len(models))
    width = 0.35

    request_sla = [all_metrics[m]["request_sla"] for m in models]
    extreme_sla = [all_metrics[m]["extreme_sla"] for m in models]

    bars1 = ax.bar(x - width/2, request_sla, width, label="Request SLA",
                   color="#4e79a7", alpha=0.85)
    bars2 = ax.bar(x + width/2, extreme_sla, width, label="Extreme Event SLA",
                   color="#e15759", alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("SLA (fraction of requests served)")
    ax.set_title("SLA Comparison: Overall vs Extreme Events\n(Extreme = demand > P99 of training data)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(min(min(request_sla), min(extreme_sla)) * 0.98, 1.002)

    # Add value labels
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f"{height:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "sla_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_extreme_event_analysis(all_results, train_df):
    """Scatter plot of test demand with extreme threshold and cold start markers."""
    # Compute P99 threshold from training data
    threshold = np.percentile(train_df["concurrency"].values, 99)

    # Use first model's results for demand (same across all models)
    first_model = list(all_results.keys())[0]
    results = all_results[first_model]

    fig, ax = plt.subplots(figsize=(14, 6))

    timestamps = results["timestamp"]
    demand = results["actual_demand"]
    is_extreme = results["is_extreme"]

    # Plot all demand
    ax.plot(timestamps, demand, color="#999999", linewidth=0.5, alpha=0.6, label="Demand")

    # Highlight extreme events
    extreme_mask = is_extreme.astype(bool)
    ax.scatter(timestamps[extreme_mask], demand[extreme_mask],
               color=COLD_COLOR, s=15, zorder=5, label=f"Extreme events (n={extreme_mask.sum()})")

    # Threshold line
    ax.axhline(y=threshold, color="#ff7f0e", linestyle="--", linewidth=2,
               label=f"P99 threshold = {threshold:,.0f}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Demand (concurrency)")
    ax.set_title("Test Set: Demand Timeline with Extreme Event Identification\n(Threshold = P99 of TRAINING data)")
    ax.legend(loc="upper right")
    plt.xticks(rotation=45)

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "extreme_event_analysis.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_demand_distribution(train_df):
    """Histogram of training demand with P90 and P99 lines."""
    fig, ax = plt.subplots(figsize=(10, 6))

    demand = train_df["concurrency"].values
    p90 = np.percentile(demand, 90)
    p99 = np.percentile(demand, 99)

    ax.hist(demand, bins=80, color="#4e79a7", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(x=p90, color="#f28e2b", linestyle="--", linewidth=2,
               label=f"P90 = {p90:,.0f}")
    ax.axvline(x=p99, color="#e15759", linestyle="--", linewidth=2,
               label=f"P99 = {p99:,.0f}")
    ax.axvline(x=demand.mean(), color="#59a14f", linestyle="-.", linewidth=1.5,
               label=f"Mean = {demand.mean():,.0f}")

    ax.set_xlabel("Concurrency (requests/minute)")
    ax.set_ylabel("Count")
    ax.set_title("Training Data: Demand Distribution\n(Used for Static P90 and extreme threshold computation)")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x/1e3:.0f}K"))

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "demand_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_baseline_vs_actual(all_results):
    """Multi-panel: predicted vs actual for each model over a representative window."""
    models = list(all_results.keys())
    n_models = len(models)

    # Choose a representative 500-timestep window from the test set
    # Pick a window that contains some extreme events for visual interest
    first_results = all_results[models[0]]
    extreme_indices = np.where(first_results["is_extreme"].values)[0]
    if len(extreme_indices) > 0:
        # Center the window around the first extreme event
        center = extreme_indices[len(extreme_indices) // 4]
        start = max(0, center - 250)
        end = min(len(first_results), start + 500)
    else:
        start = len(first_results) // 4
        end = start + 500

    fig, axes = plt.subplots(n_models, 1, figsize=(14, 3.5 * n_models), sharex=True)

    for i, model_name in enumerate(models):
        ax = axes[i]
        results = all_results[model_name]
        window = results.iloc[start:end]

        ax.plot(window["timestamp"], window["actual_demand"],
                color="#333333", linewidth=0.8, alpha=0.8, label="Actual demand")
        ax.plot(window["timestamp"], window["provisioned"],
                color=COLORS.get(model_name, "#999999"), linewidth=0.8, alpha=0.8,
                label=f"{model_name} provisioned")

        # Shade cold starts (under-provisioned)
        cold_mask = window["cold_starts"] > 0
        if cold_mask.any():
            ax.fill_between(window["timestamp"],
                          window["provisioned"], window["actual_demand"],
                          where=cold_mask, color=COLD_COLOR, alpha=0.3,
                          label="Cold starts")

        ax.set_ylabel("Concurrency")
        ax.set_title(f"{model_name}", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x/1e3:.0f}K"))

    axes[-1].set_xlabel("Time")
    plt.suptitle("Provisioned vs Actual Demand (representative test window)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = os.path.join(GRAPHS_DIR, "baseline_vs_actual.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_cold_start_timeline(all_results):
    """Multi-panel: cold start occurrences over time for each model."""
    models = list(all_results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(n_models, 1, figsize=(14, 2.5 * n_models), sharex=True)

    for i, model_name in enumerate(models):
        ax = axes[i]
        results = all_results[model_name]

        cold = results["cold_starts"].values
        timestamps = results["timestamp"]

        # Plot cold starts as vertical bars
        ax.bar(timestamps, cold, width=0.04, color=COLORS.get(model_name, "#999999"),
               alpha=0.7)

        ax.set_ylabel("Cold Starts")
        ax.set_title(f"{model_name} (total: {cold.sum():,.0f}, "
                     f"rate: {(cold > 0).mean():.2%})",
                     fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x/1e3:.0f}K"))

    axes[-1].set_xlabel("Time")
    plt.suptitle("Cold Start Timeline: Where Under-Provisioning Occurs",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = os.path.join(GRAPHS_DIR, "cold_start_timeline.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_prediction_error_distribution(all_results):
    """Histogram of prediction errors for each model."""
    models = list(all_results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 5), sharey=True)

    for i, model_name in enumerate(models):
        ax = axes[i]
        results = all_results[model_name]

        error = results["predicted"] - results["actual_demand"]
        ax.hist(error, bins=60, color=COLORS.get(model_name, "#999999"),
                alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
        ax.axvline(x=error.mean(), color="red", linestyle="--", linewidth=1,
                   label=f"Mean = {error.mean():,.0f}")
        ax.set_xlabel("Prediction Error\n(predicted - actual)")
        ax.set_title(f"{model_name}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x/1e3:.0f}K"))

    axes[0].set_ylabel("Count")
    plt.suptitle("Prediction Error Distribution\n(negative = under-provision = cold starts)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(GRAPHS_DIR, "prediction_error_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("Phase 1: Graph Generation -- Azure Dataset")
    print("=" * 60)

    os.makedirs(GRAPHS_DIR, exist_ok=True)

    # Load data
    print("\nLoading results...")
    all_metrics, all_results, train_df = load_data()

    # Generate all graphs
    print("\nGenerating graphs...")
    plot_cost_comparison(all_metrics)
    plot_sla_comparison(all_metrics)
    plot_extreme_event_analysis(all_results, train_df)
    plot_demand_distribution(train_df)
    plot_baseline_vs_actual(all_results)
    plot_cold_start_timeline(all_results)
    plot_prediction_error_distribution(all_results)

    print(f"\nAll graphs saved to: {GRAPHS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
