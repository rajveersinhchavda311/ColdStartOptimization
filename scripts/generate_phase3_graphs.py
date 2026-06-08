"""
Phase 3 Graph Generation — Sensitivity Analysis Visualizations
================================================================

Generates publication-quality figures for Phase 3A (one-at-a-time sweep)
and Phase 3B (interaction check).

Figures produced:
  1. sensitivity_curves.png  — 2×3 grid: (Request SLA, Total Cost) × (α, W, threshold)
  2. sensitivity_extreme_sla.png — 1×3 grid: Extreme SLA vs each parameter
  3. interaction_plots.png   — 3×2 grid: interaction effects for all 3 parameter pairs
  4. robustness_overview.png — dot plot of all 3A+3B configs, both models
  5. buffer_sensitivity.png  — how mean buffer changes with each parameter

Output: graphs/phase3/azure/
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
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase3", "azure")
GRAPHS_DIR  = os.path.join(PROJECT_ROOT, "graphs",  "phase3", "azure")

ANCHOR_ALPHA     = 0.99
ANCHOR_W         = 30
ANCHOR_THRESHOLD = 90

# Colors — consistent with Phase 2 palette
COLOR_REACTIVE = "#e15759"   # red
COLOR_TCN      = "#4e79a7"   # blue
COLOR_ANCHOR   = "#59a14f"   # green  (anchor value marker)

MARKERS = {"Reactive": "o", "TCN": "s"}
LINESTYLES = {"Reactive": "--", "TCN": "-"}

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    ":",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    summary_3a = pd.read_csv(os.path.join(RESULTS_DIR, "summary_3a.csv"))
    summary_3b = pd.read_csv(os.path.join(RESULTS_DIR, "summary_3b.csv"))
    return summary_3a, summary_3b


def get_sweep_data(df_3a: pd.DataFrame, sweep_name: str, model: str) -> pd.DataFrame:
    """Return rows from the specified 1-at-a-time sweep for a given model."""
    return df_3a[(df_3a["sweep"] == sweep_name) & (df_3a["model"] == model)].copy()


# ---------------------------------------------------------------------------
# Figure 1: Sensitivity curves (2×3 grid)
# ---------------------------------------------------------------------------
def plot_sensitivity_curves(df_3a: pd.DataFrame) -> None:
    """
    2-row × 3-column grid.
    Row 0: Request SLA vs (α, W, threshold).
    Row 1: Total Cost   vs (α, W, threshold).
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    sweeps = [
        ("alpha",     "α",                     ANCHOR_ALPHA,     [0.95, 0.975, 0.99],  ".3f"),
        ("W",         "Rolling window W",       ANCHOR_W,         [10, 30, 60],          "d"),
        ("threshold", "EVT threshold (Pct.)",   ANCHOR_THRESHOLD, [85, 90, 95],          "d"),
    ]
    param_cols = {"alpha": "alpha", "W": "W", "threshold": "threshold"}
    metrics    = [
        ("request_sla", "Request SLA",    lambda v: f"{v:.4f}"),
        ("total_cost",  "Total Cost (M)", lambda v: f"{v/1e6:.1f}M"),
    ]

    for col_idx, (sweep_name, xlabel, anchor_val, xvals, fmt) in enumerate(sweeps):
        param_col = param_cols[sweep_name]
        for row_idx, (metric_col, ylabel, fmt_fn) in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            for model, color in [("Reactive", COLOR_REACTIVE), ("TCN", COLOR_TCN)]:
                sub = get_sweep_data(df_3a, sweep_name, model).sort_values(param_col)
                if sub.empty:
                    continue
                xs = sub[param_col].values
                ys = sub[metric_col].values

                ax.plot(xs, ys, color=color, marker=MARKERS[model],
                        linestyle=LINESTYLES[model], linewidth=1.8,
                        markersize=6, label=model, zorder=3)

                # Annotate each point
                for x, y in zip(xs, ys):
                    ax.annotate(fmt_fn(y),
                                xy=(x, y), xytext=(0, 6),
                                textcoords="offset points",
                                ha="center", va="bottom",
                                fontsize=7, color=color)

            # Mark anchor value
            ax.axvline(anchor_val, color=COLOR_ANCHOR, linestyle=":",
                       linewidth=1.5, alpha=0.8, zorder=2, label=f"Anchor={anchor_val:{fmt}}")

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            if metric_col == "request_sla":
                all_ys = []
                for model in ["Reactive", "TCN"]:
                    sub = get_sweep_data(df_3a, sweep_name, model)
                    if not sub.empty:
                        all_ys.extend(sub[metric_col].values.tolist())
                if all_ys:
                    ymin = max(0, min(all_ys) - 0.002)
                    ymax = min(1.001, max(all_ys) + 0.002)
                    ax.set_ylim(ymin, ymax)
            elif metric_col == "total_cost":
                ax.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))

            if row_idx == 0:
                ax.set_title(f"vs {xlabel}")
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=11)

            if col_idx == 2 and row_idx == 0:
                ax.legend(loc="lower left", fontsize=8)

    fig.suptitle(
        "Phase 3A: One-at-a-Time Sensitivity Analysis\n"
        "EVT-CVaR hyperparameters — RiskAware(Reactive) and RiskAware(TCN)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, "sensitivity_curves.png")


# ---------------------------------------------------------------------------
# Figure 2: Extreme SLA sensitivity (1×3)
# ---------------------------------------------------------------------------
def plot_extreme_sla_curves(df_3a: pd.DataFrame) -> None:
    """Request SLA and Extreme SLA on same axes — shows if SLA story is robust."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sweeps = [
        ("alpha",     "α",                     ANCHOR_ALPHA,     ".3f"),
        ("W",         "Rolling window W",       ANCHOR_W,         "d"),
        ("threshold", "EVT threshold (Pct.)",   ANCHOR_THRESHOLD, "d"),
    ]
    param_cols = {"alpha": "alpha", "W": "W", "threshold": "threshold"}

    for col_idx, (sweep_name, xlabel, anchor_val, fmt) in enumerate(sweeps):
        ax = axes[col_idx]
        param_col = param_cols[sweep_name]

        for model, color in [("Reactive", COLOR_REACTIVE), ("TCN", COLOR_TCN)]:
            sub = get_sweep_data(df_3a, sweep_name, model).sort_values(param_col)
            if sub.empty:
                continue
            xs = sub[param_col].values
            ax.plot(xs, sub["request_sla"].values, color=color,
                    marker=MARKERS[model], linestyle="-", linewidth=1.8,
                    markersize=6, label=f"{model} Req. SLA", zorder=3)
            ax.plot(xs, sub["extreme_sla"].values, color=color,
                    marker=MARKERS[model], linestyle=":", linewidth=1.5,
                    markersize=5, alpha=0.75, label=f"{model} Ext. SLA", zorder=3)

        ax.axvline(anchor_val, color=COLOR_ANCHOR, linestyle=":",
                   linewidth=1.5, alpha=0.7, label=f"Anchor={anchor_val:{fmt}}")
        ax.axhline(0.99, color="gray", linestyle="--", linewidth=1.0,
                   alpha=0.5, label="SLA = 0.99")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("SLA")
        ax.set_title(f"SLA vs {xlabel}")

        all_sla = []
        for model in ["Reactive", "TCN"]:
            sub = get_sweep_data(df_3a, sweep_name, model)
            if not sub.empty:
                all_sla.extend(sub["request_sla"].tolist())
                all_sla.extend(sub["extreme_sla"].tolist())
        if all_sla:
            ax.set_ylim(max(0, min(all_sla) - 0.003), 1.002)

        if col_idx == 0:
            ax.legend(loc="lower left", fontsize=7.5, ncol=1)

    fig.suptitle(
        "Phase 3A: Request SLA and Extreme SLA Sensitivity\n"
        "(solid = Request SLA, dotted = Extreme SLA)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "sensitivity_extreme_sla.png")


# ---------------------------------------------------------------------------
# Figure 3: Interaction plots (3B)
# ---------------------------------------------------------------------------
def plot_interaction_plots(df_3b: pd.DataFrame) -> None:
    """
    3-row × 2-column figure.
    Rows: α×W interaction, α×threshold interaction, W×threshold interaction.
    Cols: Request SLA, Total Cost.

    For each interaction pair (A, B):
      x-axis = A (2 values), separate lines for each B level.
      The third parameter is averaged across its 2 levels.
    """
    interactions = [
        ("alpha",     "W",         "threshold", "α",   "W",     "P"),
        ("alpha",     "threshold", "W",         "α",   "Pct.",  "W"),
        ("W",         "threshold", "alpha",     "W",   "Pct.",  "α"),
    ]
    metrics = [
        ("request_sla", "Request SLA",    lambda v: f"{v:.4f}"),
        ("total_cost",  "Total Cost (M)", lambda v: f"{v/1e6:.1f}M"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))

    for row_idx, (x_param, line_param, avg_param,
                  x_label, line_label, avg_label) in enumerate(interactions):
        for col_idx, (metric_col, ylabel, fmt_fn) in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            for model, model_color in [("Reactive", COLOR_REACTIVE),
                                        ("TCN", COLOR_TCN)]:
                sub_m = df_3b[df_3b["model"] == model]
                if sub_m.empty:
                    continue

                x_vals    = sorted(sub_m[x_param].unique())
                line_vals = sorted(sub_m[line_param].unique())

                line_styles = ["-", "--"]
                alphas      = [0.9, 0.6]

                for li, (lv, ls, la) in enumerate(zip(line_vals, line_styles, alphas)):
                    # Average over the third parameter's levels
                    grp = sub_m[sub_m[line_param] == lv].groupby(x_param)[metric_col].mean()
                    xs  = [x for x in x_vals if x in grp.index]
                    ys  = [grp[x] for x in xs]

                    lv_fmt = f"{lv}" if isinstance(lv, int) else f"{lv:.2f}"
                    label  = f"{model} ({line_label}={lv_fmt})"

                    ax.plot(xs, ys, color=model_color, marker=MARKERS[model],
                            linestyle=ls, linewidth=1.8, markersize=6,
                            alpha=la, label=label, zorder=3)

                    for x, y in zip(xs, ys):
                        ax.annotate(fmt_fn(y),
                                    xy=(x, y), xytext=(0, 6),
                                    textcoords="offset points",
                                    ha="center", va="bottom",
                                    fontsize=6.5, color=model_color, alpha=la)

            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            title = f"{x_label} × {line_label}  [{avg_label} averaged]"
            ax.set_title(title, fontsize=10)

            if metric_col == "request_sla":
                all_ys = []
                for model in ["Reactive", "TCN"]:
                    sub_m = df_3b[df_3b["model"] == model]
                    if not sub_m.empty:
                        all_ys.extend(sub_m[metric_col].tolist())
                if all_ys:
                    ax.set_ylim(max(0, min(all_ys) - 0.003), 1.002)
                ax.axhline(0.99, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)
            else:
                ax.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))

            if row_idx == 2 and col_idx == 1:
                ax.legend(loc="best", fontsize=7.5, ncol=1)

    fig.suptitle(
        "Phase 3B: Interaction Effects (2×2×2 Factorial)\n"
        "Parallel lines = no interaction; diverging lines = interaction present\n"
        "(Average over the third parameter's two levels)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "interaction_plots.png")


# ---------------------------------------------------------------------------
# Figure 4: Robustness overview — all configs, both models
# ---------------------------------------------------------------------------
def plot_robustness_overview(df_3a: pd.DataFrame, df_3b: pd.DataFrame) -> None:
    """
    Dot plot showing Request SLA and Total Cost for every unique config × model
    in Phases 3A and 3B.  Demonstrates the "method is robust" narrative.
    """
    # Combine and deduplicate
    df_3a_clean = df_3a.drop_duplicates(subset=["config_key", "model"])
    df_3b_clean = df_3b.drop_duplicates(subset=["config_key", "model"])
    combined    = pd.concat([df_3a_clean, df_3b_clean], ignore_index=True)
    combined    = combined.drop_duplicates(subset=["config_key", "model"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for model, color in [("Reactive", COLOR_REACTIVE), ("TCN", COLOR_TCN)]:
        sub = combined[combined["model"] == model].sort_values("config_key")
        if sub.empty:
            continue
        xs = np.arange(len(sub))

        axes[0].scatter(xs, sub["request_sla"].values, color=color,
                        marker=MARKERS[model], s=50, zorder=3, label=model)
        axes[1].scatter(xs, sub["total_cost"].values / 1e6, color=color,
                        marker=MARKERS[model], s=50, zorder=3, label=model)

    # Anchor reference lines
    anchor_rows = combined[
        (combined["config_key"] == f"a{ANCHOR_ALPHA:.3f}_W{ANCHOR_W:03d}_P{ANCHOR_THRESHOLD:02d}")
    ]
    if not anchor_rows.empty:
        for model, color in [("Reactive", COLOR_REACTIVE), ("TCN", COLOR_TCN)]:
            ar = anchor_rows[anchor_rows["model"] == model]
            if not ar.empty:
                axes[0].axhline(ar["request_sla"].iloc[0], color=color,
                                linestyle=":", linewidth=1.2, alpha=0.5)
                axes[1].axhline(ar["total_cost"].iloc[0] / 1e6, color=color,
                                linestyle=":", linewidth=1.2, alpha=0.5)

    # x-tick labels (config keys, rotated)
    for model in ["Reactive", "TCN"]:
        sub = combined[combined["model"] == model].sort_values("config_key")
        if not sub.empty:
            keys = sub["config_key"].tolist()
            break

    n_configs = len(combined["config_key"].unique())
    xticks_pos = np.arange(n_configs)

    for ax, ylabel, fmt_fn in [
        (axes[0], "Request SLA", lambda v: f"{v:.4f}"),
        (axes[1], "Total Cost (M)", lambda v: f"{v:.0f}M"),
    ]:
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Config (α / W / threshold)")
        ax.set_xticks(xticks_pos)
        ax.set_xticklabels(
            sorted(combined["config_key"].unique()),
            rotation=45, ha="right", fontsize=7
        )
        ax.legend(loc="lower left", fontsize=9)

    axes[0].set_title("Request SLA — all Phase 3A+3B configs")
    axes[1].set_title("Total Cost — all Phase 3A+3B configs")
    axes[0].axhline(0.99, color="gray", linestyle="--", linewidth=1.0,
                    alpha=0.5, label="SLA=0.99 threshold")
    axes[1].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:.0f}M"))

    fig.suptitle(
        "Phase 3 Robustness Overview — All Configurations\n"
        "(Each dot = one unique (config, model) pair; dotted lines = anchor Phase 2 values)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "robustness_overview.png")


# ---------------------------------------------------------------------------
# Figure 5: Buffer sensitivity — mean buffer vs each param (3A only)
# ---------------------------------------------------------------------------
def plot_buffer_sensitivity(df_3a: pd.DataFrame) -> None:
    """
    Shows how the mean EVT-CVaR buffer changes with each hyperparameter.
    Uses the runs/ diagnostics files to compute buffer statistics.
    """
    import glob as glb

    runs_dir = os.path.join(RESULTS_DIR, "runs")
    sweeps = [
        ("alpha",     "α",                     ANCHOR_ALPHA,     ".3f"),
        ("W",         "Rolling window W",       ANCHOR_W,         "d"),
        ("threshold", "EVT threshold (Pct.)",   ANCHOR_THRESHOLD, "d"),
    ]
    param_cols = {"alpha": "alpha", "W": "W", "threshold": "threshold"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for col_idx, (sweep_name, xlabel, anchor_val, fmt) in enumerate(sweeps):
        ax = axes[col_idx]
        param_col = param_cols[sweep_name]

        for model, color in [("Reactive", COLOR_REACTIVE), ("TCN", COLOR_TCN)]:
            sub = df_3a[(df_3a["sweep"] == sweep_name) &
                        (df_3a["model"] == model)].sort_values(param_col)
            if sub.empty:
                continue

            mean_buffers = []
            std_buffers  = []
            xs           = []

            for _, row in sub.iterrows():
                diag_path = os.path.join(
                    runs_dir,
                    f"{row['config_key']}_{model}_diagnostics.csv"
                )
                if not os.path.exists(diag_path):
                    continue
                diag = pd.read_csv(diag_path)
                mean_buffers.append(np.mean(diag["buffer_t"].values))
                std_buffers.append(np.std(diag["buffer_t"].values))
                xs.append(row[param_col])

            if not xs:
                continue

            xs    = np.array(xs)
            means = np.array(mean_buffers)
            stds  = np.array(std_buffers)

            ax.plot(xs, means, color=color, marker=MARKERS[model],
                    linestyle=LINESTYLES[model], linewidth=1.8,
                    markersize=6, label=f"{model} mean(buffer)", zorder=3)
            ax.fill_between(xs, means - stds, means + stds,
                            color=color, alpha=0.12, zorder=2)

        ax.axvline(anchor_val, color=COLOR_ANCHOR, linestyle=":",
                   linewidth=1.5, alpha=0.8, label=f"Anchor={anchor_val:{fmt}}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Buffer value")
        ax.set_title(f"Mean buffer ± 1σ vs {xlabel}")
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K"))

        if col_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        "Phase 3A: EVT-CVaR Buffer Sensitivity\n"
        "(mean ± 1σ across test timesteps; shaded band = buffer volatility)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "buffer_sensitivity.png")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _save(fig, filename: str) -> None:
    path = os.path.join(GRAPHS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("="*60)
    print("Phase 3: Graph Generation — Azure Dataset")
    print("="*60)

    if not os.path.exists(os.path.join(RESULTS_DIR, "summary_3a.csv")):
        print("\nERROR: Phase 3 results not found.")
        print("  Run scripts/run_phase3.py first.")
        sys.exit(1)

    os.makedirs(GRAPHS_DIR, exist_ok=True)

    print("\nLoading results...")
    df_3a, df_3b = load_data()
    print(f"  3A rows: {len(df_3a)} | 3B rows: {len(df_3b)}")

    print("\nGenerating figures...")
    plot_sensitivity_curves(df_3a)
    plot_extreme_sla_curves(df_3a)
    plot_interaction_plots(df_3b)
    plot_robustness_overview(df_3a, df_3b)
    plot_buffer_sensitivity(df_3a)

    print(f"\nAll Phase 3 graphs saved to: {GRAPHS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
