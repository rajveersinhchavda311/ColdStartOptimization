"""
Phase 4 Graph Generation -- Component Ablation Visualizations
=============================================================

Five figures saved to graphs/phase4/azure/:

  1. ablation_sla.png          -- grouped bar: Request SLA + Extreme SLA, 5 conds x 2 models
  2. ablation_cost.png         -- stacked bar: cold + idle cost, 5 conds x 2 models
  3. ablation_incremental.png  -- delta bars for each component addition step
  4. ablation_buffer_profiles.png -- time series of buffer_t for C1-C4, TCN, 500-step window
  5. ablation_2x2_heatmap.png  -- 2x2 Request SLA heatmap (sigma x multiplier), one per model
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results", "phase4", "azure")
CONDITIONS_DIR = os.path.join(RESULTS_DIR, "conditions")
PHASE2_DIR     = os.path.join(PROJECT_ROOT, "results", "phase2", "azure")
GRAPHS_DIR     = os.path.join(PROJECT_ROOT, "graphs",  "phase4", "azure")

# K_GAUSSIAN -- same derivation as run_phase4.py
ALPHA      = 0.99
_z_alpha   = norm.ppf(ALPHA)
K_GAUSSIAN = norm.pdf(_z_alpha) / (1 - ALPHA)

CONDITION_ORDER  = ["C0_no_buffer", "C1_static_gaussian", "C2_dynamic_gaussian",
                    "C3_static_evt", "C4_dynamic_evt"]
CONDITION_LABELS = {
    "C0_no_buffer":        "C0\nNo Buffer",
    "C1_static_gaussian":  "C1\nStatic+\nGaussian",
    "C2_dynamic_gaussian": "C2\nDynamic+\nGaussian",
    "C3_static_evt":       "C3\nStatic+\nEVT",
    "C4_dynamic_evt":      "C4\nDynamic+\nEVT",
}

# Colors per condition -- sequential to show progression
COND_COLORS = {
    "C0_no_buffer":        "#adb5bd",  # light gray
    "C1_static_gaussian":  "#74c0fc",  # light blue
    "C2_dynamic_gaussian":  "#1971c2", # dark blue
    "C3_static_evt":       "#ffa94d",  # light orange
    "C4_dynamic_evt":      "#c0392b",  # dark red
}

# Model styles
MODEL_HATCH = {"Reactive": "///", "TCN": ""}
MODEL_ALPHA = {"Reactive": 0.85, "TCN": 1.0}

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
def load_summary() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))


def load_condition_diagnostics(condition_id: str, model_name: str) -> pd.DataFrame:
    path = os.path.join(CONDITIONS_DIR, f"{condition_id}_{model_name}_diagnostics.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def load_phase2_diagnostics(model_name: str) -> pd.DataFrame:
    """Load Phase 2 diagnostics for C4 buffer profile (same params as C4)."""
    path = os.path.join(PHASE2_DIR, f"RiskAware({model_name})_diagnostics.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Figure 1: Ablation SLA -- grouped bar (Request SLA + Extreme SLA)
# ---------------------------------------------------------------------------
def plot_ablation_sla(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    metrics = [
        ("request_sla", "Request SLA"),
        ("extreme_sla",  "Extreme SLA"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        n_conds  = len(CONDITION_ORDER)
        bar_w    = 0.35
        x_base   = np.arange(n_conds)

        for m_idx, model_name in enumerate(["Reactive", "TCN"]):
            offsets = x_base + (m_idx - 0.5) * bar_w
            vals    = []
            colors  = []
            for cid in CONDITION_ORDER:
                row = df[(df["condition_id"] == cid) & (df["model"] == model_name)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else 0.0)
                colors.append(COND_COLORS[cid])

            ax.bar(offsets, vals, width=bar_w,
                          color=colors,
                          alpha=MODEL_ALPHA[model_name],
                          hatch=MODEL_HATCH[model_name],
                          edgecolor="black", linewidth=0.5,
                          label=model_name)

        ax.set_xticks(x_base)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

        # Baseline reference line at 0.99
        ax.axhline(0.99, color="red", linewidth=1, linestyle="--", alpha=0.7,
                   label="SLA = 0.99")

        ymin = min(df[metric].min() * 0.995, 0.94)
        ymax = 1.0005
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    # Shared legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Reactive"),
        Patch(facecolor="white", edgecolor="black", label="TCN"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="SLA = 0.99"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower right")

    fig.suptitle("Phase 4 Ablation: SLA by Condition", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "ablation_sla.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2: Ablation cost -- stacked bar (cold cost + idle cost)
# ---------------------------------------------------------------------------
def plot_ablation_cost(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, model_name in zip(axes, ["Reactive", "TCN"]):
        sub = df[df["model"] == model_name].set_index("condition_id")
        sub = sub.loc[CONDITION_ORDER]

        x      = np.arange(len(CONDITION_ORDER))
        cold_v = sub["cold_cost"].values / 1e6
        idle_v = sub["idle_cost"].values / 1e6
        colors = [COND_COLORS[c] for c in CONDITION_ORDER]

        ax.bar(x, cold_v, label="Cold cost",
               color=colors, alpha=0.9, edgecolor="black", linewidth=0.5)
        ax.bar(x, idle_v, bottom=cold_v, label="Idle cost",
               color=colors, alpha=0.45, edgecolor="black", linewidth=0.5,
               hatch="...")

        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=8)
        ax.set_ylabel("Cost (M)")
        ax.set_title(f"Total Cost -- {model_name}")

        # Annotate totals
        for xi, (cv, iv) in enumerate(zip(cold_v, idle_v)):
            ax.text(xi, cv + iv + 0.5, f"{cv+iv:.0f}M",
                    ha="center", va="bottom", fontsize=7, rotation=0)

    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Phase 4 Ablation: Cost Decomposition", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "ablation_cost.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3: Incremental contribution
# ---------------------------------------------------------------------------
def plot_ablation_incremental(df: pd.DataFrame) -> None:
    """
    Delta bars for each component-addition transition:
      C0->C1: effect of adding any buffer (static Gaussian)
      C1->C3: Gaussian -> EVT (same static sigma)  -- EVT multiplier adds
      C1->C2: static -> dynamic sigma (same Gaussian) -- adaptivity adds
      C2->C4: Dynamic+Gaussian -> Dynamic+EVT  -- EVT on top of dynamic
      C3->C4: Static+EVT -> Dynamic+EVT         -- dynamic on top of EVT
    """
    transitions = [
        ("C0->C1", "C0_no_buffer",        "C1_static_gaussian",  "Any buffer\n(static+Gauss)"),
        ("C1->C3", "C1_static_gaussian",  "C3_static_evt",       "Gaussian->\nEVT (static)"),
        ("C1->C2", "C1_static_gaussian",  "C2_dynamic_gaussian", "static->\ndynamic (Gauss)"),
        ("C2->C4", "C2_dynamic_gaussian", "C4_dynamic_evt",      "Gauss->EVT\n(dynamic)"),
        ("C3->C4", "C3_static_evt",       "C4_dynamic_evt",      "static->\ndynamic (EVT)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_sla, ax_cost = axes

    x        = np.arange(len(transitions))
    bar_w    = 0.30
    colors_m = {"Reactive": "#e15759", "TCN": "#4e79a7"}

    for m_idx, model_name in enumerate(["Reactive", "TCN"]):
        sub = df[df["model"] == model_name].set_index("condition_id")

        sla_deltas  = []
        cost_deltas = []
        for _, from_c, to_c, _ in transitions:
            d_sla  = float(sub.loc[to_c, "request_sla"]) - float(sub.loc[from_c, "request_sla"])
            d_cost = (float(sub.loc[to_c, "total_cost"]) - float(sub.loc[from_c, "total_cost"])) / 1e6
            sla_deltas.append(d_sla * 100)   # in percentage points
            cost_deltas.append(d_cost)

        offset = (m_idx - 0.5) * bar_w
        ax_sla.bar(x + offset, sla_deltas,  width=bar_w,
                   color=colors_m[model_name], alpha=0.85,
                   edgecolor="black", linewidth=0.5, label=model_name)
        ax_cost.bar(x + offset, cost_deltas, width=bar_w,
                    color=colors_m[model_name], alpha=0.85,
                    edgecolor="black", linewidth=0.5, label=model_name)

    xlabels = [t[3] for t in transitions]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

    ax_sla.set_ylabel("Delta Request SLA (pp)")
    ax_sla.set_title("SLA gain per component addition")
    ax_cost.set_ylabel("Delta Total Cost (M)")
    ax_cost.set_title("Cost change per component addition")
    ax_sla.legend(loc="upper right")

    fig.suptitle("Phase 4 Ablation: Incremental Component Contributions",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "ablation_incremental.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4: Buffer profiles (TCN, 500-step window)
# ---------------------------------------------------------------------------
def plot_buffer_profiles(window_start: int = 500, window_len: int = 500) -> None:
    """
    Time series of buffer_t for all 4 non-zero conditions for TCN.
    C1-C3 loaded from Phase 4 conditions/; C4 loaded from Phase 2 diagnostics.
    """
    profile_sources = {
        "C1\nStatic+Gauss":   load_condition_diagnostics("C1_static_gaussian",  "TCN"),
        "C2\nDynamic+Gauss":  load_condition_diagnostics("C2_dynamic_gaussian", "TCN"),
        "C3\nStatic+EVT":     load_condition_diagnostics("C3_static_evt",       "TCN"),
        "C4\nDynamic+EVT":    load_phase2_diagnostics("TCN"),
    }

    line_colors = {
        "C1\nStatic+Gauss":  "#74c0fc",
        "C2\nDynamic+Gauss": "#1971c2",
        "C3\nStatic+EVT":    "#ffa94d",
        "C4\nDynamic+EVT":   "#c0392b",
    }
    linestyles = {
        "C1\nStatic+Gauss":  ":",
        "C2\nDynamic+Gauss": "-",
        "C3\nStatic+EVT":    "--",
        "C4\nDynamic+EVT":   "-",
    }

    fig, ax = plt.subplots(figsize=(14, 5))

    for label, diag in profile_sources.items():
        if diag is None:
            print(f"  Warning: diagnostics not found for {label}, skipping profile")
            continue
        buf = diag["buffer_t"].values
        end = min(window_start + window_len, len(buf))
        t   = np.arange(window_start, end)
        ax.plot(t, buf[window_start:end],
                label=label.replace("\n", " "),
                color=line_colors[label],
                linestyle=linestyles[label],
                linewidth=1.5, alpha=0.9)

    ax.set_xlabel(f"Timestep (test set, steps {window_start}-{window_start+window_len})")
    ax.set_ylabel("Buffer size (provisioning units)")
    ax.set_title("Phase 4 Ablation: Buffer Profiles (TCN, representative window)\n"
                 "C1/C3 = flat lines (static sigma); C2/C4 = adaptive (rolling sigma)")
    ax.legend(loc="upper right", ncol=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K"))

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "ablation_buffer_profiles.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 5: 2x2 Request SLA heatmap
# ---------------------------------------------------------------------------
def plot_2x2_heatmap(df: pd.DataFrame) -> None:
    """
    2x2 heatmap: rows = sigma type (static, dynamic), cols = multiplier (gaussian, evt).
    One subplot per model (Reactive, TCN). Cell value = Request SLA.
    C0 (no buffer) is excluded since it has no sigma/multiplier classification.
    """
    sigma_types  = ["static",  "dynamic"]
    mult_types   = ["gaussian", "evt"]
    sigma_labels = ["Static sigma\n(sigma_train)", "Dynamic sigma\n(rolling)"]
    mult_labels  = ["Gaussian\nmultiplier", "EVT\nmultiplier"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, model_name in zip(axes, ["Reactive", "TCN"]):
        sub = df[(df["model"] == model_name) & (df["sigma_type"] != "none")]
        grid = np.zeros((2, 2))
        for i, st in enumerate(sigma_types):
            for j, mt in enumerate(mult_types):
                row = sub[(sub["sigma_type"] == st) & (sub["multiplier_type"] == mt)]
                grid[i, j] = float(row["request_sla"].iloc[0]) if not row.empty else np.nan

        im = ax.imshow(grid, cmap="YlGn", aspect="auto",
                       vmin=max(0.98, np.nanmin(grid) - 0.001),
                       vmax=1.0)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(mult_labels, fontsize=9)
        ax.set_yticklabels(sigma_labels, fontsize=9)
        ax.set_title(f"{model_name}", fontweight="bold")

        # Annotate cells
        for i in range(2):
            for j in range(2):
                val = grid[i, j]
                if not np.isnan(val):
                    # Condition label for reference
                    cid_map = {
                        (0, 0): "C1", (0, 1): "C3",
                        (1, 0): "C2", (1, 1): "C4",
                    }
                    label = cid_map.get((i, j), "")
                    ax.text(j, i, f"{label}\n{val:.6f}",
                            ha="center", va="center", fontsize=9, fontweight="bold",
                            color="black" if val > np.nanmean(grid) else "white")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     format=ticker.FormatStrFormatter("%.4f"))

    fig.suptitle("Phase 4 Ablation: 2x2 Request SLA Heatmap\n"
                 "(sigma type x multiplier type)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "ablation_2x2_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("="*60)
    print("Phase 4 Graph Generation")
    print("="*60)

    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    if not os.path.exists(summary_path):
        print(f"ERROR: {summary_path} not found.")
        print("  Run scripts/run_phase4.py first.")
        sys.exit(1)

    df = load_summary()
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    print(f"\n  Loaded summary: {len(df)} rows")
    print(f"  Output directory: {GRAPHS_DIR}\n")

    print("[1/5] SLA bar chart...")
    plot_ablation_sla(df)

    print("[2/5] Cost stacked bar chart...")
    plot_ablation_cost(df)

    print("[3/5] Incremental contribution chart...")
    plot_ablation_incremental(df)

    print("[4/5] Buffer profiles...")
    plot_buffer_profiles()

    print("[5/5] 2x2 SLA heatmap...")
    plot_2x2_heatmap(df)

    print("\n[DONE] All Phase 4 graphs saved.")
    print(f"  Graphs: {GRAPHS_DIR}")


if __name__ == "__main__":
    main()
