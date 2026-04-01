"""
Sanity Checks & Visualization
==============================
For each dataset, produces:
    1. Time series plot (timestamp vs concurrency)
    2. Histogram of concurrency distribution
    3. Prints mean, max, 99th percentile statistics

Output:
    dataset/processed/{subdir}/timeseries_plot.png
    dataset/processed/{subdir}/histogram_plot.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")

# (subdirectory relative to PROCESSED_DIR, display name)
DATASETS = [
    ("huawei/combined", "Huawei Combined"),
    ("huawei/R1", "Huawei R1"),
    ("huawei/R2", "Huawei R2"),
    ("huawei/R3", "Huawei R3"),
    ("huawei/R4", "Huawei R4"),
    ("huawei/R5", "Huawei R5"),
    ("azure", "Azure"),
]


def plot_timeseries(df: pd.DataFrame, display_name: str, out_dir: str) -> None:
    """Plot timestamp vs concurrency time series."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df["timestamp"], df["concurrency"], linewidth=0.3, alpha=0.8)
    ax.set_title(f"{display_name} — Concurrency Time Series", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Concurrency (requests/min)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df) // (1440 * 7))))
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(out_dir, "timeseries_plot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_histogram(df: pd.DataFrame, display_name: str, out_dir: str) -> None:
    """Plot histogram of concurrency distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["concurrency"], bins=100, edgecolor="black", alpha=0.7)
    ax.set_title(f"{display_name} — Concurrency Distribution", fontsize=14)
    ax.set_xlabel("Concurrency (requests/min)")
    ax.set_ylabel("Frequency")
    ax.axvline(df["concurrency"].mean(), color="red", linestyle="--",
               label=f"Mean: {df['concurrency'].mean():.1f}")
    ax.axvline(np.percentile(df["concurrency"], 99), color="orange", linestyle="--",
               label=f"P99: {np.percentile(df['concurrency'], 99):.0f}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "histogram_plot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def print_stats(df: pd.DataFrame, display_name: str) -> None:
    """Print summary statistics."""
    c = df["concurrency"]
    print(f"\n  === {display_name} Statistics ===")
    print(f"    Rows:            {len(df):,}")
    print(f"    Time range:      {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"    Mean:            {c.mean():.2f}")
    print(f"    Std:             {c.std():.2f}")
    print(f"    Min:             {c.min()}")
    print(f"    Max:             {c.max()}")
    print(f"    Median:          {c.median():.1f}")
    print(f"    P95:             {np.percentile(c, 95):.0f}")
    print(f"    P99:             {np.percentile(c, 99):.0f}")
    print(f"    Zero-count mins: {(c == 0).sum()} ({(c == 0).mean()*100:.1f}%)")


def process_dataset(subdir: str, display_name: str) -> None:
    """Run all sanity checks for one dataset."""
    dataset_dir = os.path.join(PROCESSED_DIR, subdir)
    input_path = os.path.join(dataset_dir, "full_series.csv")

    if not os.path.exists(input_path):
        print(f"  [SKIP] {display_name}: {input_path} does not exist")
        return

    print(f"\n{'='*50}")
    print(f"  {display_name} Sanity Checks")
    print(f"{'='*50}")

    df = pd.read_csv(input_path, parse_dates=["timestamp"])

    # Statistics
    print_stats(df, display_name)

    # Plots
    plot_timeseries(df, display_name, dataset_dir)
    plot_histogram(df, display_name, dataset_dir)


def main():
    print("=" * 60)
    print("Sanity Checks & Visualization")
    print("=" * 60)

    for subdir, display_name in DATASETS:
        process_dataset(subdir, display_name)

    print("\n✓ All sanity checks complete")


if __name__ == "__main__":
    main()
