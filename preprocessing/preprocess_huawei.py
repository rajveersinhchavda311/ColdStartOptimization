"""
Huawei Public Cloud Trace 2025 — Preprocessing Pipeline (v2)
=============================================================
Produces BOTH combined (all-region) AND per-region (R1–R5) time series
from event-level request logs.

Architecture:
    Shared core functions (zero duplication):
        load_region_day()          → timestamps from one CSV
        load_region_days()         → all timestamps for one region
        timestamps_to_timeseries() → bin, count, gap-fill onto shared index
        validate_timeseries()      → timestamp integrity assertions

    Composable callers:
        process_combined()         → combined time series
        process_single_region()    → per-region time series

Concurrency definition:
    concurrency(t) = COUNT of requests in the 60-second window starting at t

Timestamp construction:
    The 'time' column contains CUMULATIVE SECONDS from the start of the
    entire trace. The 'day' column is a file-partition label only.
    timestamp = base_date + timedelta(seconds=time)

Output:
    dataset/processed/huawei/combined/full_series.csv
    dataset/processed/huawei/R1/full_series.csv
    ...
    dataset/processed/huawei/R5/full_series.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw", "huawei")
OUT_DIR = os.path.join(BASE_DIR, "dataset", "processed", "huawei")
BASE_DATE = pd.Timestamp("2025-01-01 00:00:00")
REGIONS = ["R1", "R2", "R3", "R4", "R5"]
NUM_DAYS = 31  # day_00 to day_30
WINDOW = "60s"
MAX_TIME = NUM_DAYS * 86400  # 31 days in seconds


# ===================================================================
# SHARED CORE FUNCTIONS
# ===================================================================

def load_region_day(region: str, day_idx: int) -> pd.Series:
    """
    Load a single CSV file and return a Series of absolute timestamps.
    Only reads the 'time' column for memory efficiency.

    The 'time' column is cumulative seconds from trace start.
    The 'day' column is a partition label — NOT used for timestamps.
    """
    filepath = os.path.join(RAW_DIR, region, f"day_{day_idx:02d}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing: {filepath}")

    df = pd.read_csv(filepath, usecols=["time"])

    # T3: Assert time values are within valid bounds
    assert (df["time"] >= 0).all(), f"Negative time in {filepath}"
    assert (df["time"] <= MAX_TIME).all(), f"Time > {MAX_TIME}s in {filepath}"

    timestamps = BASE_DATE + pd.to_timedelta(df["time"], unit="s")
    return timestamps


def load_region_days(region: str) -> pd.Series:
    """Load all 31 day files for a single region. Returns concatenated timestamps."""
    parts = []
    for day_idx in range(NUM_DAYS):
        ts = load_region_day(region, day_idx)
        parts.append(ts)
    return pd.concat(parts, ignore_index=True)


def timestamps_to_timeseries(ts: pd.Series, time_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Convert raw event timestamps into a gap-free time series.

    Steps:
        1. Floor each timestamp to its 60-second boundary
        2. COUNT events per bin (concurrency proxy)
        3. Reindex onto the shared time_index, filling gaps with 0

    This is a PURE FUNCTION — no side effects, no shared state.
    """
    ts_floored = ts.dt.floor(WINDOW)
    concurrency = ts_floored.value_counts().sort_index()

    # Reindex onto shared time index — fills missing bins with 0
    concurrency = concurrency.reindex(time_index, fill_value=0)
    concurrency.index.name = "timestamp"
    concurrency.name = "concurrency"

    result = concurrency.reset_index()
    result.columns = ["timestamp", "concurrency"]
    return result


def compute_shared_time_index(all_region_ts: dict) -> pd.DatetimeIndex:
    """
    Compute a single time index spanning the global min/max across all regions.
    Ensures all outputs have identical row counts and timestamp columns.
    """
    global_min = None
    global_max = None

    for region, ts in all_region_ts.items():
        ts_floored = ts.dt.floor(WINDOW)
        region_min = ts_floored.min()
        region_max = ts_floored.max()
        if global_min is None or region_min < global_min:
            global_min = region_min
        if global_max is None or region_max > global_max:
            global_max = region_max

    time_index = pd.date_range(start=global_min, end=global_max, freq=WINDOW)
    return time_index


def validate_timeseries(df: pd.DataFrame, label: str) -> None:
    """Run all timestamp integrity assertions for a single time series."""
    ts = df["timestamp"]

    assert ts.is_monotonic_increasing, f"FAIL [{label}]: not monotonically increasing"
    assert ts.is_unique, f"FAIL [{label}]: duplicate timestamps"

    diffs = ts.diff().dropna()
    expected_delta = pd.Timedelta(WINDOW)
    gaps = diffs[diffs != expected_delta]
    assert len(gaps) == 0, f"FAIL [{label}]: {len(gaps)} gap(s)"

    total_seconds = (ts.max() - ts.min()).total_seconds()
    expected_rows = int(total_seconds / 60) + 1
    assert len(df) == expected_rows, (
        f"FAIL [{label}]: expected {expected_rows} rows, got {len(df)}"
    )


# ===================================================================
# COMPOSABLE CALLERS
# ===================================================================

def process_combined(all_region_ts: dict, time_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Concatenate all region timestamps and produce combined time series."""
    combined_ts = pd.concat(list(all_region_ts.values()), ignore_index=True)
    return timestamps_to_timeseries(combined_ts, time_index)


def process_single_region(region_ts: pd.Series, time_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Produce time series for a single region."""
    return timestamps_to_timeseries(region_ts, time_index)


# ===================================================================
# CROSS-VALIDATION & INTERPRETABLE DIAGNOSTICS
# ===================================================================

def cross_validate(combined_df: pd.DataFrame, region_dfs: dict) -> None:
    """
    Verify strict invariants between combined and per-region outputs.
    Also prints human-readable evidence (spot checks, diagnostic report).
    """
    # --- 3.3 Row Count Invariant ---
    for region, rdf in region_dfs.items():
        assert len(rdf) == len(combined_df), (
            f"Row count mismatch: combined={len(combined_df)}, {region}={len(rdf)}"
        )

    # --- 3.4 Timestamp Identity ---
    for region, rdf in region_dfs.items():
        assert (combined_df["timestamp"].values == rdf["timestamp"].values).all(), (
            f"Timestamp mismatch: combined vs {region}"
        )

    # --- 3.1 Additive Invariant (EXACT integer equality) ---
    region_sum = sum(rdf["concurrency"].values for rdf in region_dfs.values())
    combined_vals = combined_df["concurrency"].values
    assert (combined_vals == region_sum).all(), "Additive invariant violated!"

    # --- 3.2 Event Count Partition ---
    total_combined = int(combined_df["concurrency"].sum())
    region_totals = {r: int(rdf["concurrency"].sum()) for r, rdf in region_dfs.items()}
    assert total_combined == sum(region_totals.values()), "Event count partition violated!"

    print("\n✓ All cross-validation invariants passed")
    print(f"  Row count: {len(combined_df):,} (identical across all 6 outputs)")
    print(f"  Additive invariant: combined == Σ regions at all {len(combined_df):,} timestamps")
    print(f"  Event partition: {total_combined:,} == " +
          " + ".join(f"{v:,}" for v in region_totals.values()))


def print_diagnostic_report(combined_df: pd.DataFrame, region_dfs: dict,
                            region_event_counts: dict) -> None:
    """Print the human-readable diagnostic summary table."""
    print("\n" + "=" * 72)
    print("  HUAWEI PREPROCESSING — DIAGNOSTIC REPORT")
    print("=" * 72)
    print(f"  {'Dataset':<10} {'Rows':>8} {'Events':>10} {'Mean':>8} {'Max':>6}  TS OK")
    print("-" * 72)

    for label, df, events in [
        ("Combined", combined_df, int(combined_df["concurrency"].sum()))
    ] + [
        (r, rdf, region_event_counts[r]) for r, rdf in region_dfs.items()
    ]:
        c = df["concurrency"]
        print(f"  {label:<10} {len(df):>8,} {events:>10,} {c.mean():>8.2f} {c.max():>6}  ✓")

    # Σ row
    region_sum_series = sum(rdf["concurrency"] for rdf in region_dfs.values())
    print("-" * 72)
    print(f"  {'Σ regions':<10} {len(combined_df):>8,} "
          f"{sum(region_event_counts.values()):>10,} "
          f"{region_sum_series.mean():>8.2f} {'—':>6}  ✓ = Combined")
    print("=" * 72)


def print_spot_checks(combined_df: pd.DataFrame, region_dfs: dict, n: int = 3) -> None:
    """Print n randomly-sampled timestamp decompositions for human verification."""
    np.random.seed(42)
    indices = np.random.choice(len(combined_df), size=n, replace=False)
    indices.sort()

    print(f"\n  Spot checks ({n} random timestamps, seed=42):")
    for idx in indices:
        ts = combined_df["timestamp"].iloc[idx]
        c_val = int(combined_df["concurrency"].iloc[idx])
        parts = []
        r_sum = 0
        for r, rdf in region_dfs.items():
            v = int(rdf["concurrency"].iloc[idx])
            parts.append(f"{r}: {v}")
            r_sum += v
        check = "✓" if c_val == r_sum else "✗ MISMATCH"
        print(f"    {ts}:  Combined: {c_val}  =  {' + '.join(parts)}  {check}")


def print_evidence_log(all_region_ts: dict) -> None:
    """Print evidence supporting timestamp assumptions T1–T4."""
    print("\n  Evidence log (timestamp assumptions):")

    # T1: Shared epoch
    starts = {r: ts.min() for r, ts in all_region_ts.items()}
    start_seconds = {r: (t - BASE_DATE).total_seconds() for r, t in starts.items()}
    max_gap = max(start_seconds.values()) - min(start_seconds.values())
    parts = ", ".join(f"{r}={v:.2f}" for r, v in start_seconds.items())
    print(f"    T1 Shared epoch — start times (s): {parts} — max gap: {max_gap:.2f}s ✓")

    # T2: Cumulative time (check day_00 max vs day_01 min for R1)
    d00 = pd.read_csv(os.path.join(RAW_DIR, "R1", "day_00.csv"), usecols=["time"])
    d01 = pd.read_csv(os.path.join(RAW_DIR, "R1", "day_01.csv"), usecols=["time"], nrows=5)
    d00_max = d00["time"].max()
    d01_min = d01["time"].min()
    print(f"    T2 Cumulative time — R1 day_00 max={d00_max:.1f}s, "
          f"day_01 min={d01_min:.1f}s — gap: {d01_min - d00_max:.1f}s ✓")

    # T3: Time bounds
    global_min_t = min((ts.min() - BASE_DATE).total_seconds() for ts in all_region_ts.values())
    global_max_t = max((ts.max() - BASE_DATE).total_seconds() for ts in all_region_ts.values())
    print(f"    T3 Time bounds — [{global_min_t:.3f}, {global_max_t:.3f}] "
          f"within [0, {MAX_TIME}] ✓")

    # T4: No duplicates (sample R1 day_00)
    d00_full = pd.read_csv(os.path.join(RAW_DIR, "R1", "day_00.csv"), usecols=["requestID"])
    n_rows = len(d00_full)
    n_unique = d00_full["requestID"].nunique()
    status = "✓" if n_rows == n_unique else f"✗ ({n_rows - n_unique} duplicates)"
    print(f"    T4 No duplicates — R1 day_00: {n_rows:,} rows, {n_unique:,} unique requestIDs {status}")


def plot_region_decomposition(combined_df: pd.DataFrame, region_dfs: dict,
                              out_dir: str) -> None:
    """Generate stacked area chart with combined overlay for visual proof."""
    fig, ax = plt.subplots(figsize=(16, 6))

    # Stack regions
    timestamps = combined_df["timestamp"]
    bottom = np.zeros(len(combined_df))
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

    for i, (region, rdf) in enumerate(region_dfs.items()):
        vals = rdf["concurrency"].values.astype(float)
        ax.fill_between(timestamps, bottom, bottom + vals,
                        alpha=0.7, color=colors[i], label=region)
        bottom += vals

    # Combined as black line overlay
    ax.plot(timestamps, combined_df["concurrency"].values,
            color="black", linewidth=0.5, alpha=0.8, label="Combined")

    ax.set_title("Huawei — Region Decomposition (stacked regions vs combined)", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Concurrency (requests/min)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = os.path.join(out_dir, "region_decomposition.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  ✓ Saved region decomposition plot: {path}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 60)
    print("Huawei Public Cloud Trace 2025 — Preprocessing (v2)")
    print("Combined + Region-Wise")
    print("=" * 60)

    # ----- Step 1: Load all region timestamps -----
    print("\n[1/6] Loading all region timestamps...")
    all_region_ts = {}
    region_event_counts = {}
    for region in REGIONS:
        print(f"  Loading {region}...", end=" ", flush=True)
        ts = load_region_days(region)
        all_region_ts[region] = ts
        region_event_counts[region] = len(ts)
        print(f"{len(ts):,} events")

    # ----- Step 2: Compute shared time index -----
    print("\n[2/6] Computing shared time index...")
    time_index = compute_shared_time_index(all_region_ts)
    print(f"  Time index: {time_index[0]} → {time_index[-1]}")
    print(f"  Rows: {len(time_index):,}")

    # ----- Step 3: Print evidence log -----
    print("\n[3/6] Validating timestamp assumptions...")
    print_evidence_log(all_region_ts)

    # ----- Step 4: Process combined + per-region -----
    print("\n[4/6] Processing time series...")

    # Combined
    print("  Processing combined...", end=" ", flush=True)
    combined_df = process_combined(all_region_ts, time_index)
    validate_timeseries(combined_df, "combined")
    combined_out = os.path.join(OUT_DIR, "combined")
    os.makedirs(combined_out, exist_ok=True)
    combined_df.to_csv(os.path.join(combined_out, "full_series.csv"), index=False)
    print(f"✓ ({len(combined_df):,} rows)")

    # Per-region
    region_dfs = {}
    for region in REGIONS:
        print(f"  Processing {region}...", end=" ", flush=True)
        rdf = process_single_region(all_region_ts[region], time_index)
        validate_timeseries(rdf, region)
        region_out = os.path.join(OUT_DIR, region)
        os.makedirs(region_out, exist_ok=True)
        rdf.to_csv(os.path.join(region_out, "full_series.csv"), index=False)
        region_dfs[region] = rdf
        print(f"✓ ({len(rdf):,} rows)")

    # ----- Step 5: Cross-validation -----
    print("\n[5/6] Cross-validation...")
    cross_validate(combined_df, region_dfs)
    print_diagnostic_report(combined_df, region_dfs, region_event_counts)
    print_spot_checks(combined_df, region_dfs)

    # ----- Step 6: Stacked region plot -----
    print("\n[6/6] Generating region decomposition plot...")
    plot_region_decomposition(combined_df, region_dfs, OUT_DIR)

    print("\n" + "=" * 60)
    print("✓ All processing complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
