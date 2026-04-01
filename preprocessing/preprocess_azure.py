"""
Azure Functions Dataset 2019 — Preprocessing Pipeline
======================================================
Converts per-minute per-function invocation count files (14 days)
into a minute-level time series of total invocation counts.

Concurrency definition:
    concurrency(t) = SUM of invocation counts across ALL functions for minute t

This gives a platform-wide workload intensity per minute, which serves
as the concurrency proxy.

Output:
    dataset/processed/azure/full_series.csv
    Columns: timestamp, concurrency
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw", "azure")
OUT_DIR = os.path.join(BASE_DIR, "dataset", "processed", "azure")
BASE_DATE = pd.Timestamp("2019-01-01 00:00:00")
NUM_DAYS = 14
MINUTES_PER_DAY = 1440
FILE_PATTERN = "invocations_per_function_md.anon.d{:02d}.csv"


def process_day(day_idx: int) -> pd.DataFrame:
    """
    Process a single day file.
    Returns DataFrame with columns: timestamp, concurrency
    (1440 rows — one per minute)
    """
    filename = FILE_PATTERN.format(day_idx)
    filepath = os.path.join(RAW_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing: {filepath}")

    # Read the CSV — minute columns are '1' through '1440'
    # We read only the minute columns for memory efficiency
    minute_cols = [str(m) for m in range(1, MINUTES_PER_DAY + 1)]

    # Read in chunks to handle large files (~145MB each)
    chunk_size = 10000
    minute_sums = np.zeros(MINUTES_PER_DAY, dtype=np.int64)

    for chunk in pd.read_csv(filepath, usecols=minute_cols, chunksize=chunk_size):
        # Sum across all functions (rows) for each minute (column)
        minute_sums += chunk.sum(axis=0).values.astype(np.int64)

    # Construct timestamps for this day
    # day_idx is 1-indexed (d01, d02, ..., d14)
    day_offset = pd.Timedelta(days=day_idx - 1)
    timestamps = [
        BASE_DATE + day_offset + pd.Timedelta(minutes=m)
        for m in range(MINUTES_PER_DAY)
    ]

    return pd.DataFrame({
        "timestamp": timestamps,
        "concurrency": minute_sums,
    })


def process_all_days() -> pd.DataFrame:
    """
    Process all 14 days and concatenate into a single time series.
    Returns DataFrame with 20,160 rows (14 × 1440).
    """
    frames = []

    for day_idx in range(1, NUM_DAYS + 1):
        print(f"  Processing day {day_idx:02d}/{NUM_DAYS}...")
        day_df = process_day(day_idx)
        frames.append(day_df)

    result = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows: {len(result):,}")
    return result


def validate(df: pd.DataFrame) -> None:
    """Run all timestamp integrity assertions."""
    ts = df["timestamp"]

    # Strictly sorted
    assert ts.is_monotonic_increasing, "FAIL: timestamps are not monotonically increasing"

    # No duplicates
    assert ts.is_unique, "FAIL: duplicate timestamps found"

    # No gaps — consecutive diffs should all be 60 seconds
    diffs = ts.diff().dropna()
    expected_delta = pd.Timedelta("60s")
    gaps = diffs[diffs != expected_delta]
    assert len(gaps) == 0, f"FAIL: {len(gaps)} gap(s) in time index"

    # Expected row count: 14 × 1440 = 20,160
    expected_rows = NUM_DAYS * MINUTES_PER_DAY
    assert len(df) == expected_rows, (
        f"FAIL: expected {expected_rows} rows, got {len(df)}"
    )

    print("✓ All timestamp integrity checks passed")
    print(f"  Rows: {len(df):,}")
    print(f"  Range: {ts.min()} → {ts.max()}")
    print(f"  Concurrency — mean: {df['concurrency'].mean():.2f}, "
          f"max: {df['concurrency'].max()}, "
          f"p99: {np.percentile(df['concurrency'], 99):.0f}")


def main():
    print("=" * 60)
    print("Azure Functions Dataset 2019 — Preprocessing")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Process
    df = process_all_days()

    # Validate
    print("\nRunning validation checks...")
    validate(df)

    # Save
    out_path = os.path.join(OUT_DIR, "full_series.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")
    print(f"  Shape: {df.shape}")


if __name__ == "__main__":
    main()
