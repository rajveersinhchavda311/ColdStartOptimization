"""
Huawei Dataset Preprocessing — Phase 5 External Validation
===========================================================

Regenerates train/val/test splits for all 6 Huawei subdirectories
(combined, R1–R5) with lag_1440 added and burn-in applied.

The existing splits were generated without lag_1440 and without the
1440-row burn-in. This script overwrites them with correct splits.

Feature set (matches Azure exactly):
    lag_1..lag_10  +  lag_1440  (daily seasonal lag)

Burn-in: first 1440 rows dropped (lag_1440 requires 1440 prior observations)

Split: 60/20/20 chronological on post-burn-in data
    44,640 total rows  →  43,200 post-burn-in
    → train=25,920 / val=8,640 / test=8,640

Output columns (matches Azure):
    [timestamp, concurrency, lag_1, lag_2, ..., lag_10, lag_1440]
"""

import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

HUAWEI_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "huawei")
REGIONS = ["combined", "R1", "R2", "R3", "R4", "R5"]

LAG_COLS = [f"lag_{i}" for i in range(1, 11)] + ["lag_1440"]
BURN_IN = 1440
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20
TEST_FRAC  = 0.20


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag_1..lag_10 and lag_1440 to a concurrency time series."""
    df = df.copy()
    for i in range(1, 11):
        df[f"lag_{i}"] = df["concurrency"].shift(i)
    df["lag_1440"] = df["concurrency"].shift(1440)
    return df


def split_data(df: pd.DataFrame):
    """Apply burn-in and chronological 60/20/20 split."""
    # Drop rows where any lag is NaN (first 1440 rows)
    df = df.dropna(subset=LAG_COLS).reset_index(drop=True)

    n = len(df)
    n_train = int(round(n * TRAIN_FRAC))
    n_val   = int(round(n * VAL_FRAC))
    n_test  = n - n_train - n_val

    train = df.iloc[:n_train].copy()
    val   = df.iloc[n_train:n_train + n_val].copy()
    test  = df.iloc[n_train + n_val:].copy()

    return train, val, test, n


def compute_split_info(train: pd.DataFrame, val: pd.DataFrame,
                       test: pd.DataFrame) -> dict:
    """Save row counts and demand statistics from training set."""
    conc = train["concurrency"].values
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_total_post_burnin": len(train) + len(val) + len(test),
        "train_start": str(train["timestamp"].iloc[0]),
        "train_end": str(train["timestamp"].iloc[-1]),
        "val_start": str(val["timestamp"].iloc[0]),
        "val_end": str(val["timestamp"].iloc[-1]),
        "test_start": str(test["timestamp"].iloc[0]),
        "test_end": str(test["timestamp"].iloc[-1]),
        "demand_mean": float(np.mean(conc)),
        "demand_std": float(np.std(conc)),
        "demand_P90": float(np.percentile(conc, 90)),
        "demand_P99": float(np.percentile(conc, 99)),
        "demand_min": float(np.min(conc)),
        "demand_max": float(np.max(conc)),
        "columns": ["timestamp", "concurrency"] + LAG_COLS,
        "burn_in_rows_dropped": BURN_IN,
        "note": "Splits regenerated for Phase 5 with lag_1440 added",
    }


def process_region(name: str) -> None:
    region_dir = os.path.join(HUAWEI_DIR, name)
    full_path = os.path.join(region_dir, "full_series.csv")

    print(f"\n[{name}] Loading {full_path}")
    df = pd.read_csv(full_path, parse_dates=["timestamp"])
    print(f"  Raw rows: {len(df):,}  |  Columns: {list(df.columns)}")

    assert "timestamp" in df.columns, "Missing 'timestamp' column"
    assert "concurrency" in df.columns, "Missing 'concurrency' column"

    # Sort chronologically (safety)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute lag features
    df = compute_features(df)

    # Split
    train, val, test, n_post = split_data(df)

    print(f"  Post-burn-in: {n_post:,} rows")
    print(f"  Train: {len(train):,}  |  Val: {len(val):,}  |  Test: {len(test):,}")
    print(f"  Columns: {list(train.columns)}")

    # Verify no NaN in lag columns
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        for col in LAG_COLS:
            assert split_df[col].notna().all(), \
                f"NaN found in {col} for {split_name} split of {name}"

    # Verify lag_1440 is non-null (burn-in correctly applied)
    assert train["lag_1440"].notna().all(), f"lag_1440 has NaN in train for {name}"

    # Verify chronological ordering
    assert train["timestamp"].iloc[-1] < val["timestamp"].iloc[0], \
        f"Train/val overlap in {name}"
    assert val["timestamp"].iloc[-1] < test["timestamp"].iloc[0], \
        f"Val/test overlap in {name}"

    # Save splits
    col_order = ["timestamp", "concurrency"] + LAG_COLS
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        out_path = os.path.join(region_dir, f"{split_name}.csv")
        split_df[col_order].to_csv(out_path, index=False)
        print(f"  Saved: {split_name}.csv ({len(split_df):,} rows)")

    # Save split info
    info = compute_split_info(train, val, test)
    info_path = os.path.join(region_dir, "split_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Saved: split_info.json")
    print(f"  Demand stats (train): mean={info['demand_mean']:,.1f}  "
          f"std={info['demand_std']:,.1f}  P90={info['demand_P90']:,.1f}  "
          f"P99={info['demand_P99']:,.1f}")


def main():
    print("=" * 60)
    print("Huawei Preprocessing — Phase 5")
    print("=" * 60)
    print(f"  Burn-in: {BURN_IN} rows (lag_1440 requires 1440 prior obs)")
    print(f"  Split: {TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/{TEST_FRAC:.0%} (chronological)")
    print(f"  Feature set: {LAG_COLS}")

    for region in REGIONS:
        process_region(region)

    print("\n[DONE] All regions preprocessed")
    print("\nVerification summary:")
    for region in REGIONS:
        region_dir = os.path.join(HUAWEI_DIR, region)
        info_path = os.path.join(region_dir, "split_info.json")
        with open(info_path) as f:
            info = json.load(f)
        print(f"  {region}: train={info['n_train']:,} | "
              f"val={info['n_val']:,} | test={info['n_test']:,} | "
              f"has_lag_1440={'lag_1440' in info['columns']}")


if __name__ == "__main__":
    main()
