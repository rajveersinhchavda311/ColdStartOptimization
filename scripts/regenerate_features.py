"""
Feature Regeneration Script — Adds lag_1440 (Daily Seasonal Lag)
=================================================================

Pre-Phase-3 change. Regenerates data/processed/azure/{train,val,test}.csv
with lag_1440 added alongside the existing lag_1..lag_10 features.

Why lag_1440:
    Azure Functions data has strong daily periodicity. A 10-minute input
    window (lag_1..lag_10) is blind to this. lag_1440 = concurrency[t-1440]
    is exactly the same minute of the prior day, capturing the dominant
    seasonal signal without requiring a 1440-step sequence model.

Changes from original feature_engineering.py:
    - Adds lag_1440 as a seasonal anchor feature
    - First 1440 rows (instead of 10) are dropped for NaN elimination
    - New split sizes: train ~11,232 | val ~3,744 | test ~3,744
    - Outputs directly to data/processed/azure/ (model-facing directory)

All downstream models (Reactive, StaticP90, ForecastOnly, TCN) are
updated to use this new feature set. New seasonal models also use lag_1440.
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "azure")
FULL_SERIES_PATH = os.path.join(DATA_DIR, "full_series.csv")

# Lag configuration
SHORT_LAGS = list(range(1, 11))        # lag_1 .. lag_10  (recent context)
SEASONAL_LAGS = [1440]                  # lag_1440         (daily seasonality)
ALL_LAGS = SHORT_LAGS + SEASONAL_LAGS

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag_1..lag_10 and lag_1440. Drop first 1440 rows (NaN from lag_1440)."""
    for k in ALL_LAGS:
        df[f"lag_{k}"] = df["concurrency"].shift(k)

    # Drop the first max(ALL_LAGS) = 1440 rows where any lag is NaN
    n_drop = max(ALL_LAGS)
    df = df.iloc[n_drop:].reset_index(drop=True)
    return df


def split_chronological(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)
    return train, val, test


def validate(df, train, val, test):
    """Full correctness checks — same rigour as original feature_engineering.py."""
    # 1. Lag correctness for all lags
    for k in ALL_LAGS:
        shifted = df["concurrency"].shift(k).iloc[k:]
        actual_col = df[f"lag_{k}"].iloc[k:]
        assert np.allclose(shifted.values, actual_col.values), f"lag_{k} values wrong"

    # 2. No NaN in any split
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        nan_count = split_df.isna().sum().sum()
        assert nan_count == 0, f"{nan_count} NaN values in {name}"

    # 3. Chronological ordering with no overlap
    assert train["timestamp"].max() < val["timestamp"].min(), "train/val overlap"
    assert val["timestamp"].max() < test["timestamp"].min(), "val/test overlap"

    # 4. Row counts sum correctly
    assert len(train) + len(val) + len(test) == len(df), "row count mismatch"

    print("  All validation checks passed")


def main():
    print("=" * 60)
    print("Feature Regeneration — Adding lag_1440")
    print("=" * 60)

    # Load
    df = pd.read_csv(FULL_SERIES_PATH, parse_dates=["timestamp"])
    print(f"  Loaded full_series: {len(df):,} rows")
    print(f"  Range: {df.timestamp.iloc[0]} -> {df.timestamp.iloc[-1]}")

    # Add lags
    df = add_lag_features(df)
    print(f"  After lag features: {len(df):,} rows (dropped {max(ALL_LAGS)} for NaN)")

    # Convert all lag columns to int
    lag_cols = [f"lag_{k}" for k in ALL_LAGS]
    df[lag_cols] = df[lag_cols].astype(int)

    # Split
    train, val, test = split_chronological(df)
    print(f"\n  Split (60/20/20):")
    print(f"    Train: {len(train):,} rows | {str(train.timestamp.iloc[0])[:16]} -> {str(train.timestamp.iloc[-1])[:16]}")
    print(f"    Val:   {len(val):,} rows | {str(val.timestamp.iloc[0])[:16]} -> {str(val.timestamp.iloc[-1])[:16]}")
    print(f"    Test:  {len(test):,} rows | {str(test.timestamp.iloc[0])[:16]} -> {str(test.timestamp.iloc[-1])[:16]}")
    print(f"  Columns: {list(df.columns)}")

    # Validate
    print("\n  Running validation checks...")
    validate(df, train, val, test)

    # Save
    train.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
    print(f"\n  Saved train.csv, val.csv, test.csv -> {DATA_DIR}")
    print("\nDone. All downstream models now have access to lag_1440.")


if __name__ == "__main__":
    main()
