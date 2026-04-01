"""
Feature Engineering & Train/Val/Test Splits
============================================
Applied independently to each dataset's full_series.csv.

Features:
    lag_k[t] = concurrency[t - k]   for k = 1, 2, ..., 10

Splits (chronological, no shuffling):
    train: first 60%
    val:   next  20%
    test:  final 20%

Output:
    dataset/processed/{subdir}/train.csv
    dataset/processed/{subdir}/val.csv
    dataset/processed/{subdir}/test.csv
"""

import os
import pandas as pd
import numpy as np

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

NUM_LAGS = 10
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
# TEST_FRAC = 0.20 (implicit)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag_1 through lag_NUM_LAGS columns."""
    for k in range(1, NUM_LAGS + 1):
        df[f"lag_{k}"] = df["concurrency"].shift(k)
    # Drop rows with NaN (first NUM_LAGS rows)
    df = df.iloc[NUM_LAGS:].reset_index(drop=True)
    return df


def split_chronological(df: pd.DataFrame):
    """Split into train/val/test chronologically (60/20/20)."""
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)

    return train, val, test


def validate(df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame,
             test: pd.DataFrame, dataset_name: str) -> None:
    """Run all validation checks."""
    # 1. Lag correctness
    for k in range(1, NUM_LAGS + 1):
        shifted = df["concurrency"].shift(k).iloc[k:]
        actual = df[f"lag_{k}"].iloc[k:]
        assert np.allclose(shifted.values, actual.values), (
            f"FAIL: lag_{k} values incorrect for {dataset_name}"
        )

    # 2. No NaN values in any split
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        nan_count = split_df.isna().sum().sum()
        assert nan_count == 0, (
            f"FAIL: {nan_count} NaN values in {dataset_name}/{name}"
        )

    # 3. Chronological ordering (no overlap)
    assert train["timestamp"].max() < val["timestamp"].min(), (
        f"FAIL: train/val overlap in {dataset_name}"
    )
    assert val["timestamp"].max() < test["timestamp"].min(), (
        f"FAIL: val/test overlap in {dataset_name}"
    )

    # 4. Row count integrity
    total_with_lags = len(train) + len(val) + len(test)
    assert total_with_lags == len(df), (
        f"FAIL: split row counts don't sum to total for {dataset_name}: "
        f"{total_with_lags} != {len(df)}"
    )

    print(f"  ✓ All validation checks passed for {dataset_name}")
    print(f"    Total (after lag drop): {len(df):,}")
    print(f"    Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")


def process_dataset(subdir: str, display_name: str) -> None:
    """Process one dataset end-to-end."""
    dataset_dir = os.path.join(PROCESSED_DIR, subdir)
    input_path = os.path.join(dataset_dir, "full_series.csv")

    if not os.path.exists(input_path):
        print(f"  [SKIP] {display_name}: {input_path} does not exist")
        return

    print(f"\n--- {display_name} ---")

    # Load
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    print(f"  Loaded: {len(df):,} rows")

    # Add lags
    df = add_lag_features(df)
    print(f"  After lag features: {len(df):,} rows (dropped {NUM_LAGS})")

    # Convert lag columns to int (they should be integer counts)
    lag_cols = [f"lag_{k}" for k in range(1, NUM_LAGS + 1)]
    df[lag_cols] = df[lag_cols].astype(int)

    # Split
    train, val, test = split_chronological(df)

    # Validate
    validate(df, train, val, test, display_name)

    # Save
    train.to_csv(os.path.join(dataset_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(dataset_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(dataset_dir, "test.csv"), index=False)
    print(f"  ✓ Saved train.csv, val.csv, test.csv")


def main():
    print("=" * 60)
    print("Feature Engineering & Train/Val/Test Splits")
    print("=" * 60)

    for subdir, display_name in DATASETS:
        process_dataset(subdir, display_name)

    print("\n✓ All datasets processed successfully")


if __name__ == "__main__":
    main()
