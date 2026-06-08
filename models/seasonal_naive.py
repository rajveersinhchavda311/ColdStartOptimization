"""
Seasonal Naive Baseline
========================

Strategy:
    prediction[t] = lag_1440[t] = concurrency[t - 1440]

This is the well-known "seasonal naive" forecast: use the observation
from the same period in the prior cycle as the prediction. For minute-level
data with a 1440-minute day, this means "use yesterday's value at this
exact time of day."

Rationale:
    If workloads exhibit strong daily periodicity (which Azure Functions
    data does), the best single-feature predictor may be the same minute
    yesterday rather than the last minute (lag_1). This baseline tests
    whether temporal context from the prior day alone outperforms short-term
    reactive provisioning.

    It is the seasonal counterpart to ReactiveModel. Both require zero
    training; they differ only in which historical lag they use.

    In the forecasting literature, Seasonal Naive is a standard benchmark
    that many sophisticated methods struggle to beat when seasonality is strong.

Expected behavior:
    - Tracks daily patterns well (morning peaks, overnight troughs)
    - Blind to minute-to-minute fluctuations within the day
    - No cold starts from predictable daily ramps (unlike Reactive)
    - Cold starts from day-to-day demand growth or one-off spikes

Requires:
    lag_1440 column in the input DataFrame (produced by regenerate_features.py).

No training required.
"""

import pandas as pd
import numpy as np
from models.base import BaseModel


class SeasonalNaiveModel(BaseModel):

    @property
    def name(self) -> str:
        return "Seasonal_Naive"

    @property
    def description(self) -> str:
        return "prediction = lag_1440 (same minute yesterday)"

    def fit(self, train_df: pd.DataFrame) -> None:
        """No-op: Seasonal Naive requires no training."""
        if "lag_1440" not in train_df.columns:
            raise ValueError(
                "SeasonalNaiveModel requires 'lag_1440' column. "
                "Run scripts/regenerate_features.py first."
            )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns lag_1440 as prediction.
        Does NOT access 'concurrency' — only uses lag_1440.
        """
        if "lag_1440" not in df.columns:
            raise ValueError(
                "SeasonalNaiveModel requires 'lag_1440' column. "
                "Run scripts/regenerate_features.py first."
            )
        return df["lag_1440"].values.astype(np.float64)
