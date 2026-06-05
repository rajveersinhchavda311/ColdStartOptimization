"""
Forecast-Only Baseline
=======================

Strategy:
    prediction[t] = mean(lag_1[t], lag_2[t], ..., lag_10[t])

Rationale:
    A simple moving average of the past 10 minutes. This represents
    "basic forecasting without risk awareness." It smooths out noise
    but introduces a lag in tracking rapid demand changes.

Expected behavior:
    - Smoother predictions than Reactive (less jitter)
    - Slower response to sudden spikes (averaging delays the signal)
    - Lower idle cost than Static P90 (tracks demand level)
    - Moderate cold starts during rapid transitions
    - Represents the "naive forecasting" approach

No training required — this is a pure rule-based model.
"""

import pandas as pd
import numpy as np
from models.base import BaseModel

NUM_LAGS = 10


class ForecastOnlyModel(BaseModel):

    @property
    def name(self) -> str:
        return "Forecast_Only"

    @property
    def description(self) -> str:
        return "prediction = mean(lag_1, ..., lag_10)"

    def fit(self, train_df: pd.DataFrame) -> None:
        """No-op: Forecast-Only model requires no training."""
        pass

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns the mean of lag_1 through lag_10 as the prediction.
        Does NOT access 'concurrency' column — only uses lag columns.
        """
        lag_cols = [f"lag_{k}" for k in range(1, NUM_LAGS + 1)]
        return df[lag_cols].values.mean(axis=1).astype(np.float64)
