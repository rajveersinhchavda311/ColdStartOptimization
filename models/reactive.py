"""
Reactive Baseline
==================

Strategy:
    prediction[t] = lag_1[t] = concurrency[t - 1]

Rationale:
    The simplest possible provisioning strategy — provision exactly what was
    needed one minute ago. This represents the "zero intelligence" baseline.

Expected behavior:
    - Tracks steady-state demand well (low idle during flat periods)
    - Systematically lags behind during demand spikes (high cold starts)
    - Overprovisions during demand drops (moderate idle)
    - Serves as the lower bound for forecasting performance

No training required — this is a pure rule-based model.
"""

import pandas as pd
import numpy as np
from models.base import BaseModel


class ReactiveModel(BaseModel):

    @property
    def name(self) -> str:
        return "Reactive"

    @property
    def description(self) -> str:
        return "prediction = lag_1 (provision last minute's demand)"

    def fit(self, train_df: pd.DataFrame) -> None:
        """No-op: Reactive model requires no training."""
        pass

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns lag_1 as the prediction.
        Does NOT access 'concurrency' column — only uses lag_1.
        """
        return df["lag_1"].values.astype(np.float64)
