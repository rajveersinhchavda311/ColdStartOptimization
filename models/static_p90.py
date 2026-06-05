"""
Static P90 Baseline
====================

Strategy:
    prediction[t] = P90(train_concurrency)   ∀ t

Rationale:
    A fixed provisioning level based on the 90th percentile of training data.
    This deliberately ignores temporal dynamics — it represents the
    "overprovision constantly" strategy.

Expected behavior:
    - Very few cold starts (only when demand exceeds train P90)
    - High idle cost (most timesteps have demand below P90)
    - Constant prediction regardless of recent demand patterns
    - Represents the conservative, risk-averse but wasteful approach

Design decision:
    The P90 value is computed EXCLUSIVELY from training data during fit().
    It is NEVER recalculated on validation or test data. This is critical
    to avoid information leakage.
"""

import pandas as pd
import numpy as np
from models.base import BaseModel


class StaticP90Model(BaseModel):

    def __init__(self):
        self._p90_value: float = None

    @property
    def name(self) -> str:
        return "Static_P90"

    @property
    def description(self) -> str:
        return "prediction = P90 of training concurrency (fixed constant)"

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Compute P90 from training concurrency distribution.
        This value is frozen and used as a constant prediction.
        """
        self._p90_value = float(np.percentile(train_df["concurrency"].values, 90))
        print(f"  [StaticP90] P90(train) = {self._p90_value:,.0f}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns constant P90 value for every timestep.
        Does NOT access 'concurrency' column.
        """
        if self._p90_value is None:
            raise RuntimeError("StaticP90Model.fit() must be called before predict()")
        return np.full(len(df), self._p90_value, dtype=np.float64)
