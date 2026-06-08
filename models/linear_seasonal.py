"""
Linear Seasonal Model (Learned Baseline)
=========================================

Strategy:
    Ordinary Least Squares regression on [lag_1, lag_1440].

    prediction[t] = w0 + w1 * lag_1[t] + w2 * lag_1440[t]

This is the simplest learned forecasting model that captures both:
    - Short-term momentum  (lag_1: last minute's demand)
    - Daily seasonality    (lag_1440: same minute yesterday)

Rationale:
    A linear combination of these two features is equivalent to a
    restricted ARIMA-like model with one autoregressive term and one
    seasonal term. It serves as a learned baseline that is transparent,
    fast, and interpretable — a reviewer expectation for any paper
    that claims a deep learning contribution.

    The model answers: "Is there additional value in a non-linear learned
    model (TCN) vs. the best linear combination of the two most informative
    lags?" If TCN doesn't significantly outperform LinearSeasonal, that
    signals the task does not require non-linear capacity.

Training:
    sklearn LinearRegression on [lag_1, lag_1440] from the training split.
    Normalization is applied internally for numerical stability.

Leakage prevention:
    - Fit uses only training data
    - Predict uses only lag_1 and lag_1440 (both strictly historical)
    - 'concurrency' column is never accessed in predict()
"""

import pandas as pd
import numpy as np
from models.base import BaseModel

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


FEATURE_COLS = ["lag_1", "lag_1440"]


class LinearSeasonalModel(BaseModel):
    """
    OLS regression on lag_1 and lag_1440 with z-score normalization.
    """

    def __init__(self):
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required: pip install scikit-learn")
        self._scaler = StandardScaler()
        self._model = LinearRegression()
        self._fitted = False

    @property
    def name(self) -> str:
        return "Linear_Seasonal"

    @property
    def description(self) -> str:
        if self._fitted:
            coef = self._model.coef_
            intercept = self._model.intercept_
            return (
                f"OLS on [lag_1, lag_1440] | "
                f"intercept={intercept:,.0f}, "
                f"w_lag1={coef[0]:.4f}, w_lag1440={coef[1]:.4f}"
            )
        return "OLS regression on [lag_1, lag_1440]"

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Fit OLS on training data.
        Only uses lag_1 and lag_1440 as features.
        """
        for col in FEATURE_COLS:
            if col not in train_df.columns:
                raise ValueError(
                    f"LinearSeasonalModel requires '{col}' column. "
                    "Run scripts/regenerate_features.py first."
                )

        X = train_df[FEATURE_COLS].values.astype(np.float64)
        y = train_df["concurrency"].values.astype(np.float64)

        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._fitted = True

        print(f"  [LinearSeasonal] Fitted on {len(train_df):,} samples")
        print(f"  [LinearSeasonal] Intercept: {self._model.intercept_:,.0f}")
        print(f"  [LinearSeasonal] Coef lag_1: {self._model.coef_[0]:.4f}  "
              f"lag_1440: {self._model.coef_[1]:.4f}")
        train_pred = self._model.predict(X_scaled)
        train_rmse = np.sqrt(np.mean((train_pred - y) ** 2))
        print(f"  [LinearSeasonal] Train RMSE: {train_rmse:,.0f}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using ONLY lag_1 and lag_1440.
        Does NOT access 'concurrency' column.
        """
        if not self._fitted:
            raise RuntimeError(
                "LinearSeasonalModel.fit() must be called before predict()"
            )
        X = df[FEATURE_COLS].values.astype(np.float64)
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled).astype(np.float64)
