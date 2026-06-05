"""
Risk-Aware Model Wrapper — Dynamic EVT + CVaR Provisioning
============================================================

This module implements the Phase 2 risk-aware provisioning layer.
It wraps any Phase 1 BaseModel and adds a dynamic safety buffer
derived from Extreme Value Theory (EVT) and Conditional Value-at-Risk
(CVaR).

Architecture:
    RiskAwareModel IS-A BaseModel. It wraps a Phase 1 model and
    overrides predict() to return:

        final_prediction[t] = base_prediction[t] + buffer[t]

    where:
        buffer[t] = sigma_t * CVaR_z

    sigma_t is the local rolling volatility of forecast residuals,
    and CVaR_z is a constant derived from EVT on standardized residuals.

    The buffer is DYNAMIC because sigma_t changes over time:
        - During calm periods: sigma_t is small -> small buffer
        - During volatile periods: sigma_t is large -> large buffer

Mathematical formulation:
    1. Training phase:
        - Compute residuals: epsilon = actual - base_prediction
        - Compute global volatility: sigma_train = std(epsilon)
        - Standardize: z = epsilon / sigma_train
        - Fit EVT (POT + GPD) to z -> obtain CVaR_z

    2. Inference phase (sequential):
        At timestep t:
        - base_pred[t] = base_model.predict(row_t)  [uses only lag columns]
        - Reconstruct past residual: epsilon_{t-1} = lag_1[t] - base_pred[t-1]
        - Update rolling window of residuals
        - Compute sigma_t = std(recent residuals)  [or sigma_train if warm-up]
        - buffer[t] = sigma_t * CVaR_z
        - final_pred[t] = base_pred[t] + buffer[t]

Leakage prevention:
    - The wrapper NEVER reads eval_df["concurrency"]
    - Past residuals are reconstructed via lag_1[t] = concurrency[t-1]
      which is strictly historical information available at time t
    - sigma_train is computed entirely from training data
    - CVaR_z is computed entirely from training residuals

Phase 1 compatibility:
    - The wrapped base model is completely unchanged
    - RiskAwareModel implements BaseModel, so the existing simulator
      calls model.predict() and receives the buffered predictions
    - No modifications to simulator.py, metrics.py, or extreme.py
"""

import numpy as np
import pandas as pd
from models.base import BaseModel
from evaluation.evt import fit_evt_pipeline


class RiskAwareModel(BaseModel):
    """
    Risk-aware wrapper that adds a dynamic EVT-CVaR buffer to any
    Phase 1 forecasting model.

    Parameters
    ----------
    base_model : BaseModel
        Any fitted or unfitted Phase 1 model (Reactive, ForecastOnly, TCN).
    alpha : float
        CVaR confidence level (default: 0.99).
        Higher alpha = more conservative provisioning.
    volatility_window : int
        Rolling window size for local volatility estimation (default: 30).
    evt_threshold_percentile : float
        Percentile for POT threshold on standardized residuals (default: 90).
    """

    def __init__(self, base_model: BaseModel, alpha: float = 0.99,
                 volatility_window: int = 30,
                 evt_threshold_percentile: float = 90):
        self.base_model = base_model
        self.alpha = alpha
        self.volatility_window = volatility_window
        self.evt_threshold_percentile = evt_threshold_percentile

        # Fitted parameters (set during fit)
        self.sigma_train: float = None
        self.cvar_z: float = None
        self.evt_params: dict = None

        # Diagnostics cache (set during predict)
        self._diagnostics: dict = None

    @property
    def name(self) -> str:
        return f"RiskAware({self.base_model.name})"

    @property
    def description(self) -> str:
        return (
            f"EVT-CVaR wrapper (alpha={self.alpha}, W={self.volatility_window}) "
            f"around {self.base_model.name}"
        )

    def fit(self, train_df: pd.DataFrame, **kwargs) -> None:
        """
        Fit the risk-aware model:
        1. Train the base model on training data.
        2. Compute training residuals.
        3. Compute global training volatility (sigma_train).
        4. Standardize residuals and fit EVT to obtain CVaR_z.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with columns: timestamp, concurrency, lag_1..lag_10
        **kwargs : dict
            Passed through to base model (e.g., val_df for TCN early stopping)
        """
        # Step 1: Train the base model
        print(f"  [{self.name}] Training base model: {self.base_model.name}")
        self.base_model.fit(train_df, **kwargs)

        # Step 2: Compute training residuals
        # This uses concurrency from TRAINING data (allowed during fit)
        base_preds = self.base_model.predict(train_df)
        actual = train_df["concurrency"].values.astype(np.float64)
        residuals = actual - base_preds

        print(f"  [{self.name}] Training residuals - "
              f"mean: {np.mean(residuals):,.1f}, "
              f"std: {np.std(residuals):,.1f}, "
              f"min: {np.min(residuals):,.1f}, "
              f"max: {np.max(residuals):,.1f}")

        # Step 3: Compute global training volatility
        self.sigma_train = float(np.std(residuals))
        if self.sigma_train < 1e-8:
            raise ValueError(
                f"[{self.name}] Training residual volatility is near zero "
                f"({self.sigma_train:.2e}). The base model may be trivially "
                f"perfect on training data, or produces constant predictions. "
                f"EVT cannot operate on zero-variance residuals."
            )
        print(f"  [{self.name}] sigma_train: {self.sigma_train:,.2f}")

        # Step 4: Standardize and fit EVT
        z = residuals / self.sigma_train
        print(f"  [{self.name}] Fitting EVT on {len(z):,} standardized residuals...")
        self.evt_params = fit_evt_pipeline(
            z,
            threshold_percentile=self.evt_threshold_percentile,
            alpha=self.alpha
        )
        self.cvar_z = self.evt_params["cvar_z"]
        print(f"  [{self.name}] CVaR_z = {self.cvar_z:.4f} "
              f"(buffer multiplier on sigma_t)")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate risk-aware predictions with dynamic buffer.

        CRITICAL: This method is SEQUENTIAL. At each timestep t, the
        volatility estimate uses only information available at time t.

        Information available at time t:
            - lag_1[t] = concurrency[t-1]  (past demand, from the DataFrame)
            - base_preds[t-1]              (our own past prediction)
            - All prior residuals          (lag_1[k] - base_preds[k-1] for k < t)

        This method does NOT access df["concurrency"].

        Parameters
        ----------
        df : pd.DataFrame
            Evaluation data with lag_1..lag_10 columns.

        Returns
        -------
        np.ndarray
            Risk-aware predictions: base_prediction + dynamic_buffer
        """
        if self.cvar_z is None or self.sigma_train is None:
            raise RuntimeError(
                f"{self.name}.fit() must be called before predict()"
            )

        # Step 1: Generate base predictions (uses only lag columns)
        base_preds = self.base_model.predict(df)
        n = len(df)
        lag_1 = df["lag_1"].values.astype(np.float64)

        # Step 2: Sequential volatility estimation
        sigma = np.full(n, self.sigma_train, dtype=np.float64)
        buffer = np.full(n, self.sigma_train * self.cvar_z, dtype=np.float64)

        residual_history = []

        for t in range(n):
            if t > 0:
                # Reconstruct the residual at t-1:
                #   actual[t-1] = lag_1[t]  (available from the DataFrame)
                #   predicted[t-1] = base_preds[t-1]  (our own prediction)
                past_residual = lag_1[t] - base_preds[t - 1]
                residual_history.append(past_residual)

                if len(residual_history) >= self.volatility_window:
                    window = residual_history[-self.volatility_window:]
                    sigma[t] = float(np.std(window)) + 1e-8
                # else: sigma[t] remains sigma_train (warm-up period)

                buffer[t] = sigma[t] * self.cvar_z

        # Step 3: Compute final predictions
        final_preds = base_preds + buffer

        # Step 4: Cache diagnostics for enriched output
        self._diagnostics = {
            "base_prediction": base_preds.copy(),
            "sigma_t": sigma.copy(),
            "buffer_t": buffer.copy(),
            "final_prediction": final_preds.copy(),
        }

        return final_preds.astype(np.float64)
