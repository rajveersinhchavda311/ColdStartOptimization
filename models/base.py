"""
Abstract Base Class for All Phase 1 Baseline Models
=====================================================

Design contract:
    1. fit(train_df) is called once with the training split.
    2. predict(df) returns predictions using ONLY lag columns.
       The 'concurrency' column must NEVER be accessed during prediction.
    3. All models are stateless after fit() — they do not update parameters
       during evaluation.

This base class enforces a consistent interface so the evaluation framework
can treat all models identically.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all provisioning models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name (e.g., 'Reactive')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description of the model's strategy."""
        pass

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, **kwargs) -> None:
        """
        Train or configure the model using training data only.

        For rule-based models, this may simply extract statistics (e.g., P90).
        For learned models (TCN), this performs gradient-based training.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training split with columns: timestamp, concurrency, lag_1..lag_10
        **kwargs : dict
            Optional additional parameters (e.g., val_df for early stopping)
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for each row in df.

        CRITICAL: This method must use ONLY the lag_* columns.
        It must NEVER access the 'concurrency' column.
        Violating this constraint constitutes data leakage.

        Parameters
        ----------
        df : pd.DataFrame
            Data split with columns: timestamp, concurrency, lag_1..lag_10

        Returns
        -------
        np.ndarray
            Predicted concurrency for each timestep, shape (len(df),)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.name}: {self.description}"
