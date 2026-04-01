"""
Evaluation framework for serverless autoscaling research.

Core modules:
    - config: Cost parameters
    - core: Model evaluation and metric computation
    - extreme: Extreme event identification and analysis
    - pipeline: End-to-end evaluation orchestration
    - baselines: Baseline models (Reactive, Static, Forecast, TCN)
    - compare: Multi-model fair comparison
"""

from .config import COST_IDLE, COST_COLD
from .core import evaluate_model, aggregate_results, distribution_stats
from .extreme import identify_extreme_events, analyze_extreme_events
from .pipeline import run_evaluation
from .baselines import (
    ReactiveScaler,
    StaticScaler,
    ForecastOnlyScaler,
    TCNForecastScaler,
    create_baselines,
    train_tcn_scaler,
)
from .compare import compare_models, format_comparison_table, print_comparison

__all__ = [
    'COST_IDLE',
    'COST_COLD',
    'evaluate_model',
    'aggregate_results',
    'distribution_stats',
    'identify_extreme_events',
    'analyze_extreme_events',
    'run_evaluation',
    'ReactiveScaler',
    'StaticScaler',
    'ForecastOnlyScaler',
    'TCNForecastScaler',
    'create_baselines',
    'train_tcn_scaler',
    'compare_models',
    'format_comparison_table',
    'print_comparison',
]
