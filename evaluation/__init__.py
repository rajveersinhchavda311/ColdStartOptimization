"""
Evaluation framework for serverless autoscaling research.

Core modules:
    - config: Cost parameters
    - core: Model evaluation and metric computation
    - extreme: Extreme event identification and analysis
    - pipeline: End-to-end evaluation orchestration
    - baselines: Baseline models (Reactive, Static, Forecast, MLP)
    - compare: Multi-model fair comparison
"""

from .config import COST_IDLE, COST_COLD
from .core import evaluate_model, aggregate_results, distribution_stats
from .extreme import identify_extreme_events, analyze_extreme_events
from .pipeline import run_evaluation
from models.baselines import (
    ReactiveScaler,
    StaticScaler,
    ForecastOnlyScaler,
    MLPForecastScaler,
    create_baselines,
    train_mlp_scaler,
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
    'MLPForecastScaler',
    'create_baselines',
    'train_mlp_scaler',
    'compare_models',
    'format_comparison_table',
    'print_comparison',
]
