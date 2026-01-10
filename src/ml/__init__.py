"""
Machine Learning module for Premier League match prediction.

This module provides:
- Feature engineering from raw football statistics
- Data preprocessing for ML models
- Baseline and advanced model implementations
- Ensemble methods for robust predictions
- Hyperparameter tuning with regularization
"""

from .feature_engineering import FeatureEngineer
from .data_pipeline import MLDataPipeline
from .models import (
    BaselineModel,
    XGBoostModel,
    LightGBMModel,
    NeuralNetworkModel,
    EnsemblePredictor,
)
from .tuning import (
    TuningConfig,
    tune_all_models,
    get_regularized_rf_params,
    get_regularized_xgboost_params,
    get_regularized_lightgbm_params,
)

__all__ = [
    "FeatureEngineer",
    "MLDataPipeline",
    "BaselineModel",
    "XGBoostModel",
    "LightGBMModel",
    "NeuralNetworkModel",
    "EnsemblePredictor",
    "TuningConfig",
    "tune_all_models",
    "get_regularized_rf_params",
    "get_regularized_xgboost_params",
    "get_regularized_lightgbm_params",
]
