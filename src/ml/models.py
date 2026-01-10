"""
Machine Learning Models for Premier League Match Prediction.

This module provides a collection of ML models from baseline to advanced:

1. BaselineModel - Simple heuristics (ELO-based)
2. LogisticRegressionModel - Linear baseline
3. XGBoostModel - Gradient boosting
4. LightGBMModel - Fast gradient boosting
5. NeuralNetworkModel - Deep learning approach
6. EnsemblePredictor - Combines multiple models

All models follow a consistent interface for easy comparison and ensembling.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error
)
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    
    # Common settings
    random_state: int = 42
    n_jobs: int = -1
    
    # Training settings
    early_stopping_rounds: int = 50
    validation_fraction: float = 0.1
    
    # Model-specific hyperparameters
    hyperparams: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Container for model predictions."""
    
    # Classification results
    class_predictions: np.ndarray  # Predicted classes
    class_probabilities: np.ndarray  # Probability for each class
    
    # Score predictions (optional)
    home_goals_pred: Optional[np.ndarray] = None
    away_goals_pred: Optional[np.ndarray] = None
    
    # Confidence metrics
    prediction_confidence: Optional[np.ndarray] = None
    
    def get_home_win_prob(self) -> np.ndarray:
        """Get probability of home win (class 2)."""
        return self.class_probabilities[:, 2]
    
    def get_draw_prob(self) -> np.ndarray:
        """Get probability of draw (class 1)."""
        return self.class_probabilities[:, 1]
    
    def get_away_win_prob(self) -> np.ndarray:
        """Get probability of away win (class 0)."""
        return self.class_probabilities[:, 0]


class BaseMatchPredictor(ABC):
    """
    Abstract base class for match prediction models.
    
    All models should implement:
    - fit(): Train the model
    - predict(): Make predictions
    - predict_proba(): Get probability estimates
    - evaluate(): Calculate performance metrics
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model: Any = None
        self.is_fitted: bool = False
        self.feature_names: List[str] = []
        self.classes_: np.ndarray = np.array([0, 1, 2])  # Away, Draw, Home
        
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'BaseMatchPredictor':
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict match outcomes."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for each outcome."""
        pass
    
    def predict_full(self, X: pd.DataFrame) -> PredictionResult:
        """Make full predictions with probabilities."""
        classes = self.predict(X)
        probas = self.predict_proba(X)
        
        # Confidence is the probability of the predicted class
        confidence = np.max(probas, axis=1)
        
        return PredictionResult(
            class_predictions=classes,
            class_probabilities=probas,
            prediction_confidence=confidence
        )
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Returns:
            Dictionary with various performance metrics
        """
        predictions = self.predict(X)
        probas = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'log_loss': log_loss(y, probas, labels=[0, 1, 2]),
        }
        
        if detailed:
            metrics['precision_macro'] = precision_score(y, predictions, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(y, predictions, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(y, predictions, average='macro', zero_division=0)
            
            # Per-class metrics
            for i, class_name in enumerate(['away_win', 'draw', 'home_win']):
                binary_y = (y == i).astype(int)
                binary_pred = (predictions == i).astype(int)
                
                metrics[f'{class_name}_precision'] = precision_score(binary_y, binary_pred, zero_division=0)
                metrics[f'{class_name}_recall'] = recall_score(binary_y, binary_pred, zero_division=0)
                metrics[f'{class_name}_f1'] = f1_score(binary_y, binary_pred, zero_division=0)
                
                # AUC for this class
                if len(np.unique(binary_y)) > 1:
                    metrics[f'{class_name}_auc'] = roc_auc_score(binary_y, probas[:, i])
                    
            # Confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(y, predictions).tolist()
            
            # Classification report
            metrics['classification_report'] = classification_report(
                y, predictions, 
                target_names=['Away Win', 'Draw', 'Home Win'],
                output_dict=True,
                zero_division=0
            )
            
        return metrics
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance scores."""
        if not hasattr(self.model, 'feature_importances_'):
            return None
            
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.config,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted,
            }, f)
            
        logger.info(f"Model saved to {path}")
        
    def load(self, path: str) -> 'BaseMatchPredictor':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        self.model = data['model']
        self.config = data['config']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        
        logger.info(f"Model loaded from {path}")
        return self


class BaselineModel(BaseMatchPredictor):
    """
    Baseline model using simple heuristics.
    
    Uses ELO ratings and home advantage as primary predictors.
    Good benchmark to beat with more complex models.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self.home_advantage_factor: float = 0.1
        self.elo_strength: float = 400.0
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'BaselineModel':
        """Fit baseline model (learns home advantage from data)."""
        
        self.feature_names = list(X.columns)
        
        # Learn home advantage from training data
        if y is not None:
            home_win_rate = (y == 2).mean()
            away_win_rate = (y == 0).mean()
            self.home_advantage_factor = home_win_rate - away_win_rate
            
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict match outcomes."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using ELO ratings.
        
        If ELO columns not available, uses simple home advantage.
        """
        n_samples = len(X)
        probas = np.zeros((n_samples, 3))
        
        # Check for ELO columns
        elo_diff_col = None
        for col in X.columns:
            if 'elo_diff' in col.lower():
                elo_diff_col = col
                break
                
        if elo_diff_col and elo_diff_col in X.columns:
            # Use ELO-based predictions
            elo_diff = X[elo_diff_col].values
            
            # Convert ELO diff to probabilities
            home_win_prob = 1 / (1 + 10 ** (-elo_diff / self.elo_strength))
            away_win_prob = 1 - home_win_prob
            
            # Adjust for draws (empirically ~25% of matches)
            draw_base = 0.25
            
            probas[:, 2] = home_win_prob * (1 - draw_base) + self.home_advantage_factor * 0.5
            probas[:, 0] = away_win_prob * (1 - draw_base) - self.home_advantage_factor * 0.5
            probas[:, 1] = draw_base
            
        else:
            # Simple home advantage baseline
            probas[:, 2] = 0.45  # Home win
            probas[:, 1] = 0.27  # Draw
            probas[:, 0] = 0.28  # Away win
            
        # Ensure valid probabilities
        probas = np.clip(probas, 0.01, 0.99)
        probas = probas / probas.sum(axis=1, keepdims=True)
        
        return probas


class LogisticRegressionModel(BaseMatchPredictor):
    """
    Logistic Regression model for match prediction.
    
    Good linear baseline that's fast and interpretable.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        
        default_params = {
            'max_iter': 1000,
            'solver': 'lbfgs',
            'class_weight': 'balanced',
        }
        default_params.update(self.config.hyperparams)
        
        self.model = LogisticRegression(
            random_state=self.config.random_state,
            **default_params
        )
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'LogisticRegressionModel':
        """Fit logistic regression model."""
        
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature coefficients as importance."""
        if not self.is_fitted:
            return pd.DataFrame()
            
        # Average absolute coefficients across classes
        coef_importance = np.abs(self.model.coef_).mean(axis=0)
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': coef_importance
        }).sort_values('importance', ascending=False)


class RandomForestModel(BaseMatchPredictor):
    """
    Random Forest model for match prediction.
    
    Robust ensemble method that handles non-linear relationships.
    
    Regularization Strategy:
    - max_depth=7: Prevents very deep trees that memorize training data
    - min_samples_split=20: Requires substantial data for any split
    - min_samples_leaf=10: Ensures leaves have enough samples for stable predictions
    - max_features='sqrt': Limits features per split, reducing variance
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        
        # Well-regularized defaults to prevent overfitting
        default_params = {
            'n_estimators': 200,
            'max_depth': 7,              # Shallower than default, prevents overfitting
            'min_samples_split': 20,     # Higher = more regularization
            'min_samples_leaf': 10,      # Higher = more regularization
            'max_features': 'sqrt',      # Limits features per split
            'class_weight': 'balanced',
            'bootstrap': True,
            'oob_score': True,           # Out-of-bag score for validation
        }
        default_params.update(self.config.hyperparams)
        
        self.model = RandomForestClassifier(
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            **default_params
        )
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'RandomForestModel':
        """Fit Random Forest model."""
        
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class XGBoostModel(BaseMatchPredictor):
    """
    XGBoost model for match prediction.
    
    State-of-the-art gradient boosting with regularization.
    Handles missing values and provides feature importance.
    
    Regularization Strategy:
    - max_depth=4: Very shallow trees (each tree is weak learner)
    - learning_rate=0.05: Small steps require more trees but generalize better
    - reg_alpha=0.1 (L1) + reg_lambda=1.0 (L2): Explicit regularization
    - subsample=0.8, colsample_bytree=0.7: Stochastic regularization
    - min_child_weight=5: Requires minimum samples per leaf
    - gamma=0.1: Minimum loss reduction for split
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self._xgb_available = False
        
        try:
            import xgboost as xgb
            self._xgb_available = True
            
            # Well-regularized defaults to prevent overfitting
            default_params = {
                'n_estimators': 300,
                'max_depth': 4,             # Shallow trees
                'learning_rate': 0.05,      # Small learning rate
                'min_child_weight': 5,      # Regularization: min samples per leaf
                'subsample': 0.8,           # Row sampling
                'colsample_bytree': 0.7,    # Column sampling per tree
                'reg_alpha': 0.1,           # L1 regularization
                'reg_lambda': 1.0,          # L2 regularization
                'gamma': 0.1,               # Min loss reduction for split
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
            }
            default_params.update(self.config.hyperparams)
            
            self.model = xgb.XGBClassifier(
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                **default_params
            )
        except ImportError:
            logger.warning("XGBoost not installed, falling back to GradientBoosting")
            self.model = GradientBoostingClassifier(
                random_state=self.config.random_state,
                n_estimators=200,
                max_depth=5,
            )
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'XGBoostModel':
        """Fit XGBoost model with optional early stopping."""
        
        self.feature_names = list(X.columns)
        
        if self._xgb_available and X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X, y)
            
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class LightGBMModel(BaseMatchPredictor):
    """
    LightGBM model for match prediction.
    
    Fast gradient boosting that handles categorical features.
    Often faster than XGBoost with similar accuracy.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self._lgb_available = False
        
        try:
            import lightgbm as lgb
            self._lgb_available = True
            
            # Well-regularized defaults to prevent overfitting
            # Similar strategy to XGBoost but with LightGBM-specific params
            default_params = {
                'n_estimators': 300,
                'max_depth': 4,             # Shallow trees
                'learning_rate': 0.05,      # Small learning rate
                'num_leaves': 31,           # Default, controls complexity
                'min_child_samples': 20,    # Regularization: min samples per leaf
                'subsample': 0.8,           # Row sampling
                'colsample_bytree': 0.7,    # Column sampling per tree
                'reg_alpha': 0.1,           # L1 regularization
                'reg_lambda': 1.0,          # L2 regularization
                'min_split_gain': 0.01,     # Min gain to make a split
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'verbose': -1,
            }
            default_params.update(self.config.hyperparams)
            
            self.model = lgb.LGBMClassifier(
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                **default_params
            )
        except ImportError:
            logger.warning("LightGBM not installed, falling back to GradientBoosting")
            self.model = GradientBoostingClassifier(
                random_state=self.config.random_state,
                n_estimators=200,
                max_depth=5,
            )
            
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'LightGBMModel':
        """Fit LightGBM model with optional early stopping."""
        
        self.feature_names = list(X.columns)
        
        if self._lgb_available and X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
            )
        else:
            self.model.fit(X, y)
            
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class NeuralNetworkModel(BaseMatchPredictor):
    """
    Neural Network model for match prediction.
    
    Deep learning approach using PyTorch or sklearn MLPClassifier.
    Can capture complex non-linear patterns.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        
        self._pytorch_available = False
        try:
            import torch
            import torch.nn as nn
            self._pytorch_available = True
        except ImportError:
            pass
            
        # Fallback to sklearn MLP
        from sklearn.neural_network import MLPClassifier
        
        default_params = {
            'hidden_layer_sizes': (128, 64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'batch_size': 32,
            'learning_rate': 'adaptive',
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.1,
        }
        default_params.update(self.config.hyperparams)
        
        self.model = MLPClassifier(
            random_state=self.config.random_state,
            **default_params
        )
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'NeuralNetworkModel':
        """Fit neural network model."""
        
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class EnsemblePredictor(BaseMatchPredictor):
    """
    Ensemble predictor that combines multiple models.
    
    Methods:
    - voting: Average probabilities from all models
    - weighted: Weighted average based on validation performance
    - stacking: Meta-learner on top of base predictions
    """
    
    def __init__(
        self,
        models: Optional[List[BaseMatchPredictor]] = None,
        weights: Optional[List[float]] = None,
        method: str = 'weighted',
        config: Optional[ModelConfig] = None
    ):
        super().__init__(config)
        
        self.models = models or []
        self.weights = weights
        self.method = method  # 'voting', 'weighted', 'stacking'
        self.meta_model: Optional[BaseMatchPredictor] = None
        self.model_performances: Dict[str, float] = {}
        
    def add_model(
        self,
        model: BaseMatchPredictor,
        name: Optional[str] = None
    ) -> 'EnsemblePredictor':
        """Add a model to the ensemble."""
        self.models.append(model)
        return self
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'EnsemblePredictor':
        """Fit all models in the ensemble."""
        
        self.feature_names = list(X.columns)
        
        # Fit each model
        for i, model in enumerate(self.models):
            logger.info(f"Fitting model {i+1}/{len(self.models)}: {type(model).__name__}")
            model.fit(X, y, X_val, y_val)
            
            # Evaluate on validation set for weighting
            if X_val is not None and y_val is not None:
                metrics = model.evaluate(X_val, y_val, detailed=False)
                self.model_performances[type(model).__name__] = metrics['accuracy']
                
        # Calculate weights based on validation performance
        if self.method == 'weighted' and self.weights is None and self.model_performances:
            accuracies = list(self.model_performances.values())
            # Weight proportional to accuracy
            total = sum(accuracies)
            self.weights = [acc / total for acc in accuracies]
            
        # For stacking, train meta-learner
        if self.method == 'stacking' and X_val is not None and y_val is not None:
            self._fit_meta_learner(X, y, X_val, y_val)
            
        self.is_fitted = True
        return self
    
    def _fit_meta_learner(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> None:
        """Fit meta-learner for stacking ensemble."""
        
        # Get base model predictions on validation set
        meta_features = []
        for model in self.models:
            probas = model.predict_proba(X_val)
            meta_features.append(probas)
            
        meta_X = np.hstack(meta_features)
        
        # Train meta-learner
        self.meta_model = LogisticRegression(
            random_state=self.config.random_state,
            max_iter=1000
        )
        self.meta_model.fit(meta_X, y_val)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using ensemble."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions from ensemble."""
        
        if not self.models:
            raise ValueError("No models in ensemble")
            
        # Get predictions from all models
        all_probas = []
        for model in self.models:
            probas = model.predict_proba(X)
            all_probas.append(probas)
            
        if self.method == 'stacking' and self.meta_model is not None:
            # Use meta-learner
            meta_X = np.hstack(all_probas)
            return self.meta_model.predict_proba(meta_X)
            
        elif self.method == 'weighted' and self.weights is not None:
            # Weighted average
            weighted_probas = np.zeros_like(all_probas[0])
            for proba, weight in zip(all_probas, self.weights):
                weighted_probas += proba * weight
            # Normalize to ensure probabilities sum to 1
            weighted_probas = weighted_probas / weighted_probas.sum(axis=1, keepdims=True)
            return weighted_probas
            
        else:
            # Simple voting (average)
            avg_probas = np.mean(all_probas, axis=0)
            # Normalize to ensure probabilities sum to 1
            avg_probas = avg_probas / avg_probas.sum(axis=1, keepdims=True)
            return avg_probas
    
    def get_model_contributions(self) -> pd.DataFrame:
        """Get contribution of each model to ensemble."""
        return pd.DataFrame({
            'model': [type(m).__name__ for m in self.models],
            'weight': self.weights if self.weights else [1/len(self.models)] * len(self.models),
            'validation_accuracy': [
                self.model_performances.get(type(m).__name__, np.nan)
                for m in self.models
            ]
        })
    
    def save(self, path: str) -> None:
        """Save ensemble model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'weights': self.weights,
                'method': self.method,
                'meta_model': self.meta_model,
                'model_performances': self.model_performances,
                'config': self.config,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted,
            }, f)
            
        logger.info(f"Ensemble model saved to {path}")
        
    def load(self, path: str) -> 'EnsemblePredictor':
        """Load ensemble model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        self.models = data['models']
        self.weights = data.get('weights')
        self.method = data.get('method', 'weighted')
        self.meta_model = data.get('meta_model')
        self.model_performances = data.get('model_performances', {})
        self.config = data['config']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        
        logger.info(f"Ensemble model loaded from {path}")
        return self


class ScorePredictor:
    """
    Predict exact match scores using Poisson regression.
    
    Football scores follow Poisson distribution, so we model
    expected goals for each team using regression and then
    compute score probabilities using Poisson PMF.
    
    The Poisson distribution is ideal for football because:
    1. Goals are discrete events (0, 1, 2, ...)
    2. Goals are relatively rare (average ~1.5 per team per match)
    3. Goals can be assumed roughly independent
    
    Key outputs:
    - Expected goals (λ) for home and away teams
    - Full probability matrix for all score combinations
    - Most likely score
    - Match outcome probabilities derived from scores
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.home_model: Any = None
        self.away_model: Any = None
        self.is_fitted: bool = False
        self.feature_names: List[str] = []
        
        # Historical averages for calibration
        self.avg_home_goals: float = 1.5
        self.avg_away_goals: float = 1.2
        
    def fit(
        self,
        X: pd.DataFrame,
        home_goals: pd.Series,
        away_goals: pd.Series
    ) -> 'ScorePredictor':
        """
        Fit score prediction models.
        
        Uses Gradient Boosting to predict expected goals (lambda parameter
        for Poisson distribution) based on features.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        self.feature_names = list(X.columns)
        
        # Store historical averages for calibration
        self.avg_home_goals = home_goals.mean()
        self.avg_away_goals = away_goals.mean()
        
        # Use Poisson-appropriate loss (least squares on log-transformed targets works well)
        self.home_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=10,
            random_state=self.config.random_state
        )
        self.away_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=10,
            random_state=self.config.random_state
        )
        
        self.home_model.fit(X, home_goals)
        self.away_model.fit(X, away_goals)
        
        self.is_fitted = True
        logger.info(f"ScorePredictor fitted. Avg home goals: {self.avg_home_goals:.2f}, Avg away goals: {self.avg_away_goals:.2f}")
        return self
    
    def predict_expected_goals(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict expected goals (lambda) for home and away teams.
        
        Returns:
            Tuple of (home_xg, away_xg) arrays
        """
        home_xg = self.home_model.predict(X)
        away_xg = self.away_model.predict(X)
        
        # Clip to reasonable range (0.3 to 4.0 goals expected)
        home_xg = np.clip(home_xg, 0.3, 4.0)
        away_xg = np.clip(away_xg, 0.3, 4.0)
        
        return home_xg, away_xg
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Alias for predict_expected_goals for compatibility."""
        return self.predict_expected_goals(X)
    
    def predict_score_distribution(
        self,
        X: pd.DataFrame,
        max_goals: int = 7
    ) -> np.ndarray:
        """
        Predict probability distribution over all possible scores.
        
        Uses Poisson distribution: P(X=k) = (λ^k * e^(-λ)) / k!
        
        Assumes home and away goals are independent (reasonable approximation).
        
        Args:
            X: Feature DataFrame
            max_goals: Maximum goals to consider per team
            
        Returns:
            Array of shape (n_samples, max_goals+1, max_goals+1)
            where result[i, h, a] = P(home=h, away=a | match i)
        """
        from scipy.stats import poisson
        
        home_xg, away_xg = self.predict_expected_goals(X)
        n_samples = len(X)
        
        score_probs = np.zeros((n_samples, max_goals + 1, max_goals + 1))
        
        for i in range(n_samples):
            # Pre-compute Poisson PMFs for efficiency
            home_pmf = poisson.pmf(np.arange(max_goals + 1), home_xg[i])
            away_pmf = poisson.pmf(np.arange(max_goals + 1), away_xg[i])
            
            # Outer product gives joint probability (assuming independence)
            score_probs[i] = np.outer(home_pmf, away_pmf)
                    
        return score_probs
    
    def predict_match_outcome_from_scores(
        self,
        X: pd.DataFrame,
        max_goals: int = 7
    ) -> np.ndarray:
        """
        Derive match outcome probabilities from score distribution.
        
        Returns:
            Array of shape (n_samples, 3) with [P(away), P(draw), P(home)]
        """
        score_probs = self.predict_score_distribution(X, max_goals)
        n_samples = len(X)
        
        outcome_probs = np.zeros((n_samples, 3))
        
        for i in range(n_samples):
            # Home win: home > away (upper triangle)
            home_win = np.sum(np.triu(score_probs[i], k=1))
            # Draw: home == away (diagonal)
            draw = np.sum(np.diag(score_probs[i]))
            # Away win: home < away (lower triangle)
            away_win = np.sum(np.tril(score_probs[i], k=-1))
            
            outcome_probs[i] = [away_win, draw, home_win]
            
        return outcome_probs
    
    def predict_most_likely_score(
        self,
        X: pd.DataFrame,
        max_goals: int = 7
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict most likely exact score.
        
        Returns:
            Tuple of (home_scores, away_scores, probabilities)
        """
        score_probs = self.predict_score_distribution(X, max_goals)
        
        home_scores = []
        away_scores = []
        probabilities = []
        
        for i in range(len(X)):
            idx = np.unravel_index(np.argmax(score_probs[i]), score_probs[i].shape)
            home_scores.append(idx[0])
            away_scores.append(idx[1])
            probabilities.append(score_probs[i][idx])
            
        return np.array(home_scores), np.array(away_scores), np.array(probabilities)
    
    def predict_top_scores(
        self,
        X: pd.DataFrame,
        n_top: int = 5,
        max_goals: int = 7
    ) -> List[List[Tuple[int, int, float]]]:
        """
        Get top N most likely scores for each match.
        
        Returns:
            List of lists, where each inner list contains tuples of
            (home_goals, away_goals, probability) sorted by probability
        """
        score_probs = self.predict_score_distribution(X, max_goals)
        
        results = []
        for i in range(len(X)):
            # Flatten and get top indices
            flat_probs = score_probs[i].flatten()
            top_indices = np.argsort(flat_probs)[::-1][:n_top]
            
            match_scores = []
            for idx in top_indices:
                h, a = np.unravel_index(idx, score_probs[i].shape)
                prob = score_probs[i, h, a]
                match_scores.append((int(h), int(a), float(prob)))
            
            results.append(match_scores)
            
        return results
    
    def predict_over_under(
        self,
        X: pd.DataFrame,
        threshold: float = 2.5,
        max_goals: int = 7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict over/under probabilities for total goals.
        
        Args:
            X: Features
            threshold: Goals threshold (e.g., 2.5 for over 2.5 goals)
            max_goals: Maximum goals to consider
            
        Returns:
            Tuple of (under_probs, over_probs)
        """
        score_probs = self.predict_score_distribution(X, max_goals)
        n_samples = len(X)
        
        under_probs = np.zeros(n_samples)
        over_probs = np.zeros(n_samples)
        
        threshold_int = int(threshold)
        
        for i in range(n_samples):
            for h in range(max_goals + 1):
                for a in range(max_goals + 1):
                    total = h + a
                    if total <= threshold_int:
                        under_probs[i] += score_probs[i, h, a]
                    else:
                        over_probs[i] += score_probs[i, h, a]
                        
        return under_probs, over_probs
    
    def predict_btts(
        self,
        X: pd.DataFrame,
        max_goals: int = 7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict Both Teams To Score (BTTS) probabilities.
        
        Returns:
            Tuple of (no_btts_probs, btts_probs)
        """
        score_probs = self.predict_score_distribution(X, max_goals)
        n_samples = len(X)
        
        btts_probs = np.zeros(n_samples)
        
        for i in range(n_samples):
            # BTTS: both teams score at least 1
            # Sum probabilities where h >= 1 AND a >= 1
            btts_probs[i] = np.sum(score_probs[i, 1:, 1:])
            
        return 1 - btts_probs, btts_probs
    
    def get_score_summary(
        self,
        X: pd.DataFrame,
        max_goals: int = 7
    ) -> pd.DataFrame:
        """
        Get comprehensive score prediction summary.
        
        Returns DataFrame with expected goals, most likely scores, 
        outcome probabilities, and betting market predictions.
        """
        home_xg, away_xg = self.predict_expected_goals(X)
        home_score, away_score, score_prob = self.predict_most_likely_score(X, max_goals)
        outcome_probs = self.predict_match_outcome_from_scores(X, max_goals)
        under_25, over_25 = self.predict_over_under(X, 2.5, max_goals)
        no_btts, btts = self.predict_btts(X, max_goals)
        
        return pd.DataFrame({
            'home_xg': home_xg,
            'away_xg': away_xg,
            'total_xg': home_xg + away_xg,
            'predicted_home_goals': home_score,
            'predicted_away_goals': away_score,
            'score_probability': score_prob,
            'away_win_prob': outcome_probs[:, 0],
            'draw_prob': outcome_probs[:, 1],
            'home_win_prob': outcome_probs[:, 2],
            'under_2.5_prob': under_25,
            'over_2.5_prob': over_25,
            'btts_no_prob': no_btts,
            'btts_yes_prob': btts,
        })
    
    def evaluate(
        self,
        X: pd.DataFrame,
        home_goals: pd.Series,
        away_goals: pd.Series
    ) -> Dict[str, float]:
        """Evaluate score predictions with multiple metrics."""
        
        pred_home, pred_away = self.predict_expected_goals(X)
        pred_home_score, pred_away_score, _ = self.predict_most_likely_score(X)
        outcome_probs = self.predict_match_outcome_from_scores(X)
        
        # Calculate actual outcomes
        actual_outcomes = np.where(
            home_goals > away_goals, 2,
            np.where(home_goals < away_goals, 0, 1)
        )
        predicted_outcomes = np.argmax(outcome_probs, axis=1)
        
        # Exact score accuracy
        exact_score_correct = np.sum(
            (pred_home_score == home_goals.values) & 
            (pred_away_score == away_goals.values)
        ) / len(X)
        
        return {
            'home_goals_mae': mean_absolute_error(home_goals, pred_home),
            'away_goals_mae': mean_absolute_error(away_goals, pred_away),
            'home_goals_rmse': np.sqrt(mean_squared_error(home_goals, pred_home)),
            'away_goals_rmse': np.sqrt(mean_squared_error(away_goals, pred_away)),
            'exact_score_accuracy': exact_score_correct,
            'outcome_accuracy': accuracy_score(actual_outcomes, predicted_outcomes),
        }
    
    def save(self, path: str) -> None:
        """Save score predictor to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'home_model': self.home_model,
                'away_model': self.away_model,
                'feature_names': self.feature_names,
                'avg_home_goals': self.avg_home_goals,
                'avg_away_goals': self.avg_away_goals,
                'is_fitted': self.is_fitted,
                'config': self.config,
            }, f)
        
        logger.info(f"ScorePredictor saved to {path}")
    
    def load(self, path: str) -> 'ScorePredictor':
        """Load score predictor from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.home_model = data['home_model']
        self.away_model = data['away_model']
        self.feature_names = data['feature_names']
        self.avg_home_goals = data.get('avg_home_goals', 1.5)
        self.avg_away_goals = data.get('avg_away_goals', 1.2)
        self.is_fitted = data['is_fitted']
        self.config = data.get('config', ModelConfig())
        
        logger.info(f"ScorePredictor loaded from {path}")
        return self


def create_default_ensemble() -> EnsemblePredictor:
    """Create a default ensemble with multiple model types."""
    
    config = ModelConfig()
    
    ensemble = EnsemblePredictor(
        method='weighted',
        config=config
    )
    
    # Add diverse models
    ensemble.add_model(LogisticRegressionModel(config))
    ensemble.add_model(RandomForestModel(config))
    ensemble.add_model(XGBoostModel(config))
    ensemble.add_model(LightGBMModel(config))
    
    return ensemble


def create_full_predictor() -> Tuple[EnsemblePredictor, ScorePredictor]:
    """
    Create complete prediction system.
    
    Returns ensemble for match result and score predictor for exact scores.
    """
    ensemble = create_default_ensemble()
    score_predictor = ScorePredictor()
    
    return ensemble, score_predictor
