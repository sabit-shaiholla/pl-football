"""
Hyperparameter tuning module for Premier League match prediction.

This module provides hyperparameter optimization using cross-validation
with proper regularization to prevent overfitting.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
    GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform, randint, loguniform

logger = logging.getLogger(__name__)


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    
    # Cross-validation settings
    n_folds: int = 5
    scoring: str = 'accuracy'  # 'accuracy', 'neg_log_loss', 'f1_macro'
    
    # Search settings
    n_iter: int = 50  # Number of random search iterations
    random_state: int = 42
    n_jobs: int = -1  # Use all cores
    
    # Early stopping
    early_stopping_rounds: int = 50
    
    # Verbose
    verbose: int = 1


# ============================================================================
# HYPERPARAMETER SEARCH SPACES
# ============================================================================

def get_logistic_search_space() -> Dict[str, Any]:
    """
    Logistic Regression hyperparameter search space.
    
    Key regularization parameters:
    - C: Inverse of regularization strength (lower = more regularization)
    - penalty: L1 (sparse), L2 (ridge), or elasticnet
    """
    return {
        'C': loguniform(1e-4, 10),  # Regularization strength
        'penalty': ['l2'],  # L2 works well with lbfgs
        'solver': ['lbfgs'],
        'max_iter': [1000, 2000],
        'class_weight': ['balanced', None],
    }


def get_random_forest_search_space() -> Dict[str, Any]:
    """
    Random Forest hyperparameter search space.
    
    Key regularization parameters:
    - max_depth: Limits tree depth (prevents overfitting)
    - min_samples_split/leaf: Requires minimum samples (regularization)
    - max_features: Limits features per split (reduces variance)
    - n_estimators: More trees = more stable (but diminishing returns)
    """
    return {
        'n_estimators': randint(50, 300),
        'max_depth': [3, 5, 7, 10, 15, None],  # Lower = more regularization
        'min_samples_split': randint(5, 50),   # Higher = more regularization
        'min_samples_leaf': randint(2, 20),    # Higher = more regularization
        'max_features': ['sqrt', 'log2', 0.3, 0.5],  # Lower = more regularization
        'class_weight': ['balanced', 'balanced_subsample', None],
        'bootstrap': [True],
        'oob_score': [True],
    }


def get_xgboost_search_space() -> Dict[str, Any]:
    """
    XGBoost hyperparameter search space.
    
    Key regularization parameters:
    - max_depth: Tree depth limit
    - learning_rate: Lower = more regularization (needs more trees)
    - reg_alpha (L1) / reg_lambda (L2): Explicit regularization
    - subsample / colsample_bytree: Stochastic regularization
    - min_child_weight: Minimum sum of instance weight in child
    """
    return {
        'n_estimators': randint(50, 500),
        'max_depth': randint(2, 8),              # Shallow trees prevent overfitting
        'learning_rate': loguniform(0.01, 0.3),  # Lower = more regularization
        'min_child_weight': randint(1, 10),      # Higher = more regularization
        'subsample': uniform(0.6, 0.4),          # 0.6-1.0, stochastic regularization
        'colsample_bytree': uniform(0.5, 0.5),   # 0.5-1.0, feature sampling
        'reg_alpha': loguniform(1e-4, 10),       # L1 regularization
        'reg_lambda': loguniform(1e-4, 10),      # L2 regularization
        'gamma': loguniform(1e-4, 1),            # Min loss reduction for split
    }


def get_lightgbm_search_space() -> Dict[str, Any]:
    """
    LightGBM hyperparameter search space.
    
    Similar regularization concepts to XGBoost but with different naming.
    """
    return {
        'n_estimators': randint(50, 500),
        'max_depth': randint(2, 8),
        'learning_rate': loguniform(0.01, 0.3),
        'num_leaves': randint(10, 100),          # Lower = more regularization
        'min_child_samples': randint(10, 100),   # Higher = more regularization
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.5, 0.5),
        'reg_alpha': loguniform(1e-4, 10),
        'reg_lambda': loguniform(1e-4, 10),
        'min_split_gain': loguniform(1e-4, 1),
    }


# ============================================================================
# TUNING FUNCTIONS
# ============================================================================

def tune_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[TuningConfig] = None
) -> Tuple[Dict[str, Any], float]:
    """
    Tune Logistic Regression with cross-validation.
    
    Returns:
        Tuple of (best_params, best_score)
    """
    config = config or TuningConfig()
    
    logger.info("Tuning Logistic Regression...")
    
    model = LogisticRegression(random_state=config.random_state)
    search_space = get_logistic_search_space()
    
    cv = StratifiedKFold(n_splits=config.n_folds, shuffle=True, 
                         random_state=config.random_state)
    
    search = RandomizedSearchCV(
        model,
        search_space,
        n_iter=config.n_iter,
        cv=cv,
        scoring=config.scoring,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbose=config.verbose
    )
    
    search.fit(X, y)
    
    logger.info(f"Best Logistic Regression score: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")
    
    return search.best_params_, search.best_score_


def tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[TuningConfig] = None
) -> Tuple[Dict[str, Any], float]:
    """
    Tune Random Forest with cross-validation.
    """
    config = config or TuningConfig()
    
    logger.info("Tuning Random Forest...")
    
    model = RandomForestClassifier(random_state=config.random_state)
    search_space = get_random_forest_search_space()
    
    cv = StratifiedKFold(n_splits=config.n_folds, shuffle=True,
                         random_state=config.random_state)
    
    search = RandomizedSearchCV(
        model,
        search_space,
        n_iter=config.n_iter,
        cv=cv,
        scoring=config.scoring,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbose=config.verbose
    )
    
    search.fit(X, y)
    
    logger.info(f"Best Random Forest score: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")
    
    return search.best_params_, search.best_score_


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[TuningConfig] = None
) -> Tuple[Dict[str, Any], float]:
    """
    Tune XGBoost with cross-validation.
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("XGBoost not installed")
        return {}, 0.0
        
    config = config or TuningConfig()
    
    logger.info("Tuning XGBoost...")
    
    # Note: Don't use early_stopping_rounds in CV since it requires eval_set
    model = xgb.XGBClassifier(
        random_state=config.random_state,
        eval_metric='mlogloss',
    )
    search_space = get_xgboost_search_space()
    
    cv = StratifiedKFold(n_splits=config.n_folds, shuffle=True,
                         random_state=config.random_state)
    
    search = RandomizedSearchCV(
        model,
        search_space,
        n_iter=config.n_iter,
        cv=cv,
        scoring=config.scoring,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbose=config.verbose
    )
    
    search.fit(X, y)
    
    logger.info(f"Best XGBoost score: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")
    
    return search.best_params_, search.best_score_


def tune_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[TuningConfig] = None
) -> Tuple[Dict[str, Any], float]:
    """
    Tune LightGBM with cross-validation.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM not installed")
        return {}, 0.0
        
    config = config or TuningConfig()
    
    logger.info("Tuning LightGBM...")
    
    model = lgb.LGBMClassifier(
        random_state=config.random_state,
        verbose=-1
    )
    search_space = get_lightgbm_search_space()
    
    cv = StratifiedKFold(n_splits=config.n_folds, shuffle=True,
                         random_state=config.random_state)
    
    search = RandomizedSearchCV(
        model,
        search_space,
        n_iter=config.n_iter,
        cv=cv,
        scoring=config.scoring,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbose=config.verbose
    )
    
    search.fit(X, y)
    
    logger.info(f"Best LightGBM score: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")
    
    return search.best_params_, search.best_score_


def tune_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[TuningConfig] = None
) -> Dict[str, Tuple[Dict[str, Any], float]]:
    """
    Tune all model types and return best parameters for each.
    
    Returns:
        Dictionary mapping model names to (best_params, best_score)
    """
    config = config or TuningConfig()
    
    results = {}
    
    # Tune each model type
    results['logistic'] = tune_logistic_regression(X, y, config)
    results['rf'] = tune_random_forest(X, y, config)
    results['xgboost'] = tune_xgboost(X, y, config)
    results['lightgbm'] = tune_lightgbm(X, y, config)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("TUNING RESULTS SUMMARY")
    print("=" * 60)
    for name, (params, score) in results.items():
        print(f"\n{name.upper()}: CV Score = {score:.4f}")
        print(f"  Best params: {params}")
    
    return results


# ============================================================================
# REGULARIZED MODEL DEFAULTS
# ============================================================================

def get_regularized_logistic_params() -> Dict[str, Any]:
    """
    Return well-regularized Logistic Regression parameters.
    
    These are conservative defaults that prevent overfitting.
    """
    return {
        'C': 0.1,  # Strong regularization
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'class_weight': 'balanced',
    }


def get_regularized_rf_params() -> Dict[str, Any]:
    """
    Return well-regularized Random Forest parameters.
    
    Key regularization choices:
    - max_depth=7: Prevents very deep trees
    - min_samples_split=20: Requires substantial data for splits
    - min_samples_leaf=10: Ensures leaves have enough samples
    - max_features='sqrt': Limits feature sampling
    """
    return {
        'n_estimators': 200,
        'max_depth': 7,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'bootstrap': True,
        'oob_score': True,
    }


def get_regularized_xgboost_params() -> Dict[str, Any]:
    """
    Return well-regularized XGBoost parameters.
    
    Key regularization choices:
    - max_depth=4: Shallow trees
    - learning_rate=0.05: Small steps (requires more trees)
    - reg_alpha=0.1, reg_lambda=1.0: L1 and L2 regularization
    - subsample=0.8, colsample_bytree=0.8: Stochastic sampling
    - min_child_weight=5: Requires min samples per leaf
    """
    return {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.05,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'gamma': 0.1,
        'eval_metric': 'mlogloss',
    }


def get_regularized_lightgbm_params() -> Dict[str, Any]:
    """
    Return well-regularized LightGBM parameters.
    """
    return {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_split_gain': 0.01,
        'verbose': -1,
    }


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Command-line interface for hyperparameter tuning."""
    import argparse
    import sys
    sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
    
    from src.ml.data_pipeline import MLDataPipeline, PipelineConfig
    
    parser = argparse.ArgumentParser(description='Tune ML models for match prediction')
    parser.add_argument('--target', type=str, default='match_result',
                       help='Target variable to predict')
    parser.add_argument('--n-iter', type=int, default=50,
                       help='Number of random search iterations')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'logistic', 'rf', 'xgboost', 'lightgbm'],
                       help='Model to tune')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # Load data
    pipeline_config = PipelineConfig()
    pipeline = MLDataPipeline(pipeline_config)
    train_df, val_df, test_df = pipeline.run_pipeline(
        target=args.target,
        save_processed=False
    )
    
    # Prepare data
    X_train = train_df[pipeline.feature_columns]
    y_train = train_df[args.target]
    
    # Combine train and val for tuning (CV will handle validation)
    X = pd.concat([X_train, val_df[pipeline.feature_columns]])
    y = pd.concat([y_train, val_df[args.target]])
    
    print(f"\nTuning on {len(X)} samples with {len(pipeline.feature_columns)} features")
    
    # Configure tuning
    config = TuningConfig(
        n_iter=args.n_iter,
        n_folds=args.n_folds,
        scoring='accuracy'
    )
    
    # Tune models
    if args.model == 'all':
        results = tune_all_models(X, y, config)
    else:
        tune_fn = {
            'logistic': tune_logistic_regression,
            'rf': tune_random_forest,
            'xgboost': tune_xgboost,
            'lightgbm': tune_lightgbm,
        }[args.model]
        
        best_params, best_score = tune_fn(X, y, config)
        print(f"\nBest {args.model} score: {best_score:.4f}")
        print(f"Best params: {best_params}")


if __name__ == '__main__':
    main()
