#!/usr/bin/env python3
"""
Training script for Premier League Match Prediction models.

Usage:
    python -m src.ml.train --target match_result --model ensemble
    python -m src.ml.train --target home_goals --model xgboost
    python -m src.ml.train --evaluate-all
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

from .data_pipeline import MLDataPipeline, PipelineConfig
from .feature_engineering import FeatureConfig
from .models import (
    BaselineModel,
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    NeuralNetworkModel,
    EnsemblePredictor,
    ScorePredictor,
    ModelConfig,
    create_default_ensemble,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Premier League Match Prediction models'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='match_result',
        choices=['match_result', 'home_win', 'home_goals', 'total_goals', 'over_2_5'],
        help='Target variable to predict'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='ensemble',
        choices=['baseline', 'logistic', 'rf', 'xgboost', 'lightgbm', 'nn', 'ensemble'],
        help='Model type to train'
    )
    
    parser.add_argument(
        '--seasons',
        type=str,
        nargs='+',
        default=['2023-2024', '2022-2023', '2021-2022', '2020-2021'],
        help='Seasons to include in training'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--evaluate-all',
        action='store_true',
        help='Evaluate all model types and compare'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save processed data or models'
    )
    
    return parser.parse_args()


def get_model(model_name: str, config: Optional[ModelConfig] = None):
    """Get model instance by name."""
    config = config or ModelConfig()
    
    models = {
        'baseline': BaselineModel,
        'logistic': LogisticRegressionModel,
        'rf': RandomForestModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'nn': NeuralNetworkModel,
    }
    
    if model_name == 'ensemble':
        return create_default_ensemble()
    
    return models[model_name](config)


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list,
    target: str,
    model_name: str = 'ensemble',
) -> Dict[str, Any]:
    """
    Train a model and return evaluation results.
    """
    logger.info(f"Training {model_name} model for target: {target}")
    
    # Prepare data
    X_train = train_df[feature_columns]
    y_train = train_df[target]
    X_val = val_df[feature_columns]
    y_val = val_df[target]
    X_test = test_df[feature_columns]
    y_test = test_df[target]
    
    # Get model
    model = get_model(model_name)
    
    # Train
    model.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    train_metrics = model.evaluate(X_train, y_train, detailed=False)
    val_metrics = model.evaluate(X_val, y_val, detailed=False)
    test_metrics = model.evaluate(X_test, y_test, detailed=True)
    
    results = {
        'model_name': model_name,
        'target': target,
        'train_accuracy': train_metrics['accuracy'],
        'val_accuracy': val_metrics['accuracy'],
        'test_accuracy': test_metrics['accuracy'],
        'test_log_loss': test_metrics['log_loss'],
        'test_metrics': test_metrics,
        'model': model,
    }
    
    # Get feature importance if available
    importance = model.get_feature_importance()
    if importance is not None:
        results['feature_importance'] = importance
        
    return results


def evaluate_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list,
    target: str = 'match_result',
) -> pd.DataFrame:
    """
    Evaluate all model types and compare.
    """
    model_names = ['baseline', 'logistic', 'rf', 'xgboost', 'lightgbm', 'ensemble']
    results = []
    
    for model_name in model_names:
        try:
            result = train_model(
                train_df, val_df, test_df,
                feature_columns, target, model_name
            )
            
            results.append({
                'Model': model_name,
                'Train Acc': f"{result['train_accuracy']:.3f}",
                'Val Acc': f"{result['val_accuracy']:.3f}",
                'Test Acc': f"{result['test_accuracy']:.3f}",
                'Log Loss': f"{result['test_log_loss']:.3f}",
            })
            
            logger.info(f"{model_name}: Test Accuracy = {result['test_accuracy']:.3f}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            
    return pd.DataFrame(results)


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 70)
    print("PREMIER LEAGUE MATCH PREDICTION - MODEL TRAINING")
    print("=" * 70)
    
    # Configure pipeline
    pipeline_config = PipelineConfig(
        seasons=args.seasons,
        use_time_split=True,
    )
    
    # Create and run pipeline
    logger.info("Running data pipeline...")
    pipeline = MLDataPipeline(pipeline_config)
    
    try:
        train_df, val_df, test_df = pipeline.run_pipeline(
            target=args.target,
            save_processed=not args.no_save
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
        
    feature_columns = pipeline.feature_columns
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)} matches")
    print(f"  Val:   {len(val_df)} matches")
    print(f"  Test:  {len(test_df)} matches")
    print(f"  Features: {len(feature_columns)}")
    
    # Check target distribution
    if args.target in train_df.columns:
        print(f"\nTarget distribution (train):")
        print(train_df[args.target].value_counts(normalize=True))
        
    # Train models
    if args.evaluate_all:
        print("\n" + "=" * 70)
        print("EVALUATING ALL MODELS")
        print("=" * 70)
        
        comparison = evaluate_all_models(
            train_df, val_df, test_df,
            feature_columns, args.target
        )
        
        print("\nModel Comparison:")
        print(comparison.to_string(index=False))
        
    else:
        # Train single model
        result = train_model(
            train_df, val_df, test_df,
            feature_columns, args.target, args.model
        )
        
        print("\n" + "=" * 70)
        print("TRAINING RESULTS")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Target: {args.target}")
        print(f"Train Accuracy: {result['train_accuracy']:.3f}")
        print(f"Val Accuracy:   {result['val_accuracy']:.3f}")
        print(f"Test Accuracy:  {result['test_accuracy']:.3f}")
        print(f"Test Log Loss:  {result['test_log_loss']:.3f}")
        
        # Show classification report
        if 'classification_report' in result['test_metrics']:
            print("\nClassification Report:")
            for class_name, metrics in result['test_metrics']['classification_report'].items():
                if isinstance(metrics, dict):
                    print(f"  {class_name}: precision={metrics.get('precision', 0):.2f}, "
                          f"recall={metrics.get('recall', 0):.2f}, f1={metrics.get('f1-score', 0):.2f}")
                    
        # Show feature importance
        if 'feature_importance' in result:
            print("\nTop 15 Features:")
            print(result['feature_importance'].head(15).to_string(index=False))
            
        # Save model if requested
        if not args.no_save:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / f"{args.model}_{args.target}.pkl"
            result['model'].save(str(model_path))
            print(f"\nModel saved to: {model_path}")
    
    # Also train score predictor if we have goals data
    if 'home_goals' in train_df.columns and 'away_goals' in train_df.columns:
        print("\n" + "=" * 70)
        print("TRAINING SCORE PREDICTOR (Poisson Model)")
        print("=" * 70)
        
        score_predictor = ScorePredictor()
        
        X_train = train_df[feature_columns]
        X_test = test_df[feature_columns]
        
        score_predictor.fit(
            X_train,
            train_df['home_goals'],
            train_df['away_goals']
        )
        
        # Evaluate
        score_metrics = score_predictor.evaluate(
            X_test,
            test_df['home_goals'],
            test_df['away_goals']
        )
        
        print(f"\nScore Prediction Metrics:")
        print(f"  Home Goals MAE:  {score_metrics['home_goals_mae']:.3f}")
        print(f"  Away Goals MAE:  {score_metrics['away_goals_mae']:.3f}")
        print(f"  Home Goals RMSE: {score_metrics['home_goals_rmse']:.3f}")
        print(f"  Away Goals RMSE: {score_metrics['away_goals_rmse']:.3f}")
        print(f"  Exact Score Acc: {score_metrics['exact_score_accuracy']:.1%}")
        print(f"  Outcome Accuracy: {score_metrics['outcome_accuracy']:.1%}")
        
        # Save score predictor
        if not args.no_save:
            output_dir = Path(args.output_dir)
            score_model_path = output_dir / "score_predictor.pkl"
            score_predictor.save(str(score_model_path))
            print(f"\nScore predictor saved to: {score_model_path}")
            
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
