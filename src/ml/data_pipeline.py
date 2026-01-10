"""
ML Data Pipeline for Premier League Match Prediction.

This module provides the complete data pipeline from raw FBREF data
to ML-ready training and inference datasets.

Pipeline stages:
1. Load raw data from CSV files
2. Clean and standardize data
3. Engineer features
4. Create train/test splits
5. Handle missing values
6. Scale features for models
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

from .feature_engineering import FeatureEngineer, FeatureConfig

logger = logging.getLogger(__name__)


# Standard name mappings for FBREF data
TEAM_NAME_MAPPINGS = {
    "Manchester Utd": "Manchester United",
    "Newcastle Utd": "Newcastle United",
    "Nott'ham Forest": "Nottingham Forest",
    "Nottingham Forest": "Nottingham Forest",
    "Tottenham": "Tottenham Hotspur",
    "Spurs": "Tottenham Hotspur",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Brighton": "Brighton & Hove Albion",
    "Brighton and Hove Albion": "Brighton & Hove Albion",
    "Leicester": "Leicester City",
    "Leeds": "Leeds United",
    "Sheffield Utd": "Sheffield United",
    "Luton": "Luton Town",
    "Ipswich": "Ipswich Town",
}


@dataclass
class PipelineConfig:
    """Configuration for the ML data pipeline."""
    
    # Data directories
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    
    # Seasons to process
    seasons: List[str] = field(default_factory=lambda: [
        "2023-2024", "2022-2023", "2021-2022", "2020-2021"
    ])
    
    # Train/test split configuration
    test_size: float = 0.2
    validation_size: float = 0.1
    use_time_split: bool = True  # Use temporal split instead of random
    
    # Feature engineering config
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    
    # Imputation strategy: 'mean', 'median', 'knn', 'zero'
    imputation_strategy: str = 'median'
    
    # Scaling strategy: 'standard', 'minmax', 'robust', None
    scaling_strategy: str = 'standard'
    
    # Feature selection
    drop_correlated_features: bool = True
    correlation_threshold: float = 0.95
    
    # Minimum samples per class for classification
    min_class_samples: int = 50


class MLDataPipeline:
    """
    End-to-end data pipeline for ML model training.
    
    Handles:
    - Loading and combining multi-season data
    - Feature engineering integration
    - Train/validation/test splitting
    - Missing value imputation
    - Feature scaling
    - Feature selection
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.raw_data_dir = Path(self.config.raw_data_dir)
        self.processed_data_dir = Path(self.config.processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_engineer = FeatureEngineer(self.config.feature_config)
        self.scaler: Optional[Any] = None
        self.imputer: Optional[Any] = None
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        
    def run_pipeline(
        self,
        target: str = 'match_result',
        save_processed: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the complete data pipeline.
        
        Args:
            target: Target variable to predict
            save_processed: Whether to save processed data
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Starting ML data pipeline")
        
        # 1. Load raw data
        all_fixtures, all_team_stats = self._load_all_seasons()
        
        if all_fixtures.empty:
            raise ValueError("No fixture data loaded")
            
        logger.info(f"Loaded {len(all_fixtures)} fixtures across {len(self.config.seasons)} seasons")
        
        # 2. Clean data
        all_fixtures = self._clean_fixtures(all_fixtures)
        all_team_stats = self._clean_team_stats(all_team_stats)
        
        # 3. Engineer features
        featured_data = self.feature_engineer.engineer_features(
            fixtures=all_fixtures,
            team_stats=all_team_stats,
            historical_fixtures=all_fixtures  # Use all data for form calculation
        )
        
        logger.info(f"Engineered {len(featured_data.columns)} features")
        
        # 4. Prepare features and target
        X, y = self._prepare_features_target(featured_data, target)
        
        # 5. Split data
        train_data, val_data, test_data = self._split_data(featured_data, target)
        
        # 6. Handle missing values
        train_data, val_data, test_data = self._impute_missing(
            train_data, val_data, test_data
        )
        
        # 7. Scale features
        train_data, val_data, test_data = self._scale_features(
            train_data, val_data, test_data
        )
        
        # 8. Feature selection
        if self.config.drop_correlated_features:
            train_data, val_data, test_data = self._remove_correlated_features(
                train_data, val_data, test_data
            )
            
        # 9. Save if requested
        if save_processed:
            self._save_processed_data(train_data, val_data, test_data, target)
            
        logger.info(f"Pipeline complete. Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _load_all_seasons(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from all configured seasons."""
        all_fixtures = []
        all_team_stats = []
        
        for season in self.config.seasons:
            season_dir = self.raw_data_dir / season
            
            if not season_dir.exists():
                logger.warning(f"Season directory not found: {season_dir}")
                continue
                
            # Load fixtures
            fixtures_path = season_dir / "fixtures.csv"
            if fixtures_path.exists():
                fixtures = pd.read_csv(fixtures_path)
                fixtures['season'] = season
                all_fixtures.append(fixtures)
                
            # Load team stats
            team_stats = self._load_season_team_stats(season_dir, season)
            if not team_stats.empty:
                all_team_stats.append(team_stats)
                
        # Combine all seasons
        fixtures_df = pd.concat(all_fixtures, ignore_index=True) if all_fixtures else pd.DataFrame()
        team_stats_df = pd.concat(all_team_stats, ignore_index=True) if all_team_stats else pd.DataFrame()
        
        return fixtures_df, team_stats_df
    
    def _load_season_team_stats(self, season_dir: Path, season: str) -> pd.DataFrame:
        """Load and merge all team statistics for a season."""
        
        # Define stat files to load
        stat_files = {
            'standard_for': 'standard_stats_for.csv',
            'standard_against': 'standard_stats_against.csv',
            'shooting_for': 'shooting_for.csv',
            'shooting_against': 'shooting_against.csv',
            'passing_for': 'passing_for.csv',
            'passing_against': 'passing_against.csv',
            'possession_for': 'possession_for.csv',
            'possession_against': 'possession_against.csv',
            'defensive_for': 'defensive_actions_for.csv',
            'defensive_against': 'defensive_actions_against.csv',
            'gca_for': 'goal_shot_creation_for.csv',
            'gca_against': 'goal_shot_creation_against.csv',
            'goalkeeping_for': 'goalkeeping_for.csv',
            'goalkeeping_against': 'goalkeeping_against.csv',
            'misc_for': 'misc_for.csv',
            'misc_against': 'misc_against.csv',
        }
        
        base_df = None
        
        for stat_key, filename in stat_files.items():
            file_path = season_dir / filename
            if not file_path.exists():
                continue
                
            df = pd.read_csv(file_path)
            
            # Find team column
            team_col = self._find_team_column(df)
            if not team_col:
                continue
            
            # Clean team names BEFORE merging - remove "vs " prefix from against stats
            df[team_col] = df[team_col].str.replace(r'^vs\s+', '', regex=True).str.strip()
                
            # Rename columns with prefix to avoid conflicts
            prefix = stat_key.replace('_for', '_f').replace('_against', '_a')
            rename_dict = {
                col: f"{prefix}_{col}" 
                for col in df.columns 
                if col != team_col and col not in ['season', 'stat_type']
            }
            df = df.rename(columns=rename_dict)
            
            if base_df is None:
                base_df = df
                base_df = base_df.rename(columns={team_col: 'team'})
            else:
                # Merge on team
                df = df.rename(columns={team_col: 'team'})
                base_df = base_df.merge(
                    df.drop(columns=['season', 'stat_type'], errors='ignore'),
                    on='team',
                    how='outer',
                    suffixes=('', '_dup')
                )
                # Remove duplicate columns
                base_df = base_df.loc[:, ~base_df.columns.str.endswith('_dup')]
                
        if base_df is not None:
            base_df['season'] = season
            
        return base_df if base_df is not None else pd.DataFrame()
    
    def _clean_fixtures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize fixtures data."""
        df = df.copy()
        
        # Standardize column names
        col_mappings = {
            'wk': 'gameweek',
            'gameweek': 'gameweek',
            'day': 'day_of_week',
            'dayofweek': 'day_of_week',
            'date': 'date',
            'start_time': 'kickoff_time',
            'home': 'home_team',
            'home_team': 'home_team',
            'away': 'away_team',
            'away_team': 'away_team',
            'home_xg': 'home_xg',
            'xg': 'home_xg',
            'away_xg': 'away_xg',
            'xg_1': 'away_xg',
            'score': 'score',
            'attendance': 'attendance',
            'venue': 'venue',
            'referee': 'referee',
        }
        
        df.columns = df.columns.str.lower()
        df = df.rename(columns={k: v for k, v in col_mappings.items() if k in df.columns})
        
        # Parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        # Parse score
        if 'score' in df.columns and 'home_goals' not in df.columns:
            score_split = df['score'].astype(str).str.extract(r'(\d+)[–-](\d+)')
            df['home_goals'] = pd.to_numeric(score_split[0], errors='coerce')
            df['away_goals'] = pd.to_numeric(score_split[1], errors='coerce')
            
        # Clean team names
        if 'home_team' in df.columns:
            df['home_team'] = df['home_team'].replace(TEAM_NAME_MAPPINGS)
        if 'away_team' in df.columns:
            df['away_team'] = df['away_team'].replace(TEAM_NAME_MAPPINGS)
            
        # Convert xG columns to numeric
        for col in ['home_xg', 'away_xg']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Drop matches without scores (future matches)
        df = df.dropna(subset=['home_goals', 'away_goals'])
        
        return df
    
    def _clean_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize team statistics."""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Standardize column names
        df.columns = [
            col.lower().replace(' ', '_').replace('-', '_')
            for col in df.columns
        ]
        
        # Clean team names
        if 'team' in df.columns:
            df['team'] = df['team'].replace(TEAM_NAME_MAPPINGS)
            # Remove "vs " prefix from against stats
            df['team'] = df['team'].str.replace(r'^vs\s+', '', regex=True)
            
        # Convert numeric columns
        for col in df.columns:
            if col not in ['team', 'season', 'stat_type']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    
    def _prepare_features_target(
        self, 
        df: pd.DataFrame, 
        target: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix and target vector."""
        
        # Identify target columns (match outcomes we want to predict)
        target_cols = [
            'match_result', 'home_win', 'away_win', 'draw',
            'home_goals', 'away_goals', 'total_goals', 'goal_difference',
            'over_2_5', 'over_1_5', 'btts', 'goal_diff'
        ]
        self.target_columns = [c for c in target_cols if c in df.columns]
        
        # Identify non-feature columns (metadata + leaky match-day stats)
        # CRITICAL: Exclude any columns that contain match-day information
        # which would leak the outcome (e.g., actual match xG, actual score)
        non_feature_cols = [
            # Metadata columns
            'date', 'home_team', 'away_team', 'season', 'gameweek',
            'venue', 'referee', 'score', 'day_of_week', 'kickoff_time',
            'match_report', 'notes', 'attendance', 'start_time', 'dayofweek',
            'home_team_merge', 'away_team_merge',
            # Leaky match-day columns (actual match stats, not pre-match averages)
            'home_xg', 'away_xg', 'xg', 'xga',
        ] + self.target_columns
        
        # Additional leaky patterns - features derived from match outcomes
        leaky_patterns = [
            'xg_overperformance',  # uses match-level goals - xG
            'goal_conversion',     # uses match-level goals
            'shot_accuracy',       # if calculated from match data
        ]
        
        # Get feature columns (numeric only, excluding leaky columns)
        feature_cols = []
        for col in df.columns:
            if col in non_feature_cols:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            # Check for leaky patterns
            if any(pattern in col.lower() for pattern in leaky_patterns):
                continue
            feature_cols.append(col)
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df[target] if target in df.columns else None
        
        return X, y
    
    def _split_data(
        self, 
        df: pd.DataFrame, 
        target: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets."""
        
        if self.config.use_time_split:
            # Temporal split - train on older data, test on recent
            df = df.sort_values('date')
            
            n = len(df)
            train_end = int(n * (1 - self.config.test_size - self.config.validation_size))
            val_end = int(n * (1 - self.config.test_size))
            
            train_data = df.iloc[:train_end].copy()
            val_data = df.iloc[train_end:val_end].copy()
            test_data = df.iloc[val_end:].copy()
            
        else:
            # Random split with stratification
            train_val, test_data = train_test_split(
                df,
                test_size=self.config.test_size,
                stratify=df[target] if target in df.columns else None,
                random_state=42
            )
            
            val_ratio = self.config.validation_size / (1 - self.config.test_size)
            train_data, val_data = train_test_split(
                train_val,
                test_size=val_ratio,
                stratify=train_val[target] if target in train_val.columns else None,
                random_state=42
            )
            
        return train_data, val_data, test_data
    
    def _impute_missing(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Impute missing values."""
        
        # Only impute feature columns that exist in train
        feature_cols = [c for c in self.feature_columns if c in train.columns]
        
        if not feature_cols:
            return train, val, test
        
        # Remove columns that have all NaN values
        valid_cols = [c for c in feature_cols if train[c].notna().any()]
        if len(valid_cols) < len(feature_cols):
            dropped = set(feature_cols) - set(valid_cols)
            logger.warning(f"Dropping {len(dropped)} columns with all NaN values")
            self.feature_columns = [c for c in self.feature_columns if c in valid_cols]
            feature_cols = valid_cols
        
        if not feature_cols:
            return train, val, test
            
        # Choose imputer
        if self.config.imputation_strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        else:
            strategy = self.config.imputation_strategy
            if strategy == 'zero':
                self.imputer = SimpleImputer(strategy='constant', fill_value=0)
            else:
                self.imputer = SimpleImputer(strategy=strategy)
                
        # Fit on training data
        train_features = train[feature_cols].copy()
        self.imputer.fit(train_features)
        
        # Transform all sets - handle the numpy array properly
        train_imputed = self.imputer.transform(train[feature_cols])
        val_imputed = self.imputer.transform(val[feature_cols])
        test_imputed = self.imputer.transform(test[feature_cols])
        
        # Assign back to dataframes
        train = train.copy()
        val = val.copy()
        test = test.copy()
        
        for i, col in enumerate(feature_cols):
            train[col] = train_imputed[:, i]
            val[col] = val_imputed[:, i]
            test[col] = test_imputed[:, i]
        
        return train, val, test
        
        return train, val, test
    
    def _scale_features(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Scale features using specified strategy."""
        
        if self.config.scaling_strategy is None:
            return train, val, test
            
        feature_cols = [c for c in self.feature_columns if c in train.columns]
        
        if not feature_cols:
            return train, val, test
            
        # Choose scaler
        if self.config.scaling_strategy == 'standard':
            self.scaler = StandardScaler()
        elif self.config.scaling_strategy == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.config.scaling_strategy == 'robust':
            self.scaler = RobustScaler()
        else:
            return train, val, test
            
        # Fit on training data
        self.scaler.fit(train[feature_cols])
        
        # Transform all sets - handle numpy array properly
        train_scaled = self.scaler.transform(train[feature_cols])
        val_scaled = self.scaler.transform(val[feature_cols])
        test_scaled = self.scaler.transform(test[feature_cols])
        
        # Assign back to dataframes
        train = train.copy()
        val = val.copy()
        test = test.copy()
        
        for i, col in enumerate(feature_cols):
            train[col] = train_scaled[:, i]
            val[col] = val_scaled[:, i]
            test[col] = test_scaled[:, i]
        
        return train, val, test
    
    def _remove_correlated_features(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Remove highly correlated features."""
        
        feature_cols = [c for c in self.feature_columns if c in train.columns]
        
        if len(feature_cols) < 2:
            return train, val, test
            
        # Calculate correlation matrix
        corr_matrix = train[feature_cols].corr().abs()
        
        # Select upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation above threshold
        to_drop = [
            col for col in upper.columns 
            if any(upper[col] > self.config.correlation_threshold)
        ]
        
        if to_drop:
            logger.info(f"Dropping {len(to_drop)} highly correlated features")
            self.feature_columns = [c for c in self.feature_columns if c not in to_drop]
            
            train = train.drop(columns=to_drop, errors='ignore')
            val = val.drop(columns=to_drop, errors='ignore')
            test = test.drop(columns=to_drop, errors='ignore')
            
        return train, val, test
    
    def _save_processed_data(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        target: str
    ) -> None:
        """Save processed data to disk."""
        
        # Save without timestamps - overwrite each time
        train.to_csv(self.processed_data_dir / "train.csv", index=False)
        val.to_csv(self.processed_data_dir / "val.csv", index=False)
        test.to_csv(self.processed_data_dir / "test.csv", index=False)
        
        # Save feature list
        feature_info = {
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'target': target,
        }
        
        import json
        with open(self.processed_data_dir / "feature_info.json", 'w') as f:
            json.dump(feature_info, f, indent=2)
            
        logger.info(f"Saved processed data to {self.processed_data_dir}")
        
    def _find_team_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the team column in a DataFrame."""
        for col in ['team', 'squad', 'Team', 'Squad']:
            if col in df.columns:
                return col
        for col in df.columns:
            if 'team' in col.lower() or 'squad' in col.lower():
                return col
        return None
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the features used."""
        return {
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'n_features': len(self.feature_columns),
            'feature_groups': self.feature_engineer.get_feature_importance_groups(),
        }


def create_ml_pipeline(
    config: Optional[PipelineConfig] = None
) -> MLDataPipeline:
    """Factory function to create the ML pipeline."""
    return MLDataPipeline(config=config)
