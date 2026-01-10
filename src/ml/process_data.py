#!/usr/bin/env python3
"""
Script to process raw FBREF data into ML-ready processed data.

Usage:
    python -m src.ml.process_data
    
Or run directly:
    python src/ml/process_data.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Team name standardization
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


def process_raw_data(
    raw_data_dir: str = "data/raw",
    processed_data_dir: str = "data/processed",
    seasons: list = None
):
    """
    Process all raw data files and create processed datasets.
    
    Creates:
    - all_fixtures.csv: All matches with results and xG
    - all_team_stats.csv: All team statistics merged
    - ml_features.csv: Feature-engineered dataset ready for ML
    """
    raw_path = Path(raw_data_dir)
    processed_path = Path(processed_data_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect seasons if not provided
    if seasons is None:
        seasons = sorted([
            d.name for d in raw_path.iterdir() 
            if d.is_dir() and d.name[0].isdigit()
        ])
    
    logger.info(f"Processing seasons: {seasons}")
    
    # 1. Combine all fixtures
    all_fixtures = combine_fixtures(raw_path, seasons)
    if not all_fixtures.empty:
        fixtures_path = processed_path / "all_fixtures.csv"
        all_fixtures.to_csv(fixtures_path, index=False)
        logger.info(f"Saved {len(all_fixtures)} fixtures to {fixtures_path}")
    
    # 2. Combine all team stats
    all_team_stats = combine_team_stats(raw_path, seasons)
    if not all_team_stats.empty:
        stats_path = processed_path / "all_team_stats.csv"
        all_team_stats.to_csv(stats_path, index=False)
        logger.info(f"Saved team stats for {len(all_team_stats)} team-seasons to {stats_path}")
    
    # 3. Create ML-ready features
    if not all_fixtures.empty and not all_team_stats.empty:
        ml_features = create_ml_features(all_fixtures, all_team_stats)
        features_path = processed_path / "ml_features.csv"
        ml_features.to_csv(features_path, index=False)
        logger.info(f"Saved {len(ml_features)} match features to {features_path}")
        
        # Also create train/test splits
        create_train_test_split(ml_features, processed_path)
    
    logger.info("Processing complete!")
    return all_fixtures, all_team_stats


def combine_fixtures(raw_path: Path, seasons: list) -> pd.DataFrame:
    """Combine fixtures from all seasons."""
    all_fixtures = []
    
    for season in seasons:
        fixtures_file = raw_path / season / "fixtures.csv"
        if not fixtures_file.exists():
            logger.warning(f"No fixtures found for {season}")
            continue
            
        df = pd.read_csv(fixtures_file)
        df['season'] = season
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Rename common columns
        col_mappings = {
            'home': 'home_team',
            'away': 'away_team',
            'xg': 'home_xg',
            'xg.1': 'away_xg',
        }
        df = df.rename(columns={k: v for k, v in col_mappings.items() if k in df.columns})
        
        # Parse score if needed
        if 'score' in df.columns and 'home_goals' not in df.columns:
            score_split = df['score'].astype(str).str.extract(r'(\d+)[–-](\d+)')
            df['home_goals'] = pd.to_numeric(score_split[0], errors='coerce')
            df['away_goals'] = pd.to_numeric(score_split[1], errors='coerce')
        
        # Clean team names
        if 'home_team' in df.columns:
            df['home_team'] = df['home_team'].replace(TEAM_NAME_MAPPINGS)
        if 'away_team' in df.columns:
            df['away_team'] = df['away_team'].replace(TEAM_NAME_MAPPINGS)
            
        # Parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        # Convert xG to numeric
        for col in ['home_xg', 'away_xg']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows without scores (future matches)
        df = df.dropna(subset=['home_goals', 'away_goals'])
        
        all_fixtures.append(df)
        logger.info(f"  {season}: {len(df)} matches")
    
    if not all_fixtures:
        return pd.DataFrame()
        
    combined = pd.concat(all_fixtures, ignore_index=True)
    
    # Add derived columns
    combined['match_result'] = combined.apply(
        lambda x: 2 if x['home_goals'] > x['away_goals']
        else (0 if x['home_goals'] < x['away_goals'] else 1),
        axis=1
    )
    combined['total_goals'] = combined['home_goals'] + combined['away_goals']
    combined['goal_diff'] = combined['home_goals'] - combined['away_goals']
    
    return combined.sort_values('date').reset_index(drop=True)


def combine_team_stats(raw_path: Path, seasons: list) -> pd.DataFrame:
    """Combine team statistics from all seasons."""
    
    # Stat files to load with prefixes
    stat_files = {
        'std': ('standard_stats_for.csv', 'standard_stats_against.csv'),
        'sht': ('shooting_for.csv', 'shooting_against.csv'),
        'pas': ('passing_for.csv', 'passing_against.csv'),
        'pos': ('possession_for.csv', 'possession_against.csv'),
        'def': ('defensive_actions_for.csv', 'defensive_actions_against.csv'),
        'gca': ('goal_shot_creation_for.csv', 'goal_shot_creation_against.csv'),
        'gk': ('goalkeeping_for.csv', 'goalkeeping_against.csv'),
        'msc': ('misc_for.csv', 'misc_against.csv'),
    }
    
    all_stats = []
    
    for season in seasons:
        season_path = raw_path / season
        if not season_path.exists():
            continue
            
        # Start with a base DataFrame
        base_df = None
        
        for prefix, (for_file, against_file) in stat_files.items():
            # Load "for" stats
            for_path = season_path / for_file
            if for_path.exists():
                df_for = pd.read_csv(for_path)
                team_col = find_team_column(df_for)
                if team_col:
                    df_for = df_for.rename(columns={team_col: 'team'})
                    df_for['team'] = df_for['team'].replace(TEAM_NAME_MAPPINGS)
                    
                    # Prefix columns
                    df_for = prefix_columns(df_for, f'{prefix}_f', exclude=['team', 'season', 'stat_type'])
                    
                    if base_df is None:
                        base_df = df_for
                    else:
                        base_df = base_df.merge(
                            df_for.drop(columns=['season', 'stat_type'], errors='ignore'),
                            on='team', how='outer', suffixes=('', '_dup')
                        )
                        base_df = base_df.loc[:, ~base_df.columns.str.endswith('_dup')]
            
            # Load "against" stats
            against_path = season_path / against_file
            if against_path.exists():
                df_against = pd.read_csv(against_path)
                team_col = find_team_column(df_against)
                if team_col:
                    df_against = df_against.rename(columns={team_col: 'team'})
                    # Remove "vs " prefix
                    df_against['team'] = df_against['team'].str.replace(r'^vs\s+', '', regex=True)
                    df_against['team'] = df_against['team'].replace(TEAM_NAME_MAPPINGS)
                    
                    # Prefix columns
                    df_against = prefix_columns(df_against, f'{prefix}_a', exclude=['team', 'season', 'stat_type'])
                    
                    if base_df is None:
                        base_df = df_against
                    else:
                        base_df = base_df.merge(
                            df_against.drop(columns=['season', 'stat_type'], errors='ignore'),
                            on='team', how='outer', suffixes=('', '_dup')
                        )
                        base_df = base_df.loc[:, ~base_df.columns.str.endswith('_dup')]
        
        if base_df is not None:
            base_df['season'] = season
            all_stats.append(base_df)
            logger.info(f"  {season}: {len(base_df)} teams, {len(base_df.columns)} columns")
    
    if not all_stats:
        return pd.DataFrame()
        
    return pd.concat(all_stats, ignore_index=True)


def find_team_column(df: pd.DataFrame) -> str:
    """Find the team/squad column."""
    for col in ['team', 'squad', 'Team', 'Squad']:
        if col in df.columns:
            return col
    return None


def prefix_columns(df: pd.DataFrame, prefix: str, exclude: list) -> pd.DataFrame:
    """Add prefix to column names except excluded ones."""
    df = df.copy()
    rename_dict = {
        col: f'{prefix}_{col}' 
        for col in df.columns 
        if col not in exclude
    }
    return df.rename(columns=rename_dict)


def create_ml_features(fixtures: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """Create ML-ready feature dataset."""
    
    # Merge home team stats
    home_stats = team_stats.copy()
    home_cols = {col: f'home_{col}' for col in home_stats.columns if col not in ['team', 'season']}
    home_stats = home_stats.rename(columns=home_cols)
    home_stats = home_stats.rename(columns={'team': 'home_team'})
    
    features = fixtures.merge(
        home_stats,
        on=['home_team', 'season'],
        how='left'
    )
    
    # Merge away team stats
    away_stats = team_stats.copy()
    away_cols = {col: f'away_{col}' for col in away_stats.columns if col not in ['team', 'season']}
    away_stats = away_stats.rename(columns=away_cols)
    away_stats = away_stats.rename(columns={'team': 'away_team'})
    
    features = features.merge(
        away_stats,
        on=['away_team', 'season'],
        how='left'
    )
    
    # Create differential features using pd.concat for better performance
    home_numeric_cols = [c for c in features.columns if c.startswith('home_') and features[c].dtype in ['float64', 'int64']]
    
    diff_data = {}
    for home_col in home_numeric_cols:
        away_col = home_col.replace('home_', 'away_')
        if away_col in features.columns:
            diff_col = home_col.replace('home_', 'diff_')
            diff_data[diff_col] = features[home_col] - features[away_col]
    
    if diff_data:
        diff_df = pd.DataFrame(diff_data)
        features = pd.concat([features, diff_df], axis=1)
    
    logger.info(f"Created {len(features.columns)} total columns")
    
    return features


def create_train_test_split(
    features: pd.DataFrame, 
    output_path: Path,
    test_size: float = 0.2,
    val_size: float = 0.1
):
    """Create temporal train/val/test split."""
    
    # Sort by date
    features = features.sort_values('date').reset_index(drop=True)
    
    n = len(features)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    
    train_df = features.iloc[:train_end]
    val_df = features.iloc[train_end:val_end]
    test_df = features.iloc[val_end:]
    
    # Save splits
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    logger.info(f"Created splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


if __name__ == '__main__':
    print("=" * 60)
    print("PROCESSING RAW DATA TO PROCESSED DATA")
    print("=" * 60)
    
    fixtures, stats = process_raw_data()
    
    print("\n" + "=" * 60)
    print("OUTPUT FILES:")
    print("=" * 60)
    print("  data/processed/all_fixtures.csv    - All match results")
    print("  data/processed/all_team_stats.csv  - Team statistics")
    print("  data/processed/ml_features.csv     - ML-ready features")
    print("  data/processed/train.csv           - Training set")
    print("  data/processed/val.csv             - Validation set")
    print("  data/processed/test.csv            - Test set")
