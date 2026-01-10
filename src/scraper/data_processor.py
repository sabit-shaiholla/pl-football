"""
Data processing utilities for FBREF scraped data.
Cleans, transforms, and prepares data for ML model training.

This module processes already-scraped CSV files from the raw data directory.
It does NOT load any web data - it only combines and transforms existing files.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

from .config import (
    SEASONS,
    TABLES_CONFIG,
    FEATURE_GROUPS,
    ScraperSettings,
)

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processes raw FBREF data into ML-ready features.
    
    This class works exclusively with already-scraped CSV files.
    It does NOT perform any web scraping - only data transformation.
    
    Key transformations:
    1. Clean and standardize column names
    2. Convert string values to numeric
    3. Handle missing values
    4. Compute derived features
    5. Aggregate team-level features for match prediction
    """
    
    def __init__(self, raw_data_dir: str = "data/raw",
                 processed_data_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_season_data(self, season: str) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files for a season from the raw data directory.
        
        This only reads local files - no web requests.
        """
        season_dir = self.raw_data_dir / season
        data = {}
        
        if not season_dir.exists():
            logger.warning(f"Season directory not found: {season_dir}")
            return data
            
        for csv_file in season_dir.glob("*.csv"):
            key = csv_file.stem
            try:
                df = pd.read_csv(csv_file)
                data[key] = df
                logger.info(f"Loaded {key}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                
        return data
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        df = df.copy()
        
        # Remove special characters and standardize
        df.columns = [
            re.sub(r'[^\w\s]', '', col).strip().replace(' ', '_').lower()
            for col in df.columns
        ]
        
        # Handle duplicate column names
        cols = []
        seen = {}
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                cols.append(col)
        df.columns = cols
        
        return df
    
    def convert_to_numeric(self, df: pd.DataFrame, 
                          exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Convert string columns to numeric where possible."""
        df = df.copy()
        exclude = exclude_cols or ['squad', 'season', 'stat_type', 'team', 'player']
        
        for col in df.columns:
            if col.lower() in [e.lower() for e in exclude]:
                continue
                
            # Try to convert to numeric
            if df[col].dtype == object:
                # Remove commas and convert
                try:
                    df[col] = df[col].astype(str).str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
                    
        return df
    
    def clean_squad_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize team/squad names across datasets."""
        df = df.copy()
        
        # Common name mappings (FBREF uses various formats)
        name_mappings = {
            "Manchester Utd": "Manchester United",
            "Manchester City": "Manchester City",
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
        
        squad_col = None
        for col in ['squad', 'Squad', 'team', 'Team']:
            if col in df.columns:
                squad_col = col
                break
                
        if squad_col:
            df[squad_col] = df[squad_col].replace(name_mappings)
            
        return df
    
    def process_fixtures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process fixtures data for match outcome prediction.
        
        Creates features for:
        - Match outcome (W/D/L)
        - Goals scored/conceded
        - xG differentials
        - Home/Away flags
        """
        df = df.copy()
        df = self.clean_column_names(df)
        
        # Standardize key columns
        col_mappings = {
            'wk': 'gameweek',
            'day': 'day_of_week',
            'date': 'date',
            'time': 'kickoff_time',
            'home': 'home_team',
            'away': 'away_team',
            'xg': 'home_xg',
            'xg_1': 'away_xg',
            'score': 'score',
            'attendance': 'attendance',
            'venue': 'venue',
            'referee': 'referee',
        }
        
        df = df.rename(columns={k: v for k, v in col_mappings.items() if k in df.columns})
        
        # Parse score into goals
        if 'score' in df.columns:
            score_split = df['score'].astype(str).str.extract(r'(\d+)[–-](\d+)')
            df['home_goals'] = pd.to_numeric(score_split[0], errors='coerce')
            df['away_goals'] = pd.to_numeric(score_split[1], errors='coerce')
            
            # Determine match outcome from home perspective
            df['home_result'] = df.apply(
                lambda x: 'W' if x['home_goals'] > x['away_goals'] 
                         else ('L' if x['home_goals'] < x['away_goals'] else 'D'),
                axis=1
            )
            df['away_result'] = df['home_result'].map({'W': 'L', 'L': 'W', 'D': 'D'})
            
            # Goal difference
            df['home_gd'] = df['home_goals'] - df['away_goals']
            df['away_gd'] = -df['home_gd']
            
        # Parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['month'] = df['date'].dt.month
            df['day_of_year'] = df['date'].dt.dayofyear
            
        # Clean team names
        for col in ['home_team', 'away_team']:
            if col in df.columns:
                df = self.clean_squad_names(df)
                
        return df
    
    def process_squad_stats(self, df: pd.DataFrame, 
                           stat_type: str = 'for') -> pd.DataFrame:
        """Process squad statistics table."""
        df = df.copy()
        df = self.clean_column_names(df)
        df = self.clean_squad_names(df)
        df = self.convert_to_numeric(df)
        
        # Add prefix for stat type
        prefix = 'for_' if stat_type == 'for' else 'against_'
        
        # Rename numeric columns with prefix (except identifiers)
        id_cols = ['squad', 'season', 'stat_type', 'rk', 'pl', 'age']
        for col in df.columns:
            if col not in id_cols and df[col].dtype in ['float64', 'int64']:
                df = df.rename(columns={col: f"{prefix}{col}"})
                
        return df
    
    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived features that are predictive for match outcomes.
        
        These are engineered features based on domain expertise in football analytics.
        """
        df = df.copy()
        
        # Attacking efficiency
        if 'for_gls' in df.columns and 'for_sh' in df.columns:
            df['attacking_efficiency'] = df['for_gls'] / df['for_sh'].replace(0, np.nan)
            
        # Defensive solidity (goals against per shot faced)
        if 'against_gls' in df.columns and 'against_sh' in df.columns:
            df['defensive_efficiency'] = df['against_gls'] / df['against_sh'].replace(0, np.nan)
            
        # xG over/underperformance
        if 'for_gls' in df.columns and 'for_xg' in df.columns:
            df['xg_overperformance'] = df['for_gls'] - df['for_xg']
            
        # xGA over/underperformance (negative is good - conceding less than expected)
        if 'against_gls' in df.columns and 'against_xg' in df.columns:
            df['xga_overperformance'] = df['against_gls'] - df['against_xg']
            
        # Shot quality (xG per shot)
        if 'for_xg' in df.columns and 'for_sh' in df.columns:
            df['shot_quality'] = df['for_xg'] / df['for_sh'].replace(0, np.nan)
            
        # Possession effectiveness (progressive actions per possession %)
        if 'for_poss' in df.columns and 'for_prgc' in df.columns:
            df['possession_effectiveness'] = df['for_prgc'] / df['for_poss'].replace(0, np.nan)
            
        # Defensive pressure (tackles + interceptions per 90)
        if 'for_tkl' in df.columns and 'for_int' in df.columns:
            df['defensive_pressure'] = df['for_tkl'] + df['for_int']
            
        return df
    
    def merge_all_stats(self, season_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all statistical tables into a single DataFrame per team.
        
        This creates a comprehensive feature set for each team-season combination.
        """
        # Start with standard stats if available
        base_df = None
        
        for key in ['standard_stats_for', 'standard_stats']:
            if key in season_data:
                base_df = season_data[key].copy()
                base_df = self.clean_column_names(base_df)
                base_df = self.clean_squad_names(base_df)
                break
                
        if base_df is None:
            logger.warning("No standard stats found to use as base")
            return pd.DataFrame()
            
        # Get the squad column name (FBREF uses 'team' or 'squad')
        squad_col = None
        for candidate in ['squad', 'team', 'Squad', 'Team']:
            if candidate in base_df.columns:
                squad_col = candidate
                break
        
        if not squad_col:
            for col in base_df.columns:
                if 'squad' in col.lower() or 'team' in col.lower():
                    squad_col = col
                    break
                    
        if not squad_col:
            logger.error(f"Cannot find squad column. Available columns: {list(base_df.columns)[:10]}")
            return base_df
        
        logger.debug(f"Using '{squad_col}' as squad column")
            
        # Merge other tables
        tables_to_merge = [
            'shooting_for', 'shooting_against',
            'passing_for', 'passing_against',
            'goal_shot_creation_for', 'goal_shot_creation_against',
            'defensive_actions_for', 'defensive_actions_against',
            'possession_for', 'possession_against',
            'goalkeeping_for', 'goalkeeping_against',
            'misc_for', 'misc_against',
        ]
        
        for table_key in tables_to_merge:
            if table_key not in season_data:
                continue
                
            df_to_merge = season_data[table_key].copy()
            df_to_merge = self.clean_column_names(df_to_merge)
            df_to_merge = self.clean_squad_names(df_to_merge)
            
            # Find matching squad column (can be 'team' or 'squad')
            merge_squad_col = None
            for candidate in ['squad', 'team', 'Squad', 'Team']:
                if candidate in df_to_merge.columns:
                    merge_squad_col = candidate
                    break
            
            if not merge_squad_col:
                for col in df_to_merge.columns:
                    if 'squad' in col.lower() or 'team' in col.lower():
                        merge_squad_col = col
                        break
                    
            if not merge_squad_col:
                logger.debug(f"Skipping {table_key}: no squad/team column found")
                continue
                
            # Add prefix to avoid column conflicts
            prefix = table_key.replace('_for', '_f_').replace('_against', '_a_')
            rename_dict = {}
            for col in df_to_merge.columns:
                if col != merge_squad_col and col not in ['season', 'stat_type']:
                    rename_dict[col] = f"{prefix}_{col}"
                    
            df_to_merge = df_to_merge.rename(columns=rename_dict)
            
            # Merge on squad
            base_df = base_df.merge(
                df_to_merge[[merge_squad_col] + list(rename_dict.values())],
                left_on=squad_col,
                right_on=merge_squad_col,
                how='left',
                suffixes=('', '_dup')
            )
            
            # Remove duplicate columns
            base_df = base_df.loc[:, ~base_df.columns.str.endswith('_dup')]
            
        return base_df
    
    def create_match_features(self, fixtures_df: pd.DataFrame,
                             team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create match-level features by combining team statistics.
        
        For each match, we combine home and away team stats to create
        features that can be used for match outcome prediction.
        """
        matches = fixtures_df.copy()
        team_stats = team_stats_df.copy()
        
        # Ensure we have the required columns (can be 'team' or 'squad')
        squad_col = None
        for candidate in ['squad', 'team', 'Squad', 'Team']:
            if candidate in team_stats.columns:
                squad_col = candidate
                break
        
        if not squad_col:
            for col in team_stats.columns:
                if 'squad' in col.lower() or 'team' in col.lower():
                    squad_col = col
                    break
                
        if not squad_col:
            logger.error(f"Cannot find squad column in team stats. Available: {list(team_stats.columns)[:10]}")
            return matches
        
        # Check if matches has valid team columns
        if 'home_team' not in matches.columns or 'away_team' not in matches.columns:
            logger.warning("Missing home_team or away_team columns in fixtures")
            return matches
        
        # Ensure team columns are strings (not float from empty values)
        matches['home_team'] = matches['home_team'].astype(str).replace('nan', '')
        matches['away_team'] = matches['away_team'].astype(str).replace('nan', '')
        
        # Filter out matches with missing team names
        valid_matches = matches[(matches['home_team'] != '') & (matches['away_team'] != '')]
        if len(valid_matches) < len(matches):
            logger.warning(f"Filtered out {len(matches) - len(valid_matches)} matches with missing team names")
            matches = valid_matches
        
        if matches.empty:
            logger.warning("No valid matches with team names to process")
            return matches
            
        # Get numeric columns for team stats
        numeric_cols = team_stats.select_dtypes(include=[np.number]).columns.tolist()
        
        # Ensure squad column is string type
        team_stats[squad_col] = team_stats[squad_col].astype(str)
        
        # Merge home team stats
        home_stats = team_stats[[squad_col] + numeric_cols].copy()
        home_stats = home_stats.add_prefix('home_')
        home_stats = home_stats.rename(columns={f'home_{squad_col}': 'home_team_merge'})
        
        matches = matches.merge(
            home_stats,
            left_on='home_team',
            right_on='home_team_merge',
            how='left'
        )
        
        # Merge away team stats
        away_stats = team_stats[[squad_col] + numeric_cols].copy()
        away_stats = away_stats.add_prefix('away_')
        away_stats = away_stats.rename(columns={f'away_{squad_col}': 'away_team_merge'})
        
        matches = matches.merge(
            away_stats,
            left_on='away_team',
            right_on='away_team_merge',
            how='left'
        )
        
        # Create differential features (home - away)
        for col in numeric_cols:
            home_col = f'home_{col}'
            away_col = f'away_{col}'
            if home_col in matches.columns and away_col in matches.columns:
                matches[f'diff_{col}'] = matches[home_col] - matches[away_col]
                
        # Clean up merge columns
        matches = matches.drop(columns=['home_team_merge', 'away_team_merge'], errors='ignore')
        
        return matches
    
    def process_all_seasons(self, seasons: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Process all seasons and combine into a single dataset.
        
        This method loads data from local CSV files only.
        No web scraping is performed.
        """
        seasons_to_process = seasons or SEASONS
        all_data = []
        
        for season in seasons_to_process:
            logger.info(f"Processing season {season}")
            
            try:
                # Load from local CSV files
                season_data = self.load_season_data(season)
                
                if not season_data:
                    logger.warning(f"No data found for season {season}")
                    continue
                    
                # Merge all stats for this season
                team_stats = self.merge_all_stats(season_data)
                
                if team_stats.empty:
                    continue
                    
                # Compute derived features
                team_stats = self.compute_derived_features(team_stats)
                team_stats['season'] = season
                
                # Process fixtures if available
                if 'fixtures' in season_data:
                    fixtures = self.process_fixtures(season_data['fixtures'])
                    fixtures['season'] = season
                    
                    # Create match-level features
                    match_features = self.create_match_features(fixtures, team_stats)
                    all_data.append(match_features)
                else:
                    all_data.append(team_stats)
                    
            except Exception as e:
                logger.error(f"Error processing season {season}: {e}")
                continue
                
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save processed data
            output_path = self.processed_data_dir / 'all_seasons_processed.csv'
            combined_df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
            
            return combined_df
            
        return pd.DataFrame()


class FeatureEngineer:
    """
    Advanced feature engineering for match prediction models.
    
    Creates rolling averages, form indicators, and head-to-head features.
    """
    
    def __init__(self):
        pass
    
    def _find_column(self, df: pd.DataFrame, candidates: list) -> Optional[str]:
        """Find the first matching column from a list of candidates."""
        for col in candidates:
            if col in df.columns:
                return col
            # Case-insensitive search
            for df_col in df.columns:
                if df_col.lower() == col.lower():
                    return df_col
        return None
        
    def compute_rolling_form(self, df: pd.DataFrame, 
                            window: int = 5,
                            team_col: str = None,
                            date_col: str = None) -> pd.DataFrame:
        """
        Compute rolling form features over last N matches.
        
        Form is a crucial predictor - recent performance matters more than season averages.
        """
        df = df.copy()
        
        # Find team column
        team_candidates = ['squad', 'team', 'club', 'home_team', 'away_team']
        team_col = team_col or self._find_column(df, team_candidates)
        
        # Find date column
        date_candidates = ['date', 'match_date', 'kickoff', 'datetime']
        date_col = date_col or self._find_column(df, date_candidates)
        
        if not team_col:
            logger.warning("No team column found, skipping rolling form calculation")
            return df
            
        if not date_col:
            logger.warning("No date column found, skipping rolling form calculation")
            return df
            
        # Sort by team and date
        try:
            df = df.sort_values([team_col, date_col])
        except Exception as e:
            logger.warning(f"Could not sort by {team_col}, {date_col}: {e}")
            return df
        
        # Key metrics to compute rolling averages for
        rolling_metrics = [
            'gls', 'goals', 'xg', 'sh', 'shots', 'sot', 'shots_on_target',
            'poss', 'possession', 'goals_conceded', 'xga', 'shots_faced'
        ]
        
        for metric in rolling_metrics:
            if metric in df.columns:
                col_name = f'{metric}_last_{window}'
                try:
                    df[col_name] = df.groupby(team_col)[metric].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                except Exception as e:
                    logger.debug(f"Could not compute rolling {metric}: {e}")
                
        # Points from last N games
        result_col = self._find_column(df, ['result', 'home_result', 'away_result'])
        if result_col:
            df['points'] = df[result_col].map({'W': 3, 'D': 1, 'L': 0})
            try:
                df[f'points_last_{window}'] = df.groupby(team_col)['points'].transform(
                    lambda x: x.rolling(window, min_periods=1).sum()
                )
            except Exception as e:
                logger.debug(f"Could not compute rolling points: {e}")
            
        return df
    
    def compute_home_away_splits(self, df: pd.DataFrame,
                                 team_col: str = None) -> pd.DataFrame:
        """
        Compute separate home and away performance metrics.
        
        Home advantage is real in football - splitting these improves predictions.
        """
        df = df.copy()
        
        # This would require match-level data with home/away flags
        # Implementation depends on data structure
        
        return df
    
    def add_elo_ratings(self, df: pd.DataFrame,
                        k_factor: float = 20.0,
                        home_advantage: float = 100.0) -> pd.DataFrame:
        """
        Compute Elo ratings for teams based on match results.
        
        Elo is a powerful predictor that captures relative team strength.
        """
        # Find date column
        date_col = self._find_column(df, ['date', 'match_date', 'kickoff', 'datetime'])
        if not date_col:
            logger.warning("No date column found for Elo calculation")
            return df
            
        df = df.copy()
        try:
            df = df.sort_values(date_col)
        except Exception as e:
            logger.warning(f"Could not sort by {date_col} for Elo: {e}")
            return df
        
        # Initialize Elo ratings
        elo_ratings = {}
        initial_elo = 1500.0
        
        elo_home = []
        elo_away = []
        
        # Find home/away team columns
        home_team_col = self._find_column(df, ['home_team', 'home', 'team_home'])
        away_team_col = self._find_column(df, ['away_team', 'away', 'team_away'])
        
        if not home_team_col or not away_team_col:
            logger.warning("No home/away team columns found for Elo calculation")
            return df
        
        for idx, row in df.iterrows():
            home_team = row.get(home_team_col)
            away_team = row.get(away_team_col)
            
            if pd.isna(home_team) or pd.isna(away_team):
                elo_home.append(np.nan)
                elo_away.append(np.nan)
                continue
                
            # Get current ratings
            home_elo = elo_ratings.get(home_team, initial_elo)
            away_elo = elo_ratings.get(away_team, initial_elo)
            
            # Store pre-match Elo
            elo_home.append(home_elo)
            elo_away.append(away_elo)
            
            # Calculate expected scores
            home_expected = 1 / (1 + 10 ** ((away_elo - home_elo - home_advantage) / 400))
            away_expected = 1 - home_expected
            
            # Determine actual result
            result_col = self._find_column(df, ['home_result', 'result'])
            home_result = row.get(result_col) if result_col else None
            if home_result == 'W':
                home_actual, away_actual = 1, 0
            elif home_result == 'L':
                home_actual, away_actual = 0, 1
            else:
                home_actual, away_actual = 0.5, 0.5
                
            # Update Elo ratings
            elo_ratings[home_team] = home_elo + k_factor * (home_actual - home_expected)
            elo_ratings[away_team] = away_elo + k_factor * (away_actual - away_expected)
            
        df['home_elo'] = elo_home
        df['away_elo'] = elo_away
        df['elo_diff'] = df['home_elo'] - df['away_elo']
        
        return df


def main():
    """Process all scraped data from local CSV files."""
    processor = DataProcessor()
    
    # Process all seasons from local files
    processed_data = processor.process_all_seasons()
    
    if not processed_data.empty:
        logger.info(f"Processed {len(processed_data)} rows")
        
        # Apply feature engineering
        engineer = FeatureEngineer()
        
        # Check for date column variants
        date_candidates = ['date', 'match_date', 'kickoff', 'datetime']
        has_date = any(col in processed_data.columns or 
                       col.lower() in [c.lower() for c in processed_data.columns] 
                       for col in date_candidates)
        
        if has_date:
            processed_data = engineer.add_elo_ratings(processed_data)
            processed_data = engineer.compute_rolling_form(processed_data)
        else:
            logger.info("No date column found, skipping time-based features")
            
        # Save final dataset
        output_path = Path('data/processed/ml_ready_dataset.csv')
        processed_data.to_csv(output_path, index=False)
        logger.info(f"Saved ML-ready dataset to {output_path}")
        

if __name__ == "__main__":
    main()
