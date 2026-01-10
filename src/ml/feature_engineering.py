"""
Feature Engineering for Premier League Match Prediction.

This module implements comprehensive feature engineering based on football analytics
domain expertise. Features are designed to capture:

1. Team Attacking Strength - ability to score goals
2. Team Defensive Strength - ability to prevent goals  
3. Form and Momentum - recent performance trends
4. Expected Goals (xG) metrics - probabilistic goal expectations
5. Playing Style indicators - possession, passing, pressing
6. Set-piece threat and discipline
7. Home/Away performance differentials

All features are engineered to maximize predictive power for match outcomes.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Rolling window sizes for form calculation
    form_windows: List[int] = field(default_factory=lambda: [3, 5, 10])
    
    # ELO rating parameters
    elo_k_factor: float = 32.0
    elo_home_advantage: float = 100.0
    elo_initial: float = 1500.0
    
    # Feature scaling method: 'standard', 'minmax', 'robust'
    scaling_method: str = 'standard'
    
    # Whether to include advanced metrics
    include_advanced_metrics: bool = True
    
    # Minimum matches before using team stats (cold start handling)
    min_matches_for_stats: int = 3


class FeatureEngineer:
    """
    Comprehensive feature engineering for football match prediction.
    
    This class transforms raw FBREF statistics into ML-ready features,
    applying football analytics domain knowledge to create predictive features.
    
    Feature Categories:
    ------------------
    1. ATTACKING FEATURES
       - Goals, xG, shots, shot quality
       - Goal creation actions (SCA, GCA)
       - Progressive actions (carries, passes)
       
    2. DEFENSIVE FEATURES  
       - Goals against, xGA, shots against
       - Tackles, interceptions, blocks
       - Pressing intensity
       
    3. POSSESSION & STYLE
       - Possession %, pass completion
       - Progressive passing distance
       - Territory (touches by zone)
       
    4. SET PIECES & DISCIPLINE
       - Free kicks, corners, penalties
       - Yellow/red cards, fouls
       
    5. FORM & MOMENTUM
       - Rolling average results
       - Points per game trends
       - xG trend analysis
       
    6. HEAD-TO-HEAD
       - Historical matchup results
       - Recent encounters
       
    7. ELO RATINGS
       - Dynamic team strength rating
       - Home/away adjusted ratings
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.elo_ratings: Dict[str, float] = {}
        self.team_stats_cache: Dict[str, pd.DataFrame] = {}
    
    def calculate_elo_ratings(self, fixtures: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate ELO ratings from historical fixtures.
        
        Args:
            fixtures: Historical match data with home_team, away_team, and either
                      home_goals/away_goals columns OR a score column like "4–2"
            
        Returns:
            Dictionary mapping team names to ELO ratings
        """
        # Reset ratings
        self.elo_ratings = {}
        
        # Make a copy and parse score if needed
        historical = fixtures.copy()
        
        # Parse score column if home_goals/away_goals don't exist
        if 'home_goals' not in historical.columns and 'score' in historical.columns:
            score_split = historical['score'].astype(str).str.extract(r'(\d+)[–-](\d+)')
            historical['home_goals'] = pd.to_numeric(score_split[0], errors='coerce')
            historical['away_goals'] = pd.to_numeric(score_split[1], errors='coerce')
        
        # Drop rows without valid scores
        historical = historical.dropna(subset=['home_goals', 'away_goals'])
        
        # Sort by date
        historical = historical.sort_values('date')
        
        for _, match in historical.iterrows():
            home_team = match.get('home_team', '')
            away_team = match.get('away_team', '')
            
            if not home_team or not away_team:
                continue
                
            # Initialize ratings if new team
            if home_team not in self.elo_ratings:
                self.elo_ratings[home_team] = self.config.elo_initial
            if away_team not in self.elo_ratings:
                self.elo_ratings[away_team] = self.config.elo_initial
                
            # Get current ratings
            home_elo = self.elo_ratings[home_team]
            away_elo = self.elo_ratings[away_team]
            
            # Calculate expected scores
            home_expected = self._elo_expected_score(
                home_elo + self.config.elo_home_advantage, away_elo
            )
            
            # Get actual result
            home_goals = match.get('home_goals', 0) or 0
            away_goals = match.get('away_goals', 0) or 0
            
            if home_goals > away_goals:
                home_actual, away_actual = 1.0, 0.0
            elif home_goals < away_goals:
                home_actual, away_actual = 0.0, 1.0
            else:
                home_actual, away_actual = 0.5, 0.5
                
            # Update ratings
            self.elo_ratings[home_team] = home_elo + self.config.elo_k_factor * (home_actual - home_expected)
            self.elo_ratings[away_team] = away_elo + self.config.elo_k_factor * (away_actual - (1 - home_expected))
        
        return self.elo_ratings
        
    def engineer_features(
        self,
        fixtures: pd.DataFrame,
        team_stats: pd.DataFrame,
        historical_fixtures: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Main feature engineering pipeline.
        
        Args:
            fixtures: Match fixtures with home_team, away_team, date
            team_stats: Aggregated team statistics per season
            historical_fixtures: Past match results for form calculation
            
        Returns:
            DataFrame with all engineered features for each match
        """
        logger.info(f"Engineering features for {len(fixtures)} matches")
        
        # Make copies to avoid modifying originals
        matches = fixtures.copy()
        
        # 1. Merge team-level season statistics
        matches = self._merge_team_stats(matches, team_stats)
        
        # 2. Engineer attacking features
        matches = self._engineer_attacking_features(matches)
        
        # 3. Engineer defensive features
        matches = self._engineer_defensive_features(matches)
        
        # 4. Engineer possession and style features
        matches = self._engineer_possession_features(matches)
        
        # 5. Engineer set piece and discipline features
        matches = self._engineer_setpiece_discipline_features(matches)
        
        # 6. Engineer differential features (home - away)
        matches = self._engineer_differential_features(matches)
        
        # 7. Calculate form features (if historical data available)
        if historical_fixtures is not None and len(historical_fixtures) > 0:
            matches = self._engineer_form_features(matches, historical_fixtures)
            
        # 8. Calculate ELO ratings
        if historical_fixtures is not None and len(historical_fixtures) > 0:
            matches = self._engineer_elo_features(matches, historical_fixtures)
            
        # 9. Engineer head-to-head features
        if historical_fixtures is not None and len(historical_fixtures) > 0:
            matches = self._engineer_h2h_features(matches, historical_fixtures)
        
        # 10. Create target variables
        matches = self._create_target_variables(matches)
        
        logger.info(f"Engineered {len(matches.columns)} total features")
        
        return matches
    
    def _merge_team_stats(
        self, 
        matches: pd.DataFrame, 
        team_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge team-level statistics to match data by team AND season."""
        
        # Find squad column
        squad_col = self._find_squad_column(team_stats)
        if not squad_col:
            logger.warning("Cannot find squad column in team stats")
            return matches
            
        # Get numeric columns (exclude season from merge columns)
        numeric_cols = [c for c in team_stats.select_dtypes(include=[np.number]).columns.tolist()
                       if c != 'season']
        
        # Merge home team stats (on team AND season to avoid duplicates)
        home_stats = team_stats[[squad_col, 'season'] + numeric_cols].copy()
        home_stats.columns = ['home_team_key', 'home_season_key'] + [f'home_{c}' for c in numeric_cols]
        
        matches = matches.merge(
            home_stats,
            left_on=['home_team', 'season'],
            right_on=['home_team_key', 'home_season_key'],
            how='left'
        ).drop(columns=['home_team_key', 'home_season_key'], errors='ignore')
        
        # Merge away team stats (on team AND season)
        away_stats = team_stats[[squad_col, 'season'] + numeric_cols].copy()
        away_stats.columns = ['away_team_key', 'away_season_key'] + [f'away_{c}' for c in numeric_cols]
        
        matches = matches.merge(
            away_stats,
            left_on=['away_team', 'season'],
            right_on=['away_team_key', 'away_season_key'],
            how='left'
        ).drop(columns=['away_team_key', 'away_season_key'], errors='ignore')
        
        return matches
    
    def _engineer_attacking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer attacking strength features.
        
        Football Analytics Perspective:
        - xG (Expected Goals) is the gold standard for attack quality
        - Non-penalty xG (npxG) removes penalty luck
        - Shot volume vs quality tradeoff matters
        - Goal creation actions show build-up quality
        - Progressive actions indicate territory gain
        """
        df = df.copy()
        
        # xG-based attacking metrics
        for prefix in ['home', 'away']:
            # Core xG features (already in data)
            xg_col = f'{prefix}_xg' if f'{prefix}_xg' in df.columns else None
            npxg_col = f'{prefix}_npxg' if f'{prefix}_npxg' in df.columns else None
            goals_col = f'{prefix}_goals' if f'{prefix}_goals' in df.columns else None
            
            # Look for alternative column names
            for col in df.columns:
                if 'xg' in col.lower() and prefix in col.lower() and 'assist' not in col.lower():
                    if xg_col is None:
                        xg_col = col
                    if 'npxg' in col.lower() and npxg_col is None:
                        npxg_col = col
                        
            # xG overperformance (goals - xG)
            # Positive = finishing above expectation (skill or luck)
            # Negative = underperforming chances
            if xg_col and goals_col and xg_col in df.columns and goals_col in df.columns:
                df[f'{prefix}_xg_overperformance'] = df[goals_col] - df[xg_col]
            
            # Shot quality (xG per shot)
            shots_col = None
            for col in df.columns:
                if 'shots' in col.lower() or col.lower().endswith('_sh'):
                    if prefix in col.lower():
                        shots_col = col
                        break
                        
            if xg_col and shots_col and xg_col in df.columns and shots_col in df.columns:
                df[f'{prefix}_shot_quality'] = df[xg_col] / df[shots_col].replace(0, np.nan)
                
            # Shooting accuracy (shots on target %)
            sot_col = None
            for col in df.columns:
                if ('sot' in col.lower() or 'on_target' in col.lower()) and prefix in col.lower():
                    if 'pct' not in col.lower():
                        sot_col = col
                        break
                        
            if shots_col and sot_col and shots_col in df.columns and sot_col in df.columns:
                df[f'{prefix}_shot_accuracy'] = df[sot_col] / df[shots_col].replace(0, np.nan)
                
            # Goal conversion rate
            if goals_col and shots_col and goals_col in df.columns and shots_col in df.columns:
                df[f'{prefix}_goal_conversion'] = df[goals_col] / df[shots_col].replace(0, np.nan)
                
        return df
    
    def _engineer_defensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer defensive strength features.
        
        Football Analytics Perspective:
        - xGA (Expected Goals Against) shows defensive quality
        - Tackles + Interceptions = defensive actions
        - Blocks show last-ditch defending
        - Clean sheet % is outcome-based but predictive
        - Errors leading to goals are critical
        """
        df = df.copy()
        
        for prefix in ['home', 'away']:
            # Find relevant columns dynamically
            ga_col = None
            xga_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if prefix in col_lower:
                    if 'goals_against' in col_lower or col_lower.endswith('_ga'):
                        ga_col = col
                    if 'xg' in col_lower and 'against' in col_lower:
                        xga_col = col
                        
            # xGA overperformance (negative = conceding less than expected = good)
            if ga_col and xga_col and ga_col in df.columns and xga_col in df.columns:
                df[f'{prefix}_xga_overperformance'] = df[ga_col] - df[xga_col]
                
            # Defensive actions per 90
            tkl_col = None
            int_col = None
            blocks_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if prefix in col_lower:
                    if 'tackles' in col_lower or col_lower.endswith('_tkl'):
                        tkl_col = col
                    if 'interceptions' in col_lower or col_lower.endswith('_int'):
                        int_col = col
                    if 'blocks' in col_lower:
                        blocks_col = col
                        
            if tkl_col and int_col and tkl_col in df.columns and int_col in df.columns:
                df[f'{prefix}_defensive_actions'] = df[tkl_col] + df[int_col]
                
            if blocks_col and blocks_col in df.columns:
                if tkl_col and int_col:
                    df[f'{prefix}_total_defensive_actions'] = (
                        df[tkl_col] + df[int_col] + df[blocks_col]
                    )
                    
        return df
    
    def _engineer_possession_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer possession and playing style features.
        
        Football Analytics Perspective:
        - Raw possession % is less predictive than what you do with it
        - Progressive passes/carries show territorial advancement
        - Pass completion in attacking third matters more
        - Touches in attacking areas indicate dominance
        """
        df = df.copy()
        
        for prefix in ['home', 'away']:
            # Find relevant columns
            poss_col = None
            prog_passes_col = None
            prog_carries_col = None
            touches_att_col = None
            pass_cmp_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if prefix in col_lower:
                    if 'possession' in col_lower or 'poss' in col_lower:
                        poss_col = col
                    if 'progressive' in col_lower and 'pass' in col_lower:
                        prog_passes_col = col
                    if 'progressive' in col_lower and 'carr' in col_lower:
                        prog_carries_col = col
                    if 'touches' in col_lower and 'att' in col_lower:
                        touches_att_col = col
                    if 'passes_pct' in col_lower or 'pass_cmp' in col_lower:
                        pass_cmp_col = col
            
            # Possession effectiveness (progressive actions per possession %)
            if poss_col and prog_passes_col and poss_col in df.columns and prog_passes_col in df.columns:
                df[f'{prefix}_possession_effectiveness'] = (
                    df[prog_passes_col] / df[poss_col].replace(0, np.nan)
                )
                
            # Progressive action rate
            if prog_passes_col and prog_carries_col:
                if prog_passes_col in df.columns and prog_carries_col in df.columns:
                    df[f'{prefix}_progressive_actions'] = df[prog_passes_col] + df[prog_carries_col]
                    
            # Attacking territory dominance
            if touches_att_col and touches_att_col in df.columns:
                total_touches_col = None
                for col in df.columns:
                    if 'touches' in col.lower() and prefix in col.lower() and 'def' not in col.lower() and 'att' not in col.lower():
                        total_touches_col = col
                        break
                        
                if total_touches_col and total_touches_col in df.columns:
                    df[f'{prefix}_attacking_territory'] = (
                        df[touches_att_col] / df[total_touches_col].replace(0, np.nan)
                    )
                    
        return df
    
    def _engineer_setpiece_discipline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer set piece threat and discipline features.
        
        Football Analytics Perspective:
        - Set pieces account for ~30% of goals
        - Penalties are high-conversion chances
        - Discipline (cards/fouls) affects match dynamics
        - Aerial dominance matters for set pieces
        """
        df = df.copy()
        
        for prefix in ['home', 'away']:
            # Find relevant columns
            yellow_col = None
            red_col = None
            fouls_col = None
            pens_won_col = None
            pens_con_col = None
            aerials_won_col = None
            aerials_lost_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if prefix in col_lower:
                    if 'yellow' in col_lower or 'crdy' in col_lower:
                        yellow_col = col
                    if 'red' in col_lower or 'crdr' in col_lower:
                        red_col = col
                    if 'fouls' in col_lower or col_lower.endswith('_fls'):
                        fouls_col = col
                    if 'pens_won' in col_lower or 'pkwon' in col_lower:
                        pens_won_col = col
                    if 'pens_con' in col_lower or 'pkcon' in col_lower:
                        pens_con_col = col
                    if 'aerials_won' in col_lower:
                        aerials_won_col = col
                    if 'aerials_lost' in col_lower:
                        aerials_lost_col = col
                        
            # Discipline score (weighted cards + fouls)
            # Higher = more disciplined (fewer infractions)
            if yellow_col and fouls_col and yellow_col in df.columns and fouls_col in df.columns:
                discipline_score = df[fouls_col] + df[yellow_col] * 3
                if red_col and red_col in df.columns:
                    discipline_score += df[red_col] * 10
                df[f'{prefix}_indiscipline_score'] = discipline_score
                
            # Penalty differential
            if pens_won_col and pens_con_col:
                if pens_won_col in df.columns and pens_con_col in df.columns:
                    df[f'{prefix}_penalty_differential'] = df[pens_won_col] - df[pens_con_col]
                    
            # Aerial dominance
            if aerials_won_col and aerials_lost_col:
                if aerials_won_col in df.columns and aerials_lost_col in df.columns:
                    total_aerials = df[aerials_won_col] + df[aerials_lost_col]
                    df[f'{prefix}_aerial_dominance'] = (
                        df[aerials_won_col] / total_aerials.replace(0, np.nan)
                    )
                    
        return df
    
    def _engineer_differential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create differential features (home - away) for direct comparison.
        
        Football Analytics Perspective:
        - Match outcomes depend on relative strength, not absolute
        - Differentials capture head-to-head advantage
        - Key differentials: xG, possession, progressive actions
        
        NOTE: Excludes match-outcome features (goals, xg at match level) to prevent data leakage.
        """
        df = df.copy()
        
        # Exclude columns that are match outcomes or derived from them (data leakage!)
        leaky_base_cols = {
            'home_goals', 'away_goals',  # Actual match goals
            'home_xg', 'away_xg',        # Actual match xG
        }
        
        # Find all home_ columns and create differentials using pd.concat for performance
        home_cols = [c for c in df.columns if c.startswith('home_')]
        
        diff_data = {}
        for home_col in home_cols:
            # Skip leaky columns
            if home_col in leaky_base_cols:
                continue
                
            away_col = home_col.replace('home_', 'away_')
            if away_col in df.columns:
                # Check if columns are numeric
                if pd.api.types.is_numeric_dtype(df[home_col]) and pd.api.types.is_numeric_dtype(df[away_col]):
                    feature_name = home_col.replace('home_', 'diff_')
                    diff_data[feature_name] = df[home_col] - df[away_col]
        
        if diff_data:
            diff_df = pd.DataFrame(diff_data, index=df.index)
            df = pd.concat([df, diff_df], axis=1)
                    
        return df
    
    def _engineer_form_features(
        self, 
        matches: pd.DataFrame, 
        historical: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate recent form features using rolling windows.
        
        Football Analytics Perspective:
        - Recent form (last 5 games) is highly predictive
        - Distinguish home vs away form
        - xG form captures underlying performance vs results
        - Points per game trend shows momentum
        """
        matches = matches.copy()
        
        # Ensure date columns are datetime
        if 'date' in matches.columns:
            matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
        if 'date' in historical.columns:
            historical['date'] = pd.to_datetime(historical['date'], errors='coerce')
            
        # Sort by date
        historical = historical.sort_values('date')
        
        # Calculate form for each window size
        for window in self.config.form_windows:
            # Initialize form columns
            matches[f'home_form_{window}'] = np.nan
            matches[f'away_form_{window}'] = np.nan
            matches[f'home_xg_form_{window}'] = np.nan
            matches[f'away_xg_form_{window}'] = np.nan
            
            for idx, match in matches.iterrows():
                match_date = match.get('date')
                if pd.isna(match_date):
                    continue
                    
                # Home team form
                home_team = match.get('home_team', '')
                if home_team:
                    home_form = self._get_team_form(
                        historical, home_team, match_date, window
                    )
                    matches.loc[idx, f'home_form_{window}'] = home_form.get('points_avg', np.nan)
                    matches.loc[idx, f'home_xg_form_{window}'] = home_form.get('xg_diff_avg', np.nan)
                    
                # Away team form
                away_team = match.get('away_team', '')
                if away_team:
                    away_form = self._get_team_form(
                        historical, away_team, match_date, window
                    )
                    matches.loc[idx, f'away_form_{window}'] = away_form.get('points_avg', np.nan)
                    matches.loc[idx, f'away_xg_form_{window}'] = away_form.get('xg_diff_avg', np.nan)
                    
        return matches
    
    def _get_team_form(
        self, 
        historical: pd.DataFrame,
        team: str,
        before_date: pd.Timestamp,
        window: int
    ) -> Dict[str, float]:
        """Calculate team form from recent matches."""
        
        # Filter matches before the date involving this team
        team_matches = historical[
            (historical['date'] < before_date) &
            ((historical['home_team'] == team) | (historical['away_team'] == team))
        ].tail(window)
        
        if len(team_matches) < self.config.min_matches_for_stats:
            return {'points_avg': np.nan, 'xg_diff_avg': np.nan}
            
        points = []
        xg_diffs = []
        
        for _, match in team_matches.iterrows():
            is_home = match.get('home_team') == team
            
            # Calculate points
            home_goals = match.get('home_goals', 0) or 0
            away_goals = match.get('away_goals', 0) or 0
            
            if is_home:
                if home_goals > away_goals:
                    points.append(3)
                elif home_goals == away_goals:
                    points.append(1)
                else:
                    points.append(0)
                    
                # xG differential
                home_xg = match.get('home_xg', 0) or 0
                away_xg = match.get('away_xg', 0) or 0
                xg_diffs.append(home_xg - away_xg)
            else:
                if away_goals > home_goals:
                    points.append(3)
                elif away_goals == home_goals:
                    points.append(1)
                else:
                    points.append(0)
                    
                home_xg = match.get('home_xg', 0) or 0
                away_xg = match.get('away_xg', 0) or 0
                xg_diffs.append(away_xg - home_xg)
                
        return {
            'points_avg': np.mean(points) if points else np.nan,
            'xg_diff_avg': np.mean(xg_diffs) if xg_diffs else np.nan
        }
    
    def _engineer_elo_features(
        self, 
        matches: pd.DataFrame, 
        historical: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate ELO ratings for teams.
        
        Football Analytics Perspective:
        - ELO captures long-term team strength
        - Adjusts based on match outcomes and opponent strength
        - Home advantage is built into rating
        - More reliable than league position
        """
        matches = matches.copy()
        
        # Initialize ELO ratings
        self.elo_ratings = {}
        
        # Sort historical by date and process
        historical = historical.sort_values('date').copy()
        
        for _, match in historical.iterrows():
            home_team = match.get('home_team', '')
            away_team = match.get('away_team', '')
            
            if not home_team or not away_team:
                continue
                
            # Initialize ratings if new team
            if home_team not in self.elo_ratings:
                self.elo_ratings[home_team] = self.config.elo_initial
            if away_team not in self.elo_ratings:
                self.elo_ratings[away_team] = self.config.elo_initial
                
            # Get current ratings
            home_elo = self.elo_ratings[home_team]
            away_elo = self.elo_ratings[away_team]
            
            # Calculate expected scores
            home_expected = self._elo_expected_score(
                home_elo + self.config.elo_home_advantage, away_elo
            )
            away_expected = 1 - home_expected
            
            # Get actual result
            home_goals = match.get('home_goals', 0) or 0
            away_goals = match.get('away_goals', 0) or 0
            
            if home_goals > away_goals:
                home_actual, away_actual = 1.0, 0.0
            elif home_goals < away_goals:
                home_actual, away_actual = 0.0, 1.0
            else:
                home_actual, away_actual = 0.5, 0.5
                
            # Update ratings
            self.elo_ratings[home_team] = home_elo + self.config.elo_k_factor * (home_actual - home_expected)
            self.elo_ratings[away_team] = away_elo + self.config.elo_k_factor * (away_actual - away_expected)
            
        # Now apply ELO to matches
        matches['home_elo'] = matches['home_team'].map(
            lambda x: self.elo_ratings.get(x, self.config.elo_initial)
        )
        matches['away_elo'] = matches['away_team'].map(
            lambda x: self.elo_ratings.get(x, self.config.elo_initial)
        )
        matches['elo_diff'] = matches['home_elo'] - matches['away_elo']
        
        # ELO-based win probability
        matches['elo_home_win_prob'] = matches.apply(
            lambda x: self._elo_expected_score(
                x['home_elo'] + self.config.elo_home_advantage, x['away_elo']
            ), axis=1
        )
        
        return matches
    
    def _elo_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score based on ELO ratings."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def _engineer_h2h_features(
        self, 
        matches: pd.DataFrame, 
        historical: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Engineer head-to-head features between teams.
        
        Football Analytics Perspective:
        - Some teams have psychological edge over others
        - Historical H2H matters especially for derbies
        - Recent H2H more relevant than ancient history
        """
        matches = matches.copy()
        
        matches['h2h_home_wins'] = 0
        matches['h2h_draws'] = 0
        matches['h2h_away_wins'] = 0
        matches['h2h_home_goals_avg'] = np.nan
        matches['h2h_away_goals_avg'] = np.nan
        
        for idx, match in matches.iterrows():
            home_team = match.get('home_team', '')
            away_team = match.get('away_team', '')
            match_date = match.get('date')
            
            if not home_team or not away_team or pd.isna(match_date):
                continue
                
            # Find previous H2H matches (last 5 years)
            h2h_matches = historical[
                (historical['date'] < match_date) &
                (
                    ((historical['home_team'] == home_team) & (historical['away_team'] == away_team)) |
                    ((historical['home_team'] == away_team) & (historical['away_team'] == home_team))
                )
            ].tail(10)  # Last 10 H2H encounters
            
            if len(h2h_matches) < 2:
                continue
                
            home_wins = 0
            away_wins = 0
            draws = 0
            home_goals = []
            away_goals = []
            
            for _, h2h in h2h_matches.iterrows():
                h2h_home = h2h.get('home_team')
                h2h_home_goals = h2h.get('home_goals', 0) or 0
                h2h_away_goals = h2h.get('away_goals', 0) or 0
                
                if h2h_home == home_team:
                    # Same fixture orientation
                    home_goals.append(h2h_home_goals)
                    away_goals.append(h2h_away_goals)
                    if h2h_home_goals > h2h_away_goals:
                        home_wins += 1
                    elif h2h_home_goals < h2h_away_goals:
                        away_wins += 1
                    else:
                        draws += 1
                else:
                    # Reversed fixture
                    home_goals.append(h2h_away_goals)
                    away_goals.append(h2h_home_goals)
                    if h2h_away_goals > h2h_home_goals:
                        home_wins += 1
                    elif h2h_away_goals < h2h_home_goals:
                        away_wins += 1
                    else:
                        draws += 1
                        
            matches.loc[idx, 'h2h_home_wins'] = home_wins
            matches.loc[idx, 'h2h_draws'] = draws
            matches.loc[idx, 'h2h_away_wins'] = away_wins
            matches.loc[idx, 'h2h_home_goals_avg'] = np.mean(home_goals) if home_goals else np.nan
            matches.loc[idx, 'h2h_away_goals_avg'] = np.mean(away_goals) if away_goals else np.nan
            
        # H2H dominance score
        total_h2h = matches['h2h_home_wins'] + matches['h2h_draws'] + matches['h2h_away_wins']
        matches['h2h_home_dominance'] = (
            matches['h2h_home_wins'] / total_h2h.replace(0, np.nan)
        )
        
        return matches
    
    def _create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for ML models.
        
        Targets:
        - match_result: 0=away_win, 1=draw, 2=home_win (classification)
        - home_win: binary for home win
        - home_goals, away_goals: for score prediction (regression)
        - total_goals: for over/under prediction
        """
        df = df.copy()
        
        # Ensure we have goal columns
        if 'home_goals' in df.columns and 'away_goals' in df.columns:
            # Match result (categorical)
            df['match_result'] = df.apply(
                lambda x: 2 if x['home_goals'] > x['away_goals']
                else (0 if x['home_goals'] < x['away_goals'] else 1),
                axis=1
            )
            
            # Binary outcomes
            df['home_win'] = (df['home_goals'] > df['away_goals']).astype(int)
            df['away_win'] = (df['home_goals'] < df['away_goals']).astype(int)
            df['draw'] = (df['home_goals'] == df['away_goals']).astype(int)
            
            # Goal totals
            df['total_goals'] = df['home_goals'] + df['away_goals']
            df['goal_difference'] = df['home_goals'] - df['away_goals']
            
            # Over/under thresholds
            df['over_2_5'] = (df['total_goals'] > 2.5).astype(int)
            df['over_1_5'] = (df['total_goals'] > 1.5).astype(int)
            df['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
            
        return df
    
    def _find_squad_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the squad/team column in a DataFrame."""
        for col in ['squad', 'team', 'Squad', 'Team']:
            if col in df.columns:
                return col
        for col in df.columns:
            if 'squad' in col.lower() or 'team' in col.lower():
                return col
        return None
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Return feature groups organized by predictive importance.
        
        This helps with feature selection and model interpretation.
        """
        return {
            'tier_1_high_importance': [
                # xG-based features (strongest predictors)
                'diff_xg', 'diff_npxg', 'home_xg', 'away_xg',
                'home_xg_overperformance', 'away_xg_overperformance',
                'home_shot_quality', 'away_shot_quality',
                # ELO ratings
                'elo_diff', 'home_elo', 'away_elo', 'elo_home_win_prob',
                # Form
                'home_form_5', 'away_form_5', 'home_xg_form_5', 'away_xg_form_5',
            ],
            'tier_2_medium_high_importance': [
                # Defensive metrics
                'diff_goals_against', 'home_xga_overperformance', 'away_xga_overperformance',
                'diff_defensive_actions', 
                # Shot metrics
                'diff_shots', 'diff_shots_on_target', 
                'home_goal_conversion', 'away_goal_conversion',
                # Form variants
                'home_form_3', 'away_form_3', 'home_form_10', 'away_form_10',
            ],
            'tier_3_medium_importance': [
                # Possession
                'diff_possession', 'diff_progressive_passes', 'diff_progressive_carries',
                'home_possession_effectiveness', 'away_possession_effectiveness',
                # H2H
                'h2h_home_wins', 'h2h_away_wins', 'h2h_home_dominance',
                # Goal creation
                'diff_sca', 'diff_gca',
            ],
            'tier_4_lower_importance': [
                # Discipline
                'home_indiscipline_score', 'away_indiscipline_score',
                'diff_cards_yellow', 'diff_fouls',
                # Set pieces
                'home_penalty_differential', 'away_penalty_differential',
                # Aerials
                'home_aerial_dominance', 'away_aerial_dominance',
            ],
        }


def create_feature_pipeline(
    raw_data_dir: str = "data/raw",
    processed_data_dir: str = "data/processed",
    config: Optional[FeatureConfig] = None
) -> FeatureEngineer:
    """
    Factory function to create a configured FeatureEngineer.
    
    Args:
        raw_data_dir: Directory containing raw scraped data
        processed_data_dir: Directory for output
        config: Optional feature engineering configuration
        
    Returns:
        Configured FeatureEngineer instance
    """
    return FeatureEngineer(config=config)
