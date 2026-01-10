"""
Match Prediction Module for Premier League.

This module provides functionality to predict outcomes for upcoming matches
using trained ML models.
"""

import argparse
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MatchPrediction:
    """Container for match prediction results."""
    
    home_team: str
    away_team: str
    date: str
    
    # Probabilities
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    
    # Predicted outcome
    predicted_outcome: str  # 'Home Win', 'Draw', 'Away Win'
    confidence: float
    
    # Additional info
    home_elo: Optional[float] = None
    away_elo: Optional[float] = None
    elo_expected_home: Optional[float] = None
    
    # Score predictions (Poisson model)
    predicted_home_goals: Optional[int] = None
    predicted_away_goals: Optional[int] = None
    score_probability: Optional[float] = None
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    top_scores: Optional[List[Tuple[int, int, float]]] = None
    
    # Betting market predictions
    over_2_5_prob: Optional[float] = None
    btts_prob: Optional[float] = None
    
    def __str__(self) -> str:
        """Pretty print the prediction."""
        lines = [
            "=" * 60,
            f"MATCH PREDICTION: {self.home_team} vs {self.away_team}",
            f"Date: {self.date}",
            "=" * 60,
            "",
            "PREDICTED OUTCOME:",
            f"  >>> {self.predicted_outcome} (Confidence: {self.confidence:.1%}) <<<",
            "",
            "WIN PROBABILITIES:",
            f"  Home Win ({self.home_team}): {self.home_win_prob:.1%}",
            f"  Draw:                        {self.draw_prob:.1%}",
            f"  Away Win ({self.away_team}): {self.away_win_prob:.1%}",
        ]
        
        # Add score predictions if available
        if self.predicted_home_goals is not None:
            lines.extend([
                "",
                "SCORE PREDICTION (Poisson Model):",
                f"  Expected Score: {self.home_team} {self.predicted_home_goals} - {self.predicted_away_goals} {self.away_team}",
                f"  Score Probability: {self.score_probability:.1%}" if self.score_probability else "",
            ])
            
            if self.home_xg is not None:
                lines.append(f"  Expected Goals: {self.home_xg:.2f} - {self.away_xg:.2f}")
            
            if self.top_scores:
                lines.append("")
                lines.append("TOP 5 MOST LIKELY SCORES:")
                for i, (h, a, prob) in enumerate(self.top_scores[:5], 1):
                    lines.append(f"  {i}. {h}-{a}: {prob:.1%}")
        
        # Add betting market predictions
        if self.over_2_5_prob is not None:
            lines.extend([
                "",
                "BETTING MARKETS:",
                f"  Over 2.5 Goals: {self.over_2_5_prob:.1%}",
                f"  Under 2.5 Goals: {1 - self.over_2_5_prob:.1%}",
            ])
            
        if self.btts_prob is not None:
            lines.append(f"  Both Teams To Score: {self.btts_prob:.1%}")
        
        if self.home_elo is not None:
            lines.extend([
                "",
                "ELO RATINGS:",
                f"  {self.home_team}: {self.home_elo:.0f}",
                f"  {self.away_team}: {self.away_elo:.0f}",
                f"  ELO Expected Home Win: {self.elo_expected_home:.1%}",
            ])
            
        lines.append("=" * 60)
        return "\n".join(lines)


class MatchPredictor:
    """
    Predictor for upcoming Premier League matches.
    
    Uses trained ML models and current team statistics to predict outcomes.
    Includes score prediction using Poisson distribution.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        score_model_path: Optional[str] = None,
        data_dir: str = "data/raw",
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved outcome model file. If None, uses default.
            score_model_path: Path to saved score predictor. If None, uses default.
            data_dir: Directory containing raw team statistics.
        """
        self.model_path = model_path or "models/ensemble_match_result.pkl"
        self.score_model_path = score_model_path or "models/score_predictor.pkl"
        self.data_dir = Path(data_dir)
        
        self.model = None
        self.score_predictor = None
        self.feature_columns = None
        self.team_stats = None
        self.elo_ratings = None
        
    def load_model(self) -> None:
        """Load the trained outcome model."""
        model_file = Path(self.model_path)
        
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train a model first using: "
                "uv run python -m src.ml.train --target match_result --model ensemble"
            )
            
        with open(model_file, 'rb') as f:
            saved = pickle.load(f)
        
        # Handle both ensemble format (models) and single model format (model)
        if 'models' in saved:
            # Ensemble format - create an EnsemblePredictor wrapper
            from src.ml.models import EnsemblePredictor
            ensemble = EnsemblePredictor()
            ensemble.models = saved['models']
            ensemble.weights = saved.get('weights')
            ensemble.method = saved.get('method', 'weighted')
            ensemble.meta_model = saved.get('meta_model')
            ensemble.model_performances = saved.get('model_performances', {})
            ensemble.feature_names = saved.get('feature_names', [])
            ensemble.is_fitted = saved.get('is_fitted', True)
            self.model = ensemble
        else:
            # Single model format
            self.model = saved['model']
        
        self.feature_columns = saved.get('feature_names', saved.get('feature_columns', []))
        
        logger.info(f"Loaded outcome model from {self.model_path}")
        
    def load_score_predictor(self) -> None:
        """Load the trained score predictor (Poisson-based)."""
        score_model_file = Path(self.score_model_path)
        
        if not score_model_file.exists():
            logger.warning(
                f"Score predictor not found at {self.score_model_path}. "
                "Score predictions will not be available. "
                "Train with: uv run python -m src.ml.train --target match_result --model ensemble"
            )
            return
            
        try:
            from src.ml.models import ScorePredictor
            self.score_predictor = ScorePredictor()
            self.score_predictor.load(str(score_model_file))
            logger.info(f"Loaded score predictor from {self.score_model_path}")
        except Exception as e:
            logger.warning(f"Failed to load score predictor: {e}")
            self.score_predictor = None
        
    def load_team_stats(self, season: str = "2024-2025") -> None:
        """
        Load current team statistics.
        
        Args:
            season: Season to load stats for (e.g., "2024-2025")
        """
        season_dir = self.data_dir / season
        
        if not season_dir.exists():
            # Fall back to most recent available season
            available = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
            if available:
                season_dir = available[-1]
                logger.warning(f"Season {season} not found, using {season_dir.name}")
            else:
                raise FileNotFoundError(f"No season data found in {self.data_dir}")
        
        # Load all stat files and merge
        stat_files = {
            'standard_f': 'standard_stats_for.csv',
            'standard_a': 'standard_stats_against.csv',
            'shooting_f': 'shooting_for.csv',
            'shooting_a': 'shooting_against.csv',
            'passing_f': 'passing_for.csv',
            'passing_a': 'passing_against.csv',
            'possession_f': 'possession_for.csv',
            'possession_a': 'possession_against.csv',
            'defensive_f': 'defensive_actions_for.csv',
            'defensive_a': 'defensive_actions_against.csv',
            'gca_f': 'goal_shot_creation_for.csv',
            'gca_a': 'goal_shot_creation_against.csv',
            'goalkeeping_f': 'goalkeeping_for.csv',
            'goalkeeping_a': 'goalkeeping_against.csv',
            'misc_f': 'misc_for.csv',
            'misc_a': 'misc_against.csv',
        }
        
        base_df = None
        
        for stat_key, filename in stat_files.items():
            file_path = season_dir / filename
            if not file_path.exists():
                continue
                
            df = pd.read_csv(file_path)
            
            # Find team column
            team_col = None
            for col in df.columns:
                if col.lower() in ['team', 'squad']:
                    team_col = col
                    break
            if not team_col:
                continue
            
            # Clean team names (remove "vs " prefix from against stats)
            df[team_col] = df[team_col].str.replace(r'^vs\s+', '', regex=True).str.strip()
            
            # Rename columns with prefix
            rename_dict = {
                col: f"{stat_key}_{col}"
                for col in df.columns
                if col != team_col
            }
            df = df.rename(columns=rename_dict)
            
            if base_df is None:
                base_df = df.rename(columns={team_col: 'team'})
            else:
                df = df.rename(columns={team_col: 'team'})
                base_df = base_df.merge(df, on='team', how='outer')
        
        if base_df is not None:
            base_df['season'] = season_dir.name
            self.team_stats = base_df
            logger.info(f"Loaded stats for {len(base_df)} teams from {season_dir.name}")
        else:
            raise ValueError("Could not load team statistics")
            
    def _calculate_elo_ratings(self) -> Dict[str, float]:
        """
        Calculate ELO ratings from historical fixtures.
        
        Returns default ratings if historical data isn't available.
        """
        # Default ELO ratings based on typical Premier League hierarchy
        default_ratings = {
            'Manchester City': 1850,
            'Arsenal': 1820,
            'Liverpool': 1810,
            'Manchester Utd': 1720,
            'Manchester United': 1720,
            'Chelsea': 1750,
            'Tottenham': 1720,
            'Newcastle Utd': 1700,
            'Newcastle United': 1700,
            'Aston Villa': 1680,
            'Brighton': 1660,
            'West Ham': 1640,
            'Bournemouth': 1600,
            'Fulham': 1600,
            'Wolves': 1600,
            'Crystal Palace': 1590,
            'Brentford': 1600,
            'Everton': 1580,
            'Nott\'ham Forest': 1590,
            'Nottingham Forest': 1590,
            'Leicester City': 1570,
            'Ipswich Town': 1500,
            'Southampton': 1500,
        }
        
        # Try to load historical fixtures and calculate actual ELO
        try:
            from src.ml.feature_engineering import FeatureEngineer
            
            # Load all historical fixtures
            all_fixtures = []
            for season_dir in sorted(self.data_dir.iterdir()):
                if not season_dir.is_dir():
                    continue
                fixtures_file = season_dir / 'fixtures.csv'
                if fixtures_file.exists():
                    df = pd.read_csv(fixtures_file)
                    df['season'] = season_dir.name
                    all_fixtures.append(df)
                    
            if all_fixtures:
                fixtures = pd.concat(all_fixtures, ignore_index=True)
                
                # Use feature engineer to calculate ELO
                fe = FeatureEngineer()
                elo_ratings = fe.calculate_elo_ratings(fixtures)
                
                if elo_ratings:
                    logger.info("Calculated ELO ratings from historical data")
                    return elo_ratings
                    
        except Exception as e:
            logger.warning(f"Could not calculate ELO from history: {e}")
            
        return default_ratings
    
    def _normalize_team_name(self, name: str) -> str:
        """Normalize team name to match database."""
        # Common name mappings
        mappings = {
            'Manchester United': 'Manchester Utd',
            'Manchester City': 'Manchester City',
            'Newcastle United': 'Newcastle Utd',
            'Nottingham Forest': "Nott'ham Forest",
            'Tottenham Hotspur': 'Tottenham',
            'West Ham United': 'West Ham',
            'Brighton and Hove Albion': 'Brighton',
            'Brighton & Hove Albion': 'Brighton',
            'Wolverhampton Wanderers': 'Wolves',
            'Sheffield United': 'Sheffield Utd',
            'Leeds United': 'Leeds United',
        }
        return mappings.get(name, name)
    
    def _prepare_match_features(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
    ) -> pd.DataFrame:
        """
        Prepare feature vector for a match prediction.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Date of the match
            
        Returns:
            DataFrame with features for the match
        """
        if self.team_stats is None:
            self.load_team_stats()
            
        # Normalize team names
        home_team_norm = self._normalize_team_name(home_team)
        away_team_norm = self._normalize_team_name(away_team)
        
        # Get team stats
        home_stats = self.team_stats[self.team_stats['team'] == home_team_norm]
        away_stats = self.team_stats[self.team_stats['team'] == away_team_norm]
        
        if home_stats.empty:
            # Try fuzzy match
            for team in self.team_stats['team'].unique():
                if home_team_norm.lower() in team.lower() or team.lower() in home_team_norm.lower():
                    home_stats = self.team_stats[self.team_stats['team'] == team]
                    logger.info(f"Matched '{home_team}' to '{team}'")
                    break
                    
        if away_stats.empty:
            for team in self.team_stats['team'].unique():
                if away_team_norm.lower() in team.lower() or team.lower() in away_team_norm.lower():
                    away_stats = self.team_stats[self.team_stats['team'] == team]
                    logger.info(f"Matched '{away_team}' to '{team}'")
                    break
        
        if home_stats.empty:
            raise ValueError(f"Team not found: {home_team}. Available: {self.team_stats['team'].unique()}")
        if away_stats.empty:
            raise ValueError(f"Team not found: {away_team}. Available: {self.team_stats['team'].unique()}")
            
        # Build feature vector
        features = {}
        
        # Add home team stats with 'home_' prefix
        for col in home_stats.columns:
            if col not in ['team', 'season']:
                # Rename to match expected format
                new_col = f"home_{col}"
                features[new_col] = home_stats[col].values[0]
                
        # Add away team stats with 'away_' prefix
        for col in away_stats.columns:
            if col not in ['team', 'season']:
                new_col = f"away_{col}"
                features[new_col] = away_stats[col].values[0]
                
        # Add differential features
        for home_col in list(features.keys()):
            if home_col.startswith('home_'):
                away_col = home_col.replace('home_', 'away_')
                if away_col in features:
                    try:
                        home_val = float(features[home_col]) if pd.notna(features[home_col]) else 0
                        away_val = float(features[away_col]) if pd.notna(features[away_col]) else 0
                        diff_col = home_col.replace('home_', 'diff_')
                        features[diff_col] = home_val - away_val
                    except (ValueError, TypeError):
                        pass
                        
        # Add ELO features
        elo_ratings = self._calculate_elo_ratings()
        home_elo = elo_ratings.get(home_team_norm, elo_ratings.get(home_team, 1500))
        away_elo = elo_ratings.get(away_team_norm, elo_ratings.get(away_team, 1500))
        
        features['home_elo'] = home_elo
        features['away_elo'] = away_elo
        features['elo_diff'] = home_elo - away_elo
        
        # ELO expected probability
        elo_expected = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        features['elo_home_win_prob'] = elo_expected
        
        self.elo_ratings = {'home': home_elo, 'away': away_elo, 'expected': elo_expected}
        
        # Add form features (default to neutral if not available)
        for window in [3, 5, 10]:
            features[f'home_form_{window}'] = 1.5  # Neutral form (1.5 points per game)
            features[f'away_form_{window}'] = 1.5
            features[f'home_xg_form_{window}'] = 0.0  # Neutral xG differential
            features[f'away_xg_form_{window}'] = 0.0
            
        # Add H2H features (default to neutral)
        features['h2h_home_wins'] = 0
        features['h2h_away_wins'] = 0
        features['h2h_draws'] = 0
        features['h2h_home_goals_avg'] = 1.0
        features['h2h_away_goals_avg'] = 1.0
        features['h2h_home_dominance'] = 0.5
        
        # Create DataFrame
        match_df = pd.DataFrame([features])
        
        # Ensure we have all required feature columns
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in match_df.columns:
                    match_df[col] = 0  # Default to 0 for missing features
                    
            # Select only the feature columns in the right order
            match_df = match_df.reindex(columns=self.feature_columns, fill_value=0)
            
        return match_df
    
    def predict(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[str] = None,
    ) -> MatchPrediction:
        """
        Predict the outcome of a match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Date of the match (YYYY-MM-DD format)
            
        Returns:
            MatchPrediction object with probabilities and outcome
        """
        if self.model is None:
            self.load_model()
            
        if match_date is None:
            match_date = datetime.now().strftime("%Y-%m-%d")
        else:
            # Parse and reformat date
            try:
                dt = datetime.strptime(match_date, "%Y-%m-%d")
            except ValueError:
                dt = datetime.strptime(match_date, "%B %d, %Y")
            match_date = dt.strftime("%Y-%m-%d")
            
        # Prepare features
        features = self._prepare_match_features(
            home_team, 
            away_team,
            datetime.strptime(match_date, "%Y-%m-%d")
        )
        
        # Get predictions
        probas = self.model.predict_proba(features)[0]
        
        # Map class indices to outcomes
        # Classes: 0 = Away Win, 1 = Draw, 2 = Home Win
        away_win_prob = probas[0]
        draw_prob = probas[1]
        home_win_prob = probas[2]
        
        # Determine predicted outcome
        predicted_class = np.argmax(probas)
        outcomes = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        predicted_outcome = outcomes[predicted_class]
        confidence = probas[predicted_class]
        
        # Get score predictions if score predictor is available
        predicted_home_goals = None
        predicted_away_goals = None
        score_probability = None
        home_xg = None
        away_xg = None
        top_scores = None
        over_2_5_prob = None
        btts_prob = None
        
        if self.score_predictor is not None:
            try:
                # Get expected goals (returns arrays, get first element)
                home_xg_arr, away_xg_arr = self.score_predictor.predict_expected_goals(features)
                home_xg = float(home_xg_arr[0])
                away_xg = float(away_xg_arr[0])
                
                # Get most likely score (returns 3 arrays: home_scores, away_scores, probabilities)
                home_goals_arr, away_goals_arr, prob_arr = \
                    self.score_predictor.predict_most_likely_score(features)
                predicted_home_goals = int(home_goals_arr[0])
                predicted_away_goals = int(away_goals_arr[0])
                score_probability = float(prob_arr[0])
                
                # Get top 5 scores (returns list for each sample)
                top_scores_result = self.score_predictor.predict_top_scores(features, n_top=5)
                if top_scores_result and len(top_scores_result) > 0:
                    top_scores = top_scores_result[0]  # Get first sample's top scores
                
                # Get betting market predictions
                # predict_over_under returns (under_probs, over_probs) arrays
                under_probs, over_probs_arr = self.score_predictor.predict_over_under(features, threshold=2.5)
                over_2_5_prob = float(over_probs_arr[0])
                
                # predict_btts returns (no_btts_probs, btts_probs) arrays
                no_btts_arr, btts_arr = self.score_predictor.predict_btts(features)
                btts_prob = float(btts_arr[0])
                
            except Exception as e:
                logger.warning(f"Score prediction failed: {e}")
                import traceback
                traceback.print_exc()
        
        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            date=match_date,
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            home_elo=self.elo_ratings.get('home') if self.elo_ratings else None,
            away_elo=self.elo_ratings.get('away') if self.elo_ratings else None,
            elo_expected_home=self.elo_ratings.get('expected') if self.elo_ratings else None,
            predicted_home_goals=predicted_home_goals,
            predicted_away_goals=predicted_away_goals,
            score_probability=score_probability,
            home_xg=home_xg,
            away_xg=away_xg,
            top_scores=top_scores,
            over_2_5_prob=over_2_5_prob,
            btts_prob=btts_prob,
        )
    
    def predict_multiple(
        self,
        matches: List[Tuple[str, str, str]],
    ) -> List[MatchPrediction]:
        """
        Predict outcomes for multiple matches.
        
        Args:
            matches: List of (home_team, away_team, date) tuples
            
        Returns:
            List of MatchPrediction objects
        """
        predictions = []
        for home, away, date in matches:
            try:
                pred = self.predict(home, away, date)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Failed to predict {home} vs {away}: {e}")
                
        return predictions


def main():
    """Command-line interface for match prediction."""
    parser = argparse.ArgumentParser(description='Predict Premier League match outcomes')
    parser.add_argument('--home', type=str, required=True,
                       help='Home team name')
    parser.add_argument('--away', type=str, required=True,
                       help='Away team name')
    parser.add_argument('--date', type=str, default=None,
                       help='Match date (YYYY-MM-DD)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained outcome model')
    parser.add_argument('--score-model', type=str, default=None,
                       help='Path to trained score predictor')
    parser.add_argument('--season', type=str, default='2024-2025',
                       help='Season for team stats')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Create predictor
    predictor = MatchPredictor(
        model_path=args.model,
        score_model_path=args.score_model,
    )
    
    # Load models and team stats
    predictor.load_model()
    predictor.load_score_predictor()
    predictor.load_team_stats(args.season)
    
    # Make prediction
    prediction = predictor.predict(
        home_team=args.home,
        away_team=args.away,
        match_date=args.date,
    )
    
    # Print result
    print(prediction)


if __name__ == '__main__':
    main()
