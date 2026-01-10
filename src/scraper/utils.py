"""
Utility functions for the FBREF scraper.
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def create_cache_key(url: str, params: Optional[Dict] = None) -> str:
    """Create a unique cache key for a URL and parameters."""
    key_string = url
    if params:
        key_string += json.dumps(params, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()


def load_cached_data(cache_dir: str, cache_key: str, 
                     max_age_hours: int = 24) -> Optional[pd.DataFrame]:
    """
    Load cached data if it exists and is not expired.
    
    Args:
        cache_dir: Directory containing cached files
        cache_key: Unique key for this cached item
        max_age_hours: Maximum age in hours before cache expires
        
    Returns:
        DataFrame if valid cache exists, None otherwise
    """
    cache_path = Path(cache_dir) / f"{cache_key}.csv"
    
    if not cache_path.exists():
        return None
        
    # Check age
    file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age_hours = (datetime.now() - file_time).total_seconds() / 3600
    
    if age_hours > max_age_hours:
        logger.debug(f"Cache expired for {cache_key}")
        return None
        
    try:
        return pd.read_csv(cache_path)
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None


def save_to_cache(df: pd.DataFrame, cache_dir: str, cache_key: str) -> None:
    """Save DataFrame to cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_path = cache_dir / f"{cache_key}.csv"
    df.to_csv(cache_path, index=False)
    logger.debug(f"Saved to cache: {cache_path}")


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters."""
    if not isinstance(text, str):
        return str(text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special Unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text


def parse_score(score_str: str) -> tuple:
    """
    Parse a score string like '2-1' or '2–1' into (home_goals, away_goals).
    
    Returns:
        Tuple of (home_goals, away_goals) or (None, None) if parsing fails
    """
    if not score_str or pd.isna(score_str):
        return (None, None)
        
    # Handle various dash types
    score_str = str(score_str).replace('–', '-').replace('—', '-')
    
    match = re.match(r'(\d+)\s*-\s*(\d+)', score_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))
        
    return (None, None)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if pd.isna(value):
        return default
    try:
        # Remove commas for numbers like "1,234"
        if isinstance(value, str):
            value = value.replace(',', '')
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int."""
    return int(safe_float(value, float(default)))


def standardize_team_name(name: str) -> str:
    """
    Standardize Premier League team names to a canonical form.
    
    This handles various abbreviations and alternate names used by FBREF.
    """
    if not name or pd.isna(name):
        return name
        
    name = str(name).strip()
    
    # Canonical mappings
    mappings = {
        # Full name variations
        "Manchester Utd": "Manchester United",
        "Manchester United": "Manchester United",
        "Man United": "Manchester United",
        "Man Utd": "Manchester United",
        
        "Manchester City": "Manchester City",
        "Man City": "Manchester City",
        
        "Newcastle Utd": "Newcastle United",
        "Newcastle United": "Newcastle United",
        
        "Nott'ham Forest": "Nottingham Forest",
        "Nottingham Forest": "Nottingham Forest",
        "Nott'm Forest": "Nottingham Forest",
        
        "Tottenham": "Tottenham Hotspur",
        "Tottenham Hotspur": "Tottenham Hotspur",
        "Spurs": "Tottenham Hotspur",
        
        "West Ham": "West Ham United",
        "West Ham United": "West Ham United",
        
        "Wolves": "Wolverhampton Wanderers",
        "Wolverhampton": "Wolverhampton Wanderers",
        "Wolverhampton Wanderers": "Wolverhampton Wanderers",
        
        "Brighton": "Brighton & Hove Albion",
        "Brighton and Hove Albion": "Brighton & Hove Albion",
        "Brighton & Hove Albion": "Brighton & Hove Albion",
        
        "Leicester": "Leicester City",
        "Leicester City": "Leicester City",
        
        "Leeds": "Leeds United",
        "Leeds United": "Leeds United",
        
        "Sheffield Utd": "Sheffield United",
        "Sheffield United": "Sheffield United",
        
        "Luton": "Luton Town",
        "Luton Town": "Luton Town",
        
        "Ipswich": "Ipswich Town",
        "Ipswich Town": "Ipswich Town",
        
        # Standard names (no change needed)
        "Arsenal": "Arsenal",
        "Aston Villa": "Aston Villa",
        "Bournemouth": "AFC Bournemouth",
        "AFC Bournemouth": "AFC Bournemouth",
        "Brentford": "Brentford",
        "Chelsea": "Chelsea",
        "Crystal Palace": "Crystal Palace",
        "Everton": "Everton",
        "Fulham": "Fulham",
        "Liverpool": "Liverpool",
        "Southampton": "Southampton",
        "Burnley": "Burnley",
        "Watford": "Watford",
        "Norwich": "Norwich City",
        "Norwich City": "Norwich City",
        "West Brom": "West Bromwich Albion",
        "West Bromwich Albion": "West Bromwich Albion",
        "Huddersfield": "Huddersfield Town",
        "Huddersfield Town": "Huddersfield Town",
        "Cardiff": "Cardiff City",
        "Cardiff City": "Cardiff City",
    }
    
    return mappings.get(name, name)


def get_season_from_date(date: Union[str, datetime]) -> str:
    """
    Determine the Premier League season from a match date.
    
    PL seasons run from August to May, so:
    - Aug 2024 - May 2025 = "2024-2025"
    
    Args:
        date: Date string or datetime object
        
    Returns:
        Season string like "2024-2025"
    """
    if isinstance(date, str):
        try:
            date = pd.to_datetime(date)
        except:
            return None
            
    if pd.isna(date):
        return None
        
    year = date.year
    month = date.month
    
    # Season starts in August (month 8)
    if month >= 8:
        return f"{year}-{year + 1}"
    else:
        return f"{year - 1}-{year}"


def calculate_form_points(results: List[str], last_n: int = 5) -> int:
    """
    Calculate points from last N results.
    
    Args:
        results: List of 'W', 'D', 'L' strings (most recent first)
        last_n: Number of matches to consider
        
    Returns:
        Total points from last N matches
    """
    point_map = {'W': 3, 'D': 1, 'L': 0}
    return sum(point_map.get(r, 0) for r in results[:last_n])


def calculate_goal_difference(goals_for: List[int], 
                              goals_against: List[int],
                              last_n: int = 5) -> int:
    """Calculate goal difference from last N matches."""
    gf = sum(goals_for[:last_n]) if goals_for else 0
    ga = sum(goals_against[:last_n]) if goals_against else 0
    return gf - ga


def merge_dataframes_on_team(dfs: List[pd.DataFrame],
                             team_col: str = 'squad',
                             suffixes: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Merge multiple DataFrames on team/squad column.
    
    Args:
        dfs: List of DataFrames to merge
        team_col: Column name containing team names
        suffixes: List of suffixes for each DataFrame
        
    Returns:
        Merged DataFrame
    """
    if not dfs:
        return pd.DataFrame()
        
    if len(dfs) == 1:
        return dfs[0].copy()
        
    if suffixes is None:
        suffixes = [f'_{i}' for i in range(len(dfs))]
        
    result = dfs[0].copy()
    
    for i, df in enumerate(dfs[1:], 1):
        result = result.merge(
            df,
            on=team_col,
            how='outer',
            suffixes=('', suffixes[i])
        )
        
    return result


def export_to_excel(dataframes: Dict[str, pd.DataFrame],
                    output_path: str) -> None:
    """
    Export multiple DataFrames to an Excel file with separate sheets.
    
    Args:
        dataframes: Dictionary of {sheet_name: DataFrame}
        output_path: Path for the Excel file
    """
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, df in dataframes.items():
                # Excel sheet names max 31 chars
                sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
        logger.info(f"Exported to {output_path}")
    except ImportError:
        logger.error("openpyxl not installed. Install with: pip install openpyxl")
    except Exception as e:
        logger.error(f"Failed to export to Excel: {e}")


def print_data_summary(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print a summary of a DataFrame."""
    print(f"\n{'=' * 50}")
    print(f"Summary: {name}")
    print(f"{'=' * 50}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    missing_info = pd.DataFrame({'count': missing, 'pct': missing_pct})
    missing_info = missing_info[missing_info['count'] > 0].sort_values('count', ascending=False)
    if not missing_info.empty:
        print(missing_info.head(10))
    else:
        print("  No missing values!")
    print(f"{'=' * 50}\n")
