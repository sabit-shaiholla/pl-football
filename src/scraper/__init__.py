"""
FBREF Scraper Package for Premier League Data.

This package provides comprehensive tools for scraping football statistics
from FBREF and preparing data for ML-based match prediction models.
"""

from .config import (
    CURRENT_SEASON,
    SEASONS,
    TABLES_CONFIG,
    FEATURE_GROUPS,
    ScraperSettings,
    TableConfig,
)
from .fbref_scraper import FBREFScraper, FBREFDataExtractor
from .data_processor import DataProcessor, FeatureEngineer
from .utils import (
    standardize_team_name,
    parse_score,
    safe_float,
    safe_int,
    get_season_from_date,
    calculate_form_points,
    print_data_summary,
)

__all__ = [
    # Core classes
    'FBREFScraper',
    'FBREFDataExtractor',
    'DataProcessor',
    'FeatureEngineer',
    # Configuration
    'CURRENT_SEASON',
    'SEASONS',
    'TABLES_CONFIG',
    'FEATURE_GROUPS',
    'ScraperSettings',
    'TableConfig',
    # Utilities
    'standardize_team_name',
    'parse_score',
    'safe_float',
    'safe_int',
    'get_season_from_date',
    'calculate_form_points',
    'print_data_summary',
]

__version__ = "1.0.0"
