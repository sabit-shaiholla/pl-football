#!/usr/bin/env python3
"""
FBREF Premier League Data Scraper - Main Entry Point

A comprehensive web scraper for collecting Premier League statistics from FBREF.
Designed for use in ML-based match prediction models.

Usage:
    python -m src.scraper.run_scraper --help
    python -m src.scraper.run_scraper scrape --seasons 2024-2025 2023-2024
    python -m src.scraper.run_scraper scrape --all-seasons
    python -m src.scraper.run_scraper process
    python -m src.scraper.run_scraper full-pipeline

Author: Lead Data Scientist / ML Engineer
"""

import argparse
import logging
import sys
from pathlib import Path

# Configure logging - terminal only, no file logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def scrape_command(args):
    """Execute the scraping command."""
    from . import FBREFScraper, ScraperSettings, SEASONS, TABLES_CONFIG, CURRENT_SEASON
    
    # Determine which seasons to scrape
    if getattr(args, 'all_seasons', False):
        seasons = SEASONS
    elif getattr(args, 'season', None):
        seasons = [args.season]  # Single season specified
    elif getattr(args, 'seasons', None):
        seasons = args.seasons
    else:
        seasons = [CURRENT_SEASON]  # Default to current season
        
    logger.info(f"Will scrape seasons: {seasons}")
    
    # Determine which tables to scrape
    if getattr(args, 'all_tables', False):
        tables = list(TABLES_CONFIG.keys())
    elif getattr(args, 'tables', None):
        tables = args.tables
    else:
        # Default: Priority tables for ML
        tables = [
            'standard_stats',
            'shooting',
            'goal_shot_creation',
            'defensive_actions',
            'passing',
            'possession',
            'goalkeeping',
            'misc',
        ]
        
    logger.info(f"Will scrape tables: {tables}")
    
    # Configure scraper settings
    output_dir = getattr(args, 'output_dir', None) or getattr(args, 'data_dir', 'data')
    settings = ScraperSettings(
        headless=not getattr(args, 'visible', False),
        min_delay=getattr(args, 'min_delay', 4.0),
        max_delay=getattr(args, 'max_delay', 8.0),
        output_dir=output_dir,
        raw_data_dir=f"{output_dir}/raw",
        processed_data_dir=f"{output_dir}/processed",
    )
    
    # Run scraper
    with FBREFScraper(settings) as scraper:
        all_data = scraper.scrape_all_seasons(
            seasons=seasons,
            tables=tables
        )
        
    logger.info("Scraping complete!")
    
    # Print summary
    for season, data in all_data.get('seasons', {}).items():
        tables_scraped = len(data.get('tables', {}))
        has_fixtures = data.get('fixtures') is not None
        logger.info(f"  {season}: {tables_scraped} tables, fixtures: {has_fixtures}")
        
    return all_data


def process_command(args):
    """
    Execute the data processing command.
    
    This processes already-scraped CSV files from the raw data directory.
    It does NOT perform any web scraping.
    """
    from . import DataProcessor, FeatureEngineer, SEASONS, CURRENT_SEASON
    
    processor = DataProcessor(
        raw_data_dir=f"{args.data_dir}/raw",
        processed_data_dir=f"{args.data_dir}/processed",
    )
    
    # Determine seasons to process
    if hasattr(args, 'season') and args.season:
        seasons = [args.season]  # Single season specified
    elif args.seasons:
        seasons = args.seasons
    else:
        seasons = SEASONS
        
    logger.info(f"Processing seasons: {seasons}")
    
    # Process all data from local CSV files
    processed_data = processor.process_all_seasons(seasons)
    
    if processed_data.empty:
        logger.warning("No data was processed!")
        return None
        
    logger.info(f"Processed {len(processed_data)} rows")
    
    # Apply feature engineering
    engineer = FeatureEngineer()
    
    if 'date' in processed_data.columns:
        logger.info("Adding Elo ratings...")
        processed_data = engineer.add_elo_ratings(processed_data)
        
        logger.info("Computing rolling form metrics...")
        processed_data = engineer.compute_rolling_form(processed_data, window=5)
        
    # Save final dataset
    output_path = Path(f"{args.data_dir}/processed/ml_ready_dataset.csv")
    processed_data.to_csv(output_path, index=False)
    logger.info(f"Saved ML-ready dataset to {output_path}")
    
    # Print feature summary
    numeric_cols = processed_data.select_dtypes(include=['number']).columns
    logger.info(f"Dataset contains {len(numeric_cols)} numeric features")
    
    return processed_data


def full_pipeline_command(args):
    """Execute full scraping and processing pipeline."""
    logger.info("=" * 60)
    logger.info("FBREF PREMIER LEAGUE DATA PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Scrape
    logger.info("\n[Step 1/2] Scraping data from FBREF...")
    args.output_dir = args.data_dir
    scrape_command(args)
    
    # Step 2: Process
    logger.info("\n[Step 2/2] Processing scraped data...")
    process_command(args)
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)


def validate_command(args):
    """Validate scraped data for completeness and accuracy."""
    from . import SEASONS, TABLES_CONFIG
    import pandas as pd
    
    data_dir = Path(args.data_dir) / "raw"
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Determine which seasons to validate
    if hasattr(args, 'season') and args.season:
        seasons_to_check = [args.season]
    else:
        seasons_to_check = SEASONS
        
    issues = []
    
    for season in seasons_to_check:
        season_dir = data_dir / season
        
        if not season_dir.exists():
            issues.append(f"Missing season directory: {season}")
            continue
            
        # Check for required files
        required_files = ['fixtures.csv', 'standard_stats_for.csv']
        for req_file in required_files:
            if not (season_dir / req_file).exists():
                issues.append(f"Missing {req_file} for {season}")
                
        # Validate data quality
        for csv_file in season_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                
                if df.empty:
                    issues.append(f"Empty file: {csv_file}")
                    continue
                    
                # Check for expected number of teams (20 for PL)
                if 'Squad' in df.columns or 'squad' in df.columns:
                    squad_col = 'Squad' if 'Squad' in df.columns else 'squad'
                    num_teams = df[squad_col].nunique()
                    if num_teams < 18:
                        issues.append(f"Only {num_teams} teams in {csv_file.name}")
                        
            except Exception as e:
                issues.append(f"Error reading {csv_file}: {e}")
                
    if issues:
        logger.warning("Data validation issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("All data validated successfully!")
        
    return issues


def main():
    parser = argparse.ArgumentParser(
        description='FBREF Premier League Data Scraper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape current season
  python -m src.scraper.run_scraper scrape

  # Scrape a specific season
  python -m src.scraper.run_scraper scrape --season 2024-2025

  # Scrape all seasons from 2018-2019
  python -m src.scraper.run_scraper scrape --all-seasons

  # Scrape multiple specific seasons
  python -m src.scraper.run_scraper scrape --seasons 2024-2025 2023-2024

  # Process a specific season (from existing CSV files)
  python -m src.scraper.run_scraper process --season 2024-2025

  # Full pipeline for a specific season
  python -m src.scraper.run_scraper full-pipeline --season 2024-2025

  # Validate data
  python -m src.scraper.run_scraper validate
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Scrape command
    scrape_parser = subparsers.add_parser('scrape', help='Scrape data from FBREF')
    scrape_parser.add_argument(
        '--season',
        type=str,
        help='Single season to scrape (e.g., 2024-2025)'
    )
    scrape_parser.add_argument(
        '--seasons', 
        nargs='+', 
        help='Multiple seasons to scrape (e.g., 2024-2025 2023-2024)'
    )
    scrape_parser.add_argument(
        '--all-seasons',
        action='store_true',
        help='Scrape all seasons from 2018-2019 to current'
    )
    scrape_parser.add_argument(
        '--tables',
        nargs='+',
        help='Specific tables to scrape'
    )
    scrape_parser.add_argument(
        '--all-tables',
        action='store_true',
        help='Scrape all available tables'
    )
    scrape_parser.add_argument(
        '--output-dir',
        default='data',
        help='Output directory for scraped data'
    )
    scrape_parser.add_argument(
        '--visible',
        action='store_true',
        help='Run browser in visible mode (not headless)'
    )
    scrape_parser.add_argument(
        '--min-delay',
        type=float,
        default=4.0,
        help='Minimum delay between requests (seconds)'
    )
    scrape_parser.add_argument(
        '--max-delay',
        type=float,
        default=8.0,
        help='Maximum delay between requests (seconds)'
    )
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process scraped data from local CSV files')
    process_parser.add_argument(
        '--data-dir',
        default='data',
        help='Data directory'
    )
    process_parser.add_argument(
        '--season',
        type=str,
        help='Single season to process (e.g., 2024-2025)'
    )
    process_parser.add_argument(
        '--seasons',
        nargs='+',
        help='Multiple seasons to process'
    )
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser('full-pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument(
        '--season',
        type=str,
        help='Single season to scrape and process (e.g., 2024-2025)'
    )
    pipeline_parser.add_argument(
        '--seasons',
        nargs='+',
        help='Multiple seasons to scrape and process'
    )
    pipeline_parser.add_argument(
        '--all-seasons',
        action='store_true',
        help='Process all seasons'
    )
    pipeline_parser.add_argument(
        '--all-tables',
        action='store_true',
        help='Scrape all tables'
    )
    pipeline_parser.add_argument(
        '--data-dir',
        default='data',
        help='Data directory'
    )
    pipeline_parser.add_argument(
        '--visible',
        action='store_true',
        help='Run browser in visible mode'
    )
    pipeline_parser.add_argument(
        '--min-delay',
        type=float,
        default=4.0,
        help='Minimum delay between requests'
    )
    pipeline_parser.add_argument(
        '--max-delay',
        type=float,
        default=8.0,
        help='Maximum delay between requests'
    )
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate scraped data')
    validate_parser.add_argument(
        '--data-dir',
        default='data',
        help='Data directory'
    )
    validate_parser.add_argument(
        '--season',
        type=str,
        help='Single season to validate (e.g., 2024-2025)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'scrape':
        scrape_command(args)
    elif args.command == 'process':
        process_command(args)
    elif args.command == 'full-pipeline':
        full_pipeline_command(args)
    elif args.command == 'validate':
        validate_command(args)
    else:
        parser.print_help()
        

if __name__ == '__main__':
    main()
