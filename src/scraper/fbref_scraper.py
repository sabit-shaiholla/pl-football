"""
Core FBREF Scraper using Selenium with anti-detection measures.
Handles 403 errors, rate limiting, and robust data extraction.

COMPLETELY REWRITTEN for:
1. Accurate data extraction using FBREF's data-stat attributes
2. Faster scraping by extracting multiple tables from single page loads
3. Exponential backoff with jitter for robust error handling
"""

import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
)
from webdriver_manager.chrome import ChromeDriverManager

from .config import (
    FBREF_BASE_URL,
    PREMIER_LEAGUE_COMP_ID,
    CURRENT_SEASON,
    SEASONS,
    TABLES_CONFIG,
    TABLES_WITH_AGAINST_STATS,
    FIXTURES_CONFIG,
    ScraperSettings,
    TableConfig,
)

logger = logging.getLogger(__name__)


class ExponentialBackoff:
    """
    Exponential backoff with jitter for robust retry logic.
    """
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, 
                 max_retries: int = 5, jitter: float = 0.5):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.jitter = jitter
        
    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = self.base_delay * (2 ** attempt)
        delay = min(delay, self.max_delay)
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        return max(0.1, delay)
    
    def wait(self, attempt: int) -> None:
        """Wait with exponential backoff."""
        delay = self.get_delay(attempt)
        logger.info(f"Backoff wait: {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})")
        time.sleep(delay)


class FBREFScraper:
    """
    Robust FBREF scraper using Selenium to bypass anti-scraping measures.
    
    Key features:
    - Uses data-stat attributes for accurate column mapping
    - Extracts multiple tables from single page load (faster)
    - Exponential backoff with jitter for robust retries
    - Proper handling of FBREF's commented-out tables
    """
    
    def __init__(self, settings: Optional[ScraperSettings] = None):
        """Initialize the scraper with optional custom settings."""
        self.settings = settings or ScraperSettings()
        self.driver: Optional[webdriver.Chrome] = None
        self.backoff = ExponentialBackoff(
            base_delay=2.0,
            max_delay=120.0,
            max_retries=self.settings.max_retries,
            jitter=0.5
        )
        self._setup_directories()
        self._request_count = 0
        self._last_request_time = 0
        
    def _setup_directories(self) -> None:
        """Create output directories if they don't exist."""
        for dir_path in [self.settings.output_dir, 
                         self.settings.raw_data_dir, 
                         self.settings.processed_data_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    def _get_random_user_agent(self) -> str:
        """Return a random user agent string."""
        return random.choice(self.settings.user_agents)
    
    def _smart_delay(self) -> None:
        """
        Implement smart rate limiting with variable delays.
        """
        self._request_count += 1
        base = random.uniform(self.settings.min_delay, self.settings.max_delay)
        
        # Add extra delay every N requests to avoid detection patterns
        if self._request_count % 5 == 0:
            base += random.uniform(2, 5)
        if self._request_count % 10 == 0:
            base += random.uniform(5, 10)
            
        # Ensure minimum time between requests
        elapsed = time.time() - self._last_request_time
        if elapsed < self.settings.min_delay:
            base += (self.settings.min_delay - elapsed)
            
        logger.debug(f"Smart delay: {base:.2f}s (request #{self._request_count})")
        time.sleep(base)
        self._last_request_time = time.time()
        
    def _setup_driver(self) -> webdriver.Chrome:
        """Configure Chrome WebDriver with enhanced anti-detection."""
        chrome_options = Options()
        
        user_agent = self._get_random_user_agent()
        chrome_options.add_argument(f"user-agent={user_agent}")
        
        if self.settings.headless:
            chrome_options.add_argument("--headless=new")
        
        chrome_options.add_argument(
            f"--window-size={self.settings.window_size[0]},{self.settings.window_size[1]}"
        )
        
        # Enhanced stealth settings
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--lang=en-US,en;q=0.9")
        chrome_options.add_argument("--enable-features=NetworkService,NetworkServiceInProcess")
        chrome_options.add_argument("--disable-features=IsolateOrigins,site-per-process")
        
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        prefs = {
            "profile.default_content_setting_values.notifications": 2,
            "credentials_enable_service": False,
            "profile.password_manager_enabled": False,
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Enhanced anti-detection scripts
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => {
                            const plugins = [
                                {name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer'},
                                {name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai'},
                                {name: 'Native Client', filename: 'internal-nacl-plugin'}
                            ];
                            plugins.length = 3;
                            return plugins;
                        }
                    });
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en']
                    });
                    window.chrome = {
                        runtime: {},
                        loadTimes: function() {},
                        csi: function() {},
                        app: {}
                    };
                    const originalQuery = window.navigator.permissions.query;
                    window.navigator.permissions.query = (parameters) => (
                        parameters.name === 'notifications' ?
                            Promise.resolve({ state: Notification.permission }) :
                            originalQuery(parameters)
                    );
                """
            }
        )
        
        driver.set_page_load_timeout(self.settings.page_load_timeout)
        
        return driver
    
    def start(self) -> None:
        """Start the browser session."""
        if self.driver is None:
            logger.info("Starting Chrome browser...")
            self.driver = self._setup_driver()
            self._request_count = 0
            
    def stop(self) -> None:
        """Stop the browser session."""
        if self.driver:
            logger.info("Closing browser...")
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
            
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    def _navigate_with_retry(self, url: str) -> bool:
        """Navigate to URL with exponential backoff retry."""
        for attempt in range(self.backoff.max_retries):
            try:
                logger.info(f"Navigating to: {url}")
                self.driver.get(url)
                
                WebDriverWait(self.driver, self.settings.page_load_timeout).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )
                
                page_source = self.driver.page_source.lower()
                title = self.driver.title.lower()
                
                if any(x in title for x in ['403', 'forbidden', 'blocked', 'denied']):
                    raise WebDriverException("403 Forbidden detected in title")
                if any(x in page_source for x in ['access denied', 'rate limit']):
                    raise WebDriverException("Access blocked detected in page")
                    
                time.sleep(1.5)
                return True
                
            except (TimeoutException, WebDriverException) as e:
                logger.warning(f"Navigation failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.backoff.max_retries - 1:
                    self.backoff.wait(attempt)
                    
                    if attempt >= 2:
                        logger.info("Restarting browser for fresh session...")
                        self.stop()
                        time.sleep(random.uniform(5, 10))
                        self.start()
                else:
                    logger.error(f"Max retries reached for URL: {url}")
                    return False
                    
        return False
    
    def _get_page_soup(self) -> BeautifulSoup:
        """Get BeautifulSoup object from current page."""
        return BeautifulSoup(self.driver.page_source, 'lxml')
    
    def _uncomment_tables(self, soup: BeautifulSoup) -> BeautifulSoup:
        """
        FBREF hides some tables in HTML comments for lazy loading.
        This extracts and uncomments them.
        """
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        
        for comment in comments:
            if '<table' in str(comment) and 'id=' in str(comment):
                try:
                    comment_soup = BeautifulSoup(str(comment), 'lxml')
                    tables = comment_soup.find_all('table')
                    for table in tables:
                        if comment.parent:
                            comment.parent.append(table)
                except Exception as e:
                    logger.debug(f"Error uncommenting table: {e}")
                    
        return soup
    
    def _parse_table_with_data_stat(self, table) -> pd.DataFrame:
        """
        Parse FBREF table using data-stat attributes for accurate column mapping.
        
        This is the KEY to accurate data extraction.
        """
        if table is None:
            return pd.DataFrame()
            
        thead = table.find('thead')
        if not thead:
            return pd.DataFrame()
            
        header_rows = thead.find_all('tr')
        header_row = header_rows[-1] if header_rows else None
        
        if not header_row:
            return pd.DataFrame()
            
        # Extract column names from data-stat attributes
        columns = []
        header_cells = header_row.find_all(['th', 'td'])
        
        for cell in header_cells:
            data_stat = cell.get('data-stat', '')
            if data_stat:
                columns.append(data_stat)
            else:
                columns.append(cell.get_text(strip=True))
        
        tbody = table.find('tbody')
        if not tbody:
            return pd.DataFrame()
            
        rows = tbody.find_all('tr')
        data = []
        
        for row in rows:
            row_classes = row.get('class', [])
            if any(c in row_classes for c in ['spacer', 'thead', 'partial_table']):
                continue
                
            cells = row.find_all(['td', 'th'])
            row_data = {}
            
            for cell in cells:
                data_stat = cell.get('data-stat', '')
                if not data_stat:
                    continue
                    
                link = cell.find('a')
                if link:
                    value = link.get_text(strip=True)
                else:
                    value = cell.get_text(strip=True)
                    
                row_data[data_stat] = value
                
            if row_data:
                data.append(row_data)
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        ordered_cols = [c for c in columns if c in df.columns]
        remaining_cols = [c for c in df.columns if c not in ordered_cols]
        df = df[ordered_cols + remaining_cols]
        
        df = self._convert_numeric_columns(df)
        
        return df
    
    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert string columns to numeric where appropriate."""
        string_cols = {'team', 'squad', 'player', 'nation', 'pos', 'comp', 
                       'nationality', 'birth_year', 'matches',
                       'home_team', 'away_team', 'venue', 'referee', 
                       'match_report', 'notes', 'date', 'dayofweek',
                       'start_time', 'score'}
        
        for col in df.columns:
            if col.lower() in string_cols or any(s in col.lower() for s in ['team', 'venue', 'ref', 'date', 'day', 'time', 'score', 'report', 'note']):
                continue
                
            try:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
                
        return df
    
    def _find_table_by_id(self, soup: BeautifulSoup, table_id: str):
        """Find a table by ID, checking both regular and commented tables."""
        table = soup.find('table', {'id': table_id})
        
        if table:
            return table
            
        div = soup.find('div', {'id': table_id})
        if div:
            table = div.find('table')
            if table:
                return table
                
        variations = [
            table_id,
            f"stats_{table_id.replace('stats_', '')}",
            table_id.replace('stats_squads_', 'stats_'),
        ]
        
        for var_id in variations:
            table = soup.find('table', {'id': var_id})
            if table:
                return table
                
        return None
    
    def _build_table_url(self, season: str, table_config: TableConfig) -> str:
        """Build URL for a specific stats page."""
        if season == CURRENT_SEASON:
            return f"{FBREF_BASE_URL}/en/comps/{PREMIER_LEAGUE_COMP_ID}/{table_config.url_suffix}/Premier-League-Stats"
        else:
            return f"{FBREF_BASE_URL}/en/comps/{PREMIER_LEAGUE_COMP_ID}/{season}/{table_config.url_suffix}/{season}-Premier-League-Stats"
    
    def _build_fixtures_url(self, season: str) -> str:
        """Build URL for fixtures/schedule page."""
        if season == CURRENT_SEASON:
            return f"{FBREF_BASE_URL}/en/comps/{PREMIER_LEAGUE_COMP_ID}/schedule/Premier-League-Scores-and-Fixtures"
        else:
            return f"{FBREF_BASE_URL}/en/comps/{PREMIER_LEAGUE_COMP_ID}/{season}/schedule/{season}-Premier-League-Scores-and-Fixtures"
    
    def scrape_table_from_page(self, soup: BeautifulSoup, table_id: str, 
                                stat_type: str = 'for') -> Optional[pd.DataFrame]:
        """Extract a single table from page soup."""
        table = self._find_table_by_id(soup, table_id)
        
        if table is None:
            logger.warning(f"Table {table_id} not found")
            return None
            
        df = self._parse_table_with_data_stat(table)
        
        if df is not None and not df.empty:
            df['stat_type'] = stat_type
            logger.info(f"Extracted {table_id}: {len(df)} rows, {len(df.columns)} columns")
            
        return df
    
    def scrape_all_tables_from_url(self, url: str, table_configs: List[tuple], 
                                    season: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Scrape multiple tables from a single URL.
        """
        results = {}
        
        if not self._navigate_with_retry(url):
            return results
            
        self._smart_delay()
        
        soup = self._get_page_soup()
        soup = self._uncomment_tables(soup)
        
        for table_key, table_config in table_configs:
            table_data = {}
            
            for_table_id = table_config.table_id
            df_for = self.scrape_table_from_page(soup, for_table_id, 'for')
            
            if df_for is not None and not df_for.empty:
                df_for['season'] = season
                table_data['for'] = df_for
                
            if table_key in TABLES_WITH_AGAINST_STATS:
                against_table_id = table_config.table_id.replace('_for', '_against')
                df_against = self.scrape_table_from_page(soup, against_table_id, 'against')
                
                if df_against is not None and not df_against.empty:
                    df_against['season'] = season
                    table_data['against'] = df_against
                    
            if table_data:
                results[table_key] = table_data
                
        return results
    
    def scrape_fixtures(self, season: str) -> Optional[pd.DataFrame]:
        """Scrape match fixtures/results for a season."""
        url = self._build_fixtures_url(season)
        
        if not self._navigate_with_retry(url):
            return None
            
        self._smart_delay()
        
        soup = self._get_page_soup()
        table = self._find_table_by_id(soup, FIXTURES_CONFIG['table_id'])
        
        if table is None:
            table = soup.find('table', {'id': 'sched_all'})
            
        if table is None:
            # Try to find any schedule table with the season
            for t in soup.find_all('table'):
                tid = t.get('id', '')
                if 'sched' in tid and season.replace('-', '_') in tid.replace('-', '_'):
                    table = t
                    break
        
        if table is None:
            # Try any schedule table
            for t in soup.find_all('table'):
                if 'sched' in t.get('id', ''):
                    table = t
                    break
            
        if table is None:
            logger.warning(f"Fixtures table not found for {season}")
            return None
        
        logger.debug(f"Found fixtures table with ID: {table.get('id')}")
            
        df = self._parse_fixtures_table(table)
        
        if df is not None and not df.empty:
            df['season'] = season
            logger.info(f"Scraped fixtures for {season}: {len(df)} matches")
            
        return df
    
    def _parse_fixtures_table(self, table) -> pd.DataFrame:
        """
        Parse FBREF fixtures table with special handling for team columns.
        
        FBREF fixtures have specific data-stat values that may differ from regular tables.
        """
        if table is None:
            return pd.DataFrame()
        
        tbody = table.find('tbody')
        if not tbody:
            return pd.DataFrame()
        
        rows = tbody.find_all('tr')
        data = []
        
        for row in rows:
            row_classes = row.get('class', [])
            if any(c in row_classes for c in ['spacer', 'thead', 'partial_table']):
                continue
            
            cells = row.find_all(['td', 'th'])
            row_data = {}
            
            for cell in cells:
                data_stat = cell.get('data-stat', '')
                if not data_stat:
                    continue
                
                # Get text value
                link = cell.find('a')
                if link:
                    value = link.get_text(strip=True)
                else:
                    value = cell.get_text(strip=True)
                
                # Handle score specially (extract from text)
                if data_stat == 'score' and '–' in value:
                    row_data['score'] = value
                else:
                    row_data[data_stat] = value
            
            if row_data:
                data.append(row_data)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Standardize column names for fixtures
        column_mappings = {
            'gameweek': 'gameweek',
            'wk': 'gameweek',
            'day': 'dayofweek',
            'dayofweek': 'dayofweek',
            'date': 'date',
            'time': 'start_time',
            'start_time': 'start_time',
            'home_team': 'home_team',
            'squad_a': 'home_team',
            'team_a': 'home_team',
            'home': 'home_team',
            'home_xg': 'home_xg',
            'xg_a': 'home_xg',
            'xg': 'home_xg',
            'score': 'score',
            'away_xg': 'away_xg',
            'xg_b': 'away_xg',
            'xg_1': 'away_xg',
            'away_team': 'away_team',
            'squad_b': 'away_team',
            'team_b': 'away_team',
            'away': 'away_team',
            'attendance': 'attendance',
            'venue': 'venue',
            'referee': 'referee',
            'match_report': 'match_report',
            'notes': 'notes',
        }
        
        # Apply mappings
        df = df.rename(columns={k: v for k, v in column_mappings.items() if k in df.columns})
        
        # Log columns found for debugging
        logger.debug(f"Fixtures columns after parsing: {list(df.columns)}")
        
        # Check if we got team columns
        if 'home_team' not in df.columns or 'away_team' not in df.columns:
            logger.warning(f"Missing team columns! Found: {list(df.columns)}")
        
        # Convert numeric columns
        df = self._convert_numeric_columns(df)
        
        return df
    
    def scrape_season(self, season: str, 
                      tables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Scrape all tables for a single season efficiently.
        """
        tables_to_scrape = tables or list(TABLES_CONFIG.keys())
        
        season_data = {
            'season': season,
            'scraped_at': datetime.now().isoformat(),
            'tables': {},
            'fixtures': None,
        }
        
        logger.info(f"{'=' * 50}")
        logger.info(f"Starting scrape for season {season}")
        logger.info(f"{'=' * 50}")
        
        # Group tables by URL suffix for efficient scraping
        url_groups = {}
        for table_key in tables_to_scrape:
            if table_key not in TABLES_CONFIG:
                continue
            config = TABLES_CONFIG[table_key]
            suffix = config.url_suffix
            if suffix not in url_groups:
                url_groups[suffix] = []
            url_groups[suffix].append((table_key, config))
        
        # Scrape each URL group
        for suffix, table_list in url_groups.items():
            try:
                url = self._build_table_url(season, table_list[0][1])
                logger.info(f"Scraping {len(table_list)} table(s) from {suffix} page...")
                
                tables_data = self.scrape_all_tables_from_url(url, table_list, season)
                season_data['tables'].update(tables_data)
                
            except Exception as e:
                logger.error(f"Error scraping {suffix} for {season}: {e}")
                continue
        
        # Scrape fixtures
        try:
            logger.info("Scraping fixtures...")
            fixtures_df = self.scrape_fixtures(season)
            if fixtures_df is not None:
                season_data['fixtures'] = fixtures_df
        except Exception as e:
            logger.error(f"Error scraping fixtures for {season}: {e}")
        
        return season_data
    
    def scrape_all_seasons(self, seasons: Optional[List[str]] = None,
                           tables: Optional[List[str]] = None) -> Dict[str, Any]:
        """Scrape data for multiple seasons."""
        seasons_to_scrape = seasons or SEASONS
        all_data = {
            'scraped_at': datetime.now().isoformat(),
            'seasons': {}
        }
        
        logger.info(f"Starting scrape for {len(seasons_to_scrape)} seasons")
        
        for season in seasons_to_scrape:
            try:
                season_data = self.scrape_season(season, tables)
                all_data['seasons'][season] = season_data
                
                self._save_season_data(season, season_data)
                
                delay = random.uniform(8, 15)
                logger.info(f"Waiting {delay:.1f}s before next season...")
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error scraping season {season}: {e}")
                time.sleep(random.uniform(15, 30))
                continue
                
        return all_data
    
    def _save_season_data(self, season: str, data: Dict[str, Any]) -> None:
        """Save season data to CSV files."""
        season_dir = Path(self.settings.raw_data_dir) / season
        season_dir.mkdir(parents=True, exist_ok=True)
        
        if data.get('fixtures') is not None:
            fixtures_path = season_dir / 'fixtures.csv'
            data['fixtures'].to_csv(fixtures_path, index=False)
            logger.info(f"Saved fixtures to {fixtures_path}")
        
        for table_key, table_data in data.get('tables', {}).items():
            for stat_type, df in table_data.items():
                filename = f"{table_key}_{stat_type}.csv"
                filepath = season_dir / filename
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {filename} ({len(df)} rows)")


class FBREFDataExtractor:
    """
    Alternative extraction using cloudscraper for direct HTTP requests.
    Faster than Selenium but may be blocked more easily.
    """
    
    def __init__(self, settings: Optional[ScraperSettings] = None):
        self.settings = settings or ScraperSettings()
        self.backoff = ExponentialBackoff(
            base_delay=2.0,
            max_delay=120.0,
            max_retries=5,
            jitter=0.5
        )
        self._setup_directories()
        self._init_scraper()
        
    def _setup_directories(self) -> None:
        for dir_path in [self.settings.output_dir, 
                         self.settings.raw_data_dir, 
                         self.settings.processed_data_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    def _init_scraper(self) -> None:
        try:
            import cloudscraper
            self.session = cloudscraper.create_scraper(
                browser={
                    'browser': 'chrome',
                    'platform': 'darwin',
                    'desktop': True,
                },
                delay=10,
            )
        except ImportError:
            import requests
            self.session = requests.Session()
            
        self.session.headers.update({
            'User-Agent': self.settings.user_agents[0],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        })
            
    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch page with exponential backoff."""
        for attempt in range(self.backoff.max_retries):
            try:
                self.session.headers['User-Agent'] = random.choice(self.settings.user_agents)
                
                delay = random.uniform(self.settings.min_delay, self.settings.max_delay)
                if attempt > 0:
                    delay += self.backoff.get_delay(attempt)
                time.sleep(delay)
                
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 429:
                    logger.warning("Rate limited (429)")
                    self.backoff.wait(attempt)
                elif response.status_code == 403:
                    logger.warning("Forbidden (403)")
                    self.backoff.wait(attempt)
                    self._init_scraper()
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                self.backoff.wait(attempt)
                
        return None


def main():
    """Main entry point for the scraper."""
    settings = ScraperSettings(
        headless=True,
        min_delay=3.0,
        max_delay=6.0,
    )
    
    priority_tables = [
        'standard_stats',
        'shooting',
        'goal_shot_creation',
        'defensive_actions',
        'passing',
        'possession',
        'goalkeeping',
        'misc',
    ]
    
    with FBREFScraper(settings) as scraper:
        all_data = scraper.scrape_all_seasons(
            seasons=SEASONS,
            tables=priority_tables
        )
        
        logger.info("Scraping complete!")
        return all_data


if __name__ == "__main__":
    main()
