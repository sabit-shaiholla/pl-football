"""
Configuration settings for FBREF scraper.
Contains all constants, URLs, and feature mappings for data collection.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# Base URLs
FBREF_BASE_URL = "https://fbref.com"
PREMIER_LEAGUE_COMP_ID = "9"

# Current season (update this each season)
CURRENT_SEASON = "2025-2026"

# Season range for scraping (current to 2018-2019)
SEASONS = [
    "2025-2026",
    "2024-2025",
    "2023-2024",
    "2022-2023",
    "2021-2022",
    "2020-2021",
    "2019-2020",
    "2018-2019",
]

# Current season URL format differs from historical
CURRENT_SEASON_URL = f"{FBREF_BASE_URL}/en/comps/{PREMIER_LEAGUE_COMP_ID}/Premier-League-Stats"
HISTORICAL_SEASON_URL_TEMPLATE = f"{FBREF_BASE_URL}/en/comps/{PREMIER_LEAGUE_COMP_ID}/{{season}}/{{season}}-Premier-League-Stats"


@dataclass
class TableConfig:
    """Configuration for each FBREF table."""
    name: str
    table_id: str
    url_suffix: str
    description: str
    importance_for_prediction: str  # HIGH, MEDIUM, LOW
    key_features: List[str] = field(default_factory=list)


# All available tables on FBREF with their IDs and importance for match prediction
# As an experienced DS/MLE, I've ranked these by predictive value
TABLES_CONFIG: Dict[str, TableConfig] = {
    # HIGH IMPORTANCE - Core metrics directly tied to match outcomes
    "standard_stats": TableConfig(
        name="Squad Standard Stats",
        table_id="stats_squads_standard_for",
        url_suffix="stats",
        description="Core performance metrics including goals, assists, xG, xAG",
        importance_for_prediction="HIGH",
        key_features=["Gls", "Ast", "G+A", "G-PK", "xG", "xAG", "npxG", "PrgC", "PrgP", "PrgR"]
    ),
    "shooting": TableConfig(
        name="Squad Shooting",
        table_id="stats_squads_shooting_for",
        url_suffix="shooting",
        description="Shot quality and conversion metrics",
        importance_for_prediction="HIGH",
        key_features=["Sh", "SoT", "SoT%", "Sh/90", "SoT/90", "G/Sh", "G/SoT", "Dist", "xG", "npxG", "npxG/Sh"]
    ),
    "goal_shot_creation": TableConfig(
        name="Squad Goal and Shot Creation",
        table_id="stats_squads_gca_for",
        url_suffix="gca",
        description="Actions leading to shots and goals - key attacking threat indicator",
        importance_for_prediction="HIGH",
        key_features=["SCA", "SCA90", "GCA", "GCA90", "PassLive", "PassDead", "TO", "Sh", "Fld", "Def"]
    ),
    "defensive_actions": TableConfig(
        name="Squad Defensive Actions",
        table_id="stats_squads_defense_for",
        url_suffix="defense",
        description="Tackles, interceptions, blocks - defensive solidity metrics",
        importance_for_prediction="HIGH",
        key_features=["Tkl", "TklW", "Def 3rd", "Mid 3rd", "Att 3rd", "Int", "Tkl+Int", "Clr", "Err", "Blocks"]
    ),
    
    # MEDIUM-HIGH IMPORTANCE - Strongly predictive secondary metrics
    "passing": TableConfig(
        name="Squad Passing",
        table_id="stats_squads_passing_for",
        url_suffix="passing",
        description="Pass completion and progressive passing metrics",
        importance_for_prediction="MEDIUM-HIGH",
        key_features=["Cmp", "Att", "Cmp%", "TotDist", "PrgDist", "Cmp%.1", "Cmp%.2", "Cmp%.3", "KP", "1/3", "PPA", "CrsPA", "PrgP"]
    ),
    "possession": TableConfig(
        name="Squad Possession",
        table_id="stats_squads_possession_for",
        url_suffix="possession",
        description="Ball control, carries, and progressive ball movement",
        importance_for_prediction="MEDIUM-HIGH",
        key_features=["Poss", "Touches", "Def Pen", "Def 3rd", "Mid 3rd", "Att 3rd", "Att Pen", "Carries", "PrgC", "PrgDist", "Succ", "Att"]
    ),
    "goalkeeping": TableConfig(
        name="Squad Goalkeeping",
        table_id="stats_squads_keeper_for",
        url_suffix="keepers",
        description="Basic GK stats - saves, clean sheets, goals against",
        importance_for_prediction="MEDIUM-HIGH",
        key_features=["GA", "GA90", "SoTA", "Saves", "Save%", "W", "D", "L", "CS", "CS%", "PKatt", "PKA", "PKsv", "PKm"]
    ),
    
    # MEDIUM IMPORTANCE - Useful context and style indicators
    "advanced_goalkeeping": TableConfig(
        name="Squad Advanced Goalkeeping",
        table_id="stats_squads_keeper_adv_for",
        url_suffix="keepersadv",
        description="Advanced GK metrics - PSxG, crosses, sweeper actions",
        importance_for_prediction="MEDIUM",
        key_features=["PSxG", "PSxG/SoT", "PSxG+/-", "Stp", "Stp%", "#OPA", "AvgDist"]
    ),
    "pass_types": TableConfig(
        name="Squad Pass Types",
        table_id="stats_squads_passing_types_for",
        url_suffix="passing_types",
        description="Detailed pass type breakdown",
        importance_for_prediction="MEDIUM",
        key_features=["Live", "Dead", "FK", "TB", "Sw", "Crs", "TI", "CK", "In", "Out", "Str", "Cmp", "Off", "Blocks"]
    ),
    "misc": TableConfig(
        name="Squad Miscellaneous Stats",
        table_id="stats_squads_misc_for",
        url_suffix="misc",
        description="Fouls, cards, aerials - discipline and physical play",
        importance_for_prediction="MEDIUM",
        key_features=["CrdY", "CrdR", "2CrdY", "Fls", "Fld", "Off", "Crs", "Int", "TklW", "PKwon", "PKcon", "OG", "Recov", "Won", "Lost", "Won%"]
    ),
    
    # LOWER IMPORTANCE - Contextual but less predictive
    "playing_time": TableConfig(
        name="Squad Playing Time",
        table_id="stats_squads_playing_time_for",
        url_suffix="playingtime",
        description="Minutes played, starts, substitutions - squad rotation indicators",
        importance_for_prediction="LOW",
        key_features=["MP", "Min", "Mn/MP", "Min%", "90s", "Starts", "Mn/Start", "Compl", "Subs", "Mn/Sub", "unSub", "PPM", "onG", "onGA", "+/-", "onxG", "onxGA"]
    ),
}

# Tables to scrape "Against" stats (opponent metrics) - critical for match prediction
TABLES_WITH_AGAINST_STATS = [
    "standard_stats",
    "shooting", 
    "goalkeeping",
    "advanced_goalkeeping",
    "passing",
    "pass_types",
    "goal_shot_creation",
    "defensive_actions",
    "possession",
    "misc",
]

# Fixture/Match data for historical results
FIXTURES_CONFIG = {
    "table_id": "sched_all",
    "url_suffix": "schedule",
}


@dataclass
class ScraperSettings:
    """Runtime settings for the scraper."""
    # Request delays (in seconds) - IMPORTANT to avoid rate limiting
    min_delay: float = 3.0
    max_delay: float = 7.0
    page_load_timeout: int = 30
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 10.0
    
    # Output settings
    output_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    
    # Browser settings
    headless: bool = True
    window_size: tuple = (1920, 1080)
    
    # User agents rotation for avoiding detection
    user_agents: List[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ])


# Feature groups for ML model input - organized by prediction task relevance
FEATURE_GROUPS = {
    "attacking_strength": [
        "Gls", "xG", "npxG", "Sh", "SoT", "SoT%", "G/Sh", "G/SoT",
        "SCA", "SCA90", "GCA", "GCA90", "PrgC", "PrgP"
    ],
    "defensive_strength": [
        "GA", "GA90", "Save%", "CS", "CS%", "Tkl", "TklW", "Int", 
        "Tkl+Int", "Clr", "Blocks", "Err"
    ],
    "possession_quality": [
        "Poss", "Cmp%", "PrgDist", "Touches", "Carries", "PrgC",
        "Att 3rd", "Att Pen", "KP", "1/3", "PPA"
    ],
    "set_piece_threat": [
        "FK", "CK", "PKwon", "PKatt", "PKsv"
    ],
    "discipline": [
        "CrdY", "CrdR", "Fls", "Fld", "Off"
    ],
    "physical_dominance": [
        "Won", "Lost", "Won%", "Recov"
    ],
}
