# Premier League Match Prediction - FBREF Data Pipeline

This is my end-to-end pipeline for scraping Premier League stats from [FBREF](https://fbref.com/), engineering features, and training models to predict match outcomes. I built it to be practical, readable, and easy to extend as I experiment with new features and models.

## 🎯 Project Overview

The pipeline scrapes, processes, and engineers features from FBREF’s Premier League statistics to train ML models that predict match outcomes (Win/Draw/Loss).

### Key Results

| Metric | Value |
|--------|-------|
| **Best Test Accuracy** | 61.2% (Ensemble) |
| Baseline (Random) | 33.3% |
| Features Engineered | 584 |
| Training Matches | 1,064 |
| Seasons Covered | 2020-2026 |

### Key Features

- **Robust Selenium-based scraper** that handles 403s and common anti-scraping blocks
- **Multi-season support** (2020-2021 to 2025-2026)
- **Comprehensive table coverage** — 17 statistical categories per season
- **Advanced feature engineering** including ELO ratings, form metrics, and differential features
- **Ensemble ML models** — Random Forest, XGBoost, LightGBM with weighted voting
- **Match prediction CLI** — Predict upcoming matches with probability estimates
- **Production-ready code** with solid error handling and logging

## 📊 Data Sources (FBREF Tables)

I scrape the following tables, prioritized by predictive value for match outcomes:

### High Importance (Core Predictors)

| Table | Description | Key Features |
|-------|-------------|--------------|
| **Squad Standard Stats** | Core performance metrics | Goals, xG, xAG, Progressive actions |
| **Squad Shooting** | Shot quality and conversion | Shots, SoT%, G/Sh, G/SoT, npxG/Sh |
| **Squad Goal & Shot Creation** | Attacking threat indicators | SCA, SCA90, GCA, GCA90 |
| **Squad Defensive Actions** | Defensive solidity | Tackles, Interceptions, Blocks, Clearances |

### Medium-High Importance

| Table | Description | Key Features |
|-------|-------------|--------------|
| **Squad Passing** | Ball progression quality | Completion%, Progressive Distance, Key Passes |
| **Squad Possession** | Ball control metrics | Possession%, Touches, Carries, Progressive Carries |
| **Squad Goalkeeping** | Goalkeeper performance | Save%, Clean Sheets, GA90 |

### Medium Importance (Style Indicators)

| Table | Description | Key Features |
|-------|-------------|--------------|
| **Squad Advanced Goalkeeping** | Advanced GK metrics | PSxG, Post-Shot xG +/- |
| **Squad Pass Types** | Passing patterns | Through Balls, Switches, Crosses |
| **Squad Miscellaneous** | Discipline & aerials | Yellow/Red Cards, Fouls, Aerial Duels |

### Lower Importance (Context)

| Table | Description | Key Features |
|-------|-------------|--------------|
| **Squad Playing Time** | Rotation patterns | Minutes, Starts, Substitutions |

## 🏗️ Architecture

```
pl-football/
├── src/
│   ├── scraper/
│   │   ├── __init__.py          # Package exports
│   │   ├── config.py            # Configuration and constants
│   │   ├── fbref_scraper.py     # Core Selenium scraper
│   │   ├── data_processor.py    # Data cleaning
│   │   └── run_scraper.py       # Scraper CLI
│   └── ml/
│       ├── __init__.py          # Package exports
│       ├── feature_engineering.py # Feature creation (1128 features)
│       ├── data_pipeline.py     # ML data pipeline
│       ├── models.py            # 6 ML models + Ensemble
│       ├── train.py             # Training CLI
│       ├── predict.py           # Prediction interface
│       ├── tuning.py            # Hyperparameter tuning
│       └── process_data.py      # Data combination
├── data/
│   ├── raw/                     # Scraped FBREF data by season
│   │   ├── 2025-2026/           # 17 CSV files per season
│   │   ├── 2024-2025/
│   │   └── ...
│   └── processed/               # ML-ready datasets
│       ├── ml_features.csv      # Engineered features
│       ├── train.csv            # Training split
│       ├── val.csv              # Validation split
│       └── test.csv             # Test split
├── models/
│   └── ensemble_match_result.pkl  # Trained model
├── docs/
│   ├── ML_PIPELINE_DOCUMENTATION.md
│   └── FEATURE_ENGINEERING_ML_REFERENCE.md
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Chrome browser installed (for Selenium scraper)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Setup with uv (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd pl-football

# Install dependencies with uv
uv sync

# Run commands with uv
uv run python -m src.ml.train --target match_result --model ensemble
```

### Setup with pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### Data Scraping

```bash
# Scrape current season (2025-2026)
uv run python -m src.scraper.run_scraper --season 2025-2026

# Scrape specific season
uv run python -m src.scraper.run_scraper --season 2023-2024
```

### Model Training

```bash
# Train ensemble model (recommended)
uv run python -m src.ml.train --target match_result --model ensemble

# Train specific model
uv run python -m src.ml.train --target match_result --model xgboost

# Evaluate all models
uv run python -m src.ml.train --target match_result --evaluate-all
```

### Making Predictions

```bash
# Predict upcoming match
uv run python -m src.ml.predict \
    --home "Manchester United" \
    --away "Manchester City" \
    --date "2026-01-17" \
    --season "2025-2026"

# Example output:
# ============================================================
# MATCH PREDICTION: Manchester United vs Manchester City
# Date: 2026-01-17
# ============================================================
#
# PREDICTED OUTCOME:
#   >>> Draw (Confidence: 42.9%) <<<
#
# WIN PROBABILITIES:
#   Home Win (Manchester United): 20.5%
#   Draw:                        42.9%
#   Away Win (Manchester City): 36.6%
#
# ELO RATINGS:
#   Manchester United: 1556
#   Manchester City: 1743
#   ELO Expected Home Win: 25.4%
# ============================================================
```

### Data Processing

```bash
# Process scraped data into ML features
uv run python -m src.ml.process_data
```

## 🧠 Feature Engineering

The pipeline creates 1,128 advanced features (584 after correlation filtering):

### Feature Categories

| Category | Examples | Count |
|----------|----------|-------|
| **Attacking** | xG, shots, goal conversion, progressive passes | ~150 |
| **Defensive** | xGA, tackles, interceptions, blocks, clean sheets | ~120 |
| **Possession** | Possession %, touches by zone, progressive carries | ~100 |
| **Form** | Rolling 3/5/10 match averages for all metrics | ~200 |
| **ELO Ratings** | Dynamic team strength, home/away ratings | ~10 |
| **Set Pieces** | Corners, penalties, free kicks | ~50 |
| **Differential** | Home vs Away comparisons for all stats | ~500 |

### ELO Ratings

- Dynamic strength ratings from 2,280 historical matches
- K-factor: 32, Home advantage: +100
- Top teams: Arsenal (1792), Man City (1743), Aston Villa (1730)

### Data Leakage Prevention

Explicitly excluded features:
- `home_goals`, `away_goals` - Actual match outcome
- `home_xg`, `away_xg` - Match-level xG (post-match)
- `diff_goals` - Derived from outcome

## ⚙️ Configuration

Key settings in `src/scraper/config.py`:

```python
# Seasons to scrape
SEASONS = [
    "2025-2026",
    "2024-2025",
    "2023-2024",
    # ... back to 2020-2021
]

# Request delays to avoid rate limiting
ScraperSettings(
    min_delay=3.0,      # Minimum seconds between requests
    max_delay=7.0,      # Maximum seconds between requests
    page_load_timeout=30,
    max_retries=3,
)
```

## 🛡️ Anti-Detection Measures

The scraper implements multiple techniques to avoid blocking:

1. **User Agent Rotation** - Cycles through realistic browser user agents
2. **Random Delays** - Variable timing between requests (3-7 seconds)
3. **Selenium Stealth** - Removes automation detection flags
4. **CDP Commands** - Masks `navigator.webdriver` property
5. **Retry Logic** - Automatic retries with exponential backoff
6. **Session Refresh** - New browser session on repeated failures

## 📈 ML Model Performance

The processed dataset supports multiple prediction tasks:

### Model Comparison (January 2026)

| Model | Train Acc | Val Acc | Test Acc | Log Loss |
|-------|-----------|---------|----------|----------|
| Baseline | 42.2% | 48.7% | 47.0% | 1.058 |
| Logistic | 66.4% | 57.9% | 53.6% | 1.185 |
| Random Forest | 81.0% | 65.1% | 60.5% | 0.909 |
| XGBoost | 100.0% | 63.2% | 60.5% | 0.860 |
| LightGBM | 100.0% | 63.2% | 59.2% | 0.914 |
| **Ensemble** | 99.6% | **65.8%** | **61.2%** | 0.861 |

### Feature Engineering Highlights

- **584 features** after correlation filtering (from 1,128 raw)
- **7 feature categories**: Attacking, Defensive, Possession, Form, ELO, Set Pieces, Differential
- **ELO ratings** calculated from 2,280 historical matches
- **Form features** with 3, 5, and 10 match rolling windows

## 📚 Documentation

- [ML Pipeline Documentation](docs/ML_PIPELINE_DOCUMENTATION.md) - Complete guide to training and predictions
- [Feature Engineering Reference](docs/FEATURE_ENGINEERING_ML_REFERENCE.md) - Detailed feature descriptions

## ⚠️ Important Notes

1. **FBREF data update (Jan 2026)**: OPTA requested FBREF to remove advanced stats, so the scraping solution in this repository no longer works for advanced tables. The raw and processed datasets used in this project are already included in this repository. See the official update: https://www.sports-reference.com/blog/2026/01/fbref-stathead-data-update/

2. **Rate Limiting**: FBREF has rate limits. The scraper uses conservative delays (4-8 seconds) to be respectful.

3. **Data Accuracy**: The scraper extracts data exactly as displayed on FBREF. Multi-level headers are properly parsed.

4. **Legal Considerations**: This tool is for personal/educational use. Respect FBREF's terms of service.

5. **Storage**: Each season's data is ~1-2 MB. Full dataset with all seasons: ~15-20 MB.

## 🐛 Troubleshooting

### 403 Forbidden Errors
- Increase delays in scraper settings
- The scraper auto-retries with new sessions

### Chrome Driver Issues
- The scraper uses `webdriver-manager` for automatic driver management
- Ensure Chrome browser is installed and up to date

### Model Training Issues
- Ensure data is scraped first: `uv run python -m src.scraper.run_scraper --season 2025-2026`
- Check `data/raw/` for CSV files

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! 

---

*Last Updated: February 2026*
