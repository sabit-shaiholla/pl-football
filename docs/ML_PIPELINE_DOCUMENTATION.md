# Premier League Match Prediction - ML Pipeline Documentation

> **Last Updated:** January 10, 2026  
> **Model Version:** 1.0  
> **Best Test Accuracy:** 61.2% (Ensemble)

## Table of Contents

1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [Feature Engineering](#feature-engineering)
4. [Machine Learning Models](#machine-learning-models)
5. [Training Results](#training-results)
6. [Usage Commands](#usage-commands)
7. [Project Structure](#project-structure)
8. [Future Improvements](#future-improvements)

---

## Overview

This ML pipeline predicts Premier League match outcomes (Home Win, Draw, Away Win) using historical team statistics from FBREF. The system uses an ensemble of gradient boosting models with 584 engineered features.

### Key Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | 61.2% |
| Baseline (Random) | 33.3% |
| Home Win Rate (Historical) | 42.2% |
| Number of Features | 584 |
| Training Matches | 1,064 |
| Test Matches | 304 |

---

## Data Sources

### Raw Data Structure

Data is scraped from FBREF for each Premier League season:

```
data/raw/
├── 2020-2021/
├── 2021-2022/
├── 2022-2023/
├── 2023-2024/
├── 2024-2025/
└── 2025-2026/
    ├── fixtures.csv                    # Match results and xG
    ├── standard_stats_for.csv          # Goals, assists, xG
    ├── standard_stats_against.csv      # Defensive stats
    ├── shooting_for.csv                # Shots, shot quality
    ├── shooting_against.csv
    ├── passing_for.csv                 # Pass completion, progressive passes
    ├── passing_against.csv
    ├── possession_for.csv              # Possession %, touches by zone
    ├── possession_against.csv
    ├── defensive_actions_for.csv       # Tackles, interceptions, blocks
    ├── defensive_actions_against.csv
    ├── goal_shot_creation_for.csv      # SCA, GCA metrics
    ├── goal_shot_creation_against.csv
    ├── goalkeeping_for.csv             # Save %, clean sheets
    ├── goalkeeping_against.csv
    ├── misc_for.csv                    # Cards, fouls, aerials
    └── misc_against.csv
```

### Fixtures Data Fields

| Field | Description |
|-------|-------------|
| `home_team` | Home team name |
| `away_team` | Away team name |
| `score` | Final score (e.g., "4–2") |
| `home_xg` | Home team expected goals |
| `away_xg` | Away team expected goals |
| `date` | Match date |
| `gameweek` | Premier League gameweek |

---

## Feature Engineering

### Feature Categories (1,128 raw → 584 after selection)

#### 1. Attacking Features
Features capturing a team's ability to create and score goals.

| Feature | Description | Football Analytics Insight |
|---------|-------------|---------------------------|
| `goals_per_90` | Goals scored per 90 minutes | Core attacking output |
| `xg_per_90` | Expected goals per 90 | Quality of chances created |
| `shots_per_90` | Shots per 90 minutes | Volume of attacks |
| `shots_on_target_pct` | Shot accuracy | Finishing quality |
| `npxg_per_90` | Non-penalty xG | True open-play threat |
| `gca_per_90` | Goal-creating actions | Final third creativity |
| `sca_per_90` | Shot-creating actions | Chance creation volume |
| `progressive_passes` | Passes moving ball forward | Build-up quality |
| `progressive_carries` | Carries moving ball forward | Dribbling threat |

#### 2. Defensive Features
Features capturing a team's ability to prevent goals.

| Feature | Description | Football Analytics Insight |
|---------|-------------|---------------------------|
| `goals_against_per_90` | Goals conceded per 90 | Defensive output |
| `xga_per_90` | Expected goals against | Chance quality allowed |
| `tackles_won_pct` | Tackle success rate | Defensive duels |
| `interceptions_per_90` | Interceptions per 90 | Reading the game |
| `blocks_per_90` | Blocks per 90 | Last-ditch defending |
| `clearances_per_90` | Clearances per 90 | Aerial/zone defense |
| `pressures_per_90` | Pressing intensity | High press effectiveness |
| `clean_sheet_pct` | Clean sheet percentage | Shutout ability |

#### 3. Possession & Style Features
Features capturing playing style and territory control.

| Feature | Description | Football Analytics Insight |
|---------|-------------|---------------------------|
| `possession_pct` | Average possession | Game control |
| `pass_completion_pct` | Pass accuracy | Technical quality |
| `touches_att_3rd` | Touches in attacking third | Territory dominance |
| `touches_def_3rd` | Touches in defensive third | Defensive territory |
| `progressive_pass_distance` | Total progressive distance | Vertical passing |
| `passes_into_final_third` | Entries to final third | Attack frequency |
| `crosses_per_90` | Crosses attempted | Wide play emphasis |

#### 4. Set Pieces & Discipline
Features capturing dead-ball situations and fair play.

| Feature | Description | Football Analytics Insight |
|---------|-------------|---------------------------|
| `corners_per_90` | Corners won | Set piece opportunities |
| `fouls_committed` | Fouls per game | Discipline issues |
| `fouls_drawn` | Fouls won | Drawing free kicks |
| `yellow_cards_per_90` | Yellows per game | Aggression/discipline |
| `red_cards_per_90` | Reds per game | Sending off risk |
| `penalties_won` | Penalties earned | Box threat |
| `penalties_conceded` | Penalties given away | Defensive risk |

#### 5. Form & Momentum Features
Rolling averages capturing recent performance trends.

| Feature | Description | Window |
|---------|-------------|--------|
| `form_3_ppg` | Points per game | Last 3 matches |
| `form_5_ppg` | Points per game | Last 5 matches |
| `form_10_ppg` | Points per game | Last 10 matches |
| `form_3_xg` | Average xG | Last 3 matches |
| `form_5_xg` | Average xG | Last 5 matches |
| `form_3_goals` | Goals scored | Last 3 matches |
| `form_3_goals_against` | Goals conceded | Last 3 matches |
| `win_streak` | Consecutive wins | Current |
| `unbeaten_streak` | Matches without loss | Current |

#### 6. ELO Ratings
Dynamic team strength rating system.

| Feature | Description | Parameters |
|---------|-------------|------------|
| `home_elo` | Home team ELO rating | Initial: 1500 |
| `away_elo` | Away team ELO rating | K-factor: 32 |
| `elo_diff` | Rating difference | Home advantage: 100 |
| `elo_home_win_prob` | ELO-based win probability | - |

**ELO Update Formula:**
```
New_Rating = Old_Rating + K × (Actual - Expected)
Expected = 1 / (1 + 10^((Opponent_Rating - Rating) / 400))
```

#### 7. Differential Features
Home vs Away team comparisons (key predictors).

| Feature Pattern | Description |
|-----------------|-------------|
| `diff_xg_per_90` | Home xG - Away xG |
| `diff_goals_against_per_90` | Defensive comparison |
| `diff_possession_pct` | Possession difference |
| `diff_pass_completion_pct` | Technical quality gap |
| `diff_tackles_won_pct` | Defensive duel comparison |
| `ratio_xg_per_90` | Home/Away xG ratio |

### Feature Selection Process

1. **Initial Features:** 1,128 engineered features
2. **Correlation Filtering:** Remove features with correlation > 0.95 → 619 features
3. **Final Selection:** 584 features after pipeline processing

### Data Leakage Prevention

The following features are explicitly excluded to prevent data leakage:

```python
EXCLUDED_FEATURES = [
    'home_goals', 'away_goals',      # Actual match outcome
    'home_xg', 'away_xg',            # Match-level xG (post-match)
    'total_goals', 'goal_difference', # Derived from outcome
    'match_result', 'home_win',       # Target variables
    'diff_goals',                     # Leaks goal information
]
```

---

## Machine Learning Models

### Model Architecture

#### 1. Baseline Model
Simple ELO-based prediction using expected win probabilities.

```python
# Prediction based on ELO ratings only
if elo_home_win_prob > 0.45:
    predict = "Home Win"
elif elo_home_win_prob < 0.35:
    predict = "Away Win"
else:
    predict = "Draw"
```

#### 2. Logistic Regression
Linear baseline with L2 regularization.

```python
LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
```

#### 3. Random Forest (Regularized)
Ensemble of decision trees with overfitting prevention.

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=7,              # Regularization
    min_samples_split=20,     # Regularization
    min_samples_leaf=10,      # Regularization
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)
```

#### 4. XGBoost (Regularized)
Gradient boosting with strong regularization.

```python
XGBClassifier(
    n_estimators=500,
    max_depth=4,              # Shallow trees
    learning_rate=0.05,       # Slow learning
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,            # L1 regularization
    reg_lambda=1.0,           # L2 regularization
    gamma=0.1,                # Min split loss
    min_child_weight=5,
    early_stopping_rounds=50,
    random_state=42
)
```

#### 5. LightGBM (Regularized)
Fast gradient boosting with leaf-wise growth.

```python
LGBMClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    num_leaves=15,            # Regularization
    min_child_samples=20,     # Regularization
    min_split_gain=0.01,      # Regularization
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)
```

#### 6. Ensemble (Weighted Voting)
Combines multiple models with validation-based weighting.

```python
EnsemblePredictor(
    models=[
        LogisticRegressionModel(),
        RandomForestModel(),
        XGBoostModel(),
        LightGBMModel()
    ],
    method='weighted'  # Weights based on validation accuracy
)
```

#### 7. Score Predictor (Poisson Distribution)
Predicts exact scores using Poisson regression for expected goals.

```python
ScorePredictor(
    home_model=GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=10,
        subsample=0.8
    ),
    away_model=GradientBoostingRegressor(...)
)
```

**Poisson Model Theory:**

Goals in football follow a Poisson distribution where:
- $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$
- $\lambda$ = expected goals (xG)

For a match score prediction:
- $P(\text{home}=h, \text{away}=a) = P_{\text{home}}(h) \times P_{\text{away}}(a)$

**Score Predictor Outputs:**
- Expected home/away goals
- Most likely score with probability
- Top N most likely scores
- Over/Under total goals probabilities
- Both Teams To Score (BTTS) probabilities

---

## Training Results

### Model Comparison (January 10, 2026)

| Model | Train Acc | Val Acc | Test Acc | Log Loss | Overfit Gap |
|-------|-----------|---------|----------|----------|-------------|
| Baseline | 42.2% | 48.7% | 47.0% | 1.058 | - |
| Logistic | 66.4% | 57.9% | 53.6% | 1.185 | 12.8% |
| Random Forest | 81.0% | 65.1% | 60.5% | 0.909 | 20.5% |
| XGBoost | 100.0% | 63.2% | 60.5% | 0.860 | 39.5% |
| LightGBM | 100.0% | 63.2% | 59.2% | 0.914 | 40.8% |
| **Ensemble** | 99.6% | **65.8%** | **61.2%** | 0.861 | 38.4% |

### Classification Report (Ensemble on Test Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Away Win (0) | 0.59 | 0.67 | 0.63 | 105 |
| Draw (1) | 0.38 | 0.07 | 0.12 | 68 |
| Home Win (2) | 0.64 | 0.83 | 0.72 | 131 |
| **Weighted Avg** | 0.57 | 0.61 | 0.56 | 304 |

### Score Predictor Results (Poisson Model)

| Metric | Value |
|--------|-------|
| Home Goals MAE | 0.868 |
| Away Goals MAE | 0.786 |
| Home Goals RMSE | 1.098 |
| Away Goals RMSE | 0.998 |
| Exact Score Accuracy | 12.5% |
| Avg Home xG | 1.48 |
| Avg Away xG | 1.28 |

**Note:** Exact score prediction is inherently difficult (~12.5% accuracy is reasonable given the combinatorial nature of possible scores). The model's value lies in:
1. Probabilistic score distributions
2. Betting market predictions (Over/Under, BTTS)
3. Expected goals estimations

### Key Observations

1. **Draws are hardest to predict** - Only 7% recall, 38% precision
2. **Home wins easiest to predict** - 83% recall, models favor home advantage
3. **Ensemble provides best generalization** - Highest validation and test accuracy
4. **Regularization helps** - Reduced overfitting from 50%+ to ~20% for RF
5. **61% accuracy beats random (33%)** by 28 percentage points

### Target Distribution

| Outcome | Train % | Test % |
|---------|---------|--------|
| Home Win | 42.2% | 43.1% |
| Away Win | 35.1% | 34.5% |
| Draw | 22.7% | 22.4% |

---

## Usage Commands

### Environment Setup

```bash
# Install dependencies
uv sync

# Or with pip
pip install -r requirements.txt
```

### Data Scraping

```bash
# Scrape current season data
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

# Train with custom seasons
uv run python -m src.ml.train --seasons 2022-2023 2023-2024 2024-2025
```

### Hyperparameter Tuning

```bash
# Tune XGBoost
uv run python -c "
from src.ml.tuning import tune_xgboost
best_params, cv_score = tune_xgboost(X_train, y_train)
print(f'Best CV Score: {cv_score:.4f}')
"
```

### Making Predictions

```bash
# Predict upcoming match
uv run python -m src.ml.predict \
    --home "Arsenal" \
    --away "Manchester City" \
    --date "2026-01-17" \
    --season "2023-2024"

# Example output:
# ============================================================
# MATCH PREDICTION: Arsenal vs Manchester City
# Date: 2026-01-17
# ============================================================
# 
# PREDICTED OUTCOME:
#   >>> Draw (Confidence: 39.5%) <<<
# 
# WIN PROBABILITIES:
#   Home Win (Arsenal): 21.4%
#   Draw:                        39.5%
#   Away Win (Manchester City): 39.1%
# 
# SCORE PREDICTION (Poisson Model):
#   Expected Score: Arsenal 1 - 2 Manchester City
#   Score Probability: 8.8%
#   Expected Goals: 1.58 - 2.01
# 
# TOP 5 MOST LIKELY SCORES:
#   1. 1-2: 8.8%
#   2. 1-1: 8.8%
#   3. 2-2: 7.0%
#   4. 2-1: 6.9%
#   5. 1-3: 5.9%
# 
# BETTING MARKETS:
#   Over 2.5 Goals: 69.4%
#   Under 2.5 Goals: 30.6%
#   Both Teams To Score: 68.7%
# 
# ELO RATINGS:
#   Arsenal: 1792
#   Manchester City: 1743
#   ELO Expected Home Win: 57.0%
# ============================================================
```

### Processing Raw Data

```bash
# Process all raw data into ML features
uv run python -m src.ml.process_data

# Output: data/processed/ml_features.csv
```

---

## Project Structure

```
pl-football/
├── data/
│   ├── raw/                    # Scraped FBREF data by season
│   │   └── {season}/           # 17 CSV files per season
│   └── processed/              # ML-ready data
│       ├── ml_features.csv     # All engineered features
│       ├── train.csv           # Training set
│       ├── val.csv             # Validation set
│       ├── test.csv            # Test set
│       └── feature_info.json   # Feature metadata
├── models/
│   ├── ensemble_match_result.pkl  # Trained ensemble model
│   └── score_predictor.pkl        # Poisson score predictor
├── src/
│   ├── scraper/                # FBREF data scraping
│   │   ├── fbref_scraper.py
│   │   ├── data_processor.py
│   │   └── run_scraper.py
│   └── ml/                     # Machine learning pipeline
│       ├── feature_engineering.py  # Feature creation
│       ├── data_pipeline.py        # Data processing
│       ├── models.py               # Model definitions
│       ├── train.py                # Training script
│       ├── predict.py              # Prediction interface
│       ├── tuning.py               # Hyperparameter tuning
│       └── process_data.py         # Data combination
├── docs/
│   └── ML_PIPELINE_DOCUMENTATION.md  # This file
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Future Improvements

### Short-term

1. **Player-level features** - Incorporate individual player stats and injuries
2. **Betting odds integration** - Use market odds as features
3. ~~**Score prediction** - Poisson regression for exact scores~~ ✅ Implemented
4. **Probability calibration** - Improve probability estimates

### Medium-term

1. **Neural network models** - Deep learning for complex patterns
2. **Time series features** - LSTM for sequential match dependencies
3. **Head-to-head features** - Historical matchup analysis
4. **Manager effects** - Coaching tenure and style features

### Long-term

1. **Real-time predictions** - Live match outcome updates
2. **Multi-league support** - Extend to other European leagues
3. **Transfer market impact** - Factor in player transfers
4. **Automated retraining** - Weekly model updates

---

## References

- [FBREF](https://fbref.com/) - Data source
- [Expected Goals (xG) Explained](https://fbref.com/en/expected-goals-model-explained/) - xG methodology
- [ELO Rating System](https://en.wikipedia.org/wiki/Elo_rating_system) - Rating calculations
- [XGBoost Documentation](https://xgboost.readthedocs.io/) - Model reference
- [LightGBM Documentation](https://lightgbm.readthedocs.io/) - Model reference

---

## Appendix: Current ELO Rankings (Jan 2026)

| Rank | Team | ELO Rating |
|------|------|------------|
| 1 | Arsenal | 1792 |
| 2 | Manchester City | 1743 |
| 3 | Aston Villa | 1730 |
| 4 | Liverpool | 1689 |
| 5 | Chelsea | 1624 |
| 6 | Newcastle Utd | 1611 |
| 7 | Brighton | 1604 |
| 8 | Brentford | 1596 |
| 9 | Fulham | 1580 |
| 10 | Sunderland | 1567 |
| 11 | Crystal Palace | 1566 |
| 12 | Everton | 1556 |
| 13 | Manchester Utd | 1556 |
| 14 | Bournemouth | 1540 |
| 15 | Nott'ham Forest | 1508 |
| 16 | Tottenham | 1496 |
| 17 | Leeds United | 1459 |
| 18 | West Ham | 1413 |
| 19 | Wolves | 1381 |
| 20 | Southampton | 1261 |

*ELO calculated from 2,280 matches (2020-2026)*
