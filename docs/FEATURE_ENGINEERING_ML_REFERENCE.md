# Premier League Match Prediction - Feature Engineering & ML Pipeline Documentation

## Overview

This document provides comprehensive documentation of the feature engineering approach and ML pipeline for predicting Premier League match outcomes. It serves as a reference for AI agents and developers working on this project.

> **Last Updated:** January 10, 2026  

---

## Table of Contents

1. [Data Sources & Structure](#data-sources--structure)
2. [Feature Engineering Strategy](#feature-engineering-strategy)
3. [Feature Categories Deep Dive](#feature-categories-deep-dive)
4. [ML Pipeline Architecture](#ml-pipeline-architecture)
5. [Model Selection & Ensemble](#model-selection--ensemble)
6. [Target Variables](#target-variables)
7. [Feature Importance Analysis](#feature-importance-analysis)
8. [Implementation Reference](#implementation-reference)

---

## Data Sources & Structure

### Raw Data Files

The project uses FBREF statistics scraped for multiple Premier League seasons (2020-2026). Each season contains:

| File Category | Files | Description |
|---------------|-------|-------------|
| **Fixtures** | `fixtures.csv` | Match results, xG, attendance, referees |
| **Standard Stats** | `standard_stats_for.csv`, `standard_stats_against.csv` | Goals, assists, xG, xAG, progressive actions |
| **Shooting** | `shooting_for.csv`, `shooting_against.csv` | Shots, SoT, conversion rates, shot distance |
| **Passing** | `passing_for.csv`, `passing_against.csv` | Pass completion, progressive passes, key passes |
| **Possession** | `possession_for.csv`, `possession_against.csv` | Ball control, carries, touches by zone |
| **Defensive** | `defensive_actions_for.csv`, `defensive_actions_against.csv` | Tackles, interceptions, blocks, clearances |
| **Goal Creation** | `goal_shot_creation_for.csv`, `goal_shot_creation_against.csv` | SCA, GCA, creation methods |
| **Goalkeeping** | `goalkeeping_for.csv`, `goalkeeping_against.csv` | Saves, clean sheets, penalty saves |
| **Miscellaneous** | `misc_for.csv`, `misc_against.csv` | Cards, fouls, aerials, recoveries |

### Data Structure

```
data/raw/
├── 2025-2026/
│   ├── fixtures.csv           # 380 matches per season
│   ├── standard_stats_for.csv # 20 teams
│   ├── standard_stats_against.csv
│   └── ... (17 stat files per season)
├── 2024-2025/
├── 2023-2024/
├── 2022-2023/
├── 2021-2022/
└── 2020-2021/
```

---

## Feature Engineering Strategy

### Design Principles

1. **Domain Knowledge First**: Features are designed based on football analytics expertise
2. **Relative Strength**: Match outcomes depend on relative team strength, not absolute metrics
3. **Temporal Awareness**: Recent form matters more than season averages
4. **Multiple Perspectives**: Capture attacking, defensive, possession, and set-piece dimensions

### Feature Engineering Pipeline

```python
# Pipeline flow
Raw Stats → Team Aggregation → Feature Engineering → Differential Features → Form Features → ELO Ratings → Final Features
```

---

## Feature Categories Deep Dive

### 1. Expected Goals (xG) Based Features ⭐ HIGH IMPORTANCE

xG is the most predictive metric in modern football analytics.

| Feature | Formula | Football Meaning |
|---------|---------|------------------|
| `xg` | Direct from data | Expected goals based on shot quality |
| `npxg` | xG - penalty xG | Non-penalty expected goals (removes penalty luck) |
| `xg_overperformance` | Goals - xG | Finishing above/below expectation |
| `shot_quality` | xG / Shots | Average quality of chances created |
| `xga` | Opponent xG | Expected goals against (defensive quality) |
| `xga_overperformance` | Goals Against - xGA | Defensive over/underperformance |

**Football Analytics Insight:**
- Teams with positive xG overperformance often regress to mean
- npxG is more stable and predictive than raw xG
- xG differential (xG - xGA) is the best single predictor of long-term success

### 2. Shooting Features ⭐ HIGH IMPORTANCE

| Feature | Description | Predictive Value |
|---------|-------------|------------------|
| `shots` | Total shots per 90 | Volume indicator |
| `shots_on_target` | Shots on target per 90 | Accuracy indicator |
| `shot_accuracy` | SoT / Shots | Finishing quality |
| `goal_conversion` | Goals / Shots | Clinical finishing |
| `shot_distance` | Average shot distance | Playing style indicator |

**Football Analytics Insight:**
- Shot volume alone isn't predictive - quality matters
- Top teams create shots from better positions (lower average distance)
- Shot accuracy above 35% indicates clinical finishing

### 3. Goal Creation Features ⭐ HIGH IMPORTANCE

| Feature | Description | Predictive Value |
|---------|-------------|------------------|
| `sca` | Shot Creating Actions | Actions leading to shots |
| `sca_per90` | SCA per 90 minutes | Attacking threat rate |
| `gca` | Goal Creating Actions | Actions directly leading to goals |
| `gca_per90` | GCA per 90 minutes | Creative quality |
| `sca_from_passes` | SCA from live passes | Playing style |
| `sca_from_dribbles` | SCA from take-ons | Individual quality |

**Football Analytics Insight:**
- GCA is more predictive than SCA (closer to goal)
- SCA sources indicate playing style (pass-heavy vs dribble-heavy)
- Top teams average 3.5+ GCA per 90

### 4. Defensive Features ⭐ HIGH IMPORTANCE

| Feature | Description | Predictive Value |
|---------|-------------|------------------|
| `tackles` | Tackles attempted | Defensive engagement |
| `tackles_won` | Successful tackles | Defensive quality |
| `interceptions` | Interceptions per 90 | Reading the game |
| `blocks` | Shot blocks | Last-ditch defending |
| `clearances` | Clearances per 90 | Defensive clearances |
| `errors` | Errors leading to shots | Defensive liability |
| `defensive_actions` | Tackles + Interceptions | Overall defensive intensity |

**Football Analytics Insight:**
- High-block teams have more tackles in attacking third
- Errors leading to shots are highly predictive of goals against
- Clean sheet % is outcome-based but correlates with xGA

### 5. Passing Features 📊 MEDIUM-HIGH IMPORTANCE

| Feature | Description | Predictive Value |
|---------|-------------|------------------|
| `pass_completion` | Pass completion % | Technical quality |
| `progressive_passes` | Passes advancing 10+ yards | Territorial gain |
| `passes_into_final_third` | Passes into attacking third | Attacking intent |
| `passes_into_penalty_area` | Passes into box | Chance creation |
| `key_passes` | Passes leading to shots | Creative output |

**Football Analytics Insight:**
- Raw pass completion is less predictive than progressive passing
- Passes into penalty area strongly correlate with xG
- Long pass completion % indicates playing style

### 6. Possession Features 📊 MEDIUM-HIGH IMPORTANCE

| Feature | Description | Predictive Value |
|---------|-------------|------------------|
| `possession` | Ball possession % | Style indicator |
| `touches_att_3rd` | Touches in attacking third | Territorial dominance |
| `touches_att_pen` | Touches in penalty area | Goal threat |
| `progressive_carries` | Carries advancing 10+ yards | Ball progression |
| `take_ons_success` | Dribble success rate | Individual quality |

**Football Analytics Insight:**
- Possession alone doesn't predict outcomes (correlation ~0.15)
- What matters is QUALITY of possession (progressive actions)
- Touches in attacking penalty area highly correlates with xG

### 7. Form Features 📈 HIGH IMPORTANCE

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `form_3` | Last 3 match points avg | Rolling mean |
| `form_5` | Last 5 match points avg | Rolling mean |
| `form_10` | Last 10 match points avg | Rolling mean |
| `xg_form_5` | Last 5 match xG differential | Rolling mean |
| `home_form` | Home-only form | Rolling mean (home matches) |
| `away_form` | Away-only form | Rolling mean (away matches) |

**Football Analytics Insight:**
- 5-game form is sweet spot (balances recency and sample size)
- xG-based form is more stable than results-based form
- Home/away form differs significantly for most teams

### 8. ELO Rating Features 📈 HIGH IMPORTANCE

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `elo_rating` | Team strength rating | Dynamic ELO calculation |
| `elo_diff` | Home ELO - Away ELO | Strength differential |
| `elo_home_win_prob` | ELO-based probability | 1 / (1 + 10^(-diff/400)) |

**ELO Configuration:**
- K-factor: 32 (standard for football)
- Home advantage: +100 ELO points
- Initial rating: 1500

**Football Analytics Insight:**
- ELO is one of the best single predictors of match outcome
- Captures long-term team strength better than league position
- More stable than form-based metrics

### 9. Set Piece & Discipline Features 📊 MEDIUM IMPORTANCE

| Feature | Description | Predictive Value |
|---------|-------------|------------------|
| `penalties_won` | PKs won per season | Set piece threat |
| `penalties_conceded` | PKs conceded | Defensive liability |
| `yellow_cards` | Yellow cards per 90 | Discipline issues |
| `red_cards` | Red cards per season | Major discipline issues |
| `fouls_committed` | Fouls per 90 | Aggression level |
| `aerial_won_pct` | Aerial duel win % | Physical dominance |

**Football Analytics Insight:**
- ~30% of goals come from set pieces
- Penalty differential is surprisingly predictive
- Aerial dominance matters for set piece defense

### 10. Head-to-Head Features 📊 MEDIUM IMPORTANCE

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `h2h_home_wins` | Home wins in H2H | Last 10 encounters |
| `h2h_draws` | Draws in H2H | Last 10 encounters |
| `h2h_away_wins` | Away wins in H2H | Last 10 encounters |
| `h2h_home_goals_avg` | Avg home goals | Last 10 encounters |
| `h2h_home_dominance` | Home win rate in H2H | h2h_home_wins / total |

**Football Analytics Insight:**
- H2H history matters especially for derbies
- Recent H2H (2-3 years) more predictive than ancient history
- Some teams have psychological edge over specific opponents

### 11. Differential Features (Home - Away) ⭐ CRITICAL

All numeric features are converted to differentials:

| Feature Type | Example | Meaning |
|--------------|---------|---------|
| `diff_xg` | home_xg - away_xg | xG advantage |
| `diff_possession` | home_poss - away_poss | Control advantage |
| `diff_shots` | home_shots - away_shots | Shot volume advantage |
| `diff_tackles` | home_tkl - away_tkl | Defensive pressure diff |

**Football Analytics Insight:**
- Differentials are often more predictive than absolute values
- Match outcomes are determined by relative strength
- Allows direct comparison regardless of league averages

---

## ML Pipeline Architecture

### Pipeline Stages

```
1. Data Loading
   └── Load CSV files for all seasons
   
2. Data Cleaning
   ├── Standardize column names
   ├── Clean team names
   ├── Convert to numeric types
   └── Handle missing values
   
3. Feature Engineering
   ├── Merge team stats to matches
   ├── Create attacking features
   ├── Create defensive features
   ├── Create possession features
   ├── Create form features
   ├── Calculate ELO ratings
   └── Create differential features
   
4. Train/Test Split
   └── Temporal split (train on older, test on recent)
   
5. Preprocessing
   ├── Missing value imputation (median)
   ├── Feature scaling (standard)
   └── Remove highly correlated features (>0.95)
   
6. Model Training
   └── Train ensemble of models
```

### Configuration

```python
PipelineConfig(
    seasons=["2023-2024", "2022-2023", "2021-2022", "2020-2021"],
    test_size=0.2,
    validation_size=0.1,
    use_time_split=True,
    imputation_strategy='median',
    scaling_strategy='standard',
    correlation_threshold=0.95
)
```

---

## Model Selection & Ensemble

### Model Hierarchy

| Model | Type | Strengths | Use Case |
|-------|------|-----------|----------|
| **BaselineModel** | Heuristic | Fast, interpretable | Benchmark |
| **LogisticRegression** | Linear | Fast, probabilistic | Linear baseline |
| **RandomForest** | Ensemble | Robust, no scaling needed | Good default |
| **XGBoost** | Boosting | State-of-the-art, handles missing | Production |
| **LightGBM** | Boosting | Fast, good with categoricals | Production |
| **NeuralNetwork** | Deep Learning | Complex patterns | Experimental |

### Ensemble Strategy

```python
EnsemblePredictor(
    models=[
        LogisticRegressionModel(),
        RandomForestModel(),
        XGBoostModel(),
        LightGBMModel(),
    ],
    method='weighted',  # Weight by validation accuracy
)
```

**Ensemble Methods:**
- `voting`: Simple average of probabilities
- `weighted`: Weighted average based on validation performance
- `stacking`: Meta-learner trained on base model predictions

### Score Prediction (Poisson Model)

For exact score prediction:

```python
# Football scores follow Poisson distribution
P(home_goals=h, away_goals=a) = Poisson(h|λ_home) × Poisson(a|λ_away)

# Where λ_home and λ_away are predicted expected goals
```

---

## Target Variables

### Classification Targets

| Target | Values | Description |
|--------|--------|-------------|
| `match_result` | 0, 1, 2 | Away win, Draw, Home win |
| `home_win` | 0, 1 | Binary home win |
| `draw` | 0, 1 | Binary draw |
| `away_win` | 0, 1 | Binary away win |

### Regression Targets

| Target | Range | Description |
|--------|-------|-------------|
| `home_goals` | 0-9 | Home team goals |
| `away_goals` | 0-9 | Away team goals |
| `total_goals` | 0-12 | Total match goals |
| `goal_difference` | -9 to 9 | Home - Away goals |

### Over/Under Targets

| Target | Description |
|--------|-------------|
| `over_2_5` | Total goals > 2.5 |
| `over_1_5` | Total goals > 1.5 |
| `btts` | Both teams to score |

---

## Feature Importance Analysis

### Tier 1 - Highest Importance

```
1. diff_xg (xG differential)
2. elo_diff (ELO rating differential)
3. home_form_5 (5-game form)
4. diff_npxg (non-penalty xG differential)
5. home_xg_overperformance
6. away_xg_overperformance
7. elo_home_win_prob
8. home_shot_quality
```

### Tier 2 - High Importance

```
9. diff_shots_on_target
10. diff_gca (Goal Creating Actions differential)
11. home_form_3
12. diff_defensive_actions
13. home_xg_form_5
14. diff_progressive_passes
```

### Tier 3 - Medium Importance

```
15. diff_possession
16. h2h_home_dominance
17. diff_tackles_won
18. home_aerial_dominance
19. diff_clean_sheets
```

### Tier 4 - Lower Importance

```
20. diff_yellow_cards
21. home_indiscipline_score
22. diff_fouls
23. attendance (normalized)
24. day_of_week
```

---

## Implementation Reference

### File Structure

```
src/ml/
├── __init__.py              # Package exports
├── feature_engineering.py   # FeatureEngineer class (1128 features)
├── data_pipeline.py         # MLDataPipeline class
├── models.py                # All ML models (6 models)
├── train.py                 # Training CLI script
├── predict.py               # Prediction interface
├── tuning.py                # Hyperparameter tuning
└── process_data.py          # Data combination utilities
```

### Quick Start

```bash
# Train ensemble model
uv run python -m src.ml.train --target match_result --model ensemble

# Evaluate all models
uv run python -m src.ml.train --target match_result --evaluate-all

# Make predictions
uv run python -m src.ml.predict \
    --home "Manchester United" \
    --away "Manchester City" \
    --date "2026-01-17" \
    --season "2025-2026"
```

### Python API

```python
from src.ml import MLDataPipeline, EnsemblePredictor, create_default_ensemble

# Create and run pipeline
pipeline = MLDataPipeline()
train, val, test = pipeline.run_pipeline(target='match_result')

# Train ensemble
ensemble = create_default_ensemble()
ensemble.fit(train[pipeline.feature_columns], train['match_result'],
             val[pipeline.feature_columns], val['match_result'])

# Evaluate
metrics = ensemble.evaluate(test[pipeline.feature_columns], test['match_result'])
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Log Loss: {metrics['log_loss']:.3f}")
```

### Adding New Features

```python
from src.ml.feature_engineering import FeatureEngineer

class CustomFeatureEngineer(FeatureEngineer):
    def _engineer_custom_features(self, df):
        # Add your custom features
        df['custom_feature'] = df['some_col'] / df['other_col']
        return df
```

---

## Model Performance Benchmarks

### Current Performance (January 2026)

| Model | Train Acc | Val Acc | Test Acc | Log Loss |
|-------|-----------|---------|----------|----------|
| Baseline (ELO) | 42.2% | 48.7% | 47.0% | 1.058 |
| Logistic Regression | 66.4% | 57.9% | 53.6% | 1.185 |
| Random Forest | 81.0% | 65.1% | 60.5% | 0.909 |
| XGBoost | 100.0% | 63.2% | 60.5% | 0.860 |
| LightGBM | 100.0% | 63.2% | 59.2% | 0.914 |
| **Ensemble** | 99.6% | **65.8%** | **61.2%** | 0.861 |

### Training Data

- Training Matches: 1,064
- Validation Matches: 152  
- Test Matches: 304
- Features: 584 (after correlation filtering)

### Class Distribution (Training Set)

- Home Win: 42.2%
- Away Win: 35.1%
- Draw: 22.7%

---

## Future Improvements

1. **Player-Level Features**: Incorporate individual player stats and availability
2. **Injury Data**: Add injury news impact on team strength
3. **Betting Odds**: Use market odds as additional features
4. **Contextual Features**: Cup competition, fixture congestion, derby flags
5. **Transfer Window**: Account for squad changes mid-season
6. **Manager Changes**: Track managerial changes and honeymoon periods

---

## References

- [FBref Statistics](https://fbref.com/en/comps/9/Premier-League-Stats)
- [StatsBomb xG Model](https://statsbomb.com/articles/soccer/introducing-expected-goals-on-target-xgot/)
- [ELO Rating System](https://en.wikipedia.org/wiki/Elo_rating_system)
- [Poisson Distribution in Football](https://www.pinnacle.com/en/betting-articles/Soccer/how-to-calculate-poisson-distribution/MD62MLXG3H35COYJ)

---