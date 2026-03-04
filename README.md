# March Madness Predictor

A machine learning pipeline that predicts NCAA Tournament game outcomes for the [Kaggle March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2025) competition.

**Model:** Stacked ensemble (logistic + random forest + gradient boosting) — 69.8% accuracy / 0.573 log loss (season-aware walk-forward CV across 1,382 tournament games, 2003–2024)

## Setup

```bash
git clone https://github.com/adamleash-ui/march-madness-predictor.git
cd march-madness-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place your Kaggle API credentials at `~/.kaggle/kaggle.json` (chmod 600), then accept the competition rules at kaggle.com/competitions/march-machine-learning-mania-2025/data.

## Usage

```bash
# 1. Download data
python src/get_data.py

# 2. Engineer features
python src/features.py

# 3. Train ensemble + generate submission
python src/predictor.py

# 4. Visualize predictions
python src/visualize.py
```

## Project Structure

```
march-madness-predictor/
├── data/                        # Downloaded CSVs and generated feature files (gitignored)
├── models/                      # Saved model (bracket_predictor.joblib)
├── notebooks/                   # Jupyter notebooks for exploration
├── plots/                       # Generated visualizations
├── src/
│   ├── get_data.py              # Kaggle dataset download
│   ├── features.py              # Feature engineering pipeline
│   ├── predictor.py             # Ensemble training and submission generation
│   └── visualize.py             # Prediction visualizations
└── requirements.txt
```

## Features

22 matchup features computed as **Team1 − Team2 differentials**:

| Category | Features |
|---|---|
| Seeding | Seed number |
| Rankings | Massey ordinal mean & best rank (pre-Selection Sunday) |
| Efficiency | Offensive efficiency, defensive efficiency, net efficiency |
| Scoring | Avg points scored, avg points allowed, avg margin |
| Shooting | FG%, 3P%, FT% |
| Pace | Tempo (possessions per game) |
| Recent form | Win rate & net efficiency in last 10 games; trend vs season average |
| Other | Offensive rebound rate, AST/TO ratio, steals/game, blocks/game |

## Model Pipeline

1. **Base learners** — logistic regression, random forest, gradient boosting, each in a `StandardScaler` pipeline
2. **Season-aware OOF** — walk-forward cross-validation (train on seasons 1..N, validate on N+1) generates out-of-fold predictions for each base learner without leaking future data
3. **Meta-learner** — logistic regression learns optimal blend weights from OOF predictions (logistic +2.43, random forest +1.09, gradient boosting +0.59)
4. **Final fit** — each base model is retrained on all data with isotonic calibration (5-fold `CalibratedClassifierCV`)
5. **Submission** — all 507,108 required matchup pairs predicted; pairs without data default to 0.5

## CV Results

Season-aware walk-forward CV (the temporally honest metric):

| Model | Accuracy | ROC-AUC | Log Loss |
|---|---|---|---|
| Logistic | 69.8% | 0.777 | 0.573 |
| Random Forest | 69.6% | 0.764 | 0.577 |
| Gradient Boosting | 67.3% | 0.753 | 0.632 |
| **Ensemble** | **69.5%** | **0.770** | **0.575** |

Season-aware CV reveals gradient boosting overfits significantly (0.594 → 0.632 log loss vs random CV), which is why the meta-learner down-weights it.

## Visualizations

| Plot | Description |
|---|---|
| `1_pred_distribution.png` | Distribution of predicted win probabilities |
| `2_seed_win_prob.png` | Average model win probability by seed differential |
| `3_upset_heatmap.png` | Historical upset rates by seed matchup |
| `4_calibration_curve.png` | Predicted vs. actual win rates |
| `5_feature_importances.png` | Feature importance by logistic model coefficient magnitude |
| `6_top_predictions.png` | Most competitive predicted matchups (seed diff ≤ 4) |
