# March Madness Predictor

A machine learning pipeline that predicts NCAA Tournament game outcomes for the [Kaggle March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2025) competition.

**Model:** Calibrated logistic regression — 70.8% accuracy / 0.557 log loss (5-fold CV across 1,382 tournament games, 2003–2024)

## Setup

```bash
git clone https://github.com/<your-username>/march-madness-predictor.git
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

# 3. Train model + generate submission
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
│   ├── predictor.py             # Model training and submission generation
│   └── visualize.py             # Prediction visualizations
└── requirements.txt
```

## Features

17 matchup features computed as **Team1 − Team2 differentials**:

| Category | Features |
|---|---|
| Seeding | Seed number |
| Rankings | Massey ordinal mean & best rank (pre-Selection Sunday) |
| Efficiency | Offensive efficiency, defensive efficiency, net efficiency |
| Scoring | Avg points scored, avg points allowed, avg margin |
| Shooting | FG%, 3P%, FT% |
| Other | Offensive rebound rate, AST/TO ratio, steals/game, blocks/game |

## Model Pipeline

1. **Candidate models** — logistic regression, random forest, gradient boosting — evaluated via 5-fold stratified CV
2. **Selection** — best model chosen by log loss
3. **Calibration** — isotonic regression calibration (5-fold) for well-calibrated probabilities
4. **Submission** — all 507,108 required matchup pairs predicted; pairs without seed/ranking data default to 0.5

## Visualizations

| Plot | Description |
|---|---|
| `1_pred_distribution.png` | Distribution of predicted win probabilities |
| `2_seed_win_prob.png` | Average model win probability by seed differential |
| `3_upset_heatmap.png` | Historical upset rates by seed matchup |
| `4_calibration_curve.png` | Predicted vs. actual win rates |
| `5_feature_importances.png` | Feature importance by model coefficient magnitude |
| `6_top_predictions.png` | Most competitive predicted matchups (seed diff ≤ 4) |
