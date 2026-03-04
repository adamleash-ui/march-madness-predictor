"""
March Madness bracket predictor.

Trains on historical tournament matchup features produced by features.py
and generates win-probability predictions in Kaggle submission format.

Uses season-aware walk-forward cross-validation: train on seasons 1..N,
validate on season N+1. This mirrors real deployment conditions where we
predict a tournament before it happens.

Ensemble approach: trains logistic regression, random forest, and gradient
boosting independently, then learns optimal blend weights via a meta-learner
trained on out-of-fold predictions (stacking).
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"

FEATURE_COLS = [
    "diff_win_rate", "diff_avg_score", "diff_avg_score_allowed", "diff_avg_margin",
    "diff_off_eff", "diff_def_eff", "diff_net_eff",
    "diff_tempo",
    "diff_fg_pct", "diff_fg3_pct", "diff_ft_pct",
    "diff_oreb_rate", "diff_ast_to_ratio", "diff_stl_per_game", "diff_blk_per_game",
    "diff_seed_num", "diff_massey_rank_mean", "diff_massey_rank_min",
]

N_FOLDS = 5
MIN_TRAIN_SEASONS = 5   # require at least 5 seasons of history before validating


def load_training_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Returns (X, y, seasons) — seasons needed for walk-forward CV."""
    path = DATA_DIR / "train_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run src/features.py first to generate train_features.csv")
    df = pd.read_csv(path).dropna(subset=FEATURE_COLS)
    return df[FEATURE_COLS], df["label"], df["Season"]


# ---------------------------------------------------------------------------
# Season-aware walk-forward CV splitter
# ---------------------------------------------------------------------------

class SeasonWalkForwardCV:
    """
    Walk-forward cross-validator that splits by season.

    For each validation season S (starting after MIN_TRAIN_SEASONS):
        train  = all games from seasons before S
        valid  = all games from season S

    This mirrors real-world use: we only know past seasons when predicting
    a future tournament.
    """

    def __init__(self, min_train_seasons: int = MIN_TRAIN_SEASONS):
        self.min_train_seasons = min_train_seasons

    def split(self, X: pd.DataFrame, y: pd.Series, seasons: pd.Series):
        unique_seasons = sorted(seasons.unique())
        for i, val_season in enumerate(unique_seasons):
            if i < self.min_train_seasons:
                continue
            train_idx = seasons[seasons < val_season].index
            val_idx   = seasons[seasons == val_season].index
            yield (
                X.index.get_indexer(train_idx),
                X.index.get_indexer(val_idx),
            )

    def get_n_splits(self, seasons: pd.Series) -> int:
        return max(0, seasons.nunique() - self.min_train_seasons)


def _season_cross_validate(pipe, X, y, seasons, scoring_fns: dict) -> dict:
    """Run walk-forward CV and return mean scores."""
    cv = SeasonWalkForwardCV()
    fold_scores = {k: [] for k in scoring_fns}

    for train_idx, val_idx in cv.split(X, y, seasons):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)

        for name, fn in scoring_fns.items():
            fold_scores[name].append(fn(y_val, probs if "loss" in name or "auc" in name or "brier" in name else preds))

    return {k: np.mean(v) for k, v in fold_scores.items()}


def _season_oof_predict(pipe, X, y, seasons) -> np.ndarray:
    """Collect out-of-fold probability predictions using walk-forward CV."""
    cv = SeasonWalkForwardCV()
    oof = np.full(len(X), np.nan)

    for train_idx, val_idx in cv.split(X, y, seasons):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        pipe.fit(X_tr, y_tr)
        oof[val_idx] = pipe.predict_proba(X_val)[:, 1]

    return oof


# ---------------------------------------------------------------------------
# Base pipelines
# ---------------------------------------------------------------------------

def build_base_pipelines() -> dict[str, Pipeline]:
    """Three base learners, each wrapped in a StandardScaler pipeline."""
    return {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=500, max_depth=4, min_samples_leaf=10,
                random_state=42, n_jobs=-1,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Evaluation — random CV vs season-aware CV side-by-side
# ---------------------------------------------------------------------------

def evaluate_models(X: pd.DataFrame, y: pd.Series, seasons: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two DataFrames:
        random_results   — classic 5-fold stratified CV
        seasonal_results — walk-forward season CV
    """
    random_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    scoring = ["accuracy", "roc_auc", "neg_log_loss", "neg_brier_score"]
    scoring_fns = {
        "accuracy":    lambda yt, yp: accuracy_score(yt, yp >= 0.5),
        "roc_auc":     roc_auc_score,
        "log_loss":    log_loss,
        "brier_score": brier_score_loss,
    }

    random_results = []
    seasonal_results = []
    oof_preds = {}

    for name, pipe in build_base_pipelines().items():
        print(f"  Evaluating {name}...")

        # Random CV
        r = cross_validate(pipe, X, y, cv=random_cv, scoring=scoring)
        random_results.append({
            "model": name,
            "accuracy":    r["test_accuracy"].mean(),
            "roc_auc":     r["test_roc_auc"].mean(),
            "log_loss":   -r["test_neg_log_loss"].mean(),
            "brier_score":-r["test_neg_brier_score"].mean(),
        })

        # Season-aware CV
        s = _season_cross_validate(pipe, X, y, seasons, scoring_fns)
        seasonal_results.append({"model": name, **s})

        # OOF preds (season-aware) for ensemble meta-learner
        oof_preds[name] = _season_oof_predict(pipe, X, y, seasons)

    # Ensemble evaluation (season-aware OOF, ignoring NaN rows with no history)
    print("  Evaluating ensemble...")
    mask = ~np.isnan(list(oof_preds.values())[0])
    blend = np.column_stack([oof_preds[n][mask] for n in oof_preds]).mean(axis=1)
    y_masked = y.values[mask]
    random_results.append({
        "model": "ensemble",
        "accuracy":    accuracy_score(y_masked, blend >= 0.5),
        "roc_auc":     roc_auc_score(y_masked, blend),
        "log_loss":    log_loss(y_masked, blend),
        "brier_score": brier_score_loss(y_masked, blend),
    })
    seasonal_results.append({
        "model": "ensemble",
        "accuracy":    accuracy_score(y_masked, blend >= 0.5),
        "roc_auc":     roc_auc_score(y_masked, blend),
        "log_loss":    log_loss(y_masked, blend),
        "brier_score": brier_score_loss(y_masked, blend),
    })

    rdf = pd.DataFrame(random_results).set_index("model").round(4)
    sdf = pd.DataFrame(seasonal_results).set_index("model").round(4)
    return rdf, sdf


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class EnsemblePredictor:
    """
    Stacked ensemble of logistic, random forest, and gradient boosting.

    Uses season-aware walk-forward CV to generate OOF predictions for the
    meta-learner, avoiding leakage from future seasons into past predictions.
    """

    def __init__(self):
        self.base_models: dict[str, CalibratedClassifierCV] = {}
        self.meta_model: LogisticRegression | None = None
        self.base_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, seasons: pd.Series):
        pipelines = build_base_pipelines()
        self.base_names = list(pipelines.keys())

        # Step 1: season-aware OOF predictions for meta-learner
        oof_matrix = np.zeros((len(X), len(pipelines)))
        for i, (name, pipe) in enumerate(pipelines.items()):
            print(f"  Generating season-aware OOF: {name}...")
            oof = _season_oof_predict(pipe, X, y, seasons)
            # Fill early seasons (no history) with 0.5
            oof = np.where(np.isnan(oof), 0.5, oof)
            oof_matrix[:, i] = oof

        # Step 2: fit meta-learner
        print("  Fitting meta-learner...")
        self.meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.meta_model.fit(oof_matrix, y)

        weights = self.meta_model.coef_[0]
        print("  Blend weights:")
        for name, w in zip(self.base_names, weights):
            print(f"    {name:<20} {w:+.4f}")

        # Step 3: fit each base model on ALL data
        for name, pipe in pipelines.items():
            print(f"  Training final base model: {name}...")
            calibrated = CalibratedClassifierCV(pipe, cv=N_FOLDS, method="isotonic")
            calibrated.fit(X, y)
            self.base_models[name] = calibrated

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        meta_X = np.column_stack([
            self.base_models[name].predict_proba(X)[:, 1]
            for name in self.base_names
        ])
        return self.meta_model.predict_proba(meta_X)

    def save(self, path: Path):
        joblib.dump(self, path)
        print(f"  Saved ensemble to {path}")

    @staticmethod
    def load(path: Path) -> "EnsemblePredictor":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def generate_submission(model: EnsemblePredictor) -> pd.DataFrame:
    """Predict win probabilities for ALL submission matchups. Missing → 0.5."""
    sub_path = DATA_DIR / "submission_features.csv"
    sample_path = DATA_DIR / "SampleSubmissionStage1.csv"
    if not sub_path.exists():
        raise FileNotFoundError("Run src/features.py first to generate submission_features.csv")

    sub = pd.read_csv(sub_path)
    sample = pd.read_csv(sample_path)[["ID"]]

    has_data = sub[FEATURE_COLS].notna().all(axis=1)
    probs = np.full(len(sub), 0.5)
    if has_data.any():
        probs[has_data] = model.predict_proba(sub.loc[has_data, FEATURE_COLS])[:, 1]

    pred_df = pd.DataFrame({"ID": sub["ID"], "Pred": probs.round(4)})
    out = sample.merge(pred_df, on="ID", how="left")
    out["Pred"] = out["Pred"].fillna(0.5)

    out_path = DATA_DIR / "submission.csv"
    out.to_csv(out_path, index=False)
    print(f"  {has_data.sum():,} rows modeled, {len(out) - has_data.sum():,} filled with 0.5")
    print(f"  Saved {len(out):,} predictions to {out_path}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading training data...")
    X, y, seasons = load_training_data()
    print(f"  {X.shape[0]} games, {X.shape[1]} features, {seasons.nunique()} seasons\n")

    print("Comparing models: random CV vs season-aware walk-forward CV...")
    random_results, seasonal_results = evaluate_models(X, y, seasons)

    print("\n--- Random 5-fold CV ---")
    print(random_results.to_string())
    print("\n--- Season-aware walk-forward CV ---")
    print(seasonal_results.to_string())

    print("\nTraining ensemble on all data (season-aware OOF)...")
    ensemble = EnsemblePredictor()
    ensemble.fit(X, y, seasons)

    MODEL_DIR.mkdir(exist_ok=True)
    ensemble.save(MODEL_DIR / "bracket_predictor.joblib")

    print("\nGenerating submission predictions...")
    generate_submission(ensemble)

    print("\nDone.")
