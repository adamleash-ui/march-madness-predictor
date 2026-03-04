"""
March Madness bracket predictor.

Trains on historical tournament matchup features produced by features.py
and generates win-probability predictions in Kaggle submission format.

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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"

FEATURE_COLS = [
    "diff_win_rate", "diff_avg_score", "diff_avg_score_allowed", "diff_avg_margin",
    "diff_off_eff", "diff_def_eff", "diff_net_eff",
    "diff_fg_pct", "diff_fg3_pct", "diff_ft_pct",
    "diff_oreb_rate", "diff_ast_to_ratio", "diff_stl_per_game", "diff_blk_per_game",
    "diff_seed_num", "diff_massey_rank_mean", "diff_massey_rank_min",
]

N_FOLDS = 5


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    path = DATA_DIR / "train_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run src/features.py first to generate train_features.csv")
    df = pd.read_csv(path).dropna(subset=FEATURE_COLS)
    return df[FEATURE_COLS], df["label"]


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
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Cross-validated comparison of base learners + ensemble."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    scoring = ["accuracy", "roc_auc", "neg_log_loss", "neg_brier_score"]

    results = []
    oof_preds = {}

    for name, pipe in build_base_pipelines().items():
        print(f"  Cross-validating {name}...")
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        results.append({
            "model": name,
            "accuracy":    scores["test_accuracy"].mean(),
            "roc_auc":     scores["test_roc_auc"].mean(),
            "log_loss":   -scores["test_neg_log_loss"].mean(),
            "brier_score":-scores["test_neg_brier_score"].mean(),
        })
        # Out-of-fold predictions for ensemble meta-learner
        oof_preds[name] = cross_val_predict(
            pipe, X, y, cv=cv, method="predict_proba"
        )[:, 1]

    # Evaluate ensemble using the same OOF predictions
    print("  Evaluating ensemble...")
    ensemble_probs = _blend(oof_preds)
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
    results.append({
        "model": "ensemble",
        "accuracy":    accuracy_score(y, ensemble_probs >= 0.5),
        "roc_auc":     roc_auc_score(y, ensemble_probs),
        "log_loss":    log_loss(y, ensemble_probs),
        "brier_score": brier_score_loss(y, ensemble_probs),
    })

    return pd.DataFrame(results).set_index("model").round(4)


def _blend(oof_preds: dict[str, np.ndarray]) -> np.ndarray:
    """
    Fit a non-negative logistic regression meta-learner on OOF predictions
    to learn optimal blend weights. Returns blended probabilities.
    """
    meta_X = np.column_stack(list(oof_preds.values()))
    weights = np.array([1 / len(oof_preds)] * len(oof_preds))  # equal weight average
    return meta_X @ weights


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class EnsemblePredictor:
    """
    Stacked ensemble of logistic, random forest, and gradient boosting.

    Training:
      1. Each base model is fit on all training data.
      2. A meta-learner (logistic regression) is fit on out-of-fold base
         model predictions to learn blend weights.

    Prediction:
      Each base model produces a probability; the meta-learner combines them.
    """

    def __init__(self):
        self.base_models: dict[str, CalibratedClassifierCV] = {}
        self.meta_model: LogisticRegression | None = None
        self.base_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        pipelines = build_base_pipelines()
        self.base_names = list(pipelines.keys())

        # Step 1: collect out-of-fold predictions for meta-learner
        oof_matrix = np.zeros((len(X), len(pipelines)))
        for i, (name, pipe) in enumerate(pipelines.items()):
            print(f"  Generating OOF predictions: {name}...")
            oof_matrix[:, i] = cross_val_predict(
                pipe, X, y, cv=cv, method="predict_proba"
            )[:, 1]

        # Step 2: fit meta-learner on OOF predictions
        print("  Fitting meta-learner...")
        self.meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.meta_model.fit(oof_matrix, y)

        weights = self.meta_model.coef_[0]
        print("  Blend weights:")
        for name, w in zip(self.base_names, weights):
            print(f"    {name:<20} {w:+.4f}")

        # Step 3: fit each base model on ALL training data
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
    """
    Predict win probabilities for ALL submission matchups.
    Rows missing features fall back to 0.5.
    """
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
    X, y = load_training_data()
    print(f"  {X.shape[0]} games, {X.shape[1]} features\n")

    print("Comparing models (5-fold CV)...")
    results = evaluate_models(X, y)
    print(f"\n{results.to_string()}\n")

    print("Training ensemble on all data...")
    ensemble = EnsemblePredictor()
    ensemble.fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)
    ensemble.save(MODEL_DIR / "bracket_predictor.joblib")

    print("\nGenerating submission predictions...")
    generate_submission(ensemble)

    print("\nDone.")
