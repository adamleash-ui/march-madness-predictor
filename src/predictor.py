"""
March Madness bracket predictor.

Trains on historical tournament matchup features produced by features.py
and generates win-probability predictions in Kaggle submission format.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
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


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    path = DATA_DIR / "train_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run src/features.py first to generate train_features.csv")
    df = pd.read_csv(path).dropna(subset=FEATURE_COLS)
    return df[FEATURE_COLS], df["label"]


def build_pipelines() -> dict[str, Pipeline]:
    """Three candidate models, all wrapped in a StandardScaler pipeline."""
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


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Cross-validated comparison across all candidate models."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "roc_auc", "neg_log_loss", "neg_brier_score"]

    results = []
    for name, pipe in build_pipelines().items():
        print(f"  Cross-validating {name}...")
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        results.append({
            "model": name,
            "accuracy":    scores["test_accuracy"].mean(),
            "roc_auc":     scores["test_roc_auc"].mean(),
            "log_loss":   -scores["test_neg_log_loss"].mean(),
            "brier_score":-scores["test_neg_brier_score"].mean(),
        })

    return pd.DataFrame(results).set_index("model").round(4)


def train_final_model(X: pd.DataFrame, y: pd.Series, model_name: str) -> Pipeline:
    """
    Train the chosen model on all data with Platt scaling for calibration,
    then save it to models/.
    """
    pipe = build_pipelines()[model_name]
    # Calibrate probabilities with isotonic regression (5-fold internal CV)
    calibrated = CalibratedClassifierCV(pipe, cv=5, method="isotonic")
    calibrated.fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)
    out_path = MODEL_DIR / "bracket_predictor.joblib"
    joblib.dump({"model": calibrated, "features": FEATURE_COLS}, out_path)
    print(f"  Saved to {out_path}")
    return calibrated


def feature_importances(model: CalibratedClassifierCV) -> pd.Series:
    """
    Extract feature importances from the underlying estimator.
    Averages across calibration folds for robustness.
    """
    importances = []
    for est in model.calibrated_classifiers_:
        base = est.estimator
        # Walk through the Pipeline to get the raw classifier
        clf = base.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances.append(clf.feature_importances_)
        elif hasattr(clf, "coef_"):
            importances.append(np.abs(clf.coef_[0]))

    if not importances:
        return pd.Series(dtype=float)

    avg = np.mean(importances, axis=0)
    return pd.Series(avg, index=FEATURE_COLS).sort_values(ascending=False)


def generate_submission(model: CalibratedClassifierCV) -> pd.DataFrame:
    """
    Predict win probabilities for ALL submission matchups.

    Rows with complete features get model predictions.
    Rows missing features (no seed / Massey data) fall back to 0.5
    so the submission always matches the sample file exactly.
    """
    sub_path = DATA_DIR / "submission_features.csv"
    sample_path = DATA_DIR / "SampleSubmissionStage1.csv"
    if not sub_path.exists():
        raise FileNotFoundError("Run src/features.py first to generate submission_features.csv")

    sub = pd.read_csv(sub_path)
    sample = pd.read_csv(sample_path)[["ID"]]  # ground-truth ID list

    # Predict where we have complete features; impute 0.5 elsewhere
    has_data = sub[FEATURE_COLS].notna().all(axis=1)
    probs = np.full(len(sub), 0.5)
    if has_data.any():
        probs[has_data] = model.predict_proba(sub.loc[has_data, FEATURE_COLS])[:, 1]

    pred_df = pd.DataFrame({"ID": sub["ID"], "Pred": probs.round(4)})

    # Left-join onto the full sample to guarantee correct row count & order
    out = sample.merge(pred_df, on="ID", how="left")
    out["Pred"] = out["Pred"].fillna(0.5)

    out_path = DATA_DIR / "submission.csv"
    out.to_csv(out_path, index=False)
    modeled = has_data.sum()
    print(f"  {modeled:,} rows modeled, {len(out) - modeled:,} filled with 0.5")
    print(f"  Saved {len(out):,} predictions to {out_path}")
    return out


if __name__ == "__main__":
    print("Loading training data...")
    X, y = load_training_data()
    print(f"  {X.shape[0]} games, {X.shape[1]} features\n")

    print("Comparing models (5-fold CV)...")
    results = evaluate_models(X, y)
    print(f"\n{results.to_string()}\n")

    best_model = results["log_loss"].idxmin()
    print(f"Best model by log loss: {best_model}")

    print(f"\nTraining final {best_model} model on all data...")
    model = train_final_model(X, y, best_model)

    print("\nFeature importances:")
    imps = feature_importances(model)
    if not imps.empty:
        for feat, imp in imps.items():
            bar = "█" * int(imp * 40)
            print(f"  {feat:<30} {imp:.4f} {bar}")

    print("\nGenerating submission predictions...")
    generate_submission(model)

    print("\nDone.")
