"""
Visualization suite for March Madness bracket predictions.

Run directly to generate and save all plots:
    python src/visualize.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from sklearn.calibration import calibration_curve
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.predictor import EnsemblePredictor, FEATURE_COLS  # noqa: F401 — needed for joblib unpickling

DATA_DIR  = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
PLOT_DIR  = Path(__file__).parent.parent / "plots"

BLUE = "#1f6fad"
RED  = "#c0392b"
GOLD = "#f39c12"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str):
    PLOT_DIR.mkdir(exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


def _load_artifacts():
    sub  = pd.read_csv(DATA_DIR / "submission.csv")
    subf = pd.read_csv(DATA_DIR / "submission_features.csv")
    train = pd.read_csv(DATA_DIR / "train_features.csv")
    seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    tourney = pd.read_csv(DATA_DIR / "MNCAATourneyCompactResults.csv")
    model = joblib.load(MODEL_DIR / "bracket_predictor.joblib")
    features = FEATURE_COLS
    return sub, subf, train, seeds, tourney, model, features


# ---------------------------------------------------------------------------
# 1. Prediction probability distribution
# ---------------------------------------------------------------------------

def plot_pred_distribution(sub: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(sub["Pred"], bins=60, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.axvline(0.5, color=RED, linestyle="--", linewidth=1.5, label="50 % (coin flip)")
    ax.set_xlabel("Predicted Win Probability (lower-seed team)")
    ax.set_ylabel("Number of Matchups")
    ax.set_title("Distribution of Predicted Win Probabilities")
    ax.legend()
    fig.tight_layout()
    _save(fig, "1_pred_distribution.png")


# ---------------------------------------------------------------------------
# 2. Seed-matchup win-probability heatmap  (1 vs 16, 2 vs 15, …)
# ---------------------------------------------------------------------------

def _seed_win_prob(subf: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    merged = subf[["ID", "diff_seed_num"]].copy()
    merged = merged.merge(sub, on="ID")
    merged = merged.dropna(subset=["diff_seed_num"])
    merged["seed_diff"] = merged["diff_seed_num"].astype(int)
    return merged.groupby("seed_diff")["Pred"].mean().reset_index()


def plot_seed_win_prob(subf: pd.DataFrame, sub: pd.DataFrame):
    df = _seed_win_prob(subf, sub)
    # Only classic first-round pairings: diffs -15 … +15 in steps of 1
    df = df[df["seed_diff"].between(-15, 15)]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [RED if p < 0.5 else BLUE for p in df["Pred"]]
    ax.bar(df["seed_diff"], df["Pred"], color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Seed Differential  (T1 seed − T2 seed)\nnegative = T1 is the better seed")
    ax.set_ylabel("Avg Predicted Win Prob (T1)")
    ax.set_title("Average Model Win Probability by Seed Differential")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    fig.tight_layout()
    _save(fig, "2_seed_win_prob.png")


# ---------------------------------------------------------------------------
# 3. Historical upset rate by seed matchup  (triangular heatmap)
# ---------------------------------------------------------------------------

def _build_upset_matrix(seeds_raw: pd.DataFrame, tourney: pd.DataFrame):
    seeds = seeds_raw.copy()
    seeds["seed_num"] = seeds["Seed"].str[1:3].astype(int)
    t = tourney.merge(seeds[["Season","TeamID","seed_num"]],
                      left_on=["Season","WTeamID"], right_on=["Season","TeamID"])
    t = t.rename(columns={"seed_num":"W_seed"}).drop(columns="TeamID")
    t = t.merge(seeds[["Season","TeamID","seed_num"]],
                left_on=["Season","LTeamID"], right_on=["Season","TeamID"])
    t = t.rename(columns={"seed_num":"L_seed"}).drop(columns="TeamID")

    # For each unique seed pair, record whether the better (lower #) seed won
    t["better_seed"] = t[["W_seed","L_seed"]].min(axis=1)
    t["worse_seed"]  = t[["W_seed","L_seed"]].max(axis=1)
    t["better_won"]  = t["W_seed"] < t["L_seed"]

    pivot = t.groupby(["better_seed","worse_seed"])["better_won"].mean().unstack()
    return pivot


def plot_upset_heatmap(seeds_raw: pd.DataFrame, tourney: pd.DataFrame):
    pivot = _build_upset_matrix(seeds_raw, tourney)
    seeds_range = list(range(1, 17))
    matrix = pd.DataFrame(index=seeds_range, columns=seeds_range, dtype=float)
    for r in seeds_range:
        for c in seeds_range:
            if c > r and r in pivot.index and c in pivot.columns:
                matrix.loc[r, c] = pivot.loc[r, c]

    cmap = LinearSegmentedColormap.from_list("rw", [RED, "white", BLUE])
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix.values.astype(float), cmap=cmap, vmin=0, vmax=1,
                   aspect="auto", origin="upper")
    ax.set_xticks(range(16))
    ax.set_xticklabels(seeds_range, fontsize=8)
    ax.set_yticks(range(16))
    ax.set_yticklabels(seeds_range, fontsize=8)
    ax.set_xlabel("Worse Seed (higher number)")
    ax.set_ylabel("Better Seed (lower number)")
    ax.set_title("Historical Win Rate of Better Seed by Matchup\n(blue=better seed dominates, red=frequent upset)")

    # annotate cells
    for i in range(16):
        for j in range(16):
            val = matrix.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=7, color="black")

    plt.colorbar(im, ax=ax, label="Better-seed win rate")
    fig.tight_layout()
    _save(fig, "3_upset_heatmap.png")


# ---------------------------------------------------------------------------
# 4. Calibration curve
# ---------------------------------------------------------------------------

def plot_calibration(model, train: pd.DataFrame):
    features = [c for c in train.columns if c.startswith("diff_")]
    Xtr = train[features].dropna()
    ytr = train.loc[Xtr.index, "label"]

    prob_pred = model.predict_proba(Xtr)[:, 1]
    frac_pos, mean_pred = calibration_curve(ytr, prob_pred, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color=BLUE, linewidth=2,
            markersize=6, label="Model")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (actual win rate)")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, "4_calibration_curve.png")


# ---------------------------------------------------------------------------
# 5. Feature importances
# ---------------------------------------------------------------------------

def plot_feature_importances(model: EnsemblePredictor):
    """Extract and plot feature importances from the logistic base model."""
    logistic = model.base_models.get("logistic")
    if logistic is None:
        print("  Skipping feature importances (logistic model not found)")
        return

    importances = []
    for est in logistic.calibrated_classifiers_:
        clf = est.estimator.named_steps["clf"]
        if hasattr(clf, "coef_"):
            importances.append(np.abs(clf.coef_[0]))

    if not importances:
        print("  Skipping feature importances (coef_ not available)")
        return

    avg = np.mean(importances, axis=0)
    feat_labels = [f.replace("diff_", "").replace("_", " ") for f in FEATURE_COLS]
    s = pd.Series(avg, index=feat_labels).sort_values()

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = [GOLD if v == s.max() else BLUE for v in s.values]
    s.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Feature Importances — Logistic Base Model (avg |coefficient|)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    _save(fig, "5_feature_importances.png")


# ---------------------------------------------------------------------------
# 6. 2025 high-confidence predictions
# ---------------------------------------------------------------------------

def plot_top_predictions(sub: pd.DataFrame, subf: pd.DataFrame, n: int = 20):
    merged = sub.merge(subf[["ID", "diff_seed_num", "diff_net_eff"]], on="ID")
    # Restrict to seeded-vs-seeded matchups with close seeds (|diff| <= 4)
    merged = merged.dropna(subset=["diff_seed_num"])
    merged = merged[merged["diff_seed_num"].abs() <= 4]
    # Use most recent season available
    merged["season"] = merged["ID"].str.split("_").str[0]
    latest = merged["season"].max()
    latest_df = merged[merged["season"] == latest].copy()
    # "Toss-up" games: closest to 50%
    latest_df["distance_from_even"] = (latest_df["Pred"] - 0.5).abs()
    top = latest_df.nsmallest(n, "distance_from_even").sort_values("Pred")

    colors = [RED if p < 0.5 else BLUE for p in top["Pred"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top)), top["Pred"] - 0.5, color=colors,
            edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["ID"].str[5:], fontsize=8)
    ax.axvline(0, color="black", linewidth=1)
    # Annotate: always place text just inside the center line for readability
    for i, (_, row) in enumerate(top.iterrows()):
        edge = row["Pred"] - 0.5
        label = f'{row["Pred"]:.1%}'
        if edge >= 0:
            ax.text(edge + 0.001, i, label, va="center", ha="left", fontsize=7.5, color="white" if edge > 0.01 else "black")
        else:
            ax.text(edge - 0.001, i, label, va="center", ha="right", fontsize=7.5, color="white" if edge < -0.01 else "black")
    ax.set_xlabel("Predicted Edge over 50 %")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title(f"{latest} Closest Predicted Matchups (seed diff ≤ 4)\n"
                 f"blue = T1 favored  |  red = T2 favored")
    fig.tight_layout()
    _save(fig, "6_top_predictions.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data and model...")
    sub, subf, train, seeds_raw, tourney, model, features = _load_artifacts()

    print("Generating plots...")
    plot_pred_distribution(sub)
    plot_seed_win_prob(subf, sub)
    plot_upset_heatmap(seeds_raw, tourney)
    plot_calibration(model, train)
    plot_feature_importances(model)
    plot_top_predictions(sub, subf)

    print("\nAll plots saved to plots/")
