"""
Generate 2026 tournament predictions using Stage 2 submission pairs.
Since the 2026 bracket hasn't been announced, there are no official seeds.
We use Massey ranking percentile as a seed proxy so the model still runs.

Run from project root:
    python web/scripts/predict_2025.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.features import (
    compute_team_season_stats,
    compute_recent_form,
    compute_massey_ranks,
    compute_sos,
    compute_adjusted_efficiency,
    compute_all_coach_features,
    build_team_profile,
    build_submission_data,
    extract_seeds,
    DIFF_STATS,
)
from src.predictor import EnsemblePredictor, FEATURE_COLS

DATA  = ROOT / "data"
MODEL = ROOT / "models" / "bracket_predictor.joblib"
OUT   = DATA / "submission_2025.csv"

SEASON = 2026


def infer_seeds_from_massey(massey_raw: pd.DataFrame, season: int, n_teams: int = 68) -> pd.DataFrame:
    """
    Create synthetic seed assignments from Massey ordinals for teams without
    official seeds. Maps the top n_teams by consensus rank to seeds 1-16 across
    4 placeholder regions.
    """
    pre = massey_raw[
        (massey_raw["Season"] == season) &
        (massey_raw["RankingDayNum"] <= 133)
    ]
    consensus = pre.groupby("TeamID")["OrdinalRank"].mean().sort_values()
    top = consensus.head(n_teams).reset_index()
    top["rank_pos"] = range(1, len(top) + 1)

    regions = ["W", "X", "Y", "Z"]
    seed_rows = []
    for i, row in top.iterrows():
        region = regions[i % 4]
        seed_num = (i // 4) + 1
        seed_str = f"{region}{seed_num:02d}"
        seed_rows.append({
            "Season": season,
            "TeamID": int(row["TeamID"]),
            "Seed": seed_str,
            "seed_num": seed_num,
            "seed_region": region,
        })
    return pd.DataFrame(seed_rows)


def main():
    print("Loading data...")
    detailed    = pd.read_csv(DATA / "MRegularSeasonDetailedResults.csv")
    compact     = pd.read_csv(DATA / "MRegularSeasonCompactResults.csv")
    massey_raw  = pd.read_csv(DATA / "MMasseyOrdinals.csv")
    seeds_raw   = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
    coaches_raw = pd.read_csv(DATA / "MTeamCoaches.csv")
    stage2      = pd.read_csv(DATA / "SampleSubmissionStage2.csv")

    print("Computing 2026 team features...")
    team_stats  = compute_team_season_stats(detailed)
    recent_form = compute_recent_form(detailed)
    massey      = compute_massey_ranks(massey_raw)
    sos         = compute_sos(compact, team_stats)
    adj_eff     = compute_adjusted_efficiency(detailed, team_stats)
    coach_features = compute_all_coach_features(coaches_raw, seeds_raw)

    # Use real seeds if available; otherwise infer from Massey rankings
    real_seeds = seeds_raw[seeds_raw["Season"] == SEASON]
    if len(real_seeds) == 0:
        print(f"  No {SEASON} seeds found — inferring from Massey rankings")
        seeds = infer_seeds_from_massey(massey_raw, SEASON)
    else:
        seeds = extract_seeds(seeds_raw)

    profile = build_team_profile(team_stats, seeds, massey, recent_form, sos, adj_eff, coach_features)

    print("Building 2026 submission features...")
    sub_df = build_submission_data(stage2, profile)

    # Fill any remaining NaN diffs with 0 (equal assumption)
    fill_cols = [
        "diff_seed_num", "diff_massey_rank_mean", "diff_massey_rank_min",
        "diff_sos", "diff_adj_off_eff", "diff_adj_def_eff", "diff_adj_net_eff",
        "diff_coach_exp_years", "diff_coach_tourney_apps",
    ]
    for col in fill_cols:
        if col in sub_df.columns:
            sub_df[col] = sub_df[col].fillna(0)

    print("Loading model and predicting...")
    model = EnsemblePredictor.load(MODEL)

    has_data = sub_df[FEATURE_COLS].notna().all(axis=1)
    probs = np.full(len(sub_df), 0.5)
    if has_data.any():
        probs[has_data] = model.predict_proba(sub_df.loc[has_data, FEATURE_COLS])[:, 1]

    out_df = pd.DataFrame({"ID": sub_df["ID"], "Pred": probs.round(4)})
    out = stage2[["ID"]].merge(out_df, on="ID", how="left")
    out["Pred"] = out["Pred"].fillna(0.5)
    out.to_csv(OUT, index=False)

    print(f"  {has_data.sum():,} rows modeled, {(~has_data).sum():,} filled with 0.5")
    print(f"  Saved {len(out):,} predictions → {OUT}")


if __name__ == "__main__":
    main()
