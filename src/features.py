"""
Feature engineering for March Madness bracket prediction.

Builds per-team season statistics and matchup feature vectors
suitable for training a binary classifier.

Matchup target: 1 if the lower TeamID wins, 0 if the higher TeamID wins
(matches the Kaggle submission format: Season_Team1_Team2, Team1 < Team2).
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# 1. Per-team regular-season stats
# ---------------------------------------------------------------------------

def compute_team_season_stats(detailed: pd.DataFrame) -> pd.DataFrame:
    """
    Derive per-team, per-season efficiency and box-score averages
    from MRegularSeasonDetailedResults.

    Returns one row per (Season, TeamID) with columns:
        win_rate, avg_score, avg_score_allowed, avg_margin,
        off_eff, def_eff, net_eff,
        fg_pct, fg3_pct, ft_pct,
        oreb_rate, dreb_rate, ast_to_ratio,
        stl_per_game, blk_per_game, num_games
    """

    def possessions(fga, fgm, fta, or_, to):
        # Simplified possession estimate
        return fga - or_ + to + 0.475 * fta

    rows = []

    for perspective, prefix_us, prefix_them in [
        ("W", "W", "L"),   # winner's perspective
        ("L", "L", "W"),   # loser's perspective
    ]:
        us = prefix_us
        them = prefix_them

        g = detailed.copy()
        g["TeamID"] = g[f"{us}TeamID"]
        g["won"] = 1 if perspective == "W" else 0
        g["score"] = g[f"{us}Score"]
        g["score_allowed"] = g[f"{them}Score"]
        g["FGM"] = g[f"{us}FGM"]
        g["FGA"] = g[f"{us}FGA"]
        g["FGM3"] = g[f"{us}FGM3"]
        g["FGA3"] = g[f"{us}FGA3"]
        g["FTM"] = g[f"{us}FTM"]
        g["FTA"] = g[f"{us}FTA"]
        g["OR"] = g[f"{us}OR"]
        g["DR"] = g[f"{us}DR"]
        g["Ast"] = g[f"{us}Ast"]
        g["TO"] = g[f"{us}TO"]
        g["Stl"] = g[f"{us}Stl"]
        g["Blk"] = g[f"{us}Blk"]
        g["opp_FGA"] = g[f"{them}FGA"]
        g["opp_FGM"] = g[f"{them}FGM"]
        g["opp_FTA"] = g[f"{them}FTA"]
        g["opp_OR"] = g[f"{them}OR"]
        g["opp_TO"] = g[f"{them}TO"]

        # Possessions
        g["poss"] = possessions(g["FGA"], g["FGM"], g["FTA"], g["OR"], g["TO"])
        g["opp_poss"] = possessions(g["opp_FGA"], g["opp_FGM"], g["opp_FTA"], g["opp_OR"], g["opp_TO"])

        # Per-100-possession efficiency
        g["off_eff"] = g["score"] / g["poss"] * 100
        g["def_eff"] = g["score_allowed"] / g["opp_poss"] * 100

        keep = ["Season", "TeamID", "won", "score", "score_allowed",
                "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
                "OR", "DR", "Ast", "TO", "Stl", "Blk", "off_eff", "def_eff", "poss"]
        rows.append(g[keep])

    combined = pd.concat(rows, ignore_index=True)

    agg = combined.groupby(["Season", "TeamID"]).agg(
        num_games=("won", "count"),
        win_rate=("won", "mean"),
        avg_score=("score", "mean"),
        avg_score_allowed=("score_allowed", "mean"),
        off_eff=("off_eff", "mean"),
        def_eff=("def_eff", "mean"),
        total_poss=("poss", "sum"),
        FGM=("FGM", "sum"),
        FGA=("FGA", "sum"),
        FGM3=("FGM3", "sum"),
        FGA3=("FGA3", "sum"),
        FTM=("FTM", "sum"),
        FTA=("FTA", "sum"),
        OR=("OR", "sum"),
        DR=("DR", "sum"),
        Ast=("Ast", "sum"),
        TO=("TO", "sum"),
        Stl=("Stl", "sum"),
        Blk=("Blk", "sum"),
    ).reset_index()

    agg["avg_margin"] = agg["avg_score"] - agg["avg_score_allowed"]
    agg["net_eff"] = agg["off_eff"] - agg["def_eff"]
    agg["tempo"] = agg["total_poss"] / agg["num_games"]   # possessions per game
    agg["fg_pct"] = agg["FGM"] / agg["FGA"]
    agg["fg3_pct"] = agg["FGM3"] / agg["FGA3"]
    agg["ft_pct"] = agg["FTM"] / agg["FTA"]
    agg["oreb_rate"] = agg["OR"] / (agg["OR"] + agg["DR"])
    agg["ast_to_ratio"] = agg["Ast"] / agg["TO"].replace(0, np.nan)
    agg["stl_per_game"] = agg["Stl"] / agg["num_games"]
    agg["blk_per_game"] = agg["Blk"] / agg["num_games"]

    drop_cols = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR",
                 "Ast", "TO", "Stl", "Blk", "total_poss"]
    return agg.drop(columns=drop_cols)


# ---------------------------------------------------------------------------
# 2. Recent form
# ---------------------------------------------------------------------------

def compute_recent_form(detailed: pd.DataFrame, n_games: int = 10) -> pd.DataFrame:
    """
    Compute per-team stats over their last N regular-season games by DayNum.

    Returns one row per (Season, TeamID) with:
        recent_win_rate  — win rate in last N games
        recent_net_eff   — net efficiency (off - def) in last N games
        trend_win_rate   — recent_win_rate minus full-season win_rate
        trend_net_eff    — recent_net_eff minus full-season net_eff
    """

    def possessions(fga, fgm, fta, or_, to):
        return fga - or_ + to + 0.475 * fta

    rows = []
    for perspective, us, them in [("W", "W", "L"), ("L", "L", "W")]:
        g = detailed.copy()
        g["TeamID"]        = g[f"{us}TeamID"]
        g["won"]           = 1 if perspective == "W" else 0
        g["score"]         = g[f"{us}Score"]
        g["score_allowed"] = g[f"{them}Score"]
        g["poss"]     = possessions(g[f"{us}FGA"],  g[f"{us}FGM"],  g[f"{us}FTA"],  g[f"{us}OR"],  g[f"{us}TO"])
        g["opp_poss"] = possessions(g[f"{them}FGA"], g[f"{them}FGM"], g[f"{them}FTA"], g[f"{them}OR"], g[f"{them}TO"])
        g["off_eff"]  = g["score"]         / g["poss"]     * 100
        g["def_eff"]  = g["score_allowed"] / g["opp_poss"] * 100
        rows.append(g[["Season", "TeamID", "DayNum", "won", "off_eff", "def_eff"]])

    combined = pd.concat(rows, ignore_index=True)

    # Keep the last N games per team-season by DayNum
    combined = combined.sort_values(["Season", "TeamID", "DayNum"])
    recent = combined.groupby(["Season", "TeamID"]).tail(n_games)

    season_agg = combined.groupby(["Season", "TeamID"]).agg(
        season_win_rate=("won", "mean"),
        season_net_eff=("off_eff", "mean"),   # proxy; def_eff subtracted below
    ).reset_index()
    season_agg["season_net_eff"] -= combined.groupby(["Season", "TeamID"])["def_eff"].mean().values

    recent_agg = recent.groupby(["Season", "TeamID"]).agg(
        recent_win_rate=("won", "mean"),
        recent_off_eff=("off_eff", "mean"),
        recent_def_eff=("def_eff", "mean"),
    ).reset_index()
    recent_agg["recent_net_eff"] = recent_agg["recent_off_eff"] - recent_agg["recent_def_eff"]

    merged = season_agg.merge(recent_agg[["Season", "TeamID", "recent_win_rate", "recent_net_eff"]],
                              on=["Season", "TeamID"], how="left")
    merged["trend_win_rate"] = merged["recent_win_rate"] - merged["season_win_rate"]
    merged["trend_net_eff"]  = merged["recent_net_eff"]  - merged["season_net_eff"]

    return merged[["Season", "TeamID", "recent_win_rate", "recent_net_eff",
                   "trend_win_rate", "trend_net_eff"]]


# ---------------------------------------------------------------------------
# 3. Tournament seeds
# ---------------------------------------------------------------------------

def extract_seeds(seeds: pd.DataFrame) -> pd.DataFrame:
    """
    Parse MNCAATourneySeeds into numeric seed (1-16) and region.
    Returns (Season, TeamID, seed_num, seed_region).
    """
    df = seeds.copy()
    df["seed_region"] = df["Seed"].str[0]          # W / X / Y / Z
    df["seed_num"] = df["Seed"].str[1:3].astype(int)
    # Play-in games have 'a'/'b' suffix — seed number is still valid
    return df[["Season", "TeamID", "seed_num", "seed_region"]]


# ---------------------------------------------------------------------------
# 3. Massey Ordinals — consensus ranking
# ---------------------------------------------------------------------------

def compute_massey_ranks(massey: pd.DataFrame, day_cutoff: int = 133) -> pd.DataFrame:
    """
    Average Massey ordinal rank across all rating systems,
    using only rankings published before Selection Sunday (day ~133).

    Lower rank = better team.
    Returns (Season, TeamID, massey_rank_mean, massey_rank_min).
    """
    pre = massey[massey["RankingDayNum"] <= day_cutoff]
    agg = pre.groupby(["Season", "TeamID"])["OrdinalRank"].agg(
        massey_rank_mean="mean",
        massey_rank_min="min",
    ).reset_index()
    return agg


# ---------------------------------------------------------------------------
# 4. Build matchup feature rows
# ---------------------------------------------------------------------------

DIFF_STATS = [
    "win_rate", "avg_score", "avg_score_allowed", "avg_margin",
    "off_eff", "def_eff", "net_eff",
    "tempo",
    "recent_win_rate", "recent_net_eff", "trend_win_rate", "trend_net_eff",
    "fg_pct", "fg3_pct", "ft_pct",
    "oreb_rate", "ast_to_ratio", "stl_per_game", "blk_per_game",
    "seed_num", "massey_rank_mean", "massey_rank_min",
]


def build_team_profile(
    team_stats: pd.DataFrame,
    seeds: pd.DataFrame,
    massey: pd.DataFrame,
    recent_form: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all per-team features into a single (Season, TeamID) table."""
    profile = team_stats.merge(seeds, on=["Season", "TeamID"], how="left")
    profile = profile.merge(massey, on=["Season", "TeamID"], how="left")
    profile = profile.merge(recent_form, on=["Season", "TeamID"], how="left")
    return profile


def make_matchup_row(t1: pd.Series, t2: pd.Series) -> dict:
    """
    Compute stat differentials: team1 - team2.
    Also include raw seed values for both teams.
    """
    row = {}
    for col in DIFF_STATS:
        v1 = t1.get(col, np.nan)
        v2 = t2.get(col, np.nan)
        row[f"diff_{col}"] = v1 - v2
    return row


def build_training_data(
    tourney: pd.DataFrame,
    profile: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per tournament game with matchup feature differentials.

    Target (label):
        1  →  lower TeamID won
        0  →  higher TeamID won
    This matches the Kaggle submission convention.
    """
    profile_idx = profile.set_index(["Season", "TeamID"])
    feature_rows = []
    labels = []

    for _, game in tourney.iterrows():
        season = game["Season"]
        w, loser = game["WTeamID"], game["LTeamID"]
        t1, t2 = (w, loser) if w < loser else (loser, w)   # t1 always the lower ID
        label = 1 if w < loser else 0                       # 1 if lower-ID team won

        try:
            p1 = profile_idx.loc[(season, t1)]
            p2 = profile_idx.loc[(season, t2)]
        except KeyError:
            continue  # skip if profile missing (e.g. pre-2003 detailed data)

        row = make_matchup_row(p1, p2)
        row["Season"] = season
        row["T1"] = t1
        row["T2"] = t2
        feature_rows.append(row)
        labels.append(label)

    df = pd.DataFrame(feature_rows)
    df["label"] = labels
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Build submission rows (all possible matchups for a season)
# ---------------------------------------------------------------------------

def build_submission_data(
    submission: pd.DataFrame,
    profile: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build feature rows for every matchup in the Kaggle submission file.
    Returns DataFrame aligned with submission ID order.
    """
    profile_idx = profile.set_index(["Season", "TeamID"])
    feature_rows = []
    ids = []

    for _, row in submission.iterrows():
        mid = row["ID"]
        season, t1, t2 = map(int, mid.split("_"))

        try:
            p1 = profile_idx.loc[(season, t1)]
            p2 = profile_idx.loc[(season, t2)]
        except KeyError:
            feature_rows.append({f"diff_{s}": np.nan for s in DIFF_STATS})
            ids.append(mid)
            continue

        feat = make_matchup_row(p1, p2)
        feat["Season"] = season
        feat["T1"] = t1
        feat["T2"] = t2
        feature_rows.append(feat)
        ids.append(mid)

    df = pd.DataFrame(feature_rows)
    df.insert(0, "ID", ids)
    return df


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------

def build_all_features(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full feature pipeline.

    Returns:
        train_df  — matchup rows for past tournament games (with 'label')
        sub_df    — matchup rows for submission pairs
    """
    if verbose:
        print("Loading raw data...")
    detailed = pd.read_csv(DATA_DIR / "MRegularSeasonDetailedResults.csv")
    tourney  = pd.read_csv(DATA_DIR / "MNCAATourneyCompactResults.csv")
    seeds_raw = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    massey_raw = pd.read_csv(DATA_DIR / "MMasseyOrdinals.csv")
    submission = pd.read_csv(DATA_DIR / "SampleSubmissionStage1.csv")

    if verbose:
        print("Computing team season stats...")
    team_stats = compute_team_season_stats(detailed)

    if verbose:
        print("Computing recent form (last 10 games)...")
    recent_form = compute_recent_form(detailed)

    if verbose:
        print("Extracting seeds...")
    seeds = extract_seeds(seeds_raw)

    if verbose:
        print("Computing Massey rankings...")
    massey = compute_massey_ranks(massey_raw)

    if verbose:
        print("Building team profiles...")
    profile = build_team_profile(team_stats, seeds, massey, recent_form)

    if verbose:
        print("Building training matchups...")
    train_df = build_training_data(tourney, profile)

    if verbose:
        print("Building submission matchups...")
    sub_df = build_submission_data(submission, profile)

    if verbose:
        diff_cols = [c for c in train_df.columns if c.startswith("diff_")]
        print(f"\nTraining set:   {train_df.shape[0]} games x {len(diff_cols)} features")
        print(f"Submission set: {sub_df.shape[0]} matchups")
        print("\nFeature columns:\n  " + "\n  ".join(diff_cols))
        print(f"\nLabel balance:\n{train_df['label'].value_counts()}")

    return train_df, sub_df


if __name__ == "__main__":
    train_df, sub_df = build_all_features()
    train_df.to_csv(DATA_DIR / "train_features.csv", index=False)
    sub_df.to_csv(DATA_DIR / "submission_features.csv", index=False)
    print("\nSaved to data/train_features.csv and data/submission_features.csv")
