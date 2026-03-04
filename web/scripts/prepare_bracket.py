"""
Prepare bracket.json for the web app.

Reads Kaggle data files and the model's submission.csv, builds the 2024
tournament bracket with model predictions and actual results, and writes
web/public/data/bracket.json.

Run from the project root:
    python web/scripts/prepare_bracket.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "web" / "public" / "data" / "bracket.json"

SEASON = 2024

REGION_NAMES = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}

# First-round seed matchups (lower seed vs higher seed)
FIRST_ROUND_PAIRS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]


def get_prob(prob_lookup: dict, t1: int, t2: int) -> float:
    """Return P(t1 beats t2). Handles ordering."""
    lo, hi = min(t1, t2), max(t1, t2)
    key = f"{lo}_{hi}"
    p = prob_lookup.get(key, 0.5)
    return p if t1 == lo else 1 - p


def build_game(t1: dict, t2: dict, prob_lookup: dict, actual_winners: set) -> dict:
    prob_t1 = get_prob(prob_lookup, t1["id"], t2["id"])
    predicted_winner = 1 if prob_t1 >= 0.5 else 2
    # Determine actual winner if the game occurred
    actual_winner = None
    if t1["id"] in actual_winners and t2["id"] not in actual_winners:
        actual_winner = 1
    elif t2["id"] in actual_winners and t1["id"] not in actual_winners:
        actual_winner = 2
    return {
        "t1": t1,
        "t2": t2,
        "prob": round(max(prob_t1, 1 - prob_t1), 4),  # always the favored team's prob
        "probT1": round(prob_t1, 4),
        "predictedWinner": predicted_winner,
        "actualWinner": actual_winner,
    }


def main():
    print("Loading data...")
    teams_df  = pd.read_csv(DATA / "MTeams.csv")
    seeds_df  = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
    results_df = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")
    sub_df    = pd.read_csv(DATA / "submission.csv")

    # Team name lookup
    team_names = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))

    # 2024 seeds — parse region + seed number
    s24 = seeds_df[seeds_df["Season"] == SEASON].copy()
    s24["region"] = s24["Seed"].str[0]
    s24["seed_str"] = s24["Seed"].str[1:3]          # '01', '11a' → '01', '11'
    s24["seed_num"] = s24["seed_str"].str[:2].astype(int)
    # For play-in games (11a/11b, 16a/16b) keep only one representative per seed
    s24 = s24.sort_values("Seed").drop_duplicates(subset=["region","seed_num"], keep="first")

    # Build region → seed_num → team_info
    region_teams = {}
    for _, row in s24.iterrows():
        r = row["region"]
        tid = int(row["TeamID"])
        region_teams.setdefault(r, {})[int(row["seed_num"])] = {
            "id": tid,
            "name": team_names.get(tid, f"Team {tid}"),
            "seed": int(row["seed_num"]),
            "region": r,
        }

    # Win-probability lookup (season 2024 rows in submission.csv)
    # ID format: SEASON_T1_T2 where T1 < T2
    sub24 = sub_df[sub_df["ID"].str.startswith(f"{SEASON}_")].copy()
    prob_lookup = {}
    for _, row in sub24.iterrows():
        parts = row["ID"].split("_")
        lo, hi = int(parts[1]), int(parts[2])
        prob_lookup[f"{lo}_{hi}"] = float(row["Pred"])

    print(f"  {len(prob_lookup):,} 2024 matchup predictions loaded")

    # Actual 2024 tournament winners (any team that won at least one game)
    r24 = results_df[results_df["Season"] == SEASON]
    actual_winners = set(r24["WTeamID"].astype(int))

    print(f"  {len(actual_winners)} teams won at least one 2024 tournament game")

    # Build region brackets
    regions_out = {}
    region_winners = {}  # region → team that won the region (Final Four entrant)

    for region_code in ["W", "X", "Y", "Z"]:
        rt = region_teams.get(region_code, {})
        rounds = []

        # Round 1: 8 games
        r1 = []
        for s1, s2 in FIRST_ROUND_PAIRS:
            t1 = rt.get(s1)
            t2 = rt.get(s2)
            if t1 and t2:
                r1.append(build_game(t1, t2, prob_lookup, actual_winners))
        rounds.append(r1)

        # Rounds 2–4: advance predicted winners
        current_round = r1
        for rnd in range(2, 5):   # rounds of 32, 16, 8
            next_round = []
            for i in range(0, len(current_round), 2):
                g1 = current_round[i]
                g2 = current_round[i + 1] if i + 1 < len(current_round) else None
                if g2 is None:
                    break
                w1 = g1["t1"] if g1["predictedWinner"] == 1 else g1["t2"]
                w2 = g2["t1"] if g2["predictedWinner"] == 1 else g2["t2"]
                next_round.append(build_game(w1, w2, prob_lookup, actual_winners))
            rounds.append(next_round)
            current_round = next_round

        # Region winner = winner of Elite Eight game
        if current_round:
            elite_eight = current_round[0]
            region_winners[region_code] = (
                elite_eight["t1"] if elite_eight["predictedWinner"] == 1
                else elite_eight["t2"]
            )

        regions_out[region_code] = {
            "name": REGION_NAMES[region_code],
            "rounds": rounds,
        }

    # Final Four: W vs X, Y vs Z (standard bracket pairing)
    final_four = []
    ff_pairings = [("W", "X"), ("Y", "Z")]
    ff_winners = []
    for r1_code, r2_code in ff_pairings:
        t1 = region_winners.get(r1_code)
        t2 = region_winners.get(r2_code)
        if t1 and t2:
            g = build_game(t1, t2, prob_lookup, actual_winners)
            final_four.append(g)
            ff_winners.append(g["t1"] if g["predictedWinner"] == 1 else g["t2"])

    # Championship
    championship = None
    if len(ff_winners) == 2:
        championship = build_game(ff_winners[0], ff_winners[1], prob_lookup, actual_winners)

    # Build all-pairs probability table for the matchup predictor
    # Only include 2024 tournament teams
    tourney_teams = []
    for r in region_teams.values():
        for team in r.values():
            tourney_teams.append(team)
    pair_probs = {}
    for i, ta in enumerate(tourney_teams):
        for tb in tourney_teams[i+1:]:
            lo, hi = min(ta["id"], tb["id"]), max(ta["id"], tb["id"])
            p = prob_lookup.get(f"{lo}_{hi}", 0.5)
            pair_probs[f"{lo}_{hi}"] = round(p, 4)

    output = {
        "season": SEASON,
        "teams": {str(k): v for k, v in team_names.items()},
        "tourneyTeams": tourney_teams,
        "pairProbs": pair_probs,
        "regions": regions_out,
        "finalFour": final_four,
        "championship": championship,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    print(f"\nWrote {OUT}")
    if championship:
        champ = championship["t1"] if championship["predictedWinner"] == 1 else championship["t2"]
        print(f"Predicted champion: [{champ['seed']}] {champ['name']}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
