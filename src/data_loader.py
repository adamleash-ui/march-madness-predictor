import requests
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def fetch_team_stats(url: str, save_as: str) -> pd.DataFrame:
    """Fetch team stats from a URL and cache locally."""
    dest = DATA_DIR / save_as
    if dest.exists():
        print(f"Loading cached data from {dest}")
        return pd.read_csv(dest)

    print(f"Fetching data from {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    return pd.read_csv(dest)


def build_matchup_features(
    team1: pd.Series, team2: pd.Series, stat_cols: list[str]
) -> pd.Series:
    """Compute stat differentials between two teams for model input."""
    return team1[stat_cols] - team2[stat_cols]
