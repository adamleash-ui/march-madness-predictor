"""
Download the March Machine Learning Mania 2025 dataset from Kaggle
and preview the key CSV files.

Usage:
    python src/get_data.py
"""

import zipfile
from pathlib import Path

import kaggle
import pandas as pd

COMPETITION = "march-machine-learning-mania-2025"
DATA_DIR = Path(__file__).parent.parent / "data"
FILES_TO_LOAD = [
    "MRegularSeasonCompactResults.csv",
    "MNCAATourneyCompactResults.csv",
]


def download_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading competition data: {COMPETITION}")
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(
        COMPETITION,
        path=DATA_DIR,
        quiet=False,
    )

    zip_path = DATA_DIR / f"{COMPETITION}.zip"
    if zip_path.exists():
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()
        print("Extraction complete.")


def load_and_preview(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{filename} not found in {DATA_DIR}. Run download first."
        )
    df = pd.read_csv(path)
    print(f"\n{'='*60}")
    print(f"  {filename}")
    print(f"  Shape: {df.shape}")
    print(f"{'='*60}")
    print(df.head())
    return df


if __name__ == "__main__":
    download_data()
    dfs = {f: load_and_preview(f) for f in FILES_TO_LOAD}
