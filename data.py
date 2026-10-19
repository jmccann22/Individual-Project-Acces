"""Download, load, and clean nflverse play-by-play data for fourth-down decisions."""
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import config

try:  # parquet is smaller/faster, but needs pyarrow
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

# Only these columns are kept from the ~370 in the raw file.
NEEDED_COLS = [
    "season", "season_type", "game_id", "play_id", "posteam", "home_team",
    "posteam_type", "down", "ydstogo", "yardline_100", "qtr",
    "game_seconds_remaining", "score_differential", "play_type",
    "fourth_down_converted", "fourth_down_failed",
]
REQUIRED_COLS = ["season", "season_type", "game_id", "down", "ydstogo",
                 "yardline_100", "qtr", "game_seconds_remaining",
                 "score_differential", "play_type"]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "fourth-down-student-project"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dest)


def _read_parquet(path: Path) -> pd.DataFrame:
    cols = [c for c in NEEDED_COLS if c in pq.ParquetFile(path).schema.names]
    return pd.read_parquet(path, columns=cols)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=lambda c: c in NEEDED_COLS, low_memory=False)


def load_season(season: int, data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """Load one season of play-by-play, downloading it the first time."""
    pq_path = data_dir / f"play_by_play_{season}.parquet"
    csv_path = data_dir / f"play_by_play_{season}.csv.gz"

    if HAS_PYARROW and pq_path.exists():
        return _read_parquet(pq_path)
    if csv_path.exists():
        return _read_csv(csv_path)

    if HAS_PYARROW:
        try:
            print(f"  downloading {season} (parquet)...")
            _download(f"{config.BASE_URL}/play_by_play_{season}.parquet", pq_path)
            return _read_parquet(pq_path)
        except Exception as err:  # fall back to CSV
            print(f"  parquet failed ({err}); trying csv.gz")
    print(f"  downloading {season} (csv.gz)...")
    _download(f"{config.BASE_URL}/play_by_play_{season}.csv.gz", csv_path)
    return _read_csv(csv_path)


def load_seasons(seasons, data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    frames = []
    for s in seasons:
        print(f"Loading {s}")
        frames.append(load_season(s, data_dir))
    pbp = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED_COLS if c not in pbp.columns]
    if missing:
        raise KeyError(f"Expected columns missing from the data: {missing}")
    return pbp


def clean_fourth_downs(pbp: pd.DataFrame):
    """Keep regular-season 4th-down decisions and build the label.

    go_for_it = 1 when the offense ran or passed (an actual attempt)
    go_for_it = 0 when it punted or kicked a field goal
    QB kneels/spikes and penalty 'no_play' rows are dropped (not real decisions).
    """
    counts = {"raw_plays": int(len(pbp))}

    df = pbp[(pbp["season_type"] == "REG") & (pbp["down"] == 4)].copy()
    counts["regular_season_fourth_downs"] = int(len(df))

    df = df[df["play_type"].isin(["pass", "run", "punt", "field_goal"])].copy()
    counts["fourth_down_decisions"] = int(len(df))

    df[config.TARGET] = df["play_type"].isin(["pass", "run"]).astype(int)

    if "posteam_type" in df.columns:
        df["is_home"] = (df["posteam_type"] == "home").astype(int)
    else:
        df["is_home"] = (df["posteam"] == df["home_team"]).astype(int)

    # Conversion result (only meaningful for go-for-it plays); used by the backup model.
    conv = df["fourth_down_converted"] if "fourth_down_converted" in df.columns else np.nan
    df["converted"] = np.where(df[config.TARGET] == 1, conv, np.nan)

    before = len(df)
    df = df.dropna(subset=config.FEATURES + ["season"])
    counts["dropped_missing_values"] = int(before - len(df))
    counts["rows_kept"] = int(len(df))
    counts["go_for_it_rate"] = round(float(df[config.TARGET].mean()), 4)

    keep = ["season", "game_id", "play_id", "posteam", "play_type"] + config.FEATURES + \
           [config.TARGET, "converted"]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)
    return df, counts


def save_processed(df: pd.DataFrame) -> Path:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PROCESSED_DIR / "fourth_down_decisions.csv"
    df.to_csv(path, index=False)
    df.head(25).to_csv(config.PROCESSED_DIR / "sample_25_rows.csv", index=False)
    return path
