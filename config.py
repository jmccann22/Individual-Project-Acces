"""Shared settings for the NFL fourth-down decision project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"            # downloaded nflverse files are cached here
PROCESSED_DIR = ROOT / "data" / "processed" # cleaned fourth-down table
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# nflverse play-by-play release files (public, no login needed)
BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp"

# Seasons to use. The LAST season is held out as the test set (time-based split),
# so the model is always evaluated on games it has never seen.
FIRST_SEASON = 2019
LAST_SEASON = 2024

# Model inputs, taken from the approved proposal:
#   field position, yards to go, quarter, time left, score difference, home/away
FEATURES = [
    "yardline_100",            # yards from the opponent's end zone (35 = opponent's 35)
    "ydstogo",                 # yards needed for a first down
    "qtr",                     # 1-4, 5 = overtime
    "game_seconds_remaining",  # seconds left in the game
    "score_differential",      # offense score minus defense score
    "is_home",                 # 1 if the offense is the home team
]
TARGET = "go_for_it"           # 1 = went for it (run/pass), 0 = kicked (punt/field goal)

# Simple non-AI baseline: "go for it on short yardage when in the opponent's half"
RULE_MAX_YDSTOGO = 2
RULE_MAX_YARDLINE_100 = 50

RANDOM_STATE = 42
