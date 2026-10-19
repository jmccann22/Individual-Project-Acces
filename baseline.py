"""Non-AI baselines the machine learning models must beat."""
import pandas as pd
from sklearn.metrics import f1_score

import config


def always_kick(df: pd.DataFrame):
    """Majority-class baseline: never go for it."""
    return pd.Series(0, index=df.index)


def rule_based_go(df: pd.DataFrame,
                  max_ydstogo: int = config.RULE_MAX_YDSTOGO,
                  max_yardline_100: int = config.RULE_MAX_YARDLINE_100):
    """Simple rule: go for it on short yardage when in the opponent's half."""
    return ((df["ydstogo"] <= max_ydstogo) &
            (df["yardline_100"] <= max_yardline_100)).astype(int)


def tune_rule(train_df: pd.DataFrame):
    """Search the best (yards-to-go, field-position) cutoffs on TRAINING data only.

    This gives the ML models a stronger, fairer opponent than a hand-picked rule.
    """
    best = (-1.0, config.RULE_MAX_YDSTOGO, config.RULE_MAX_YARDLINE_100)
    for ytg in range(1, 11):
        for yl in range(20, 101, 5):
            pred = rule_based_go(train_df, ytg, yl)
            f1 = f1_score(train_df[config.TARGET], pred, zero_division=0)
            if f1 > best[0]:
                best = (f1, ytg, yl)
    return {"max_ydstogo": best[1], "max_yardline_100": best[2], "train_f1": best[0]}
