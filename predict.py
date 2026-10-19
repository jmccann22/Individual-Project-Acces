"""Enter a 4th-down situation, get a recommendation.

Examples:
    python predict.py                     # runs a built-in demo situation
    python predict.py --yardline 38 --ydstogo 2 --qtr 4 --clock 5:30 --score-diff -3 --home

--yardline is measured from the OPPONENT's end zone: 38 means the opponent's 38-yard line,
70 means your own 30. --score-diff is your score minus the opponent's.
"""
import argparse

import joblib
import pandas as pd

import config
from baseline import rule_based_go


def load_bundle(name: str = "best_model"):
    path = config.MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run `python train.py` first.")
    return joblib.load(path)


def game_seconds_remaining(qtr: int, minutes: int, seconds: int) -> int:
    clock = minutes * 60 + seconds
    return (4 - qtr) * 900 + clock if qtr <= 4 else clock


def describe_spot(yardline_100: int) -> str:
    if yardline_100 == 50:
        return "midfield"
    return f"opp {yardline_100}" if yardline_100 < 50 else f"own {100 - yardline_100}"


def recommend(yardline_100, ydstogo, qtr, minutes, seconds, score_differential, is_home,
              decision_bundle=None, conversion_bundle=None):
    if not 1 <= yardline_100 <= 99:
        raise ValueError("yardline must be between 1 and 99 (yards from the opponent's end zone).")
    if not 1 <= ydstogo <= yardline_100:
        raise ValueError("yards to go must be at least 1 and no more than the yardline.")
    if qtr not in (1, 2, 3, 4, 5):
        raise ValueError("quarter must be 1-4 (5 = overtime).")
    if not (0 <= seconds <= 59 and 0 <= minutes <= 15):
        raise ValueError("clock must be between 0:00 and 15:00.")

    decision_bundle = decision_bundle or load_bundle("best_model")
    row = pd.DataFrame([{
        "yardline_100": yardline_100, "ydstogo": ydstogo, "qtr": qtr,
        "game_seconds_remaining": game_seconds_remaining(qtr, minutes, seconds),
        "score_differential": score_differential, "is_home": int(bool(is_home)),
    }])[decision_bundle["features"]]

    p_go = float(decision_bundle["model"].predict_proba(row)[0, 1])
    out = {
        "situation": (f"4th & {ydstogo} at {describe_spot(yardline_100)} | Q{qtr} {minutes}:{seconds:02d} | "
                      f"score diff {score_differential:+d} | {'home' if is_home else 'away'}"),
        "model": decision_bundle["name"],
        "p_go": p_go,
        "recommendation": "Go for it" if p_go >= 0.5 else "Punt / kick",
        "rule_recommendation": "Go for it" if int(rule_based_go(row).iloc[0]) else "Punt / kick",
        "p_convert": None,
    }
    if conversion_bundle is None:
        try:
            conversion_bundle = load_bundle("conversion_model")
        except FileNotFoundError:
            pass
    if conversion_bundle is not None:
        out["p_convert"] = float(conversion_bundle["model"].predict_proba(row[conversion_bundle["features"]])[0, 1])
    return out


def print_result(r):
    print(f"\nSituation: {r['situation']}")
    print(f"Model ({r['model']}) recommends: {r['recommendation'].upper()}   [P(go for it) = {r['p_go']:.2f}]")
    if r["p_convert"] is not None:
        print(f"Estimated chance of converting if they go: {r['p_convert']:.2f}")
    print(f"Simple-rule baseline says: {r['rule_recommendation']}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yardline", type=int, default=38, help="yards from opponent end zone (default 38)")
    ap.add_argument("--ydstogo", type=int, default=2, help="yards to go (default 2)")
    ap.add_argument("--qtr", type=int, default=4, help="quarter 1-4, 5=OT (default 4)")
    ap.add_argument("--clock", default="5:30", help="clock in the quarter as M:SS (default 5:30)")
    ap.add_argument("--score-diff", type=int, default=-3, help="your score minus opponent (default -3)")
    ap.add_argument("--home", action="store_true", help="offense is the home team")
    a = ap.parse_args()
    m, s = (int(x) for x in a.clock.split(":"))
    try:
        print_result(recommend(a.yardline, a.ydstogo, a.qtr, m, s, a.score_diff, a.home))
    except (ValueError, FileNotFoundError) as err:
        raise SystemExit(f"Error: {err}")


if __name__ == "__main__":
    main()
