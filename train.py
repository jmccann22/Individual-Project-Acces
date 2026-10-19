"""Train and evaluate the fourth-down decision models.

Usage:
    python train.py                       # seasons in config.py (downloads data first time)
    python train.py --first-season 2018 --last-season 2024

Outputs (all under results/ and models/):
    results/metrics.json          every number used in the report
    results/summary.md            ready-to-copy tables for the checkpoint template
    results/figures/*.png         figures to paste into the report
    models/best_model.joblib      go-for-it model chosen by cross-validation
    models/conversion_model.joblib  backup model: P(convert | go for it)
"""
import argparse
import json
import textwrap

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from baseline import always_kick, rule_based_go, tune_rule
from data import clean_fourth_downs, load_seasons, save_processed


def evaluate(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if y_prob is not None else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
            for k, v in out.items()}


def make_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000,
                               random_state=config.RANDOM_STATE)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1),
    }


def plot_go_rate(df):
    d = df.assign(ytg=df["ydstogo"].clip(upper=15))
    rate = d.groupby("ytg")[config.TARGET].mean()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(rate.index.astype(int), rate.values * 100, color="#2b6cb0")
    ax.set_xlabel("Yards to go on 4th down (15 = 15+)")
    ax.set_ylabel("Teams that went for it (%)")
    ax.set_title("Historical 4th-down go-for-it rate by distance")
    fig.tight_layout()
    path = config.FIGURES_DIR / "go_rate_by_yards_to_go.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_comparison(results):
    names = list(results)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(x - 0.2, [results[n]["accuracy"] for n in names], 0.4, label="Accuracy", color="#a0aec0")
    ax.bar(x + 0.2, [results[n]["f1"] for n in names], 0.4, label="F1 (go for it)", color="#2b6cb0")
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(n, 16) for n in names], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title("Held-out test season: baselines vs. machine learning")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = config.FIGURES_DIR / "model_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def write_summary(counts, split, tuned, results, cv, conv, importances):
    lines = ["# Results summary (copy these numbers into the checkpoint report)", ""]
    lines += ["## Data",
              f"- Seasons loaded: {split['seasons']}",
              f"- Raw plays loaded: {counts['raw_plays']:,}",
              f"- Regular-season 4th-down plays: {counts['regular_season_fourth_downs']:,}",
              f"- 4th-down decisions kept (go / punt / field goal): {counts['rows_kept']:,}",
              f"- Go-for-it rate: {counts['go_for_it_rate'] * 100:.1f}%",
              f"- Train: {split['train_n']:,} plays (seasons {split['train_seasons']}); "
              f"Test: {split['test_n']:,} plays (season {split['test_season']})", ""]
    lines += [f"## Test-season results (season {split['test_season']})",
              "| Method | Accuracy | Precision | Recall | F1 (go) | ROC AUC |",
              "|---|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {fmt(r['accuracy'])} | {fmt(r['precision'])} | "
                     f"{fmt(r['recall'])} | {fmt(r['f1'])} | {fmt(r['roc_auc'])} |")
    lines += ["", "## Cross-validation on training seasons (F1, grouped by game)"]
    for name, s in cv.items():
        lines.append(f"- {name}: {s['mean']:.3f} +/- {s['std']:.3f}")
    lines += ["", f"Best model by cross-validation: **{split['best_model']}**", ""]
    lines += ["## Backup model: conversion success (go-for-it plays only)",
              f"- Test attempts: {conv['test_n']}; actual conversion rate {conv['test_rate']:.3f}",
              f"- ROC AUC {conv['roc_auc']:.3f}; accuracy {conv['accuracy']:.3f}", ""]
    lines += ["## Random Forest feature importance"]
    for k, v in sorted(importances.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {v:.3f}")
    (config.RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-season", type=int, default=config.FIRST_SEASON)
    ap.add_argument("--last-season", type=int, default=config.LAST_SEASON)
    args = ap.parse_args()

    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(exist_ok=True)

    # ---- data ----
    seasons = list(range(args.first_season, args.last_season + 1))
    pbp = load_seasons(seasons)
    df, counts = clean_fourth_downs(pbp)
    save_processed(df)
    print("\nData counts:", json.dumps(counts, indent=2))

    test_season = args.last_season
    train_df = df[df["season"] < test_season]
    test_df = df[df["season"] == test_season]
    if train_df.empty or test_df.empty:
        raise SystemExit("Need at least two seasons (one to train on, one to test on).")
    X_tr, y_tr = train_df[config.FEATURES], train_df[config.TARGET]
    X_te, y_te = test_df[config.FEATURES], test_df[config.TARGET]
    split = {"seasons": f"{seasons[0]}-{seasons[-1]}", "train_seasons": f"{seasons[0]}-{test_season - 1}",
             "test_season": test_season, "train_n": len(train_df), "test_n": len(test_df)}

    # ---- baselines ----
    tuned = tune_rule(train_df)
    hand_name = f"Rule: 4th & <={config.RULE_MAX_YDSTOGO}, opp. half"
    tuned_name = f"Tuned rule: 4th & <={tuned['max_ydstogo']}, opp. {tuned['max_yardline_100']}"
    results = {
        "Always kick (majority)": evaluate(y_te, always_kick(test_df)),
        hand_name: evaluate(y_te, rule_based_go(test_df)),
        tuned_name: evaluate(y_te, rule_based_go(test_df, tuned["max_ydstogo"], tuned["max_yardline_100"])),
    }

    # ---- machine learning: pick the model by grouped CV on training seasons only ----
    models = make_models()
    cv_scores, cv_summary = {}, {}
    cv = GroupKFold(n_splits=5)
    for name, model in models.items():
        s = cross_val_score(model, X_tr, y_tr, groups=train_df["game_id"], cv=cv, scoring="f1")
        cv_summary[name] = {"mean": float(s.mean()), "std": float(s.std())}
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        results[name] = evaluate(y_te, (prob >= 0.5).astype(int), prob)
    best_name = max(cv_summary, key=lambda n: cv_summary[n]["mean"])
    split["best_model"] = best_name
    joblib.dump({"model": models[best_name], "name": best_name, "features": config.FEATURES},
                config.MODEL_DIR / "best_model.joblib")
    for name, model in models.items():
        joblib.dump({"model": model, "name": name, "features": config.FEATURES},
                    config.MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    rf = models["Random Forest"]
    importances = {f: float(v) for f, v in zip(config.FEATURES, rf.feature_importances_)}

    # ---- backup model: will a go-for-it attempt convert? ----
    go_tr = train_df[(train_df[config.TARGET] == 1)].dropna(subset=["converted"])
    go_te = test_df[(test_df[config.TARGET] == 1)].dropna(subset=["converted"])
    conv_model = RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                        random_state=config.RANDOM_STATE, n_jobs=-1)
    conv_model.fit(go_tr[config.FEATURES], go_tr["converted"].astype(int))
    p_conv = conv_model.predict_proba(go_te[config.FEATURES])[:, 1]
    conv = {"test_n": int(len(go_te)), "test_rate": float(go_te["converted"].mean()),
            "roc_auc": float(roc_auc_score(go_te["converted"].astype(int), p_conv)),
            "accuracy": float(accuracy_score(go_te["converted"].astype(int), (p_conv >= 0.5).astype(int)))}
    joblib.dump({"model": conv_model, "name": "Conversion RF", "features": config.FEATURES},
                config.MODEL_DIR / "conversion_model.joblib")

    # ---- save everything ----
    plot_go_rate(df)
    plot_comparison(results)
    metrics = {"counts": counts, "split": {k: (int(v) if isinstance(v, (np.integer,)) else v)
                                           for k, v in split.items()},
               "tuned_rule": tuned, "test_results": results, "cv_f1": cv_summary,
               "conversion_model": conv, "rf_feature_importance": importances}
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    write_summary(counts, split, tuned, results, cv_summary, conv, importances)

    print(f"\n=== Test season {test_season} ===")
    for name, r in results.items():
        print(f"{name:40s} acc={fmt(r['accuracy'])} f1={fmt(r['f1'])} "
              f"prec={fmt(r['precision'])} rec={fmt(r['recall'])} auc={fmt(r['roc_auc'])}")
    print(f"\nBest model by CV: {best_name}")
    print("Saved: results/summary.md, results/metrics.json, results/figures/, models/")


if __name__ == "__main__":
    main()
