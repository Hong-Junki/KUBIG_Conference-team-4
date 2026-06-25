"""
LightGBM + SE score training script.

Adds macis_se_score from full_se.parquet to baseline features.
Does NOT modify any baseline files.

Usage (from project root):
    python modeling/train_lightgbm_se.py

Outputs:
    outputs/models/lightgbm_se.pkl
    outputs/reports/lightgbm_se_val_metrics.json
    outputs/reports/model_comparison_baseline_vs_se.md
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import date as dt_date
import joblib
import pandas as pd
import lightgbm as lgb

from utils import (
    TARGET_COL, DATE_COL, CATEGORICAL_COLS, LABEL_META_COLS,
    TRAIN_PATH, VAL_PATH,
    REPORT_PATH as BASELINE_REPORT_PATH,
    MODEL_DIR, PRED_DIR, REPORT_DIR,
    ensure_output_dirs,
    get_feature_columns,
    make_xy,
)
from evaluate import compute_all_metrics

# ── SE-specific paths ─────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SE_SCORE_PATH   = os.path.join(_ROOT, "input", "processed", "dataset", "full_se.parquet")
MODEL_PATH_SE   = os.path.join(MODEL_DIR, "lightgbm_se.pkl")
REPORT_PATH_SE  = os.path.join(REPORT_DIR, "lightgbm_se_val_metrics.json")
COMPARISON_PATH = os.path.join(REPORT_DIR, "model_comparison_baseline_vs_se.md")

# ── Hyperparameters (identical to baseline for fair comparison) ───────────────
RANDOM_SEED          = 42
NUM_BOOST_ROUND      = 1000
EARLY_STOPPING_ROUNDS = 50
LOG_PERIOD           = 100

LGB_PARAMS = {
    "objective":        "binary",
    "metric":           "average_precision",
    "boosting_type":    "gbdt",
    "learning_rate":    0.05,
    "num_leaves":       63,
    "min_child_samples": 20,
    "subsample":        0.8,
    "subsample_freq":   1,
    "colsample_bytree": 0.8,
    "seed":             RANDOM_SEED,
    "verbose":          -1,
}


# ── SE merge helpers ──────────────────────────────────────────────────────────

def load_se_scores():
    """Load only date, country, macis_se_score from full_se.parquet."""
    se = pd.read_parquet(SE_SCORE_PATH, columns=["date", "country", "macis_se_score"])
    return se


def merge_se_score(df, se_df, split_name):
    """
    Left-merge macis_se_score onto df by date + country.
    Validates that row count does not change after merge.
    """
    n_before = len(df)
    merged = df.merge(se_df, on=["date", "country"], how="left")
    n_after = len(merged)

    if n_before != n_after:
        raise ValueError(
            f"{split_name}: row count changed after merge "
            f"({n_before:,} → {n_after:,}). Check for duplicate keys in full_se.parquet."
        )

    n_null = merged["macis_se_score"].isnull().sum()
    status = f"null={n_null}"
    if n_null > 0:
        print(f"  WARNING: {split_name} has {n_null:,} null macis_se_score → filling with 0")
        merged["macis_se_score"] = merged["macis_se_score"].fillna(0)
    else:
        status = "null=0 ✓"

    print(f"  {split_name}: {n_after:,} rows  macis_se_score {status}")
    return merged


# ── Comparison report ─────────────────────────────────────────────────────────

def generate_comparison_report(baseline_metrics, se_metrics, n_feat_base, n_feat_se):
    def fmt(v):
        return f"{v:.4f}"

    def delta(se_v, base_v):
        d = se_v - base_v
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"

    lines = [
        "# Baseline vs SE Score — Model Comparison",
        "",
        f"Generated: {dt_date.today().isoformat()}",
        "",
        "## Feature Set",
        "",
        "| Model | Total features | macis_se_score |",
        "|-------|---------------|----------------|",
        f"| Baseline LightGBM | {n_feat_base} | No |",
        f"| LightGBM + SE     | {n_feat_se}  | Yes |",
        "",
        "## Validation Set Metrics",
        "",
        "| Metric | Baseline | + SE | Delta |",
        "|--------|----------|------|-------|",
        f"| PR-AUC            | {fmt(baseline_metrics['pr_auc'])} "
        f"| {fmt(se_metrics['pr_auc'])} "
        f"| {delta(se_metrics['pr_auc'], baseline_metrics['pr_auc'])} |",
        f"| P@top5%           | {fmt(baseline_metrics['p_at_top5pct'])} "
        f"| {fmt(se_metrics['p_at_top5pct'])} "
        f"| {delta(se_metrics['p_at_top5pct'], baseline_metrics['p_at_top5pct'])} |",
        f"| R@P≥0.10          | {fmt(baseline_metrics['recall_at_precision_010'])} "
        f"| {fmt(se_metrics['recall_at_precision_010'])} "
        f"| {delta(se_metrics['recall_at_precision_010'], baseline_metrics['recall_at_precision_010'])} |",
        f"| ECE               | {fmt(baseline_metrics['ece'])} "
        f"| {fmt(se_metrics['ece'])} "
        f"| {delta(se_metrics['ece'], baseline_metrics['ece'])} |",
        f"| Best iteration    | {baseline_metrics.get('best_iteration', '?')} "
        f"| {se_metrics.get('best_iteration', '?')} | — |",
        "",
        "## Interpretation",
        "",
    ]

    pr_improved = se_metrics["pr_auc"] > baseline_metrics["pr_auc"]
    p5_improved = se_metrics["p_at_top5pct"] > baseline_metrics["p_at_top5pct"]
    ece_improved = se_metrics["ece"] < baseline_metrics["ece"]

    if pr_improved and p5_improved:
        verdict = (
            f"macis_se_score가 **PR-AUC와 P@top5% 모두를 개선**했습니다. "
            f"SE 모델(`predictions__lightgbm_se__byeonghyeon.csv`)을 팀 제출 후보로 우선 고려하세요."
        )
    elif pr_improved:
        verdict = (
            f"macis_se_score가 **PR-AUC는 개선**했지만 P@top5%는 저하됐습니다. "
            f"운영 alert 정밀도보다 전반적 순위 품질이 중요하다면 SE 모델이 유리합니다."
        )
    elif p5_improved:
        verdict = (
            f"macis_se_score가 **P@top5%는 개선**했지만 PR-AUC는 저하됐습니다. "
            f"상위 알림 정밀도가 중요하다면 SE 모델이 유리합니다."
        )
    else:
        verdict = (
            "macis_se_score 추가가 val 성능을 개선하지 못했습니다. "
            "baseline(`predictions__lightgbm__byeonghyeon.csv`)을 팀 제출 후보로 사용하세요."
        )

    if ece_improved:
        verdict += " ECE도 개선되어 확률 calibration 품질이 향상됐습니다."
    else:
        verdict += " ECE는 개선되지 않았습니다."

    lines.append(verdict)
    lines.append("")
    lines.append("## Submission Files")
    lines.append("")
    lines.append("| Model | Prediction file |")
    lines.append("|-------|----------------|")
    lines.append("| Baseline | `outputs/predictions/predictions__lightgbm__byeonghyeon.csv` |")
    lines.append("| + SE     | `outputs/predictions/predictions__lightgbm_se__byeonghyeon.csv` |")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_output_dirs()
    sep = "=" * 60

    # ── 1. Load SE scores ─────────────────────────────────────────────────────
    print(sep)
    print("Step 1 — Load SE scores")
    print(sep)
    se_df = load_se_scores()
    print(f"  full_se.parquet → extracted {len(se_df):,} rows  "
          f"(date, country, macis_se_score)")
    print(f"  macis_se_score  null={se_df['macis_se_score'].isnull().sum()}  "
          f"min={se_df['macis_se_score'].min():.2f}  "
          f"max={se_df['macis_se_score'].max():.2f}")

    # ── 2. Load and merge (train + val only; test reserved) ───────────────────
    print()
    print(sep)
    print("Step 2 — Load and merge (test not loaded)")
    print(sep)
    train_raw = pd.read_parquet(TRAIN_PATH)
    val_raw   = pd.read_parquet(VAL_PATH)
    print(f"  train raw: {len(train_raw):,} rows")
    print(f"  val   raw: {len(val_raw):,} rows")
    print()

    train = merge_se_score(train_raw, se_df, "train")
    val   = merge_se_score(val_raw,   se_df, "val")

    # ── 3. Feature summary (auto-computed) ────────────────────────────────────
    print()
    print(sep)
    print("Step 3 — Feature summary")
    print(sep)
    feature_cols = get_feature_columns(train)   # macis_se_score auto-included
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    cat_cols     = [c for c in feature_cols if c in CATEGORICAL_COLS]

    print(f"  Total features             : {len(feature_cols)}")
    print(f"  ├─ Numeric                 : {len(numeric_cols)}  (baseline 54 + macis_se_score 1)")
    print(f"  └─ Categorical             : {len(cat_cols)}  {cat_cols}")

    # ── 4. X / y split ────────────────────────────────────────────────────────
    X_train, y_train = make_xy(train, feature_cols)
    X_val,   y_val   = make_xy(val,   feature_cols)

    pos_train = int(y_train.sum())
    neg_train = int((y_train == 0).sum())
    scale_pos_weight = neg_train / pos_train

    print()
    print(f"  Train y_escalation: {pos_train:,} pos / {neg_train:,} neg  "
          f"(rate={pos_train/len(y_train):.4f})")
    print(f"  Val   y_escalation: {int(y_val.sum()):,} pos / {int((y_val==0).sum()):,} neg  "
          f"(rate={y_val.mean():.4f})")
    print(f"  scale_pos_weight   : {scale_pos_weight:.2f}")

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 4 — Training")
    print(sep)

    dtrain = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_val, label=y_val,
        categorical_feature=cat_cols,
        reference=dtrain,
        free_raw_data=False,
    )

    params = {**LGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=True),
        lgb.log_evaluation(LOG_PERIOD),
    ]

    model = lgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )
    print(f"\nBest iteration: {model.best_iteration}")

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 5 — Validation metrics")
    print(sep)
    val_prob = model.predict(X_val, num_iteration=model.best_iteration)
    se_metrics = compute_all_metrics(y_val.values, val_prob)

    print(f"  PR-AUC              : {se_metrics['pr_auc']:.4f}")
    print(f"  P@top5%             : {se_metrics['p_at_top5pct']:.4f}")
    print(f"  R@Precision≥0.10    : {se_metrics['recall_at_precision_010']:.4f}")
    print(f"  ECE                 : {se_metrics['ece']:.4f}")
    print(f"  Positive rate       : {se_metrics['positive_rate']:.4f}")
    print(f"  n_samples           : {se_metrics['n_samples']:,}")

    # ── 7. Baseline comparison ────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 6 — Baseline vs SE comparison")
    print(sep)

    if os.path.exists(BASELINE_REPORT_PATH):
        with open(BASELINE_REPORT_PATH, "r") as f:
            baseline_metrics = json.load(f)

        print(f"  {'Metric':<28} {'Baseline':>10} {'+ SE':>10} {'Delta':>10}")
        print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10}")
        for key, label in [
            ("pr_auc",                   "PR-AUC"),
            ("p_at_top5pct",             "P@top5%"),
            ("recall_at_precision_010",  "R@P≥0.10"),
            ("ece",                      "ECE"),
        ]:
            base_v = baseline_metrics[key]
            se_v   = se_metrics[key]
            d      = se_v - base_v
            sign   = "+" if d >= 0 else ""
            print(f"  {label:<28} {base_v:>10.4f} {se_v:>10.4f} {sign}{d:>9.4f}")
    else:
        print("  baseline metrics not found — run train_lightgbm.py first for comparison")
        baseline_metrics = None

    # ── 8. Save model artifact ────────────────────────────────────────────────
    artifact = {
        "model":              model,
        "feature_cols":       feature_cols,
        "country_categories": list(X_train["country"].cat.categories),
    }
    joblib.dump(artifact, MODEL_PATH_SE)

    se_metrics_out = {**se_metrics, "best_iteration": model.best_iteration}
    with open(REPORT_PATH_SE, "w", encoding="utf-8") as f:
        json.dump(se_metrics_out, f, indent=2)

    print()
    print(f"Model saved   : {MODEL_PATH_SE}")
    print(f"Metrics saved : {REPORT_PATH_SE}")

    # ── 9. Comparison report ──────────────────────────────────────────────────
    if baseline_metrics is not None:
        n_feat_base = baseline_metrics.get("n_features", 55)   # fallback
        report_md = generate_comparison_report(
            baseline_metrics, se_metrics_out,
            n_feat_base=n_feat_base,
            n_feat_se=len(feature_cols),
        )
        with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Report saved  : {COMPARISON_PATH}")

    print(sep)


if __name__ == "__main__":
    main()
