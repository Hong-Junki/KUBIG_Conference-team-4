"""
LightGBM baseline training script.

Usage (from project root):
    python modeling/train_lightgbm.py

Outputs:
    outputs/models/lightgbm_baseline.pkl       — model artifact
    outputs/reports/lightgbm_val_metrics.json  — validation metrics
"""

import sys
import os

# Ensure this file's directory is on sys.path so sibling imports work.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import joblib
import pandas as pd
import lightgbm as lgb

from utils import (
    TARGET_COL, DATE_COL, CATEGORICAL_COLS, LABEL_META_COLS,
    TRAIN_PATH, VAL_PATH,
    MODEL_PATH, REPORT_PATH,
    ensure_output_dirs,
    get_feature_columns,
    make_xy,
)
from evaluate import compute_all_metrics

# ── Hyperparameters ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50
LOG_PERIOD = 100

LGB_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",   # PR-AUC; requires lightgbm >= 3.3.0
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "seed": RANDOM_SEED,
    "verbose": -1,
}


def main():
    ensure_output_dirs()
    sep = "=" * 60

    # ── 1. Load train and val (test is NOT loaded here) ───────────────────────
    print(sep)
    print("Step 1 — Load data")
    print(sep)
    print("Loading train and val sets (test is reserved for prediction only)...")
    train = pd.read_parquet(TRAIN_PATH)
    val = pd.read_parquet(VAL_PATH)
    print(f"  train : {train.shape[0]:>6,} rows  "
          f"({train[DATE_COL].min().date()} ~ {train[DATE_COL].max().date()})")
    print(f"  val   : {val.shape[0]:>6,} rows  "
          f"({val[DATE_COL].min().date()} ~ {val[DATE_COL].max().date()})")

    # ── 2. Feature columns (auto-computed, never hardcoded) ───────────────────
    print()
    print(sep)
    print("Step 2 — Feature summary")
    print(sep)
    feature_cols = get_feature_columns(train)
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_COLS]

    print(f"  Excluded (label/meta/date) : {len(LABEL_META_COLS) + 1} cols  "
          f"({', '.join(LABEL_META_COLS + [DATE_COL])})")
    print(f"  Total features             : {len(feature_cols)}")
    print(f"  ├─ Numeric                 : {len(numeric_cols)}")
    print(f"  └─ Categorical             : {len(cat_cols)}  {cat_cols}")

    # ── 3. X / y split ────────────────────────────────────────────────────────
    X_train, y_train = make_xy(train, feature_cols)
    X_val, y_val = make_xy(val, feature_cols)

    pos_train = int(y_train.sum())
    neg_train = int((y_train == 0).sum())
    pos_val = int(y_val.sum())
    neg_val = int((y_val == 0).sum())
    scale_pos_weight = neg_train / pos_train

    print()
    print(f"  Train y_escalation: {pos_train:,} pos / {neg_train:,} neg  "
          f"(rate={pos_train / len(y_train):.4f})")
    print(f"  Val   y_escalation: {pos_val:,} pos / {neg_val:,} neg  "
          f"(rate={pos_val / len(y_val):.4f})")
    print(f"  scale_pos_weight   : {scale_pos_weight:.2f}")

    # ── 4. LightGBM Datasets ──────────────────────────────────────────────────
    # reference=dtrain ensures val uses the same binning as train.
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

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 3 — Training")
    print(sep)
    params = {**LGB_PARAMS, "scale_pos_weight": scale_pos_weight}

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=True),
        lgb.log_evaluation(LOG_PERIOD),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )
    print(f"\nBest iteration: {model.best_iteration}")

    # ── 6. Evaluate on val ───────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 4 — Validation metrics")
    print(sep)
    val_prob = model.predict(X_val, num_iteration=model.best_iteration)
    metrics = compute_all_metrics(y_val.values, val_prob)

    print(f"  PR-AUC              : {metrics['pr_auc']:.4f}")
    print(f"  P@top5%             : {metrics['p_at_top5pct']:.4f}")
    print(f"  R@Precision≥0.10    : {metrics['recall_at_precision_010']:.4f}")
    print(f"  ECE                 : {metrics['ece']:.4f}")
    print(f"  Positive rate       : {metrics['positive_rate']:.4f}")
    print(f"  n_samples           : {metrics['n_samples']:,}")

    # ── 7. Save model artifact ────────────────────────────────────────────────
    # Store feature_cols and country categories alongside the model so that
    # predict_lightgbm.py can reconstruct exactly the same feature setup
    # without re-loading training data.
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "country_categories": list(X_train["country"].cat.categories),
    }
    joblib.dump(artifact, MODEL_PATH)
    print()
    print(f"Model saved   : {MODEL_PATH}")

    # ── 8. Save metrics ───────────────────────────────────────────────────────
    metrics_out = {**metrics, "best_iteration": model.best_iteration}
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metrics saved : {REPORT_PATH}")
    print(sep)


if __name__ == "__main__":
    main()
