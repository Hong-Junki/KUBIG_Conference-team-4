"""
Integrated retraining experiment script using the updated processed dataset.

Runs 4 experiments in a single pass and produces a validation comparison report.
Does NOT generate test predictions — run a separate prediction script after selecting
the best experiment.
Does NOT overwrite or modify any existing model/report/prediction files.

Experiment configurations
─────────────────────────
  updated_mask_included   : full 211,816-row train, acled_missing_mask kept as feature
  updated_mask_excluded   : full 211,816-row train, acled_missing_mask dropped
  updated_mask0_only      : only mask=0 rows (~182,554), acled_missing_mask dropped
  updated_2022_2023_only  : only 2022-2023 rows (~42,340), acled_missing_mask dropped

SE-score coverage note
──────────────────────
full_se.parquet covers 2022-01-01 ~ 2025-03-28 only.
For Exp-1/2, ~169,476 of 211,816 train rows have no SE score (null → filled 0).
For Exp-3,   ~140,214 of 182,554 train rows have no SE score (null → filled 0).
For Exp-4,   all 42,340 rows have real SE scores (0 nulls).

Usage (from project root):
    python modeling/train_lightgbm_se_updated.py

Outputs
───────
  outputs/models/lightgbm_se_updated_{name}.pkl              (4 model artifacts)
  outputs/reports/lightgbm_se_updated_{name}_val_metrics.json  (4 metric JSONs)
  outputs/reports/model_comparison_updated_experiments.csv
  outputs/reports/model_comparison_updated_experiments.md
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import date as dt_date

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve

from utils import (
    TARGET_COL, DATE_COL, CATEGORICAL_COLS, LABEL_META_COLS,
    MODEL_DIR, PRED_DIR, REPORT_DIR,
    ensure_output_dirs,
    make_xy,
)
from evaluate import compute_all_metrics

# ── Project root ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)

# ── Data paths ────────────────────────────────────────────────────────────────
TRAIN_PATH   = os.path.join(_ROOT, "input", "processed", "dataset", "train.parquet")
VAL_PATH     = os.path.join(_ROOT, "input", "processed", "dataset", "val.parquet")
SE_PATH      = os.path.join(_ROOT, "input", "processed", "dataset", "full_se.parquet")
OLD_SE_REPORT = os.path.join(REPORT_DIR, "lightgbm_se_val_metrics.json")

# ── Hyperparameters (identical to existing scripts for fair comparison) ───────
RANDOM_SEED           = 42
NUM_BOOST_ROUND       = 1000
EARLY_STOPPING_ROUNDS = 50
LOG_PERIOD            = 100

LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "average_precision",
    "boosting_type":     "gbdt",
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 20,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.8,
    "seed":              RANDOM_SEED,
    "verbose":           -1,
}

# ── Experiment registry ───────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "name":         "updated_mask_included",
        "label":        "Updated full train + mask included",
        "train_filter": None,
        "exclude_mask": False,
    },
    {
        "name":         "updated_mask_excluded",
        "label":        "Updated full train + mask excluded",
        "train_filter": None,
        "exclude_mask": True,
    },
    {
        "name":         "updated_mask0_only",
        "label":        "mask=0 rows only + mask excluded",
        "train_filter": "mask0",
        "exclude_mask": True,
    },
    {
        "name":         "updated_2022_2023_only",
        "label":        "2022-2023 train only + mask excluded",
        "train_filter": "2022_2023",
        "exclude_mask": True,
    },
]


# ── Additional metric functions (not modifying evaluate.py) ──────────────────

def compute_recall_at_precision_threshold(y_true, y_prob, min_precision):
    """Maximum recall achievable while keeping precision >= min_precision."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    valid = precision >= min_precision
    if not valid.any():
        return 0.0
    return float(recall[valid].max())


def compute_brier_score(y_true, y_prob):
    """Mean squared error between predicted probabilities and true labels."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def compute_extended_metrics(y_true, y_prob):
    """
    Extends compute_all_metrics() from evaluate.py with three additional metrics:
      - recall_at_precision_020  (R@P≥0.20)
      - recall_at_precision_030  (R@P≥0.30)
      - brier_score
    evaluate.py is NOT modified.
    """
    base = compute_all_metrics(y_true, y_prob)
    base["recall_at_precision_020"] = compute_recall_at_precision_threshold(
        y_true, y_prob, min_precision=0.20
    )
    base["recall_at_precision_030"] = compute_recall_at_precision_threshold(
        y_true, y_prob, min_precision=0.30
    )
    base["brier_score"] = compute_brier_score(y_true, y_prob)
    return base


# ── Feature column builder ────────────────────────────────────────────────────

def get_updated_feature_columns(df, exclude_mask):
    """
    Return feature columns, excluding label/meta/date columns.
    If exclude_mask is True, also drop acled_missing_mask.
    """
    exclude = set(LABEL_META_COLS) | {DATE_COL}
    if exclude_mask:
        exclude.add("acled_missing_mask")
    return [c for c in df.columns if c not in exclude]


# ── SE merge helper ───────────────────────────────────────────────────────────

def merge_se_score(df, se_df, split_name):
    """
    Left-merge macis_se_score onto df by date + country.
    Rows without a matching SE entry are filled with 0.
    Row count is validated to not change after merge.
    """
    n_before = len(df)
    merged = df.merge(se_df, on=["date", "country"], how="left")
    if len(merged) != n_before:
        raise ValueError(
            f"{split_name}: row count changed after SE merge "
            f"({n_before:,} → {len(merged):,}). Check for duplicate keys in full_se.parquet."
        )
    n_null = merged["macis_se_score"].isnull().sum()
    if n_null > 0:
        pct = n_null / n_before * 100
        print(f"    WARNING {split_name}: {n_null:,} / {n_before:,} rows ({pct:.1f}%) "
              f"have no SE score → filling with 0")
        merged["macis_se_score"] = merged["macis_se_score"].fillna(0.0)
    else:
        print(f"    {split_name}: SE merge complete, null=0 ✓")
    return merged


# ── Single experiment runner ──────────────────────────────────────────────────

def run_experiment(exp_cfg, train_raw, val_merged, se_df):
    """
    Run one experiment end-to-end.
    Returns a dict containing metrics and metadata.
    """
    sep = "-" * 60
    name   = exp_cfg["name"]
    label  = exp_cfg["label"]
    f_mode = exp_cfg["train_filter"]
    ex_mask = exp_cfg["exclude_mask"]

    print()
    print("=" * 60)
    print(f"EXPERIMENT: {name}")
    print(f"  {label}")
    print("=" * 60)

    # ── Step 1: Apply train filter ────────────────────────────────────────────
    print(sep)
    print("Step 1 — Apply train filter")
    if f_mode is None:
        train_filtered = train_raw.copy()
        print(f"  No filter applied  →  {len(train_filtered):,} rows")
    elif f_mode == "mask0":
        train_filtered = train_raw[train_raw["acled_missing_mask"] == 0].copy()
        print(f"  mask=0 filter  →  {len(train_filtered):,} rows "
              f"(dropped {len(train_raw) - len(train_filtered):,} mask=1 rows)")
    elif f_mode == "2022_2023":
        train_filtered = train_raw[train_raw[DATE_COL].dt.year >= 2022].copy()
        print(f"  2022+ filter   →  {len(train_filtered):,} rows "
              f"(dropped {len(train_raw) - len(train_filtered):,} pre-2022 rows)")
    else:
        raise ValueError(f"Unknown train_filter: {f_mode}")

    # ── Step 2: Merge SE scores ───────────────────────────────────────────────
    print(sep)
    print("Step 2 — Merge SE scores")
    train_se = merge_se_score(train_filtered, se_df, "train")

    # ── Step 3: Build feature list ────────────────────────────────────────────
    print(sep)
    print("Step 3 — Feature summary")
    feature_cols = get_updated_feature_columns(train_se, exclude_mask=ex_mask)
    cat_cols     = [c for c in feature_cols if c in CATEGORICAL_COLS]
    num_cols     = [c for c in feature_cols if c not in CATEGORICAL_COLS]

    print(f"  acled_missing_mask : {'excluded' if ex_mask else 'INCLUDED as feature'}")
    print(f"  macis_se_score     : included")
    print(f"  Total features     : {len(feature_cols)}")
    print(f"  ├─ Numeric         : {len(num_cols)}")
    print(f"  └─ Categorical     : {len(cat_cols)}  {cat_cols}")

    # ── Step 4: X/y split ─────────────────────────────────────────────────────
    X_train, y_train = make_xy(train_se,   feature_cols)
    X_val,   y_val   = make_xy(val_merged, feature_cols)

    pos_train = int(y_train.sum())
    neg_train = int((y_train == 0).sum())
    scale_pos_weight = neg_train / pos_train

    print()
    print(f"  Train y_escalation: {pos_train:,} pos / {neg_train:,} neg  "
          f"(rate={pos_train/len(y_train):.4f})")
    print(f"  Val   y_escalation: {int(y_val.sum()):,} pos / {int((y_val==0).sum()):,} neg  "
          f"(rate={y_val.mean():.4f})")
    print(f"  scale_pos_weight  : {scale_pos_weight:.2f}")

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 4 — Training")
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
    params    = {**LGB_PARAMS, "scale_pos_weight": scale_pos_weight}
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
    print(f"\n  Best iteration: {model.best_iteration}")

    # ── Step 6: Evaluate on val ───────────────────────────────────────────────
    print()
    print(sep)
    print("Step 5 — Validation metrics")
    val_prob = model.predict(X_val, num_iteration=model.best_iteration)
    metrics  = compute_extended_metrics(y_val.values, val_prob)

    print(f"  PR-AUC              : {metrics['pr_auc']:.4f}")
    print(f"  P@top5%             : {metrics['p_at_top5pct']:.4f}")
    print(f"  R@P≥0.10            : {metrics['recall_at_precision_010']:.4f}")
    print(f"  R@P≥0.20            : {metrics['recall_at_precision_020']:.4f}")
    print(f"  R@P≥0.30            : {metrics['recall_at_precision_030']:.4f}")
    print(f"  ECE                 : {metrics['ece']:.4f}")
    print(f"  Brier score         : {metrics['brier_score']:.4f}")

    # ── Step 7: Save model artifact ───────────────────────────────────────────
    artifact = {
        "model":              model,
        "feature_cols":       feature_cols,
        "country_categories": list(X_train["country"].cat.categories),
        "experiment":         name,
        "train_rows":         len(train_se),
        "n_features":         len(feature_cols),
    }
    model_path = os.path.join(MODEL_DIR, f"lightgbm_se_{name}.pkl")
    joblib.dump(artifact, model_path)
    print(f"\n  Model saved   : {model_path}")

    # ── Step 8: Save per-experiment metrics JSON ──────────────────────────────
    metrics_out = {
        **metrics,
        "best_iteration": model.best_iteration,
        "n_features":     len(feature_cols),
        "train_rows":     len(train_se),
        "experiment":     name,
    }
    report_path = os.path.join(REPORT_DIR, f"lightgbm_se_{name}_val_metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Metrics saved : {report_path}")

    return metrics_out


# ── Comparison report generators ─────────────────────────────────────────────

def generate_comparison_csv(results, old_metrics):
    """Return a CSV string with one row per experiment plus the old-model row."""
    METRIC_KEYS = [
        "pr_auc", "p_at_top5pct",
        "recall_at_precision_010",
        "recall_at_precision_020",
        "recall_at_precision_030",
        "ece", "brier_score",
        "best_iteration", "n_features", "train_rows",
    ]
    METRIC_LABELS = [
        "PR-AUC", "P@top5%",
        "R@P≥0.10", "R@P≥0.20", "R@P≥0.30",
        "ECE", "Brier", "best_iter", "n_feat", "train_rows",
    ]
    header = "experiment," + ",".join(METRIC_LABELS)
    rows = []

    # Old model row (R@P≥0.20/0.30 and Brier not available for old model)
    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    old_row_vals = [_fmt(old_metrics.get(k, "N/A")) for k in METRIC_KEYS]
    rows.append("old_lightgbm_se (2022-2023)," + ",".join(old_row_vals))

    for r in results:
        vals = [_fmt(r.get(k, "N/A")) for k in METRIC_KEYS]
        rows.append(r["experiment"] + "," + ",".join(vals))

    return header + "\n" + "\n".join(rows)


def generate_comparison_md(results, old_metrics):
    """
    Generate a Korean-language Markdown comparison report.
    Interpretation is derived dynamically from actual metric values.
    """
    def fmt(v, decimals=4):
        if isinstance(v, float):
            return f"{v:.{decimals}f}"
        return str(v)

    def delta_str(new_v, ref_v):
        if not isinstance(new_v, float) or not isinstance(ref_v, float):
            return "N/A"
        d = new_v - ref_v
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"

    # index results by name for easy lookup
    by_name = {r["experiment"]: r for r in results}
    exp1 = by_name.get("updated_mask_included", {})
    exp2 = by_name.get("updated_mask_excluded", {})
    exp3 = by_name.get("updated_mask0_only", {})
    exp4 = by_name.get("updated_2022_2023_only", {})

    old_pr   = old_metrics.get("pr_auc", None)
    exp2_pr  = exp2.get("pr_auc", None)
    exp4_pr  = exp4.get("pr_auc", None)
    exp1_pr  = exp1.get("pr_auc", None)
    exp3_pr  = exp3.get("pr_auc", None)

    # ── Analysis logic ────────────────────────────────────────────────────────
    # Q1: Does 2014-2023 training period help? (Exp-2 vs Exp-4)
    if exp2_pr is not None and exp4_pr is not None:
        if exp2_pr - exp4_pr > 0.005:
            q1 = (f"Exp-2(전체 기간)의 PR-AUC({fmt(exp2_pr)})가 "
                  f"Exp-4(2022-2023)의 PR-AUC({fmt(exp4_pr)})보다 "
                  f"{exp2_pr - exp4_pr:.4f} 높습니다. "
                  "**2014~2023 학습 기간 확장이 유의미한 성능 향상을 가져왔습니다.**")
        elif exp4_pr - exp2_pr > 0.005:
            q1 = (f"Exp-4(2022-2023)의 PR-AUC({fmt(exp4_pr)})가 "
                  f"Exp-2(전체 기간)의 PR-AUC({fmt(exp2_pr)})보다 높습니다. "
                  "**기간 확장이 오히려 성능에 불리하게 작용했습니다.** "
                  "2014~2021 구간의 데이터가 최근 패턴과 다소 이질적일 가능성이 있습니다.")
        else:
            q1 = (f"Exp-2({fmt(exp2_pr)})와 Exp-4({fmt(exp4_pr)})의 PR-AUC 차이가 "
                  "0.005 이내로 미미합니다. "
                  "**학습 기간 확장의 효과가 명확하지 않습니다.** "
                  "두 실험의 다른 지표(P@top5%, ECE)도 함께 참고하세요.")
    else:
        q1 = "Exp-2 또는 Exp-4 결과를 확인할 수 없습니다."

    # Q2: Should acled_missing_mask be included or excluded? (Exp-1 vs Exp-2)
    if exp1_pr is not None and exp2_pr is not None:
        if exp2_pr - exp1_pr > 0.002:
            q2 = (f"Exp-2(mask 제외, PR-AUC={fmt(exp2_pr)})가 "
                  f"Exp-1(mask 포함, PR-AUC={fmt(exp1_pr)})보다 성능이 높습니다. "
                  "**acled_missing_mask는 피처에서 제외하는 것이 유리합니다.** "
                  "val/test에서 mask가 항상 0인 분포 불일치가 모델에 부정적으로 작용한 것으로 보입니다.")
        elif exp1_pr - exp2_pr > 0.002:
            q2 = (f"Exp-1(mask 포함, PR-AUC={fmt(exp1_pr)})가 "
                  f"Exp-2(mask 제외, PR-AUC={fmt(exp2_pr)})보다 성능이 높습니다. "
                  "**acled_missing_mask를 피처로 포함하는 것이 유리합니다.** "
                  "모델이 ACLED 결측 여부를 유용한 정보로 활용하고 있음을 시사합니다.")
        else:
            q2 = (f"Exp-1({fmt(exp1_pr)})과 Exp-2({fmt(exp2_pr)})의 차이가 미미합니다. "
                  "두 설정이 거의 동등하므로, **배포 안정성 측면에서 mask 제외(Exp-2)를 권장합니다.** "
                  "val/test 분포와의 일관성이 더 높기 때문입니다.")
    else:
        q2 = "Exp-1 또는 Exp-2 결과를 확인할 수 없습니다."

    # Q3: Are mask=1 rows harmful or useful? (Exp-2 vs Exp-3)
    if exp2_pr is not None and exp3_pr is not None:
        if exp2_pr - exp3_pr > 0.002:
            q3 = (f"Exp-2(mask=1 포함, PR-AUC={fmt(exp2_pr)})가 "
                  f"Exp-3(mask=0만, PR-AUC={fmt(exp3_pr)})보다 성능이 높습니다. "
                  "**mask=1 행(ACLED 결측 구간)이 모델 학습에 유용하게 작용했습니다.** "
                  "데이터 양의 증가 효과가 노이즈를 상회한 것으로 해석됩니다.")
        elif exp3_pr - exp2_pr > 0.002:
            q3 = (f"Exp-3(mask=0만, PR-AUC={fmt(exp3_pr)})가 "
                  f"Exp-2(mask=1 포함, PR-AUC={fmt(exp2_pr)})보다 성능이 높습니다. "
                  "**mask=1 행이 피처 노이즈로 작용하여 성능에 해롭습니다.** "
                  "ACLED 파생 피처가 0으로 채워진 행들이 모델을 혼란스럽게 한 것으로 보입니다.")
        else:
            q3 = (f"Exp-2({fmt(exp2_pr)})와 Exp-3({fmt(exp3_pr)})의 차이가 미미합니다. "
                  "**mask=1 행의 영향이 중립적입니다.** "
                  "데이터 양이 많은 Exp-2(전체 행 사용)를 권장합니다.")
    else:
        q3 = "Exp-2 또는 Exp-3 결과를 확인할 수 없습니다."

    # Q4: Best candidate
    valid_results = [r for r in results if "pr_auc" in r]
    best = max(valid_results, key=lambda r: r["pr_auc"]) if valid_results else None
    if best:
        best_name = best["experiment"]
        best_pr   = best["pr_auc"]
        # Secondary: if top two are within 0.003, prefer mask_excluded for stability
        sorted_by_pr = sorted(valid_results, key=lambda r: r["pr_auc"], reverse=True)
        if (len(sorted_by_pr) >= 2
                and sorted_by_pr[0]["pr_auc"] - sorted_by_pr[1]["pr_auc"] < 0.003
                and best_name == "updated_mask_included"):
            best_name = "updated_mask_excluded"
            best      = by_name[best_name]
            best_pr   = best["pr_auc"]
            q4 = (f"PR-AUC 기준 최고값은 `updated_mask_included`이나 2위와의 차이가 0.003 미만입니다. "
                  f"배포 안정성을 고려해 **`{best_name}` (PR-AUC={fmt(best_pr)})를 후보 모델로 권장합니다.**")
        else:
            q4 = f"**`{best_name}` (PR-AUC={fmt(best_pr)})를 후보 모델로 권장합니다.**"
    else:
        q4 = "결과를 확인할 수 없습니다."

    # Q5: Calibration
    best_ece = best.get("ece", None) if best else None
    if best_ece is not None:
        if best_ece > 0.10:
            q5 = (f"후보 모델의 ECE={fmt(best_ece)}로 0.10을 초과합니다. "
                  "**캘리브레이션 재실행을 강력히 권장합니다.** "
                  "기존 `calibrate_lightgbm_se.py`를 새 모델에 맞게 실행하세요.")
        elif best_ece > 0.05:
            q5 = (f"후보 모델의 ECE={fmt(best_ece)}로 0.05~0.10 구간입니다. "
                  "**캘리브레이션 재실행을 권장합니다.** "
                  "특히 대시보드의 확률 임계값 알림 기능을 사용한다면 캘리브레이션 후 ECE 개선을 확인하세요.")
        else:
            q5 = (f"후보 모델의 ECE={fmt(best_ece)}로 0.05 이하입니다. "
                  "캘리브레이션 없이도 사용 가능한 수준이지만, "
                  "기존 캘리브레이터를 새 모델에 재적합(refit)하는 것을 권장합니다.")
    else:
        q5 = "ECE 정보를 확인할 수 없습니다."

    # ── Table ─────────────────────────────────────────────────────────────────
    header = "| 실험 | 학습 행 수 | 피처 수 | PR-AUC | P@top5% | R@P≥0.10 | R@P≥0.20 | R@P≥0.30 | ECE | Brier | Best iter |"
    divider = "|------|-----------|--------|--------|---------|---------|---------|---------|-----|-------|-----------|"

    def row(name, r, is_old=False):
        label_col  = f"**{name}**" if not is_old else f"*{name}*"
        train_rows = r.get("train_rows", "N/A")
        rows_str   = f"{train_rows:,}" if isinstance(train_rows, int) else str(train_rows)
        return (
            f"| {label_col} "
            f"| {rows_str} "
            f"| {r.get('n_features', 'N/A')} "
            f"| {fmt(r.get('pr_auc', 0))} "
            f"| {fmt(r.get('p_at_top5pct', 0))} "
            f"| {fmt(r.get('recall_at_precision_010', 0))} "
            f"| {fmt(r.get('recall_at_precision_020', 'N/A'))} "
            f"| {fmt(r.get('recall_at_precision_030', 'N/A'))} "
            f"| {fmt(r.get('ece', 0))} "
            f"| {fmt(r.get('brier_score', 'N/A'))} "
            f"| {r.get('best_iteration', 'N/A')} |"
        )

    old_row = {
        "train_rows":    42340,
        "n_features":    55,
        "pr_auc":        old_metrics.get("pr_auc", 0),
        "p_at_top5pct":  old_metrics.get("p_at_top5pct", 0),
        "recall_at_precision_010": old_metrics.get("recall_at_precision_010", 0),
        "recall_at_precision_020": "N/A",
        "recall_at_precision_030": "N/A",
        "ece":           old_metrics.get("ece", 0),
        "brier_score":   "N/A",
        "best_iteration": old_metrics.get("best_iteration", "N/A"),
    }

    table_rows = [row("old_lightgbm_se (2022-2023)", old_row, is_old=True)]
    for r in results:
        table_rows.append(row(r["experiment"], r))

    # ── Delta table (vs old model) ────────────────────────────────────────────
    delta_header  = "| 실험 | ΔPR-AUC | ΔP@top5% | ΔR@P≥0.10 | ΔECE |"
    delta_divider = "|------|---------|---------|---------|------|"
    delta_rows = []
    for r in results:
        delta_rows.append(
            f"| {r['experiment']} "
            f"| {delta_str(r.get('pr_auc'), old_metrics.get('pr_auc'))} "
            f"| {delta_str(r.get('p_at_top5pct'), old_metrics.get('p_at_top5pct'))} "
            f"| {delta_str(r.get('recall_at_precision_010'), old_metrics.get('recall_at_precision_010'))} "
            f"| {delta_str(r.get('ece'), old_metrics.get('ece'))} |"
        )

    lines = [
        "# 업데이트 데이터셋 재학습 실험 비교 보고서",
        "",
        f"생성일: {dt_date.today().isoformat()}",
        "",
        "---",
        "",
        "## 실험 설정 요약",
        "",
        "| 실험명 | 학습 기간 | mask=1 포함 | acled_missing_mask 피처 | SE null 비율 |",
        "|--------|---------|------------|------------------------|------------|",
        "| old_lightgbm_se | 2022~2023 | ✗ | ✗ | 0% |",
        "| updated_mask_included | 2014~2023 | ✓ | **포함** | ~80% |",
        "| updated_mask_excluded | 2014~2023 | ✓ | 제외 | ~80% |",
        "| updated_mask0_only | 2014~2023 | **✗** | 제외 | ~77% |",
        "| updated_2022_2023_only | 2022~2023 | ✗ | 제외 | 0% |",
        "",
        "> **SE null 비율 주의**: full_se.parquet은 2022~2025 기간만 커버합니다.",
        "> Exp-1/2/3에서 2014~2021 구간의 macis_se_score는 null → 0으로 채워집니다.",
        "> Exp-4만 SE 점수가 100% 실측값입니다.",
        "",
        "---",
        "",
        "## 검증셋 성능 비교 (val: 2024-01-01 ~ 2024-06-30)",
        "",
        header,
        divider,
    ] + table_rows + [
        "",
        "### 구버전 대비 Delta (val 기준)",
        "",
        delta_header,
        delta_divider,
    ] + delta_rows + [
        "",
        "---",
        "",
        "## 분석 및 해석",
        "",
        "### 1. 학습 기간 확장 (2014~2023 vs 2022~2023) 효과",
        "",
        q1,
        "",
        "### 2. `acled_missing_mask` 포함 vs 제외",
        "",
        q2,
        "",
        "### 3. mask=1 행(ACLED 결측 구간)의 유해성 여부",
        "",
        q3,
        "",
        "### 4. 최적 후보 모델 선정",
        "",
        q4,
        "",
        "### 5. 캘리브레이션 재실행 필요 여부",
        "",
        q5,
        "",
        "---",
        "",
        "## 다음 단계",
        "",
        "1. 위 권장 후보 모델을 확정한 후 `predict_lightgbm_se.py` 에 해당하는 updated 예측 스크립트를 실행하세요.",
        "2. 캘리브레이션이 권장되는 경우 기존 `calibrate_lightgbm_se.py`를 참조하여 새 모델에 재적합하세요.",
        "3. 대시보드 업데이트 전에 val 예측 분포와 기존 예측 분포를 비교하여 이상 없음을 확인하세요.",
        "",
        "---",
        "",
        "*이 보고서는 `modeling/train_lightgbm_se_updated.py`에 의해 자동 생성되었습니다.*",
    ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_output_dirs()
    sep = "=" * 60

    print(sep)
    print("train_lightgbm_se_updated.py — 4-experiment integrated runner")
    print(sep)

    # ── Load SE scores (shared across all experiments) ────────────────────────
    print()
    print(sep)
    print("Loading SE scores")
    print(sep)
    se_df = pd.read_parquet(SE_PATH, columns=["date", "country", "macis_se_score"])
    print(f"  full_se.parquet: {len(se_df):,} rows  "
          f"({se_df['date'].min().date()} ~ {se_df['date'].max().date()})")
    print(f"  macis_se_score null={se_df['macis_se_score'].isnull().sum()}  "
          f"min={se_df['macis_se_score'].min():.4f}  "
          f"max={se_df['macis_se_score'].max():.4f}")

    # ── Load raw train / val ──────────────────────────────────────────────────
    print()
    print(sep)
    print("Loading train and val (test NOT loaded)")
    print(sep)
    train_raw = pd.read_parquet(TRAIN_PATH)
    val_raw   = pd.read_parquet(VAL_PATH)
    print(f"  train raw : {len(train_raw):,} rows  "
          f"({train_raw[DATE_COL].min().date()} ~ {train_raw[DATE_COL].max().date()})")
    print(f"  val   raw : {len(val_raw):,} rows  "
          f"({val_raw[DATE_COL].min().date()} ~ {val_raw[DATE_COL].max().date()})")

    # Pre-merge val SE scores once (val is shared across all experiments)
    print()
    print("  Merging SE scores onto val...")
    val_merged = merge_se_score(val_raw, se_df, "val")

    # ── Load old model metrics for comparison ─────────────────────────────────
    old_metrics = {}
    if os.path.exists(OLD_SE_REPORT):
        with open(OLD_SE_REPORT, "r") as f:
            old_metrics = json.load(f)
        print()
        print(f"  Old model metrics loaded from: {OLD_SE_REPORT}")
        print(f"    PR-AUC={old_metrics.get('pr_auc', 'N/A'):.4f}  "
              f"P@top5%={old_metrics.get('p_at_top5pct', 'N/A'):.4f}  "
              f"ECE={old_metrics.get('ece', 'N/A'):.4f}")
    else:
        print()
        print(f"  WARNING: old model metrics not found at {OLD_SE_REPORT}")

    # ── Run all experiments ───────────────────────────────────────────────────
    all_results = []
    for exp_cfg in EXPERIMENTS:
        result = run_experiment(exp_cfg, train_raw, val_merged, se_df)
        all_results.append(result)

    # ── Generate comparison outputs ───────────────────────────────────────────
    print()
    print(sep)
    print("Generating comparison reports")
    print(sep)

    csv_str = generate_comparison_csv(all_results, old_metrics)
    csv_path = os.path.join(REPORT_DIR, "model_comparison_updated_experiments.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_str)
    print(f"  CSV saved : {csv_path}")

    md_str = generate_comparison_md(all_results, old_metrics)
    md_path = os.path.join(REPORT_DIR, "model_comparison_updated_experiments.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_str)
    print(f"  MD saved  : {md_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print(sep)
    print("SUMMARY — Validation PR-AUC by experiment")
    print(sep)
    if old_metrics:
        print(f"  {'old_lightgbm_se (2022-2023)':<40} PR-AUC={old_metrics.get('pr_auc', 0):.4f}  "
              f"(reference)")
    for r in sorted(all_results, key=lambda x: x.get("pr_auc", 0), reverse=True):
        delta = ""
        if old_metrics and "pr_auc" in r and "pr_auc" in old_metrics:
            d = r["pr_auc"] - old_metrics["pr_auc"]
            delta = f"  Δ{'+' if d >= 0 else ''}{d:.4f}"
        print(f"  {r['experiment']:<40} PR-AUC={r.get('pr_auc', 0):.4f}{delta}")

    best = max(all_results, key=lambda r: r.get("pr_auc", 0))
    print()
    print(f"  Best experiment : {best['experiment']}")
    print(f"  Best PR-AUC     : {best.get('pr_auc', 0):.4f}")
    print(sep)


if __name__ == "__main__":
    main()
