"""
LightGBM + SE + time-series derived features experiment.

Loads train + val only (test set is NEVER loaded).
Derives temporal features on train+val concat (sorted by country, date) so that
val's early rows can use train's trailing history — this is NOT leakage because
only past observations are used.

Usage (from project root):
    python modeling/train_lightgbm_se_time_features.py

Outputs:
    outputs/models/lightgbm_se_time_features.pkl
    outputs/reports/time_feature_experiment_val.json
    outputs/reports/time_feature_experiment_val.md
    outputs/reports/time_feature_importance_top30.csv

IMPORTANT — Submission CSV format:
    Final submission MUST have exactly three columns: date,country,y_prob
    Do NOT add extra columns. This script does NOT produce a submission CSV.
    Filename format: predictions__{model_name}__byeonghyeon.csv
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import date as dt_date

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, average_precision_score

from utils import (
    TARGET_COL, DATE_COL, CATEGORICAL_COLS, LABEL_META_COLS,
    TRAIN_PATH, VAL_PATH,
    MODEL_DIR, REPORT_DIR,
    ensure_output_dirs,
    get_feature_columns,
    make_xy,
)
from evaluate import compute_pr_auc, compute_ece
from train_lightgbm_se import load_se_scores, merge_se_score

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH_TF    = os.path.join(MODEL_DIR, "lightgbm_se_time_features.pkl")
JSON_PATH        = os.path.join(REPORT_DIR, "time_feature_experiment_val.json")
MD_PATH          = os.path.join(REPORT_DIR, "time_feature_experiment_val.md")
FEAT_IMP_CSV     = os.path.join(REPORT_DIR, "time_feature_importance_top30.csv")
SE_REPORT_PATH   = os.path.join(REPORT_DIR, "lightgbm_se_diagnostic_val.json")

# ── Hyperparameters (identical to SE model for fair comparison) ───────────────
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

# ── Feature-engineering constants ─────────────────────────────────────────────
EPSILON    = 1e-6
# ratio clip: cap at 20× to prevent explosion when 14d/30d window is near-zero.
# Values beyond ±20 are treated as extreme and provide no additional rank signal.
RATIO_CLIP = 20.0
# z-score clip: ±5 standard deviations; beyond this the signal is noise
ZSCORE_CLIP = 5.0

# Rolling window for z-score baseline (days)
ROLLING_WINDOW = 90


# ── Base feature discovery ─────────────────────────────────────────────────────

def find_base_features(df):
    """
    Return lists of base feature names that have all three windows (7d, 14d, 30d).
    Only acled_* and gdelt_* prefixes are considered.
    """
    candidates = {}
    for col in df.columns:
        for suffix in ("_7d", "_14d", "_30d"):
            if col.endswith(suffix):
                prefix = col[: -len(suffix)]
                if prefix.startswith(("acled_", "gdelt_")):
                    candidates.setdefault(prefix, set()).add(suffix)

    # Keep only those that have all three windows
    base_features = sorted(
        p for p, windows in candidates.items()
        if {"_7d", "_14d", "_30d"} <= windows
    )
    return base_features


# ── Derived feature builders ──────────────────────────────────────────────────

def add_diff_features(df, base_features, new_cols):
    """
    Short-vs-medium difference.
    feat_7d - feat_14d/2  ≈ compares daily rate over last 7d vs last 14d.
    feat_7d - feat_30d/4  ≈ compares daily rate over last 7d vs last 30d.
    No time-series shift needed: both windows are pre-computed from past data only.
    """
    for base in base_features:
        col7, col14, col30 = f"{base}_7d", f"{base}_14d", f"{base}_30d"

        name_d14 = f"tf_diff14_{base}"
        df[name_d14] = df[col7] - df[col14] / 2.0
        new_cols.append(name_d14)

        name_d30 = f"tf_diff30_{base}"
        df[name_d30] = df[col7] - df[col30] / 4.0
        new_cols.append(name_d30)

    return df


def add_ratio_features(df, base_features, new_cols):
    """
    Short-vs-medium ratio.
    7d / (14d/2 + ε)  and  7d / (30d/4 + ε).
    Clipped to ±RATIO_CLIP to prevent explosion when denominator is near-zero.
    Epsilon added before division; clip applied after.
    No time-series shift needed.
    """
    for base in base_features:
        col7, col14, col30 = f"{base}_7d", f"{base}_14d", f"{base}_30d"

        name_r14 = f"tf_ratio14_{base}"
        df[name_r14] = (df[col7] / (df[col14] / 2.0 + EPSILON)).clip(
            -RATIO_CLIP, RATIO_CLIP
        )
        new_cols.append(name_r14)

        name_r30 = f"tf_ratio30_{base}"
        df[name_r30] = (df[col7] / (df[col30] / 4.0 + EPSILON)).clip(
            -RATIO_CLIP, RATIO_CLIP
        )
        new_cols.append(name_r30)

    return df


def add_zscore_features(df, base_features, new_cols):
    """
    Country-level rolling z-score over a 90-day window.

    For each country group, sorted by date:
      rolling_mean = feat_7d.shift(1).rolling(ROLLING_WINDOW, min_periods=7).mean()
      rolling_std  = feat_7d.shift(1).rolling(ROLLING_WINDOW, min_periods=7).std()
      z = (feat_7d - rolling_mean) / (rolling_std + EPSILON)

    Why shift(1)?
      shift(1) shifts the series forward by 1 row BEFORE applying the rolling window.
      This means the rolling window at time t is built from rows [t-ROLLING_WINDOW, t-1],
      explicitly excluding the current row's value.
      Without shift(1), the current row would be included in its own baseline — leakage.

    NaN fill: rows with insufficient history (< min_periods) produce NaN z-scores → kept
    as NaN and later removed if all-null, otherwise kept for LightGBM to handle natively.
    """
    df = df.sort_values([DATE_COL, "country"]).copy()

    for base in base_features:
        col7 = f"{base}_7d"
        name_z = f"tf_z90_{base}"

        def _compute_z(grp):
            # shift(1): exclude current row from rolling baseline
            shifted = grp[col7].shift(1)
            rm = shifted.rolling(ROLLING_WINDOW, min_periods=7).mean()
            rs = shifted.rolling(ROLLING_WINDOW, min_periods=7).std()
            z = (grp[col7] - rm) / (rs + EPSILON)
            return z.clip(-ZSCORE_CLIP, ZSCORE_CLIP)

        df[name_z] = df.groupby("country", group_keys=False).apply(_compute_z)
        new_cols.append(name_z)

    return df


def add_momentum_features(df, base_features, new_cols):
    """
    Momentum: current 7d value minus value 3 rows ago (per country, sorted by date).
    Uses shift(3) within each country group.

    Why shift(3)?
      When sorted by country+date, shift(3) gives the value 3 days prior for that country.
      This captures recent acceleration. No future information is used.
    """
    df = df.sort_values([DATE_COL, "country"]).copy()

    for base in base_features:
        col7 = f"{base}_7d"
        name_m = f"tf_mom3_{base}"

        def _compute_mom(grp):
            return grp[col7] - grp[col7].shift(3)

        df[name_m] = df.groupby("country", group_keys=False).apply(_compute_mom)
        new_cols.append(name_m)

    return df


def add_interaction_features(df, base_features, new_cols):
    """
    macis_se_score × derived signal features.
    Selected: fatalities z-score, event_count z-score, goldstein z-score,
              fatalities diff30, mentions diff30, tone diff30.
    These capture "SE-elevated region with unusual spike" — a joint signal.
    The z-score and diff30 features are themselves leakage-free (see above),
    so their product with macis_se_score is also leakage-free.
    """
    interaction_targets = []

    # z-score interactions (if generated)
    for base in ["acled_fatalities", "acled_event_count", "gdelt_goldstein_mean"]:
        z_col = f"tf_z90_{base}"
        if z_col in df.columns:
            interaction_targets.append(("z", base, z_col))

    # diff30 interactions
    for base in ["acled_fatalities", "gdelt_mentions_sum", "gdelt_tone_mean"]:
        d_col = f"tf_diff30_{base}"
        if d_col in df.columns:
            interaction_targets.append(("d30", base, d_col))

    for kind, base, col in interaction_targets:
        name_i = f"tf_ix_se_{kind}_{base}"
        df[name_i] = df["macis_se_score"] * df[col]
        new_cols.append(name_i)

    return df


# ── Quality filter ─────────────────────────────────────────────────────────────

def drop_bad_cols(df, new_cols, verbose=True):
    """
    Remove derived columns that are: inf/-inf, all-null, or constant.
    Returns (df, kept_cols, dropped_report).
    """
    kept, dropped = [], []
    for col in new_cols:
        s = df[col]
        if s.isin([np.inf, -np.inf]).any():
            dropped.append((col, "contains inf/-inf"))
            continue
        if s.isnull().all():
            dropped.append((col, "all-null"))
            continue
        if s.nunique(dropna=True) <= 1:
            dropped.append((col, "constant"))
            continue
        kept.append(col)

    if verbose and dropped:
        print(f"  Dropped {len(dropped)} derived features:")
        for col, reason in dropped:
            print(f"    - {col}  [{reason}]")

    df = df.drop(columns=[col for col, _ in dropped])
    return df, kept, dropped


# ── Metric helpers (same as diagnose script) ──────────────────────────────────

def p_at_top_k(y_true, y_prob, k):
    n = len(y_true)
    top_k = max(1, int(np.ceil(n * k)))
    top_idx = np.argsort(y_prob)[::-1][:top_k]
    return float(np.mean(y_true[top_idx]))


def recall_at_top_k(y_true, y_prob, k):
    n = len(y_true)
    top_k = max(1, int(np.ceil(n * k)))
    top_idx = np.argsort(y_prob)[::-1][:top_k]
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    return float(y_true[top_idx].sum() / n_pos)


def recall_at_precision_threshold(y_true, y_prob, min_precision):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    valid = precision >= min_precision
    if not valid.any():
        return 0.0
    return float(recall[valid].max())


def monthly_pr_auc(val_df, y_prob_arr):
    df = val_df[[DATE_COL, TARGET_COL]].copy()
    df["_y_prob"] = y_prob_arr
    df["_month"] = pd.to_datetime(df[DATE_COL]).dt.to_period("M")
    results = {}
    for month, grp in df.groupby("_month"):
        y_t = grp[TARGET_COL].values
        y_p = grp["_y_prob"].values
        results[str(month)] = (
            float(average_precision_score(y_t, y_p)) if y_t.sum() > 0 else None
        )
    return results


def compute_full_metrics(y_true, y_prob, val_df_for_monthly=None):
    m = {
        "pr_auc":                  compute_pr_auc(y_true, y_prob),
        "p_at_top1pct":            p_at_top_k(y_true, y_prob, 0.01),
        "p_at_top3pct":            p_at_top_k(y_true, y_prob, 0.03),
        "p_at_top5pct":            p_at_top_k(y_true, y_prob, 0.05),
        "p_at_top10pct":           p_at_top_k(y_true, y_prob, 0.10),
        "recall_at_top1pct":       recall_at_top_k(y_true, y_prob, 0.01),
        "recall_at_top3pct":       recall_at_top_k(y_true, y_prob, 0.03),
        "recall_at_top5pct":       recall_at_top_k(y_true, y_prob, 0.05),
        "recall_at_top10pct":      recall_at_top_k(y_true, y_prob, 0.10),
        "recall_at_precision_010": recall_at_precision_threshold(y_true, y_prob, 0.10),
        "recall_at_precision_020": recall_at_precision_threshold(y_true, y_prob, 0.20),
        "recall_at_precision_030": recall_at_precision_threshold(y_true, y_prob, 0.30),
        "ece":                     compute_ece(y_true, y_prob),
        "n_samples":               int(len(y_true)),
        "n_positives":             int(y_true.sum()),
        "positive_rate":           float(y_true.mean()),
    }
    if val_df_for_monthly is not None:
        m["monthly_pr_auc"] = monthly_pr_auc(val_df_for_monthly, y_prob)
    return m


# ── Markdown report ────────────────────────────────────────────────────────────

def generate_md_report(
    se_metrics,
    tf_metrics,
    feature_info,
    feat_df,
    dropped_features,
):
    today = dt_date.today().isoformat()
    L = []

    L += [
        "# LightGBM + SE + Time Features — Validation Experiment Report",
        "",
        f"Generated: {today}",
        f"Model artifact: `outputs/models/lightgbm_se_time_features.pkl`",
        f"Evaluated on: **val set only** (2024-01 ~ 2024-06)",
        "",
        "> **⚠️ 제출 CSV 형식 주의사항**",
        "> 최종 제출 파일은 반드시 `date,country,y_prob` **세 컬럼만** 유지해야 합니다.",
        "> 이 스크립트는 제출 CSV를 생성하지 않습니다.",
        "> 제출 파일명 형식: `predictions__{model_name}__byeonghyeon.csv`",
        "",
    ]

    # 1. Feature engineering overview
    fi = feature_info
    L += [
        "## 1. Feature Engineering Overview",
        "",
        "### train+val concat 후 feature 생성 이유",
        "",
        "val 초반부(2024-01)의 rolling z-score와 momentum feature는 train 말미(2023-12 등)의",
        "과거 관측값을 baseline으로 사용해야 정상적으로 계산됩니다.",
        "train과 val을 따로 처리하면 val 초반부에 NaN이 다수 발생하거나",
        "rolling baseline이 truncated되어 z-score가 왜곡됩니다.",
        "concat 후 country+date 정렬로 feature를 생성하고 split으로 재분리하면",
        "이 문제를 해결할 수 있으며, 시간상 과거 데이터만 사용하므로 leakage가 아닙니다.",
        "(test set은 concat에 포함하지 않았습니다.)",
        "",
        "### Leakage 방지 방식",
        "",
        "| Feature 유형 | Leakage 방지 방식 |",
        "|-------------|-----------------|",
        "| Difference / Ratio | 동일 row의 7d/14d/30d window는 각각 과거 데이터만 포함 — 추가 조치 불필요 |",
        "| Rolling z-score | `shift(1)` 적용: rolling window에서 현재 row 제외 (t 시점 z-score = [t-90, t-1] 기반) |",
        "| Momentum lag3 | `shift(3)` 적용: 3일 전 값과의 차이 — 미래 정보 없음 |",
        "| Interaction | z-score·diff 파생값 × macis_se_score — 모두 과거 정보 기반 |",
        "",
        "### Feature 생성 결과",
        "",
        f"| 항목 | 값 |",
        f"|------|-----|",
        f"| Base feature 수 (7d/14d/30d 모두 존재) | {fi['n_base_features']} |",
        f"| 생성된 derived feature 수 | {fi['n_derived_generated']} |",
        f"| 품질 필터로 제거된 feature 수 | {fi['n_dropped']} |",
        f"| 최종 사용 derived feature 수 | {fi['n_derived_kept']} |",
        f"| 기존 SE feature 수 (macis_se_score 포함) | {fi['n_se_features']} |",
        f"| 최종 전체 feature 수 | {fi['n_total_features']} |",
        "",
    ]

    if fi["base_feature_names"]:
        L.append("**대상 base features:**")
        L.append("")
        for name in fi["base_feature_names"]:
            L.append(f"- `{name}` (→ `{name}_7d/14d/30d`)")
        L.append("")

    if dropped_features:
        L += [
            "**제거된 derived features:**",
            "",
            "| Feature | 제거 이유 |",
            "|---------|---------|",
        ]
        for col, reason in dropped_features:
            L.append(f"| `{col}` | {reason} |")
        L.append("")

    # 2. Model comparison
    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    def delta_str(tf_v, se_v):
        if tf_v is None or se_v is None:
            return "—"
        d = tf_v - se_v
        sign = "+" if d >= 0 else ""
        marker = " ▲" if d > 0 else (" ▼" if d < 0 else "")
        return f"{sign}{d:.4f}{marker}"

    L += [
        "## 2. Model Comparison — Val Set Metrics",
        "",
        "| Metric | SE (기존) | SE+TF (이번) | Delta |",
        "|--------|----------|-------------|-------|",
    ]

    compare_keys = [
        ("pr_auc",                  "PR-AUC"),
        ("p_at_top1pct",            "P@top1%"),
        ("p_at_top3pct",            "P@top3%"),
        ("p_at_top5pct",            "P@top5%"),
        ("p_at_top10pct",           "P@top10%"),
        ("recall_at_top1pct",       "R@top1%"),
        ("recall_at_top3pct",       "R@top3%"),
        ("recall_at_top5pct",       "R@top5%"),
        ("recall_at_top10pct",      "R@top10%"),
        ("recall_at_precision_010", "R@P≥0.10"),
        ("recall_at_precision_020", "R@P≥0.20"),
        ("recall_at_precision_030", "R@P≥0.30"),
        ("ece",                     "ECE (↓ good)"),
    ]

    for key, label in compare_keys:
        se_v  = se_metrics.get(key)
        tf_v  = tf_metrics.get(key)
        is_ece = "ece" in key
        if is_ece and se_v is not None and tf_v is not None:
            d = tf_v - se_v
            sign = "+" if d >= 0 else ""
            marker = " ▼ (개선)" if d < 0 else (" ▲ (악화)" if d > 0 else "")
            delta_s = f"{sign}{d:.4f}{marker}"
        else:
            delta_s = delta_str(tf_v, se_v)
        L.append(f"| {label} | {fmt(se_v)} | {fmt(tf_v)} | {delta_s} |")

    L.append("")

    # 3. Monthly PR-AUC comparison
    L += [
        "## 3. Monthly PR-AUC",
        "",
        "| Month | SE (기존) | SE+TF (이번) | Delta |",
        "|-------|----------|-------------|-------|",
    ]
    se_monthly  = se_metrics.get("monthly_pr_auc", {})
    tf_monthly  = tf_metrics.get("monthly_pr_auc", {})
    for month in sorted(set(list(se_monthly.keys()) + list(tf_monthly.keys()))):
        se_v = se_monthly.get(month)
        tf_v = tf_monthly.get(month)
        L.append(f"| {month} | {fmt(se_v)} | {fmt(tf_v)} | {delta_str(tf_v, se_v)} |")
    L.append("")

    # 4. Feature importance top 30
    L += [
        "## 4. Feature Importance Top 30 (Gain 기준)",
        "",
        "| Rank | Feature | Importance (Gain) | Derived? |",
        "|------|---------|-------------------|---------|",
    ]
    for i, row in feat_df.iterrows():
        is_derived = "✓" if row["feature"].startswith("tf_") else ""
        L.append(
            f"| {i + 1} | {row['feature']} | {row['importance_gain']:.2f} | {is_derived} |"
        )
    L.append("")

    # 5. Final verdict
    pr_improved  = tf_metrics["pr_auc"] > se_metrics.get("pr_auc", 0)
    p5_improved  = tf_metrics["p_at_top5pct"] > se_metrics.get("p_at_top5pct", 0)
    ece_improved = tf_metrics["ece"] < se_metrics.get("ece", 1)
    improved_keys = [
        k for k, _ in compare_keys
        if se_metrics.get(k) is not None
        and tf_metrics.get(k) is not None
        and (
            (k != "ece" and tf_metrics[k] > se_metrics[k])
            or (k == "ece" and tf_metrics[k] < se_metrics[k])
        )
    ]
    degraded_keys = [
        k for k, _ in compare_keys
        if se_metrics.get(k) is not None
        and tf_metrics.get(k) is not None
        and (
            (k != "ece" and tf_metrics[k] < se_metrics[k])
            or (k == "ece" and tf_metrics[k] > se_metrics[k])
        )
    ]

    derived_in_top10 = feat_df.head(10)["feature"].str.startswith("tf_").sum()

    if pr_improved and p5_improved:
        adopt_verdict = (
            "**채택 권장**: PR-AUC와 P@top5% 모두 개선됐습니다. "
            "SE+TF 모델(`lightgbm_se_time_features.pkl`)을 다음 후보로 사용하세요."
        )
    elif pr_improved:
        adopt_verdict = (
            "**조건부 채택**: PR-AUC는 개선됐으나 P@top5%는 저하됐습니다. "
            "전반적 순위 품질을 중시하면 채택, 상위 알림 정밀도를 중시하면 기존 SE 유지."
        )
    elif p5_improved:
        adopt_verdict = (
            "**조건부 채택**: P@top5%는 개선됐으나 PR-AUC는 저하됐습니다. "
            "상위 알림 정밀도를 중시하면 채택, 전반적 순위 품질을 중시하면 기존 SE 유지."
        )
    else:
        adopt_verdict = (
            "**채택 보류**: PR-AUC와 P@top5% 모두 개선되지 않았습니다. "
            "기존 SE 모델(`lightgbm_se.pkl`)을 유지하세요."
        )

    improved_labels  = [label for k, label in compare_keys if k in improved_keys]
    degraded_labels  = [label for k, label in compare_keys if k in degraded_keys]

    L += [
        "## 5. Verdict",
        "",
        adopt_verdict,
        "",
        f"**개선된 지표 ({len(improved_labels)}개):** "
        + (", ".join(improved_labels) if improved_labels else "없음"),
        "",
        f"**악화된 지표 ({len(degraded_labels)}개):** "
        + (", ".join(degraded_labels) if degraded_labels else "없음"),
        "",
        f"**Top-10 feature 중 derived feature 포함 수:** {derived_in_top10}개",
        "",
        "---",
        "",
        "*이 리포트는 val set 전용 진단입니다. test set은 사용되지 않았습니다.*",
    ]

    return "\n".join(L)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_output_dirs()
    sep = "=" * 60

    # ── 1. Load SE scores ─────────────────────────────────────────────────────
    print(sep)
    print("Step 1 — Load SE scores")
    print(sep)
    se_df = load_se_scores()
    print(f"  full_se.parquet → {len(se_df):,} rows")

    # ── 2. Load train + val (TEST SET NOT LOADED) ─────────────────────────────
    print()
    print(sep)
    print("Step 2 — Load train + val  [test set NOT loaded]")
    print(sep)
    train_raw = pd.read_parquet(TRAIN_PATH)
    val_raw   = pd.read_parquet(VAL_PATH)
    print(f"  train raw: {len(train_raw):,} rows  "
          f"({pd.to_datetime(train_raw[DATE_COL]).min().date()} ~ "
          f"{pd.to_datetime(train_raw[DATE_COL]).max().date()})")
    print(f"  val   raw: {len(val_raw):,} rows  "
          f"({pd.to_datetime(val_raw[DATE_COL]).min().date()} ~ "
          f"{pd.to_datetime(val_raw[DATE_COL]).max().date()})")

    # ── 3. Merge SE scores ────────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 3 — Merge SE scores")
    print(sep)
    train_se = merge_se_score(train_raw, se_df, "train")
    val_se   = merge_se_score(val_raw,   se_df, "val")

    # ── 4. Concat → derive features → split back ─────────────────────────────
    print()
    print(sep)
    print("Step 4 — Concat train+val, derive time features, split back")
    print(sep)
    print("  Reason: val early rows need train's trailing history for rolling features.")
    print("  Leakage-free: only past observations used (shift/lag operations).")

    train_se["_split"] = "train"
    val_se["_split"]   = "val"
    combined = pd.concat([train_se, val_se], ignore_index=True)
    combined = combined.sort_values([DATE_COL, "country"]).reset_index(drop=True)
    print(f"  Combined: {len(combined):,} rows")

    # Discover base features
    base_features = find_base_features(combined)
    print(f"\n  Found {len(base_features)} base features with 7d/14d/30d windows:")
    for bf in base_features:
        print(f"    {bf}")

    # Generate derived features
    new_cols = []
    combined = add_diff_features(combined, base_features, new_cols)
    combined = add_ratio_features(combined, base_features, new_cols)
    combined = add_zscore_features(combined, base_features, new_cols)
    combined = add_momentum_features(combined, base_features, new_cols)
    combined = add_interaction_features(combined, base_features, new_cols)

    n_generated = len(new_cols)
    print(f"\n  Generated {n_generated} derived features")

    # Quality filter
    print()
    combined, kept_cols, dropped_features = drop_bad_cols(combined, new_cols)
    n_kept = len(kept_cols)
    print(f"  Kept {n_kept} derived features after quality filter")

    # Split back
    train = combined[combined["_split"] == "train"].drop(columns=["_split"])
    val   = combined[combined["_split"] == "val"].drop(columns=["_split"])
    print(f"\n  train after split: {len(train):,} rows")
    print(f"  val   after split: {len(val):,} rows")

    # ── 5. Feature columns ─────────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 5 — Feature summary")
    print(sep)
    feature_cols = get_feature_columns(train)
    # get_feature_columns excludes LABEL_META_COLS + DATE_COL; _split already dropped
    n_se_features = len(feature_cols) - n_kept  # approximate baseline
    print(f"  SE features (approx): {n_se_features}")
    print(f"  Derived kept        : {n_kept}")
    print(f"  Total features      : {len(feature_cols)}")

    feature_info = {
        "n_base_features":    len(base_features),
        "base_feature_names": base_features,
        "n_derived_generated": n_generated,
        "n_dropped":          len(dropped_features),
        "n_derived_kept":     n_kept,
        "n_se_features":      n_se_features,
        "n_total_features":   len(feature_cols),
    }

    # ── 6. X / y split ─────────────────────────────────────────────────────────
    X_train, y_train = make_xy(train, feature_cols)
    X_val,   y_val   = make_xy(val,   feature_cols)

    pos_train = int(y_train.sum())
    neg_train = int((y_train == 0).sum())
    scale_pos_weight = neg_train / pos_train
    print(f"\n  Train y_escalation: {pos_train:,} pos / {neg_train:,} neg  "
          f"(rate={pos_train/len(y_train):.4f})")
    print(f"  Val   y_escalation: {int(y_val.sum()):,} pos / {int((y_val==0).sum()):,} neg  "
          f"(rate={y_val.mean():.4f})")
    print(f"  scale_pos_weight   : {scale_pos_weight:.2f}")

    # ── 7. Train ─────────────────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 6 — Training")
    print(sep)

    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
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

    # ── 8. Evaluate on val ───────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 7 — Evaluate on val set")
    print(sep)
    y_prob = model.predict(X_val, num_iteration=model.best_iteration)
    tf_metrics = compute_full_metrics(y_val.values, y_prob, val_df_for_monthly=val)

    print(f"\n  {'Metric':<32} {'Value':>8}")
    print(f"  {'-'*32} {'-'*8}")
    for k, v in tf_metrics.items():
        if k not in ("monthly_pr_auc", "n_samples", "n_positives", "positive_rate"):
            print(f"  {k:<32} {v:>8.4f}")
    print()
    for month, v in tf_metrics.get("monthly_pr_auc", {}).items():
        val_str = f"{v:.4f}" if v is not None else "N/A"
        print(f"  Monthly PR-AUC {month}        {val_str:>8}")

    # ── 9. Load SE baseline metrics for comparison ───────────────────────────
    print()
    print(sep)
    print("Step 8 — Load SE baseline metrics for comparison")
    print(sep)
    if os.path.exists(SE_REPORT_PATH):
        with open(SE_REPORT_PATH, "r") as f:
            se_diag = json.load(f)
        # Flatten nested metric groups from diagnostic JSON
        se_metrics = {}
        se_metrics.update(se_diag.get("core_metrics", {}))
        for k, v in se_diag.get("p_at_top_k", {}).items():
            se_metrics[f"p_at_{k}"] = v
        for k, v in se_diag.get("recall_at_top_k", {}).items():
            se_metrics[f"recall_at_{k}"] = v
        for k, v in se_diag.get("recall_at_precision", {}).items():
            mapping = {"p_geq_010": "recall_at_precision_010",
                       "p_geq_020": "recall_at_precision_020",
                       "p_geq_030": "recall_at_precision_030"}
            se_metrics[mapping.get(k, k)] = v
        se_metrics["monthly_pr_auc"] = se_diag.get("monthly_pr_auc", {})
        print(f"  Loaded SE baseline: {SE_REPORT_PATH}")
    else:
        print(f"  WARNING: SE diagnostic not found at {SE_REPORT_PATH}")
        print("           Run diagnose_lightgbm_se.py first for comparison.")
        se_metrics = {}

    # ── 10. Feature importance ───────────────────────────────────────────────
    print()
    print(sep)
    print("Step 9 — Feature importance (gain)")
    print(sep)
    feat_imp   = model.feature_importance(importance_type="gain")
    feat_names = model.feature_name()
    feat_df = (
        pd.DataFrame({"feature": feat_names, "importance_gain": feat_imp})
        .sort_values("importance_gain", ascending=False)
        .head(30)
        .reset_index(drop=True)
    )
    print(feat_df.to_string(index=False))

    # ── 11. Save outputs ─────────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 10 — Save outputs")
    print(sep)

    # Model artifact
    artifact = {
        "model":              model,
        "feature_cols":       feature_cols,
        "country_categories": list(X_train["country"].cat.categories),
        "feature_info":       feature_info,
    }
    joblib.dump(artifact, MODEL_PATH_TF)
    print(f"  Saved model: {MODEL_PATH_TF}")

    # JSON report
    out_json = {
        "feature_info":  feature_info,
        "tf_metrics":    {k: v for k, v in tf_metrics.items()},
        "se_metrics":    {k: v for k, v in se_metrics.items()},
        "dropped_features": [{"feature": c, "reason": r} for c, r in dropped_features],
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print(f"  Saved JSON : {JSON_PATH}")

    # Feature importance CSV
    feat_df.to_csv(FEAT_IMP_CSV, index=False)
    print(f"  Saved CSV  : {FEAT_IMP_CSV}")

    # Markdown report
    md_text = generate_md_report(se_metrics, tf_metrics, feature_info, feat_df, dropped_features)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"  Saved MD   : {MD_PATH}")

    print()
    print("Done.")
    print(sep)


if __name__ == "__main__":
    main()
