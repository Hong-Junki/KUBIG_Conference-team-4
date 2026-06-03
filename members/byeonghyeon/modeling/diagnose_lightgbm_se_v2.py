"""
Validation diagnostic report for LightGBM + SE model — Version 2.

Loads outputs/models/lightgbm_se.pkl and produces extended diagnostics
on the val set ONLY. Train and test parquets are loaded for SE-merge
row-count verification but NO predictions are generated from them.

Usage (from project root):
    python modeling/diagnose_lightgbm_se_v2.py

Outputs
-------
outputs/predictions/val_predictions__lightgbm_se__byeonghyeon.csv
outputs/reports/diagnostics/country_diagnostics.csv
outputs/reports/diagnostics/monthly_diagnostics.csv
outputs/reports/diagnostics/top5pct_alerts.csv
outputs/reports/diagnostics/false_positives.csv
outputs/reports/diagnostics/false_negatives_outside_top5pct.csv
outputs/reports/diagnostics/false_negatives_lowest30.csv
outputs/reports/diagnostics/feature_importance_top30.csv
outputs/reports/diagnosis_lightgbm_se.md
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import date as dt_date
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score

from utils import (
    TARGET_COL, DATE_COL,
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    MODEL_DIR, PRED_DIR, REPORT_DIR,
    ensure_output_dirs,
)
from evaluate import compute_pr_auc, compute_ece, compute_recall_at_precision
from train_lightgbm_se import load_se_scores, merge_se_score

# ── Output paths ───────────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join(MODEL_DIR, "lightgbm_se.pkl")
DIAG_DIR        = os.path.join(REPORT_DIR, "diagnostics")
VAL_PRED_PATH   = os.path.join(PRED_DIR,  "val_predictions__lightgbm_se__byeonghyeon.csv")
MD_REPORT_PATH  = os.path.join(REPORT_DIR, "diagnosis_lightgbm_se.md")

COUNTRY_CSV    = os.path.join(DIAG_DIR, "country_diagnostics.csv")
MONTHLY_CSV    = os.path.join(DIAG_DIR, "monthly_diagnostics.csv")
TOP5_CSV       = os.path.join(DIAG_DIR, "top5pct_alerts.csv")
FP_CSV         = os.path.join(DIAG_DIR, "false_positives.csv")
FN_OUT5_CSV    = os.path.join(DIAG_DIR, "false_negatives_outside_top5pct.csv")
FN_LOW30_CSV   = os.path.join(DIAG_DIR, "false_negatives_lowest30.csv")
FEAT_IMP_CSV   = os.path.join(DIAG_DIR, "feature_importance_top30.csv")


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _p_at_k(y_true, y_prob, k):
    n = len(y_true)
    top_k = max(1, int(np.ceil(n * k)))
    top_idx = np.argsort(y_prob)[::-1][:top_k]
    return float(np.mean(y_true[top_idx]))


def _r_at_k(y_true, y_prob, k):
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    n = len(y_true)
    top_k = max(1, int(np.ceil(n * k)))
    top_idx = np.argsort(y_prob)[::-1][:top_k]
    return float(y_true[top_idx].sum() / n_pos)


def _top5_mask(y_prob):
    n = len(y_prob)
    top_k = max(1, int(np.ceil(n * 0.05)))
    mask = np.zeros(n, dtype=bool)
    mask[np.argsort(y_prob)[::-1][:top_k]] = True
    return mask


def _fmt_date(val):
    try:
        return str(pd.to_datetime(val).date())
    except Exception:
        return str(val)


# ── Country diagnostics ────────────────────────────────────────────────────────

def compute_country_diagnostics(val_df, y_prob_arr):
    df = val_df[[DATE_COL, "country", TARGET_COL]].copy().reset_index(drop=True)
    df["_y_prob"] = y_prob_arr

    rows = []
    for country, grp in df.groupby("country"):
        y_t = grp[TARGET_COL].values
        y_p = grp["_y_prob"].values
        n = len(grp)
        n_pos = int(y_t.sum())

        if n_pos > 0 and n_pos < n:
            pr_auc = round(float(average_precision_score(y_t, y_p)), 4)
        else:
            pr_auc = None

        top_k = max(1, int(np.ceil(n * 0.05)))
        top_idx = np.argsort(y_p)[::-1][:top_k]
        p_top5 = round(float(y_t[top_idx].mean()), 4)

        rows.append({
            "country":       country,
            "n_samples":     n,
            "n_positives":   n_pos,
            "positive_rate": round(float(y_t.mean()), 4),
            "pr_auc":        pr_auc,
            "p_at_top5pct":  p_top5,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("pr_auc", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


# ── Monthly diagnostics ────────────────────────────────────────────────────────

def compute_monthly_diagnostics(val_df, y_prob_arr):
    df = val_df[[DATE_COL, TARGET_COL]].copy().reset_index(drop=True)
    df["_y_prob"] = y_prob_arr
    df["_month"] = pd.to_datetime(df[DATE_COL]).dt.to_period("M")

    rows = []
    for month, grp in df.groupby("_month"):
        y_t = grp[TARGET_COL].values
        y_p = grp["_y_prob"].values
        n = len(grp)
        n_pos = int(y_t.sum())

        pr_auc = None
        if n_pos > 0 and n_pos < n:
            pr_auc = round(float(average_precision_score(y_t, y_p)), 4)

        top_k = max(1, int(np.ceil(n * 0.05)))
        top_idx = np.argsort(y_p)[::-1][:top_k]
        p_top5 = round(float(y_t[top_idx].mean()), 4) if n_pos > 0 else 0.0

        rows.append({
            "month":         str(month),
            "n_samples":     n,
            "n_positives":   n_pos,
            "positive_rate": round(float(y_t.mean()), 4),
            "pr_auc":        pr_auc,
            "p_at_top5pct":  p_top5,
        })

    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


# ── Alert / FP / FN builders ───────────────────────────────────────────────────

def build_top5_alerts(val_df, y_prob_arr):
    df = val_df[[DATE_COL, "country", TARGET_COL]].copy().reset_index(drop=True)
    df["y_prob"] = y_prob_arr
    df["_in_top5"] = _top5_mask(y_prob_arr)
    alerts = df[df["_in_top5"]].drop(columns=["_in_top5"]).copy()
    alerts["alert_type"] = np.where(alerts[TARGET_COL] == 1, "TP", "FP")
    return alerts.sort_values("y_prob", ascending=False).reset_index(drop=True)


def build_false_positives(val_df, y_prob_arr):
    df = val_df[[DATE_COL, "country", TARGET_COL]].copy().reset_index(drop=True)
    df["y_prob"] = y_prob_arr
    df["_in_top5"] = _top5_mask(y_prob_arr)
    fp = df[df["_in_top5"] & (df[TARGET_COL] == 0)].drop(columns=["_in_top5"])
    return fp.sort_values("y_prob", ascending=False).reset_index(drop=True)


def build_false_negatives(val_df, y_prob_arr):
    """
    Returns two DataFrames:
    - fn_outside: all positives not in the top 5% alert set
    - fn_lowest30: 30 lowest-scored positives, sorted by y_prob ascending
    """
    df = val_df[[DATE_COL, "country", TARGET_COL]].copy().reset_index(drop=True)
    df["y_prob"] = y_prob_arr
    df["_in_top5"] = _top5_mask(y_prob_arr)

    positives = df[df[TARGET_COL] == 1].copy()
    fn_outside = (
        positives[~positives["_in_top5"]]
        .drop(columns=["_in_top5"])
        .sort_values("y_prob", ascending=False)
        .reset_index(drop=True)
    )
    fn_lowest30 = (
        positives
        .drop(columns=["_in_top5"])
        .sort_values("y_prob", ascending=True)
        .head(30)
        .reset_index(drop=True)
    )
    return fn_outside, fn_lowest30


# ── Markdown report (Korean) ───────────────────────────────────────────────────

def generate_md_report(overview, core, monthly_df, country_df,
                       top5_df, fp_df, fn_out5_df, fn_low30_df, feat_df):
    today = dt_date.today().isoformat()
    n_alerts = len(top5_df)
    n_tp = int((top5_df["alert_type"] == "TP").sum())
    n_fp = int((top5_df["alert_type"] == "FP").sum())
    n_fn_out5 = len(fn_out5_df)

    L = []

    # Header
    L += [
        "# LightGBM + SE 모델 — 검증 진단 리포트 (v2)",
        "",
        f"생성일: {today}",
        f"모델 아티팩트: `outputs/models/lightgbm_se.pkl`",
        f"평가 데이터: **val 세트 전용** (test 세트는 모델 선택 및 예측에 미사용)",
        "",
    ]

    # 1. 개요
    L += [
        "## 1. 개요",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
        f"| 샘플 수 | {overview['n_samples']:,} |",
        f"| 양성(분쟁 상승) 수 | {overview['n_positives']:,} |",
        f"| 양성 비율 | {overview['positive_rate']:.4f} |",
        f"| 피처 수 | {overview['n_features']} |",
        f"| 최적 반복 횟수 (best iteration) | {overview['best_iteration']} |",
        "",
    ]

    # 2. 핵심 메트릭
    L += [
        "## 2. 핵심 메트릭",
        "",
        "| 메트릭 | 값 | 설명 |",
        "|--------|-----|------|",
        f"| PR-AUC | {core['pr_auc']:.4f} | 전체 Precision-Recall 곡선 아래 면적 |",
        f"| P@top5% | {core['p_at_top5pct']:.4f} | 상위 5% 알림의 정밀도 |",
        f"| R@P≥0.10 | {core['r_at_p010']:.4f} | Precision ≥ 0.10 구간에서의 최대 재현율 |",
        f"| R@P≥0.20 | {core['r_at_p020']:.4f} | Precision ≥ 0.20 구간에서의 최대 재현율 |",
        f"| R@P≥0.30 | {core['r_at_p030']:.4f} | Precision ≥ 0.30 구간에서의 최대 재현율 |",
        f"| ECE | {core['ece']:.4f} | 예측 확률 보정 오차 (낮을수록 우수) |",
        "",
    ]

    # 3. Top-k% 정밀도 / 재현율
    L += [
        "## 3. Top-k% 정밀도 / 재현율",
        "",
        "| k% | Precision | Recall |",
        "|----|-----------|--------|",
    ]
    for k_label in ["1%", "3%", "5%", "10%"]:
        key = k_label.replace("%", "pct")
        p = core[f"p_at_top{key}"]
        r = core[f"r_at_top{key}"]
        L.append(f"| {k_label} | {p:.4f} | {r:.4f} |")
    L.append("")

    # 4. 월별 진단
    L += [
        "## 4. 월별 진단",
        "",
        "| 월 | 샘플 수 | 양성 수 | 양성 비율 | PR-AUC | P@top5% |",
        "|----|--------|--------|---------|--------|---------|",
    ]
    for _, row in monthly_df.iterrows():
        pr_str = f"{row['pr_auc']:.4f}" if pd.notna(row["pr_auc"]) else "N/A"
        L.append(
            f"| {row['month']} | {row['n_samples']:,} | {row['n_positives']} "
            f"| {row['positive_rate']:.4f} | {pr_str} | {row['p_at_top5pct']:.4f} |"
        )
    L.append("")

    # 5. 국가별 진단
    valid_c = country_df[country_df["pr_auc"].notna()].reset_index(drop=True)
    top10 = valid_c.head(10)
    bot10 = valid_c.tail(10).iloc[::-1]

    L += [
        "## 5. 국가별 진단",
        "",
        "### 5-1. PR-AUC 상위 10개국",
        "",
        "| 국가 | 샘플 수 | 양성 수 | 양성 비율 | PR-AUC | P@top5% |",
        "|------|--------|--------|---------|--------|---------|",
    ]
    for _, row in top10.iterrows():
        L.append(
            f"| {row['country']} | {row['n_samples']:,} | {row['n_positives']} "
            f"| {row['positive_rate']:.4f} | {row['pr_auc']:.4f} | {row['p_at_top5pct']:.4f} |"
        )
    L += [
        "",
        "### 5-2. PR-AUC 하위 10개국",
        "",
        "| 국가 | 샘플 수 | 양성 수 | 양성 비율 | PR-AUC | P@top5% |",
        "|------|--------|--------|---------|--------|---------|",
    ]
    for _, row in bot10.iterrows():
        L.append(
            f"| {row['country']} | {row['n_samples']:,} | {row['n_positives']} "
            f"| {row['positive_rate']:.4f} | {row['pr_auc']:.4f} | {row['p_at_top5pct']:.4f} |"
        )
    L.append("")

    # 6. Top 5% 알림 분석
    L += [
        "## 6. Top 5% 알림 분석",
        "",
        f"- 총 알림 수: **{n_alerts:,}건**",
        f"- TP (실제 분쟁 상승): **{n_tp:,}건** ({n_tp/n_alerts*100:.1f}%)",
        f"- FP (오탐): **{n_fp:,}건** ({n_fp/n_alerts*100:.1f}%)",
        "",
    ]
    alert_by_country = (
        top5_df.groupby("country").size()
        .sort_values(ascending=False)
        .head(10)
    )
    L += [
        "### 알림 집중 상위 10개국 (TP + FP 합산)",
        "",
        "| 국가 | 알림 횟수 |",
        "|------|----------|",
    ]
    for country, cnt in alert_by_country.items():
        L.append(f"| {country} | {cnt} |")
    L.append("")

    # 7. 오탐(FP) / 미탐(FN) 분석
    L += [
        "## 7. 오탐(FP) / 미탐(FN) 분석",
        "",
        f"- **오탐(FP)**: top 5% 알림 중 실제 음성 — {n_fp:,}건",
        f"  - 저장 위치: `outputs/reports/diagnostics/false_positives.csv`",
        f"- **미탐(FN, top5% 외 양성)**: {n_fn_out5:,}건",
        f"  - 저장 위치: `outputs/reports/diagnostics/false_negatives_outside_top5pct.csv`",
        f"- **최저 확률 양성 30건**: `outputs/reports/diagnostics/false_negatives_lowest30.csv`",
        "",
    ]
    if len(fn_low30_df) > 0:
        L += [
            "### 가장 낮은 예측 확률 양성 샘플 (상위 10건 미리보기)",
            "",
            "| date | country | y_prob |",
            "|------|---------|--------|",
        ]
        for _, row in fn_low30_df.head(10).iterrows():
            L.append(f"| {_fmt_date(row[DATE_COL])} | {row['country']} | {row['y_prob']:.6f} |")
        L.append("")

    # 8. 피처 중요도 Top 30
    L += [
        "## 8. 피처 중요도 Top 30 (Gain 기준)",
        "",
        "| 순위 | 피처 | 중요도 (Gain) |",
        "|------|------|--------------|",
    ]
    for i, row in feat_df.iterrows():
        L.append(f"| {i + 1} | {row['feature']} | {row['importance_gain']:.2f} |")
    L += [
        "",
        "---",
        "",
        "*이 리포트는 val 세트 전용 진단입니다.*  ",
        "*test 세트는 모델 선택·튜닝·예측에 사용되지 않았습니다.*",
    ]

    return "\n".join(L)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ensure_output_dirs()
    os.makedirs(DIAG_DIR, exist_ok=True)
    SEP = "=" * 60

    # ── Step 1: Load model artifact ───────────────────────────────────────────
    print(SEP)
    print("Step 1 — 모델 아티팩트 로드")
    print(SEP)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: 모델 파일 없음 → {MODEL_PATH}")
        sys.exit(1)

    artifact           = joblib.load(MODEL_PATH)
    model              = artifact["model"]
    feature_cols       = artifact["feature_cols"]
    country_categories = artifact["country_categories"]

    print(f"  로드 완료  : {MODEL_PATH}")
    print(f"  피처 수    : {len(feature_cols)}")
    print(f"  국가 카테고리: {len(country_categories)}")
    print(f"  최적 반복  : {model.best_iteration}")

    # ── Step 2: Load SE scores ────────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 2 — SE 점수 로드 (full_se.parquet)")
    print(SEP)

    se_df = load_se_scores()
    print(f"  {len(se_df):,} 행  (date, country, macis_se_score)")

    # ── Step 3: Load & merge all splits (no test predictions) ────────────────
    print()
    print(SEP)
    print("Step 3 — 데이터 로드 및 SE 병합  [test 세트: 검증 전용, 예측 없음]")
    print(SEP)

    train_raw = pd.read_parquet(TRAIN_PATH)
    val_raw   = pd.read_parquet(VAL_PATH)
    test_raw  = pd.read_parquet(TEST_PATH)

    print(f"  train: {len(train_raw):,} 행  "
          f"({pd.to_datetime(train_raw[DATE_COL]).min().date()} ~ "
          f"{pd.to_datetime(train_raw[DATE_COL]).max().date()})")
    print(f"  val  : {len(val_raw):,} 행  "
          f"({pd.to_datetime(val_raw[DATE_COL]).min().date()} ~ "
          f"{pd.to_datetime(val_raw[DATE_COL]).max().date()})")
    print(f"  test : {len(test_raw):,} 행  (※ 예측 생성 없음)")
    print()

    merge_se_score(train_raw, se_df, "train")   # row-count validation
    val = merge_se_score(val_raw, se_df, "val")
    merge_se_score(test_raw,  se_df, "test")    # row-count validation

    # ── Step 4: Prepare val features ─────────────────────────────────────────
    print()
    print(SEP)
    print("Step 4 — val 피처 준비")
    print(SEP)

    val = val.copy().reset_index(drop=True)
    val["country"] = pd.Categorical(val["country"], categories=country_categories)
    X_val = val[feature_cols].copy()
    X_val["country"] = X_val["country"].astype("category")
    y_val = val[TARGET_COL].values

    print(f"  양성: {int(y_val.sum()):,} / {len(y_val):,}  (비율={y_val.mean():.4f})")

    # ── Step 5: Predict on val ────────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 5 — val 예측 생성")
    print(SEP)

    y_prob = model.predict(X_val, num_iteration=model.best_iteration)
    print(f"  y_prob 범위: [{y_prob.min():.6f}, {y_prob.max():.6f}]")

    # ── Step 6: Save val predictions ─────────────────────────────────────────
    print()
    print(SEP)
    print("Step 6 — val 예측 CSV 저장")
    print(SEP)

    val_pred_df = pd.DataFrame({
        "date":    pd.to_datetime(val[DATE_COL]).dt.strftime("%Y-%m-%d"),
        "country": val["country"].astype(str),
        "y_true":  y_val.astype(int),
        "y_prob":  y_prob,
    })
    val_pred_df.to_csv(VAL_PRED_PATH, index=False)
    print(f"  저장 완료: {VAL_PRED_PATH}  ({len(val_pred_df):,} 행)")

    # ── Step 7: Core metrics ──────────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 7 — 핵심 메트릭 계산")
    print(SEP)

    core = {
        "pr_auc":        compute_pr_auc(y_val, y_prob),
        "p_at_top1pct":  _p_at_k(y_val, y_prob, 0.01),
        "p_at_top3pct":  _p_at_k(y_val, y_prob, 0.03),
        "p_at_top5pct":  _p_at_k(y_val, y_prob, 0.05),
        "p_at_top10pct": _p_at_k(y_val, y_prob, 0.10),
        "r_at_top1pct":  _r_at_k(y_val, y_prob, 0.01),
        "r_at_top3pct":  _r_at_k(y_val, y_prob, 0.03),
        "r_at_top5pct":  _r_at_k(y_val, y_prob, 0.05),
        "r_at_top10pct": _r_at_k(y_val, y_prob, 0.10),
        "r_at_p010":     compute_recall_at_precision(y_val, y_prob, 0.10),
        "r_at_p020":     compute_recall_at_precision(y_val, y_prob, 0.20),
        "r_at_p030":     compute_recall_at_precision(y_val, y_prob, 0.30),
        "ece":           compute_ece(y_val, y_prob),
    }

    print(f"\n  {'메트릭':<36} {'값':>8}")
    print(f"  {'-'*36} {'-'*8}")
    for k, v in core.items():
        print(f"  {k:<36} {v:>8.4f}")

    # ── Step 8: Country diagnostics ───────────────────────────────────────────
    print()
    print(SEP)
    print("Step 8 — 국가별 진단")
    print(SEP)

    country_df = compute_country_diagnostics(val, y_prob)
    country_df.to_csv(COUNTRY_CSV, index=False)
    n_with_auc = country_df["pr_auc"].notna().sum()
    print(f"  {len(country_df)}개국  (PR-AUC 산출 가능: {n_with_auc}개국)")
    print(f"  저장 완료: {COUNTRY_CSV}")

    # ── Step 9: Monthly diagnostics ───────────────────────────────────────────
    print()
    print(SEP)
    print("Step 9 — 월별 진단")
    print(SEP)

    monthly_df = compute_monthly_diagnostics(val, y_prob)
    monthly_df.to_csv(MONTHLY_CSV, index=False)
    print(monthly_df.to_string(index=False))
    print(f"\n  저장 완료: {MONTHLY_CSV}")

    # ── Step 10: Top 5% alert analysis ────────────────────────────────────────
    print()
    print(SEP)
    print("Step 10 — Top 5% 알림 분석")
    print(SEP)

    top5_df = build_top5_alerts(val, y_prob)
    top5_df.to_csv(TOP5_CSV, index=False)
    n_tp = int((top5_df["alert_type"] == "TP").sum())
    n_fp = int((top5_df["alert_type"] == "FP").sum())
    print(f"  알림 수: {len(top5_df):,}  (TP={n_tp}, FP={n_fp})")
    print(f"  정밀도(P@top5%): {n_tp / len(top5_df):.4f}")
    print(f"  저장 완료: {TOP5_CSV}")

    # ── Step 11: False positives ──────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 11 — 오탐(FP) 저장")
    print(SEP)

    fp_df = build_false_positives(val, y_prob)
    fp_df.to_csv(FP_CSV, index=False)
    print(f"  오탐 수: {len(fp_df):,}건")
    print(f"  저장 완료: {FP_CSV}")

    # ── Step 12: False negatives (two views) ──────────────────────────────────
    print()
    print(SEP)
    print("Step 12 — 미탐(FN) 저장 — 두 가지 뷰")
    print(SEP)

    fn_out5_df, fn_low30_df = build_false_negatives(val, y_prob)
    fn_out5_df.to_csv(FN_OUT5_CSV, index=False)
    fn_low30_df.to_csv(FN_LOW30_CSV, index=False)
    print(f"  뷰 1 (top5% 외 양성): {len(fn_out5_df):,}건  → {FN_OUT5_CSV}")
    print(f"  뷰 2 (최저 확률 30건): {len(fn_low30_df)}건  → {FN_LOW30_CSV}")
    print()
    print("  [최저 확률 양성 상위 10건]")
    print(f"  {'date':<12} {'country':<30} {'y_prob':>8}")
    print(f"  {'-'*12} {'-'*30} {'-'*8}")
    for _, row in fn_low30_df.head(10).iterrows():
        print(f"  {_fmt_date(row[DATE_COL]):<12} {str(row['country']):<30} {row['y_prob']:>8.6f}")

    # ── Step 13: Feature importance top 30 ────────────────────────────────────
    print()
    print(SEP)
    print("Step 13 — 피처 중요도 Top 30 (Gain)")
    print(SEP)

    feat_df = (
        pd.DataFrame({
            "feature":          model.feature_name(),
            "importance_gain":  model.feature_importance(importance_type="gain"),
        })
        .sort_values("importance_gain", ascending=False)
        .head(30)
        .reset_index(drop=True)
    )
    feat_df.to_csv(FEAT_IMP_CSV, index=False)
    print(feat_df.to_string(index=False))
    print(f"\n  저장 완료: {FEAT_IMP_CSV}")

    # ── Step 14: Markdown report ──────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 14 — 마크다운 리포트 생성 (한국어)")
    print(SEP)

    overview = {
        "n_samples":      int(len(y_val)),
        "n_positives":    int(y_val.sum()),
        "positive_rate":  float(y_val.mean()),
        "n_features":     len(feature_cols),
        "best_iteration": int(model.best_iteration),
    }

    md_text = generate_md_report(
        overview, core, monthly_df, country_df,
        top5_df, fp_df, fn_out5_df, fn_low30_df, feat_df,
    )
    with open(MD_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"  저장 완료: {MD_REPORT_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("진단 완료 — 생성된 파일 목록")
    print(SEP)
    for path in [
        VAL_PRED_PATH,
        COUNTRY_CSV, MONTHLY_CSV, TOP5_CSV,
        FP_CSV, FN_OUT5_CSV, FN_LOW30_CSV, FEAT_IMP_CSV,
        MD_REPORT_PATH,
    ]:
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists}  {path}")
    print(SEP)


if __name__ == "__main__":
    main()
