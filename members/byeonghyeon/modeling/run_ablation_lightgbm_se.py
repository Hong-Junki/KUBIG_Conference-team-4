"""
Feature group ablation study for LightGBM + SE model.

Trains 12 LightGBM variants on different feature subsets and evaluates
each on the val set only. Test set is never loaded.

Usage (from project root):
    python modeling/run_ablation_lightgbm_se.py

Outputs
-------
outputs/reports/ablation_lightgbm_se.csv
outputs/reports/ablation_lightgbm_se.md
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import date as dt_date
import numpy as np
import pandas as pd
import lightgbm as lgb

from utils import (
    TARGET_COL, DATE_COL, LABEL_META_COLS,
    TRAIN_PATH, VAL_PATH,
    REPORT_DIR,
    ensure_output_dirs,
)
from evaluate import compute_pr_auc, compute_ece, compute_p_at_top_k, compute_recall_at_precision
from train_lightgbm_se import load_se_scores, merge_se_score

# ── Output paths ───────────────────────────────────────────────────────────────
ABLATION_CSV = os.path.join(REPORT_DIR, "ablation_lightgbm_se.csv")
ABLATION_MD  = os.path.join(REPORT_DIR, "ablation_lightgbm_se.md")

# ── Hyperparameters (identical to train_lightgbm_se.py for fair comparison) ───
RANDOM_SEED           = 42
NUM_BOOST_ROUND       = 1000
EARLY_STOPPING_ROUNDS = 50

LGB_BASE_PARAMS = {
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


# ── Feature set definitions ────────────────────────────────────────────────────

def build_feature_sets(all_feature_cols):
    """Build 12 named feature subsets from the full post-merge column list."""
    acled_cols  = sorted(c for c in all_feature_cols if c.startswith("acled_"))
    gdelt_cols  = sorted(c for c in all_feature_cols if c.startswith("gdelt_"))
    econ_cols   = sorted(c for c in all_feature_cols if c.startswith("econ_"))
    se_col      = [c for c in all_feature_cols if c == "macis_se_score"]
    country_col = [c for c in all_feature_cols if c == "country"]

    return {
        "full_with_se":                   list(all_feature_cols),
        "no_se":                          [c for c in all_feature_cols if c != "macis_se_score"],
        "no_country":                     [c for c in all_feature_cols if c != "country"],
        "no_se_no_country":               [c for c in all_feature_cols
                                           if c not in ("macis_se_score", "country")],
        "se_only":                        se_col,
        "country_only":                   country_col,
        "acled_only":                     acled_cols,
        "gdelt_only":                     gdelt_cols,
        "econ_only":                      econ_cols,
        "acled_gdelt_only":               acled_cols + gdelt_cols,
        "acled_gdelt_se":                 acled_cols + gdelt_cols + se_col,
        "acled_gdelt_econ_se_no_country": acled_cols + gdelt_cols + econ_cols + se_col,
    }


# ── Single experiment ──────────────────────────────────────────────────────────

def run_experiment(train_df, val_df, feature_cols, country_categories, label):
    """
    Train LightGBM on feature_cols and evaluate on val_df.
    Returns a metrics dict, or None if feature_cols is empty.
    """
    if not feature_cols:
        print(f"  [{label}] 피처 없음 — skip")
        return None

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].values
    X_val   = val_df[feature_cols].copy()
    y_val   = val_df[TARGET_COL].values

    cat_cols = []
    if "country" in feature_cols:
        for df in (X_train, X_val):
            df["country"] = pd.Categorical(
                df["country"], categories=country_categories
            ).astype("category")
        cat_cols = ["country"]

    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    spw = neg / pos

    params = {**LGB_BASE_PARAMS, "scale_pos_weight": spw}

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False),
        lgb.log_evaluation(9999),   # suppress per-iteration output
    ]

    if cat_cols:
        dtrain = lgb.Dataset(
            X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False
        )
        dval = lgb.Dataset(
            X_val, label=y_val, categorical_feature=cat_cols,
            reference=dtrain, free_raw_data=False
        )
    else:
        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain, free_raw_data=False)

    model = lgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )

    y_prob = model.predict(X_val, num_iteration=model.best_iteration)

    return {
        "feature_set":    label,
        "n_features":     len(feature_cols),
        "pr_auc":         compute_pr_auc(y_val, y_prob),
        "p_at_top5pct":   compute_p_at_top_k(y_val, y_prob, k=0.05),
        "r_at_p010":      compute_recall_at_precision(y_val, y_prob, 0.10),
        "r_at_p020":      compute_recall_at_precision(y_val, y_prob, 0.20),
        "r_at_p030":      compute_recall_at_precision(y_val, y_prob, 0.30),
        "ece":            compute_ece(y_val, y_prob),
        "best_iteration": int(model.best_iteration),
    }


# ── Markdown report (Korean) ───────────────────────────────────────────────────

def generate_md_report(df):
    today = dt_date.today().isoformat()

    # Index by feature_set for easy lookup
    r = df.set_index("feature_set")

    def v(name, col):
        return float(r.loc[name, col])

    full = "full_with_se"

    # Key deltas
    delta_se_prauc = v(full, "pr_auc") - v("no_se", "pr_auc")
    delta_se_p5    = v(full, "p_at_top5pct") - v("no_se", "p_at_top5pct")
    delta_co_prauc = v(full, "pr_auc") - v("no_country", "pr_auc")
    delta_co_p5    = v(full, "p_at_top5pct") - v("no_country", "p_at_top5pct")

    # Best overall
    best_prauc_row = df.loc[df["pr_auc"].idxmax()]
    best_p5_row    = df.loc[df["p_at_top5pct"].idxmax()]

    # Best pure-signal set (no SE, no country)
    pure_signal_sets = ["acled_only", "gdelt_only", "econ_only", "acled_gdelt_only"]
    pure_signal_df   = df[df["feature_set"].isin(pure_signal_sets)]
    best_signal      = pure_signal_df.loc[pure_signal_df["pr_auc"].idxmax()]

    # Best explainable set (no country, for dashboard use)
    no_country_candidates = [
        "no_country", "no_se_no_country", "se_only",
        "acled_only", "gdelt_only", "econ_only",
        "acled_gdelt_only", "acled_gdelt_se", "acled_gdelt_econ_se_no_country",
    ]
    explain_df  = df[df["feature_set"].isin(no_country_candidates)]
    best_explain = explain_df.loc[explain_df["pr_auc"].idxmax()]

    # Recommendation logic (data-driven)
    country_dominant = delta_co_prauc > 0.02
    se_dominant      = delta_se_prauc > 0.02
    poor_ece         = v(full, "ece") > 0.15

    if country_dominant:
        recommendation = "CatBoost"
        rec_reason = (
            f"country 제거 시 PR-AUC가 {delta_co_prauc:.4f} 하락합니다. "
            "국가 고정효과가 강하므로 CatBoost의 Ordered Target Statistics 인코딩이 "
            "국가 범주를 더 정밀하게 포착할 수 있습니다."
        )
    elif poor_ece:
        recommendation = "Calibration (Platt scaling 또는 Isotonic Regression)"
        rec_reason = (
            f"전체 피처 모델 ECE={v(full, 'ece'):.4f}로 확률 보정 품질이 낮습니다. "
            "보정 레이어 추가로 실용 신뢰도를 빠르게 개선할 수 있습니다."
        )
    elif se_dominant:
        recommendation = "EBM (Explainable Boosting Machine)"
        rec_reason = (
            f"macis_se_score 제거 시 PR-AUC가 {delta_se_prauc:.4f} 하락합니다. "
            "EBM의 shape function으로 SE 점수 비선형 임계 구간을 시각화하면 "
            "도메인 전문가 검증 및 대시보드 설명에 직접 활용할 수 있습니다."
        )
    else:
        recommendation = "EBM (Explainable Boosting Machine)"
        rec_reason = (
            "SE·country 기여도가 모두 제한적입니다. "
            "EBM으로 피처별 shape function을 분석해 다음 피처 엔지니어링 방향을 설정하는 것을 권장합니다."
        )

    L = []

    # ── Header ────────────────────────────────────────────────────────────────
    L += [
        "# LightGBM + SE — 피처 그룹 기여도 Ablation 실험 리포트",
        "",
        f"생성일: {today}",
        "평가 데이터: **val 세트 전용** (2024-01 ~ 2024-06, test 세트 미사용)",
        "하이퍼파라미터: `train_lightgbm_se.py`와 동일",
        "(learning_rate=0.05, num_leaves=63, early_stopping_rounds=50)",
        "",
    ]

    # ── 1. 전체 결과 표 ────────────────────────────────────────────────────────
    sorted_df = df.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    L += [
        "## 1. 전체 실험 결과 (PR-AUC 내림차순)",
        "",
        "| 피처 세트 | n_feat | PR-AUC | P@top5% | R@P≥0.10 | R@P≥0.20 | R@P≥0.30 | ECE | best_iter |",
        "|-----------|--------|--------|---------|---------|---------|---------|-----|-----------|",
    ]
    for _, row in sorted_df.iterrows():
        marker = "  ← 기준선" if row["feature_set"] == full else ""
        L.append(
            f"| `{row['feature_set']}`{marker} | {int(row['n_features'])} "
            f"| {row['pr_auc']:.4f} | {row['p_at_top5pct']:.4f} "
            f"| {row['r_at_p010']:.4f} | {row['r_at_p020']:.4f} | {row['r_at_p030']:.4f} "
            f"| {row['ece']:.4f} | {int(row['best_iteration'])} |"
        )
    L.append("")

    # ── 2. macis_se_score 의존도 ───────────────────────────────────────────────
    se_pct = delta_se_prauc / v(full, "pr_auc") * 100 if v(full, "pr_auc") > 0 else 0
    L += [
        "## 2. macis_se_score 의존도",
        "",
        "| 비교 | PR-AUC | P@top5% |",
        "|------|--------|---------|",
        f"| `full_with_se` (기준선) | {v(full, 'pr_auc'):.4f} | {v(full, 'p_at_top5pct'):.4f} |",
        f"| `no_se` (SE 제거) | {v('no_se', 'pr_auc'):.4f} | {v('no_se', 'p_at_top5pct'):.4f} |",
        f"| `se_only` (SE만) | {v('se_only', 'pr_auc'):.4f} | {v('se_only', 'p_at_top5pct'):.4f} |",
        f"| delta (full − no_se) | **{delta_se_prauc:+.4f}** | **{delta_se_p5:+.4f}** |",
        "",
    ]
    if abs(delta_se_prauc) > 0.03:
        L.append(
            f"macis_se_score 제거 시 PR-AUC가 **{delta_se_prauc:.4f}** ({se_pct:.1f}%) 하락합니다. "
            f"모델이 SE 점수에 **강하게 의존**하고 있습니다. "
            f"`se_only` 단독 PR-AUC가 {v('se_only', 'pr_auc'):.4f}로 상당한 예측력을 가지며, "
            "SE 점수가 대부분의 분쟁 신호를 이미 요약하고 있음을 시사합니다."
        )
    elif abs(delta_se_prauc) > 0.01:
        L.append(
            f"macis_se_score 제거 시 PR-AUC가 **{delta_se_prauc:.4f}** ({se_pct:.1f}%) 하락합니다. "
            "SE 점수의 기여도가 **중간 수준**입니다."
        )
    else:
        L.append(
            f"macis_se_score 제거 시 PR-AUC 변화가 **{delta_se_prauc:.4f}**로 미미합니다. "
            "다른 피처들이 SE 점수의 정보를 상당 부분 커버하고 있습니다."
        )
    L.append("")

    # ── 3. country 고정효과 강도 ───────────────────────────────────────────────
    co_pct = delta_co_prauc / v(full, "pr_auc") * 100 if v(full, "pr_auc") > 0 else 0
    L += [
        "## 3. country 고정효과 강도",
        "",
        "| 비교 | PR-AUC | P@top5% |",
        "|------|--------|---------|",
        f"| `full_with_se` (기준선) | {v(full, 'pr_auc'):.4f} | {v(full, 'p_at_top5pct'):.4f} |",
        f"| `no_country` (country 제거) | {v('no_country', 'pr_auc'):.4f} | {v('no_country', 'p_at_top5pct'):.4f} |",
        f"| `country_only` (country만) | {v('country_only', 'pr_auc'):.4f} | {v('country_only', 'p_at_top5pct'):.4f} |",
        f"| delta (full − no_country) | **{delta_co_prauc:+.4f}** | **{delta_co_p5:+.4f}** |",
        "",
    ]
    if abs(delta_co_prauc) > 0.03:
        L.append(
            f"country 제거 시 PR-AUC가 **{delta_co_prauc:.4f}** ({co_pct:.1f}%) 하락합니다. "
            "국가 고정효과가 **매우 강합니다**. "
            "국가별 베이스라인 위험률 차이가 크며, "
            "CatBoost의 native categorical 인코딩이 이를 더 정밀하게 모델링할 수 있습니다."
        )
    elif abs(delta_co_prauc) > 0.01:
        L.append(
            f"country 제거 시 PR-AUC가 **{delta_co_prauc:.4f}** ({co_pct:.1f}%) 하락합니다. "
            "국가 고정효과가 **중간 수준**으로 존재합니다."
        )
    else:
        L.append(
            f"country 제거 시 PR-AUC 변화가 **{delta_co_prauc:.4f}**로 미미합니다. "
            "국가 고정효과가 모델 성능에 미치는 영향이 제한적입니다."
        )
    L.append("")

    # ── 4. 순수 신호 피처 유효성 ────────────────────────────────────────────────
    signal_ratio = best_signal["pr_auc"] / v(full, "pr_auc") * 100
    L += [
        "## 4. 순수 신호 피처 (ACLED/GDELT/경제) 단독 유효성",
        "",
        "SE 점수와 country를 제외한 개별 신호 그룹의 예측력:",
        "",
        "| 피처 세트 | n_feat | PR-AUC | P@top5% | ECE |",
        "|-----------|--------|--------|---------|-----|",
    ]
    for fs in ["no_se_no_country", "acled_only", "gdelt_only", "econ_only", "acled_gdelt_only",
               "acled_gdelt_econ_se_no_country"]:
        if fs in r.index:
            row = r.loc[fs]
            L.append(
                f"| `{fs}` | {int(row['n_features'])} | {row['pr_auc']:.4f} "
                f"| {row['p_at_top5pct']:.4f} | {row['ece']:.4f} |"
            )
    L.append("")

    L.append(
        f"SE·country 없이 가장 좋은 단독 신호 그룹: "
        f"**`{best_signal['feature_set']}`** "
        f"(PR-AUC={best_signal['pr_auc']:.4f}, 기준선 대비 {signal_ratio:.1f}%)"
    )
    L.append("")

    if signal_ratio < 60:
        L.append(
            "ACLED/GDELT/경제 신호의 단독 예측력이 낮습니다. "
            "SE 점수와 country가 핵심 신호를 담당하며, 이들 없이는 성능이 크게 저하됩니다."
        )
    elif signal_ratio < 80:
        L.append(
            "ACLED/GDELT/경제 신호가 기준선의 60~80% 수준의 예측력을 보입니다. "
            "SE·country 의존도는 높지만 순수 이벤트/미디어 신호도 독립적 가치를 가집니다."
        )
    else:
        L.append(
            "ACLED/GDELT/경제 신호만으로도 기준선의 80% 이상 성능을 달성합니다. "
            "SE 점수와 country 없이도 충분한 예측력이 확보됩니다."
        )
    L.append("")

    # ── 5. 대시보드 설명 적합 피처 세트 ──────────────────────────────────────────
    L += [
        "## 5. 대시보드 설명 적합 피처 세트",
        "",
        "대시보드 사용자는 '이 국가가 왜 고위험인가'를 이해해야 합니다. "
        "`country` 고정효과는 순환 논리(국가이기 때문에 위험)가 되므로, "
        "country 없이 가장 높은 PR-AUC를 달성하는 피처 세트가 설명에 적합합니다.",
        "",
        f"**추천 피처 세트**: `{best_explain['feature_set']}`",
        f"- PR-AUC: **{best_explain['pr_auc']:.4f}**",
        f"- P@top5%: **{best_explain['p_at_top5pct']:.4f}**",
        f"- 피처 수: {int(best_explain['n_features'])}",
        "",
    ]

    # ── 6. 다음 실험 추천 ─────────────────────────────────────────────────────
    L += [
        "## 6. 다음 실험 추천",
        "",
        f"### 추천: {recommendation}",
        "",
        rec_reason,
        "",
    ]

    # ── 요약 ─────────────────────────────────────────────────────────────────
    L += [
        "---",
        "",
        "## 요약",
        "",
        f"| 항목 | 결과 |",
        f"|------|------|",
        f"| PR-AUC 최고 피처 세트 | `{best_prauc_row['feature_set']}` ({best_prauc_row['pr_auc']:.4f}) |",
        f"| P@top5% 최고 피처 세트 | `{best_p5_row['feature_set']}` ({best_p5_row['p_at_top5pct']:.4f}) |",
        f"| macis_se_score 기여도 | PR-AUC {delta_se_prauc:+.4f} / P@top5% {delta_se_p5:+.4f} |",
        f"| country 기여도 | PR-AUC {delta_co_prauc:+.4f} / P@top5% {delta_co_p5:+.4f} |",
        f"| 순수 신호 최고 PR-AUC | `{best_signal['feature_set']}` = {best_signal['pr_auc']:.4f} |",
        f"| 대시보드 추천 피처 세트 | `{best_explain['feature_set']}` (PR-AUC {best_explain['pr_auc']:.4f}) |",
        f"| 다음 실험 추천 | {recommendation} |",
        "",
        "*이 리포트는 val 세트 전용 ablation입니다. test 세트는 사용되지 않았습니다.*",
    ]

    return "\n".join(L)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ensure_output_dirs()
    SEP = "=" * 60

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print(SEP)
    print("Step 1 — 데이터 로드 및 SE 병합  [test 세트 미로드]")
    print(SEP)

    se_df     = load_se_scores()
    train_raw = pd.read_parquet(TRAIN_PATH)
    val_raw   = pd.read_parquet(VAL_PATH)

    print(f"  train: {len(train_raw):,} 행  "
          f"({pd.to_datetime(train_raw[DATE_COL]).min().date()} ~ "
          f"{pd.to_datetime(train_raw[DATE_COL]).max().date()})")
    print(f"  val  : {len(val_raw):,} 행  "
          f"({pd.to_datetime(val_raw[DATE_COL]).min().date()} ~ "
          f"{pd.to_datetime(val_raw[DATE_COL]).max().date()})")
    print()

    train = merge_se_score(train_raw, se_df, "train").reset_index(drop=True)
    val   = merge_se_score(val_raw,   se_df, "val").reset_index(drop=True)

    # ── Step 2: Detect feature groups ─────────────────────────────────────────
    print()
    print(SEP)
    print("Step 2 — 피처 그룹 탐지 (prefix 기반)")
    print(SEP)

    exclude = set(LABEL_META_COLS) | {DATE_COL}
    all_feature_cols = [c for c in train.columns if c not in exclude]

    acled_cols  = sorted(c for c in all_feature_cols if c.startswith("acled_"))
    gdelt_cols  = sorted(c for c in all_feature_cols if c.startswith("gdelt_"))
    econ_cols   = sorted(c for c in all_feature_cols if c.startswith("econ_"))
    other_cols  = [c for c in all_feature_cols
                   if c not in acled_cols + gdelt_cols + econ_cols
                   and c not in ("macis_se_score", "country")]

    print(f"  전체 피처          : {len(all_feature_cols)}")
    print(f"  ├─ acled_*        : {len(acled_cols)}")
    print(f"  ├─ gdelt_*        : {len(gdelt_cols)}")
    print(f"  ├─ econ_*         : {len(econ_cols)}")
    print(f"  ├─ macis_se_score : 1")
    print(f"  ├─ country        : 1")
    if other_cols:
        print(f"  └─ 기타           : {len(other_cols)}  {other_cols}")
    else:
        print(f"  └─ 기타           : 0")

    feature_sets       = build_feature_sets(all_feature_cols)
    country_categories = sorted(train["country"].dropna().unique().tolist())
    print(f"\n  국가 카테고리: {len(country_categories)}개")

    # ── Step 3: Run ablation ───────────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 3 — 12개 피처 세트 학습 및 평가")
    print(SEP)

    results = []
    for i, (set_name, feat_cols) in enumerate(feature_sets.items(), 1):
        print(f"\n  [{i:02d}/12] {set_name}  (n_features={len(feat_cols)})")
        metrics = run_experiment(train, val, feat_cols, country_categories, set_name)
        if metrics is not None:
            results.append(metrics)
            print(
                f"           PR-AUC={metrics['pr_auc']:.4f}  "
                f"P@top5%={metrics['p_at_top5pct']:.4f}  "
                f"ECE={metrics['ece']:.4f}  "
                f"best_iter={metrics['best_iteration']}"
            )

    # ── Step 4: Assemble results DataFrame ────────────────────────────────────
    col_order = [
        "feature_set", "n_features", "pr_auc", "p_at_top5pct",
        "r_at_p010", "r_at_p020", "r_at_p030", "ece", "best_iteration",
    ]
    results_df = pd.DataFrame(results)[col_order]

    # ── Step 5: Print comparison table ────────────────────────────────────────
    print()
    print(SEP)
    print("Step 4 — 결과 비교 (PR-AUC 내림차순)")
    print(SEP)
    display_df = results_df.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    print(f"\n  {'feature_set':<36} {'n_feat':>6} {'PR-AUC':>8} {'P@top5%':>8} {'ECE':>7} {'best_iter':>10}")
    print(f"  {'-'*36} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*10}")
    for _, row in display_df.iterrows():
        marker = " ←" if row["feature_set"] == "full_with_se" else ""
        print(
            f"  {row['feature_set']:<36} {int(row['n_features']):>6} "
            f"{row['pr_auc']:>8.4f} {row['p_at_top5pct']:>8.4f} "
            f"{row['ece']:>7.4f} {int(row['best_iteration']):>10}{marker}"
        )

    # ── Step 6: Save outputs ───────────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 5 — 결과 저장")
    print(SEP)

    results_df.to_csv(ABLATION_CSV, index=False)
    print(f"  저장 완료: {ABLATION_CSV}")

    md_text = generate_md_report(results_df)
    with open(ABLATION_MD, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"  저장 완료: {ABLATION_MD}")

    print()
    print(SEP)
    print("Ablation 완료.")
    print(SEP)


if __name__ == "__main__":
    main()
