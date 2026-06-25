"""
Feature group ablation study for LightGBM + SE + time-series derived features.

Experiments:
  A. SE baseline only
  B. SE + diff features   (tf_diff14_*, tf_diff30_*)
  C. SE + ratio features  (tf_ratio14_*, tf_ratio30_*)
  D. SE + z-score features (tf_z90_*)
  E. SE + momentum features (tf_mom3_*)
  F. SE + interaction features (tf_ix_se_*)
  G. SE + best groups (conservative auto-selection: PR-AUC >= SE AND P@top5% >= SE)

Test set is NEVER loaded. No submission CSV is produced.
train+val are concat-sorted for feature generation, then split back.

Usage (from project root):
    python modeling/train_lightgbm_se_time_feature_ablation.py

Outputs:
    outputs/reports/time_feature_ablation_val.md
    outputs/reports/time_feature_ablation_val.json
    outputs/reports/time_feature_ablation_summary.csv

IMPORTANT — Submission CSV format:
    final submission: date,country,y_prob only.
    This script does NOT produce a submission CSV.
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

# ── Output paths ──────────────────────────────────────────────────────────────
MD_PATH      = os.path.join(REPORT_DIR, "time_feature_ablation_val.md")
JSON_PATH    = os.path.join(REPORT_DIR, "time_feature_ablation_val.json")
CSV_PATH     = os.path.join(REPORT_DIR, "time_feature_ablation_summary.csv")
SE_DIAG_PATH = os.path.join(REPORT_DIR, "lightgbm_se_diagnostic_val.json")

# ── Shared hyperparameters (identical to SE model for fair comparison) ─────────
RANDOM_SEED           = 42
NUM_BOOST_ROUND       = 1000
EARLY_STOPPING_ROUNDS = 50
LOG_PERIOD            = 200   # less verbose for multi-experiment run

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

# Feature engineering constants
EPSILON     = 1e-6
RATIO_CLIP  = 20.0   # cap ratios at ±20 to suppress explosion near-zero denominator
ZSCORE_CLIP = 5.0    # cap z-scores at ±5 standard deviations
ROLLING_WIN = 90


# ── Feature engineering helpers ────────────────────────────────────────────────

def find_base_features(df):
    """Columns with all three windows (7d/14d/30d), acled_* or gdelt_* prefix."""
    candidates = {}
    for col in df.columns:
        for sfx in ("_7d", "_14d", "_30d"):
            if col.endswith(sfx):
                prefix = col[: -len(sfx)]
                if prefix.startswith(("acled_", "gdelt_")):
                    candidates.setdefault(prefix, set()).add(sfx)
    return sorted(p for p, w in candidates.items() if {"_7d", "_14d", "_30d"} <= w)


def add_diff_features(df, bases, out):
    """7d - 14d/2 and 7d - 30d/4. No time-axis shift needed (all pre-computed windows)."""
    for b in bases:
        n1 = f"tf_diff14_{b}"; df[n1] = df[f"{b}_7d"] - df[f"{b}_14d"] / 2.0; out.append(n1)
        n2 = f"tf_diff30_{b}"; df[n2] = df[f"{b}_7d"] - df[f"{b}_30d"] / 4.0; out.append(n2)
    return df


def add_ratio_features(df, bases, out):
    """7d / (14d/2 + ε) and 7d / (30d/4 + ε), clipped to ±RATIO_CLIP."""
    for b in bases:
        n1 = f"tf_ratio14_{b}"
        df[n1] = (df[f"{b}_7d"] / (df[f"{b}_14d"] / 2.0 + EPSILON)).clip(-RATIO_CLIP, RATIO_CLIP)
        out.append(n1)
        n2 = f"tf_ratio30_{b}"
        df[n2] = (df[f"{b}_7d"] / (df[f"{b}_30d"] / 4.0 + EPSILON)).clip(-RATIO_CLIP, RATIO_CLIP)
        out.append(n2)
    return df


def add_zscore_features(df, bases, out):
    """
    Country-level rolling 90d z-score.
    shift(1) applied before rolling so the current row is NOT part of its own baseline.
    z = (feat_7d[t] - mean([t-90, t-1])) / (std([t-90, t-1]) + ε), clipped ±5.
    """
    df = df.sort_values([DATE_COL, "country"]).copy()
    for b in bases:
        col7 = f"{b}_7d"
        name = f"tf_z90_{b}"

        def _z(grp, col=col7):
            sh = grp[col].shift(1)
            rm = sh.rolling(ROLLING_WIN, min_periods=7).mean()
            rs = sh.rolling(ROLLING_WIN, min_periods=7).std()
            return ((grp[col] - rm) / (rs + EPSILON)).clip(-ZSCORE_CLIP, ZSCORE_CLIP)

        df[name] = df.groupby("country", group_keys=False).apply(_z)
        out.append(name)
    return df


def add_momentum_features(df, bases, out):
    """
    Country-level lag-3 momentum: feat_7d[t] - feat_7d[t-3].
    shift(3) within each country group; uses only past values.
    """
    df = df.sort_values([DATE_COL, "country"]).copy()
    for b in bases:
        col7 = f"{b}_7d"
        name = f"tf_mom3_{b}"

        def _mom(grp, col=col7):
            return grp[col] - grp[col].shift(3)

        df[name] = df.groupby("country", group_keys=False).apply(_mom)
        out.append(name)
    return df


def add_interaction_features(df, bases, out):
    """
    macis_se_score × z-score (3 bases) and × diff30 (3 bases).
    Both z-score and diff30 are leakage-free, so their products are too.
    """
    for b in ["acled_fatalities", "acled_event_count", "gdelt_goldstein_mean"]:
        zc = f"tf_z90_{b}"
        if zc in df.columns:
            nm = f"tf_ix_se_z_{b}"; df[nm] = df["macis_se_score"] * df[zc]; out.append(nm)
    for b in ["acled_fatalities", "gdelt_mentions_sum", "gdelt_tone_mean"]:
        dc = f"tf_diff30_{b}"
        if dc in df.columns:
            nm = f"tf_ix_se_d30_{b}"; df[nm] = df["macis_se_score"] * df[dc]; out.append(nm)
    return df


# ── Metric helpers ─────────────────────────────────────────────────────────────

def p_at_k(y_true, y_prob, k):
    top = np.argsort(y_prob)[::-1][: max(1, int(np.ceil(len(y_true) * k)))]
    return float(np.mean(y_true[top]))


def r_at_k(y_true, y_prob, k):
    top = np.argsort(y_prob)[::-1][: max(1, int(np.ceil(len(y_true) * k)))]
    n_pos = int(y_true.sum())
    return float(y_true[top].sum() / n_pos) if n_pos else 0.0


def r_at_prec(y_true, y_prob, min_prec):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    valid = prec >= min_prec
    return float(rec[valid].max()) if valid.any() else 0.0


def monthly_auc(val_df, y_prob):
    df = val_df[[DATE_COL, TARGET_COL]].copy()
    df["_p"] = y_prob
    df["_m"] = pd.to_datetime(df[DATE_COL]).dt.to_period("M")
    out = {}
    for m, g in df.groupby("_m"):
        yt, yp = g[TARGET_COL].values, g["_p"].values
        out[str(m)] = float(average_precision_score(yt, yp)) if yt.sum() > 0 else None
    return out


def full_metrics(y_true, y_prob, val_df=None):
    m = {
        "pr_auc":                  compute_pr_auc(y_true, y_prob),
        "p_at_top1pct":            p_at_k(y_true, y_prob, 0.01),
        "p_at_top3pct":            p_at_k(y_true, y_prob, 0.03),
        "p_at_top5pct":            p_at_k(y_true, y_prob, 0.05),
        "p_at_top10pct":           p_at_k(y_true, y_prob, 0.10),
        "recall_at_top1pct":       r_at_k(y_true, y_prob, 0.01),
        "recall_at_top3pct":       r_at_k(y_true, y_prob, 0.03),
        "recall_at_top5pct":       r_at_k(y_true, y_prob, 0.05),
        "recall_at_top10pct":      r_at_k(y_true, y_prob, 0.10),
        "recall_at_precision_010": r_at_prec(y_true, y_prob, 0.10),
        "recall_at_precision_020": r_at_prec(y_true, y_prob, 0.20),
        "recall_at_precision_030": r_at_prec(y_true, y_prob, 0.30),
        "ece":                     compute_ece(y_true, y_prob),
    }
    if val_df is not None:
        m["monthly_pr_auc"] = monthly_auc(val_df, y_prob)
    return m


# ── Training helper ────────────────────────────────────────────────────────────

def train_and_eval(train_df, val_df, feature_cols, tag):
    """Train one LightGBM model and return (metrics_dict, model, best_iter)."""
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    X_tr, y_tr = make_xy(train_df, feature_cols)
    X_va, y_va = make_xy(val_df,   feature_cols)

    spw = float((y_tr == 0).sum()) / float(y_tr.sum())
    params = {**LGB_PARAMS, "scale_pos_weight": spw}

    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
    dval   = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols,
                         reference=dtrain, free_raw_data=False)

    model = lgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False),
            lgb.log_evaluation(LOG_PERIOD),
        ],
    )
    y_prob = model.predict(X_va, num_iteration=model.best_iteration)
    m = full_metrics(y_va.values, y_prob, val_df=val_df)
    m["best_iteration"] = int(model.best_iteration)
    m["n_features"]     = len(feature_cols)

    # top-5 feature importance for the summary in the report
    imp = model.feature_importance(importance_type="gain")
    names = model.feature_name()
    top5 = sorted(zip(names, imp), key=lambda x: x[1], reverse=True)[:5]
    m["top5_features"] = [{"feature": n, "gain": float(g)} for n, g in top5]

    print(f"  [{tag}] iter={model.best_iteration:3d}  "
          f"PR-AUC={m['pr_auc']:.4f}  P@5%={m['p_at_top5pct']:.4f}  "
          f"ECE={m['ece']:.4f}  n_feat={len(feature_cols)}")
    return m, model


# ── Markdown report ────────────────────────────────────────────────────────────

def generate_md(results, se_base, g_groups_used, g_is_empty, today):
    """results: dict[tag] -> metrics dict. se_base: SE metrics dict."""
    L = []

    COMPARE_KEYS = [
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

    DELTA_KEYS = [
        ("pr_auc",                  "PR-AUC Δ"),
        ("p_at_top1pct",            "P@top1% Δ"),
        ("p_at_top5pct",            "P@top5% Δ"),
        ("recall_at_precision_010", "R@P≥0.10 Δ"),
        ("ece",                     "ECE Δ"),
    ]

    TAG_LABELS = {
        "A": "SE baseline",
        "B": "SE + diff",
        "C": "SE + ratio",
        "D": "SE + z-score",
        "E": "SE + momentum",
        "F": "SE + interaction",
        "G": "SE + best groups",
    }

    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    def delta(tv, sv, is_ece=False):
        if tv is None or sv is None:
            return "—"
        d = tv - sv
        sign = "+" if d >= 0 else ""
        if is_ece:
            mark = " ▼(개선)" if d < 0 else (" ▲(악화)" if d > 0 else "")
        else:
            mark = " ▲" if d > 0 else (" ▼" if d < 0 else "")
        return f"{sign}{d:.4f}{mark}"

    # Header
    L += [
        "# Feature Group Ablation — Validation Report",
        "",
        f"Generated: {today}",
        "Evaluated on: **val set only** (2024-01 ~ 2024-06)",
        "Test set: **NOT loaded**  |  Submission CSV: **NOT produced**",
        "",
        "> **⚠️ 제출 CSV 형식 주의사항**",
        "> 최종 제출 파일은 반드시 `date,country,y_prob` **세 컬럼만** 유지해야 합니다.",
        "",
    ]

    # 1. Experiment overview
    L += ["## 1. Experiment Overview", ""]
    L += [
        "| Experiment | Feature groups added | n_derived |",
        "|-----------|---------------------|-----------|",
        "| A — SE baseline | (없음) | 0 |",
        "| B — SE + diff | tf_diff14_*, tf_diff30_* | 16 |",
        "| C — SE + ratio | tf_ratio14_*, tf_ratio30_* | 16 |",
        "| D — SE + z-score | tf_z90_* | 8 |",
        "| E — SE + momentum | tf_mom3_* | 8 |",
        "| F — SE + interaction | tf_ix_se_* | 6 |",
    ]
    if g_is_empty:
        L.append("| G — best groups | **선택된 group 없음 → 기존 SE 유지** | 0 |")
    else:
        g_label = " + ".join(g_groups_used) if g_groups_used else "없음"
        g_n = results["G"]["n_features"] - results["A"]["n_features"]
        L.append(f"| G — best groups | {g_label} | {g_n} |")
    L.append("")

    L += [
        "**G 선택 기준 (보수적):**",
        "",
        "- PR-AUC ≥ SE baseline **AND** P@top5% ≥ SE baseline 인 group만 채택",
        "- P@top5% < SE baseline 인 group은 PR-AUC 개선 여부에 관계없이 제외",
        "- P@top1% 개선이 있어도 P@top5%, R@P≥0.10, ECE 중 하나라도 악화되면 대시보드 기본 모델로 채택하지 않음",
        "",
    ]

    # 2. Delta summary (5 key metrics × 6 experiments vs SE)
    L += ["## 2. Key Delta Summary vs SE Baseline", ""]
    header = "| Metric | " + " | ".join(TAG_LABELS[t] for t in "BCDEFG" if t in results) + " |"
    sep    = "|--------|" + "|".join("-------" for t in "BCDEFG" if t in results) + "|"
    L += [header, sep]

    for key, label in DELTA_KEYS:
        is_ece = "ece" in key
        sv = se_base.get(key)
        row = f"| {label} |"
        for tag in [t for t in "BCDEFG" if t in results]:
            tv = results[tag].get(key)
            row += f" {delta(tv, sv, is_ece)} |"
        L.append(row)
    L.append("")

    # 3. Full metrics table per experiment
    L += ["## 3. Full Metrics — All Experiments", ""]
    tags_available = [t for t in "ABCDEFG" if t in results]
    header = "| Metric | " + " | ".join(TAG_LABELS[t] for t in tags_available) + " |"
    sep    = "|--------|" + "|".join("-------" for _ in tags_available) + "|"
    L += [header, sep]

    for key, label in COMPARE_KEYS:
        is_ece = "ece" in key
        row = f"| {label} |"
        for tag in tags_available:
            v = results[tag].get(key)
            if tag == "A":
                row += f" {fmt(v)} |"
            else:
                sv = se_base.get(key)
                tv = v
                row += f" {fmt(tv)} ({delta(tv, sv, is_ece)}) |"
        L.append(row)
    L.append("")

    # 4. Monthly PR-AUC
    L += ["## 4. Monthly PR-AUC", ""]
    months = sorted({
        m for tag in tags_available
        for m in results[tag].get("monthly_pr_auc", {}).keys()
    })
    header = "| Month | " + " | ".join(TAG_LABELS[t] for t in tags_available) + " |"
    sep    = "|-------|" + "|".join("-------" for _ in tags_available) + "|"
    L += [header, sep]

    for month in months:
        is_apr = month == "2024-04"
        row = f"| {month}{'  ⚠️' if is_apr else ''} |"
        for tag in tags_available:
            v = results[tag].get("monthly_pr_auc", {}).get(month)
            row += f" {fmt(v)} |"
        L.append(row)
    L.append("")

    # Monthly statistics per experiment
    L += ["### Monthly PR-AUC Statistics", ""]
    L += [
        "| Experiment | Mean | Min | 2024-04 | 2024-04 악화? |",
        "|-----------|------|-----|---------|-------------|",
    ]
    se_apr = se_base.get("monthly_pr_auc", {}).get("2024-04")
    for tag in tags_available:
        monthly = results[tag].get("monthly_pr_auc", {})
        vals = [v for v in monthly.values() if v is not None]
        mean_v = np.mean(vals) if vals else None
        min_v  = np.min(vals)  if vals else None
        apr_v  = monthly.get("2024-04")
        if tag == "A":
            apr_flag = "—"
        else:
            apr_flag = "✓ 악화" if (apr_v is not None and se_apr is not None and apr_v < se_apr) else "—"
        L.append(
            f"| {TAG_LABELS[tag]} | {fmt(mean_v)} | {fmt(min_v)} | {fmt(apr_v)} | {apr_flag} |"
        )
    L.append("")

    # 5. Feature importance top-5 per experiment
    L += ["## 5. Feature Importance Top-5 per Experiment (Gain)", ""]
    for tag in tags_available:
        top5 = results[tag].get("top5_features", [])
        L.append(f"**{TAG_LABELS[tag]}**")
        L.append("")
        if top5:
            L.append("| Rank | Feature | Gain | Derived? |")
            L.append("|------|---------|------|---------|")
            for i, ft in enumerate(top5, 1):
                is_derived = "✓" if ft["feature"].startswith("tf_") else ""
                L.append(f"| {i} | {ft['feature']} | {ft['gain']:.1f} | {is_derived} |")
        else:
            L.append("(없음)")
        L.append("")

    # 6. Group-level verdict
    L += ["## 6. Group-level Analysis", ""]
    L += [
        "### 어떤 group이 P@top1%를 올렸는가?",
        "",
    ]
    se_p1 = se_base.get("p_at_top1pct", 0)
    improved_p1 = [TAG_LABELS[t] for t in "BCDEF" if t in results
                   and results[t].get("p_at_top1pct", 0) > se_p1]
    L.append("개선 그룹: " + (", ".join(improved_p1) if improved_p1 else "없음"))
    L.append("")

    L += ["### 어떤 group이 P@top5%를 악화시켰는가?", ""]
    se_p5 = se_base.get("p_at_top5pct", 0)
    degraded_p5 = [TAG_LABELS[t] for t in "BCDEF" if t in results
                   and results[t].get("p_at_top5pct", 0) < se_p5]
    L.append("악화 그룹: " + (", ".join(degraded_p5) if degraded_p5 else "없음"))
    L.append("")

    L += ["### 어떤 group이 R@P≥0.10을 악화시켰는가?", ""]
    se_rp = se_base.get("recall_at_precision_010", 0)
    degraded_rp = [TAG_LABELS[t] for t in "BCDEF" if t in results
                   and results[t].get("recall_at_precision_010", 0) < se_rp]
    L.append("악화 그룹: " + (", ".join(degraded_rp) if degraded_rp else "없음"))
    L.append("")

    L += ["### 2024-04 PR-AUC가 악화된 group", ""]
    if se_apr is not None:
        apr_degraded = [TAG_LABELS[t] for t in "BCDEF" if t in results
                        and (results[t].get("monthly_pr_auc", {}).get("2024-04") or 0) < se_apr]
        L.append("2024-04 악화 그룹: " + (", ".join(apr_degraded) if apr_degraded else "없음"))
    else:
        L.append("(SE 2024-04 baseline 없음)")
    L.append("")

    L += ["### 월별 PR-AUC 안정성 (분산 기준)", ""]
    se_monthly_vals = [v for v in se_base.get("monthly_pr_auc", {}).values() if v is not None]
    se_std = np.std(se_monthly_vals) if se_monthly_vals else None
    L += [
        "| Experiment | Monthly std | SE 대비 안정성 |",
        "|-----------|-------------|--------------|",
    ]
    for tag in tags_available:
        monthly = results[tag].get("monthly_pr_auc", {})
        vals = [v for v in monthly.values() if v is not None]
        std_v = np.std(vals) if vals else None
        if tag == "A" or se_std is None or std_v is None:
            stability = "—"
        else:
            stability = "✓ 안정" if std_v <= se_std else "△ 불안정"
        L.append(f"| {TAG_LABELS[tag]} | {fmt(std_v)} | {stability} |")
    L.append("")

    # 7. Final verdict
    L += ["## 7. Final Verdict", ""]

    # Conditional notes per group
    L += ["### Group별 평가", ""]
    for tag in "BCDEF":
        if tag not in results:
            continue
        m = results[tag]
        pr  = m.get("pr_auc", 0)
        p5  = m.get("p_at_top5pct", 0)
        rp  = m.get("recall_at_precision_010", 0)
        ece = m.get("ece", 1)
        se_pr  = se_base.get("pr_auc", 0)
        se_p5  = se_base.get("p_at_top5pct", 0)
        se_rp  = se_base.get("recall_at_precision_010", 0)
        se_ece = se_base.get("ece", 1)

        pr_ok  = pr  >= se_pr
        p5_ok  = p5  >= se_p5
        rp_ok  = rp  >= se_rp
        ece_ok = ece <= se_ece

        if pr_ok and p5_ok and rp_ok and ece_ok:
            verdict = "✅ 전면 채택 가능 (모든 핵심 지표 개선)"
        elif pr_ok and p5_ok:
            verdict = "✅ G 조합 채택 (PR-AUC·P@top5% 모두 개선, 일부 지표 소폭 악화)"
        elif pr_ok and not p5_ok:
            verdict = "⚠️ 조건부 (PR-AUC 개선, P@top5% 악화 → G 조합 제외)"
        elif not pr_ok and p5_ok:
            verdict = "⚠️ 조건부 (P@top5% 유지, PR-AUC 악화 → G 조합 제외)"
        else:
            verdict = "❌ 채택 불가 (PR-AUC·P@top5% 모두 악화)"
        L.append(f"- **{TAG_LABELS[tag]}**: {verdict}")
    L.append("")

    # G verdict
    if g_is_empty:
        L += [
            "### G 최종 판단",
            "",
            "**선택된 group 없음 → 기존 SE 모델 유지**",
            "",
            "보수적 선택 기준(PR-AUC ≥ SE AND P@top5% ≥ SE)을 동시에 만족하는 feature group이 없습니다.",
            "추가 feature 없이 기존 `lightgbm_se.pkl`을 계속 사용하세요.",
            "",
        ]
    else:
        g_pr   = results["G"].get("pr_auc", 0)
        g_p5   = results["G"].get("p_at_top5pct", 0)
        se_pr  = se_base.get("pr_auc", 0)
        se_p5  = se_base.get("p_at_top5pct", 0)
        adopt  = g_pr >= se_pr and g_p5 >= se_p5
        g_label = " + ".join(g_groups_used)
        L += [
            "### G 최종 판단",
            "",
            f"G 조합: **{g_label}**",
            "",
        ]
        if adopt:
            L += [
                f"**채택 권장**: G 모델이 SE 대비 PR-AUC({g_pr:.4f} vs {se_pr:.4f})와 "
                f"P@top5%({g_p5:.4f} vs {se_p5:.4f}) 모두를 유지하거나 개선했습니다.",
                "다음 단계로 `lightgbm_se_time_features_g.pkl`을 후보 모델로 사용하세요.",
            ]
        else:
            L += [
                f"**채택 보류**: G 모델이 조합 후 기대를 충족하지 못했습니다. "
                f"PR-AUC={g_pr:.4f}({'+' if g_pr-se_pr >= 0 else ''}{g_pr-se_pr:.4f}), "
                f"P@top5%={g_p5:.4f}({'+' if g_p5-se_p5 >= 0 else ''}{g_p5-se_p5:.4f}).",
                "기존 SE 모델을 유지하는 것을 권장합니다.",
            ]
    L.append("")

    # Final recommendation
    L += ["### 최종 추천", ""]
    if g_is_empty:
        L.append("→ **기존 SE 유지** (`lightgbm_se.pkl`). 모든 derived group이 P@top5% 또는 PR-AUC를 악화시킵니다.")
    else:
        g_pr  = results["G"].get("pr_auc", 0)
        g_p5  = results["G"].get("p_at_top5pct", 0)
        se_pr = se_base.get("pr_auc", 0)
        se_p5 = se_base.get("p_at_top5pct", 0)
        if g_pr >= se_pr and g_p5 >= se_p5:
            L.append(f"→ **특정 feature group 추가** ({' + '.join(g_groups_used)}). G 모델을 채택하세요.")
        else:
            L.append("→ **기존 SE 유지**. G 조합이 기준을 통과했으나 조합 후 성능이 충분하지 않습니다.")
        L.append("→ SE+TF 전체 모델(54개 동시 투입)은 ECE·P@top5% 악화로 **폐기**.")

    L += [
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

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(sep)
    print("Step 1 — Load data")
    print(sep)
    se_df     = load_se_scores()
    train_raw = pd.read_parquet(TRAIN_PATH)
    val_raw   = pd.read_parquet(VAL_PATH)
    print(f"  train raw: {len(train_raw):,}  val raw: {len(val_raw):,}")

    train_se = merge_se_score(train_raw, se_df, "train")
    val_se   = merge_se_score(val_raw,   se_df, "val")

    # ── 2. Concat → derive → split ────────────────────────────────────────────
    print()
    print(sep)
    print("Step 2 — Concat train+val, generate all derived features, split back")
    print(sep)
    print("  [train+val concat reason] val early rows need train trailing history for")
    print("  rolling z-score and momentum. Only past values used → NOT leakage.")

    train_se["_split"] = "train"
    val_se["_split"]   = "val"
    combined = (pd.concat([train_se, val_se], ignore_index=True)
                  .sort_values([DATE_COL, "country"])
                  .reset_index(drop=True))

    bases    = find_base_features(combined)
    all_cols = []

    combined = add_diff_features(combined, bases, all_cols)
    combined = add_ratio_features(combined, bases, all_cols)
    combined = add_zscore_features(combined, bases, all_cols)
    combined = add_momentum_features(combined, bases, all_cols)
    combined = add_interaction_features(combined, bases, all_cols)

    print(f"  Base features    : {len(bases)}  → {bases}")
    print(f"  Derived features : {len(all_cols)}")

    train_all = combined[combined["_split"] == "train"].drop(columns=["_split"])
    val_all   = combined[combined["_split"] == "val"].drop(columns=["_split"])

    # ── 3. Define feature groups ───────────────────────────────────────────────
    group_prefixes = {
        "B": ("tf_diff",),
        "C": ("tf_ratio",),
        "D": ("tf_z90",),
        "E": ("tf_mom3",),
        "F": ("tf_ix_se",),
    }
    group_cols = {
        tag: [c for c in all_cols if any(c.startswith(p) for p in pfx)]
        for tag, pfx in group_prefixes.items()
    }
    for tag, cols in group_cols.items():
        print(f"  Group {tag}: {len(cols)} features")

    # SE base feature columns (no derived)
    se_feature_cols = get_feature_columns(train_all)
    # Remove any derived cols from the SE base list
    se_feature_cols = [c for c in se_feature_cols if c not in all_cols]

    # ── 4. Load SE baseline metrics ───────────────────────────────────────────
    print()
    print(sep)
    print("Step 3 — Load SE baseline metrics")
    print(sep)
    if os.path.exists(SE_DIAG_PATH):
        with open(SE_DIAG_PATH) as f:
            diag = json.load(f)
        se_base = {}
        se_base.update(diag.get("core_metrics", {}))
        for k, v in diag.get("p_at_top_k", {}).items():
            se_base[f"p_at_{k}"] = v
        for k, v in diag.get("recall_at_top_k", {}).items():
            se_base[f"recall_at_{k}"] = v
        for k, v in diag.get("recall_at_precision", {}).items():
            mapping = {"p_geq_010": "recall_at_precision_010",
                       "p_geq_020": "recall_at_precision_020",
                       "p_geq_030": "recall_at_precision_030"}
            se_base[mapping.get(k, k)] = v
        se_base["monthly_pr_auc"] = diag.get("monthly_pr_auc", {})
        print(f"  SE baseline loaded from {SE_DIAG_PATH}")
        print(f"  SE PR-AUC={se_base['pr_auc']:.4f}  P@top5%={se_base['p_at_top5pct']:.4f}")
    else:
        print(f"  WARNING: {SE_DIAG_PATH} not found. Run diagnose_lightgbm_se.py first.")
        se_base = {}

    # ── 5. Run experiments A–F ─────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 4 — Run experiments A–F")
    print(sep)

    results = {}

    # A: SE baseline (re-train for apples-to-apples best_iteration)
    print("\n[A] SE baseline")
    results["A"], _ = train_and_eval(train_all, val_all, se_feature_cols, "A")

    # B-F: SE base + one group each
    for tag in "BCDEF":
        feat_cols = se_feature_cols + group_cols[tag]
        print(f"\n[{tag}] {list(group_prefixes[tag])}")
        results[tag], _ = train_and_eval(train_all, val_all, feat_cols, tag)

    # ── 6. Conservative G selection ───────────────────────────────────────────
    print()
    print(sep)
    print("Step 5 — Conservative G selection")
    print(sep)
    print("  Criterion: PR-AUC >= SE baseline  AND  P@top5% >= SE baseline")
    print("  (P@top5% < baseline → excluded regardless of PR-AUC)")

    se_pr = se_base.get("pr_auc", results["A"]["pr_auc"])
    se_p5 = se_base.get("p_at_top5pct", results["A"]["p_at_top5pct"])

    g_groups = []
    for tag in "BCDEF":
        m = results[tag]
        pr_ok = m["pr_auc"]        >= se_pr
        p5_ok = m["p_at_top5pct"]  >= se_p5
        flag  = "✅ 채택" if (pr_ok and p5_ok) else "❌ 제외"
        print(f"  Group {tag}: PR-AUC={m['pr_auc']:.4f}(>={se_pr:.4f}? {pr_ok})  "
              f"P@top5%={m['p_at_top5pct']:.4f}(>={se_p5:.4f}? {p5_ok})  {flag}")
        if pr_ok and p5_ok:
            g_groups.append(tag)

    g_groups_used = [f"Group {t}" for t in g_groups]
    g_is_empty = len(g_groups) == 0

    if g_is_empty:
        print("\n  → No groups pass. G = 기존 SE 유지 (학습 생략)")
        results["G"] = {**results["A"]}  # copy A metrics to represent G=SE
        results["G"]["_g_note"] = "no_groups_selected"
    else:
        g_feat_cols = se_feature_cols + [c for t in g_groups for c in group_cols[t]]
        print(f"\n  → Selected: {g_groups}  ({len(g_feat_cols)} total features)")
        print("\n[G] Best groups combined")
        results["G"], g_model = train_and_eval(train_all, val_all, g_feat_cols, "G")
        # Save G model artifact
        g_model_path = os.path.join(MODEL_DIR, "lightgbm_se_time_features_g.pkl")
        X_tr, y_tr = make_xy(train_all, g_feat_cols)
        artifact = {
            "model":              g_model,
            "feature_cols":       g_feat_cols,
            "country_categories": list(X_tr["country"].cat.categories),
            "g_groups":           g_groups,
        }
        joblib.dump(artifact, g_model_path)
        print(f"  G model saved: {g_model_path}")

    # ── 7. Save outputs ────────────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 6 — Save outputs")
    print(sep)

    today = dt_date.today().isoformat()

    # JSON
    out_json = {
        "se_baseline":   se_base,
        "g_groups_used": g_groups_used,
        "g_is_empty":    g_is_empty,
        "experiments":   results,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print(f"  Saved JSON : {JSON_PATH}")

    # Summary CSV
    TAG_LABELS = {
        "A": "SE baseline", "B": "SE+diff", "C": "SE+ratio",
        "D": "SE+z-score", "E": "SE+momentum", "F": "SE+interaction", "G": "SE+best_G",
    }
    csv_rows = []
    for tag in [t for t in "ABCDEFG" if t in results]:
        m = results[tag]
        monthly_vals = [v for v in m.get("monthly_pr_auc", {}).values() if v is not None]
        csv_rows.append({
            "experiment":     TAG_LABELS[tag],
            "n_features":     m.get("n_features", ""),
            "pr_auc":         round(m.get("pr_auc", float("nan")), 4),
            "p_at_top1pct":   round(m.get("p_at_top1pct", float("nan")), 4),
            "p_at_top5pct":   round(m.get("p_at_top5pct", float("nan")), 4),
            "recall_at_p010": round(m.get("recall_at_precision_010", float("nan")), 4),
            "ece":            round(m.get("ece", float("nan")), 4),
            "monthly_mean":   round(float(np.mean(monthly_vals)), 4) if monthly_vals else "",
            "monthly_min":    round(float(np.min(monthly_vals)), 4) if monthly_vals else "",
            "apr_pr_auc":     round(m.get("monthly_pr_auc", {}).get("2024-04", float("nan")), 4),
            "best_iteration": m.get("best_iteration", ""),
        })
    pd.DataFrame(csv_rows).to_csv(CSV_PATH, index=False)
    print(f"  Saved CSV  : {CSV_PATH}")

    # Markdown
    md_text = generate_md(results, se_base, g_groups_used, g_is_empty, today)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"  Saved MD   : {MD_PATH}")

    print()
    print("Done.")
    print(sep)


if __name__ == "__main__":
    main()
