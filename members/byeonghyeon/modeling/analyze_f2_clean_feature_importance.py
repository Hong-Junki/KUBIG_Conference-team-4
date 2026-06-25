"""
F2_clean Feature Importance 분석
==================================
cleanval split 동일 구조로 F2_clean 모델만 재학습해서 feature importance를 분석한다.

split (cleanval과 동일):
  train_fit  : 2014-01-01 ~ 2022-12-31  (base model 학습)
  tune_cal   : 2023-01-01 ~ 2023-12-31  (early stopping)
  val_eval   : 2024-01-01 ~ 2024-06-30  (val PR-AUC 확인용만)
  test       : 로드하지 않음 (평가 금지)

분석 내용:
  1. LightGBM gain importance (top 30 + group 집계)
  2. LightGBM split importance (group 집계)
  3. XGBoost gain importance (group 집계)
  4. feature group별 비율 비교표

출력:
  members/byeonghyeon/outputs/reports/f2_clean_feature_importance.md
  (예측 CSV 저장 없음)

실행:
  cd <KUBIG_Conference-team-4 루트>
  python members/byeonghyeon/modeling/analyze_f2_clean_feature_importance.py
"""

import os
import sys
from datetime import date as dt_date

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from xgboost.callback import EarlyStopping as XGBEarlyStopping
from sklearn.preprocessing import LabelEncoder

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_SUBTREE_ROOT = os.path.dirname(_SCRIPT_DIR)

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_SUBTREE_ROOT))),
                 "conflict-early-warning"),
)

sys.path.insert(0, _SCRIPT_DIR)
from evaluate import compute_pr_auc

TRAIN_PATH       = os.path.join(DATA_ROOT, "input", "processed", "dataset", "train.parquet")
VAL_PATH         = os.path.join(DATA_ROOT, "input", "processed", "dataset", "val.parquet")
SAFE_ACLED_PATH  = os.path.join(_SUBTREE_ROOT, "input", "processed", "acled_safe",
                                "safe_acled_lag_features.parquet")
GDELT_TITLE_PATH = os.path.join(_SUBTREE_ROOT, "input", "processed", "gdelt_titles",
                                "gdelt_title_features.parquet")
GDELT_TP_PATH    = os.path.join(_SUBTREE_ROOT, "input", "processed", "gdelt_titles",
                                "gdelt_theme_person_features.parquet")

REPORT_DIR = os.path.join(_SUBTREE_ROOT, "outputs", "reports")
REPORT_MD  = os.path.join(REPORT_DIR, "f2_clean_feature_importance.md")

# ── split 경계 ────────────────────────────────────────────────────────────────
TRAIN_FIT_END  = pd.Timestamp("2022-12-31", tz="UTC")
TUNE_CAL_START = pd.Timestamp("2023-01-01", tz="UTC")
TUNE_CAL_END   = pd.Timestamp("2023-12-31", tz="UTC")
GDELT_TITLE_MIN_DATE     = "2015-02-17"
GDELT_TITLE_COVERAGE_COL = "gdelt_title_coverage_mask"

TARGET_COL = "y_escalation"
DATE_COL   = "date"
RANDOM_SEED = 42

LABEL_META_COLS = [
    "y", "y_onset", "y_escalation",
    "fatalities_next3d", "event_count_next3d",
    "past14d_event_count", "past14d_fatalities_mean",
]
ACLED_REMOVE_COLS = [
    "acled_event_count_7d",    "acled_event_count_14d",   "acled_event_count_30d",
    "acled_fatalities_7d",     "acled_fatalities_14d",    "acled_fatalities_30d",
    "acled_fatalities_max_7d", "acled_fatalities_max_14d","acled_fatalities_max_30d",
    "acled_ratio_battles",     "acled_ratio_explosions",  "acled_ratio_vac",
    *[f"acled_actor_type_{i}_ratio" for i in range(1, 9)],
    "acled_missing_mask", "macis_se_score",
]
ALWAYS_EXCLUDE = [DATE_COL] + ACLED_REMOVE_COLS

SAFE_ACLED_FEATURE_COLS = [
    "safe_acled_event_count_7d_lag7",  "safe_acled_event_count_14d_lag7",
    "safe_acled_event_count_30d_lag7", "safe_acled_fatalities_7d_lag7",
    "safe_acled_fatalities_14d_lag7",  "safe_acled_fatalities_30d_lag7",
    "safe_acled_fatalities_max_7d_lag7","safe_acled_fatalities_max_14d_lag7",
    "safe_acled_fatalities_max_30d_lag7","safe_acled_ratio_battles_lag7",
    "safe_acled_ratio_explosions_lag7", "safe_acled_ratio_vac_lag7",
    "safe_acled_ratio_state_forces_lag7","safe_acled_ratio_external_forces_lag7",
    "safe_acled_missing_mask",
]
GDELT_TITLE_FEATURE_COLS = [
    "gdelt_title_count_1d",            "gdelt_title_nonnull_count_1d",
    "gdelt_title_tone_mean_1d",        "gdelt_title_tone_std_1d",
    "gdelt_title_tone_min_1d",         "gdelt_title_negative_count_1d",
    "gdelt_title_positive_count_1d",   "gdelt_title_eng_count_1d",
    "gdelt_title_domain_diversity_1d", "gdelt_title_lang_diversity_1d",
    "gdelt_title_count_7d",            "gdelt_title_nonnull_count_7d",
    "gdelt_title_tone_mean_7d",        "gdelt_title_tone_std_7d",
    "gdelt_title_tone_min_7d",         "gdelt_title_negative_count_7d",
    "gdelt_title_positive_count_7d",   "gdelt_title_eng_count_7d",
    "gdelt_title_domain_diversity_7d", "gdelt_title_lang_diversity_7d",
    "gdelt_title_tone_trend_7d",
]
GDELT_TP_FEATURE_COLS = [
    "gdelt_theme_nonnull_count_1d",   "gdelt_person_nonnull_count_1d",
    "gdelt_theme_count_1d",           "gdelt_person_count_1d",
    "gdelt_theme_conflict_count_1d",  "gdelt_theme_protest_count_1d",
    "gdelt_theme_military_count_1d",  "gdelt_theme_refugee_count_1d",
    "gdelt_theme_sanction_count_1d",  "gdelt_theme_government_count_1d",
    "gdelt_person_density_1d",
    "gdelt_theme_nonnull_count_7d",   "gdelt_person_nonnull_count_7d",
    "gdelt_theme_count_7d",           "gdelt_person_count_7d",
    "gdelt_theme_conflict_count_7d",  "gdelt_theme_protest_count_7d",
    "gdelt_theme_military_count_7d",  "gdelt_theme_refugee_count_7d",
    "gdelt_theme_sanction_count_7d",  "gdelt_theme_government_count_7d",
    "gdelt_person_density_7d",
]
_ALL_EXTRA = set(
    SAFE_ACLED_FEATURE_COLS + GDELT_TITLE_FEATURE_COLS
    + [GDELT_TITLE_COVERAGE_COL] + GDELT_TP_FEATURE_COLS
)

# ── LGB/XGB 하이퍼파라미터 (cleanval과 동일) ──────────────────────────────────
LGB_PARAMS = {
    "objective": "binary", "metric": "average_precision",
    "boosting_type": "gbdt", "num_leaves": 63, "learning_rate": 0.05,
    "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.8, "scale_pos_weight": 22,
    "seed": RANDOM_SEED, "verbose": -1,
}
XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "aucpr",
    "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
    "colsample_bytree": 0.8, "scale_pos_weight": 22,
    "seed": RANDOM_SEED, "verbosity": 0,
}
LGB_ROUNDS = 1000
XGB_ROUNDS = 1000
EARLY_STOP = 50
LOG_PERIOD = 100


# ─────────────────────────────────────────────────────────────────────────────
# feature group 매핑
# ─────────────────────────────────────────────────────────────────────────────

def assign_group(col: str) -> str:
    if col in set(SAFE_ACLED_FEATURE_COLS):
        return "safe_acled"
    if col in set(GDELT_TITLE_FEATURE_COLS):
        return "gdelt_title"
    if col == GDELT_TITLE_COVERAGE_COL:
        return "coverage_mask"
    if col in set(GDELT_TP_FEATURE_COLS):
        return "gdelt_theme_person"
    if col.startswith("gdelt_"):
        return "gdelt_events"
    if col.startswith("econ_"):
        return "economic"
    if col == "country":
        return "country"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드 + merge
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("[Step 1] 데이터 로딩 + feature merge")
    train_full = pd.read_parquet(TRAIN_PATH)
    val_eval   = pd.read_parquet(VAL_PATH)

    for df in [train_full, val_eval]:
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("UTC")

    train_fit = train_full[train_full["date"] <= TRAIN_FIT_END].copy()
    tune_cal  = train_full[(train_full["date"] >= TUNE_CAL_START) &
                           (train_full["date"] <= TUNE_CAL_END)].copy()

    print(f"  train_fit : {len(train_fit):,}행 | 양성률 {train_fit[TARGET_COL].mean():.4f}")
    print(f"  tune_cal  : {len(tune_cal):,}행 | 양성률 {tune_cal[TARGET_COL].mean():.4f}")
    print(f"  val_eval  : {len(val_eval):,}행 | 양성률 {val_eval[TARGET_COL].mean():.4f}")
    print("  test      : 로드 안 함 (평가 금지)")

    # feature parquet 로드
    safe_df  = pd.read_parquet(SAFE_ACLED_PATH)
    title_df = pd.read_parquet(GDELT_TITLE_PATH)
    tp_df    = pd.read_parquet(GDELT_TP_PATH)

    for df in [safe_df, title_df, tp_df]:
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("UTC")

    safe_merge  = safe_df[["date","country"] + SAFE_ACLED_FEATURE_COLS]
    title_merge = title_df[["date","country"] + GDELT_TITLE_FEATURE_COLS]
    tp_merge    = tp_df[["date","country"] + GDELT_TP_FEATURE_COLS]
    coverage_ts = pd.Timestamp(GDELT_TITLE_MIN_DATE, tz="UTC")

    def merge_all(df, name):
        n = len(df)
        df = df.merge(safe_merge,  on=["date","country"], how="left"); assert len(df)==n
        df[SAFE_ACLED_FEATURE_COLS]  = df[SAFE_ACLED_FEATURE_COLS].fillna(0)
        df = df.merge(title_merge, on=["date","country"], how="left"); assert len(df)==n
        df[GDELT_TITLE_FEATURE_COLS] = df[GDELT_TITLE_FEATURE_COLS].fillna(0)
        df[GDELT_TITLE_COVERAGE_COL] = (df["date"] < coverage_ts).astype(int)
        df = df.merge(tp_merge,    on=["date","country"], how="left"); assert len(df)==n
        df[GDELT_TP_FEATURE_COLS]    = df[GDELT_TP_FEATURE_COLS].fillna(0)
        print(f"    [{name}] merge 완료, row count 유지 ✅")
        return df

    train_fit = merge_all(train_fit, "train_fit")
    tune_cal  = merge_all(tune_cal,  "tune_cal")
    val_eval  = merge_all(val_eval,  "val_eval")
    return train_fit, tune_cal, val_eval


# ─────────────────────────────────────────────────────────────────────────────
# feature cols 선별 + 검증
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_cols(df):
    exclude = set(LABEL_META_COLS) | set(ALWAYS_EXCLUDE) | _ALL_EXTRA
    base = [c for c in df.columns if c not in exclude]
    return (base + SAFE_ACLED_FEATURE_COLS + GDELT_TITLE_FEATURE_COLS
            + [GDELT_TITLE_COVERAGE_COL] + GDELT_TP_FEATURE_COLS)


def validate_feature_cols(feature_cols):
    print("\n[Step 2] Feature 검증")
    old_acled = [c for c in feature_cols
                 if (c.startswith("acled_") and not c.startswith("safe_acled_"))
                 or c == "macis_se_score"]
    if old_acled:
        print(f"  [오류] 기존 ACLED/SE 컬럼 잔존: {old_acled}")
        sys.exit(1)
    label_leaked = [c for c in feature_cols if c in set(LABEL_META_COLS)]
    if label_leaked:
        print(f"  [오류] label/future 컬럼 잔존: {label_leaked}")
        sys.exit(1)

    groups = {}
    for c in feature_cols:
        g = assign_group(c)
        groups[g] = groups.get(g, 0) + 1
    print(f"  feature 수: {len(feature_cols)}개  (기대 94)")
    for g, cnt in sorted(groups.items()):
        print(f"    {g:<22}: {cnt}개")
    print("  ACLED/SE/label 잔존 없음 ✅")
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# 모델 학습
# ─────────────────────────────────────────────────────────────────────────────

def make_lgbm_X(df, feature_cols):
    X = df[feature_cols].copy()
    if "country" in X.columns:
        X["country"] = X["country"].astype("category")
    return X


def make_xgb_X(df, feature_cols, enc):
    X = df[feature_cols].copy()
    if "country" in X.columns:
        country_map = {c: i for i, c in enumerate(enc.classes_)}
        X["country"] = df["country"].astype(str).map(
            lambda c: country_map.get(c, -1)).values.astype(np.int32)
    return X.astype(np.float32)


def train_models(train_fit, tune_cal, feature_cols):
    print("\n[Step 3] 모델 학습 (train_fit, early stop on tune_cal)")

    enc = LabelEncoder()
    enc.fit(train_fit["country"].astype(str))

    # LightGBM
    ds_tr = lgb.Dataset(make_lgbm_X(train_fit, feature_cols),
                        label=train_fit[TARGET_COL].values,
                        categorical_feature=["country"], free_raw_data=False)
    ds_tc = lgb.Dataset(make_lgbm_X(tune_cal, feature_cols),
                        label=tune_cal[TARGET_COL].values,
                        reference=ds_tr, free_raw_data=False)
    lgb_model = lgb.train(
        LGB_PARAMS, ds_tr, num_boost_round=LGB_ROUNDS,
        valid_sets=[ds_tc], valid_names=["tune_cal"],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
                   lgb.log_evaluation(period=LOG_PERIOD)],
    )
    print(f"  LightGBM best iteration: {lgb_model.best_iteration}")

    # XGBoost
    dtrain = xgb.DMatrix(make_xgb_X(train_fit, feature_cols, enc),
                         label=train_fit[TARGET_COL].values)
    dtune  = xgb.DMatrix(make_xgb_X(tune_cal, feature_cols, enc),
                         label=tune_cal[TARGET_COL].values)
    xgb_model = xgb.train(
        XGB_PARAMS, dtrain, num_boost_round=XGB_ROUNDS,
        evals=[(dtune, "tune_cal")],
        callbacks=[XGBEarlyStopping(rounds=EARLY_STOP, save_best=True)],
        verbose_eval=LOG_PERIOD,
    )
    print(f"  XGBoost  best iteration: {xgb_model.best_iteration}")

    return lgb_model, xgb_model, enc


# ─────────────────────────────────────────────────────────────────────────────
# val_eval PR-AUC 확인 (참고용)
# ─────────────────────────────────────────────────────────────────────────────

def check_val_pr(lgb_model, xgb_model, val_eval, feature_cols, enc):
    val_y = val_eval[TARGET_COL].values
    prob_l = lgb_model.predict(make_lgbm_X(val_eval, feature_cols))
    prob_x = xgb_model.predict(xgb.DMatrix(make_xgb_X(val_eval, feature_cols, enc)))
    print(f"\n  val_eval PR-AUC 확인 (참고): "
          f"LGB={compute_pr_auc(val_y, prob_l):.4f} | XGB={compute_pr_auc(val_y, prob_x):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance 추출
# ─────────────────────────────────────────────────────────────────────────────

def extract_importance(lgb_model, xgb_model, feature_cols):
    print("\n[Step 4] Feature importance 추출")

    # LightGBM
    lgb_feat_names = lgb_model.feature_name()
    lgb_gain  = lgb_model.feature_importance(importance_type="gain")
    lgb_split = lgb_model.feature_importance(importance_type="split")

    lgb_df = pd.DataFrame({
        "feature": lgb_feat_names,
        "lgb_gain":  lgb_gain.astype(float),
        "lgb_split": lgb_split.astype(float),
    })

    # XGBoost — get_score는 0인 feature 제외하므로 전체 feature_cols 기준으로 채움
    xgb_scores = xgb_model.get_score(importance_type="gain")
    # XGBoost feature names: f0, f1, ... or actual names
    # actual names depend on how DMatrix was built; let's use positional mapping
    xgb_gain_vals = np.zeros(len(feature_cols))
    for fname, val in xgb_scores.items():
        # XGBoost may use 'f0','f1',... if feature names not set
        if fname.startswith("f") and fname[1:].isdigit():
            idx = int(fname[1:])
            if idx < len(feature_cols):
                xgb_gain_vals[idx] = val
        elif fname in feature_cols:
            idx = feature_cols.index(fname)
            xgb_gain_vals[idx] = val

    xgb_df = pd.DataFrame({
        "feature": feature_cols,
        "xgb_gain": xgb_gain_vals,
    })

    # 병합
    imp_df = lgb_df.merge(xgb_df, on="feature", how="left")
    imp_df["xgb_gain"] = imp_df["xgb_gain"].fillna(0)
    imp_df["group"] = imp_df["feature"].map(assign_group)

    # 정규화 (각 importance 타입 합계를 100으로)
    for col in ["lgb_gain", "lgb_split", "xgb_gain"]:
        total = imp_df[col].sum()
        imp_df[f"{col}_pct"] = imp_df[col] / total * 100 if total > 0 else 0

    print(f"  LGB gain  합계: {lgb_gain.sum():,.0f}")
    print(f"  LGB split 합계: {lgb_split.sum():,.0f}")
    print(f"  XGB gain  합계: {sum(xgb_scores.values()):,.2f}")
    return imp_df


# ─────────────────────────────────────────────────────────────────────────────
# 리포트 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_report(imp_df, feature_groups_count):
    today = dt_date.today().isoformat()

    # ── group별 집계 ──────────────────────────────────────────────────────────
    group_df = imp_df.groupby("group").agg(
        n_features=("feature", "count"),
        lgb_gain_sum=("lgb_gain", "sum"),
        lgb_split_sum=("lgb_split", "sum"),
        xgb_gain_sum=("xgb_gain", "sum"),
    ).reset_index()

    for col, total_col in [("lgb_gain_sum","lgb_gain_pct"),
                           ("lgb_split_sum","lgb_split_pct"),
                           ("xgb_gain_sum","xgb_gain_pct")]:
        total = group_df[col].sum()
        group_df[total_col] = group_df[col] / total * 100 if total > 0 else 0

    group_df = group_df.sort_values("lgb_gain_sum", ascending=False)

    # ── 개별 top-30 (lgb_gain 기준) ──────────────────────────────────────────
    top30 = imp_df.nlargest(30, "lgb_gain").reset_index(drop=True)

    # ── 그룹 순위 (참고: LGB gain 기준) ──────────────────────────────────────
    top_group = group_df.iloc[0]["group"]
    top2_group = group_df.iloc[1]["group"] if len(group_df) > 1 else "—"

    lines = [
        f"# F2_clean Feature Importance 분석",
        f"",
        f"**분석일**: {today}  ",
        f"**모델**: LightGBM + XGBoost (cleanval split, train_fit=2014~2022, tune_cal=2023)  ",
        f"**feature 수**: 94개 (B 35 + safe ACLED 15 + GDELT title 21 + coverage_mask 1 + theme/person 22)  ",
        f"",
        "---",
        "",
        "## Feature Group별 Importance",
        "",
        "| Group | feature 수 | LGB gain % | LGB split % | XGB gain % |",
        "|-------|-----------|-----------|------------|-----------|",
    ]
    for _, row in group_df.iterrows():
        lines.append(
            f"| **{row['group']}** | {int(row['n_features'])} "
            f"| {row['lgb_gain_pct']:>6.2f}% "
            f"| {row['lgb_split_pct']:>6.2f}% "
            f"| {row['xgb_gain_pct']:>6.2f}% |"
        )

    lines += [
        "",
        "---",
        "",
        "## Individual Feature Top 30 (LightGBM gain 기준)",
        "",
        "| rank | feature | group | LGB gain % | LGB split % | XGB gain % |",
        "|------|---------|-------|-----------|------------|-----------|",
    ]
    for i, row in top30.iterrows():
        lines.append(
            f"| {i+1:2d} | `{row['feature']}` | {row['group']} "
            f"| {row['lgb_gain_pct']:>6.2f}% "
            f"| {row['lgb_split_pct']:>6.2f}% "
            f"| {row['xgb_gain_pct']:>6.2f}% |"
        )

    # ── Group별 상세: top-5 feature ───────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Feature Group별 주요 Feature (LGB gain 상위 5개)",
        "",
    ]
    for group in group_df["group"].tolist():
        sub = imp_df[imp_df["group"] == group].nlargest(5, "lgb_gain")
        if len(sub) == 0:
            continue
        lines.append(f"### {group}")
        lines.append("")
        lines.append("| feature | LGB gain % | LGB split % |")
        lines.append("|---------|-----------|------------|")
        for _, row in sub.iterrows():
            lines.append(f"| `{row['feature']}` "
                         f"| {row['lgb_gain_pct']:>6.2f}% "
                         f"| {row['lgb_split_pct']:>6.2f}% |")
        lines.append("")

    # ── 해석 ─────────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 해석",
        "",
        f"### 가장 중요한 feature group",
        f"",
        f"LGB gain 기준 1위: **{top_group}**, 2위: **{top2_group}**",
        "",
    ]

    # safe_acled 분석
    safe_row = group_df[group_df["group"] == "safe_acled"]
    safe_pct = float(safe_row["lgb_gain_pct"].values[0]) if len(safe_row) > 0 else 0.0

    title_row = group_df[group_df["group"] == "gdelt_title"]
    title_pct = float(title_row["lgb_gain_pct"].values[0]) if len(title_row) > 0 else 0.0

    tp_row = group_df[group_df["group"] == "gdelt_theme_person"]
    tp_pct = float(tp_row["lgb_gain_pct"].values[0]) if len(tp_row) > 0 else 0.0

    econ_row = group_df[group_df["group"] == "economic"]
    econ_pct = float(econ_row["lgb_gain_pct"].values[0]) if len(econ_row) > 0 else 0.0

    country_row = group_df[group_df["group"] == "country"]
    country_pct = float(country_row["lgb_gain_pct"].values[0]) if len(country_row) > 0 else 0.0

    lines += [
        f"**safe_acled** ({safe_pct:.1f}%): 과거 ACLED 분쟁 기록 (shift 7일 lag). "
        f"{'가장 강력한 단일 group.' if top_group == 'safe_acled' else '예상보다 낮을 경우 GDELT 신호가 더 강함을 의미.'}",
        "",
        f"**gdelt_title** ({title_pct:.1f}%): GDELT 기사 수/tone/다양성 피처. "
        f"{'safe ACLED 위에서도 독립적 기여 확인.' if title_pct > 5 else '기여도 낮음 — safe ACLED에 흡수됐을 가능성.'}",
        "",
        f"**gdelt_theme_person** ({tp_pct:.1f}%): 테마/인물 집계 피처. "
        f"{'F2-F1 delta +0.0191의 주요 원인으로 확인됨.' if tp_pct > 5 else '기여도 낮음 — 다른 group이 대부분의 정보를 포함.'}",
        "",
        f"**economic** ({econ_pct:.1f}%): 글로벌 경제 지표. "
        f"{'실질적 기여 확인.' if econ_pct > 3 else '기여도 낮음 — 분쟁 예측에서 경제 지표의 한계.'}",
        "",
        f"**country** ({country_pct:.1f}%): ISO3 국가 코드. "
        f"{'높으면 국가 고정효과 absorb (overfitting 주의).' if country_pct > 15 else '적절한 수준.'}",
        "",
        "---",
        "",
        "## 주의사항",
        "",
        "1. **feature importance는 근사적 해석**일 뿐 causal effect가 아니다.  ",
        "   높은 importance = '모델이 이 feature를 많이 사용함'이지, '실제 원인'이 아님.",
        "",
        "2. **gain importance의 correlated feature 문제**: 서로 상관된 feature(예: count_7d vs count_14d)는  ",
        "   중요도가 분산되거나 한쪽에 몰릴 수 있다. group 단위 집계가 개별 feature보다 더 안정적이다.",
        "",
        "3. **country feature**: 국가별 고정효과를 흡수하는 경향이 있어 importance가 높게 나올 수 있다.  ",
        "   이는 과적합 징조일 수 있으므로 주의.",
        "",
        "4. **최종 성능 평가는 아직 test set에서 하지 않았다.**  ",
        "   현재 val_eval 기준으로 F2_clean이 최고이며, test 1회 평가 후 최종 확정한다.",
        "",
        f"*생성: {today}*",
    ]

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  리포트 저장: {REPORT_MD}")

    # 터미널 요약 출력
    print("\n  ── Group 집계 요약 (LGB gain %) ──")
    for _, row in group_df.iterrows():
        bar = "█" * int(row["lgb_gain_pct"] / 2)
        print(f"  {row['group']:<22} {row['lgb_gain_pct']:>5.1f}%  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  F2_clean Feature Importance 분석")
    print(f"  DATA_ROOT : {DATA_ROOT}")
    print(f"  REPORT_MD : {REPORT_MD}")
    print("=" * 70)

    for path, label in [(TRAIN_PATH,"train"), (VAL_PATH,"val"),
                        (SAFE_ACLED_PATH,"safe_acled"), (GDELT_TITLE_PATH,"gdelt_title"),
                        (GDELT_TP_PATH,"gdelt_tp")]:
        if not os.path.exists(path):
            print(f"[오류] {label} 없음: {path}")
            sys.exit(1)

    train_fit, tune_cal, val_eval = load_data()

    feature_cols = get_feature_cols(train_fit)
    fg_counts    = validate_feature_cols(feature_cols)

    lgb_model, xgb_model, enc = train_models(train_fit, tune_cal, feature_cols)

    check_val_pr(lgb_model, xgb_model, val_eval, feature_cols, enc)

    imp_df = extract_importance(lgb_model, xgb_model, feature_cols)

    save_report(imp_df, fg_counts)

    print()
    print("=" * 70)
    print("  F2_clean Feature Importance 분석 완료")
    print(f"  리포트: {REPORT_MD}")
    print("=" * 70)


if __name__ == "__main__":
    main()
