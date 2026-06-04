"""
Clean Validation Comparison: F0_clean / F1_clean / F2_clean
============================================================
기존 F0/F1 스크립트에서 val set이 early stopping, meta C 선택, calibration fit,
평가에 중복 사용되던 문제를 해결한다.

핵심 split 설계 (기존 train parquet를 시간 기준으로 재분할):
  train_fit  : 2014-01-01 ~ 2022-12-31  → base model 학습 + OOF 생성
  tune_cal   : 2023-01-01 ~ 2023-12-31  → early stopping, meta C 선택, calibration fit
  val_eval   : 2024-01-01 ~ 2024-06-30  → 순수 평가 전용 (어떤 학습에도 사용 안 함)
  test       : 2024-07-01 ~ (평가 미수행)

val_eval 역할 보장:
  - LightGBM/XGBoost early stopping: tune_cal ✅
  - Meta LogReg C 선택:              tune_cal ✅
  - Platt/Isotonic calibration fit:  tune_cal ✅
  - 최종 PR-AUC/ECE/Brier 평가:      val_eval ✅ (fit 없음)

비교 feature set:
  F0_clean (50개):  B 35 + safe ACLED 15
  F1_clean (72개):  F0 + GDELT title 21 + coverage_mask 1
  F2_clean (94개):  F1 + GDELT theme/person 22

OOF fold 설계 (train_fit 내부, 5 folds):
  fold1: train ≤2017 → predict 2018
  fold2: train ≤2018 → predict 2019
  fold3: train ≤2019 → predict 2020
  fold4: train ≤2020 → predict 2021
  fold5: train ≤2021 → predict 2022

출력:
  members/byeonghyeon/outputs/reports/safe_acled_cleanval_f0_f1_f2_comparison.md
  (예측 CSV 저장 없음)

실행:
  cd <KUBIG_Conference-team-4 루트>
  python members/byeonghyeon/modeling/run_stacking_safe_acled_compare_f0_f1_f2_cleanval.py

  Smoke test (~10분):
  SMOKE_TEST=1 python members/byeonghyeon/modeling/run_stacking_safe_acled_compare_f0_f1_f2_cleanval.py
"""

import os
import sys
from datetime import date as dt_date

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from xgboost.callback import EarlyStopping as XGBEarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_SUBTREE_ROOT = os.path.dirname(_SCRIPT_DIR)

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_SUBTREE_ROOT))),
                 "conflict-early-warning"),
)

sys.path.insert(0, _SCRIPT_DIR)
from evaluate import compute_pr_auc, compute_p_at_top_k, compute_recall_at_precision, compute_ece

# ── Smoke test ────────────────────────────────────────────────────────────────
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

EXPERIMENT_TAG = "cleanval_compare" + ("_smoke" if SMOKE_TEST else "")
RANDOM_SEED    = 42

# ── 입력 경로 ─────────────────────────────────────────────────────────────────
TRAIN_PATH          = os.path.join(DATA_ROOT, "input", "processed", "dataset", "train.parquet")
VAL_PATH            = os.path.join(DATA_ROOT, "input", "processed", "dataset", "val.parquet")
TEST_PATH           = os.path.join(DATA_ROOT, "input", "processed", "dataset", "test.parquet")
SAFE_ACLED_PATH     = os.path.join(_SUBTREE_ROOT, "input", "processed", "acled_safe",
                                   "safe_acled_lag_features.parquet")
GDELT_TITLE_PATH    = os.path.join(_SUBTREE_ROOT, "input", "processed", "gdelt_titles",
                                   "gdelt_title_features.parquet")
GDELT_TP_PATH       = os.path.join(_SUBTREE_ROOT, "input", "processed", "gdelt_titles",
                                   "gdelt_theme_person_features.parquet")

# ── 출력 경로 ─────────────────────────────────────────────────────────────────
REPORT_DIR = os.path.join(_SUBTREE_ROOT, "outputs", "reports")
REPORT_MD  = os.path.join(REPORT_DIR, "safe_acled_cleanval_f0_f1_f2_comparison.md")

# ── 날짜 split 경계 ───────────────────────────────────────────────────────────
TRAIN_FIT_END = pd.Timestamp("2022-12-31", tz="UTC")
TUNE_CAL_START = pd.Timestamp("2023-01-01", tz="UTC")
TUNE_CAL_END   = pd.Timestamp("2023-12-31", tz="UTC")

GDELT_TITLE_MIN_DATE     = "2015-02-17"
GDELT_TITLE_COVERAGE_COL = "gdelt_title_coverage_mask"

# ── 컬럼 상수 ─────────────────────────────────────────────────────────────────
TARGET_COL = "y_escalation"
DATE_COL   = "date"

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
    "acled_missing_mask",
    "macis_se_score",
]
ALWAYS_EXCLUDE = [DATE_COL] + ACLED_REMOVE_COLS

# safe ACLED lag feature 15개
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

# GDELT title feature 21개
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

# GDELT theme/person feature 22개
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

# 모든 extra feature col 집합 (base B cols 선별 시 제외용)
_ALL_EXTRA = set(
    SAFE_ACLED_FEATURE_COLS
    + GDELT_TITLE_FEATURE_COLS
    + [GDELT_TITLE_COVERAGE_COL]
    + GDELT_TP_FEATURE_COLS
)

# ── OOF fold 설계 (train_fit 내부 5 folds) ────────────────────────────────────
_ALL_FOLDS = [
    {"name": "OOF_F1", "train_end": 2017, "pred_year": 2018},
    {"name": "OOF_F2", "train_end": 2018, "pred_year": 2019},
    {"name": "OOF_F3", "train_end": 2019, "pred_year": 2020},
    {"name": "OOF_F4", "train_end": 2020, "pred_year": 2021},
    {"name": "OOF_F5", "train_end": 2021, "pred_year": 2022},
]
OOF_FOLDS = _ALL_FOLDS[-1:] if SMOKE_TEST else _ALL_FOLDS

# ── 하이퍼파라미터 ─────────────────────────────────────────────────────────────
LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "average_precision",
    "boosting_type":     "gbdt",
    "num_leaves":        63,
    "learning_rate":     0.05,
    "min_child_samples": 20,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.8,
    "scale_pos_weight":  22,
    "seed":              RANDOM_SEED,
    "verbose":           -1,
}
LGB_ROUNDS_OOF   = 50  if SMOKE_TEST else 500
LGB_ROUNDS_FINAL = 100 if SMOKE_TEST else 1000
EARLY_STOP       = 20  if SMOKE_TEST else 50
LOG_PERIOD       = 50  if SMOKE_TEST else 100

XGB_PARAMS_OOF = {
    "objective":        "binary:logistic",
    "eval_metric":      "aucpr",
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 22,
    "seed":             RANDOM_SEED,
    "verbosity":        0,
}
XGB_PARAMS_FINAL = {**XGB_PARAMS_OOF, "verbosity": 0}
XGB_ROUNDS_OOF   = 50  if SMOKE_TEST else 500
XGB_ROUNDS_FINAL = 100 if SMOKE_TEST else 1000

META_C_CANDIDATES = [0.01, 0.1, 1.0, 10.0]

EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _recall_at_prec(y_true, y_prob, min_prec):
    precision, recall, _ = precision_recall_curve(np.asarray(y_true), np.asarray(y_prob))
    valid = precision >= min_prec
    return float(recall[valid].max()) if valid.any() else 0.0


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        "pr_auc":                  compute_pr_auc(y_true, y_prob),
        "p_at_top5pct":            compute_p_at_top_k(y_true, y_prob, k=0.05),
        "recall_at_precision_010": compute_recall_at_precision(y_true, y_prob, min_precision=0.10),
        "recall_at_precision_020": _recall_at_prec(y_true, y_prob, 0.20),
        "brier_score":             float(np.mean((y_prob - y_true.astype(float)) ** 2)),
        "ece":                     compute_ece(y_true, y_prob),
        "positive_rate":           float(y_true.mean()),
        "n_samples":               int(len(y_true)),
    }


def apply_platt(platt, prob):
    return platt.predict_proba(np.asarray(prob).reshape(-1, 1))[:, 1]


def apply_isotonic(iso, prob):
    return iso.predict(np.asarray(prob))


# ─────────────────────────────────────────────────────────────────────────────
# Step 0 — 경로 검증
# ─────────────────────────────────────────────────────────────────────────────

def validate_setup(run_f2: bool):
    print("=" * 70)
    print(f"  Clean Validation Comparison: {EXPERIMENT_TAG}")
    if SMOKE_TEST:
        print("  *** SMOKE TEST: OOF 1 fold, 라운드 축소 ***")
        print(f"  OOF folds: {[f['name'] for f in OOF_FOLDS]}")
    print(f"  DATA_ROOT      : {DATA_ROOT}")
    print(f"  SAFE_ACLED     : {SAFE_ACLED_PATH}")
    print(f"  GDELT_TITLE    : {GDELT_TITLE_PATH}")
    print(f"  GDELT_TP       : {GDELT_TP_PATH}")
    print(f"  REPORT_MD      : {REPORT_MD}")
    print(f"  Split: train_fit ≤2022-12-31 | tune_cal 2023 | val_eval 2024-H1")
    print(f"  Experiments    : F0_clean(50) / F1_clean(72) / F2_clean(94, F2={'ON' if run_f2 else 'SKIP'})")
    print(f"  채택 기준      : B+0.003=0.0594 (F0), F0+0.003 (F1), F1+0.003 (F2)")
    print("=" * 70)

    if not os.path.isdir(DATA_ROOT):
        print(f"[오류] DATA_ROOT 없음: {DATA_ROOT}")
        sys.exit(1)

    for path, label in [(TRAIN_PATH, "train"), (VAL_PATH, "val"), (TEST_PATH, "test")]:
        if not os.path.exists(path):
            print(f"[오류] {label}.parquet 없음: {path}")
            sys.exit(1)

    for path, label in [(SAFE_ACLED_PATH, "safe_acled"), (GDELT_TITLE_PATH, "gdelt_title")]:
        if not os.path.exists(path):
            print(f"[오류] {label} parquet 없음: {path}")
            sys.exit(1)

    os.makedirs(REPORT_DIR, exist_ok=True)
    print("  → 필수 입력 파일 존재 확인 ✅")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — 데이터 로드 + feature merge
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data(run_f2: bool):
    """모든 split에 모든 feature를 한 번에 merge한다."""
    print("\n[Step 1] 데이터 로딩 + feature merge")

    train_full = pd.read_parquet(TRAIN_PATH)
    val_eval   = pd.read_parquet(VAL_PATH)
    test       = pd.read_parquet(TEST_PATH)

    # 날짜 정규화
    for df in [train_full, val_eval, test]:
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("UTC")

    # train_fit / tune_cal 분리
    train_fit = train_full[train_full["date"] <= TRAIN_FIT_END].copy()
    tune_cal  = train_full[(train_full["date"] >= TUNE_CAL_START) &
                           (train_full["date"] <= TUNE_CAL_END)].copy()

    print(f"  train_fit : {len(train_fit):,}행 | "
          f"{train_fit['date'].min().date()} ~ {train_fit['date'].max().date()} | "
          f"양성률 {train_fit[TARGET_COL].mean():.4f}")
    print(f"  tune_cal  : {len(tune_cal):,}행 | "
          f"{tune_cal['date'].min().date()} ~ {tune_cal['date'].max().date()} | "
          f"양성률 {tune_cal[TARGET_COL].mean():.4f}")
    print(f"  val_eval  : {len(val_eval):,}행 | "
          f"{val_eval['date'].min().date()} ~ {val_eval['date'].max().date()} | "
          f"양성률 {val_eval[TARGET_COL].mean():.4f}")
    print(f"  test      : {len(test):,}행 | "
          f"{test['date'].min().date()} ~ {test['date'].max().date()} (미평가)")

    splits = {"train_fit": train_fit, "tune_cal": tune_cal,
              "val_eval": val_eval, "test": test}

    # safe ACLED merge
    safe_df = pd.read_parquet(SAFE_ACLED_PATH)
    if safe_df["date"].dt.tz is None:
        safe_df["date"] = safe_df["date"].dt.tz_localize("UTC")
    safe_merge = safe_df[["date", "country"] + SAFE_ACLED_FEATURE_COLS]
    print(f"\n  safe ACLED : {len(safe_df):,}행 | {safe_df['country'].nunique()}개국")

    # GDELT title merge
    title_df = pd.read_parquet(GDELT_TITLE_PATH)
    if title_df["date"].dt.tz is None:
        title_df["date"] = title_df["date"].dt.tz_localize("UTC")
    title_merge = title_df[["date", "country"] + GDELT_TITLE_FEATURE_COLS]
    print(f"  GDELT title: {len(title_df):,}행 | {title_df['country'].nunique()}개국")

    # GDELT theme/person merge (F2용, 없으면 skip)
    tp_merge = None
    if run_f2:
        if os.path.exists(GDELT_TP_PATH):
            tp_df = pd.read_parquet(GDELT_TP_PATH)
            if tp_df["date"].dt.tz is None:
                tp_df["date"] = tp_df["date"].dt.tz_localize("UTC")
            tp_merge = tp_df[["date", "country"] + GDELT_TP_FEATURE_COLS]
            print(f"  GDELT theme/person: {len(tp_df):,}행 | {tp_df['country'].nunique()}개국")
        else:
            print(f"  [경고] GDELT theme/person parquet 없음 → F2_clean 건너뜀")
            run_f2 = False

    coverage_ts = pd.Timestamp(GDELT_TITLE_MIN_DATE, tz="UTC")

    def merge_features(split_df, split_name):
        n_before = len(split_df)

        # safe ACLED
        merged = split_df.merge(safe_merge, on=["date", "country"], how="left")
        assert len(merged) == n_before, f"row count changed: safe ACLED [{split_name}]"
        n_safe = merged[SAFE_ACLED_FEATURE_COLS].isnull().sum().sum()
        merged[SAFE_ACLED_FEATURE_COLS] = merged[SAFE_ACLED_FEATURE_COLS].fillna(0)

        # GDELT title
        merged = merged.merge(title_merge, on=["date", "country"], how="left")
        assert len(merged) == n_before, f"row count changed: GDELT title [{split_name}]"
        n_title = merged[GDELT_TITLE_FEATURE_COLS].isnull().sum().sum()
        merged[GDELT_TITLE_FEATURE_COLS] = merged[GDELT_TITLE_FEATURE_COLS].fillna(0)

        # coverage_mask
        merged[GDELT_TITLE_COVERAGE_COL] = (merged["date"] < coverage_ts).astype(int)

        # GDELT theme/person
        if tp_merge is not None:
            merged = merged.merge(tp_merge, on=["date", "country"], how="left")
            assert len(merged) == n_before, f"row count changed: GDELT TP [{split_name}]"
            n_tp = merged[GDELT_TP_FEATURE_COLS].isnull().sum().sum()
            merged[GDELT_TP_FEATURE_COLS] = merged[GDELT_TP_FEATURE_COLS].fillna(0)
        else:
            n_tp = 0

        n_cov = int((merged[GDELT_TITLE_COVERAGE_COL] == 1).sum())
        print(f"    [{split_name}] safe_null={n_safe:,}→0 | title_null={n_title:,}→0 | "
              f"tp_null={n_tp:,}→0 | cov_mask=1: {n_cov:,}행")
        return merged

    print()
    for key in ("train_fit", "tune_cal", "val_eval", "test"):
        splits[key] = merge_features(splits[key], key)

    return splits, run_f2


# ─────────────────────────────────────────────────────────────────────────────
# Feature cols 선별 및 검증
# ─────────────────────────────────────────────────────────────────────────────

def _base_cols(df):
    """B feature 35개: label/exclude/extra 전부 제외 후 남는 컬럼."""
    exclude = set(LABEL_META_COLS) | set(ALWAYS_EXCLUDE) | _ALL_EXTRA
    return [c for c in df.columns if c not in exclude]


def get_feature_cols(df, experiment: str):
    base = _base_cols(df)
    if experiment == "F0_clean":
        return base + SAFE_ACLED_FEATURE_COLS
    if experiment == "F1_clean":
        return base + SAFE_ACLED_FEATURE_COLS + GDELT_TITLE_FEATURE_COLS + [GDELT_TITLE_COVERAGE_COL]
    if experiment == "F2_clean":
        return (base + SAFE_ACLED_FEATURE_COLS + GDELT_TITLE_FEATURE_COLS
                + [GDELT_TITLE_COVERAGE_COL] + GDELT_TP_FEATURE_COLS)
    raise ValueError(f"Unknown experiment: {experiment}")


_EXPECTED = {"F0_clean": 50, "F1_clean": 72, "F2_clean": 94}


def validate_feature_cols(feature_cols, experiment: str):
    print(f"\n  [{experiment}] Feature 검증")

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

    expected = _EXPECTED[experiment]
    safe_n  = sum(1 for c in SAFE_ACLED_FEATURE_COLS if c in feature_cols)
    title_n = sum(1 for c in GDELT_TITLE_FEATURE_COLS if c in feature_cols)
    mask_n  = 1 if GDELT_TITLE_COVERAGE_COL in feature_cols else 0
    tp_n    = sum(1 for c in GDELT_TP_FEATURE_COLS if c in feature_cols)
    b_n     = len(feature_cols) - safe_n - title_n - mask_n - tp_n

    print(f"    B:{b_n} + safe_ACLED:{safe_n} + title:{title_n} + mask:{mask_n} + TP:{tp_n}"
          f" = {len(feature_cols)}개  (기대 {expected})")

    if len(feature_cols) != expected:
        print(f"  [경고] feature 수 불일치: {len(feature_cols)} ≠ {expected}")
    print(f"    ACLED/SE/label 잔존 없음 ✅")


# ─────────────────────────────────────────────────────────────────────────────
# Feature 행렬 생성 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def fit_country_encoder(train_fit_df):
    enc = LabelEncoder()
    enc.fit(train_fit_df["country"].astype(str))
    return enc


def make_lgbm_X(df, feature_cols):
    X = df[feature_cols].copy()
    if "country" in X.columns:
        X["country"] = X["country"].astype("category")
    return X


def make_xgb_X(df, feature_cols, country_enc):
    X = df[feature_cols].copy()
    if "country" in X.columns:
        country_map = {c: i for i, c in enumerate(country_enc.classes_)}
        codes = df["country"].astype(str).map(lambda c: country_map.get(c, -1))
        n_unk = int((codes == -1).sum())
        if n_unk:
            print(f"    WARNING: {n_unk} unknown country codes → -1")
        X["country"] = codes.values.astype(np.int32)
    return X.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# OOF 생성 (train_fit 내부, tune_cal/val_eval 미사용)
# ─────────────────────────────────────────────────────────────────────────────

def run_oof_folds(train_fit, feature_cols, country_enc):
    """train_fit 내부에서만 OOF 생성. tune_cal/val_eval 개입 없음."""
    oof_rows_lgbm, oof_rows_xgb = [], []
    fold_summaries = []

    for fold in OOF_FOLDS:
        fname, train_end, pred_yr = fold["name"], fold["train_end"], fold["pred_year"]

        fold_tr = train_fit[train_fit[DATE_COL].dt.year <= train_end]
        fold_pr = train_fit[train_fit[DATE_COL].dt.year == pred_yr]

        y_tr = fold_tr[TARGET_COL].values
        y_pr = fold_pr[TARGET_COL].values

        print(f"\n    {fname}: 학습 {len(fold_tr):,}행 (≤{train_end}) "
              f"→ 예측 {pred_yr} ({len(fold_pr):,}행, 양성률:{y_pr.mean():.3f})")

        ds_tr     = lgb.Dataset(make_lgbm_X(fold_tr, feature_cols), label=y_tr,
                                categorical_feature=["country"], free_raw_data=False)
        lgbm_fold = lgb.train(LGB_PARAMS, ds_tr, num_boost_round=LGB_ROUNDS_OOF)
        prob_l    = lgbm_fold.predict(make_lgbm_X(fold_pr, feature_cols))
        auc_l     = compute_pr_auc(y_pr, prob_l)
        print(f"      LightGBM PR-AUC: {auc_l:.4f}")

        dtrain_xgb = xgb.DMatrix(make_xgb_X(fold_tr, feature_cols, country_enc), label=y_tr)
        xgb_fold   = xgb.train(XGB_PARAMS_OOF, dtrain_xgb, num_boost_round=XGB_ROUNDS_OOF)
        prob_x     = xgb_fold.predict(xgb.DMatrix(make_xgb_X(fold_pr, feature_cols, country_enc)))
        auc_x      = compute_pr_auc(y_pr, prob_x)
        print(f"      XGBoost  PR-AUC: {auc_x:.4f}")

        dates, countries = fold_pr[DATE_COL].dt.strftime("%Y-%m-%d").values, fold_pr["country"].values
        for d, c, yt, pl, px in zip(dates, countries, y_pr, prob_l, prob_x):
            oof_rows_lgbm.append({"date": d, "country": c, "y_true": int(yt), "y_prob_oof": float(pl)})
            oof_rows_xgb.append( {"date": d, "country": c, "y_true": int(yt), "y_prob_oof": float(px)})

        fold_summaries.append({
            "name": fname, "pred_year": pred_yr,
            "n_train": len(fold_tr), "n_pred": len(fold_pr),
            "positive_rate": round(float(y_pr.mean()), 4),
            "lgbm_pr_auc": round(float(auc_l), 4),
            "xgb_pr_auc":  round(float(auc_x), 4),
        })

    return pd.DataFrame(oof_rows_lgbm), pd.DataFrame(oof_rows_xgb), fold_summaries


# ─────────────────────────────────────────────────────────────────────────────
# Final base model 학습 (train_fit, early stop on tune_cal)
# ─────────────────────────────────────────────────────────────────────────────

def train_final_lgbm(train_fit, tune_cal, feature_cols):
    print("\n    [3a] Final LightGBM (train_fit, early stop on tune_cal)")
    ds_tr = lgb.Dataset(make_lgbm_X(train_fit, feature_cols), label=train_fit[TARGET_COL].values,
                        categorical_feature=["country"], free_raw_data=False)
    ds_tc = lgb.Dataset(make_lgbm_X(tune_cal, feature_cols), label=tune_cal[TARGET_COL].values,
                        reference=ds_tr, free_raw_data=False)
    model = lgb.train(
        LGB_PARAMS, ds_tr, num_boost_round=LGB_ROUNDS_FINAL,
        valid_sets=[ds_tc], valid_names=["tune_cal"],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
                   lgb.log_evaluation(period=LOG_PERIOD)],
    )
    print(f"      Best iteration: {model.best_iteration}")
    return model


def train_final_xgb(train_fit, tune_cal, feature_cols, country_enc):
    print("\n    [3b] Final XGBoost (train_fit, early stop on tune_cal)")
    dtrain = xgb.DMatrix(make_xgb_X(train_fit, feature_cols, country_enc),
                         label=train_fit[TARGET_COL].values)
    dtune  = xgb.DMatrix(make_xgb_X(tune_cal, feature_cols, country_enc),
                         label=tune_cal[TARGET_COL].values)
    model = xgb.train(
        XGB_PARAMS_FINAL, dtrain, num_boost_round=XGB_ROUNDS_FINAL,
        evals=[(dtune, "tune_cal")],
        callbacks=[XGBEarlyStopping(rounds=EARLY_STOP, save_best=True)],
        verbose_eval=LOG_PERIOD,
    )
    print(f"      Best iteration: {model.best_iteration}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Meta learner (OOF으로 학습, tune_cal으로 C 선택)
# ─────────────────────────────────────────────────────────────────────────────

def train_meta_logreg(oof_lgbm, oof_xgb, tune_prob_l, tune_prob_x, tune_y):
    """
    meta learner fit: OOF predictions (train_fit 내부)
    C 선택: tune_cal predictions + tune_cal labels
    val_eval label은 절대 사용 안 함
    """
    oof = oof_lgbm[["date","country","y_true","y_prob_oof"]].merge(
        oof_xgb[["date","country","y_prob_oof"]].rename(columns={"y_prob_oof":"y_prob_oof_x"}),
        on=["date","country"], how="inner")
    X_oof = oof[["y_prob_oof","y_prob_oof_x"]].values
    y_oof = oof["y_true"].values
    X_tune_meta = np.column_stack([tune_prob_l, tune_prob_x])

    print(f"\n    [4] Meta LogReg (OOF {len(X_oof):,}행 학습, tune_cal C 선택)")
    best_c, best_auc = None, -1.0
    for c in META_C_CANDIDATES:
        m = LogisticRegression(class_weight="balanced", C=c, solver="lbfgs",
                               max_iter=1000, random_state=RANDOM_SEED)
        m.fit(X_oof, y_oof)
        auc = compute_pr_auc(tune_y, m.predict_proba(X_tune_meta)[:, 1])
        print(f"      C={c:.2f} → tune_cal PR-AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc, best_c = auc, c

    print(f"      → 선택된 C = {best_c}  (tune_cal PR-AUC {best_auc:.4f})")
    meta = LogisticRegression(class_weight="balanced", C=best_c, solver="lbfgs",
                              max_iter=1000, random_state=RANDOM_SEED)
    meta.fit(X_oof, y_oof)
    return meta, best_c


# ─────────────────────────────────────────────────────────────────────────────
# Calibration (tune_cal로 fit)
# ─────────────────────────────────────────────────────────────────────────────

def fit_calibrators(tune_stack_raw, tune_y):
    """Platt + Isotonic calibration을 tune_cal stacking predictions로 fit."""
    print("\n    [5] Calibration fit on tune_cal")
    platt = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    platt.fit(tune_stack_raw.reshape(-1, 1), tune_y)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(tune_stack_raw, tune_y)
    print("      Platt + Isotonic fit 완료 (tune_cal 기준)")
    return platt, iso


# ─────────────────────────────────────────────────────────────────────────────
# 단일 실험 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(exp_name: str, splits: dict, country_enc: LabelEncoder):
    """
    하나의 실험(F0/F1/F2)을 clean split 기준으로 실행.
    tune_cal: early stop / meta C / calibration fit
    val_eval: 최종 평가만
    """
    print(f"\n{'='*70}")
    print(f"  실험: {exp_name}")
    print(f"{'='*70}")

    train_fit = splits["train_fit"]
    tune_cal  = splits["tune_cal"]
    val_eval  = splits["val_eval"]
    test      = splits["test"]

    # feature cols 결정 + 검증
    feature_cols = get_feature_cols(train_fit, exp_name)
    validate_feature_cols(feature_cols, exp_name)
    print(f"  feature_cols: {len(feature_cols)}개")

    # OOF (train_fit 내부)
    print(f"\n  [Step 2] OOF 생성 ({len(OOF_FOLDS)} fold, train_fit 내부)")
    oof_lgbm, oof_xgb, fold_summaries = run_oof_folds(train_fit, feature_cols, country_enc)

    # Final base models (train_fit 학습, tune_cal early stop)
    lgbm_final = train_final_lgbm(train_fit, tune_cal, feature_cols)
    xgb_final  = train_final_xgb(train_fit, tune_cal, feature_cols, country_enc)

    # Predictions on tune_cal (meta C 선택 + calibration fit용)
    tune_y      = tune_cal[TARGET_COL].values
    tune_prob_l = lgbm_final.predict(make_lgbm_X(tune_cal, feature_cols))
    tune_prob_x = xgb_final.predict(xgb.DMatrix(make_xgb_X(tune_cal, feature_cols, country_enc)))
    print(f"\n    tune_cal → LGB PR-AUC: {compute_pr_auc(tune_y, tune_prob_l):.4f} | "
          f"XGB PR-AUC: {compute_pr_auc(tune_y, tune_prob_x):.4f}")

    # Meta learner (OOF 학습, tune_cal C 선택)
    meta_model, best_c = train_meta_logreg(oof_lgbm, oof_xgb, tune_prob_l, tune_prob_x, tune_y)

    # tune_cal stacking raw (calibration fit용)
    X_tune_meta    = np.column_stack([tune_prob_l, tune_prob_x])
    tune_stack_raw = meta_model.predict_proba(X_tune_meta)[:, 1]

    # Calibration fit on tune_cal
    platt_cal, iso_cal = fit_calibrators(tune_stack_raw, tune_y)

    # Predictions on val_eval (순수 평가용)
    val_y       = val_eval[TARGET_COL].values
    val_prob_l  = lgbm_final.predict(make_lgbm_X(val_eval, feature_cols))
    val_prob_x  = xgb_final.predict(xgb.DMatrix(make_xgb_X(val_eval, feature_cols, country_enc)))

    X_val_meta    = np.column_stack([val_prob_l, val_prob_x])
    val_stack_raw = meta_model.predict_proba(X_val_meta)[:, 1]
    val_platt     = apply_platt(platt_cal, val_stack_raw)
    val_iso       = apply_isotonic(iso_cal, val_stack_raw)

    # val_eval 지표 계산
    print(f"\n    [6] val_eval 지표 계산 (평가 전용, fit 없음)")
    all_metrics = {
        "lgbm":              compute_metrics(val_y, val_prob_l),
        "xgb":               compute_metrics(val_y, val_prob_x),
        "stacking_raw":      compute_metrics(val_y, val_stack_raw),
        "stacking_platt":    compute_metrics(val_y, val_platt),
        "stacking_isotonic": compute_metrics(val_y, val_iso),
    }

    header = f"    {'모델':<25} {'PR-AUC':>8} {'P@5%':>8} {'Brier':>8} {'ECE':>8}"
    print(f"\n{header}")
    print(f"    {'-'*55}")
    label_map = {
        "lgbm": "LightGBM", "xgb": "XGBoost",
        "stacking_raw": "Stacking (raw)",
        "stacking_platt": "Stacking (Platt)",
        "stacking_isotonic": "Stacking (Isotonic)",
    }
    best_pr = max(v["pr_auc"] for v in all_metrics.values())
    for k, lb in label_map.items():
        mv   = all_metrics[k]
        flag = " ←" if abs(mv["pr_auc"] - best_pr) < 1e-6 else ""
        print(f"    {lb:<25} {mv['pr_auc']:>8.4f} {mv['p_at_top5pct']:>8.4f} "
              f"{mv['brier_score']:>8.4f} {mv['ece']:>8.4f}{flag}")

    platt_pr = all_metrics["stacking_platt"]["pr_auc"]
    print(f"\n    Stacking Platt PR-AUC (val_eval): {platt_pr:.4f}")

    return {
        "name":          exp_name,
        "feature_n":     len(feature_cols),
        "metrics":       all_metrics,
        "best_c":        best_c,
        "fold_summaries":fold_summaries,
        "platt_pr_auc":  platt_pr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 결과 리포트 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_report(results: list):
    today = dt_date.today().isoformat()

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and v == v else "—"

    def delta_str(val, ref):
        if val != val or ref != ref:
            return "—"
        d = val - ref
        if d > 0.0005:  return f"+{d:.4f} ↑"
        if d < -0.0005: return f"{d:.4f} ↓"
        return f"{d:.4f} ≈"

    # 각 실험의 Stacking Platt PR-AUC 추출
    prs = {r["name"]: r["platt_pr_auc"] for r in results}
    f0  = prs.get("F0_clean", float("nan"))
    f1  = prs.get("F1_clean", float("nan"))
    f2  = prs.get("F2_clean", float("nan"))

    smoke_tag = " (⚠️ SMOKE TEST — 참고용)" if SMOKE_TEST else ""

    lines = [
        f"# Clean Validation: F0/F1/F2 비교 결과{smoke_tag}",
        f"",
        f"**실험명**: `{EXPERIMENT_TAG}`  ",
        f"**실행일**: {today}  ",
        f"",
        "## Split 설계",
        "",
        "| Split | 기간 | 역할 |",
        "|-------|------|------|",
        "| train_fit | 2014-01-01 ~ 2022-12-31 | base model 학습, OOF 생성 |",
        "| tune_cal  | 2023-01-01 ~ 2023-12-31 | early stopping, meta C 선택, calibration fit |",
        "| val_eval  | 2024-01-01 ~ 2024-06-30 | **순수 평가 전용** (어떤 학습에도 사용 안 함) |",
        "| test      | 2024-07-01 ~             | 아직 평가하지 않음 |",
        "",
        "> val_eval은 early stopping, meta C 선택, calibration fit 어디에도 사용하지 않았다.  ",
        "> 따라서 ECE/Brier/PR-AUC는 기존 F0/F1 스크립트보다 덜 낙관적인 추정이다.  ",
        "> 단, 절대 수치는 여전히 test에서 달라질 수 있다.",
        "",
        "---",
        "",
        "## Feature 구성",
        "",
        "| 실험 | feature 수 | 구성 |",
        "|------|-----------|------|",
        "| F0_clean | 50 | B 35 + safe ACLED 15 |",
        "| F1_clean | 72 | F0 + GDELT title 21 + coverage_mask 1 |",
        "| F2_clean | 94 | F1 + GDELT theme/person 22 |",
        "",
        "---",
        "",
        "## 실험별 val_eval 성능",
        "",
    ]

    model_order = [
        ("lgbm",             "LightGBM"),
        ("xgb",              "XGBoost"),
        ("stacking_raw",     "Stacking (raw)"),
        ("stacking_platt",   "Stacking (Platt) ★"),
        ("stacking_isotonic","Stacking (Isotonic)"),
    ]

    for r in results:
        exp_name = r["name"]
        mv_dict  = r["metrics"]
        lines += [
            f"### {exp_name} (feature {r['feature_n']}개)",
            "",
            f"| 모델 | PR-AUC | P@5% | R@P≥.10 | Brier | ECE |",
            f"|------|--------|------|---------|-------|-----|",
        ]
        for key, label in model_order:
            if key not in mv_dict:
                continue
            mv = mv_dict[key]
            lines.append(
                f"| {label} "
                f"| {fmt(mv.get('pr_auc',float('nan')))} "
                f"| {fmt(mv.get('p_at_top5pct',float('nan')))} "
                f"| {fmt(mv.get('recall_at_precision_010',float('nan')))} "
                f"| {fmt(mv.get('brier_score',float('nan')))} "
                f"| {fmt(mv.get('ece',float('nan')))} |"
            )
        lines += ["", f"Meta LogReg C: {r['best_c']}", ""]

    # OOF fold 요약
    lines += ["---", "", "## OOF fold 요약 (train_fit 내부)", ""]
    if SMOKE_TEST:
        lines.append("> ⚠️ SMOKE TEST: 1 fold만 실행됨.")
        lines.append("")
    for r in results:
        lines.append(f"### {r['name']}")
        lines += [
            "| fold | pred_year | n_train | n_pred | pos_rate | LGBM PR-AUC | XGB PR-AUC |",
            "|------|-----------|---------|--------|----------|-------------|------------|",
        ]
        for fs in r["fold_summaries"]:
            lines.append(
                f"| {fs['name']} | {fs['pred_year']} "
                f"| {fs['n_train']:,} | {fs['n_pred']:,} "
                f"| {fs['positive_rate']:.4f} "
                f"| {fs['lgbm_pr_auc']:.4f} | {fs['xgb_pr_auc']:.4f} |"
            )
        lines.append("")

    # 실험 간 비교
    lines += ["---", "", "## 실험 간 Stacking Platt PR-AUC 비교", ""]
    ref_b = 0.0564
    ref_c = 0.0653
    lines += [
        f"| 실험 | feature 수 | Stacking Platt PR-AUC | vs B(0.0564) | vs C(0.0653) | 판정 |",
        f"|------|-----------|----------------------|-------------|-------------|------|",
        f"| B baseline (ACLED-free) | 35 | 0.0564 | — | — | 비교 기준 |",
        f"| C baseline (ACLED-free+title) | 57 | 0.0653 | +0.0089 ↑ | — | 비교 기준 |",
    ]
    thresholds = {"F0_clean": ref_b + 0.003, "F1_clean": f0 + 0.003, "F2_clean": f1 + 0.003}
    for r in results:
        pr   = r["platt_pr_auc"]
        thr  = thresholds.get(r["name"], float("nan"))
        adopted = pr >= thr if (pr == pr and thr == thr) else False
        lines.append(
            f"| **{r['name']}** | {r['feature_n']} | **{fmt(pr)}** "
            f"| {delta_str(pr, ref_b)} | {delta_str(pr, ref_c)} "
            f"| {'✅ 채택' if adopted else '❌ 미채택'} (기준≥{fmt(thr)}) |"
        )

    # 단계별 delta
    lines += [
        "",
        "### 단계별 개선폭 (Stacking Platt, val_eval 기준)",
        "",
        f"- F0_clean           : {fmt(f0)}",
        f"- F1_clean           : {fmt(f1)}  (F1-F0 delta: {delta_str(f1, f0)})",
        f"- F2_clean           : {fmt(f2)}  (F2-F1 delta: {delta_str(f2, f1)})",
    ]

    # ECE/Brier 해석 주의사항
    lines += [
        "",
        "---",
        "",
        "## ECE/Brier 해석",
        "",
        "- calibration fit: tune_cal (2023) 기준  ",
        "- calibration 평가: val_eval (2024-H1) 기준  ",
        "- **기존 스크립트(F0/F1)와 달리 fit set ≠ eval set이므로 ECE/Brier가 실제 calibration 품질을 반영함**  ",
        "- 다만 Platt calibration은 단조 변환이므로 PR-AUC는 raw/Platt 간 거의 동일  ",
        "- 모델 선택 기준: PR-AUC (rank 기반, calibration 무관)",
        "",
        "---",
        "",
        "## test set 평가 정책",
        "",
        "> **test set은 최종 feature/model 구조가 확정된 시점에 딱 한 번만 평가한다.**  ",
        "> 현재는 F0/F1/F2 비교 단계이며, test set은 아직 평가하지 않았다.  ",
        "> val_eval 지표로 방향을 결정하고, 최종 모델 선택 후 test PR-AUC를 1회 측정한다.  ",
        "",
        f"*생성: {today}*",
    ]

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  리포트 저장: {REPORT_MD}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    run_f2 = os.path.exists(GDELT_TP_PATH)

    validate_setup(run_f2)

    # 데이터 로드 + feature merge
    splits, run_f2 = load_all_data(run_f2)

    # Country encoder는 train_fit 기준으로 한 번만 fit
    country_enc = fit_country_encoder(splits["train_fit"])
    print(f"\n  XGBoost country encoder: {len(country_enc.classes_)}개 국가 (train_fit 기준)")

    # 실험 목록
    experiments = ["F0_clean", "F1_clean"]
    if run_f2:
        experiments.append("F2_clean")

    # 실험 실행
    all_results = []
    for exp_name in experiments:
        result = run_experiment(exp_name, splits, country_enc)
        all_results.append(result)

    # 최종 요약
    print()
    print("=" * 70)
    print("  Clean Validation 비교 완료 (val_eval = 순수 평가 전용)")
    print(f"  {'실험':<12} {'feature':>8} {'Stacking Platt PR-AUC':>22}")
    print(f"  {'-'*44}")
    for r in all_results:
        print(f"  {r['name']:<12} {r['feature_n']:>8}개 {r['platt_pr_auc']:>22.4f}")
    print("=" * 70)

    save_report(all_results)


if __name__ == "__main__":
    main()
