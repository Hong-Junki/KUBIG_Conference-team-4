"""
Experiment F0: Safe ACLED Lag + GDELT events + Economic + Country (SE-free)
============================================================================
B ACLED-free baseline (35개 피처) 위에 leakage-safe ACLED lag feature 15개를
추가해 safe ACLED 정보의 기여도를 측정한다.

목적:
  B baseline (val Stacking Platt PR-AUC=0.0564) 대비 +0.003 이상 개선되는지 확인.
  채택 기준: Stacking Platt PR-AUC ≥ 0.0594

Feature set (50개):
  B 피처 (35개):
    GDELT events (19): gdelt_goldstein_mean/std/tone_mean/mentions_sum/event_count
                       × {7d,14d,30d}, gdelt_quadclass_{1-4}_ratio
    Economic (15):     econ_{vix,wti,gold,dxy,stlfsi4} × {level, pct_1d, pct_7d}
    Country (1):       country

  safe ACLED lag 피처 (15개):
    count/fatalities (9):  safe_acled_{event_count,fatalities,fatalities_max}
                           × {7d,14d,30d}_lag7
    event type ratio (3):  safe_acled_ratio_{battles,explosions,vac}_lag7
    actor type ratio (2):  safe_acled_ratio_{state_forces,external_forces}_lag7
    missing mask (1):      safe_acled_missing_mask

제외:
  - 기존 acled_* feature: 사용하지 않음
  - acled_missing_mask (구버전): safe_acled_missing_mask 사용
  - macis_se_score: leakage 가능성으로 제외
  - event_count_next3d, fatalities_next3d: future label
  - y, y_onset, y_escalation: label

leakage-free 보장:
  safe ACLED lag feature는 shift(7) 후 rolling으로 계산.
  t일 feature는 최대 t-7일까지의 ACLED 데이터만 참조.
  label window (t+1~t+3)와 feature window (~t-7) 사이 8일 gap 확보.

safe ACLED 피처 소스:
  members/byeonghyeon/input/processed/acled_safe/safe_acled_lag_features.parquet
  (build_safe_acled_lag_features.py로 생성, gitignore로 추적 안 됨)

OOF design: B/C와 동일한 expanding-window 6-fold
  F1: train ≤2017 → predict 2018
  F2: train ≤2018 → predict 2019
  F3: train ≤2019 → predict 2020
  F4: train ≤2020 → predict 2021
  F5: train ≤2021 → predict 2022
  F6: train ≤2022 → predict 2023

출력:
  members/byeonghyeon/outputs/reports/safe_acled_f0_results.md
  (예측 CSV 저장 없음)

실행:
  cd <KUBIG_Conference-team-4 루트>
  python members/byeonghyeon/modeling/run_stacking_safe_acled_f0.py

  Smoke test (~5분):
  SMOKE_TEST=1 python members/byeonghyeon/modeling/run_stacking_safe_acled_f0.py
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
_SUBTREE_ROOT = os.path.dirname(_SCRIPT_DIR)  # members/byeonghyeon/

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_SUBTREE_ROOT))),
                 "conflict-early-warning"),
)

sys.path.insert(0, _SCRIPT_DIR)
from evaluate import compute_pr_auc, compute_p_at_top_k, compute_recall_at_precision, compute_ece

# ── Smoke test ────────────────────────────────────────────────────────────────
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

# ── 실험 식별자 ────────────────────────────────────────────────────────────────
EXPERIMENT  = "stacking_safe_acled_f0" + ("_smoke" if SMOKE_TEST else "")
RANDOM_SEED = 42

# ── 입력 경로 ─────────────────────────────────────────────────────────────────
TRAIN_PATH        = os.path.join(DATA_ROOT, "input", "processed", "dataset", "train.parquet")
VAL_PATH          = os.path.join(DATA_ROOT, "input", "processed", "dataset", "val.parquet")
TEST_PATH         = os.path.join(DATA_ROOT, "input", "processed", "dataset", "test.parquet")
SAFE_ACLED_PATH   = os.path.join(_SUBTREE_ROOT, "input", "processed", "acled_safe",
                                 "safe_acled_lag_features.parquet")

# ── 출력 경로 ─────────────────────────────────────────────────────────────────
REPORT_DIR = os.path.join(_SUBTREE_ROOT, "outputs", "reports")
REPORT_MD  = os.path.join(REPORT_DIR, "safe_acled_f0_results.md")

# ── 컬럼 상수 ─────────────────────────────────────────────────────────────────
TARGET_COL = "y_escalation"
DATE_COL   = "date"

LABEL_META_COLS = [
    "y", "y_onset", "y_escalation",
    "fatalities_next3d", "event_count_next3d",
    "past14d_event_count", "past14d_fatalities_mean",
]

# 기존 ACLED feature와 SE score는 모두 제외
ACLED_REMOVE_COLS = [
    "acled_event_count_7d",    "acled_event_count_14d",   "acled_event_count_30d",
    "acled_fatalities_7d",     "acled_fatalities_14d",    "acled_fatalities_30d",
    "acled_fatalities_max_7d", "acled_fatalities_max_14d","acled_fatalities_max_30d",
    "acled_ratio_battles",     "acled_ratio_explosions",  "acled_ratio_vac",
    *[f"acled_actor_type_{i}_ratio" for i in range(1, 9)],
    "acled_missing_mask",      # 구버전 mask — safe_acled_missing_mask 사용
    "macis_se_score",
]
ALWAYS_EXCLUDE = [DATE_COL] + ACLED_REMOVE_COLS

# safe ACLED lag feature 15개
SAFE_ACLED_FEATURE_COLS = [
    # count / fatalities 계열 (9개)
    "safe_acled_event_count_7d_lag7",
    "safe_acled_event_count_14d_lag7",
    "safe_acled_event_count_30d_lag7",
    "safe_acled_fatalities_7d_lag7",
    "safe_acled_fatalities_14d_lag7",
    "safe_acled_fatalities_30d_lag7",
    "safe_acled_fatalities_max_7d_lag7",
    "safe_acled_fatalities_max_14d_lag7",
    "safe_acled_fatalities_max_30d_lag7",
    # event type ratio 계열 (3개)
    "safe_acled_ratio_battles_lag7",
    "safe_acled_ratio_explosions_lag7",
    "safe_acled_ratio_vac_lag7",
    # actor type ratio 계열 (2개, 실제 데이터 존재 확인)
    "safe_acled_ratio_state_forces_lag7",
    "safe_acled_ratio_external_forces_lag7",
    # 결측 마스크
    "safe_acled_missing_mask",
]

# B feature 35개 (검증용)
FEATURE_COLS_B = [
    "gdelt_goldstein_mean_7d",  "gdelt_goldstein_mean_14d",  "gdelt_goldstein_mean_30d",
    "gdelt_goldstein_std_7d",   "gdelt_goldstein_std_14d",   "gdelt_goldstein_std_30d",
    "gdelt_tone_mean_7d",       "gdelt_tone_mean_14d",       "gdelt_tone_mean_30d",
    "gdelt_mentions_sum_7d",    "gdelt_mentions_sum_14d",    "gdelt_mentions_sum_30d",
    "gdelt_event_count_7d",     "gdelt_event_count_14d",     "gdelt_event_count_30d",
    "gdelt_quadclass_1_ratio",  "gdelt_quadclass_2_ratio",
    "gdelt_quadclass_3_ratio",  "gdelt_quadclass_4_ratio",
    "econ_vix",     "econ_vix_pct_1d",     "econ_vix_pct_7d",
    "econ_wti",     "econ_wti_pct_1d",     "econ_wti_pct_7d",
    "econ_gold",    "econ_gold_pct_1d",    "econ_gold_pct_7d",
    "econ_dxy",     "econ_dxy_pct_1d",     "econ_dxy_pct_7d",
    "econ_stlfsi4", "econ_stlfsi4_pct_1d", "econ_stlfsi4_pct_7d",
    "country",
]

# ── OOF fold 설계 ─────────────────────────────────────────────────────────────
_ALL_FOLDS = [
    {"name": "F1", "train_end": 2017, "pred_year": 2018},
    {"name": "F2", "train_end": 2018, "pred_year": 2019},
    {"name": "F3", "train_end": 2019, "pred_year": 2020},
    {"name": "F4", "train_end": 2020, "pred_year": 2021},
    {"name": "F5", "train_end": 2021, "pred_year": 2022},
    {"name": "F6", "train_end": 2022, "pred_year": 2023},
]
OOF_FOLDS = _ALL_FOLDS[-1:] if SMOKE_TEST else _ALL_FOLDS

# ── 하이퍼파라미터 (B/C와 동일) ───────────────────────────────────────────────
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
XGB_PARAMS_FINAL = {**XGB_PARAMS_OOF, "verbosity": 1}
XGB_ROUNDS_OOF   = 50  if SMOKE_TEST else 500
XGB_ROUNDS_FINAL = 100 if SMOKE_TEST else 1000

META_C_CANDIDATES = [0.01, 0.1, 1.0, 10.0]

# 비교 기준
BASELINE_B_PR_AUC = 0.0564   # B ACLED-free
BASELINE_C_PR_AUC = 0.0653   # C ACLED-free + title
ADOPT_THRESHOLD   = BASELINE_B_PR_AUC + 0.003  # F0 채택 기준 ≥ 0.0594

EXPECTED_N_FEATURES = 50  # B:35 + safe ACLED:15


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
        "recall_at_precision_030": _recall_at_prec(y_true, y_prob, 0.30),
        "brier_score":             float(np.mean((y_prob - y_true.astype(float)) ** 2)),
        "ece":                     compute_ece(y_true, y_prob),
        "positive_rate":           float(y_true.mean()),
        "n_samples":               int(len(y_true)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 0 — 경로 및 파일 검증
# ─────────────────────────────────────────────────────────────────────────────

def validate_setup():
    print("=" * 70)
    print(f"  실험: {EXPERIMENT}")
    if SMOKE_TEST:
        print("  *** SMOKE TEST 모드: OOF 1 fold, 라운드 축소 ***")
        print(f"  OOF folds: {[f['name'] for f in OOF_FOLDS]}")
        print(f"  LGB rounds: OOF={LGB_ROUNDS_OOF}, Final={LGB_ROUNDS_FINAL}")
        print(f"  XGB rounds: OOF={XGB_ROUNDS_OOF}, Final={XGB_ROUNDS_FINAL}")
    print(f"  DATA_ROOT        : {DATA_ROOT}")
    print(f"  SAFE_ACLED_PATH  : {SAFE_ACLED_PATH}")
    print(f"  REPORT_MD        : {REPORT_MD}")
    print(f"  B baseline PR-AUC: {BASELINE_B_PR_AUC}")
    print(f"  C baseline PR-AUC: {BASELINE_C_PR_AUC}")
    print(f"  채택 기준        : Stacking Platt PR-AUC ≥ {ADOPT_THRESHOLD:.4f} (B+0.003)")
    print("=" * 70)

    if not os.path.isdir(DATA_ROOT):
        print(f"\n[오류] DATA_ROOT 없음: {DATA_ROOT}")
        print("  cd <KUBIG_Conference-team-4 루트> 에서 실행하거나")
        print("  DATA_ROOT=/path/to/conflict-early-warning 로 지정하세요.")
        sys.exit(1)

    for path, label in [(TRAIN_PATH, "train"), (VAL_PATH, "val"), (TEST_PATH, "test")]:
        if not os.path.exists(path):
            print(f"[오류] {label}.parquet 없음: {path}")
            sys.exit(1)

    if not os.path.exists(SAFE_ACLED_PATH):
        print(f"\n[오류] safe ACLED lag feature parquet 없음: {SAFE_ACLED_PATH}")
        print("  먼저 실행하세요:")
        print("    python members/byeonghyeon/modeling/build_safe_acled_lag_features.py")
        sys.exit(1)

    print("  → 입력 파일 존재 확인 ✅")
    os.makedirs(REPORT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — 데이터 로드 + safe ACLED lag feature merge
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data():
    print("\n[Step 1] 데이터 로딩 + safe ACLED lag feature merge")

    train = pd.read_parquet(TRAIN_PATH)
    val   = pd.read_parquet(VAL_PATH)
    test  = pd.read_parquet(TEST_PATH)

    print(f"  train : {len(train):,}행 | "
          f"{train[DATE_COL].min().date()} ~ {train[DATE_COL].max().date()} | "
          f"양성률 {train[TARGET_COL].mean():.4f}")
    print(f"  val   : {len(val):,}행 | "
          f"{val[DATE_COL].min().date()} ~ {val[DATE_COL].max().date()} | "
          f"양성률 {val[TARGET_COL].mean():.4f}")
    print(f"  test  : {len(test):,}행 | "
          f"{test[DATE_COL].min().date()} ~ {test[DATE_COL].max().date()}")

    # safe ACLED lag feature 로드
    safe_df = pd.read_parquet(SAFE_ACLED_PATH)
    print(f"\n  safe ACLED (F0) : {len(safe_df):,}행 | "
          f"{safe_df['date'].min().date()} ~ {safe_df['date'].max().date()} | "
          f"{safe_df['country'].nunique()}개국")

    # 날짜 타입 정규화
    for df in [train, val, test, safe_df]:
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("UTC")

    safe_merge = safe_df[["date", "country"] + SAFE_ACLED_FEATURE_COLS]

    def merge_safe(split_df, split_name):
        n_before = len(split_df)
        merged = split_df.merge(safe_merge, on=["date", "country"], how="left")
        assert len(merged) == n_before, f"row count changed after safe ACLED merge [{split_name}]"
        n_null = merged[SAFE_ACLED_FEATURE_COLS].isnull().sum().sum()
        merged[SAFE_ACLED_FEATURE_COLS] = merged[SAFE_ACLED_FEATURE_COLS].fillna(0)
        print(f"    [{split_name}] safe ACLED null {n_null:,}→0 채움")
        return merged

    print()
    train = merge_safe(train, "train")
    val   = merge_safe(val,   "val")
    test  = merge_safe(test,  "test")

    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Feature selection & validation
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_cols(df):
    """기존 ACLED/SE/label/future 제외 + safe ACLED lag feature 포함."""
    all_safe = set(SAFE_ACLED_FEATURE_COLS)
    exclude  = set(LABEL_META_COLS) | set(ALWAYS_EXCLUDE)
    base_cols = [c for c in df.columns if c not in exclude and c not in all_safe]
    return base_cols + SAFE_ACLED_FEATURE_COLS


def validate_feature_cols(feature_cols):
    print("\n[Step 0] Feature 검증")

    # 누수 방지: 기존 ACLED / SE / label 잔존 확인
    old_acled_leaked = [c for c in feature_cols
                        if (c.startswith("acled_") and not c.startswith("safe_acled_"))
                        or c == "macis_se_score"]
    if old_acled_leaked:
        print(f"  [오류] 기존 ACLED/SE 컬럼 잔존: {old_acled_leaked}")
        sys.exit(1)

    label_leaked = [c for c in feature_cols if c in set(LABEL_META_COLS)]
    if label_leaked:
        print(f"  [오류] label/future 컬럼 잔존: {label_leaked}")
        sys.exit(1)

    b_in   = [c for c in FEATURE_COLS_B if c in feature_cols]
    safe_in = [c for c in SAFE_ACLED_FEATURE_COLS if c in feature_cols]

    gdelt_ev = [c for c in feature_cols
                if c.startswith("gdelt_") and c not in set(SAFE_ACLED_FEATURE_COLS)]
    econ     = [c for c in feature_cols if c.startswith("econ_")]

    print(f"  feature_cols ({len(feature_cols)}개):")
    print(f"    B: GDELT events      : {len(gdelt_ev)}개")
    print(f"    B: Economic          : {len(econ)}개")
    print(f"    B: Country           : {'country' in feature_cols}")
    print(f"    B 합계               : {len(b_in)}개  (기대 35)")
    print(f"    safe ACLED lag       : {len(safe_in)}개  (기대 15)")
    print(f"    총계                 : {len(feature_cols)}개  (기대 {EXPECTED_N_FEATURES})")
    print(f"  기존 ACLED/SE/label 잔존 없음 ✅")

    if len(b_in) != 35:
        print(f"  [경고] B feature 수 불일치: {len(b_in)} ≠ 35")
    if len(safe_in) != 15:
        print(f"  [경고] safe ACLED feature 수 불일치: {len(safe_in)} ≠ 15")
    if len(feature_cols) != EXPECTED_N_FEATURES:
        print(f"  [경고] 총 feature 수 불일치: {len(feature_cols)} ≠ {EXPECTED_N_FEATURES}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature 행렬 생성 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def fit_country_encoder(train_df):
    enc = LabelEncoder()
    enc.fit(train_df["country"].astype(str))
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
# Step 2 — OOF 생성
# ─────────────────────────────────────────────────────────────────────────────

def run_oof_folds(train_df, feature_cols, country_enc):
    print(f"\n[Step 2] OOF 생성 (expanding-window, "
          f"{len(OOF_FOLDS)} fold{'s' if len(OOF_FOLDS)>1 else ''} × 2 base models)")
    print("  OOF fold: 고정 라운드 학습 (early stopping 없음)")

    oof_rows_lgbm, oof_rows_xgb = [], []
    fold_summaries = []

    for fold in OOF_FOLDS:
        fname     = fold["name"]
        train_end = fold["train_end"]
        pred_yr   = fold["pred_year"]

        fold_tr = train_df[train_df[DATE_COL].dt.year <= train_end]
        fold_pr = train_df[train_df[DATE_COL].dt.year == pred_yr]

        y_tr = fold_tr[TARGET_COL].values
        y_pr = fold_pr[TARGET_COL].values
        n_tr, n_pr = len(fold_tr), len(fold_pr)

        print(f"\n  {fname}: 학습 {n_tr:,}행 (≤{train_end}) "
              f"→ 예측 {pred_yr} ({n_pr:,}행, 양성률:{y_pr.mean():.3f})")

        # LightGBM
        ds_tr     = lgb.Dataset(
            make_lgbm_X(fold_tr, feature_cols), label=y_tr,
            categorical_feature=["country"], free_raw_data=False,
        )
        lgbm_fold = lgb.train(LGB_PARAMS, ds_tr, num_boost_round=LGB_ROUNDS_OOF)
        prob_l    = lgbm_fold.predict(make_lgbm_X(fold_pr, feature_cols))
        auc_l     = compute_pr_auc(y_pr, prob_l)
        print(f"    LightGBM PR-AUC: {auc_l:.4f}")

        # XGBoost
        dtrain_xgb = xgb.DMatrix(make_xgb_X(fold_tr, feature_cols, country_enc), label=y_tr)
        dpred_xgb  = xgb.DMatrix(make_xgb_X(fold_pr, feature_cols, country_enc))
        xgb_fold   = xgb.train(XGB_PARAMS_OOF, dtrain_xgb, num_boost_round=XGB_ROUNDS_OOF)
        prob_x     = xgb_fold.predict(dpred_xgb)
        auc_x      = compute_pr_auc(y_pr, prob_x)
        print(f"    XGBoost  PR-AUC: {auc_x:.4f}")

        dates     = fold_pr[DATE_COL].dt.strftime("%Y-%m-%d").values
        countries = fold_pr["country"].values

        for d, c, yt, pl, px in zip(dates, countries, y_pr, prob_l, prob_x):
            oof_rows_lgbm.append({"date": d, "country": c, "y_true": int(yt), "y_prob_oof": float(pl)})
            oof_rows_xgb.append( {"date": d, "country": c, "y_true": int(yt), "y_prob_oof": float(px)})

        fold_summaries.append({
            "name":          fname,
            "pred_year":     pred_yr,
            "n_train":       n_tr,
            "n_pred":        n_pr,
            "positive_rate": round(float(y_pr.mean()), 4),
            "lgbm_pr_auc":   round(float(auc_l), 4),
            "xgb_pr_auc":    round(float(auc_x), 4),
        })

    return pd.DataFrame(oof_rows_lgbm), pd.DataFrame(oof_rows_xgb), fold_summaries


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Final base model 학습
# ─────────────────────────────────────────────────────────────────────────────

def train_final_lgbm(train_df, val_df, feature_cols):
    print("\n[Step 3a] Final LightGBM 학습 (full train, val early stopping)")
    ds_tr = lgb.Dataset(
        make_lgbm_X(train_df, feature_cols), label=train_df[TARGET_COL].values,
        categorical_feature=["country"], free_raw_data=False,
    )
    ds_vl = lgb.Dataset(
        make_lgbm_X(val_df, feature_cols), label=val_df[TARGET_COL].values,
        reference=ds_tr, free_raw_data=False,
    )
    model = lgb.train(
        LGB_PARAMS, ds_tr,
        num_boost_round=LGB_ROUNDS_FINAL,
        valid_sets=[ds_vl], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=True),
            lgb.log_evaluation(period=LOG_PERIOD),
        ],
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model


def train_final_xgb(train_df, val_df, feature_cols, country_enc):
    print("\n[Step 3b] Final XGBoost 학습 (full train, val early stopping)")
    dtrain = xgb.DMatrix(
        make_xgb_X(train_df, feature_cols, country_enc),
        label=train_df[TARGET_COL].values,
    )
    dval = xgb.DMatrix(
        make_xgb_X(val_df, feature_cols, country_enc),
        label=val_df[TARGET_COL].values,
    )
    model = xgb.train(
        XGB_PARAMS_FINAL, dtrain,
        num_boost_round=XGB_ROUNDS_FINAL,
        evals=[(dval, "val")],
        callbacks=[XGBEarlyStopping(rounds=EARLY_STOP, save_best=True)],
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Base model val/test 예측
# ─────────────────────────────────────────────────────────────────────────────

def predict_base_models(lgbm_model, xgb_model, val_df, test_df, feature_cols, country_enc):
    print("\n[Step 4] Base model val/test 예측")
    val_y = val_df[TARGET_COL].values

    val_prob_l  = lgbm_model.predict(make_lgbm_X(val_df,  feature_cols))
    val_prob_x  = xgb_model.predict( xgb.DMatrix(make_xgb_X(val_df,  feature_cols, country_enc)))
    test_prob_l = lgbm_model.predict(make_lgbm_X(test_df, feature_cols))
    test_prob_x = xgb_model.predict( xgb.DMatrix(make_xgb_X(test_df, feature_cols, country_enc)))

    print(f"  LightGBM val PR-AUC : {compute_pr_auc(val_y, val_prob_l):.4f}")
    print(f"  XGBoost  val PR-AUC : {compute_pr_auc(val_y, val_prob_x):.4f}")

    return {
        "val_y":  val_y,
        "val_l":  val_prob_l,  "val_x":  val_prob_x,
        "test_l": test_prob_l, "test_x": test_prob_x,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Meta Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────

def train_meta_logreg(oof_lgbm, oof_xgb, preds):
    print("\n[Step 5] Meta LogReg 학습 + C 탐색 (val PR-AUC 기준)")

    oof = oof_lgbm[["date", "country", "y_true", "y_prob_oof"]].merge(
        oof_xgb[["date", "country", "y_prob_oof"]].rename(
            columns={"y_prob_oof": "y_prob_oof_x"}),
        on=["date", "country"], how="inner",
    )
    X_oof      = oof[["y_prob_oof", "y_prob_oof_x"]].values
    y_oof      = oof["y_true"].values
    X_val_meta = np.column_stack([preds["val_l"], preds["val_x"]])
    y_val      = preds["val_y"]

    print(f"  Meta 학습 데이터: {len(X_oof):,}행 | 양성률: {y_oof.mean():.4f}")

    best_c, best_auc = None, -1.0
    for c in META_C_CANDIDATES:
        m = LogisticRegression(class_weight="balanced", C=c, solver="lbfgs",
                               max_iter=1000, random_state=RANDOM_SEED)
        m.fit(X_oof, y_oof)
        auc = compute_pr_auc(y_val, m.predict_proba(X_val_meta)[:, 1])
        print(f"    C={c:.2f} → val PR-AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc, best_c = auc, c

    print(f"  → 선택된 C = {best_c}  (val PR-AUC {best_auc:.4f})")

    meta = LogisticRegression(class_weight="balanced", C=best_c, solver="lbfgs",
                              max_iter=1000, random_state=RANDOM_SEED)
    meta.fit(X_oof, y_oof)
    return meta, best_c


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Platt calibration
# ─────────────────────────────────────────────────────────────────────────────

def fit_calibrators(val_stack_raw, val_y):
    print("\n[Step 6] Calibration (val 기준)")
    platt = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    platt.fit(val_stack_raw.reshape(-1, 1), val_y)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_stack_raw, val_y)
    print("  Platt + Isotonic fit 완료")
    return platt, iso


def apply_platt(platt, prob):
    return platt.predict_proba(np.asarray(prob).reshape(-1, 1))[:, 1]


def apply_isotonic(iso, prob):
    return iso.predict(np.asarray(prob))


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — 결과 리포트 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_report(all_metrics, best_c, fold_summaries, feature_cols):
    today = dt_date.today().isoformat()

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and v == v else "—"

    def delta_str(val, ref):
        d = val - ref
        if d > 0.0005:  return f"+{d:.4f} ↑"
        if d < -0.0005: return f"{d:.4f} ↓"
        return f"{d:.4f} ≈"

    platt_pr = all_metrics.get("stacking_platt", {}).get("pr_auc", float("nan"))
    adopted  = platt_pr >= ADOPT_THRESHOLD
    delta_b  = platt_pr - BASELINE_B_PR_AUC
    delta_c  = platt_pr - BASELINE_C_PR_AUC
    smoke_tag = " (⚠️ SMOKE TEST — 1 fold, 결과 참고용)" if SMOKE_TEST else ""

    gdelt_n  = sum(1 for c in feature_cols
                   if c.startswith("gdelt_") and not c.startswith("safe_"))
    econ_n   = sum(1 for c in feature_cols if c.startswith("econ_"))
    safe_n   = sum(1 for c in SAFE_ACLED_FEATURE_COLS if c in feature_cols)
    country_n = 1 if "country" in feature_cols else 0

    lines = [
        f"# 실험 F0: Safe ACLED Lag + B feature 결과{smoke_tag}",
        f"",
        f"**실험명**: `{EXPERIMENT}`  ",
        f"**실행일**: {today}  ",
        f"**B baseline val PR-AUC**: {BASELINE_B_PR_AUC} (ACLED-free, 비교 기준)  ",
        f"**C baseline val PR-AUC**: {BASELINE_C_PR_AUC} (ACLED-free + title, 참고)  ",
        f"**채택 기준**: Stacking Platt PR-AUC ≥ {ADOPT_THRESHOLD:.4f} (B + 0.003)  ",
        f"",
        f"> ⚠️ val 지표는 (1) final base model early stopping, (2) meta C 탐색,",
        f"> (3) Platt calibration에 val을 모두 사용하므로 낙관적 추정임.",
        f"> test는 아직 평가하지 않았다.",
        f"",
        "---",
        "",
        "## 실험 F0 채택 여부",
        "",
        f"**Stacking Platt PR-AUC (val) = {fmt(platt_pr)}**",
        "",
        f"| 비교 기준 | 기준값 | F0 PR-AUC | delta | 판정 |",
        f"|-----------|--------|-----------|-------|------|",
        f"| B baseline (ACLED-free) | {BASELINE_B_PR_AUC:.4f} | {fmt(platt_pr)} "
        f"| {delta_str(platt_pr, BASELINE_B_PR_AUC)} | "
        f"{'✅ F0-B 달성' if delta_b >= 0.003 else '❌ F0-B 미달'} |",
        f"| C baseline (ACLED-free+title) | {BASELINE_C_PR_AUC:.4f} | {fmt(platt_pr)} "
        f"| {delta_str(platt_pr, BASELINE_C_PR_AUC)} | — |",
        f"| 채택 기준 (B+0.003) | {ADOPT_THRESHOLD:.4f} | {fmt(platt_pr)} "
        f"| — | {'✅ **채택**' if adopted else '❌ **미채택**'} |",
        "",
        "---",
        "",
        "## 피처 구성",
        "",
        "| 카테고리 | 피처 수 |",
        "|----------|---------|",
        f"| GDELT events (B) | {gdelt_n} |",
        f"| Economic (B)     | {econ_n} |",
        f"| Country (B)      | {country_n} |",
        f"| B 합계           | {gdelt_n + econ_n + country_n} |",
        f"| safe ACLED lag (신규) | {safe_n} |",
        f"| **F0 총계**      | **{len(feature_cols)}** |",
        "",
        "safe ACLED lag feature 설계:  ",
        "- shift(7) 후 rolling → t일 feature는 최대 t-7일 ACLED만 참조  ",
        "- label window (t+1~t+3) 와 feature window (~t-7) 사이 **8일 gap** 확보  ",
        "- macis_se_score 및 기존 acled_* feature 완전 제외  ",
        "",
        "---",
        "",
        "## val 지표 비교 (B / C / F0)",
        "",
        "| 모델 | F0 PR-AUC | B PR-AUC | F0-B delta | C PR-AUC | F0-C delta | P@5% | R@P≥.10 | Brier | ECE |",
        "|------|-----------|----------|------------|----------|------------|------|---------|-------|-----|",
    ]

    model_order = [
        ("lgbm",             "LightGBM"),
        ("xgb",              "XGBoost"),
        ("stacking_raw",     "Stacking (raw)"),
        ("stacking_platt",   "Stacking (Platt) ★"),
        ("stacking_isotonic","Stacking (Isotonic)"),
    ]
    b_ref = {"lgbm": 0.0601, "xgb": 0.0546, "stacking_raw":  0.0564,
             "stacking_platt": BASELINE_B_PR_AUC, "stacking_isotonic": 0.0563}
    c_ref = {"lgbm": 0.0601, "xgb": 0.0546, "stacking_raw":  0.0564,
             "stacking_platt": BASELINE_C_PR_AUC, "stacking_isotonic": 0.0563}

    for key, label in model_order:
        if key not in all_metrics:
            continue
        mv   = all_metrics[key]
        f0pr = mv.get("pr_auc", float("nan"))
        bpr  = b_ref.get(key, float("nan"))
        cpr  = c_ref.get(key, float("nan"))
        db   = delta_str(f0pr, bpr)  if (f0pr == f0pr and bpr == bpr) else "—"
        dc   = delta_str(f0pr, cpr)  if (f0pr == f0pr and cpr == cpr) else "—"
        lines.append(
            f"| {label} "
            f"| {fmt(f0pr)} "
            f"| {bpr:.4f} | {db} "
            f"| {cpr:.4f} | {dc} "
            f"| {fmt(mv.get('p_at_top5pct', float('nan')))} "
            f"| {fmt(mv.get('recall_at_precision_010', float('nan')))} "
            f"| {fmt(mv.get('brier_score', float('nan')))} "
            f"| {fmt(mv.get('ece', float('nan')))} |"
        )

    lines += [
        "",
        "---",
        "",
        "## OOF fold 요약",
        "",
    ]
    if SMOKE_TEST:
        lines.append("> ⚠️ SMOKE TEST: F6 (2023 예측) 1 fold만 실행됨. 아래 수치는 참고용.")
        lines.append("")
    lines += [
        "| fold | pred_year | n_train | n_pred | pos_rate | LGBM PR-AUC | XGB PR-AUC |",
        "|------|-----------|---------|--------|----------|-------------|------------|",
    ]
    for fs in fold_summaries:
        lines.append(
            f"| {fs['name']} | {fs['pred_year']} "
            f"| {fs['n_train']:,} | {fs['n_pred']:,} "
            f"| {fs['positive_rate']:.4f} "
            f"| {fs['lgbm_pr_auc']:.4f} "
            f"| {fs['xgb_pr_auc']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Meta LogReg",
        "",
        f"선택된 C: **{best_c}**  ",
        "",
        "---",
        "",
        "## 해석 주의사항",
        "",
        "1. 채택 기준: B + 0.003 = 0.0594. 이를 초과하면 safe ACLED lag feature 효과 확인.",
        "2. C(0.0653) 대비 F0 성능 비교: safe ACLED만 추가했을 때의 기여도 측정.",
        "3. Isotonic ECE ≈ 0은 val 과적합 (val 셋 크기 작음). Platt 채택.",
        "4. macis_se_score와 기존 acled_* feature는 완전 제외됨.",
        "5. safe ACLED feature는 shift(7)+rolling으로 leakage-free 보장.",
        "",
        "## test set 평가 정책",
        "",
        "> **test set은 최종 feature/model 구조가 확정된 시점에 딱 한 번만 평가한다.**  ",
        "> 현재는 F 실험 비교 단계이며, test set은 아직 평가하지 않았다.  ",
        "> val 지표로만 실험 방향을 결정하고, 최종 모델 선택 후 test PR-AUC를 1회 측정한다.  ",
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
    validate_setup()

    train_df, val_df, test_df = load_all_data()

    feature_cols = get_feature_cols(train_df)
    validate_feature_cols(feature_cols)

    country_enc = fit_country_encoder(train_df)
    print(f"\n  XGBoost country encoder: {len(country_enc.classes_)}개 국가")
    print(f"  최종 feature_cols: {len(feature_cols)}개 "
          f"(B 35 + safe ACLED 15 = {EXPECTED_N_FEATURES})")

    oof_lgbm, oof_xgb, fold_summaries = run_oof_folds(train_df, feature_cols, country_enc)

    lgbm_final = train_final_lgbm(train_df, val_df, feature_cols)
    xgb_final  = train_final_xgb(train_df, val_df, feature_cols, country_enc)

    preds = predict_base_models(lgbm_final, xgb_final, val_df, test_df,
                                feature_cols, country_enc)

    meta_model, best_c = train_meta_logreg(oof_lgbm, oof_xgb, preds)

    print("\n[Step 6] Stacking 예측 + Calibration")
    val_y = preds["val_y"]

    val_stack_raw  = meta_model.predict_proba(
        np.column_stack([preds["val_l"], preds["val_x"]])
    )[:, 1]
    test_stack_raw = meta_model.predict_proba(  # noqa: F841
        np.column_stack([preds["test_l"], preds["test_x"]])
    )[:, 1]

    platt_cal, iso_cal = fit_calibrators(val_stack_raw, val_y)

    val_stack_platt = apply_platt(platt_cal, val_stack_raw)
    val_stack_iso   = apply_isotonic(iso_cal, val_stack_raw)

    print("\n[Step 7] val 지표 계산")
    all_metrics = {
        "lgbm":              compute_metrics(val_y, preds["val_l"]),
        "xgb":               compute_metrics(val_y, preds["val_x"]),
        "stacking_raw":      compute_metrics(val_y, val_stack_raw),
        "stacking_platt":    compute_metrics(val_y, val_stack_platt),
        "stacking_isotonic": compute_metrics(val_y, val_stack_iso),
    }

    header = f"  {'모델':<28} {'PR-AUC':>8} {'P@5%':>8} {'Brier':>8} {'ECE':>8}"
    print(f"\n{header}")
    print(f"  {'-'*60}")
    labels = {
        "lgbm":              "LightGBM          ",
        "xgb":               "XGBoost           ",
        "stacking_raw":      "Stacking (raw)    ",
        "stacking_platt":    "Stacking (Platt)  ",
        "stacking_isotonic": "Stacking (Isotonic)",
    }
    best_pr = max(v["pr_auc"] for v in all_metrics.values())
    for k, lb in labels.items():
        mv   = all_metrics[k]
        flag = " ← best" if abs(mv["pr_auc"] - best_pr) < 1e-6 else ""
        print(f"  {lb:<28} {mv['pr_auc']:>8.4f} {mv['p_at_top5pct']:>8.4f} "
              f"{mv['brier_score']:>8.4f} {mv['ece']:>8.4f}{flag}")

    platt_pr = all_metrics["stacking_platt"]["pr_auc"]
    adopted  = platt_pr >= ADOPT_THRESHOLD

    print(f"\n  B baseline PR-AUC     : {BASELINE_B_PR_AUC}")
    print(f"  C baseline PR-AUC     : {BASELINE_C_PR_AUC}")
    print(f"  F0 Stacking Platt     : {platt_pr:.4f}")
    print(f"  delta (F0 - B)        : {platt_pr - BASELINE_B_PR_AUC:+.4f}")
    print(f"  delta (F0 - C)        : {platt_pr - BASELINE_C_PR_AUC:+.4f}")
    print(f"  채택 기준             : ≥ {ADOPT_THRESHOLD:.4f} (B + 0.003)")
    print(f"  채택 여부             : {'✅ 채택' if adopted else '❌ 미채택'}")

    save_report(all_metrics, best_c, fold_summaries, feature_cols)

    print()
    print("=" * 70)
    print("  실험 F0 완료")
    print(f"  Stacking Platt PR-AUC : {platt_pr:.4f}")
    print(f"  F0 - B delta          : {platt_pr - BASELINE_B_PR_AUC:+.4f}")
    print(f"  F0 - C delta          : {platt_pr - BASELINE_C_PR_AUC:+.4f}")
    print(f"  채택 여부             : {'채택 ✅' if adopted else '미채택 ❌'}")
    print(f"  리포트               : {REPORT_MD}")
    print("=" * 70)


if __name__ == "__main__":
    main()
