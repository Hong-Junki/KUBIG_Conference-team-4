"""
Experiment B: ACLED-free Stacking Baseline
==========================================
Reference model (stacking_tree_only_12y_with_mask_feature) 대비
ACLED 기반 피처를 전부 제거하고 GDELT events + 경제지표 + country만으로
동일한 stacking 구조를 학습한다.

목적:
  ACLED 없이 달성 가능한 성능 상한을 측정한다.
  이 결과가 실험 C (ACLED-free + GDELT titles) 의 비교 기준점이 된다.

Feature set (35개):
  GDELT events (19):
    gdelt_goldstein_mean/std/tone_mean/mentions_sum/event_count × {7d, 14d, 30d} = 15개
    gdelt_quadclass_{1,2,3,4}_ratio = 4개
  Economic (15):
    econ_{vix,wti,gold,dxy,stlfsi4} × {level, pct_1d, pct_7d} = 15개
  Country (1):
    country (ISO3, categorical)

제거 항목 (Reference 대비):
  acled_event_count/fatalities/fatalities_max × {7d,14d,30d}  (9개)
  acled_ratio_{battles,explosions,vac}                         (3개)
  acled_actor_type_{1-8}_ratio                                 (8개)
  acled_missing_mask                                           (1개)
  macis_se_score  (SE merge 단계 자체를 생략)                  (1개)

OOF design: Reference와 동일한 expanding-window 6-fold
  F1: train ≤2017 → predict 2018
  F2: train ≤2018 → predict 2019
  F3: train ≤2019 → predict 2020
  F4: train ≤2020 → predict 2021
  F5: train ≤2021 → predict 2022
  F6: train ≤2022 → predict 2023

출력:
  members/byeonghyeon/outputs/reports/acled_free_baseline_results.md
  (예측 CSV는 저장하지 않음)

실행:
  cd <conflict-early-warning 프로젝트 루트>
  python members/byeonghyeon/modeling/run_stacking_acled_free_baseline.py

  또는 DATA_ROOT 환경변수로 데이터 경로 지정:
  DATA_ROOT=/path/to/conflict-early-warning \\
    python members/byeonghyeon/modeling/run_stacking_acled_free_baseline.py
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
# DATA_ROOT: train/val/test parquet가 있는 프로젝트 루트
# 환경변수 DATA_ROOT가 없으면 개인 레포 기본 경로를 사용.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SUBTREE_ROOT = os.path.dirname(_SCRIPT_DIR)  # members/byeonghyeon/

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_SUBTREE_ROOT))),
                 "conflict-early-warning"),
)

# evaluate.py가 같은 디렉토리에 있음
sys.path.insert(0, _SCRIPT_DIR)
from evaluate import compute_pr_auc, compute_p_at_top_k, compute_recall_at_precision, compute_ece

# ── Smoke test 모드 ───────────────────────────────────────────────────────────
# 환경변수 SMOKE_TEST=1 로 설정하면 빠른 기능 검증 모드로 실행된다.
#   - OOF: F6 1개 fold만 실행 (train ≤2022 → predict 2023)
#   - 라운드: OOF 50, Final 100
#   - 목적: 경로·feature 설정·파이프라인 흐름을 5분 내로 검증
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

# ── 실험 식별자 ────────────────────────────────────────────────────────────────
EXPERIMENT  = "stacking_acled_free_baseline" + ("_smoke" if SMOKE_TEST else "")
RANDOM_SEED = 42

# ── 입력 경로 ─────────────────────────────────────────────────────────────────
TRAIN_PATH = os.path.join(DATA_ROOT, "input", "processed", "dataset", "train.parquet")
VAL_PATH   = os.path.join(DATA_ROOT, "input", "processed", "dataset", "val.parquet")
TEST_PATH  = os.path.join(DATA_ROOT, "input", "processed", "dataset", "test.parquet")

# ── 출력 경로 ─────────────────────────────────────────────────────────────────
REPORT_DIR = os.path.join(_SUBTREE_ROOT, "outputs", "reports")
REPORT_MD  = os.path.join(REPORT_DIR, "acled_free_baseline_results.md")

# ── 컬럼 상수 ─────────────────────────────────────────────────────────────────
TARGET_COL = "y_escalation"
DATE_COL   = "date"

LABEL_META_COLS = [
    "y", "y_onset", "y_escalation",
    "fatalities_next3d", "event_count_next3d",
    "past14d_event_count", "past14d_fatalities_mean",
]

# ACLED-free: ACLED raw + derived 전부 제거. SE는 merge 자체를 하지 않으므로 목록에 불필요.
ACLED_REMOVE_COLS = [
    "acled_event_count_7d",    "acled_event_count_14d",   "acled_event_count_30d",
    "acled_fatalities_7d",     "acled_fatalities_14d",    "acled_fatalities_30d",
    "acled_fatalities_max_7d", "acled_fatalities_max_14d","acled_fatalities_max_30d",
    "acled_ratio_battles",     "acled_ratio_explosions",  "acled_ratio_vac",
    *[f"acled_actor_type_{i}_ratio" for i in range(1, 9)],
    "acled_missing_mask",
]
ALWAYS_EXCLUDE = [DATE_COL] + ACLED_REMOVE_COLS

# 기대 feature 목록 (검증용)
EXPECTED_FEATURE_COLS = [
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
]  # 35개

# ── OOF fold 설계 ─────────────────────────────────────────────────────────────
_ALL_FOLDS = [
    {"name": "F1", "train_end": 2017, "pred_year": 2018},
    {"name": "F2", "train_end": 2018, "pred_year": 2019},
    {"name": "F3", "train_end": 2019, "pred_year": 2020},
    {"name": "F4", "train_end": 2020, "pred_year": 2021},
    {"name": "F5", "train_end": 2021, "pred_year": 2022},
    {"name": "F6", "train_end": 2022, "pred_year": 2023},
]
# Smoke test: 마지막 1개 fold만 사용 (train ≤2022 → predict 2023)
OOF_FOLDS = _ALL_FOLDS[-1:] if SMOKE_TEST else _ALL_FOLDS

# ── 하이퍼파라미터 (Reference와 동일) ─────────────────────────────────────────
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
LGB_ROUNDS_OOF   = 50   if SMOKE_TEST else 500
LGB_ROUNDS_FINAL = 100  if SMOKE_TEST else 1000
EARLY_STOP       = 20   if SMOKE_TEST else 50
LOG_PERIOD       = 50   if SMOKE_TEST else 100

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

# Reference val PR-AUC (비교 표시용)
REFERENCE_PR_AUC = 0.2714


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
# Step 0 — 경로 및 feature_cols 검증
# ─────────────────────────────────────────────────────────────────────────────

def validate_setup():
    print("=" * 70)
    print(f"  실험: {EXPERIMENT}")
    if SMOKE_TEST:
        print("  *** SMOKE TEST 모드: OOF 1 fold, 라운드 축소 ***")
        print(f"  OOF folds: {[f['name'] for f in OOF_FOLDS]}")
        print(f"  LGB rounds: OOF={LGB_ROUNDS_OOF}, Final={LGB_ROUNDS_FINAL}")
        print(f"  XGB rounds: OOF={XGB_ROUNDS_OOF}, Final={XGB_ROUNDS_FINAL}")
    print(f"  DATA_ROOT: {DATA_ROOT}")
    print(f"  REPORT_MD: {REPORT_MD}")
    print("=" * 70)

    if not os.path.isdir(DATA_ROOT):
        print(f"\n[오류] DATA_ROOT 디렉토리가 없음: {DATA_ROOT}")
        print("  → 실행 방법:")
        print("    cd <KUBIG_Conference-team-4 루트>")
        print("    python members/byeonghyeon/modeling/run_stacking_acled_free_baseline.py")
        print("  → 또는 DATA_ROOT 명시:")
        print("    DATA_ROOT=/path/to/conflict-early-warning python ...")
        sys.exit(1)

    for path, label in [(TRAIN_PATH, "train"), (VAL_PATH, "val"), (TEST_PATH, "test")]:
        if not os.path.exists(path):
            print(f"\n[오류] {label}.parquet 없음: {path}")
            print("  → DATA_ROOT 환경변수를 conflict-early-warning 프로젝트 루트로 설정하세요.")
            sys.exit(1)
    print("  → 입력 파일 존재 확인 ✅")

    os.makedirs(REPORT_DIR, exist_ok=True)


def get_feature_cols(df):
    exclude = set(LABEL_META_COLS) | set(ALWAYS_EXCLUDE)
    cols = [c for c in df.columns if c not in exclude]
    return cols


def validate_feature_cols(feature_cols, df_columns):
    print("\n[Step 0] Feature 검증")

    missing_expected = [c for c in EXPECTED_FEATURE_COLS if c not in feature_cols]
    unexpected = [c for c in feature_cols if c not in EXPECTED_FEATURE_COLS]

    if missing_expected:
        print(f"  [경고] 기대했으나 없는 컬럼 ({len(missing_expected)}개): {missing_expected}")
    if unexpected:
        print(f"  [경고] 예상 외 컬럼 포함 ({len(unexpected)}개): {unexpected}")

    acled_leaked = [c for c in feature_cols if c.startswith("acled_") or c == "macis_se_score"]
    if acled_leaked:
        print(f"\n  [오류] ACLED/SE 컬럼이 feature_cols에 잔존: {acled_leaked}")
        sys.exit(1)

    label_leaked = [c for c in feature_cols if c in set(LABEL_META_COLS)]
    if label_leaked:
        print(f"\n  [오류] label/future 컬럼이 feature_cols에 잔존: {label_leaked}")
        sys.exit(1)

    print(f"  feature_cols ({len(feature_cols)}개):")
    gdelt = [c for c in feature_cols if c.startswith("gdelt_")]
    econ  = [c for c in feature_cols if c.startswith("econ_")]
    other = [c for c in feature_cols if not c.startswith("gdelt_") and not c.startswith("econ_")]
    print(f"    GDELT events : {len(gdelt)}개  — {gdelt}")
    print(f"    economic     : {len(econ)}개  — {econ}")
    print(f"    other/country: {len(other)}개  — {other}")
    print(f"  ACLED/SE 컬럼 완전 제거 확인 ✅")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — 데이터 로드 (SE merge 없음)
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data():
    print("\n[Step 1] 데이터 로딩 (SE merge 없음 — ACLED-free)")
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
    print("  → macis_se_score merge 건너뜀")
    return train, val, test


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
    print("\n[Step 2] OOF 생성 (expanding-window, 6 folds × 2 base models)")
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
        n_tr = len(fold_tr)
        n_pr = len(fold_pr)

        print(f"\n  {fname}: 학습 {n_tr:,}행 (≤{train_end}) "
              f"→ 예측 {pred_yr} ({n_pr:,}행, 양성률:{y_pr.mean():.3f})")

        # LightGBM
        ds_tr    = lgb.Dataset(
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
    X_oof = oof[["y_prob_oof", "y_prob_oof_x"]].values
    y_oof = oof["y_true"].values
    X_val_meta = np.column_stack([preds["val_l"], preds["val_x"]])
    y_val = preds["val_y"]

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

    m = all_metrics

    lines = []
    lines += [
        f"# 실험 B: ACLED-free Stacking Baseline 결과",
        f"",
        f"**실험명**: `{EXPERIMENT}`  ",
        f"**실행일**: {today}  ",
        f"**Reference val PR-AUC**: {REFERENCE_PR_AUC} (기존 full model, 비교 참고용)  ",
        f"",
        f"> ⚠️ val 지표는 (1) final base model early stopping, (2) meta C 탐색,",
        f"> (3) Platt calibration에 val을 모두 사용하므로 낙관적 추정임.",
        f"",
        "---",
        "",
        "## 피처 구성",
        "",
        f"| 카테고리 | 피처 수 |",
        f"|----------|---------|",
    ]
    gdelt_n = sum(1 for c in feature_cols if c.startswith("gdelt_"))
    econ_n  = sum(1 for c in feature_cols if c.startswith("econ_"))
    other_n = len(feature_cols) - gdelt_n - econ_n
    lines += [
        f"| GDELT events | {gdelt_n} |",
        f"| Economic     | {econ_n} |",
        f"| Country      | {other_n} |",
        f"| **합계**     | **{len(feature_cols)}** |",
        "",
        "제거됨: ACLED raw 20개 + acled_missing_mask + macis_se_score",
        "",
        "---",
        "",
        "## val 지표 비교",
        "",
        "| 모델 | PR-AUC | P@5% | R@P≥.10 | R@P≥.20 | Brier | ECE | delta vs Reference |",
        "|------|--------|------|---------|---------|-------|-----|--------------------|",
    ]

    model_order = [
        ("lgbm",             "LightGBM (ACLED-free)"),
        ("xgb",              "XGBoost (ACLED-free)"),
        ("stacking_raw",     "Stacking (raw)"),
        ("stacking_platt",   "Stacking (Platt)"),
        ("stacking_isotonic","Stacking (Isotonic)"),
    ]
    for key, label in model_order:
        if key not in m:
            continue
        mv = m[key]
        pr  = mv.get("pr_auc", float("nan"))
        d_str = delta_str(pr, REFERENCE_PR_AUC) if key == "stacking_platt" else "—"
        lines.append(
            f"| {label} "
            f"| {fmt(mv.get('pr_auc', float('nan')))} "
            f"| {fmt(mv.get('p_at_top5pct', float('nan')))} "
            f"| {fmt(mv.get('recall_at_precision_010', float('nan')))} "
            f"| {fmt(mv.get('recall_at_precision_020', float('nan')))} "
            f"| {fmt(mv.get('brier_score', float('nan')))} "
            f"| {fmt(mv.get('ece', float('nan')))} "
            f"| {d_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## OOF fold 요약",
        "",
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
        "1. Reference (0.2714)는 ACLED + macis_se_score를 포함한 결과이므로 직접 비교 대상이 아님.",
        "2. 실험 C (B + GDELT titles)가 B 대비 +0.003 이상이면 titles 피처 채택 기준 충족.",
        "3. Isotonic ECE ≈ 0은 val 과적합 (val 셋 크기 작음). Platt 채택.",
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
    # Step 0: 검증
    validate_setup()

    # Step 1: 데이터 로드
    train_df, val_df, test_df = load_all_data()

    # Feature cols 결정 및 검증
    feature_cols = get_feature_cols(train_df)
    validate_feature_cols(feature_cols, train_df.columns)

    # Country encoder (XGBoost용)
    country_enc = fit_country_encoder(train_df)
    print(f"\n  XGBoost country encoder: {len(country_enc.classes_)}개 국가")

    # Step 2: OOF
    oof_lgbm, oof_xgb, fold_summaries = run_oof_folds(train_df, feature_cols, country_enc)

    # Step 3: Final base models
    lgbm_final = train_final_lgbm(train_df, val_df, feature_cols)
    xgb_final  = train_final_xgb(train_df, val_df, feature_cols, country_enc)

    # Step 4: Base model predictions
    preds = predict_base_models(lgbm_final, xgb_final, val_df, test_df, feature_cols, country_enc)

    # Step 5: Meta LogReg
    meta_model, best_c = train_meta_logreg(oof_lgbm, oof_xgb, preds)

    # Step 6: Stacking 예측 + Calibration
    print("\n[Step 6] Stacking 예측 + Calibration")
    val_y = preds["val_y"]

    val_stack_raw  = meta_model.predict_proba(
        np.column_stack([preds["val_l"],  preds["val_x"]])
    )[:, 1]
    test_stack_raw = meta_model.predict_proba(
        np.column_stack([preds["test_l"], preds["test_x"]])
    )[:, 1]

    platt_cal, iso_cal = fit_calibrators(val_stack_raw, val_y)

    val_stack_platt  = apply_platt(platt_cal,  val_stack_raw)
    val_stack_iso    = apply_isotonic(iso_cal, val_stack_raw)

    # Step 7: 지표 계산
    print("\n[Step 7] val 지표 계산")
    all_metrics = {
        "lgbm":             compute_metrics(val_y, preds["val_l"]),
        "xgb":              compute_metrics(val_y, preds["val_x"]),
        "stacking_raw":     compute_metrics(val_y, val_stack_raw),
        "stacking_platt":   compute_metrics(val_y, val_stack_platt),
        "stacking_isotonic":compute_metrics(val_y, val_stack_iso),
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
        mv = all_metrics[k]
        flag = " ← best" if abs(mv["pr_auc"] - best_pr) < 1e-6 else ""
        print(f"  {lb:<28} {mv['pr_auc']:>8.4f} {mv['p_at_top5pct']:>8.4f} "
              f"{mv['brier_score']:>8.4f} {mv['ece']:>8.4f}{flag}")
    print(f"\n  Reference (full model) PR-AUC: {REFERENCE_PR_AUC}  "
          f"(ACLED+SE 포함, 직접 비교 대상 아님)")

    # Step 8: 리포트 저장
    save_report(all_metrics, best_c, fold_summaries, feature_cols)

    print()
    print("=" * 70)
    print("  실험 B 완료")
    print(f"  Stacking Platt PR-AUC : {all_metrics['stacking_platt']['pr_auc']:.4f}")
    print(f"  리포트               : {REPORT_MD}")
    print("=" * 70)


if __name__ == "__main__":
    main()
