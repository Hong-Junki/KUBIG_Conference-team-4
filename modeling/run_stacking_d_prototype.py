"""
Category D — Stacking Ensemble Prototype (tree-only, 12-year SE)
================================================================
Experiment : stacking_tree_only_12y
Policy     : full train — acled_missing_mask=1 rows included as training rows;
             acled_missing_mask excluded as a feature.
Base models: LightGBM + 12y SE, XGBoost + 12y SE   (LSTM added later via BASE_MODELS)
Meta model : Logistic Regression  (C searched over val PR-AUC; test never used)
Calibration: Platt Scaling + Isotonic Regression    (fit on val, applied to test)

OOF design : expanding-window (no random KFold)
  F1: train 2014-2017 → predict 2018
  F2: train 2014-2018 → predict 2019
  F3: train 2014-2019 → predict 2020
  F4: train 2014-2020 → predict 2021
  F5: train 2014-2021 → predict 2022
  F6: train 2014-2022 → predict 2023
  Meta-learner trains on 2018-2023 OOF rows (127,078 rows).

NOTE — Final base models use val for early stopping → val metrics mildly optimistic.
NOTE — LSTM will be added when C-category predictions are available (see BASE_MODELS).

Usage (from project root):
    python modeling/run_stacking_d_prototype.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from xgboost.callback import EarlyStopping as XGBEarlyStopping
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve

from evaluate import compute_pr_auc, compute_p_at_top_k, compute_recall_at_precision, compute_ece

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Experiment identity ───────────────────────────────────────────────────────
EXPERIMENT   = "stacking_tree_only_12y"
TRAIN_POLICY = "full"     # "full" | "mask0_only"  — change for ablation
OWNER        = "D_byeonghyeon"
RANDOM_SEED  = 42

# ── Base model registry ───────────────────────────────────────────────────────
# To add LSTM: append "lstm_classifier" and supply OOF/val/test prediction files.
BASE_MODELS = ["lgbm_se_12y", "xgb_se_12y"]

# ── Input paths ───────────────────────────────────────────────────────────────
TRAIN_PATH = os.path.join(_ROOT, "input", "processed", "dataset", "train.parquet")
VAL_PATH   = os.path.join(_ROOT, "input", "processed", "dataset", "val.parquet")
TEST_PATH  = os.path.join(_ROOT, "input", "processed", "dataset", "test.parquet")
SE_PATH    = os.path.join(_ROOT, "output", "macis_12y", "se_scores.parquet")

# ── Output directories ────────────────────────────────────────────────────────
PRED_DIR   = os.path.join(_ROOT, "outputs", "predictions")
MODEL_DIR  = os.path.join(_ROOT, "outputs", "models")
REPORT_DIR = os.path.join(_ROOT, "outputs", "reports")

# ── Column constants ──────────────────────────────────────────────────────────
TARGET_COL = "y_escalation"
DATE_COL   = "date"

LABEL_META_COLS = [
    "y", "y_onset", "y_escalation",
    "fatalities_next3d", "event_count_next3d",
    "past14d_event_count", "past14d_fatalities_mean",
]
ALWAYS_EXCLUDE = [DATE_COL, "acled_missing_mask"]

# ── OOF fold design ───────────────────────────────────────────────────────────
OOF_FOLDS = [
    {"name": "F1", "train_end": 2017, "pred_year": 2018},
    {"name": "F2", "train_end": 2018, "pred_year": 2019},
    {"name": "F3", "train_end": 2019, "pred_year": 2020},
    {"name": "F4", "train_end": 2020, "pred_year": 2021},
    {"name": "F5", "train_end": 2021, "pred_year": 2022},
    {"name": "F6", "train_end": 2022, "pred_year": 2023},
]

# ── Hyperparameters ───────────────────────────────────────────────────────────
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
LGB_ROUNDS_OOF   = 500    # fixed — no early stopping in OOF folds
LGB_ROUNDS_FINAL = 1000   # with val early stopping
EARLY_STOP       = 50
LOG_PERIOD       = 100

XGB_PARAMS_OOF = {
    "objective":        "binary:logistic",
    "eval_metric":      "aucpr",
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 22,
    "seed":             RANDOM_SEED,
    "verbosity":        0,   # silent for OOF folds
}
XGB_PARAMS_FINAL = dict(XGB_PARAMS_OOF)
XGB_PARAMS_FINAL["verbosity"] = 1   # show progress for final model

XGB_ROUNDS_OOF   = 500
XGB_ROUNDS_FINAL = 1000

META_C_CANDIDATES = [0.01, 0.1, 1.0, 10.0]

# Reference PR-AUC from best previous single model (LightGBM+SE mask0_only)
REFERENCE_PR_AUC = 0.1741

# ── Output file registry ──────────────────────────────────────────────────────
OUTPUT_FILES = {
    "oof_lgbm":         os.path.join(PRED_DIR,   f"oof_predictions__lgbm_se_12y__{OWNER}.csv"),
    "oof_xgb":          os.path.join(PRED_DIR,   f"oof_predictions__xgb_se_12y__{OWNER}.csv"),
    "val_lgbm":         os.path.join(PRED_DIR,   f"val_predictions__lgbm_se_12y__{OWNER}.csv"),
    "val_xgb":          os.path.join(PRED_DIR,   f"val_predictions__xgb_se_12y__{OWNER}.csv"),
    "test_lgbm":        os.path.join(PRED_DIR,   f"predictions__lgbm_se_12y__{OWNER}.csv"),
    "test_xgb":         os.path.join(PRED_DIR,   f"predictions__xgb_se_12y__{OWNER}.csv"),
    "val_stack":        os.path.join(PRED_DIR,   f"val_predictions__{EXPERIMENT}__{OWNER}.csv"),
    "test_stack_raw":   os.path.join(PRED_DIR,   f"predictions__{EXPERIMENT}_raw__{OWNER}.csv"),
    "test_stack_platt": os.path.join(PRED_DIR,   f"predictions__{EXPERIMENT}_platt__{OWNER}.csv"),
    "test_stack_iso":   os.path.join(PRED_DIR,   f"predictions__{EXPERIMENT}_isotonic__{OWNER}.csv"),
    "model_lgbm":       os.path.join(MODEL_DIR,  "lgbm_se_12y_D_final.pkl"),
    "model_xgb":        os.path.join(MODEL_DIR,  "xgb_se_12y_D_final.pkl"),
    "model_meta":       os.path.join(MODEL_DIR,  f"meta_logreg_{EXPERIMENT}.pkl"),
    "model_platt":      os.path.join(MODEL_DIR,  f"platt_calibrator_{EXPERIMENT}.pkl"),
    "model_iso":        os.path.join(MODEL_DIR,  f"isotonic_calibrator_{EXPERIMENT}.pkl"),
    "oof_summary":      os.path.join(REPORT_DIR, f"{EXPERIMENT}_oof_summary.json"),
    "val_metrics_json": os.path.join(REPORT_DIR, f"{EXPERIMENT}_val_metrics.json"),
    "val_metrics_md":   os.path.join(REPORT_DIR, f"{EXPERIMENT}_val_metrics.md"),
    "calib_csv":        os.path.join(REPORT_DIR, f"{EXPERIMENT}_calibration.csv"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Safety helpers
# ─────────────────────────────────────────────────────────────────────────────

def check_existing_outputs():
    conflicts = [p for p in OUTPUT_FILES.values() if os.path.exists(p)]
    if not conflicts:
        return
    print()
    print("=" * 70)
    print("  [중단] 이미 존재하는 출력 파일 발견 — 덮어쓰지 않고 종료합니다.")
    print("=" * 70)
    for p in conflicts:
        print(f"  • {os.path.relpath(p, _ROOT)}")
    print()
    sys.exit(1)


def ensure_output_dirs():
    for d in [PRED_DIR, MODEL_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _recall_at_prec(y_true, y_prob, min_prec):
    precision, recall, _ = precision_recall_curve(
        np.asarray(y_true), np.asarray(y_prob)
    )
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
# SE merge
# ─────────────────────────────────────────────────────────────────────────────

def merge_se(df, se_df, split_name):
    """Left-merge 12y SE onto df by date+country; rename; fill nulls with 0."""
    n_before = len(df)
    merged = df.merge(
        se_df[["iso3", "date", "se_score"]],
        left_on=["date", "country"],
        right_on=["date", "iso3"],
        how="left",
    ).drop(columns=["iso3"], errors="ignore").rename(
        columns={"se_score": "macis_se_score"}
    )
    if len(merged) != n_before:
        raise ValueError(
            f"[{split_name}] Row count changed after SE merge: "
            f"{n_before:,} → {len(merged):,}"
        )
    n_null = int(merged["macis_se_score"].isna().sum())
    pct    = n_null / n_before * 100
    print(f"  SE [{split_name}]: {n_null:,}/{n_before:,} null ({pct:.1f}%) → filled 0")
    merged["macis_se_score"] = merged["macis_se_score"].fillna(0.0)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_cols(df):
    exclude = set(LABEL_META_COLS) | set(ALWAYS_EXCLUDE)
    return [c for c in df.columns if c not in exclude]


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
        n_unknown = int((codes == -1).sum())
        if n_unknown:
            print(f"    WARNING: {n_unknown} unknown country codes → -1")
        X["country"] = codes.values.astype(np.int32)
    return X.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data():
    print("\n[Step 1] 데이터 로딩 및 SE 병합")
    se_df  = pd.read_parquet(SE_PATH)
    print(f"  SE 로드: {len(se_df):,}행 | "
          f"{se_df[se_df['se_score'].notna()]['date'].min().date()} ~ "
          f"{se_df[se_df['se_score'].notna()]['date'].max().date()}")

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

    if TRAIN_POLICY == "mask0_only":
        train = train[train["acled_missing_mask"] == 0].copy()
        print(f"  → mask0_only 적용 후 train: {len(train):,}행")

    train = merge_se(train, se_df, "train")
    val   = merge_se(val,   se_df, "val")
    test  = merge_se(test,  se_df, "test")

    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — OOF generation
# ─────────────────────────────────────────────────────────────────────────────

def run_oof_folds(train_df, feature_cols, country_enc):
    print("\n[Step 2] OOF 생성 (expanding-window, 6 folds × 2 base models)")
    print("  OOF fold 모델: 고정 라운드 학습 (early stopping 없음)")

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
        m1   = int((fold_pr["acled_missing_mask"] == 1).sum())

        print(f"\n  {fname}: 학습 {n_tr:,}행 (≤{train_end}) "
              f"→ 예측 {pred_yr} ({n_pr:,}행, mask=1:{m1}, 양성률:{y_pr.mean():.3f})")

        # LightGBM fold
        ds_tr    = lgb.Dataset(
            make_lgbm_X(fold_tr, feature_cols), label=y_tr,
            categorical_feature=["country"], free_raw_data=False,
        )
        lgbm_fold = lgb.train(LGB_PARAMS, ds_tr, num_boost_round=LGB_ROUNDS_OOF)
        prob_l    = lgbm_fold.predict(make_lgbm_X(fold_pr, feature_cols))
        auc_l     = compute_pr_auc(y_pr, prob_l)
        print(f"    LightGBM PR-AUC: {auc_l:.4f}")

        # XGBoost fold
        dtrain_xgb = xgb.DMatrix(make_xgb_X(fold_tr, feature_cols, country_enc), label=y_tr)
        dpred_xgb  = xgb.DMatrix(make_xgb_X(fold_pr, feature_cols, country_enc))
        xgb_fold   = xgb.train(XGB_PARAMS_OOF, dtrain_xgb, num_boost_round=XGB_ROUNDS_OOF)
        prob_x     = xgb_fold.predict(dpred_xgb)
        auc_x      = compute_pr_auc(y_pr, prob_x)
        print(f"    XGBoost  PR-AUC: {auc_x:.4f}")

        dates    = fold_pr[DATE_COL].dt.strftime("%Y-%m-%d").values
        countries = fold_pr["country"].values

        for d, c, yt, pl, px in zip(dates, countries, y_pr, prob_l, prob_x):
            oof_rows_lgbm.append({"date": d, "country": c, "y_true": int(yt), "y_prob_oof": float(pl)})
            oof_rows_xgb.append( {"date": d, "country": c, "y_true": int(yt), "y_prob_oof": float(px)})

        fold_summaries.append({
            "name": fname, "pred_year": pred_yr,
            "n_train": n_tr, "n_pred": n_pr,
            "mask1_in_pred": m1,
            "positive_rate": round(float(y_pr.mean()), 4),
            "lgbm_pr_auc":   round(float(auc_l), 4),
            "xgb_pr_auc":    round(float(auc_x), 4),
        })

    oof_lgbm = pd.DataFrame(oof_rows_lgbm)
    oof_xgb  = pd.DataFrame(oof_rows_xgb)
    total    = len(oof_lgbm)
    m1_total = sum(f["mask1_in_pred"] for f in fold_summaries)

    print(f"\n  OOF 합계: {total:,}행 | mask=1 {m1_total}행 ({m1_total/total*100:.1f}%)")
    return oof_lgbm, oof_xgb, fold_summaries


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Final base model training
# ─────────────────────────────────────────────────────────────────────────────

def train_final_lgbm(train_df, val_df, feature_cols):
    print("\n[Step 3a] Final LightGBM 학습 (full train, val early stopping)")
    X_tr = make_lgbm_X(train_df, feature_cols)
    X_vl = make_lgbm_X(val_df,   feature_cols)
    ds_tr = lgb.Dataset(X_tr, label=train_df[TARGET_COL].values,
                        categorical_feature=["country"], free_raw_data=False)
    ds_vl = lgb.Dataset(X_vl, label=val_df[TARGET_COL].values,
                        reference=ds_tr, free_raw_data=False)
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
    dtrain = xgb.DMatrix(make_xgb_X(train_df, feature_cols, country_enc),
                         label=train_df[TARGET_COL].values)
    dval   = xgb.DMatrix(make_xgb_X(val_df,   feature_cols, country_enc),
                         label=val_df[TARGET_COL].values)
    model = xgb.train(
        XGB_PARAMS_FINAL, dtrain,
        num_boost_round=XGB_ROUNDS_FINAL,
        evals=[(dval, "val")],
        callbacks=[XGBEarlyStopping(rounds=EARLY_STOP, save_best=True)],
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Base model val/test predictions
# ─────────────────────────────────────────────────────────────────────────────

def predict_base_models(lgbm_model, xgb_model, val_df, test_df, feature_cols, country_enc):
    print("\n[Step 4] Base model val/test 예측")

    val_prob_l  = lgbm_model.predict(make_lgbm_X(val_df,  feature_cols))
    val_prob_x  = xgb_model.predict( xgb.DMatrix(make_xgb_X(val_df,  feature_cols, country_enc)))
    test_prob_l = lgbm_model.predict(make_lgbm_X(test_df, feature_cols))
    test_prob_x = xgb_model.predict( xgb.DMatrix(make_xgb_X(test_df, feature_cols, country_enc)))

    val_y   = val_df[TARGET_COL].values
    val_dt  = val_df[DATE_COL].dt.strftime("%Y-%m-%d").values
    val_ct  = val_df["country"].values
    test_dt = test_df[DATE_COL].dt.strftime("%Y-%m-%d").values
    test_ct = test_df["country"].values

    print(f"  LightGBM val PR-AUC : {compute_pr_auc(val_y, val_prob_l):.4f}")
    print(f"  XGBoost  val PR-AUC : {compute_pr_auc(val_y, val_prob_x):.4f}")

    return {
        "val_date": val_dt,  "val_country": val_ct, "val_y": val_y,
        "val_l":  val_prob_l,  "val_x":  val_prob_x,
        "test_date": test_dt, "test_country": test_ct,
        "test_l": test_prob_l, "test_x": test_prob_x,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Meta Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────

def train_meta_logreg(oof_lgbm, oof_xgb, preds):
    print("\n[Step 5] Meta LogReg 학습 + C 탐색 (val PR-AUC 기준)")

    # Align OOF predictions by (date, country)
    oof = oof_lgbm[["date","country","y_true","y_prob_oof"]].merge(
        oof_xgb[["date","country","y_prob_oof"]].rename(columns={"y_prob_oof": "y_prob_oof_x"}),
        on=["date","country"], how="inner",
    )
    X_oof = oof[["y_prob_oof","y_prob_oof_x"]].values
    y_oof = oof["y_true"].values

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


def stack_predict(meta, prob_l, prob_x):
    return meta.predict_proba(np.column_stack([prob_l, prob_x]))[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Calibration
# ─────────────────────────────────────────────────────────────────────────────

def fit_calibrators(val_stack_raw, val_y):
    print("\n[Step 6] Calibration fit (val 기준 → test 적용)")
    platt = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    platt.fit(val_stack_raw.reshape(-1, 1), val_y)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_stack_raw, val_y)
    print("  Platt + Isotonic calibrators fit 완료")
    return platt, iso


def apply_platt(platt, prob):
    return platt.predict_proba(np.asarray(prob).reshape(-1, 1))[:, 1]


def apply_isotonic(iso, prob):
    return iso.predict(np.asarray(prob))


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Report generation
# ─────────────────────────────────────────────────────────────────────────────

def _metric_row(label, m):
    return (
        f"| {label:<28}"
        f"| {m['pr_auc']:.4f} "
        f"| {m['p_at_top5pct']:.4f} "
        f"| {m['recall_at_precision_010']:.4f} "
        f"| {m['recall_at_precision_020']:.4f} "
        f"| {m['recall_at_precision_030']:.4f} "
        f"| {m['brier_score']:.4f} "
        f"| {m['ece']:.4f} |"
    )


def build_val_metrics_md(all_metrics, best_c):
    stacking_pr = all_metrics.get("stacking_isotonic", {}).get("pr_auc",
                  all_metrics.get("stacking_raw", {}).get("pr_auc", 0.0))
    improved = "✓ 개선됨" if stacking_pr > REFERENCE_PR_AUC else "✗ 미개선"

    rows = "\n".join([
        _metric_row("LightGBM + 12y SE",       all_metrics["lgbm_se_12y"]),
        _metric_row("XGBoost + 12y SE",         all_metrics["xgb_se_12y"]),
        _metric_row("Stacking (raw)",            all_metrics["stacking_raw"]),
        _metric_row("Stacking (Platt)",          all_metrics["stacking_platt"]),
        _metric_row("Stacking (Isotonic)",       all_metrics["stacking_isotonic"]),
    ])

    return f"""\
# 검증 지표 — {EXPERIMENT}

**실험**: {EXPERIMENT}
**정책**: {TRAIN_POLICY}  (acled_missing_mask=1 포함, 피처에서 제외)
**Base 모델**: LightGBM + 12y SE, XGBoost + 12y SE (tree-only 1차 프로토타입)
**Meta 모델**: Logistic Regression (선택된 C = {best_c})
**비교 기준**: 단일 모델 LightGBM+SE mask0_only val PR-AUC = {REFERENCE_PR_AUC}

> ⚠ Final base 모델은 val을 early stopping에 사용 → val 지표가 소폭 낙관적일 수 있음.
> LSTM은 C담당 파일 수령 후 추가 예정 (BASE_MODELS 리스트에 항목 추가).

## 지표 비교

| 모델                         | PR-AUC | P@5%   | R@P≥.10 | R@P≥.20 | R@P≥.30 | Brier  | ECE    |
|------------------------------|--------|--------|---------|---------|---------|--------|--------|
{rows}

## 기준 대비 평가

- 단일 모델 기준 PR-AUC: **{REFERENCE_PR_AUC}**
- Stacking (Isotonic) PR-AUC: **{stacking_pr:.4f}**
- 결과: **{improved}**

## 다음 단계

1. `stacking_tree_only_12y_mask0` ablation 실행
   (mask=0 행만으로 재학습 → full-train vs mask0-only 비교)
2. C담당 LSTM 예측 파일 수령 후 BASE_MODELS에 추가 후 재실행
3. Isotonic calibrated test 파일을 대시보드 F 항목 후보로 사용
"""


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Save all outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_all(oof_lgbm, oof_xgb, preds, val_stack_df,
             test_stack_raw, test_stack_platt, test_stack_iso,
             lgbm_final, xgb_final, meta_model, platt_cal, iso_cal,
             fold_summaries, all_metrics, best_c):
    print("\n[Step 8] 결과 저장")

    # OOF prediction files
    oof_lgbm.to_csv(OUTPUT_FILES["oof_lgbm"], index=False)
    oof_xgb.to_csv( OUTPUT_FILES["oof_xgb"],  index=False)

    # Base model val predictions
    for key, y_prob in [("val_lgbm", preds["val_l"]), ("val_xgb", preds["val_x"])]:
        pd.DataFrame({
            "date": preds["val_date"], "country": preds["val_country"],
            "y_true": preds["val_y"], "y_prob": y_prob,
        }).to_csv(OUTPUT_FILES[key], index=False)

    # Base model test predictions
    for key, y_prob in [("test_lgbm", preds["test_l"]), ("test_xgb", preds["test_x"])]:
        pd.DataFrame({
            "date": preds["test_date"], "country": preds["test_country"], "y_prob": y_prob,
        }).to_csv(OUTPUT_FILES[key], index=False)

    # Stacking val prediction (all variants in one file)
    val_stack_df.to_csv(OUTPUT_FILES["val_stack"], index=False)

    # Stacking test predictions
    for key, prob in [
        ("test_stack_raw",   test_stack_raw),
        ("test_stack_platt", test_stack_platt),
        ("test_stack_iso",   test_stack_iso),
    ]:
        pd.DataFrame({
            "date": preds["test_date"], "country": preds["test_country"], "y_prob": prob,
        }).to_csv(OUTPUT_FILES[key], index=False)

    print(f"  예측 파일 저장: {len(OUTPUT_FILES)//2} 파일")

    # Model artifacts
    joblib.dump(lgbm_final, OUTPUT_FILES["model_lgbm"])
    joblib.dump(xgb_final,  OUTPUT_FILES["model_xgb"])
    joblib.dump(meta_model, OUTPUT_FILES["model_meta"])
    joblib.dump(platt_cal,  OUTPUT_FILES["model_platt"])
    joblib.dump(iso_cal,    OUTPUT_FILES["model_iso"])
    print("  모델 아티팩트 저장: 5개")

    # OOF summary JSON
    total_oof = len(oof_lgbm)
    m1_oof    = sum(f["mask1_in_pred"] for f in fold_summaries)
    oof_summary = {
        "experiment":      EXPERIMENT,
        "train_policy":    TRAIN_POLICY,
        "oof_total_rows":  total_oof,
        "oof_mask1_rows":  m1_oof,
        "oof_mask1_pct":   round(m1_oof / total_oof * 100, 2),
        "note":            "2014-2017 rows excluded from OOF. Meta-learner trains on 2018-2023.",
        "folds":           fold_summaries,
    }
    with open(OUTPUT_FILES["oof_summary"], "w") as f:
        json.dump(oof_summary, f, indent=2, ensure_ascii=False)

    # Val metrics JSON
    val_metrics_out = {
        "experiment":                EXPERIMENT,
        "train_policy":              TRAIN_POLICY,
        "best_meta_C":               best_c,
        "reference_single_pr_auc":   REFERENCE_PR_AUC,
        "note_early_stopping":       "Final base models use val for early stopping; val metrics mildly optimistic.",
        "note_lstm":                 "LSTM not included. Add to BASE_MODELS when C-category files are available.",
        "models":                    all_metrics,
    }
    with open(OUTPUT_FILES["val_metrics_json"], "w") as f:
        json.dump(val_metrics_out, f, indent=2, ensure_ascii=False)

    # Val metrics MD
    md = build_val_metrics_md(all_metrics, best_c)
    with open(OUTPUT_FILES["val_metrics_md"], "w", encoding="utf-8") as f:
        f.write(md)

    # Calibration CSV
    calib_rows = [
        {"model": k, **{mk: mv for mk, mv in all_metrics[k].items()
                        if mk not in ("positive_rate","n_samples")}}
        for k in ["stacking_raw", "stacking_platt", "stacking_isotonic"]
        if k in all_metrics
    ]
    pd.DataFrame(calib_rows).to_csv(OUTPUT_FILES["calib_csv"], index=False)

    print("  리포트 저장: 4개")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"  [카테고리 D] Stacking Prototype: {EXPERIMENT}")
    print(f"  정책: {TRAIN_POLICY} | Base: LightGBM + XGBoost | SE: 12y")
    print("=" * 70)

    check_existing_outputs()
    ensure_output_dirs()

    # ── 1. Data loading ───────────────────────────────────────────────────────
    train_df, val_df, test_df = load_all_data()

    # ── 2. Feature preparation ────────────────────────────────────────────────
    print("\n[Step 2a] 피처 준비")
    feature_cols = get_feature_cols(train_df)
    se_cols = [c for c in feature_cols if "se" in c.lower()]
    print(f"  피처 수: {len(feature_cols)}  (SE 포함: {se_cols})")

    country_enc = fit_country_encoder(train_df)
    print(f"  XGBoost country encoder: {len(country_enc.classes_)}개 국가 fit")

    # ── 3. OOF generation ─────────────────────────────────────────────────────
    oof_lgbm, oof_xgb, fold_summaries = run_oof_folds(train_df, feature_cols, country_enc)

    # ── 4. Final base models ──────────────────────────────────────────────────
    lgbm_final = train_final_lgbm(train_df, val_df, feature_cols)
    xgb_final  = train_final_xgb(train_df, val_df, feature_cols, country_enc)

    # ── 5. Base model val/test predictions ────────────────────────────────────
    preds = predict_base_models(lgbm_final, xgb_final, val_df, test_df,
                                feature_cols, country_enc)

    # ── 6. Meta LogReg ────────────────────────────────────────────────────────
    meta_model, best_c = train_meta_logreg(oof_lgbm, oof_xgb, preds)

    # ── 7. Stacking predictions ───────────────────────────────────────────────
    print("\n[Step 7] Stacking 예측 생성")
    val_stack_raw  = stack_predict(meta_model, preds["val_l"],  preds["val_x"])
    test_stack_raw = stack_predict(meta_model, preds["test_l"], preds["test_x"])

    # ── 8. Calibration ────────────────────────────────────────────────────────
    platt_cal, iso_cal = fit_calibrators(val_stack_raw, preds["val_y"])

    val_stack_platt  = apply_platt(platt_cal,    val_stack_raw)
    val_stack_iso    = apply_isotonic(iso_cal,   val_stack_raw)
    test_stack_platt = apply_platt(platt_cal,    test_stack_raw)
    test_stack_iso   = apply_isotonic(iso_cal,   test_stack_raw)

    # ── 9. Metrics ────────────────────────────────────────────────────────────
    print("\n[Step 9] val 지표 계산")
    val_y = preds["val_y"]
    all_metrics = {
        "lgbm_se_12y":       compute_metrics(val_y, preds["val_l"]),
        "xgb_se_12y":        compute_metrics(val_y, preds["val_x"]),
        "stacking_raw":      compute_metrics(val_y, val_stack_raw),
        "stacking_platt":    compute_metrics(val_y, val_stack_platt),
        "stacking_isotonic": compute_metrics(val_y, val_stack_iso),
    }

    labels = {
        "lgbm_se_12y":       "LightGBM+SE 12y   ",
        "xgb_se_12y":        "XGBoost+SE 12y    ",
        "stacking_raw":      "Stacking (raw)    ",
        "stacking_platt":    "Stacking (Platt)  ",
        "stacking_isotonic": "Stacking (Isotonic)",
    }
    print(f"\n  {'모델':<24} {'PR-AUC':>8} {'P@5%':>8} {'Brier':>8} {'ECE':>8}")
    print(f"  {'-'*56}")
    for k, lb in labels.items():
        m = all_metrics[k]
        flag = " ← best" if m["pr_auc"] == max(x["pr_auc"] for x in all_metrics.values()) else ""
        print(f"  {lb:<24} {m['pr_auc']:>8.4f} {m['p_at_top5pct']:>8.4f} "
              f"{m['brier_score']:>8.4f} {m['ece']:>8.4f}{flag}")

    # Stacking val output (multi-column)
    val_stack_df = pd.DataFrame({
        "date":                  preds["val_date"],
        "country":               preds["val_country"],
        "y_true":                val_y,
        "y_prob_lgbm":           preds["val_l"],
        "y_prob_xgb":            preds["val_x"],
        "y_prob_stack_raw":      val_stack_raw,
        "y_prob_stack_platt":    val_stack_platt,
        "y_prob_stack_isotonic": val_stack_iso,
    })

    # ── 10. Save all outputs ──────────────────────────────────────────────────
    save_all(
        oof_lgbm, oof_xgb, preds, val_stack_df,
        test_stack_raw, test_stack_platt, test_stack_iso,
        lgbm_final, xgb_final, meta_model, platt_cal, iso_cal,
        fold_summaries, all_metrics, best_c,
    )

    best_pr = max(m["pr_auc"] for m in all_metrics.values())
    print()
    print("=" * 70)
    print("  실행 완료")
    print(f"  최고 val PR-AUC : {best_pr:.4f}")
    print(f"  단일 모델 기준  : {REFERENCE_PR_AUC}")
    print(f"  비교 결과       : {'개선됨' if best_pr > REFERENCE_PR_AUC else '미개선'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
