"""
F2_clean Feature Group Ablation
================================
F2_clean (94개)에서 feature group을 하나씩 제거해 각 group의 성능 기여를 측정한다.

cleanval split (F2_clean full run과 동일):
  train_fit  : 2014-01-01 ~ 2022-12-31
  tune_cal   : 2023-01-01 ~ 2023-12-31
  val_eval   : 2024-01-01 ~ 2024-06-30 (순수 평가 전용)
  test       : 사용 금지

Baseline:
  F2_full (94개) Stacking Platt PR-AUC = 0.1027

Ablation 목록:
  F2_no_country          : country 1개 제거  → 93개
  F2_no_safe_acled       : safe_acled 15개 제거 → 79개
  F2_no_gdelt_title      : gdelt_title 21개 + coverage_mask 1개 제거 → 72개
  F2_no_gdelt_theme_person: gdelt_theme/person 22개 제거 → 72개
  F2_no_economic         : econ_* 15개 제거 → 79개
  F2_no_gdelt_events     : gdelt events 19개 제거 → 75개

실행 순서 (우선순위):
  1. no_country  2. no_gdelt_theme_person  3. no_gdelt_title
  4. no_safe_acled  5. no_economic  6. no_gdelt_events

출력:
  members/byeonghyeon/outputs/reports/f2_clean_group_ablation.md

실행:
  cd <KUBIG_Conference-team-4 루트>
  python members/byeonghyeon/modeling/analyze_f2_clean_group_ablation.py

  특정 ablation만 실행 (쉼표 구분):
  ABLATIONS=no_country,no_gdelt_theme_person \\
    python members/byeonghyeon/modeling/analyze_f2_clean_group_ablation.py
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

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_SUBTREE_ROOT = os.path.dirname(_SCRIPT_DIR)

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_SUBTREE_ROOT))),
                 "conflict-early-warning"),
)

sys.path.insert(0, _SCRIPT_DIR)
from evaluate import compute_pr_auc, compute_p_at_top_k, compute_recall_at_precision, compute_ece

TRAIN_PATH       = os.path.join(DATA_ROOT, "input", "processed", "dataset", "train.parquet")
VAL_PATH         = os.path.join(DATA_ROOT, "input", "processed", "dataset", "val.parquet")
SAFE_ACLED_PATH  = os.path.join(_SUBTREE_ROOT, "input", "processed", "acled_safe",
                                "safe_acled_lag_features.parquet")
GDELT_TITLE_PATH = os.path.join(_SUBTREE_ROOT, "input", "processed", "gdelt_titles",
                                "gdelt_title_features.parquet")
GDELT_TP_PATH    = os.path.join(_SUBTREE_ROOT, "input", "processed", "gdelt_titles",
                                "gdelt_theme_person_features.parquet")

REPORT_DIR = os.path.join(_SUBTREE_ROOT, "outputs", "reports")
REPORT_MD  = os.path.join(REPORT_DIR, "f2_clean_group_ablation.md")

TRAIN_FIT_END  = pd.Timestamp("2022-12-31", tz="UTC")
TUNE_CAL_START = pd.Timestamp("2023-01-01", tz="UTC")
TUNE_CAL_END   = pd.Timestamp("2023-12-31", tz="UTC")
GDELT_TITLE_MIN_DATE     = "2015-02-17"
GDELT_TITLE_COVERAGE_COL = "gdelt_title_coverage_mask"

TARGET_COL  = "y_escalation"
DATE_COL    = "date"
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

# ── 모델 하이퍼파라미터 ───────────────────────────────────────────────────────
LGB_PARAMS = {
    "objective": "binary", "metric": "average_precision",
    "boosting_type": "gbdt", "num_leaves": 63, "learning_rate": 0.05,
    "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.8, "scale_pos_weight": 22,
    "seed": RANDOM_SEED, "verbose": -1,
}
XGB_PARAMS_OOF = {
    "objective": "binary:logistic", "eval_metric": "aucpr",
    "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
    "colsample_bytree": 0.8, "scale_pos_weight": 22,
    "seed": RANDOM_SEED, "verbosity": 0,
}
XGB_PARAMS_FINAL = {**XGB_PARAMS_OOF}
LGB_ROUNDS_OOF = 500; LGB_ROUNDS_FINAL = 1000
XGB_ROUNDS_OOF = 500; XGB_ROUNDS_FINAL = 1000
EARLY_STOP = 50; LOG_PERIOD = 100

OOF_FOLDS = [
    {"name": "OOF_F1", "train_end": 2017, "pred_year": 2018},
    {"name": "OOF_F2", "train_end": 2018, "pred_year": 2019},
    {"name": "OOF_F3", "train_end": 2019, "pred_year": 2020},
    {"name": "OOF_F4", "train_end": 2020, "pred_year": 2021},
    {"name": "OOF_F5", "train_end": 2021, "pred_year": 2022},
]

META_C_CANDIDATES = [0.01, 0.1, 1.0, 10.0]

# F2_full baseline
F2_FULL_PR_AUC = 0.1027


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("[Step 1] 데이터 로딩 + feature merge (1회, 전 실험 공유)")
    train_full = pd.read_parquet(TRAIN_PATH)
    val_eval   = pd.read_parquet(VAL_PATH)

    for df in [train_full, val_eval]:
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("UTC")

    train_fit = train_full[train_full["date"] <= TRAIN_FIT_END].copy()
    tune_cal  = train_full[(train_full["date"] >= TUNE_CAL_START) &
                           (train_full["date"] <= TUNE_CAL_END)].copy()

    print(f"  train_fit: {len(train_fit):,}행 | tune_cal: {len(tune_cal):,}행 | "
          f"val_eval: {len(val_eval):,}행")
    print("  test: 사용 금지")

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
        df[SAFE_ACLED_FEATURE_COLS] = df[SAFE_ACLED_FEATURE_COLS].fillna(0)
        df = df.merge(title_merge, on=["date","country"], how="left"); assert len(df)==n
        df[GDELT_TITLE_FEATURE_COLS] = df[GDELT_TITLE_FEATURE_COLS].fillna(0)
        df[GDELT_TITLE_COVERAGE_COL] = (df["date"] < coverage_ts).astype(int)
        df = df.merge(tp_merge,    on=["date","country"], how="left"); assert len(df)==n
        df[GDELT_TP_FEATURE_COLS] = df[GDELT_TP_FEATURE_COLS].fillna(0)
        return df

    train_fit = merge_all(train_fit, "train_fit")
    tune_cal  = merge_all(tune_cal,  "tune_cal")
    val_eval  = merge_all(val_eval,  "val_eval")
    print("  merge 완료 ✅")
    return {"train_fit": train_fit, "tune_cal": tune_cal, "val_eval": val_eval}


# ─────────────────────────────────────────────────────────────────────────────
# feature cols 구성
# ─────────────────────────────────────────────────────────────────────────────

def _base_cols(df):
    exclude = set(LABEL_META_COLS) | set(ALWAYS_EXCLUDE) | _ALL_EXTRA
    return [c for c in df.columns if c not in exclude]


def get_f2_full_cols(df):
    base = _base_cols(df)
    return (base + SAFE_ACLED_FEATURE_COLS + GDELT_TITLE_FEATURE_COLS
            + [GDELT_TITLE_COVERAGE_COL] + GDELT_TP_FEATURE_COLS)


def get_ablation_cols(df, ablation_name):
    """F2_full에서 해당 group을 제거한 feature_cols."""
    full = get_f2_full_cols(df)
    full_set = set(full)

    remove = set()
    if ablation_name == "no_country":
        remove = {"country"}
    elif ablation_name == "no_safe_acled":
        remove = set(SAFE_ACLED_FEATURE_COLS)
    elif ablation_name == "no_gdelt_title":
        remove = set(GDELT_TITLE_FEATURE_COLS) | {GDELT_TITLE_COVERAGE_COL}
    elif ablation_name == "no_gdelt_theme_person":
        remove = set(GDELT_TP_FEATURE_COLS)
    elif ablation_name == "no_economic":
        remove = {c for c in full if c.startswith("econ_")}
    elif ablation_name == "no_gdelt_events":
        # gdelt event features: gdelt_goldstein_*, gdelt_tone_*, gdelt_mentions_*,
        #                       gdelt_event_count_*, gdelt_quadclass_*
        remove = {c for c in full
                  if c.startswith("gdelt_")
                  and c not in set(GDELT_TITLE_FEATURE_COLS)
                  and c not in set(GDELT_TP_FEATURE_COLS)
                  and c != GDELT_TITLE_COVERAGE_COL}
    else:
        raise ValueError(f"Unknown ablation: {ablation_name}")

    return [c for c in full if c not in remove]


def validate_cols(feature_cols, name):
    old_acled = [c for c in feature_cols
                 if (c.startswith("acled_") and not c.startswith("safe_acled_"))
                 or c == "macis_se_score"]
    if old_acled:
        print(f"  [오류] {name}: 기존 ACLED/SE 잔존 {old_acled}")
        sys.exit(1)
    label_leaked = [c for c in feature_cols if c in set(LABEL_META_COLS)]
    if label_leaked:
        print(f"  [오류] {name}: label/future 잔존 {label_leaked}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼: matrix 생성
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


def _recall_at_prec(y_true, y_prob, min_prec):
    precision, recall, _ = precision_recall_curve(np.asarray(y_true), np.asarray(y_prob))
    valid = precision >= min_prec
    return float(recall[valid].max()) if valid.any() else 0.0


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob, dtype=float)
    return {
        "pr_auc":      compute_pr_auc(y_true, y_prob),
        "p_at_top5pct":compute_p_at_top_k(y_true, y_prob, k=0.05),
        "r_at_p010":   compute_recall_at_precision(y_true, y_prob, min_precision=0.10),
        "brier":       float(np.mean((y_prob - y_true.astype(float))**2)),
        "ece":         compute_ece(y_true, y_prob),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 단일 ablation 실험 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(name: str, feature_cols: list, splits: dict, country_enc: LabelEncoder):
    """
    하나의 ablation 실험:
      OOF (train_fit) → final models (tune_cal early stop) → meta (tune_cal C 선택)
      → calibration (tune_cal fit) → val_eval 평가
    """
    print(f"\n  {'─'*60}")
    print(f"  실험: {name}  ({len(feature_cols)}개 feature)")
    print(f"  {'─'*60}")

    validate_cols(feature_cols, name)

    train_fit = splits["train_fit"]
    tune_cal  = splits["tune_cal"]
    val_eval  = splits["val_eval"]

    has_country = "country" in feature_cols
    cat_feat    = ["country"] if has_country else []

    # ── OOF (train_fit 내부) ─────────────────────────────────────────────────
    oof_rows_l, oof_rows_x = [], []
    for fold in OOF_FOLDS:
        fname, te, py = fold["name"], fold["train_end"], fold["pred_year"]
        fold_tr = train_fit[train_fit[DATE_COL].dt.year <= te]
        fold_pr = train_fit[train_fit[DATE_COL].dt.year == py]
        y_tr = fold_tr[TARGET_COL].values; y_pr = fold_pr[TARGET_COL].values

        ds_tr = lgb.Dataset(make_lgbm_X(fold_tr, feature_cols), label=y_tr,
                            categorical_feature=cat_feat, free_raw_data=False)
        lgbf  = lgb.train(LGB_PARAMS, ds_tr, num_boost_round=LGB_ROUNDS_OOF)
        pl    = lgbf.predict(make_lgbm_X(fold_pr, feature_cols))

        dtf = xgb.DMatrix(make_xgb_X(fold_tr, feature_cols, country_enc), label=y_tr)
        xgbf = xgb.train(XGB_PARAMS_OOF, dtf, num_boost_round=XGB_ROUNDS_OOF)
        px   = xgbf.predict(xgb.DMatrix(make_xgb_X(fold_pr, feature_cols, country_enc)))

        dates = fold_pr[DATE_COL].dt.strftime("%Y-%m-%d").values
        ctries = fold_pr["country"].values
        for d,c,yt,pli,pxi in zip(dates,ctries,y_pr,pl,px):
            oof_rows_l.append({"date":d,"country":c,"y_true":int(yt),"y_prob_oof":float(pli)})
            oof_rows_x.append({"date":d,"country":c,"y_true":int(yt),"y_prob_oof":float(pxi)})

    oof_l = pd.DataFrame(oof_rows_l)
    oof_x = pd.DataFrame(oof_rows_x)

    # ── Final base models (train_fit, early stop on tune_cal) ────────────────
    ds_tr = lgb.Dataset(make_lgbm_X(train_fit, feature_cols), label=train_fit[TARGET_COL].values,
                        categorical_feature=cat_feat, free_raw_data=False)
    ds_tc = lgb.Dataset(make_lgbm_X(tune_cal, feature_cols), label=tune_cal[TARGET_COL].values,
                        reference=ds_tr, free_raw_data=False)
    lgb_final = lgb.train(
        LGB_PARAMS, ds_tr, num_boost_round=LGB_ROUNDS_FINAL,
        valid_sets=[ds_tc], valid_names=["tune_cal"],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
                   lgb.log_evaluation(period=LOG_PERIOD)],
    )
    print(f"    LGB best_iter: {lgb_final.best_iteration}")

    dtrain = xgb.DMatrix(make_xgb_X(train_fit, feature_cols, country_enc),
                         label=train_fit[TARGET_COL].values)
    dtune  = xgb.DMatrix(make_xgb_X(tune_cal, feature_cols, country_enc),
                         label=tune_cal[TARGET_COL].values)
    xgb_final = xgb.train(
        XGB_PARAMS_FINAL, dtrain, num_boost_round=XGB_ROUNDS_FINAL,
        evals=[(dtune,"tune_cal")],
        callbacks=[XGBEarlyStopping(rounds=EARLY_STOP, save_best=True)],
        verbose_eval=LOG_PERIOD,
    )
    print(f"    XGB best_iter: {xgb_final.best_iteration}")

    # ── Tune_cal predictions (meta C 선택 + calibration fit) ─────────────────
    tune_y      = tune_cal[TARGET_COL].values
    tune_prob_l = lgb_final.predict(make_lgbm_X(tune_cal, feature_cols))
    tune_prob_x = xgb_final.predict(xgb.DMatrix(make_xgb_X(tune_cal, feature_cols, country_enc)))
    print(f"    tune_cal → LGB={compute_pr_auc(tune_y,tune_prob_l):.4f} | "
          f"XGB={compute_pr_auc(tune_y,tune_prob_x):.4f}")

    # ── Meta LogReg (OOF 학습, tune_cal C 선택) ──────────────────────────────
    oof = oof_l[["date","country","y_true","y_prob_oof"]].merge(
        oof_x[["date","country","y_prob_oof"]].rename(columns={"y_prob_oof":"y_prob_oof_x"}),
        on=["date","country"], how="inner")
    X_oof = oof[["y_prob_oof","y_prob_oof_x"]].values
    y_oof = oof["y_true"].values
    X_tune_meta = np.column_stack([tune_prob_l, tune_prob_x])

    best_c, best_auc = None, -1.0
    for c in META_C_CANDIDATES:
        m = LogisticRegression(class_weight="balanced", C=c, solver="lbfgs",
                               max_iter=1000, random_state=RANDOM_SEED)
        m.fit(X_oof, y_oof)
        auc = compute_pr_auc(tune_y, m.predict_proba(X_tune_meta)[:,1])
        if auc > best_auc:
            best_auc, best_c = auc, c
    meta = LogisticRegression(class_weight="balanced", C=best_c, solver="lbfgs",
                              max_iter=1000, random_state=RANDOM_SEED)
    meta.fit(X_oof, y_oof)
    print(f"    meta C={best_c} (tune_cal PR-AUC {best_auc:.4f})")

    # ── Calibration fit on tune_cal ───────────────────────────────────────────
    tune_stack_raw = meta.predict_proba(X_tune_meta)[:,1]
    platt = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    platt.fit(tune_stack_raw.reshape(-1,1), tune_y)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(tune_stack_raw, tune_y)

    # ── val_eval 평가 (fit 없음) ──────────────────────────────────────────────
    val_y      = val_eval[TARGET_COL].values
    val_prob_l = lgb_final.predict(make_lgbm_X(val_eval, feature_cols))
    val_prob_x = xgb_final.predict(xgb.DMatrix(make_xgb_X(val_eval, feature_cols, country_enc)))

    X_val_meta    = np.column_stack([val_prob_l, val_prob_x])
    val_stack_raw = meta.predict_proba(X_val_meta)[:,1]
    val_platt     = platt.predict_proba(val_stack_raw.reshape(-1,1))[:,1]
    val_iso       = iso.predict(val_stack_raw)

    metrics = {
        "lgbm":              compute_metrics(val_y, val_prob_l),
        "xgb":               compute_metrics(val_y, val_prob_x),
        "stacking_platt":    compute_metrics(val_y, val_platt),
        "stacking_isotonic": compute_metrics(val_y, val_iso),
    }
    platt_pr = metrics["stacking_platt"]["pr_auc"]
    delta    = platt_pr - F2_FULL_PR_AUC
    d_str    = f"{delta:+.4f} {'↓' if delta < 0 else '↑'}"
    print(f"    val_eval Stacking Platt PR-AUC: {platt_pr:.4f}  (F2_full delta: {d_str})")

    return {
        "name":         name,
        "feature_n":    len(feature_cols),
        "metrics":      metrics,
        "platt_pr_auc": platt_pr,
        "delta":        delta,
        "best_c":       best_c,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 리포트 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_report(results: list):
    today = dt_date.today().isoformat()

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and v == v else "—"

    def delta_str(d):
        if abs(d) < 0.0005: return f"{d:.4f} ≈"
        return f"{d:+.4f} {'↓' if d < 0 else '↑'}"

    # 하락폭 기준 정렬
    sorted_results = sorted(results, key=lambda r: r["delta"])

    lines = [
        f"# F2_clean Feature Group Ablation 결과",
        f"",
        f"**분석일**: {today}  ",
        f"**F2_full baseline (val_eval Stacking Platt PR-AUC)**: {F2_FULL_PR_AUC}  ",
        f"**F2_full feature 수**: 94개  ",
        f"",
        "## Split (cleanval 동일)",
        "",
        "| Split | 기간 | 역할 |",
        "|-------|------|------|",
        "| train_fit | 2014–2022 | base model 학습, OOF |",
        "| tune_cal  | 2023      | early stopping, meta C, calibration fit |",
        "| val_eval  | 2024-H1   | 순수 평가 전용 |",
        "| test      | 미사용    | 평가 금지 |",
        "",
        "---",
        "",
        "## Ablation 요약 (PR-AUC 하락폭 순)",
        "",
        f"| 실험 | feature 수 | Stacking Platt PR-AUC | F2_full delta |",
        f"|------|-----------|----------------------|--------------|",
        f"| **F2_full (baseline)** | 94 | **{F2_FULL_PR_AUC:.4f}** | — |",
    ]
    for r in sorted_results:
        d = r["delta"]
        bold = "**" if abs(d) > 0.005 else ""
        lines.append(
            f"| {bold}F2_{r['name']}{bold} | {r['feature_n']} "
            f"| {bold}{fmt(r['platt_pr_auc'])}{bold} "
            f"| {bold}{delta_str(d)}{bold} |"
        )

    # 상세 지표 비교
    lines += [
        "",
        "---",
        "",
        "## Ablation별 상세 지표",
        "",
        "| 실험 | LGB PR-AUC | XGB PR-AUC | Platt PR-AUC | P@5% | R@P≥.10 | Brier | ECE |",
        "|------|-----------|-----------|-------------|------|---------|-------|-----|",
    ]
    for r in sorted_results:
        mv_p = r["metrics"]["stacking_platt"]
        mv_l = r["metrics"]["lgbm"]
        mv_x = r["metrics"]["xgb"]
        lines.append(
            f"| F2_{r['name']} "
            f"| {fmt(mv_l['pr_auc'])} "
            f"| {fmt(mv_x['pr_auc'])} "
            f"| {fmt(mv_p['pr_auc'])} "
            f"| {fmt(mv_p['p_at_top5pct'])} "
            f"| {fmt(mv_p['r_at_p010'])} "
            f"| {fmt(mv_p['brier'])} "
            f"| {fmt(mv_p['ece'])} |"
        )

    # Group 기여도 순위
    lines += [
        "",
        "---",
        "",
        "## Feature Group 기여도 순위",
        "",
        "> PR-AUC 하락폭이 클수록 해당 group의 기여도가 높음",
        "",
        "| 순위 | Group (제거된 것) | PR-AUC 하락 | 기여도 해석 |",
        "|------|-----------------|------------|------------|",
    ]
    for rank, r in enumerate(sorted_results, 1):
        drop = abs(r["delta"])
        if drop > 0.015:     interp = "핵심 — 제거 시 대폭 하락"
        elif drop > 0.005:   interp = "유의미 — 제거 시 유의미한 하락"
        elif drop > 0.001:   interp = "부분적 — 소폭 하락"
        elif drop > -0.001:  interp = "거의 없음 — 다른 feature로 대체 가능"
        else:                interp = "역효과 — 제거 시 오히려 개선"
        lines.append(
            f"| {rank} | **{r['name']}** | {delta_str(r['delta'])} | {interp} |"
        )

    # 질문별 해석
    result_map = {r["name"]: r for r in results}

    def pr_of(name):
        return result_map[name]["platt_pr_auc"] if name in result_map else None

    def delta_of(name):
        return result_map[name]["delta"] if name in result_map else None

    def interp_delta(name, label, question):
        d = delta_of(name)
        if d is None:
            return f"  - **{question}**: 미실행"
        return (f"  - **{question}**: {label} 제거 시 PR-AUC {pr_of(name):.4f} "
                f"(F2_full delta: {delta_str(d)})")

    lines += [
        "",
        "---",
        "",
        "## 질문별 해석",
        "",
    ]

    # country
    d_c = delta_of("no_country")
    if d_c is not None:
        if abs(d_c) < 0.002:
            interp_c = "country 제거 시 성능이 거의 유지됨 → 국가 고정효과는 다른 feature로 흡수 가능."
        elif d_c < -0.005:
            interp_c = "country 제거 시 성능이 유의미하게 하락 → 모델이 국가 정보에 의존. feature importance 24.6%가 실제로 기여함."
        else:
            interp_c = "country 제거 시 소폭 하락."
        lines.append(f"**Q1. country 제거 시 성능이 얼마나 떨어지는가?**")
        lines.append(f"→ {interp_c}  ")
        lines.append(f"  (제거 전: {F2_FULL_PR_AUC:.4f}, 제거 후: {pr_of('no_country'):.4f}, delta: {delta_str(d_c)})")
        lines.append("")

    # theme/person
    d_tp = delta_of("no_gdelt_theme_person")
    if d_tp is not None:
        if d_tp < -0.008:
            interp_tp = "theme/person 제거 시 PR-AUC가 크게 하락 → F2-F1 delta +0.0191의 직접 원인 확인됨."
        elif d_tp < -0.002:
            interp_tp = "theme/person 제거 시 유의미한 하락 → F2 개선분의 일부를 theme/person이 담당."
        else:
            interp_tp = "theme/person 제거해도 성능 유지 → F2 개선은 다른 feature와의 상호작용 효과일 가능성."
        lines.append(f"**Q2. theme/person 제거 시 F2 개선분이 사라지는가?**")
        lines.append(f"→ {interp_tp}  ")
        lines.append(f"  (제거 후: {pr_of('no_gdelt_theme_person'):.4f}, delta: {delta_str(d_tp)})")
        lines.append("")

    # gdelt_title
    d_t = delta_of("no_gdelt_title")
    if d_t is not None:
        if d_t < -0.005:
            interp_t = "title 제거 시 유의미한 하락 → F1-F0 delta +0.0055의 효과 일부 확인됨."
        else:
            interp_t = "title 제거해도 성능 유지 → title의 정보가 다른 GDELT 피처로 흡수됐을 가능성."
        lines.append(f"**Q3. title 제거 시 F1 효과가 사라지는가?**")
        lines.append(f"→ {interp_t}  ")
        lines.append(f"  (제거 후: {pr_of('no_gdelt_title'):.4f}, delta: {delta_str(d_t)})")
        lines.append("")

    # safe_acled
    d_a = delta_of("no_safe_acled")
    if d_a is not None:
        if d_a < -0.010:
            interp_a = "safe ACLED 제거 시 대폭 하락 → ACLED 과거 기록이 여전히 핵심 신호."
        elif d_a < -0.003:
            interp_a = "safe ACLED 제거 시 소폭 하락 — LGB gain(4.3%)보다 실제 기여가 큼."
        else:
            interp_a = "safe ACLED 제거해도 모델이 유지됨 — GDELT 피처가 대부분의 정보를 커버."
        lines.append(f"**Q4. safe ACLED 제거 시 모델이 여전히 괜찮은가?**")
        lines.append(f"→ {interp_a}  ")
        lines.append(f"  (제거 후: {pr_of('no_safe_acled'):.4f}, delta: {delta_str(d_a)})")
        lines.append("")

    # economic
    d_e = delta_of("no_economic")
    if d_e is not None:
        if d_e < -0.005:
            interp_e = "economic 제거 시 유의미한 하락 → LGB gain 23.2%가 실제 기여를 반영."
        else:
            interp_e = "economic 제거 시 소폭 하락 → 중요하지만 필수는 아님."
        lines.append(f"**Q5. economic 제거 시 성능이 크게 떨어지는가?**")
        lines.append(f"→ {interp_e}  ")
        lines.append(f"  (제거 후: {pr_of('no_economic'):.4f}, delta: {delta_str(d_e)})")
        lines.append("")

    lines += [
        "---",
        "",
        "## test set 평가 정책",
        "",
        "> **test set은 아직 평가하지 않았다.**  ",
        "> 현재는 F2_clean ablation 분석 단계이며, 최종 feature/model 구조 확정 후  ",
        "> test PR-AUC를 딱 한 번만 측정한다.  ",
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
    print("=" * 70)
    print("  F2_clean Feature Group Ablation")
    print(f"  F2_full baseline PR-AUC = {F2_FULL_PR_AUC}")
    print(f"  DATA_ROOT : {DATA_ROOT}")
    print("=" * 70)

    for path, label in [(TRAIN_PATH,"train"), (VAL_PATH,"val"),
                        (SAFE_ACLED_PATH,"safe_acled"), (GDELT_TITLE_PATH,"gdelt_title"),
                        (GDELT_TP_PATH,"gdelt_tp")]:
        if not os.path.exists(path):
            print(f"[오류] {label} 없음: {path}")
            sys.exit(1)

    # 실행할 ablation 목록 (환경변수로 일부만 선택 가능)
    all_ablations = [
        "no_country", "no_gdelt_theme_person", "no_gdelt_title",
        "no_safe_acled", "no_economic", "no_gdelt_events",
    ]
    env_filter = os.environ.get("ABLATIONS", "")
    if env_filter:
        run_list = [a.strip() for a in env_filter.split(",") if a.strip() in all_ablations]
        print(f"  ABLATIONS 필터: {run_list}")
    else:
        run_list = all_ablations

    print(f"  실행할 ablation: {run_list}\n")

    # 데이터 로드 (1회)
    splits = load_data()
    train_fit = splits["train_fit"]

    # country encoder (train_fit 기준, 전 실험 공유)
    country_enc = LabelEncoder()
    country_enc.fit(train_fit["country"].astype(str))
    print(f"  country encoder: {len(country_enc.classes_)}개 국가")

    # F2_full feature cols 미리 구성 (검증용)
    f2_full_cols = get_f2_full_cols(train_fit)
    print(f"  F2_full feature 수: {len(f2_full_cols)}개\n")

    # Ablation 실행
    all_results = []
    for ab_name in run_list:
        feature_cols = get_ablation_cols(train_fit, ab_name)
        result = run_ablation(ab_name, feature_cols, splits, country_enc)
        all_results.append(result)

    # 최종 요약
    print()
    print("=" * 70)
    print("  Ablation 요약 (val_eval Stacking Platt PR-AUC)")
    print(f"  {'실험':<30} {'feature':>8} {'PR-AUC':>8} {'delta':>10}")
    print(f"  {'-'*58}")
    print(f"  {'F2_full (baseline)':<30} {94:>8}개 {F2_FULL_PR_AUC:>8.4f} {'—':>10}")
    for r in sorted(all_results, key=lambda x: x["delta"]):
        d = r["delta"]
        print(f"  {'F2_' + r['name']:<30} {r['feature_n']:>8}개 "
              f"{r['platt_pr_auc']:>8.4f} {d:>+10.4f}")
    print("=" * 70)

    save_report(all_results)


if __name__ == "__main__":
    main()
