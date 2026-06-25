"""
O0_clean: ACLED-free clean validation baseline.

Feature set:
  - GDELT events 19
  - Economic 15
  - Country 1

Excluded:
  - all acled_* / safe_acled_* / macis_se_score
  - GDELT title/theme/person
  - embedding/cosine/vector
  - label/future/next/past14d columns

Split:
  train_fit: 2014-01-01 ~ 2022-12-31
  tune_cal : 2023-01-01 ~ 2023-12-31
  val_eval : 2024-01-01 ~ 2024-06-30
  test     : not evaluated
"""

import os
import sys
from datetime import date as dt_date

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from xgboost.callback import EarlyStopping as XGBEarlyStopping

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SUBTREE_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(_SUBTREE_ROOT))),
        "conflict-early-warning",
    ),
)

sys.path.insert(0, _SCRIPT_DIR)
from evaluate import compute_ece, compute_p_at_top_k, compute_pr_auc, compute_recall_at_precision

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
EXPERIMENT = "o0_acled_free_clean_baseline" + ("_smoke" if SMOKE_TEST else "")
RANDOM_SEED = 42

TRAIN_PATH = os.path.join(DATA_ROOT, "input", "processed", "dataset", "train.parquet")
VAL_PATH = os.path.join(DATA_ROOT, "input", "processed", "dataset", "val.parquet")
TEST_PATH = os.path.join(DATA_ROOT, "input", "processed", "dataset", "test.parquet")

REPORT_DIR = os.path.join(_SUBTREE_ROOT, "outputs", "reports")
REPORT_MD = os.path.join(REPORT_DIR, "o0_acled_free_clean_baseline_results.md")

TARGET_COL = "y_escalation"
DATE_COL = "date"
TRAIN_FIT_END = pd.Timestamp("2022-12-31", tz="UTC")
TUNE_CAL_START = pd.Timestamp("2023-01-01", tz="UTC")
TUNE_CAL_END = pd.Timestamp("2023-12-31", tz="UTC")

BASELINE_B_OLD_PR_AUC = 0.0564
O1_THRESHOLD = BASELINE_B_OLD_PR_AUC + 0.003

FEATURE_COLS = [
    "gdelt_goldstein_mean_7d", "gdelt_goldstein_mean_14d", "gdelt_goldstein_mean_30d",
    "gdelt_goldstein_std_7d", "gdelt_goldstein_std_14d", "gdelt_goldstein_std_30d",
    "gdelt_tone_mean_7d", "gdelt_tone_mean_14d", "gdelt_tone_mean_30d",
    "gdelt_mentions_sum_7d", "gdelt_mentions_sum_14d", "gdelt_mentions_sum_30d",
    "gdelt_event_count_7d", "gdelt_event_count_14d", "gdelt_event_count_30d",
    "gdelt_quadclass_1_ratio", "gdelt_quadclass_2_ratio",
    "gdelt_quadclass_3_ratio", "gdelt_quadclass_4_ratio",
    "econ_vix", "econ_vix_pct_1d", "econ_vix_pct_7d",
    "econ_wti", "econ_wti_pct_1d", "econ_wti_pct_7d",
    "econ_gold", "econ_gold_pct_1d", "econ_gold_pct_7d",
    "econ_dxy", "econ_dxy_pct_1d", "econ_dxy_pct_7d",
    "econ_stlfsi4", "econ_stlfsi4_pct_1d", "econ_stlfsi4_pct_7d",
    "country",
]

FORBIDDEN_PATTERNS = (
    "acled_", "safe_acled_", "gdelt_title_", "gdelt_theme_", "gdelt_person_",
    "embedding", "embed", "cosine", "vector",
)
FORBIDDEN_EXACT = {
    "y", "y_onset", "y_escalation", "macis_se_score",
    "fatalities_next3d", "event_count_next3d",
    "past14d_event_count", "past14d_fatalities_mean",
}
FORBIDDEN_SUBSTR = ("future", "next", "past14d")

_ALL_FOLDS = [
    {"name": "OOF_F1", "train_end": 2017, "pred_year": 2018},
    {"name": "OOF_F2", "train_end": 2018, "pred_year": 2019},
    {"name": "OOF_F3", "train_end": 2019, "pred_year": 2020},
    {"name": "OOF_F4", "train_end": 2020, "pred_year": 2021},
    {"name": "OOF_F5", "train_end": 2021, "pred_year": 2022},
]
OOF_FOLDS = _ALL_FOLDS[-1:] if SMOKE_TEST else _ALL_FOLDS

LGB_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 22,
    "seed": RANDOM_SEED,
    "verbose": -1,
}
LGB_ROUNDS_OOF = 50 if SMOKE_TEST else 500
LGB_ROUNDS_FINAL = 100 if SMOKE_TEST else 1000
EARLY_STOP = 20 if SMOKE_TEST else 50
LOG_PERIOD = 50 if SMOKE_TEST else 100

XGB_PARAMS_OOF = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 22,
    "seed": RANDOM_SEED,
    "verbosity": 0,
}
XGB_PARAMS_FINAL = {**XGB_PARAMS_OOF, "verbosity": 0}
XGB_ROUNDS_OOF = 50 if SMOKE_TEST else 500
XGB_ROUNDS_FINAL = 100 if SMOKE_TEST else 1000
META_C_CANDIDATES = [0.01, 0.1, 1.0, 10.0]


def _recall_at_precision(y_true, y_prob, min_precision):
    precision, recall, _ = precision_recall_curve(np.asarray(y_true), np.asarray(y_prob))
    valid = precision >= min_precision
    return float(recall[valid].max()) if valid.any() else 0.0


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    pos_rate = float(y_true.mean())
    metrics = {
        "pr_auc": compute_pr_auc(y_true, y_prob),
        "p_at_top1pct": compute_p_at_top_k(y_true, y_prob, k=0.01),
        "p_at_top5pct": compute_p_at_top_k(y_true, y_prob, k=0.05),
        "p_at_top10pct": compute_p_at_top_k(y_true, y_prob, k=0.10),
        "recall_at_precision_010": compute_recall_at_precision(y_true, y_prob, min_precision=0.10),
        "brier_score": float(np.mean((y_prob - y_true.astype(float)) ** 2)),
        "ece": compute_ece(y_true, y_prob),
        "positive_rate": pos_rate,
        "n_samples": int(len(y_true)),
    }
    for k in (1, 5, 10):
        metrics[f"lift_at_top{k}pct"] = (
            metrics[f"p_at_top{k}pct"] / pos_rate if pos_rate > 0 else float("nan")
        )
    return metrics


def validate_setup():
    print("=" * 70)
    print(f"  Experiment: {EXPERIMENT}")
    if SMOKE_TEST:
        print("  SMOKE_TEST=1: OOF 1 fold, reduced boosting rounds")
    print(f"  DATA_ROOT : {DATA_ROOT}")
    print(f"  REPORT_MD : {REPORT_MD}")
    print("  Split     : train_fit<=2022 | tune_cal=2023 | val_eval=2024-H1")
    print("=" * 70)
    if not os.path.isdir(DATA_ROOT):
        raise SystemExit(f"[오류] DATA_ROOT 없음: {DATA_ROOT}")
    for path in (TRAIN_PATH, VAL_PATH, TEST_PATH):
        if not os.path.exists(path):
            raise SystemExit(f"[오류] 입력 parquet 없음: {path}")
    os.makedirs(REPORT_DIR, exist_ok=True)


def _normalize_date(df):
    if df[DATE_COL].dt.tz is None:
        df[DATE_COL] = df[DATE_COL].dt.tz_localize("UTC")
    return df


def load_splits():
    print("\n[Step 1] Load local dataset only")
    train_full = _normalize_date(pd.read_parquet(TRAIN_PATH))
    val_eval = _normalize_date(pd.read_parquet(VAL_PATH))
    _ = TEST_PATH  # test path is checked but not loaded/evaluated.

    train_fit = train_full[train_full[DATE_COL] <= TRAIN_FIT_END].copy()
    tune_cal = train_full[
        (train_full[DATE_COL] >= TUNE_CAL_START) & (train_full[DATE_COL] <= TUNE_CAL_END)
    ].copy()
    splits = {"train_fit": train_fit, "tune_cal": tune_cal, "val_eval": val_eval}
    for name, df in splits.items():
        print(
            f"  {name:<9}: {len(df):,} rows | "
            f"{df[DATE_COL].min().date()} ~ {df[DATE_COL].max().date()} | "
            f"pos_rate={df[TARGET_COL].mean():.4f} | countries={df['country'].nunique()}"
        )
    print("  test     : not loaded for evaluation")
    return splits


def validate_feature_cols(df):
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"[오류] O0 feature missing: {missing}")
    leaked = []
    for c in FEATURE_COLS:
        cl = c.lower()
        if c in FORBIDDEN_EXACT:
            leaked.append(c)
        if any(cl.startswith(p) for p in FORBIDDEN_PATTERNS):
            leaked.append(c)
        if any(s in cl for s in FORBIDDEN_SUBSTR):
            leaked.append(c)
    if leaked:
        raise SystemExit(f"[오류] forbidden O0 feature present: {sorted(set(leaked))}")
    if len(FEATURE_COLS) != 35:
        raise SystemExit(f"[오류] O0 feature count must be 35, got {len(FEATURE_COLS)}")
    print("\n[Step 2] Feature validation")
    print("  O0_clean features: 35 = GDELT events 19 + economic 15 + country 1")
    print("  Forbidden ACLED/safe_ACLED/title/theme/embedding/label/future columns absent")


def fit_country_encoder(train_fit):
    enc = LabelEncoder()
    enc.fit(train_fit["country"].astype(str))
    return enc


def make_lgbm_X(df):
    X = df[FEATURE_COLS].copy()
    X["country"] = X["country"].astype("category")
    return X


def make_xgb_X(df, enc):
    X = df[FEATURE_COLS].copy()
    cmap = {c: i for i, c in enumerate(enc.classes_)}
    X["country"] = df["country"].astype(str).map(lambda c: cmap.get(c, -1)).astype(np.int32)
    return X.astype(np.float32)


def run_oof(train_fit, enc):
    print(f"\n[Step 3] OOF within train_fit ({len(OOF_FOLDS)} fold)")
    oof_l, oof_x, summaries = [], [], []
    for fold in OOF_FOLDS:
        tr = train_fit[train_fit[DATE_COL].dt.year <= fold["train_end"]]
        pr = train_fit[train_fit[DATE_COL].dt.year == fold["pred_year"]]
        y_tr = tr[TARGET_COL].values
        y_pr = pr[TARGET_COL].values
        print(
            f"  {fold['name']}: train<={fold['train_end']} {len(tr):,} "
            f"-> pred {fold['pred_year']} {len(pr):,}, pos={y_pr.mean():.4f}"
        )
        lgb_ds = lgb.Dataset(
            make_lgbm_X(tr), label=y_tr, categorical_feature=["country"], free_raw_data=False
        )
        lgbm = lgb.train(LGB_PARAMS, lgb_ds, num_boost_round=LGB_ROUNDS_OOF)
        pl = lgbm.predict(make_lgbm_X(pr))
        dl = compute_pr_auc(y_pr, pl)

        xgbm = xgb.train(
            XGB_PARAMS_OOF,
            xgb.DMatrix(make_xgb_X(tr, enc), label=y_tr),
            num_boost_round=XGB_ROUNDS_OOF,
        )
        px = xgbm.predict(xgb.DMatrix(make_xgb_X(pr, enc)))
        dx = compute_pr_auc(y_pr, px)
        print(f"    LGB={dl:.4f} | XGB={dx:.4f}")

        for d, c, yt, p1, p2 in zip(
            pr[DATE_COL].dt.strftime("%Y-%m-%d").values,
            pr["country"].values,
            y_pr,
            pl,
            px,
        ):
            oof_l.append({"date": d, "country": c, "y_true": int(yt), "p": float(p1)})
            oof_x.append({"date": d, "country": c, "p_x": float(p2)})
        summaries.append({
            "name": fold["name"],
            "pred_year": fold["pred_year"],
            "n_train": len(tr),
            "n_pred": len(pr),
            "positive_rate": float(y_pr.mean()),
            "lgbm_pr_auc": float(dl),
            "xgb_pr_auc": float(dx),
        })
    return pd.DataFrame(oof_l), pd.DataFrame(oof_x), summaries


def train_finals(train_fit, tune_cal, enc):
    print("\n[Step 4] Final base models: train_fit, early stop on tune_cal")
    ds_tr = lgb.Dataset(
        make_lgbm_X(train_fit),
        label=train_fit[TARGET_COL].values,
        categorical_feature=["country"],
        free_raw_data=False,
    )
    ds_tc = lgb.Dataset(
        make_lgbm_X(tune_cal),
        label=tune_cal[TARGET_COL].values,
        reference=ds_tr,
        free_raw_data=False,
    )
    lgbm = lgb.train(
        LGB_PARAMS,
        ds_tr,
        num_boost_round=LGB_ROUNDS_FINAL,
        valid_sets=[ds_tc],
        valid_names=["tune_cal"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
            lgb.log_evaluation(period=LOG_PERIOD),
        ],
    )
    xgbm = xgb.train(
        XGB_PARAMS_FINAL,
        xgb.DMatrix(make_xgb_X(train_fit, enc), label=train_fit[TARGET_COL].values),
        num_boost_round=XGB_ROUNDS_FINAL,
        evals=[(xgb.DMatrix(make_xgb_X(tune_cal, enc), label=tune_cal[TARGET_COL].values), "tune_cal")],
        callbacks=[XGBEarlyStopping(rounds=EARLY_STOP, save_best=True)],
        verbose_eval=LOG_PERIOD,
    )
    return lgbm, xgbm


def train_meta_and_calibrators(oof_l, oof_x, tune_cal, lgbm, xgbm, enc):
    print("\n[Step 5] Meta LogReg and Platt calibration on tune_cal")
    oof = oof_l.merge(oof_x, on=["date", "country"], how="inner")
    X_oof = oof[["p", "p_x"]].values
    y_oof = oof["y_true"].values

    tune_y = tune_cal[TARGET_COL].values
    tune_l = lgbm.predict(make_lgbm_X(tune_cal))
    tune_x = xgbm.predict(xgb.DMatrix(make_xgb_X(tune_cal, enc)))
    X_tune = np.column_stack([tune_l, tune_x])

    best_c, best_auc = None, -1.0
    for c in META_C_CANDIDATES:
        meta_c = LogisticRegression(
            class_weight="balanced", C=c, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED
        )
        meta_c.fit(X_oof, y_oof)
        auc = compute_pr_auc(tune_y, meta_c.predict_proba(X_tune)[:, 1])
        print(f"  C={c:.2f} -> tune_cal PR-AUC={auc:.4f}")
        if auc > best_auc:
            best_c, best_auc = c, auc

    meta = LogisticRegression(
        class_weight="balanced", C=best_c, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED
    )
    meta.fit(X_oof, y_oof)
    tune_raw = meta.predict_proba(X_tune)[:, 1]

    platt = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    platt.fit(tune_raw.reshape(-1, 1), tune_y)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(tune_raw, tune_y)
    print(f"  Selected C={best_c}, tune_cal meta PR-AUC={best_auc:.4f}")
    return meta, platt, iso, best_c, {"lgbm": tune_l, "xgb": tune_x}


def evaluate_val(val_eval, lgbm, xgbm, meta, platt, iso, enc):
    print("\n[Step 6] val_eval metrics only")
    y = val_eval[TARGET_COL].values
    prob_l = lgbm.predict(make_lgbm_X(val_eval))
    prob_x = xgbm.predict(xgb.DMatrix(make_xgb_X(val_eval, enc)))
    raw = meta.predict_proba(np.column_stack([prob_l, prob_x]))[:, 1]
    platt_prob = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
    iso_prob = iso.predict(raw)
    metrics = {
        "lgbm": compute_metrics(y, prob_l),
        "xgb": compute_metrics(y, prob_x),
        "stacking_raw": compute_metrics(y, raw),
        "stacking_platt": compute_metrics(y, platt_prob),
        "stacking_isotonic": compute_metrics(y, iso_prob),
    }
    for key, label in [
        ("lgbm", "LightGBM"),
        ("xgb", "XGBoost"),
        ("stacking_platt", "Stacking Platt"),
    ]:
        m = metrics[key]
        print(
            f"  {label:<15} PR-AUC={m['pr_auc']:.4f} "
            f"P@5%={m['p_at_top5pct']:.4f} Lift@5%={m['lift_at_top5pct']:.2f}x"
        )
    return metrics


def save_report(splits, metrics, best_c, folds):
    today = dt_date.today().isoformat()

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and v == v else "-"

    platt_pr = metrics["stacking_platt"]["pr_auc"]
    decision = "O1_clean으로 확장 필요" if platt_pr < O1_THRESHOLD else "O1_clean 확장 후보이나 추가 feature ablation 필요"
    rows = []
    for key, label in [
        ("lgbm", "LightGBM"),
        ("xgb", "XGBoost"),
        ("stacking_raw", "Stacking raw"),
        ("stacking_platt", "Stacking Platt"),
        ("stacking_isotonic", "Stacking Isotonic"),
    ]:
        m = metrics[key]
        rows.append(
            f"| {label} | {fmt(m['pr_auc'])} | {fmt(m['p_at_top1pct'])} | "
            f"{fmt(m['p_at_top5pct'])} | {fmt(m['p_at_top10pct'])} | "
            f"{fmt(m['lift_at_top1pct'])} | {fmt(m['lift_at_top5pct'])} | "
            f"{fmt(m['lift_at_top10pct'])} | {fmt(m['recall_at_precision_010'])} | "
            f"{fmt(m['brier_score'])} | {fmt(m['ece'])} |"
        )

    split_lines = []
    for name in ["train_fit", "tune_cal", "val_eval"]:
        df = splits[name]
        split_lines.append(
            f"| {name} | {len(df):,} | {df[DATE_COL].min().date()} ~ "
            f"{df[DATE_COL].max().date()} | {df[TARGET_COL].mean():.4f} |"
        )

    fold_lines = [
        f"| {f['name']} | {f['pred_year']} | {f['n_train']:,} | {f['n_pred']:,} | "
        f"{f['positive_rate']:.4f} | {f['lgbm_pr_auc']:.4f} | {f['xgb_pr_auc']:.4f} |"
        for f in folds
    ]

    lines = [
        f"# O0_clean ACLED-free baseline results{' (SMOKE TEST)' if SMOKE_TEST else ''}",
        "",
        f"실행일: {today}",
        f"실험명: `{EXPERIMENT}`",
        "",
        "## Split",
        "",
        "| split | rows | period | positive rate |",
        "|---|---:|---|---:|",
        *split_lines,
        "| test | not evaluated | 2024-07-01 onward | - |",
        "",
        "## Feature Set",
        "",
        "- O0_clean feature 수: 35",
        "- 사용 feature group: GDELT events 19, economic 15, country 1",
        "- ACLED/safe_ACLED/macis_se_score/GDELT title/theme/person/embedding/cosine/vector 미사용",
        "- label/future/next/past14d 컬럼 미사용",
        "",
        "## val_eval Metrics",
        "",
        "| model | PR-AUC | P@top1% | P@top5% | P@top10% | Lift@top1% | Lift@top5% | Lift@top10% | R@P>=0.10 | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## OOF Folds",
        "",
        "| fold | pred_year | n_train | n_pred | pos_rate | LGB PR-AUC | XGB PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *fold_lines,
        "",
        "## Baseline Comparison",
        "",
        f"- old B PR-AUC: {BASELINE_B_OLD_PR_AUC:.4f}",
        f"- O0_clean Stacking Platt PR-AUC: {platt_pr:.4f}",
        f"- delta: {platt_pr - BASELINE_B_OLD_PR_AUC:+.4f}",
        f"- Meta LogReg C: {best_c}",
        f"- 판단: {decision}",
        "",
        "## Test Policy",
        "",
        "test set은 로드/평가하지 않았다. 예측 CSV와 모델 파일도 저장하지 않았다.",
    ]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved: {REPORT_MD}")


def main():
    validate_setup()
    splits = load_splits()
    validate_feature_cols(splits["train_fit"])
    enc = fit_country_encoder(splits["train_fit"])
    print(f"  country encoder: {len(enc.classes_)} countries")
    oof_l, oof_x, folds = run_oof(splits["train_fit"], enc)
    lgbm, xgbm = train_finals(splits["train_fit"], splits["tune_cal"], enc)
    meta, platt, iso, best_c, _ = train_meta_and_calibrators(
        oof_l, oof_x, splits["tune_cal"], lgbm, xgbm, enc
    )
    metrics = evaluate_val(splits["val_eval"], lgbm, xgbm, meta, platt, iso, enc)
    save_report(splits, metrics, best_c, folds)


if __name__ == "__main__":
    main()
