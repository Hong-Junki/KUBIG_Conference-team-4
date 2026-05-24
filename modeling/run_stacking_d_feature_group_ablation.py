"""
Category D — Feature Group Contribution Ablation
=================================================
Experiment family : stacking_tree_only_12y_feature_group_{group}
Base design       : identical to stacking_tree_only_12y_with_mask_feature
                    (LightGBM + XGBoost, LogReg meta, Platt+Isotonic, 6-fold OOF)

Feature groups tested:
  1. full_best               — all features + SE + mask (replicates current best)
  2. acled_only_plus_se_mask — ACLED cols + SE + mask
  3. gdelt_only_plus_se_mask — GDELT cols + SE + mask
  4. economic_only_plus_se_mask — econ cols + SE + mask
  5. se_mask_only            — SE + mask only
  6. no_acled_features       — all except ACLED cols (keep SE + mask)
  7. no_gdelt_features       — all except GDELT cols (keep SE + mask)
  8. no_economic_features    — all except econ cols (keep SE + mask)

Output:
  Validation stack prediction for every group.
  Test predictions only if group beats current best + 0.003.
  Summary CSV + MD report.

Usage (from project root):
    python modeling/run_stacking_d_feature_group_ablation.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from xgboost.callback import EarlyStopping as XGBEarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve, brier_score_loss

from evaluate import compute_pr_auc, compute_p_at_top_k, compute_recall_at_precision, compute_ece

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Identity ───────────────────────────────────────────────────────────────────
OWNER       = "D_byeonghyeon"
RANDOM_SEED = 42

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_PATH = os.path.join(_ROOT, "input", "processed", "dataset", "train.parquet")
VAL_PATH   = os.path.join(_ROOT, "input", "processed", "dataset", "val.parquet")
TEST_PATH  = os.path.join(_ROOT, "input", "processed", "dataset", "test.parquet")
SE_PATH    = os.path.join(_ROOT, "output", "macis_12y", "se_scores.parquet")

PRED_DIR   = os.path.join(_ROOT, "outputs", "predictions")
REPORT_DIR = os.path.join(_ROOT, "outputs", "reports")

# ── Column constants ───────────────────────────────────────────────────────────
TARGET_COL = "y_escalation"
DATE_COL   = "date"
COUNTRY_COL = "country"

LABEL_META_COLS = [
    "y", "y_onset", "y_escalation",
    "fatalities_next3d", "event_count_next3d",
    "past14d_event_count", "past14d_fatalities_mean",
]
ALWAYS_EXCLUDE = set(LABEL_META_COLS) | {DATE_COL, "iso3"}
# COUNTRY_COL is intentionally kept as a feature (same as current best script)

# ── Feature group definitions ──────────────────────────────────────────────────
# Columns by source (populated after data load)
_ACLED_COLS   = None   # acled_* except acled_missing_mask
_GDELT_COLS   = None   # gdelt_*
_ECON_COLS    = None   # econ_*
_ALL_FEAT_COLS = None  # all usable feature cols (after SE merge)

def _classify_columns(feat_cols: list) -> tuple:
    """Split feature columns into ACLED / GDELT / ECON / other buckets."""
    acled, gdelt, econ, other = [], [], [], []
    for c in feat_cols:
        if c in ("macis_se_score", "acled_missing_mask"):
            continue   # handled separately
        elif c.startswith("acled_"):
            acled.append(c)
        elif c.startswith("gdelt_"):
            gdelt.append(c)
        elif c.startswith("econ_"):
            econ.append(c)
        else:
            other.append(c)
    return acled, gdelt, econ, other


def build_feature_groups(all_feat_cols: list) -> dict:
    """Return {group_name: [col, ...]} using the full feature list."""
    acled, gdelt, econ, other = _classify_columns(all_feat_cols)
    se_mask = ["macis_se_score", "acled_missing_mask"]

    groups = {
        "full_best": list(all_feat_cols),
        "acled_only_plus_se_mask":    acled + se_mask,
        "gdelt_only_plus_se_mask":    gdelt + se_mask,
        "economic_only_plus_se_mask": econ  + se_mask,
        "se_mask_only":               se_mask,
        "no_acled_features":  [c for c in all_feat_cols if c not in set(acled)],
        "no_gdelt_features":  [c for c in all_feat_cols if c not in set(gdelt)],
        "no_economic_features": [c for c in all_feat_cols if c not in set(econ)],
    }
    return groups


# ── OOF folds ─────────────────────────────────────────────────────────────────
OOF_FOLDS = [
    {"name": "F1", "train_end": 2017, "pred_year": 2018},
    {"name": "F2", "train_end": 2018, "pred_year": 2019},
    {"name": "F3", "train_end": 2019, "pred_year": 2020},
    {"name": "F4", "train_end": 2020, "pred_year": 2021},
    {"name": "F5", "train_end": 2021, "pred_year": 2022},
    {"name": "F6", "train_end": 2022, "pred_year": 2023},
]

# ── Hyperparameters (identical to current best) ────────────────────────────────
LGB_PARAMS = {
    "objective": "binary", "metric": "average_precision",
    "boosting_type": "gbdt", "num_leaves": 63, "learning_rate": 0.05,
    "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.8, "scale_pos_weight": 22,
    "seed": RANDOM_SEED, "verbose": -1,
}
LGB_ROUNDS_OOF   = 500
LGB_ROUNDS_FINAL = 1000
EARLY_STOP       = 50

XGB_PARAMS_OOF = {
    "objective": "binary:logistic", "eval_metric": "aucpr",
    "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
    "colsample_bytree": 0.8, "scale_pos_weight": 22,
    "seed": RANDOM_SEED, "verbosity": 0,
}
XGB_PARAMS_FINAL = {**XGB_PARAMS_OOF, "verbosity": 0}
XGB_ROUNDS_OOF   = 500
XGB_ROUNDS_FINAL = 1000

META_C_CANDIDATES = [0.01, 0.1, 1.0, 10.0]

# ── Thresholds ─────────────────────────────────────────────────────────────────
CURRENT_BEST_PR_AUC = 0.2714   # stacking_tree_only_12y_with_mask_feature Platt
IMPROVEMENT_THRESHOLD = 0.003   # must exceed current best by this much to save test preds


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _recall_at_prec(y_true, y_prob, min_prec):
    p, r, _ = precision_recall_curve(np.asarray(y_true), np.asarray(y_prob))
    valid = p >= min_prec
    return float(r[valid].max()) if valid.any() else 0.0


def compute_metrics(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        "pr_auc":                   compute_pr_auc(y_true, y_prob),
        "p_at_top5pct":             compute_p_at_top_k(y_true, y_prob, k=0.05),
        "p_at_top10pct":            compute_p_at_top_k(y_true, y_prob, k=0.10),
        "recall_at_precision_010":  compute_recall_at_precision(y_true, y_prob, 0.10),
        "recall_at_precision_020":  _recall_at_prec(y_true, y_prob, 0.20),
        "recall_at_precision_030":  _recall_at_prec(y_true, y_prob, 0.30),
        "brier_score":              float(brier_score_loss(y_true, y_prob)),
        "ece":                      compute_ece(y_true, y_prob),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def merge_se(df: pd.DataFrame, se: pd.DataFrame) -> pd.DataFrame:
    se2 = se.rename(columns={"se_score": "macis_se_score", "iso3": "_iso3"})
    merged = df.merge(se2[["_iso3", "date", "macis_se_score"]],
                      left_on=[DATE_COL, COUNTRY_COL], right_on=["date", "_iso3"], how="left")
    if "date_x" in merged.columns:
        merged = merged.rename(columns={"date_x": DATE_COL}).drop(columns=["date_y"], errors="ignore")
    merged = merged.drop(columns=["_iso3"], errors="ignore")
    merged["macis_se_score"] = merged["macis_se_score"].fillna(0.0)
    return merged


def get_all_feat_cols(df: pd.DataFrame) -> list:
    return sorted([c for c in df.columns if c not in ALWAYS_EXCLUDE])


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation helpers (country encoding)
# ─────────────────────────────────────────────────────────────────────────────

def prep_lgbm_X(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Return feature DataFrame with country encoded as category for LightGBM."""
    X = df[feat_cols].copy()
    if "country" in feat_cols:
        X["country"] = X["country"].astype("category")
    return X


def prep_xgb_X(df: pd.DataFrame, feat_cols: list, country_enc: LabelEncoder) -> np.ndarray:
    """Return float32 array with country label-encoded for XGBoost."""
    X = df[feat_cols].copy()
    if "country" in feat_cols:
        country_map = {c: i for i, c in enumerate(country_enc.classes_)}
        X["country"] = df["country"].astype(str).map(
            lambda c: country_map.get(c, -1)
        ).astype(np.int32)
    return X.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM helpers
# ─────────────────────────────────────────────────────────────────────────────

def lgbm_oof_fold(train_df, feat_cols, fold):
    tr = train_df[train_df[DATE_COL].dt.year <= fold["train_end"]]
    va = train_df[train_df[DATE_COL].dt.year == fold["pred_year"]]
    cat_feat = ["country"] if "country" in feat_cols else "auto"
    # No valid_set in OOF — fixed rounds, no early stopping (matches current best)
    d_tr = lgb.Dataset(prep_lgbm_X(tr, feat_cols), label=tr[TARGET_COL],
                       categorical_feature=cat_feat, free_raw_data=False)
    model = lgb.train(LGB_PARAMS, d_tr, num_boost_round=LGB_ROUNDS_OOF)
    return model.predict(prep_lgbm_X(va, feat_cols)), va


def lgbm_final(train_df, val_df, feat_cols):
    cat_feat = ["country"] if "country" in feat_cols else "auto"
    d_tr = lgb.Dataset(prep_lgbm_X(train_df, feat_cols), label=train_df[TARGET_COL],
                       categorical_feature=cat_feat, free_raw_data=False)
    d_va = lgb.Dataset(prep_lgbm_X(val_df, feat_cols),   label=val_df[TARGET_COL],
                       reference=d_tr, free_raw_data=False)
    model = lgb.train(
        LGB_PARAMS, d_tr, num_boost_round=LGB_ROUNDS_FINAL,
        valid_sets=[d_va],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost helpers
# ─────────────────────────────────────────────────────────────────────────────

def xgb_oof_fold(train_df, feat_cols, fold, country_enc):
    tr = train_df[train_df[DATE_COL].dt.year <= fold["train_end"]]
    va = train_df[train_df[DATE_COL].dt.year == fold["pred_year"]]
    # No eval set in OOF — fixed rounds, no early stopping (matches current best)
    d_tr = xgb.DMatrix(prep_xgb_X(tr, feat_cols, country_enc), label=tr[TARGET_COL])
    d_va = xgb.DMatrix(prep_xgb_X(va, feat_cols, country_enc))
    model = xgb.train(XGB_PARAMS_OOF, d_tr, num_boost_round=XGB_ROUNDS_OOF)
    return model.predict(d_va), va


def xgb_final(train_df, val_df, feat_cols, country_enc):
    d_tr = xgb.DMatrix(prep_xgb_X(train_df, feat_cols, country_enc), label=train_df[TARGET_COL])
    d_va = xgb.DMatrix(prep_xgb_X(val_df,   feat_cols, country_enc), label=val_df[TARGET_COL])
    cb   = XGBEarlyStopping(rounds=EARLY_STOP, metric_name="aucpr",
                             data_name="val", maximize=True, save_best=True)
    model = xgb.train(XGB_PARAMS_FINAL, d_tr, num_boost_round=XGB_ROUNDS_FINAL,
                      evals=[(d_va, "val")], callbacks=[cb], verbose_eval=False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# One feature-group run
# ─────────────────────────────────────────────────────────────────────────────

def run_group(group_name: str, feat_cols: list,
              train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
              country_enc: LabelEncoder) -> dict:
    print(f"\n{'='*60}")
    print(f"  Group: {group_name}  ({len(feat_cols)} features)")
    print(f"{'='*60}")
    t0 = time.time()

    # ── OOF predictions ────────────────────────────────────────────────────────
    oof_lgbm_rows, oof_xgb_rows = [], []

    for fold in OOF_FOLDS:
        lgbm_probs, va = lgbm_oof_fold(train_df, feat_cols, fold)
        for i, (_, row) in enumerate(va.iterrows()):
            oof_lgbm_rows.append({
                "date": row[DATE_COL], "country": row[COUNTRY_COL],
                "y_true": row[TARGET_COL], "y_prob_oof": lgbm_probs[i]
            })

        xgb_probs, va2 = xgb_oof_fold(train_df, feat_cols, fold, country_enc)
        for i, (_, row) in enumerate(va2.iterrows()):
            oof_xgb_rows.append({
                "date": row[DATE_COL], "country": row[COUNTRY_COL],
                "y_true": row[TARGET_COL], "y_prob_oof": xgb_probs[i]
            })

    oof_lgbm_df = pd.DataFrame(oof_lgbm_rows).sort_values(["date", "country"]).reset_index(drop=True)
    oof_xgb_df  = pd.DataFrame(oof_xgb_rows).sort_values(["date", "country"]).reset_index(drop=True)

    # ── Final base models (with val early stopping) ────────────────────────────
    lgbm_model = lgbm_final(train_df, val_df, feat_cols)
    xgb_model  = xgb_final(train_df, val_df, feat_cols, country_enc)

    val_lgbm  = lgbm_model.predict(prep_lgbm_X(val_df,  feat_cols))
    val_xgb   = xgb_model.predict(xgb.DMatrix(prep_xgb_X(val_df,  feat_cols, country_enc)))
    test_lgbm = lgbm_model.predict(prep_lgbm_X(test_df, feat_cols))
    test_xgb  = xgb_model.predict(xgb.DMatrix(prep_xgb_X(test_df, feat_cols, country_enc)))

    # ── Meta OOF stack ─────────────────────────────────────────────────────────
    oof_merged = oof_lgbm_df[["date", "country", "y_true", "y_prob_oof"]].merge(
        oof_xgb_df[["date", "country", "y_prob_oof"]].rename(columns={"y_prob_oof": "y_prob_oof_xgb"}),
        on=["date", "country"], how="inner"
    )
    X_oof = oof_merged[["y_prob_oof", "y_prob_oof_xgb"]].values
    y_oof = oof_merged["y_true"].values

    X_val  = np.column_stack([val_lgbm,  val_xgb])
    X_test = np.column_stack([test_lgbm, test_xgb])

    # ── C search ──────────────────────────────────────────────────────────────
    best_c, best_pr = None, -1.0
    for c in META_C_CANDIDATES:
        meta = LogisticRegression(C=c, class_weight="balanced",
                                  max_iter=1000, random_state=RANDOM_SEED)
        meta.fit(X_oof, y_oof)
        pr = compute_pr_auc(val_df[TARGET_COL].values,
                            meta.predict_proba(X_val)[:, 1])
        if pr > best_pr:
            best_pr, best_c = pr, c

    # ── Final meta model ───────────────────────────────────────────────────────
    meta_final = LogisticRegression(C=best_c, class_weight="balanced",
                                    max_iter=1000, random_state=RANDOM_SEED)
    meta_final.fit(X_oof, y_oof)

    val_stack_raw  = meta_final.predict_proba(X_val)[:, 1]
    test_stack_raw = meta_final.predict_proba(X_test)[:, 1]

    # ── Calibration ────────────────────────────────────────────────────────────
    y_val = val_df[TARGET_COL].values

    platt = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
    platt.fit(val_stack_raw.reshape(-1, 1), y_val)
    val_platt  = platt.predict_proba(val_stack_raw.reshape(-1, 1))[:, 1]
    test_platt = platt.predict_proba(test_stack_raw.reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_stack_raw, y_val)
    val_iso  = iso.predict(val_stack_raw)
    test_iso = iso.predict(test_stack_raw)

    # ── Metrics ───────────────────────────────────────────────────────────────
    m_lgbm  = compute_metrics(y_val, val_lgbm)
    m_xgb   = compute_metrics(y_val, val_xgb)
    m_raw   = compute_metrics(y_val, val_stack_raw)
    m_platt = compute_metrics(y_val, val_platt)
    m_iso   = compute_metrics(y_val, val_iso)

    elapsed = time.time() - t0
    print(f"  LGBM PR-AUC={m_lgbm['pr_auc']:.4f}  XGB PR-AUC={m_xgb['pr_auc']:.4f}")
    print(f"  Stack Platt PR-AUC={m_platt['pr_auc']:.4f}  P@5%={m_platt['p_at_top5pct']:.4f}  "
          f"ECE={m_platt['ece']:.4f}  best_C={best_c}")
    print(f"  Elapsed: {elapsed:.1f}s")

    # ── Save val stack predictions ─────────────────────────────────────────────
    val_out_path = os.path.join(PRED_DIR,
        f"val_predictions__stacking_feature_group_{group_name}_v2__{OWNER}.csv")
    val_out = val_df[[DATE_COL, COUNTRY_COL, TARGET_COL]].copy()
    val_out = val_out.rename(columns={DATE_COL: "date", COUNTRY_COL: "country",
                                      TARGET_COL: "y_true"})
    val_out["date"] = pd.to_datetime(val_out["date"]).dt.tz_localize(None).dt.normalize()
    val_out["y_prob_lgbm"]           = val_lgbm
    val_out["y_prob_xgb"]            = val_xgb
    val_out["y_prob_stack_raw"]      = val_stack_raw
    val_out["y_prob_stack_platt"]    = val_platt
    val_out["y_prob_stack_isotonic"] = val_iso
    val_out = val_out.sort_values(["date", "country"]).reset_index(drop=True)
    assert val_out.duplicated(["date", "country"]).sum() == 0
    val_out.to_csv(val_out_path, index=False)

    # ── Save test predictions only if exceeds threshold ────────────────────────
    saved_test = False
    if m_platt["pr_auc"] > CURRENT_BEST_PR_AUC + IMPROVEMENT_THRESHOLD:
        test_meta = test_df[[DATE_COL, COUNTRY_COL]].copy()
        test_meta["date"] = pd.to_datetime(test_meta[DATE_COL]).dt.tz_localize(None).dt.normalize()
        test_meta = test_meta.rename(columns={DATE_COL: "date", COUNTRY_COL: "country"})
        for suffix, probs in [("raw", test_stack_raw), ("platt", test_platt), ("isotonic", test_iso)]:
            out = test_meta[["date", "country"]].copy()
            out["y_prob"] = probs
            out = out.sort_values(["date", "country"]).reset_index(drop=True)
            path = os.path.join(PRED_DIR,
                f"predictions__stacking_feature_group_{group_name}_{suffix}_v2__{OWNER}.csv")
            out.to_csv(path, index=False)
        saved_test = True
        print(f"  *** NEW BEST — test predictions saved for {group_name} ***")

    return {
        "group":            group_name,
        "n_features":       len(feat_cols),
        "lgbm_pr_auc":      m_lgbm["pr_auc"],
        "xgb_pr_auc":       m_xgb["pr_auc"],
        "stack_raw_pr_auc": m_raw["pr_auc"],
        "stack_platt_pr_auc": m_platt["pr_auc"],
        "stack_iso_pr_auc": m_iso["pr_auc"],
        "p_at_top5pct":     m_platt["p_at_top5pct"],
        "p_at_top10pct":    m_platt["p_at_top10pct"],
        "recall_at_prec010":m_platt["recall_at_precision_010"],
        "recall_at_prec020":m_platt["recall_at_precision_020"],
        "recall_at_prec030":m_platt["recall_at_precision_030"],
        "brier_score":      m_platt["brier_score"],
        "ece":              m_platt["ece"],
        "best_meta_c":      best_c,
        "saved_test":       saved_test,
        "elapsed_s":        round(elapsed, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Safety check
# ─────────────────────────────────────────────────────────────────────────────

def check_existing_outputs(groups: dict):
    must_not_exist = [
        os.path.join(REPORT_DIR, "stacking_tree_only_12y_feature_group_ablation_v2.csv"),
        os.path.join(REPORT_DIR, "stacking_tree_only_12y_feature_group_ablation_v2.md"),
    ]
    for g in groups:
        must_not_exist.append(os.path.join(PRED_DIR,
            f"val_predictions__stacking_feature_group_{g}_v2__{OWNER}.csv"))

    conflicts = [p for p in must_not_exist if os.path.exists(p)]
    if conflicts:
        print("\n[STOP] Output file(s) already exist:")
        for p in conflicts:
            print(f"  {os.path.relpath(p, _ROOT)}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Report writing
# ─────────────────────────────────────────────────────────────────────────────

def write_report(results: list, groups_meta: dict):
    df = pd.DataFrame(results)
    df["delta_vs_best"] = df["stack_platt_pr_auc"] - CURRENT_BEST_PR_AUC

    csv_path = os.path.join(REPORT_DIR, "stacking_tree_only_12y_feature_group_ablation_v2.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Report CSV: {os.path.relpath(csv_path, _ROOT)}")

    # Markdown report
    lines = [
        "# Feature Group Contribution Ablation",
        "",
        f"> Base experiment: `stacking_tree_only_12y_with_mask_feature`",
        f"> Current best Stacking Platt PR-AUC: **{CURRENT_BEST_PR_AUC}**",
        f"> Owner: `{OWNER}`",
        "",
        "## Feature Group Summary",
        "",
        "| Group | n_feat | LGBM PR-AUC | XGB PR-AUC | Stack Platt PR-AUC | P@5% | ECE | Best C | delta vs best |",
        "|-------|--------|-------------|------------|-------------------|------|-----|--------|---------------|",
    ]
    for _, row in df.iterrows():
        delta = row["delta_vs_best"]
        sign  = "+" if delta >= 0 else ""
        star  = " ★" if delta > 0 else ""
        lines.append(
            f"| {row['group']}{star} | {int(row['n_features'])} "
            f"| {row['lgbm_pr_auc']:.4f} | {row['xgb_pr_auc']:.4f} "
            f"| {row['stack_platt_pr_auc']:.4f} | {row['p_at_top5pct']:.4f} "
            f"| {row['ece']:.4f} | {row['best_meta_c']} "
            f"| {sign}{delta:.4f} |"
        )

    lines += [
        "",
        "## Detailed Platt Metrics per Group",
        "",
        "| Group | Platt PR-AUC | Platt P@5% | Platt P@10% | R@P≥0.10 | R@P≥0.20 | R@P≥0.30 | Brier | ECE |",
        "|-------|-------------|-----------|------------|---------|---------|---------|-------|-----|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['group']} | {row['stack_platt_pr_auc']:.4f} "
            f"| {row['p_at_top5pct']:.4f} | {row['p_at_top10pct']:.4f} "
            f"| {row['recall_at_prec010']:.4f} | {row['recall_at_prec020']:.4f} "
            f"| {row['recall_at_prec030']:.4f} | {row['brier_score']:.4f} "
            f"| {row['ece']:.4f} |"
        )

    # Group-level feature counts
    lines += [
        "",
        "## Feature Group Composition",
        "",
        "| Group | Features included |",
        "|-------|------------------|",
    ]
    for g, cols in groups_meta.items():
        lines.append(f"| {g} | {len(cols)} cols: {', '.join(cols[:5])}{'...' if len(cols)>5 else ''} |")

    lines.append("")
    md_path = os.path.join(REPORT_DIR, "stacking_tree_only_12y_feature_group_ablation_v2.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report MD:  {os.path.relpath(md_path, _ROOT)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("[1] Loading data...")
    train_raw = pd.read_parquet(TRAIN_PATH)
    val_raw   = pd.read_parquet(VAL_PATH)
    test_raw  = pd.read_parquet(TEST_PATH)
    se        = pd.read_parquet(SE_PATH)

    # ── Merge SE ──────────────────────────────────────────────────────────────
    print("[2] Merging SE scores...")
    train_df = merge_se(train_raw, se)
    val_df   = merge_se(val_raw,   se)
    test_df  = merge_se(test_raw,  se)

    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL], utc=True)
    val_df[DATE_COL]   = pd.to_datetime(val_df[DATE_COL],   utc=True)
    test_df[DATE_COL]  = pd.to_datetime(test_df[DATE_COL],  utc=True)

    all_feat_cols = get_all_feat_cols(train_df)
    print(f"  Total feature cols after SE merge: {len(all_feat_cols)}")

    # Fit country encoder for XGBoost (country as label-encoded integer)
    country_enc = LabelEncoder()
    country_enc.fit(train_df[COUNTRY_COL].astype(str))
    print(f"  Country encoder: {len(country_enc.classes_)} countries")

    # ── Build feature groups ───────────────────────────────────────────────────
    groups_dict = build_feature_groups(all_feat_cols)
    print(f"\n[3] Feature groups ({len(groups_dict)}):")
    for g, cols in groups_dict.items():
        acled_n = sum(1 for c in cols if c.startswith("acled_") and c != "acled_missing_mask")
        gdelt_n = sum(1 for c in cols if c.startswith("gdelt_"))
        econ_n  = sum(1 for c in cols if c.startswith("econ_"))
        se_n    = 1 if "macis_se_score" in cols else 0
        mask_n  = 1 if "acled_missing_mask" in cols else 0
        print(f"  {g:<35}: {len(cols):3d} cols "
              f"(ACLED={acled_n} GDELT={gdelt_n} ECON={econ_n} SE={se_n} mask={mask_n})")

    # ── Safety check ──────────────────────────────────────────────────────────
    check_existing_outputs(groups_dict)

    # ── Run each group ─────────────────────────────────────────────────────────
    print(f"\n[4] Running {len(groups_dict)} feature group experiments...")
    results = []
    for group_name, feat_cols in groups_dict.items():
        # Validate feat_cols against actual df columns
        missing = [c for c in feat_cols if c not in train_df.columns]
        if missing:
            print(f"  WARNING: {group_name} has missing columns: {missing[:5]}")
            feat_cols = [c for c in feat_cols if c in train_df.columns]
        row = run_group(group_name, feat_cols, train_df, val_df, test_df, country_enc)
        results.append(row)

    # ── Write report ──────────────────────────────────────────────────────────
    print("\n[5] Writing reports...")
    df = write_report(results, groups_dict)

    # ── Print summary ─────────────────────────────────────────────────────────
    total = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  Feature Group Ablation Summary")
    print(f"{'='*65}")
    print(f"  {'Group':<35} {'Platt PR-AUC':>12} {'P@5%':>8} {'delta':>8}")
    print(f"  {'-'*65}")
    df_sorted = df.sort_values("stack_platt_pr_auc", ascending=False)
    for _, row in df_sorted.iterrows():
        delta = row["delta_vs_best"]
        sign  = "+" if delta >= 0 else ""
        star  = " ★" if delta > IMPROVEMENT_THRESHOLD else ""
        print(f"  {row['group']:<35} {row['stack_platt_pr_auc']:>12.4f} "
              f"{row['p_at_top5pct']:>8.4f} {sign}{delta:>7.4f}{star}")
    print(f"  {'='*65}")
    print(f"  Current best (tree-only wmf): {CURRENT_BEST_PR_AUC:.4f}")
    print(f"  Total elapsed: {total/60:.1f} min")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
