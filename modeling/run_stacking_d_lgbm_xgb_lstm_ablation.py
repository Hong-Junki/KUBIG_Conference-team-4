"""
Category D — Stacking Ablation: LightGBM + XGBoost + LSTM
==========================================================
Experiment : stacking_lgbm_xgb_lstm_12y_with_mask_feature
Level 0    : LightGBM (12y+SE+mask_feature), XGBoost (12y+SE+mask_feature), LSTM (30d)
Level 1    : Logistic Regression (meta-learner, OOF-trained)
Calibration: Platt Scaling + Isotonic Regression

All Level 0 predictions are loaded from existing files — no retraining.
Meta-learner C is selected by validation PR-AUC (test set never used for selection).

Usage (from project root):
    python modeling/run_stacking_d_lgbm_xgb_lstm_ablation.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, brier_score_loss

from evaluate import compute_pr_auc, compute_p_at_top_k, compute_recall_at_precision, compute_ece

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Experiment identity ───────────────────────────────────────────────────────
EXPERIMENT = "stacking_lgbm_xgb_lstm_12y_with_mask_feature"
OWNER      = "D_byeonghyeon"

# ── Existing Level 0 prediction files ────────────────────────────────────────
LGBM_OOF  = os.path.join(_ROOT, "outputs", "predictions",
                          "oof_predictions__lgbm_se_12y_with_mask_feature__D_byeonghyeon.csv")
LGBM_VAL  = os.path.join(_ROOT, "outputs", "predictions",
                          "val_predictions__lgbm_se_12y_with_mask_feature__D_byeonghyeon.csv")
LGBM_TEST = os.path.join(_ROOT, "outputs", "predictions",
                          "predictions__lgbm_se_12y_with_mask_feature__D_byeonghyeon.csv")

XGB_OOF   = os.path.join(_ROOT, "outputs", "predictions",
                          "oof_predictions__xgb_se_12y_with_mask_feature__D_byeonghyeon.csv")
XGB_VAL   = os.path.join(_ROOT, "outputs", "predictions",
                          "val_predictions__xgb_se_12y_with_mask_feature__D_byeonghyeon.csv")
XGB_TEST  = os.path.join(_ROOT, "outputs", "predictions",
                          "predictions__xgb_se_12y_with_mask_feature__D_byeonghyeon.csv")

LSTM_OOF  = os.path.join(_ROOT, "outputs", "predictions",
                          "oof_predictions__lstm_classifier_30d__C_or_D_byeonghyeon.csv")
LSTM_VAL  = os.path.join(_ROOT, "outputs", "predictions",
                          "val_predictions__lstm_classifier_30d__C_or_D_byeonghyeon.csv")
LSTM_TEST = os.path.join(_ROOT, "outputs", "predictions",
                          "predictions__lstm_classifier_30d__C_or_D_byeonghyeon.csv")

# ── Output directories ────────────────────────────────────────────────────────
PRED_DIR   = os.path.join(_ROOT, "outputs", "predictions")
MODEL_DIR  = os.path.join(_ROOT, "outputs", "models")
REPORT_DIR = os.path.join(_ROOT, "outputs", "reports")

# ── Output file registry ──────────────────────────────────────────────────────
OUTPUT_FILES = {
    "val_stack":      os.path.join(PRED_DIR,   f"val_predictions__{EXPERIMENT}__{OWNER}.csv"),
    "test_raw":       os.path.join(PRED_DIR,   f"predictions__{EXPERIMENT}_raw__{OWNER}.csv"),
    "test_platt":     os.path.join(PRED_DIR,   f"predictions__{EXPERIMENT}_platt__{OWNER}.csv"),
    "test_isotonic":  os.path.join(PRED_DIR,   f"predictions__{EXPERIMENT}_isotonic__{OWNER}.csv"),
    "model_meta":     os.path.join(MODEL_DIR,  f"meta_logreg_{EXPERIMENT}.pkl"),
    "model_platt":    os.path.join(MODEL_DIR,  f"platt_calibrator_{EXPERIMENT}.pkl"),
    "model_isotonic": os.path.join(MODEL_DIR,  f"isotonic_calibrator_{EXPERIMENT}.pkl"),
    "metrics_json":   os.path.join(REPORT_DIR, f"{EXPERIMENT}_val_metrics.json"),
    "metrics_md":     os.path.join(REPORT_DIR, f"{EXPERIMENT}_val_metrics.md"),
    "calib_csv":      os.path.join(REPORT_DIR, f"{EXPERIMENT}_calibration.csv"),
    "compare_csv":    os.path.join(REPORT_DIR, "stacking_lgbm_xgb_lstm_vs_tree_only_comparison.csv"),
    "compare_md":     os.path.join(REPORT_DIR, "stacking_lgbm_xgb_lstm_vs_tree_only_comparison.md"),
}

# ── Baseline reference values (tree-only best) ────────────────────────────────
TREE_ONLY_REF = {
    "experiment":  "stacking_tree_only_12y_with_mask_feature",
    "lgbm_pr_auc": 0.2606,
    "xgb_pr_auc":  0.2631,
    "lstm_pr_auc": 0.1030,
    "platt_pr_auc": 0.2714,
    "platt_p5pct":  0.2689,
    "platt_ece":    0.0083,
}

META_C_CANDIDATES = [0.01, 0.1, 1.0, 10.0]
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Safety helpers
# ─────────────────────────────────────────────────────────────────────────────

def check_existing_outputs():
    conflicts = [p for p in OUTPUT_FILES.values() if os.path.exists(p)]
    if not conflicts:
        return
    print("\n[STOP] Output file(s) already exist — refusing to overwrite:")
    for p in conflicts:
        print(f"  {os.path.relpath(p, _ROOT)}")
    sys.exit(1)


def ensure_dirs():
    for d in [PRED_DIR, MODEL_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _recall_at_prec(y_true, y_prob, min_prec):
    p, r, _ = precision_recall_curve(np.asarray(y_true), np.asarray(y_prob))
    valid = p >= min_prec
    return float(r[valid].max()) if valid.any() else 0.0


def compute_metrics(y_true, y_prob):
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
# Data loading and merging
# ─────────────────────────────────────────────────────────────────────────────

def load_oof() -> pd.DataFrame:
    """Merge LGBM, XGBoost, and LSTM OOF predictions by date+country+y_true."""
    lgbm = pd.read_csv(LGBM_OOF)[["date", "country", "y_true", "y_prob_oof"]].rename(
        columns={"y_prob_oof": "lgbm"})
    xgb  = pd.read_csv(XGB_OOF)[["date", "country", "y_prob_oof"]].rename(
        columns={"y_prob_oof": "xgb"})
    lstm = pd.read_csv(LSTM_OOF)[["date", "country", "y_prob_oof"]].rename(
        columns={"y_prob_oof": "lstm"})

    oof = lgbm.merge(xgb,  on=["date", "country"], how="inner") \
              .merge(lstm, on=["date", "country"], how="inner")
    assert len(oof) == len(lgbm), f"OOF merge lost rows: {len(oof)} != {len(lgbm)}"
    print(f"  OOF merged: {len(oof):,} rows | years: {sorted(pd.to_datetime(oof['date']).dt.year.unique())}")
    return oof


def load_val() -> pd.DataFrame:
    """Merge LGBM, XGBoost, and LSTM validation predictions."""
    lgbm = pd.read_csv(LGBM_VAL)[["date", "country", "y_true", "y_prob"]].rename(
        columns={"y_prob": "lgbm"})
    xgb  = pd.read_csv(XGB_VAL)[["date", "country", "y_prob"]].rename(
        columns={"y_prob": "xgb"})
    lstm = pd.read_csv(LSTM_VAL)[["date", "country", "y_prob"]].rename(
        columns={"y_prob": "lstm"})

    val = lgbm.merge(xgb,  on=["date", "country"], how="inner") \
              .merge(lstm, on=["date", "country"], how="inner")
    assert len(val) == len(lgbm), f"Val merge lost rows: {len(val)} != {len(lgbm)}"
    print(f"  Val merged: {len(val):,} rows | pos_rate={val['y_true'].mean():.4f}")
    return val


def load_test() -> pd.DataFrame:
    """Merge LGBM, XGBoost, and LSTM test predictions (no y_true)."""
    lgbm = pd.read_csv(LGBM_TEST)[["date", "country", "y_prob"]].rename(columns={"y_prob": "lgbm"})
    xgb  = pd.read_csv(XGB_TEST)[["date",  "country", "y_prob"]].rename(columns={"y_prob": "xgb"})
    lstm = pd.read_csv(LSTM_TEST)[["date",  "country", "y_prob"]].rename(columns={"y_prob": "lstm"})

    test = lgbm.merge(xgb,  on=["date", "country"], how="inner") \
               .merge(lstm, on=["date", "country"], how="inner")
    assert len(test) == len(lgbm), f"Test merge lost rows: {len(test)} != {len(lgbm)}"
    print(f"  Test merged: {len(test):,} rows | date range: {test['date'].min()} – {test['date'].max()}")
    return test


# ─────────────────────────────────────────────────────────────────────────────
# Calibration helpers
# ─────────────────────────────────────────────────────────────────────────────

def fit_platt(y_true, y_prob_raw):
    """Platt scaling: LogReg on raw stacking probabilities."""
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(y_prob_raw.reshape(-1, 1), y_true)
    return lr


def apply_platt(model, y_prob_raw):
    return model.predict_proba(y_prob_raw.reshape(-1, 1))[:, 1]


def fit_isotonic(y_true, y_prob_raw):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob_raw, y_true)
    return iso


def apply_isotonic(model, y_prob_raw):
    return model.predict(y_prob_raw)


def build_calibration_csv(y_true, y_prob_raw, y_prob_platt, y_prob_iso, n_bins=10):
    """Build calibration comparison table across bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob_raw >= lo) & (y_prob_raw < hi if hi < 1 else y_prob_raw <= hi)
        n = mask.sum()
        if n == 0:
            continue
        rows.append({
            "bin_lo": round(lo, 2), "bin_hi": round(hi, 2), "n": int(n),
            "frac_pos":  float(y_true[mask].mean()),
            "mean_raw":  float(y_prob_raw[mask].mean()),
            "mean_platt":float(y_prob_platt[mask].mean()),
            "mean_iso":  float(y_prob_iso[mask].mean()),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Diversity and overlap analysis
# ─────────────────────────────────────────────────────────────────────────────

def top_k_set(y_prob, k=0.05):
    n = len(y_prob)
    top_k = max(1, int(np.ceil(n * k)))
    return set(np.argsort(y_prob)[::-1][:top_k])


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def overlap_ratio(set_a, set_b):
    """Fraction of set_a that also appears in set_b."""
    if not set_a:
        return 0.0
    return len(set_a & set_b) / len(set_a)


# ─────────────────────────────────────────────────────────────────────────────
# Report writing
# ─────────────────────────────────────────────────────────────────────────────

def _corr_to_md(corr_df: pd.DataFrame) -> str:
    """Render a correlation DataFrame as a markdown table without tabulate."""
    cols = list(corr_df.columns)
    header = "| | " + " | ".join(str(c) for c in cols) + " |"
    sep    = "|---|" + "---|" * len(cols)
    rows   = [header, sep]
    for idx, row in corr_df.iterrows():
        vals = " | ".join(f"{row[c]:.4f}" for c in cols)
        rows.append(f"| {idx} | {vals} |")
    return "\n".join(rows)


def write_metrics_md(all_metrics: dict, best_c: float, meta_coefs: dict,
                     corr_df: pd.DataFrame, diversity: dict):
    lines = [
        f"# Stacking LGBM+XGB+LSTM — Validation Metrics",
        f"",
        f"> Experiment: `{EXPERIMENT}` | Owner: `{OWNER}`",
        f"> Meta-learner best C: {best_c}",
        f"",
        f"## Base Model Metrics",
        f"",
        f"| Model | PR-AUC | P@5% | Brier | ECE |",
        f"|-------|--------|------|-------|-----|",
    ]
    for model in ["lgbm", "xgb", "lstm"]:
        m = all_metrics[model]
        lines.append(f"| {model.upper()} | {m['pr_auc']:.4f} | {m['p_at_top5pct']:.4f} "
                     f"| {m['brier_score']:.4f} | {m['ece']:.4f} |")

    lines += [
        f"",
        f"## Stacking Metrics",
        f"",
        f"| Variant | PR-AUC | P@5% | Brier | ECE |",
        f"|---------|--------|------|-------|-----|",
    ]
    for variant in ["stack_raw", "stack_platt", "stack_isotonic"]:
        m = all_metrics[variant]
        lines.append(f"| {variant} | {m['pr_auc']:.4f} | {m['p_at_top5pct']:.4f} "
                     f"| {m['brier_score']:.4f} | {m['ece']:.4f} |")

    lines += [
        f"",
        f"## Tree-Only vs LGBM+XGB+LSTM (Platt PR-AUC delta)",
        f"",
        f"| Item | Tree-only | +LSTM | delta |",
        f"|------|-----------|-------|-------|",
    ]
    tree = TREE_ONLY_REF
    lstm_stack = all_metrics["stack_platt"]
    for metric, tree_val, lstm_val in [
        ("PR-AUC", tree["platt_pr_auc"], lstm_stack["pr_auc"]),
        ("P@5%",   tree["platt_p5pct"],  lstm_stack["p_at_top5pct"]),
        ("ECE",    tree["platt_ece"],    lstm_stack["ece"]),
    ]:
        delta = lstm_val - tree_val
        sign  = "+" if delta >= 0 else ""
        lines.append(f"| {metric} | {tree_val:.4f} | {lstm_val:.4f} | {sign}{delta:.4f} |")

    lines += [
        f"",
        f"## Meta-Model Coefficients",
        f"",
        f"| Base Model | Coefficient |",
        f"|------------|-------------|",
    ]
    for model, coef in meta_coefs.items():
        lines.append(f"| {model} | {coef:.4f} |")

    lines += [
        f"",
        f"## Base Model Probability Correlations (Val)",
        f"",
        _corr_to_md(corr_df),
        f"",
        f"## Top-5% Alert Overlap (Jaccard / Recall)",
        f"",
        f"| Pair | Jaccard | A-in-B | B-in-A |",
        f"|------|---------|--------|--------|",
    ]
    for pair, j, a_in_b, b_in_a in diversity["overlaps"]:
        lines.append(f"| {pair} | {j:.3f} | {a_in_b:.3f} | {b_in_a:.3f} |")

    lines.append("")
    with open(OUTPUT_FILES["metrics_md"], "w") as f:
        f.write("\n".join(lines))
    print(f"  Metrics MD: {os.path.relpath(OUTPUT_FILES['metrics_md'], _ROOT)}")


def write_comparison_files(all_metrics: dict):
    """Write tree-only vs LGBM+XGB+LSTM comparison CSV and MD."""
    tree = TREE_ONLY_REF
    stk  = all_metrics["stack_platt"]
    rows = []
    metrics_to_compare = [
        ("pr_auc", "PR-AUC"),
        ("p_at_top5pct", "P@5%"),
        ("p_at_top10pct", "P@10%"),
        ("recall_at_precision_010", "Recall@Prec>=0.10"),
        ("recall_at_precision_020", "Recall@Prec>=0.20"),
        ("recall_at_precision_030", "Recall@Prec>=0.30"),
        ("brier_score", "Brier Score"),
        ("ece", "ECE"),
    ]
    ref_map = {
        "pr_auc": tree["platt_pr_auc"],
        "p_at_top5pct": tree["platt_p5pct"],
        "ece": tree["platt_ece"],
    }

    tree_full = {}
    tree_path = os.path.join(REPORT_DIR, "stacking_tree_only_12y_with_mask_feature_val_metrics.json")
    if os.path.exists(tree_path):
        with open(tree_path) as f:
            tree_full = json.load(f)

    for key, label in metrics_to_compare:
        tree_val = tree_full.get(key, ref_map.get(key, None))
        lstm_val = stk.get(key)
        delta = (lstm_val - tree_val) if tree_val is not None and lstm_val is not None else None
        rows.append({
            "metric": label,
            "tree_only_platt": tree_val,
            "lgbm_xgb_lstm_platt": lstm_val,
            "delta": delta,
        })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(OUTPUT_FILES["compare_csv"], index=False)

    lines = [
        f"# tree-only vs LGBM+XGB+LSTM Stacking Comparison (Platt 기준)",
        f"",
        f"| Metric | tree-only | +LSTM | delta |",
        f"|--------|-----------|-------|-------|",
    ]
    for _, row in comp_df.iterrows():
        tv = f"{row['tree_only_platt']:.4f}" if row["tree_only_platt"] is not None else "N/A"
        lv = f"{row['lgbm_xgb_lstm_platt']:.4f}" if row["lgbm_xgb_lstm_platt"] is not None else "N/A"
        dv = (f"+{row['delta']:.4f}" if row["delta"] >= 0 else f"{row['delta']:.4f}") \
             if row["delta"] is not None else "N/A"
        lines.append(f"| {row['metric']} | {tv} | {lv} | {dv} |")
    lines.append("")
    with open(OUTPUT_FILES["compare_md"], "w") as f:
        f.write("\n".join(lines))
    print(f"  Comparison files saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    check_existing_outputs()
    ensure_dirs()

    # ── Load and merge Level 0 predictions ────────────────────────────────────
    print("\n[1] Loading Level 0 prediction files...")
    oof  = load_oof()
    val  = load_val()
    test = load_test()

    X_oof  = oof[["lgbm", "xgb", "lstm"]].values
    y_oof  = oof["y_true"].values
    X_val  = val[["lgbm", "xgb", "lstm"]].values
    y_val  = val["y_true"].values
    X_test = test[["lgbm", "xgb", "lstm"]].values

    # ── C search for meta-learner ──────────────────────────────────────────────
    print("\n[2] Searching meta-learner C on val PR-AUC...")
    best_c, best_pr_auc = None, -1.0
    for c in META_C_CANDIDATES:
        meta = LogisticRegression(C=c, class_weight="balanced",
                                  max_iter=1000, random_state=RANDOM_SEED)
        meta.fit(X_oof, y_oof)
        val_raw_prob = meta.predict_proba(X_val)[:, 1]
        pr_auc = compute_pr_auc(y_val, val_raw_prob)
        print(f"  C={c:6} → val PR-AUC={pr_auc:.4f}")
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_c = c

    print(f"  Best C={best_c} (val PR-AUC={best_pr_auc:.4f})")

    # ── Fit final meta-model on full OOF ──────────────────────────────────────
    print("\n[3] Fitting final meta-model on full OOF...")
    meta_final = LogisticRegression(C=best_c, class_weight="balanced",
                                    max_iter=1000, random_state=RANDOM_SEED)
    meta_final.fit(X_oof, y_oof)

    coef = meta_final.coef_[0]
    meta_coefs = {"lgbm": float(coef[0]), "xgb": float(coef[1]), "lstm": float(coef[2])}
    print(f"  Meta coefs — LGBM:{coef[0]:.4f}  XGB:{coef[1]:.4f}  LSTM:{coef[2]:.4f}")

    joblib.dump(meta_final, OUTPUT_FILES["model_meta"])
    print(f"  Meta model saved: {os.path.relpath(OUTPUT_FILES['model_meta'], _ROOT)}")

    # ── Raw stacking predictions ───────────────────────────────────────────────
    val_stack_raw  = meta_final.predict_proba(X_val)[:, 1]
    test_stack_raw = meta_final.predict_proba(X_test)[:, 1]

    # ── Platt calibration ─────────────────────────────────────────────────────
    print("\n[4] Fitting Platt calibration on val...")
    platt = fit_platt(y_val, val_stack_raw)
    val_stack_platt  = apply_platt(platt, val_stack_raw)
    test_stack_platt = apply_platt(platt, test_stack_raw)
    joblib.dump(platt, OUTPUT_FILES["model_platt"])

    # ── Isotonic calibration ──────────────────────────────────────────────────
    print("  Fitting Isotonic calibration on val...")
    iso = fit_isotonic(y_val, val_stack_raw)
    val_stack_iso  = apply_isotonic(iso, val_stack_raw)
    test_stack_iso = apply_isotonic(iso, test_stack_raw)
    joblib.dump(iso, OUTPUT_FILES["model_isotonic"])

    # ── Validation metrics ─────────────────────────────────────────────────────
    print("\n[5] Computing validation metrics...")
    all_metrics = {
        "lgbm":          compute_metrics(y_val, val[:]["lgbm"]),
        "xgb":           compute_metrics(y_val, val[:]["xgb"]),
        "lstm":          compute_metrics(y_val, val[:]["lstm"]),
        "stack_raw":     compute_metrics(y_val, val_stack_raw),
        "stack_platt":   compute_metrics(y_val, val_stack_platt),
        "stack_isotonic":compute_metrics(y_val, val_stack_iso),
    }

    print("\n  Validation metrics:")
    print(f"  {'Model':<20} {'PR-AUC':>8} {'P@5%':>8} {'Brier':>8} {'ECE':>8}")
    print(f"  {'-'*56}")
    for name, m in all_metrics.items():
        print(f"  {name:<20} {m['pr_auc']:>8.4f} {m['p_at_top5pct']:>8.4f} "
              f"{m['brier_score']:>8.4f} {m['ece']:>8.4f}")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    full_metrics = {
        "experiment": EXPERIMENT,
        "best_meta_C": best_c,
        "meta_coefs": meta_coefs,
        **{f"{k}_{metric}": v
           for k, m in all_metrics.items()
           for metric, v in m.items()},
    }
    with open(OUTPUT_FILES["metrics_json"], "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"\n  Metrics JSON: {os.path.relpath(OUTPUT_FILES['metrics_json'], _ROOT)}")

    # ── Calibration CSV ───────────────────────────────────────────────────────
    calib_df = build_calibration_csv(y_val, val_stack_raw, val_stack_platt, val_stack_iso)
    calib_df.to_csv(OUTPUT_FILES["calib_csv"], index=False)
    print(f"  Calibration CSV: {os.path.relpath(OUTPUT_FILES['calib_csv'], _ROOT)}")

    # ── Diversity analysis ────────────────────────────────────────────────────
    print("\n[6] Diversity analysis...")
    # Correlation matrix
    corr_df = val[["lgbm", "xgb", "lstm"]].corr().round(4)
    print("\n  Probability correlations (val):")
    print(corr_df)

    # Top-5% alert overlap
    lgbm_top = top_k_set(val["lgbm"].values)
    xgb_top  = top_k_set(val["xgb"].values)
    lstm_top = top_k_set(val["lstm"].values)

    # Load tree-only val for comparison
    tree_val_path = os.path.join(PRED_DIR,
        "val_predictions__stacking_tree_only_12y_with_mask_feature__D_byeonghyeon.csv")
    if os.path.exists(tree_val_path):
        tree_val_df = pd.read_csv(tree_val_path)
        # Merge by (date,country) to align indices with current val
        merged_tree = val[["date","country"]].merge(
            tree_val_df[["date","country","y_prob_stack_platt"]], on=["date","country"], how="left")
        tree_top = top_k_set(merged_tree["y_prob_stack_platt"].fillna(0).values)
    else:
        tree_top = None

    lstm_stack_top = top_k_set(val_stack_platt)

    overlaps = [
        ("LGBM vs XGB",  jaccard(lgbm_top, xgb_top),  overlap_ratio(lgbm_top, xgb_top),  overlap_ratio(xgb_top,  lgbm_top)),
        ("LGBM vs LSTM", jaccard(lgbm_top, lstm_top),  overlap_ratio(lgbm_top, lstm_top), overlap_ratio(lstm_top, lgbm_top)),
        ("XGB vs LSTM",  jaccard(xgb_top,  lstm_top),  overlap_ratio(xgb_top,  lstm_top), overlap_ratio(lstm_top, xgb_top)),
    ]
    if tree_top is not None:
        overlaps.append(("tree-only vs +LSTM stack",
                         jaccard(tree_top, lstm_stack_top),
                         overlap_ratio(tree_top, lstm_stack_top),
                         overlap_ratio(lstm_stack_top, tree_top)))

    diversity = {"overlaps": overlaps}
    print("\n  Top-5% alert overlap:")
    print(f"  {'Pair':<30} {'Jaccard':>8} {'A⊂B':>8} {'B⊂A':>8}")
    for pair, j, a_in_b, b_in_a in overlaps:
        print(f"  {pair:<30} {j:>8.3f} {a_in_b:>8.3f} {b_in_a:>8.3f}")

    # ── Save validation output ────────────────────────────────────────────────
    print("\n[7] Saving prediction files...")
    val_out = val[["date", "country", "y_true", "lgbm", "xgb", "lstm"]].copy()
    val_out = val_out.rename(columns={"lgbm": "y_prob_lgbm", "xgb": "y_prob_xgb", "lstm": "y_prob_lstm"})
    val_out["y_prob_stack_raw"]      = val_stack_raw
    val_out["y_prob_stack_platt"]    = val_stack_platt
    val_out["y_prob_stack_isotonic"] = val_stack_iso

    assert val_out.duplicated(["date", "country"]).sum() == 0
    assert val_out["y_prob_stack_platt"].between(0, 1).all()
    val_out.to_csv(OUTPUT_FILES["val_stack"], index=False)
    print(f"  Val stacking saved: {len(val_out):,} rows")

    # Test prediction files (one per calibration variant)
    for key, probs in [("test_raw", test_stack_raw),
                       ("test_platt", test_stack_platt),
                       ("test_isotonic", test_stack_iso)]:
        out = test[["date", "country"]].copy()
        out["y_prob"] = probs
        assert out.duplicated(["date", "country"]).sum() == 0
        assert out["y_prob"].between(0, 1).all()
        out.to_csv(OUTPUT_FILES[key], index=False)
        print(f"  Test {key} saved: {len(out):,} rows")

    # ── Write report files ────────────────────────────────────────────────────
    print("\n[8] Writing report files...")
    write_metrics_md(all_metrics, best_c, meta_coefs, corr_df, diversity)
    write_comparison_files(all_metrics)

    # ── Final summary ──────────────────────────────────────────────────────────
    platt_m = all_metrics["stack_platt"]
    tree_pr = TREE_ONLY_REF["platt_pr_auc"]
    delta_pr = platt_m["pr_auc"] - tree_pr
    sign = "+" if delta_pr >= 0 else ""

    print(f"\n{'='*60}")
    print(f"  {EXPERIMENT}")
    print(f"{'='*60}")
    print(f"  Best meta C       : {best_c}")
    print(f"  Meta coefs        : LGBM={coef[0]:.3f}  XGB={coef[1]:.3f}  LSTM={coef[2]:.3f}")
    print(f"  Stacking Platt PR-AUC : {platt_m['pr_auc']:.4f}  "
          f"(tree-only: {tree_pr:.4f}, delta: {sign}{delta_pr:.4f})")
    print(f"  Stacking Platt P@5%   : {platt_m['p_at_top5pct']:.4f}  "
          f"(tree-only: {TREE_ONLY_REF['platt_p5pct']:.4f})")
    print(f"  Stacking Platt ECE    : {platt_m['ece']:.4f}  "
          f"(tree-only: {TREE_ONLY_REF['platt_ece']:.4f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
