"""
LightGBM + SE score prediction script.

Loads the saved SE model and generates y_prob for the test set.
The test set is NEVER used for training or model selection.

Usage (from project root):
    python modeling/predict_lightgbm_se.py

Output:
    outputs/predictions/predictions__lightgbm_se__byeonghyeon.csv
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd

from utils import (
    DATE_COL, CATEGORICAL_COLS,
    TEST_PATH, PRED_DIR,
    ensure_output_dirs,
    validate_prediction_file,
)

# ── SE-specific paths ─────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SE_SCORE_PATH = os.path.join(_ROOT, "input", "processed", "dataset", "full_se.parquet")
MODEL_PATH_SE = os.path.join(_ROOT, "outputs", "models", "lightgbm_se.pkl")
PRED_PATH_SE  = os.path.join(PRED_DIR, "predictions__lightgbm_se__byeonghyeon.csv")


def load_se_scores():
    """Load only date, country, macis_se_score from full_se.parquet."""
    return pd.read_parquet(SE_SCORE_PATH, columns=["date", "country", "macis_se_score"])


def merge_se_score(df, se_df, split_name):
    """
    Left-merge macis_se_score onto df by date + country.
    Validates that row count does not change after merge.
    """
    n_before = len(df)
    merged = df.merge(se_df, on=["date", "country"], how="left")
    n_after = len(merged)

    if n_before != n_after:
        raise ValueError(
            f"{split_name}: row count changed after merge "
            f"({n_before:,} → {n_after:,}). Check for duplicate keys in full_se.parquet."
        )

    n_null = merged["macis_se_score"].isnull().sum()
    if n_null > 0:
        print(f"  WARNING: {split_name} has {n_null:,} null macis_se_score → filling with 0")
        merged["macis_se_score"] = merged["macis_se_score"].fillna(0)

    print(f"  {split_name}: {n_after:,} rows  macis_se_score null={n_null}")
    return merged


def make_features_from_artifact(df, feature_cols, country_categories):
    """Reconstruct feature matrix using artifact metadata."""
    X = df[feature_cols].copy()
    if "country" in X.columns:
        X["country"] = pd.Categorical(X["country"], categories=country_categories)
    return X


def main():
    ensure_output_dirs()
    sep = "=" * 60

    # ── 1. Load model artifact ────────────────────────────────────────────────
    print(sep)
    print("Step 1 — Load SE model artifact")
    print(sep)
    artifact = joblib.load(MODEL_PATH_SE)
    model              = artifact["model"]
    feature_cols       = artifact["feature_cols"]
    country_categories = artifact["country_categories"]

    n_numeric = len([c for c in feature_cols if c not in CATEGORICAL_COLS])
    n_cat     = len([c for c in feature_cols if c in CATEGORICAL_COLS])
    print(f"  Model loaded       : {MODEL_PATH_SE}")
    print(f"  Best iteration     : {model.best_iteration}")
    print(f"  Features           : {len(feature_cols)} total  "
          f"({n_numeric} numeric, {n_cat} categorical)")
    print(f"  Country categories : {len(country_categories)}")

    # ── 2. Load SE scores ─────────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 2 — Load SE scores")
    print(sep)
    se_df = load_se_scores()
    print(f"  Loaded {len(se_df):,} rows from full_se.parquet")

    # ── 3. Load test set and merge ────────────────────────────────────────────
    print()
    print(sep)
    print("Step 3 — Load test set and merge SE score")
    print(sep)
    test_raw = pd.read_parquet(TEST_PATH)
    print(f"  test raw: {len(test_raw):,} rows  "
          f"({test_raw[DATE_COL].min().date()} ~ {test_raw[DATE_COL].max().date()})")

    test = merge_se_score(test_raw, se_df, "test")

    # ── 4. Build feature matrix ───────────────────────────────────────────────
    X_test = make_features_from_artifact(test, feature_cols, country_categories)

    # ── 5. Predict ────────────────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 4 — Predict")
    print(sep)
    y_prob = model.predict(X_test, num_iteration=model.best_iteration)
    print(f"  Predictions generated : {len(y_prob):,}")
    print(f"  y_prob range          : [{y_prob.min():.6f}, {y_prob.max():.6f}]")
    print(f"  y_prob mean           : {y_prob.mean():.6f}")

    # ── 6. Build and validate submission ─────────────────────────────────────
    pred_df = pd.DataFrame({
        "date":    pd.to_datetime(test[DATE_COL]).dt.strftime("%Y-%m-%d"),
        "country": test["country"].values,
        "y_prob":  y_prob,
    })

    print()
    print(sep)
    print("Step 5 — Validate submission")
    print(sep)
    validate_prediction_file(pred_df, test_raw)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    pred_df.to_csv(PRED_PATH_SE, index=False, encoding="utf-8")
    print()
    print(f"Predictions saved : {PRED_PATH_SE}")
    print(sep)


if __name__ == "__main__":
    main()
