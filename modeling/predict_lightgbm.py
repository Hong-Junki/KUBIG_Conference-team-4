"""
LightGBM baseline prediction script.

Loads the saved model artifact and generates y_prob for the test set.
The test set is NEVER used for training or model selection.

Usage (from project root):
    python modeling/predict_lightgbm.py

Output:
    outputs/predictions/predictions__lightgbm__byeonghyeon.csv
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd

from utils import (
    DATE_COL, CATEGORICAL_COLS,
    TEST_PATH, MODEL_PATH, PRED_PATH,
    ensure_output_dirs,
    validate_prediction_file,
)


def make_features_from_artifact(df, feature_cols, country_categories):
    """
    Reconstruct feature matrix using the exact column list and country category
    mapping stored in the training artifact.
    """
    X = df[feature_cols].copy()
    if "country" in X.columns:
        X["country"] = pd.Categorical(X["country"], categories=country_categories)
    return X


def main():
    ensure_output_dirs()
    sep = "=" * 60

    # ── 1. Load model artifact ────────────────────────────────────────────────
    print(sep)
    print("Step 1 — Load model artifact")
    print(sep)
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    country_categories = artifact["country_categories"]

    print(f"  Model loaded       : {MODEL_PATH}")
    print(f"  Best iteration     : {model.best_iteration}")
    print(f"  Features           : {len(feature_cols)} total  "
          f"({len([c for c in feature_cols if c not in CATEGORICAL_COLS])} numeric, "
          f"{len([c for c in feature_cols if c in CATEGORICAL_COLS])} categorical)")
    print(f"  Country categories : {len(country_categories)}")

    # ── 2. Load test set only ─────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 2 — Load test set")
    print(sep)
    test = pd.read_parquet(TEST_PATH)
    print(f"  test : {test.shape[0]:>6,} rows × {test.shape[1]} cols  "
          f"({test[DATE_COL].min().date()} ~ {test[DATE_COL].max().date()})")

    # ── 3. Build feature matrix ───────────────────────────────────────────────
    X_test = make_features_from_artifact(test, feature_cols, country_categories)

    # ── 4. Predict ────────────────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 3 — Predict")
    print(sep)
    y_prob = model.predict(X_test, num_iteration=model.best_iteration)
    print(f"  Predictions generated : {len(y_prob):,}")
    print(f"  y_prob range          : [{y_prob.min():.6f}, {y_prob.max():.6f}]")
    print(f"  y_prob mean           : {y_prob.mean():.6f}")

    # ── 5. Build submission DataFrame ────────────────────────────────────────
    pred_df = pd.DataFrame({
        "date": pd.to_datetime(test[DATE_COL]).dt.strftime("%Y-%m-%d"),
        "country": test["country"].values,
        "y_prob": y_prob,
    })

    # ── 6. Validate ───────────────────────────────────────────────────────────
    print()
    print(sep)
    print("Step 4 — Validate submission")
    print(sep)
    validate_prediction_file(pred_df, test)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    pred_df.to_csv(PRED_PATH, index=False, encoding="utf-8")
    print()
    print(f"Predictions saved : {PRED_PATH}")
    print(sep)


if __name__ == "__main__":
    main()
