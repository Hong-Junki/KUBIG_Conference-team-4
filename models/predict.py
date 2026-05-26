"""
Model inference — produces model_predictions from the feature table.

Output schema is fixed by MODEL_PREDICTION_COLUMNS regardless of the model used.
When no real model is available, generate_mock_predictions() is used as fallback.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_schema import (
    FEATURE_KEY_COLUMNS,
    FEATURE_TABLE_PATH,
    MODEL_PREDICTION_COLUMNS,
    MODEL_PREDICTIONS_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_VERSION,
    validate_required_columns,
    ensure_datetime_date_column,
)
from models.model_loader import load_model_bundle

logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = set(FEATURE_KEY_COLUMNS)


# ── Feature loading ───────────────────────────────────────────────────────────

def load_features(feature_path: str = FEATURE_TABLE_PATH) -> pd.DataFrame:
    """Loads feature table parquet and validates key columns."""
    path = Path(feature_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")

    df = pd.read_parquet(path)
    validate_required_columns(df, FEATURE_KEY_COLUMNS, table_name="feature_table")
    df = ensure_datetime_date_column(df)
    logger.info(f"Features loaded: {len(df)} rows from {path}")
    return df


def align_features_for_model(
    features: pd.DataFrame,
    feature_list: list[str] | None,
) -> np.ndarray:
    """
    Returns feature matrix X with columns ordered by feature_list.

    Falls back to all numeric non-key columns when feature_list is None.
    Fills missing columns with 0.0 and warns about them.
    """
    if feature_list is not None:
        missing_cols = [c for c in feature_list if c not in features.columns]
        if missing_cols:
            logger.warning(
                f"Feature list contains {len(missing_cols)} columns absent from "
                f"feature table — filling with 0.0: {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}"
            )
            for col in missing_cols:
                features = features.copy()
                features[col] = 0.0
        return features[feature_list].to_numpy(dtype=float)

    # Fallback: all numeric columns that are not key columns
    numeric_cols = [
        c for c in features.select_dtypes(include=[np.number]).columns
        if c not in NON_FEATURE_COLUMNS
    ]
    logger.info(f"No feature_list provided — using {len(numeric_cols)} numeric columns")
    return features[numeric_cols].to_numpy(dtype=float)


# ── Mock prediction ───────────────────────────────────────────────────────────

def generate_mock_predictions(
    features: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> pd.DataFrame:
    """
    Generates deterministic mock predicted_probability values.

    Values are derived from (country, date) so the same inputs always
    produce the same output — useful for reproducible pipeline tests.
    """
    probs = []
    for _, row in features.iterrows():
        country_val = sum(ord(c) for c in str(row["country"]))
        date_val = pd.Timestamp(row["date"]).toordinal() if hasattr(row["date"], "toordinal") else 0
        seed = (country_val * 31 + date_val) % (2**32)
        rng = np.random.default_rng(seed=seed)
        probs.append(float(np.clip(rng.beta(a=2, b=20), 0.0, 1.0)))

    now = datetime.now(timezone.utc)
    return pd.DataFrame({
        "country":               features["country"].values,
        "date":                  features["date"].values,
        "model_name":            model_name,
        "model_version":         model_version,
        "predicted_probability": probs,
        "created_at":            now,
    })


# ── Real model inference ──────────────────────────────────────────────────────

def run_model_predictions(
    features: pd.DataFrame,
    model_bundle: dict,
) -> pd.DataFrame:
    """
    Runs inference using the real model artifact, or falls back to mock.

    Attempts predict_proba first (positive class probability), then predict.
    All probabilities are clipped to [0, 1].
    """
    model = model_bundle.get("model")
    feature_list = model_bundle.get("feature_list")
    metadata = model_bundle.get("metadata", {})
    model_name = metadata.get("model_name", DEFAULT_MODEL_NAME)
    model_version = metadata.get("model_version", DEFAULT_MODEL_VERSION)

    if model is None:
        logger.info("No model artifact — generating mock predictions.")
        return generate_mock_predictions(features, model_name, model_version)

    X = align_features_for_model(features, feature_list)

    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)
        probs = raw[:, 1] if raw.ndim == 2 else raw
    elif hasattr(model, "predict"):
        probs = model.predict(X)
    else:
        logger.warning("Model has neither predict_proba nor predict — using mock.")
        return generate_mock_predictions(features, model_name, model_version)

    probs = np.clip(probs, 0.0, 1.0)
    now = datetime.now(timezone.utc)

    return pd.DataFrame({
        "country":               features["country"].values,
        "date":                  features["date"].values,
        "model_name":            model_name,
        "model_version":         model_version,
        "predicted_probability": probs.astype(float),
        "created_at":            now,
    })


# ── Save ──────────────────────────────────────────────────────────────────────

def save_model_predictions(
    predictions: pd.DataFrame,
    output_dir: str = MODEL_PREDICTIONS_DIR,
) -> tuple[Path, Path]:
    """Saves predictions as parquet + CSV. Returns (parquet_path, csv_path)."""
    validate_required_columns(predictions, MODEL_PREDICTION_COLUMNS, "model_predictions")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    parquet_path = out / "model_predictions.parquet"
    csv_path = out / "model_predictions.csv"

    predictions.to_parquet(parquet_path, index=False)
    predictions.to_csv(csv_path, index=False)

    logger.info(f"Predictions saved: {parquet_path} ({len(predictions)} rows)")
    return parquet_path, csv_path


# ── Pipeline step ─────────────────────────────────────────────────────────────

def run_inference(
    feature_path: str = FEATURE_TABLE_PATH,
    output_dir: str = MODEL_PREDICTIONS_DIR,
    artifact_dir: str = "artifacts",
) -> tuple[Path, Path]:
    """
    Full inference step: load features → load model → predict → save.

    Returns (parquet_path, csv_path) of the saved predictions.
    """
    features = load_features(feature_path)
    model_bundle = load_model_bundle(artifact_dir)
    predictions = run_model_predictions(features, model_bundle)
    return save_model_predictions(predictions, output_dir)


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument("--feature-path", default=FEATURE_TABLE_PATH)
    parser.add_argument("--output-dir", default=MODEL_PREDICTIONS_DIR)
    parser.add_argument("--artifact-dir", default="artifacts")
    args = parser.parse_args()

    parquet_path, csv_path = run_inference(
        feature_path=args.feature_path,
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
    )
    print(f"Saved: {parquet_path}")
    print(f"Saved: {csv_path}")
