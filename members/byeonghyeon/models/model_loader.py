"""
Model artifact loader.

Loads model, feature_list, and metadata from artifacts/.
Returns None (or safe defaults) when files are absent so the pipeline
can fall back to mock prediction without crashing.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_DIR = "artifacts"
DEFAULT_MODEL_PATH = "artifacts/final_model.pkl"
DEFAULT_FEATURE_LIST_PATH = "artifacts/feature_list.json"
DEFAULT_METADATA_PATH = "artifacts/model_metadata.json"

_DEFAULT_METADATA = {
    "model_name": "mock_model",
    "model_version": "v0_skeleton",
}


# ── Individual loaders ────────────────────────────────────────────────────────

def load_pickle_model(model_path: str = DEFAULT_MODEL_PATH) -> Any | None:
    """
    Loads a pickled model object.

    Tries joblib first (preferred for sklearn), falls back to pickle.
    Returns None if the file does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        logger.warning(f"Model artifact not found: {path}. Will use mock prediction.")
        return None

    try:
        import joblib
        model = joblib.load(path)
        logger.info(f"Model loaded via joblib: {path}")
        return model
    except Exception:
        pass

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded via pickle: {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        return None


def load_feature_list(feature_list_path: str = DEFAULT_FEATURE_LIST_PATH) -> list[str] | None:
    """
    Loads the ordered feature column list from a JSON file.

    Returns None if the file does not exist so callers can fall back to
    inferring features from the feature table columns.
    """
    path = Path(feature_list_path)
    if not path.exists():
        logger.warning(f"feature_list.json not found: {path}. Feature order will be inferred.")
        return None

    try:
        with open(path, encoding="utf-8") as f:
            feature_list = json.load(f)
        logger.info(f"Feature list loaded: {len(feature_list)} features from {path}")
        return feature_list
    except Exception as e:
        logger.error(f"Failed to load feature list from {path}: {e}")
        return None


def load_model_metadata(metadata_path: str = DEFAULT_METADATA_PATH) -> dict:
    """
    Loads model metadata (name, version, target variable, etc.).

    Returns a safe default dict when the file does not exist.
    """
    path = Path(metadata_path)
    if not path.exists():
        logger.warning(f"model_metadata.json not found: {path}. Using default metadata.")
        return dict(_DEFAULT_METADATA)

    try:
        with open(path, encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info(f"Model metadata loaded from {path}: {metadata}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to load metadata from {path}: {e}")
        return dict(_DEFAULT_METADATA)


# ── Bundle loader ─────────────────────────────────────────────────────────────

def load_model_bundle(artifact_dir: str = DEFAULT_ARTIFACT_DIR) -> dict:
    """
    Loads model, feature_list, and metadata as a single bundle dict.

    Returns:
        {
            "model":        <model object or None>,
            "feature_list": <list[str] or None>,
            "metadata":     <dict with model_name, model_version, ...>,
        }
    When model is None, the pipeline falls back to mock prediction.
    """
    base = Path(artifact_dir)
    bundle = {
        "model":        load_pickle_model(str(base / "final_model.pkl")),
        "feature_list": load_feature_list(str(base / "feature_list.json")),
        "metadata":     load_model_metadata(str(base / "model_metadata.json")),
    }

    if bundle["model"] is None:
        logger.info("No model artifact found — pipeline will run in mock-prediction mode.")
    else:
        logger.info(
            f"Model bundle ready: {bundle['metadata'].get('model_name')} "
            f"v{bundle['metadata'].get('model_version')}"
        )
    return bundle
