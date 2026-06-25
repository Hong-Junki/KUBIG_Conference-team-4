"""
Feature table, model_predictions, dashboard_country_risk schema constants.

Final feature list is NOT fixed here — get_example_feature_columns() returns
illustrative candidates only. Replace with artifacts/feature_list.json values
once the model team delivers the final artifact.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Key columns ─────────────────────────────────────────────────────────────
FEATURE_KEY_COLUMNS: list[str] = ["country", "date"]

# ── Model predictions output schema ─────────────────────────────────────────
MODEL_PREDICTION_COLUMNS: list[str] = [
    "country",
    "date",
    "model_name",
    "model_version",
    "predicted_probability",
    "created_at",
]

# ── Dashboard output schema ──────────────────────────────────────────────────
# Note: data_quality_flag replaces the ambiguous "confidence_level" name.
DASHBOARD_RISK_COLUMNS: list[str] = [
    "country",
    "date",
    "model_name",
    "model_version",
    "predicted_probability",
    "temperature_score",
    "risk_level",
    "delta_1d",
    "delta_7d",
    "rank_today",
    "data_quality_flag",
    "main_driver_1",
    "main_driver_2",
    "main_driver_3",
    "updated_at",
]

# ── Default identifiers (used until real artifact is loaded) ─────────────────
DEFAULT_MODEL_NAME: str = "mock_model"
DEFAULT_MODEL_VERSION: str = "v0_skeleton"

# ── Default paths ────────────────────────────────────────────────────────────
FEATURE_TABLE_PATH: str = "input/processed/features_country_daily.parquet"
MODEL_PREDICTIONS_DIR: str = "outputs/predictions"
DASHBOARD_OUTPUT_DIR: str = "outputs/dashboard"
DASHBOARD_PUBLIC_DIR: str = "public/data"
ARTIFACTS_DIR: str = "artifacts"


# ── Example feature columns (illustrative only — not the final model list) ──

def get_example_feature_columns() -> list[str]:
    """
    Returns illustrative feature column candidates for schema testing.

    These are NOT the final model features. Replace with feature_list.json
    values from the model artifact once the model team confirms the final list.
    """
    acled = [
        "acled_event_count_lag7_7d",
        "acled_fatalities_lag7_30d",
        "acled_battle_count_lag7_14d",
        "acled_missing_flag",
    ]
    gdelt_hist = [
        "gdelt_num_articles_1d",
        "gdelt_avg_tone_7d",
        "gdelt_goldstein_mean_7d",
        "gdelt_num_mentions_7d",
    ]
    gdelt_recent = [
        "gdelt_doc_volume_1d",
        "gdelt_doc_volume_zscore_30d",
        "gdelt_doc_tone_7d",
    ]
    economic = [
        "VIX",
        "WTI",
        "Gold",
        "DXY",
        "STLFSI4",
        "VIX_7d_change",
        "WTI_7d_change",
    ]
    time_features = [
        "day_of_week",
        "month",
        "week_of_year",
    ]
    return acled + gdelt_hist + gdelt_recent + economic + time_features


# ── Validation helpers ───────────────────────────────────────────────────────

def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    table_name: str = "DataFrame",
) -> None:
    """Raises ValueError if any required column is missing from df."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{table_name}] Missing required columns: {missing}"
        )


def ensure_datetime_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Coerces the 'date' column to datetime.date objects if present."""
    if "date" not in df.columns:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df
