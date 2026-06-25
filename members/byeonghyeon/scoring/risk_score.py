"""
Risk scoring — converts model_predictions into dashboard_country_risk.

Implements Candidate 1 (probability-only) as the default MVP scoring method.
Candidate 2 and Candidate 3 are provided as function skeletons.
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
    DASHBOARD_RISK_COLUMNS,
    DASHBOARD_OUTPUT_DIR,
    MODEL_PREDICTIONS_DIR,
    MODEL_PREDICTION_COLUMNS,
    validate_required_columns,
    ensure_datetime_date_column,
)

logger = logging.getLogger(__name__)

# ── Risk level thresholds ─────────────────────────────────────────────────────
# These are placeholder thresholds. Confirm in team review before production use.
THREE_LEVEL_THRESHOLDS = {"High": 60.0, "Medium": 30.0}   # score >= High → High, etc.
FOUR_LEVEL_THRESHOLDS  = {"Critical": 80.0, "High": 60.0, "Moderate": 30.0}


# ── Scoring candidates ────────────────────────────────────────────────────────

def score_probability_only(df: pd.DataFrame) -> pd.Series:
    """Candidate 1: temperature_score = predicted_probability × 100."""
    return (df["predicted_probability"] * 100).clip(0.0, 100.0)


def score_model_centered_composite(
    df: pd.DataFrame,
    news_signal: pd.Series | None = None,
    event_signal: pd.Series | None = None,
) -> pd.Series:
    """
    Candidate 2: Composite score using model probability + news + event signals.

    temperature_score = 100 × (0.75 × prob + 0.15 × news + 0.10 × event)

    TODO: Provide real news_signal and event_signal series from feature table.
    Currently falls back to Candidate 1 when signals are not supplied.
    """
    if news_signal is None or event_signal is None:
        logger.warning(
            "score_model_centered_composite: news_signal or event_signal not provided. "
            "Falling back to Candidate 1 (probability-only)."
        )
        return score_probability_only(df)

    score = 100.0 * (
        0.75 * df["predicted_probability"]
        + 0.15 * news_signal
        + 0.10 * event_signal
    )
    return score.clip(0.0, 100.0)


def score_rank_based(df: pd.DataFrame) -> pd.Series:
    """
    Candidate 3: Rank-based score — percentile rank of predicted_probability × 100.

    Groups by date to rank countries relative to each other on the same day.
    """
    return (
        df.groupby("date")["predicted_probability"]
        .rank(pct=True)
        .mul(100.0)
        .clip(0.0, 100.0)
    )


# ── Risk level assignment ─────────────────────────────────────────────────────

def assign_risk_level(
    temperature_score: pd.Series,
    scheme: str = "three_level",
) -> pd.Series:
    """
    Converts temperature_score into a categorical risk level.

    scheme="three_level": Low / Medium / High
    scheme="four_level":  Low / Moderate / High / Critical

    Thresholds are placeholders — team should confirm final values.
    """
    if scheme == "four_level":
        thresholds = FOUR_LEVEL_THRESHOLDS
        conditions = [
            temperature_score >= thresholds["Critical"],
            temperature_score >= thresholds["High"],
            temperature_score >= thresholds["Moderate"],
        ]
        choices = ["Critical", "High", "Moderate"]
        default = "Low"
    else:
        thresholds = THREE_LEVEL_THRESHOLDS
        conditions = [
            temperature_score >= thresholds["High"],
            temperature_score >= thresholds["Medium"],
        ]
        choices = ["High", "Medium"]
        default = "Low"

    return pd.Series(
        np.select(conditions, choices, default=default),
        index=temperature_score.index,
        dtype=str,
    )


# ── Delta and rank columns ────────────────────────────────────────────────────

def add_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds delta_1d and delta_7d (temperature_score change vs previous day/week).

    When historical rows for the same country are absent (e.g., single-date
    snapshot), fills with 0.0 as a safe default.
    """
    df = df.copy().sort_values(["country", "date"])
    df["delta_1d"] = (
        df.groupby("country")["temperature_score"]
        .diff(periods=1)
        .fillna(0.0)
    )
    df["delta_7d"] = (
        df.groupby("country")["temperature_score"]
        .diff(periods=7)
        .fillna(0.0)
    )
    return df


def add_rank_today(df: pd.DataFrame) -> pd.DataFrame:
    """Adds rank_today: descending rank of temperature_score within each date."""
    df = df.copy()
    df["rank_today"] = (
        df.groupby("date")["temperature_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return df


# ── Data quality flag ────────────────────────────────────────────────────────

def add_data_quality_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds data_quality_flag to indicate data coverage status.

    MVP logic:
      - "mock_or_unverified" when model_name contains "mock"
      - "ok" otherwise

    TODO: Replace with real data coverage checks:
      - Check acled_missing_flag in feature table
      - Check for recent collection failures
      - Check NaN ratio in key feature columns
    """
    df = df.copy()
    is_mock = df["model_name"].str.contains("mock", case=False, na=False)
    df["data_quality_flag"] = np.where(is_mock, "mock_or_unverified", "ok")
    return df


# ── Main builder ──────────────────────────────────────────────────────────────

def build_dashboard_country_risk(
    predictions: pd.DataFrame,
    scoring_method: str = "probability_only",
    risk_level_scheme: str = "three_level",
) -> pd.DataFrame:
    """
    Transforms model_predictions into dashboard_country_risk.

    scoring_method options: "probability_only" | "rank_based"
    risk_level_scheme options: "three_level" | "four_level"
    """
    validate_required_columns(predictions, MODEL_PREDICTION_COLUMNS, "model_predictions")
    df = ensure_datetime_date_column(predictions.copy())

    # Temperature score
    if scoring_method == "rank_based":
        df["temperature_score"] = score_rank_based(df)
    elif scoring_method == "probability_only":
        df["temperature_score"] = score_probability_only(df)
    else:
        logger.warning(f"Unknown scoring_method '{scoring_method}', using probability_only.")
        df["temperature_score"] = score_probability_only(df)

    # Derived columns
    df["risk_level"] = assign_risk_level(df["temperature_score"], scheme=risk_level_scheme)
    df = add_delta_columns(df)
    df = add_rank_today(df)
    df = add_data_quality_flag(df)

    # Optional columns — empty in MVP
    df["main_driver_1"] = ""
    df["main_driver_2"] = ""
    df["main_driver_3"] = ""
    df["updated_at"] = datetime.now(timezone.utc)

    # Enforce column order from schema
    missing_cols = [c for c in DASHBOARD_RISK_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dashboard schema columns missing: {missing_cols}")

    return df[DASHBOARD_RISK_COLUMNS]


# ── Save ──────────────────────────────────────────────────────────────────────

def save_dashboard_risk(
    df: pd.DataFrame,
    output_dir: str = DASHBOARD_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Saves dashboard_country_risk as parquet + CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    parquet_path = out / "dashboard_country_risk.parquet"
    csv_path = out / "dashboard_country_risk.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    logger.info(f"Dashboard risk saved: {parquet_path} ({len(df)} rows)")
    return parquet_path, csv_path


# ── Pipeline step ─────────────────────────────────────────────────────────────

def run_risk_scoring(
    prediction_path: str = f"{MODEL_PREDICTIONS_DIR}/model_predictions.parquet",
    output_dir: str = DASHBOARD_OUTPUT_DIR,
    scoring_method: str = "probability_only",
    risk_level_scheme: str = "three_level",
) -> tuple[Path, Path]:
    """
    Full scoring step: load predictions → score → save.

    Returns (parquet_path, csv_path) of dashboard_country_risk.
    """
    path = Path(prediction_path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    predictions = pd.read_parquet(path)
    logger.info(f"Loaded predictions: {len(predictions)} rows from {path}")

    df = build_dashboard_country_risk(
        predictions,
        scoring_method=scoring_method,
        risk_level_scheme=risk_level_scheme,
    )
    return save_dashboard_risk(df, output_dir)


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run risk scoring")
    parser.add_argument("--prediction-path",
                        default=f"{MODEL_PREDICTIONS_DIR}/model_predictions.parquet")
    parser.add_argument("--output-dir", default=DASHBOARD_OUTPUT_DIR)
    parser.add_argument("--scoring-method", default="probability_only",
                        choices=["probability_only", "rank_based"])
    parser.add_argument("--risk-level-scheme", default="three_level",
                        choices=["three_level", "four_level"])
    args = parser.parse_args()

    run_risk_scoring(
        prediction_path=args.prediction_path,
        output_dir=args.output_dir,
        scoring_method=args.scoring_method,
        risk_level_scheme=args.risk_level_scheme,
    )
