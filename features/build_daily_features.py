"""
Feature builder — skeleton implementation.

Current state: builds a mock feature table from the country list and a given
date. Real feature engineering (lag/shift, GDELT aggregation, STLFSI4 ffill)
is marked as TODO and will be implemented once the final feature list is fixed.

Output: input/processed/features_country_daily.parquet
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_schema import (
    FEATURE_KEY_COLUMNS,
    FEATURE_TABLE_PATH,
    get_example_feature_columns,
)

logger = logging.getLogger(__name__)

RAW_ACLED_DIR = Path("input/raw/acled")
RAW_ECONOMIC_PATH = Path("input/raw/economic/indicators.parquet")
FALLBACK_COUNTRIES = ["UKR", "SYR", "IRQ", "PSE", "RUS"]


# ── Country list ─────────────────────────────────────────────────────────────

def load_country_list() -> list[str]:
    """
    Returns ISO3 country list.

    Priority:
      1. collect/config.py COUNTRIES constant
      2. ISO3 codes inferred from input/raw/acled/*.parquet filenames
      3. FALLBACK_COUNTRIES hardcoded list
    """
    try:
        from collect.config import COUNTRIES
        iso3_list = [c["iso3"] for c in COUNTRIES]
        logger.info(f"Country list loaded from collect/config.py: {len(iso3_list)} countries")
        return iso3_list
    except Exception:
        pass

    if RAW_ACLED_DIR.exists():
        parquet_files = sorted(RAW_ACLED_DIR.glob("*.parquet"))
        iso3_list = [f.stem for f in parquet_files if len(f.stem) == 3]
        if iso3_list:
            logger.info(f"Country list inferred from raw/acled filenames: {len(iso3_list)} countries")
            return iso3_list

    logger.warning(f"Using fallback country list: {FALLBACK_COUNTRIES}")
    return FALLBACK_COUNTRIES


# ── Mock feature table ────────────────────────────────────────────────────────

def build_mock_feature_table(as_of_date: date | None = None) -> pd.DataFrame:
    """
    Builds a deterministic mock feature table for schema/pipeline testing.

    Uses a fixed random seed so the same (country, date) always produces
    the same feature values across pipeline runs.
    """
    target_date = as_of_date or date.today()
    countries = load_country_list()
    feature_cols = get_example_feature_columns()

    rng = np.random.default_rng(seed=42)

    rows = []
    for country in countries:
        # Deterministic seed per (country, date) pair
        country_seed = sum(ord(c) for c in country) + target_date.toordinal()
        country_rng = np.random.default_rng(seed=country_seed % (2**32))

        row = {"country": country, "date": target_date}
        for col in feature_cols:
            row[col] = float(country_rng.uniform(0.0, 1.0))
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        f"Mock feature table built: {len(df)} rows, {len(df.columns)} columns "
        f"(as_of_date={target_date})"
    )
    return df


# ── Real feature builder stubs (TODO) ────────────────────────────────────────

def _build_acled_features(countries: list[str], target_date: date) -> pd.DataFrame:
    """
    TODO: Aggregate ACLED raw events into country-date lag features.

    Steps:
      - Load input/raw/acled/{iso3}.parquet for each country
      - Filter to [target_date - max_lag, target_date]
      - Compute acled_event_count_lag7_Nd, acled_fatalities_lag7_Nd, etc.
      - Add acled_missing_flag=1 for countries with no data
    """
    raise NotImplementedError("_build_acled_features not yet implemented")


def _build_gdelt_features(countries: list[str], target_date: date) -> pd.DataFrame:
    """
    TODO: Aggregate GDELT BQ + DOC 2.0 raw data into country-date features.

    Steps:
      - Load input/raw/gdelt/{iso3}.parquet (BQ historical)
      - Load input/raw/gdelt/{iso3}_doc_vol/tone.parquet (DOC 2.0 recent)
      - Compute GoldsteinScale rolling, AvgTone rolling, volume z-score, etc.
      - Normalize NumArticles/NumMentions to correct for coverage bias
    """
    raise NotImplementedError("_build_gdelt_features not yet implemented")


def _build_economic_features(target_date: date) -> pd.DataFrame:
    """
    TODO: Build economic indicator features with STLFSI4 forward-fill.

    Steps:
      - Load input/raw/economic/indicators.parquet
      - Apply ffill to STLFSI4 (weekly → daily)
      - Compute VIX_7d_change, WTI_7d_change, etc.
      - Return single-row DataFrame for target_date
    """
    raise NotImplementedError("_build_economic_features not yet implemented")


def _build_time_features(countries: list[str], target_date: date) -> pd.DataFrame:
    """
    TODO: Build calendar-based time features (day_of_week, month, week_of_year).
    """
    raise NotImplementedError("_build_time_features not yet implemented")


# ── Main builder ──────────────────────────────────────────────────────────────

def build_daily_features(
    as_of_date: date | None = None,
    output_path: str = FEATURE_TABLE_PATH,
) -> pd.DataFrame:
    """
    Builds and saves the daily feature table.

    Currently returns a mock feature table. Once real feature engineering
    functions are implemented, replace the mock call with actual builders.
    """
    target_date = as_of_date or date.today()
    logger.info(f"Building feature table for {target_date} ...")

    # TODO: Replace with real feature builder when feature list is confirmed:
    #   acled_df   = _build_acled_features(countries, target_date)
    #   gdelt_df   = _build_gdelt_features(countries, target_date)
    #   econ_df    = _build_economic_features(target_date)
    #   time_df    = _build_time_features(countries, target_date)
    #   df = acled_df.merge(gdelt_df, ...).merge(econ_df, ...).merge(time_df, ...)

    df = build_mock_feature_table(as_of_date=target_date)
    return save_features(df, output_path)


def save_features(df: pd.DataFrame, output_path: str = FEATURE_TABLE_PATH) -> pd.DataFrame:
    """Saves feature DataFrame to parquet and returns it."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info(f"Feature table saved: {out} ({len(df)} rows)")
    return df


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Build daily feature table")
    parser.add_argument("--as-of-date", type=str, default=None,
                        help="Target date (YYYY-MM-DD). Defaults to today.")
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of_date) if args.as_of_date else None
    build_daily_features(as_of_date=as_of)
