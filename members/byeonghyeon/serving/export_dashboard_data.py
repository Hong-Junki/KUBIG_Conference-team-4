"""
Dashboard data exporter — publishes dashboard_country_risk to public/data/.

MVP: CSV and JSON export only.
DB upsert is a placeholder (NotImplementedError) for future implementation.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_schema import DASHBOARD_PUBLIC_DIR, DASHBOARD_OUTPUT_DIR

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = f"{DASHBOARD_OUTPUT_DIR}/dashboard_country_risk.parquet"
DEFAULT_OUTPUT_DIR = DASHBOARD_PUBLIC_DIR


# ── Load ──────────────────────────────────────────────────────────────────────

def load_dashboard_risk(input_path: str = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Loads dashboard_country_risk from parquet. Falls back to CSV if absent."""
    path = Path(input_path)
    if path.exists() and path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.with_suffix(".csv").exists():
        csv_path = path.with_suffix(".csv")
        logger.warning(f"Parquet not found; loading CSV instead: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Dashboard risk file not found: {path}")

    logger.info(f"Dashboard risk loaded: {len(df)} rows from {path}")
    return df


# ── Export helpers ────────────────────────────────────────────────────────────

def _prepare_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts date/datetime columns to ISO strings so they are
    JSON-serializable without custom encoders.
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif df[col].dtype == object:
            # Coerce date objects to string
            df[col] = df[col].apply(
                lambda v: v.isoformat() if hasattr(v, "isoformat") else v
            )
    return df


def export_csv(df: pd.DataFrame, output_path: str) -> Path:
    """Exports DataFrame to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df = _prepare_for_export(df)
    out_df.to_csv(path, index=False)
    logger.info(f"CSV exported: {path} ({len(df)} rows)")
    return path


def export_json(df: pd.DataFrame, output_path: str) -> Path:
    """Exports DataFrame to JSON (records orientation)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df = _prepare_for_export(df)
    records = out_df.to_dict(orient="records")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON exported: {path} ({len(df)} rows)")
    return path


def upsert_to_database_placeholder(df: pd.DataFrame) -> None:
    """
    TODO: Implement DB upsert when DB schema and connection string are confirmed.

    Planned approach:
      - Load conn_str from environment variable (never hardcode)
      - Upsert on PK: (country, date, model_version)
      - Use SQLAlchemy + pandas to_sql or raw INSERT ON CONFLICT
    """
    raise NotImplementedError(
        "DB upsert is not yet implemented. "
        "Confirm DB table schema and connection string with the team first."
    )


# ── Pipeline step ─────────────────────────────────────────────────────────────

def export_dashboard_data(
    input_path: str = DEFAULT_INPUT_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """
    Loads dashboard_country_risk and exports to CSV + JSON.

    Returns (csv_path, json_path).
    """
    df = load_dashboard_risk(input_path)

    out = Path(output_dir)
    csv_path  = export_csv(df,  str(out / "dashboard_country_risk.csv"))
    json_path = export_json(df, str(out / "dashboard_country_risk.json"))

    return csv_path, json_path


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Export dashboard data")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    csv_path, json_path = export_dashboard_data(
        input_path=args.input_path,
        output_dir=args.output_dir,
    )
    print(f"Exported: {csv_path}")
    print(f"Exported: {json_path}")
