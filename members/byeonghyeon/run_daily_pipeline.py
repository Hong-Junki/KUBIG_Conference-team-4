"""
Daily pipeline entrypoint.

Default run (no flags) executes the full pipeline without any external API calls:
  1. build_daily_features  — mock feature table
  2. run_inference         — mock predictions (no artifact needed)
  3. run_risk_scoring      — temperature_score + risk_level
  4. export_dashboard_data — CSV + JSON to public/data/

Pass --run-collect to also trigger collect_recent before step 1.
This requires API credentials in .env and is intentionally opt-in.

Docker entrypoint:
  CMD ["python", "run_daily_pipeline.py"]
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import date
from pathlib import Path

# ── Logging setup (before any local imports) ──────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_daily_pipeline")

# ── Local module imports ──────────────────────────────────────────────────────
from features.build_daily_features import build_daily_features
from features.feature_schema import (
    FEATURE_TABLE_PATH,
    MODEL_PREDICTIONS_DIR,
    DASHBOARD_OUTPUT_DIR,
    DASHBOARD_PUBLIC_DIR,
)
from models.predict import run_inference
from scoring.risk_score import run_risk_scoring
from serving.export_dashboard_data import export_dashboard_data


# ── Optional collect step ─────────────────────────────────────────────────────

def step_collect_recent() -> bool:
    """
    Runs collect_recent for ACLED, GDELT DOC 2.0, and economic indicators.

    Requires API credentials in .env. Not called by default.
    """
    try:
        logger.info("[collect] Starting collect_recent ...")
        from collect.acled_collector import collect_recent as acled_recent
        from collect.gdelt_collector import collect_recent_doc as gdelt_recent
        from collect.economic_collector import collect_recent as econ_recent

        acled_recent()
        gdelt_recent()
        econ_recent()
        logger.info("[collect] collect_recent completed.")
        return True
    except Exception:
        logger.error("[collect] collect_recent failed:\n" + traceback.format_exc())
        return False


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_build_features(as_of_date: date | None) -> bool:
    try:
        logger.info("[features] Building daily feature table ...")
        build_daily_features(as_of_date=as_of_date, output_path=FEATURE_TABLE_PATH)
        logger.info(f"[features] Done → {FEATURE_TABLE_PATH}")
        return True
    except Exception:
        logger.error("[features] build_daily_features failed:\n" + traceback.format_exc())
        return False


def step_run_inference(scoring_method: str) -> bool:
    try:
        logger.info("[inference] Running model inference ...")
        parquet_path, csv_path = run_inference(
            feature_path=FEATURE_TABLE_PATH,
            output_dir=MODEL_PREDICTIONS_DIR,
        )
        logger.info(f"[inference] Done → {parquet_path}, {csv_path}")
        return True
    except Exception:
        logger.error("[inference] run_inference failed:\n" + traceback.format_exc())
        return False


def step_run_risk_scoring(scoring_method: str, risk_level_scheme: str) -> bool:
    try:
        logger.info("[scoring] Running risk scoring ...")
        prediction_path = str(Path(MODEL_PREDICTIONS_DIR) / "model_predictions.parquet")
        parquet_path, csv_path = run_risk_scoring(
            prediction_path=prediction_path,
            output_dir=DASHBOARD_OUTPUT_DIR,
            scoring_method=scoring_method,
            risk_level_scheme=risk_level_scheme,
        )
        logger.info(f"[scoring] Done → {parquet_path}, {csv_path}")
        return True
    except Exception:
        logger.error("[scoring] run_risk_scoring failed:\n" + traceback.format_exc())
        return False


def step_export_dashboard() -> bool:
    try:
        logger.info("[export] Exporting dashboard data ...")
        input_path = str(Path(DASHBOARD_OUTPUT_DIR) / "dashboard_country_risk.parquet")
        csv_path, json_path = export_dashboard_data(
            input_path=input_path,
            output_dir=DASHBOARD_PUBLIC_DIR,
        )
        logger.info(f"[export] Done → {csv_path}, {json_path}")
        return True
    except Exception:
        logger.error("[export] export_dashboard_data failed:\n" + traceback.format_exc())
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline(
    as_of_date: date | None = None,
    run_collect: bool = False,
    scoring_method: str = "probability_only",
    risk_level_scheme: str = "three_level",
    stop_on_failure: bool = True,
) -> bool:
    logger.info("=" * 60)
    logger.info("Daily pipeline started")
    if as_of_date:
        logger.info(f"  as_of_date     : {as_of_date}")
    logger.info(f"  run_collect    : {run_collect}")
    logger.info(f"  scoring_method : {scoring_method}")
    logger.info(f"  risk_level     : {risk_level_scheme}")
    logger.info("=" * 60)

    steps = []

    if run_collect:
        steps.append(("collect_recent", lambda: step_collect_recent()))

    steps += [
        ("build_features",  lambda: step_build_features(as_of_date)),
        ("run_inference",   lambda: step_run_inference(scoring_method)),
        ("run_risk_scoring", lambda: step_run_risk_scoring(scoring_method, risk_level_scheme)),
        ("export_dashboard", lambda: step_export_dashboard()),
    ]

    for step_name, step_fn in steps:
        success = step_fn()
        if not success:
            logger.error(f"Pipeline stopped at step: {step_name}")
            if stop_on_failure:
                return False

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully.")
    logger.info("Output files:")
    output_files = [
        Path(FEATURE_TABLE_PATH),
        Path(MODEL_PREDICTIONS_DIR) / "model_predictions.parquet",
        Path(MODEL_PREDICTIONS_DIR) / "model_predictions.csv",
        Path(DASHBOARD_OUTPUT_DIR) / "dashboard_country_risk.parquet",
        Path(DASHBOARD_OUTPUT_DIR) / "dashboard_country_risk.csv",
        Path(DASHBOARD_PUBLIC_DIR) / "dashboard_country_risk.csv",
        Path(DASHBOARD_PUBLIC_DIR) / "dashboard_country_risk.json",
    ]
    for f in output_files:
        status = "OK" if f.exists() else "MISSING"
        logger.info(f"  [{status}] {f}")
    logger.info("=" * 60)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily pipeline: features → inference → scoring → dashboard export"
    )
    parser.add_argument(
        "--as-of-date", type=str, default=None,
        help="Target date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--run-collect", action="store_true", default=False,
        help="Also run collect_recent before building features (requires API keys in .env)",
    )
    parser.add_argument(
        "--scoring-method", default="probability_only",
        choices=["probability_only", "rank_based"],
        help="Risk scoring method (default: probability_only)",
    )
    parser.add_argument(
        "--risk-level-scheme", default="three_level",
        choices=["three_level", "four_level"],
        help="Risk level classification scheme (default: three_level)",
    )
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of_date) if args.as_of_date else None

    success = run_pipeline(
        as_of_date=as_of,
        run_collect=args.run_collect,
        scoring_method=args.scoring_method,
        risk_level_scheme=args.risk_level_scheme,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
