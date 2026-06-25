"""
GDELT 이벤트 증분 수집기.

기존 gdelt_collector.py의 BQ 쿼리 로직을 재사용하고,
결과를 gdelt_processed_events 테이블에 MERGE로 idempotent하게 적재한다.

적재 방식:
  1. gdelt-bq.gdeltv2.events_partitioned → 로컬 DataFrame
  2. DataFrame → staging table 적재
  3. staging → MERGE INTO gdelt_processed_events (GLOBALEVENTID 기반)
  4. staging 삭제
"""

import os
from datetime import date

import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery import SchemaField

from ..collect_gdelt_titles_gkg import GKG_TABLE  # GKG 소스 테이블명 재사용 (gkg_partitioned)
from ..config import COUNTRIES, GDELT_BQ_TABLE
from ..gdelt_collector import (
    _build_query,
    _map_fips_to_iso3,
    _parse_sqldate,
    _run_bq_query,
    dry_run_query,
)
from ..utils import date_range_months, get_logger
from .bigquery_io import (
    drop_table,
    generate_run_id,
    load_dataframe_to_staging,
    merge_into_target,
    validate_after_load,
)
from .state import (
    compute_collection_window,
    require_max_date_from_bq,
    log_state_summary,
)

logger = get_logger(__name__)

TARGET_DATASET = "conflict_ew"
TARGET_TABLE = "gdelt_processed_events"
MERGE_KEY = ["GLOBALEVENTID"]
GDELT_SCHEMA = [
    SchemaField("GLOBALEVENTID", "INTEGER"),
    SchemaField("SQLDATE", "INTEGER"),
    SchemaField("ActionGeo_CountryCode", "STRING"),
    SchemaField("EventCode", "STRING"),
    SchemaField("EventRootCode", "STRING"),
    SchemaField("QuadClass", "INTEGER"),
    SchemaField("GoldsteinScale", "FLOAT"),
    SchemaField("NumMentions", "FLOAT"),
    SchemaField("NumArticles", "FLOAT"),
    SchemaField("AvgTone", "FLOAT"),
    SchemaField("event_date", "TIMESTAMP"),
    SchemaField("iso3", "STRING"),
]
ALL_COLUMNS = [f.name for f in GDELT_SCHEMA]


def run_gdelt_incremental(
    project_id: str,
    overlap_days: int | None = None,
    forced_start: date | None = None,
    forced_end: date | None = None,
    dry_run: bool = False,
    max_gb_per_query: float = 100.0,
    run_id: str | None = None,
) -> dict:
    """
    GDELT 이벤트 증분 수집 + BQ 적재.

    Args:
        project_id: GCP 프로젝트 ID (billing)
        overlap_days: overlap 일수. None이면 기본값(3일) 사용.
        dry_run: True이면 스캔량 확인 후 실제 적재 안 함.

    Returns:
        결과 요약 dict.
    """
    run_id = run_id or generate_run_id()
    client = bigquery.Client(project=project_id)
    target_fqn = f"{project_id}.{TARGET_DATASET}.{TARGET_TABLE}"

    # 1. 수집 기간 결정 — MAX(date) 조회 실패 시 즉시 RuntimeError (fallback 없음)
    last_date = require_max_date_from_bq(
        client, target_fqn, "event_date",
        partition_col="event_date",
    )
    try:
        start, end = compute_collection_window(
            last_date, "gdelt", overlap_days, forced_start, forced_end
        )
    except ValueError as e:
        logger.warning(str(e))
        return {"source": "gdelt", "run_id": run_id, "skipped": True, "reason": str(e)}

    log_state_summary("gdelt", last_date, start, end)

    # 2. FIPS 코드 수집
    fips_codes: list[str] = []
    for c in COUNTRIES:
        codes = c["gdelt"] if isinstance(c["gdelt"], list) else [c["gdelt"]]
        fips_codes.extend(codes)

    months = date_range_months(start, end)
    total_scanned_gb = 0.0
    frames: list[pd.DataFrame] = []

    for idx, (m_start, m_end) in enumerate(months, 1):
        query = _build_query(fips_codes, m_start, m_end)

        # dry run 비용 확인
        gb = dry_run_query(query)
        total_scanned_gb += gb
        logger.info(f"  [{idx}/{len(months)}] {m_start:%Y-%m}: 예상 {gb:.2f} GB")

        if gb > max_gb_per_query:
            logger.error(f"  스캔 한도 초과 ({gb:.1f} GB > {max_gb_per_query} GB). 건너뜀.")
            continue

        if dry_run:
            continue

        df = _run_bq_query(query)
        df = _parse_sqldate(df)
        df = _map_fips_to_iso3(df)
        df = df.dropna(subset=["iso3"])

        if not df.empty:
            frames.append(df)

    logger.info(f"  총 예상 스캔: {total_scanned_gb:.2f} GB")

    if dry_run:
        return {
            "source": "gdelt",
            "run_id": run_id,
            "dry_run": True,
            "start": str(start),
            "end": str(end),
            "estimated_gb": round(total_scanned_gb, 2),
        }

    if not frames:
        logger.warning("  수집된 데이터 없음 (원천에 신규 데이터 없음)")
        return {
            "source": "gdelt",
            "run_id": run_id,
            "start": str(start),
            "end": str(end),
            "new_rows": 0,
            "passed": True,
        }

    # 3. 데이터 준비
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["GLOBALEVENTID"])
    df_all = df_all[[c for c in ALL_COLUMNS if c in df_all.columns]]

    # QuadClass를 Int64로 (NaN-safe)
    if "QuadClass" in df_all.columns:
        df_all["QuadClass"] = pd.to_numeric(df_all["QuadClass"], errors="coerce")

    logger.info(f"  수집 완료: {len(df_all):,}행 (dedup 후)")

    # 4. staging → MERGE
    staging_fqn = load_dataframe_to_staging(
        client, df_all, project_id, TARGET_DATASET,
        TARGET_TABLE, run_id, schema=GDELT_SCHEMA,
    )
    try:
        affected = merge_into_target(
            client,
            staging_fqn=staging_fqn,
            target_fqn=target_fqn,
            merge_keys=MERGE_KEY,
            columns=ALL_COLUMNS,
            update_on_match=False,  # GDELT 이벤트는 삽입만
        )
    finally:
        drop_table(client, staging_fqn)

    # 5. 검증
    partition_filter = (
        f"event_date >= TIMESTAMP('{start - __import__('datetime').timedelta(days=1)}')"
    )
    validation = validate_after_load(
        client, target_fqn, "event_date", "iso3",
        expected_min_date=start,
        expected_max_date=end,
        run_id=run_id,
        partition_filter=partition_filter,
    )

    return {
        "source": "gdelt",
        "run_id": run_id,
        "start": str(start),
        "end": str(end),
        "rows_collected": len(df_all),
        "rows_merged": affected,
        "estimated_gb": round(total_scanned_gb, 2),
        "validation": validation,
        "passed": validation.get("passed", False),
    }
