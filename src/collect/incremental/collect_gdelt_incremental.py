"""
GDELT 이벤트 증분 수집기.

두 가지 수집 모드:

[date 모드 (기존, backfill / reconciliation)]
  - start_date, end_date 기준 SQLDATE 범위 수집
  - BQ MAX(event_date) - overlap_days 자동 계산
  - 일 단위 reconciliation, 수동 backfill에 사용

[timestamp 모드 (15분 증분)]
  - DATEADDED 기준 timestamp window 수집
  - start_timestamp, end_timestamp 파라미터 필요
  - pipeline_watermarks watermark 갱신은 호출자(run_incremental)가 담당
  - DATEADDED: gdelt-bq.gdeltv2.events_partitioned의 14자리 YYYYMMDDHHMMSS 컬럼
              (NULL 0%, 15분 배치 공개 시각, cross-day lag: 99.5% 당일)

적재 방식 (공통):
  1. gdelt-bq.gdeltv2.events_partitioned → 로컬 DataFrame
  2. DataFrame → staging table 적재
  3. staging → MERGE INTO gdelt_processed_events (GLOBALEVENTID 기반)
  4. staging 삭제
"""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery import SchemaField

from ..collect_gdelt_titles_gkg import GKG_TABLE  # GKG 소스 테이블명 재사용
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
    dry_run_bytes,
    generate_run_id,
    load_dataframe_to_staging,
    merge_into_target,
    validate_after_load,
)
from .state import (
    compute_collection_window,
    dateadded_int,
    log_state_summary,
    require_max_date_from_bq,
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

# DATEADDED 기반 파티션 pruning 여유 (하루 앞 파티션 포함)
# cross-day lag: 99.5% 당일, 0.2% 다음날 → 하루 이전 파티션으로 충분
_DATEADDED_PARTITION_LOOKBACK_DAYS = 1


def _build_dateadded_query(
    fips_codes: list[str],
    start_ts: datetime,
    end_ts: datetime,
) -> str:
    """
    DATEADDED 기반 15분 증분 수집 쿼리.

    파티션 범위: start_ts 하루 전 파티션 ~ end_ts 당일 파티션
      - cross-day lag 0.2% (다음날 publish 소량) 대응
      - DATEADDED 자체 필터로 실제 window 정밀 제어

    Args:
        fips_codes: GDELT FIPS 국가 코드 목록
        start_ts: 수집 시작 UTC datetime (watermark - overlap)
        end_ts: 수집 종료 UTC datetime (현재 UTC)

    Returns:
        BQ SQL 쿼리 문자열
    """
    partition_start = (start_ts.date() - timedelta(days=_DATEADDED_PARTITION_LOOKBACK_DAYS))
    partition_end = end_ts.date()
    start_int = dateadded_int(start_ts)
    end_int = dateadded_int(end_ts)
    fips_str = ", ".join(f"'{c}'" for c in fips_codes)

    # BQ_SELECT_FIELDS 재사용 (DATEADDED는 WHERE 절만, SELECT에서 제외)
    from ..gdelt_collector import BQ_SELECT_FIELDS
    fields = ", ".join(BQ_SELECT_FIELDS)

    return f"""
    SELECT {fields}
    FROM `{GDELT_BQ_TABLE}`
    WHERE _PARTITIONTIME BETWEEN TIMESTAMP('{partition_start}') AND TIMESTAMP('{partition_end}')
      AND DATEADDED BETWEEN {start_int} AND {end_int}
      AND ActionGeo_CountryCode IN ({fips_str})
    """


def run_gdelt_incremental(
    project_id: str,
    overlap_days: int | None = None,
    forced_start: date | None = None,
    forced_end: date | None = None,
    dry_run: bool = False,
    max_gb_per_query: float = 100.0,
    run_id: str | None = None,
    # timestamp 모드 전용
    start_timestamp: datetime | None = None,
    end_timestamp: datetime | None = None,
    overlap_minutes: int = 30,
) -> dict:
    """
    GDELT 이벤트 증분 수집 + BQ 적재.

    timestamp 모드: start_timestamp, end_timestamp 둘 다 지정.
    date 모드: 기존 방식 (forced_start/forced_end 또는 BQ MAX(date) - overlap).

    Args:
        start_timestamp: timestamp 모드 시작 (UTC datetime). None이면 date 모드.
        end_timestamp: timestamp 모드 종료 (UTC datetime).
        overlap_minutes: timestamp 모드 overlap (로그용, 실제 계산은 호출자가 처리).

    Returns:
        결과 요약 dict. 'passed' 키로 성공 여부 확인.
    """
    run_id = run_id or generate_run_id()
    client = bigquery.Client(project=project_id)
    target_fqn = f"{project_id}.{TARGET_DATASET}.{TARGET_TABLE}"

    is_timestamp_mode = start_timestamp is not None and end_timestamp is not None

    if is_timestamp_mode:
        return _run_timestamp_mode(
            client=client,
            project_id=project_id,
            target_fqn=target_fqn,
            start_ts=start_timestamp,
            end_ts=end_timestamp,
            overlap_minutes=overlap_minutes,
            dry_run=dry_run,
            max_gb_per_query=max_gb_per_query,
            run_id=run_id,
        )
    else:
        return _run_date_mode(
            client=client,
            project_id=project_id,
            target_fqn=target_fqn,
            overlap_days=overlap_days,
            forced_start=forced_start,
            forced_end=forced_end,
            dry_run=dry_run,
            max_gb_per_query=max_gb_per_query,
            run_id=run_id,
        )


def _run_date_mode(
    client: bigquery.Client,
    project_id: str,
    target_fqn: str,
    overlap_days: int | None,
    forced_start: date | None,
    forced_end: date | None,
    dry_run: bool,
    max_gb_per_query: float,
    run_id: str,
) -> dict:
    """날짜 단위 수집 (기존 로직 보존)."""
    # 수집 기간 결정
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

    fips_codes = _get_fips_codes()
    months = date_range_months(start, end)
    total_scanned_gb = 0.0
    frames: list[pd.DataFrame] = []

    for idx, (m_start, m_end) in enumerate(months, 1):
        query = _build_query(fips_codes, m_start, m_end)
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
            "passed": True,
            "mode": "date",
            "start": str(start),
            "end": str(end),
            "last_bq_date": str(last_date),
            "estimated_gb": round(total_scanned_gb, 2),
        }

    return _merge_and_validate(
        client=client,
        project_id=project_id,
        target_fqn=target_fqn,
        frames=frames,
        run_id=run_id,
        extra={
            "mode": "date",
            "start": str(start),
            "end": str(end),
            "last_bq_date": str(last_date),
            "estimated_gb": round(total_scanned_gb, 2),
        },
        partition_filter=(
            f"event_date >= TIMESTAMP('{start - timedelta(days=1)}')"
        ),
        expected_min_date=start,
        expected_max_date=end,
    )


def _run_timestamp_mode(
    client: bigquery.Client,
    project_id: str,
    target_fqn: str,
    start_ts: datetime,
    end_ts: datetime,
    overlap_minutes: int,
    dry_run: bool,
    max_gb_per_query: float,
    run_id: str,
) -> dict:
    """
    DATEADDED 기반 15분 증분 수집.

    DATEADDED 필터로 실제 15분 window를 정밀 제한하고,
    _PARTITIONTIME으로 partition pruning을 추가 적용한다.
    MERGE key (GLOBALEVENTID)로 중복 없이 idempotent 적재.
    """
    fips_codes = _get_fips_codes()
    query = _build_dateadded_query(fips_codes, start_ts, end_ts)

    # dry-run 스캔량 확인
    try:
        scanned_bytes = dry_run_bytes(client, query)
        scanned_gb = scanned_bytes / 1e9
    except Exception as e:
        logger.warning(f"  DATEADDED 쿼리 dry-run 실패: {e}")
        scanned_gb = 0.0

    logger.info(f"  [timestamp] DATEADDED window 예상 스캔: {scanned_gb:.3f} GB")

    if scanned_gb > max_gb_per_query:
        msg = f"스캔 한도 초과 ({scanned_gb:.1f} GB > {max_gb_per_query} GB)"
        logger.error(f"  {msg}")
        return {"source": "gdelt", "run_id": run_id, "passed": False, "error": msg}

    if dry_run:
        return {
            "source": "gdelt",
            "run_id": run_id,
            "dry_run": True,
            "passed": True,
            "mode": "timestamp",
            "start_timestamp": start_ts.isoformat(),
            "end_timestamp": end_ts.isoformat(),
            "estimated_gb": round(scanned_gb, 3),
        }

    # 실제 수집
    df = _run_bq_query(query)
    df = _parse_sqldate(df)
    df = _map_fips_to_iso3(df)
    df = df.dropna(subset=["iso3"])

    logger.info(f"  [timestamp] 원천 수집: {len(df):,}행")

    expected_min_date = start_ts.date() - timedelta(days=_DATEADDED_PARTITION_LOOKBACK_DAYS)
    expected_max_date = end_ts.date()

    return _merge_and_validate(
        client=client,
        project_id=project_id,
        target_fqn=target_fqn,
        frames=[df] if not df.empty else [],
        run_id=run_id,
        extra={
            "mode": "timestamp",
            "start_timestamp": start_ts.isoformat(),
            "end_timestamp": end_ts.isoformat(),
            "estimated_gb": round(scanned_gb, 3),
        },
        partition_filter=(
            f"event_date >= TIMESTAMP('{expected_min_date}')"
        ),
        expected_min_date=expected_min_date,
        expected_max_date=expected_max_date,
    )


def _get_fips_codes() -> list[str]:
    """설정에서 GDELT FIPS 코드 목록 반환."""
    fips_codes: list[str] = []
    for c in COUNTRIES:
        codes = c["gdelt"] if isinstance(c["gdelt"], list) else [c["gdelt"]]
        fips_codes.extend(codes)
    return fips_codes


def _merge_and_validate(
    client: bigquery.Client,
    project_id: str,
    target_fqn: str,
    frames: list[pd.DataFrame],
    run_id: str,
    extra: dict,
    partition_filter: str,
    expected_min_date: date,
    expected_max_date: date,
) -> dict:
    """데이터 준비 → staging → MERGE → validation."""
    if not frames:
        logger.info("  수집된 데이터 없음 (원천에 신규 데이터 없음)")
        return {
            "source": "gdelt",
            "run_id": run_id,
            "new_rows": 0,
            "rows_merged": 0,
            "passed": True,
            **extra,
        }

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["GLOBALEVENTID"])
    df_all = df_all[[c for c in ALL_COLUMNS if c in df_all.columns]]

    if "QuadClass" in df_all.columns:
        df_all["QuadClass"] = pd.to_numeric(df_all["QuadClass"], errors="coerce")

    logger.info(f"  수집 완료: {len(df_all):,}행 (dedup 후)")

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

    validation = validate_after_load(
        client, target_fqn, "event_date", "iso3",
        expected_min_date=expected_min_date,
        expected_max_date=expected_max_date,
        run_id=run_id,
        partition_filter=partition_filter,
    )

    return {
        "source": "gdelt",
        "run_id": run_id,
        "rows_collected": len(df_all),
        "rows_merged": affected,
        "validation": validation,
        "passed": validation.get("passed", False),
        **extra,
    }
