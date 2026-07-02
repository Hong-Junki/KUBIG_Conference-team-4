"""
BigQuery 기반 수집 상태 관리.

GitHub Actions runner는 매번 초기화되므로 로컬 checkpoint를 운영 상태로 사용하지 않는다.

[날짜 단위 모드 (date mode)]
  수집 시작일은 항상 BigQuery target table의 MAX(date)에서 계산한다.
  economic, 수동 backfill, 7/30일 reconciliation에서 사용.

[timestamp 모드 (15분 증분)]
  pipeline_watermarks 테이블의 last_success_at을 기준으로 window를 계산한다.
  gdelt (DATEADDED 기반), gdelt_titles (GKG DATE 기반)에서 사용.
  성공 시에만 watermark를 갱신한다.

중요: MAX(date) 조회 실패 시 임의 날짜로 fallback하지 않는다.
      실제 적재 모드에서는 즉시 실패 처리한다.
"""

from datetime import date, datetime, timedelta, timezone

from google.cloud import bigquery

from ..utils import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# 날짜 단위 수집 상태 (기존 로직)
# ──────────────────────────────────────────────

DEFAULT_OVERLAP = {
    "gdelt": 3,
    "economic": 10,
    "gdelt_titles": 3,
}


def get_max_date_from_bq(
    client: bigquery.Client,
    table_fqn: str,
    date_col: str,
    partition_col: str | None = None,
) -> date | None:
    """
    BQ 테이블의 MAX(date_col)를 조회한다.

    실패 시 None을 반환하고 상위 레이어에서 처리한다.
    실제 적재 모드에서는 None 반환 시 RuntimeError를 발생시켜야 한다.

    Args:
        table_fqn: 완전한 테이블 경로 (project.dataset.table)
        date_col: 날짜 컬럼명 (DATE 또는 TIMESTAMP)
        partition_col: 파티션 컬럼명. 지정 시 마지막 파티션 범위로 스캔 비용 절감.

    Returns:
        date 또는 None (BQ 접근 불가)
    """
    try:
        if partition_col:
            partition_max = _get_max_partition_date(client, table_fqn)
            if partition_max is not None:
                filter_start = partition_max - timedelta(days=60)
                query = f"""
                SELECT MAX({date_col}) AS max_dt
                FROM `{table_fqn}`
                WHERE {partition_col} >= TIMESTAMP('{filter_start}')
                """
            else:
                query = f"SELECT MAX({date_col}) AS max_dt FROM `{table_fqn}`"
        else:
            query = f"SELECT MAX({date_col}) AS max_dt FROM `{table_fqn}`"

        rows = list(client.query(query))
        if not rows or rows[0]["max_dt"] is None:
            logger.warning(f"  {table_fqn}: MAX({date_col}) is NULL (빈 테이블?)")
            return None

        raw = rows[0]["max_dt"]
        if hasattr(raw, "date"):
            return raw.date()
        return date.fromisoformat(str(raw)[:10])

    except Exception as e:
        logger.error(f"  {table_fqn} MAX({date_col}) 조회 실패: {e}")
        return None


def _get_max_partition_date(client: bigquery.Client, table_fqn: str) -> date | None:
    """INFORMATION_SCHEMA.PARTITIONS에서 마지막 파티션 날짜 조회."""
    parts = table_fqn.split(".")
    if len(parts) != 3:
        return None
    project, dataset, table = parts
    query = f"""
    SELECT MAX(partition_id) AS max_pid
    FROM `{project}.{dataset}.INFORMATION_SCHEMA.PARTITIONS`
    WHERE table_name = '{table}'
      AND partition_id NOT IN ('__NULL__', '__UNPARTITIONED__')
    """
    try:
        rows = list(client.query(query))
        if not rows or rows[0]["max_pid"] is None:
            return None
        pid = str(rows[0]["max_pid"])  # e.g. "202603"
        if len(pid) == 6:
            return date(int(pid[:4]), int(pid[4:6]), 1)
        return None
    except Exception:
        return None


def require_max_date_from_bq(
    client: bigquery.Client,
    table_fqn: str,
    date_col: str,
    partition_col: str | None = None,
) -> date:
    """
    MAX(date) 조회 실패 시 RuntimeError를 발생시킨다.
    실제 적재 파이프라인에서 사용한다. fallback 없음.
    """
    result = get_max_date_from_bq(client, table_fqn, date_col, partition_col)
    if result is None:
        raise RuntimeError(
            f"{table_fqn} MAX({date_col}) 조회 실패. "
            "BQ 접근 권한·인증을 확인하세요. "
            "임의 날짜 fallback을 허용하지 않습니다."
        )
    return result


def compute_collection_window(
    last_date: date,
    source: str,
    overlap_days: int | None = None,
    forced_start: date | None = None,
    forced_end: date | None = None,
) -> tuple[date, date]:
    """
    날짜 단위 수집 시작/종료일 계산.

    Args:
        last_date: BQ에서 조회한 MAX(date). 반드시 유효한 date여야 한다.
        source: 소스 키 ('gdelt', 'economic', 'gdelt_titles')
        overlap_days: overlap 일수. None이면 DEFAULT_OVERLAP 사용.
        forced_start: CLI로 강제 지정한 시작일
        forced_end: CLI로 강제 지정한 종료일

    Returns:
        (start, end) 튜플
    """
    today = date.today()
    end = forced_end or (today - timedelta(days=1))

    if forced_start:
        if forced_start > end:
            raise ValueError(
                f"{source}: forced_start({forced_start}) > end({end}). 수집 기간이 없습니다."
            )
        return forced_start, end

    overlap = overlap_days if overlap_days is not None else DEFAULT_OVERLAP.get(source, 7)
    start = last_date - timedelta(days=overlap)

    if start > end:
        raise ValueError(f"{source}: start({start}) > end({end}). 수집 기간이 없습니다.")

    return start, end


def log_state_summary(source: str, last_date: date, start: date, end: date) -> None:
    logger.info(f"[{source}] 상태 요약:")
    logger.info(f"  BQ MAX(date)  : {last_date}")
    logger.info(f"  수집 시작일   : {start}")
    logger.info(f"  수집 종료일   : {end}")
    logger.info(f"  수집 기간     : {(end - start).days + 1}일")


# ──────────────────────────────────────────────
# Timestamp watermark (15분 증분 수집)
# ──────────────────────────────────────────────

WATERMARK_TABLE = "conflict_ew.pipeline_watermarks"
WATERMARK_SOURCES = frozenset({"gdelt", "gdelt_titles"})

# 최초 실행 시 watermark 없을 때의 fallback 수집 window (시간)
_FIRST_RUN_FALLBACK_HOURS = 2


def get_watermark(
    client: bigquery.Client,
    project_id: str,
    source: str,
) -> datetime | None:
    """
    pipeline_watermarks 테이블에서 source의 last_success_at 조회.

    Returns:
        UTC-aware datetime, 또는 None (최초 실행 / 행 없음)

    Raises:
        RuntimeError: 테이블이 존재하지 않을 때 (migration 미적용)
        Exception: 그 외 BQ 오류
    """
    table_fqn = f"{project_id}.{WATERMARK_TABLE}"
    query = f"""
    SELECT last_success_at
    FROM `{table_fqn}`
    WHERE source = '{source}'
    ORDER BY updated_at DESC
    LIMIT 1
    """
    try:
        rows = list(client.query(query))
        if not rows or rows[0]["last_success_at"] is None:
            logger.info(f"[{source}] watermark 없음 (최초 실행 또는 행 없음)")
            return None
        ts = rows[0]["last_success_at"]
        # BigQuery TIMESTAMP → Python datetime (timezone-aware 보장)
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        elif not hasattr(ts, "tzinfo"):
            ts = datetime.fromisoformat(str(ts)).replace(tzinfo=timezone.utc)
        logger.info(f"[{source}] watermark: {ts.isoformat()}")
        return ts
    except Exception as e:
        err_str = str(e)
        if "Not found" in err_str or "notFound" in err_str or "404" in err_str:
            raise RuntimeError(
                f"pipeline_watermarks 테이블이 없습니다: {table_fqn}\n"
                "먼저 아래 명령으로 migration을 적용하세요:\n"
                f"  bq query --use_legacy_sql=false --project_id={project_id} \\\n"
                f'    "$(sed \'s/{{PROJECT_ID}}/{project_id}/g\' migrations/create_pipeline_watermarks.sql)"'
            ) from e
        logger.error(f"[{source}] watermark 조회 실패: {e}")
        raise


def set_watermark(
    client: bigquery.Client,
    project_id: str,
    source: str,
    ts: datetime,
    run_id: str,
) -> None:
    """
    pipeline_watermarks 테이블에 watermark를 MERGE로 갱신한다.

    성공한 수집 직후에만 호출해야 한다.
    실패 시에는 호출하지 않아 기존 watermark를 유지한다.

    Args:
        ts: 수집 end_timestamp (UTC). 다음 실행의 start 기준이 된다.
        run_id: 수집 실행 ID
    """
    table_fqn = f"{project_id}.{WATERMARK_TABLE}"
    # SQL 인젝션 방지: 알파뉴메릭, 하이픈, 언더스코어만 허용
    import re as _re
    _safe_source = _re.sub(r"[^a-zA-Z0-9_\-]", "", source)[:64]
    _safe_run_id = _re.sub(r"[^a-zA-Z0-9_\-]", "", str(run_id))[:64]

    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    query = f"""
    MERGE `{table_fqn}` T
    USING (SELECT '{_safe_source}' AS source) S
    ON T.source = S.source
    WHEN MATCHED THEN
      UPDATE SET
        last_success_at = TIMESTAMP '{ts_str}',
        updated_at      = TIMESTAMP '{now_str}',
        run_id          = '{_safe_run_id}'
    WHEN NOT MATCHED THEN
      INSERT (source, last_success_at, updated_at, run_id)
      VALUES ('{_safe_source}', TIMESTAMP '{ts_str}', TIMESTAMP '{now_str}', '{_safe_run_id}')
    """
    try:
        job = client.query(query)
        job.result()
        logger.info(f"[{source}] watermark 갱신 완료: {ts_str} (run_id={_safe_run_id})")
    except Exception as e:
        logger.error(f"[{source}] watermark 갱신 실패: {e}")
        raise


def get_watermark_with_fallback(
    client: bigquery.Client,
    project_id: str,
    source: str,
    target_fqn: str,
    target_date_col: str,
) -> datetime | None:
    """
    watermark 조회 후 없으면 target 테이블 MAX(date)로 fallback.

    최초 실행(watermark 없음 + target 데이터 있음) 시 기존 데이터 이후부터 수집한다.

    Returns:
        UTC-aware datetime, 또는 None (완전 첫 실행)
    """
    ts = get_watermark(client, project_id, source)
    if ts is not None:
        return ts

    # fallback: target 테이블 MAX(date/event_date)를 timestamp로 변환
    logger.info(f"[{source}] watermark 없음 → target MAX({target_date_col}) fallback 시도")
    max_date = get_max_date_from_bq(client, target_fqn, target_date_col)
    if max_date is not None:
        # date → UTC datetime (자정 기준)
        fallback_ts = datetime(
            max_date.year, max_date.month, max_date.day,
            tzinfo=timezone.utc,
        )
        logger.info(f"[{source}] target MAX → fallback watermark: {fallback_ts.isoformat()}")
        return fallback_ts

    logger.info(f"[{source}] target도 비어 있음 → {_FIRST_RUN_FALLBACK_HOURS}시간 window 사용")
    return None


def compute_timestamp_window(
    last_success_at: datetime | None,
    overlap_minutes: int = 30,
) -> tuple[datetime, datetime]:
    """
    timestamp 기반 15분 증분 수집 window 계산.

    Args:
        last_success_at: 마지막 성공 수집의 end_timestamp (UTC).
                         None이면 현재 - FIRST_RUN_FALLBACK_HOURS.
        overlap_minutes: start를 뒤로 당기는 분. 기본 30분.
                         누락 방지를 위해 last_success_at - overlap에서 재수집.

    Returns:
        (start_ts, end_ts) UTC-aware datetime tuple
    """
    end_ts = datetime.now(timezone.utc)

    if last_success_at is None:
        start_ts = end_ts - timedelta(hours=_FIRST_RUN_FALLBACK_HOURS)
    else:
        if last_success_at.tzinfo is None:
            last_success_at = last_success_at.replace(tzinfo=timezone.utc)
        start_ts = last_success_at - timedelta(minutes=overlap_minutes)

    return start_ts, end_ts


def dateadded_int(dt: datetime) -> int:
    """
    datetime → 14자리 정수 YYYYMMDDHHMMSS.
    GDELT DATEADDED / GKG DATE 컬럼 비교에 사용.

    Args:
        dt: UTC-aware datetime (naive도 허용, UTC로 간주)

    Returns:
        예: datetime(2026,7,2,12,45,0) → 20260702124500
    """
    return int(dt.strftime("%Y%m%d%H%M%S"))


def log_timestamp_window(
    source: str,
    watermark_before: datetime | None,
    start_ts: datetime,
    end_ts: datetime,
    overlap_minutes: int,
) -> None:
    """timestamp 모드 수집 window 로그."""
    logger.info(f"[{source}] timestamp window:")
    logger.info(f"  watermark_before : {watermark_before.isoformat() if watermark_before else 'None (최초)'}")
    logger.info(f"  overlap          : {overlap_minutes}분")
    logger.info(f"  start_timestamp  : {start_ts.isoformat()}")
    logger.info(f"  end_timestamp    : {end_ts.isoformat()}")
    logger.info(f"  window           : {(end_ts - start_ts).total_seconds() / 60:.1f}분")
