"""
BigQuery 기반 수집 상태 관리.

GitHub Actions runner는 매번 초기화되므로 로컬 checkpoint를 운영 상태로 사용하지 않는다.
수집 시작일은 항상 BigQuery target table의 MAX(date)에서 계산한다.

중요: MAX(date) 조회 실패 시 임의 날짜로 fallback하지 않는다.
      실제 적재 모드에서는 즉시 실패 처리한다.
"""

from datetime import date, timedelta

from google.cloud import bigquery

from ..utils import get_logger

logger = get_logger(__name__)

# 증분 수집 기본 overlap (각 소스의 데이터 지연·수정치 반영을 위해)
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
    수집 시작/종료일 계산.

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
