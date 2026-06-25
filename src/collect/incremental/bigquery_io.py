"""
BigQuery 적재 유틸: staging table → MERGE → cleanup 패턴.

설계 원칙:
  1. 임시 staging table에 적재
  2. row count / schema 검증
  3. target table에 MERGE (idempotent)
  4. staging table 정리

DELETE+INSERT 방식은 INSERT 실패 시 데이터 공백이 생길 수 있으므로 사용하지 않는다.
"""

import uuid
from datetime import date, datetime
from typing import Any

import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery import SchemaField

from ..utils import get_logger, retry

logger = get_logger(__name__)

_MAX_BYTES_BILLED_DEFAULT = 200 * 1024**3  # 200 GB


def get_bq_client(project_id: str) -> bigquery.Client:
    return bigquery.Client(project=project_id)


def dry_run_bytes(client: bigquery.Client, query: str) -> int:
    """쿼리 스캔량(bytes) 반환. 실행하지 않음."""
    cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(query, job_config=cfg)
    return job.total_bytes_processed


# ──────────────────────────────────────────────
# staging 생성/삭제
# ──────────────────────────────────────────────

def _staging_table_id(dataset: str, base_name: str, run_id: str) -> str:
    return f"{dataset}._staging_{base_name}_{run_id.replace('-', '_')}"


def load_dataframe_to_staging(
    client: bigquery.Client,
    df: pd.DataFrame,
    project_id: str,
    dataset: str,
    base_name: str,
    run_id: str,
    schema: list[SchemaField] | None = None,
) -> str:
    """
    DataFrame을 staging table에 적재하고 완전한 테이블 ID를 반환한다.

    Args:
        schema: None이면 자동 감지 (권장: 명시 지정)
    Returns:
        staging table의 완전한 FQN (project.dataset._staging_...)
    """
    staging_id = _staging_table_id(dataset, base_name, run_id)
    table_fqn = f"{project_id}.{staging_id}"

    cfg = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        create_disposition="CREATE_IF_NEEDED",
        schema=schema,
    )
    job = client.load_table_from_dataframe(df, table_fqn, job_config=cfg)
    job.result()
    logger.info(f"  staging 적재 완료: {table_fqn} ({len(df):,}행)")
    return table_fqn


def drop_table(client: bigquery.Client, table_fqn: str) -> None:
    """테이블 삭제. 존재하지 않으면 무시."""
    try:
        client.delete_table(table_fqn, not_found_ok=True)
        logger.info(f"  staging 삭제: {table_fqn}")
    except Exception as e:
        logger.warning(f"  staging 삭제 실패 (무시): {table_fqn} — {e}")


# ──────────────────────────────────────────────
# MERGE
# ──────────────────────────────────────────────

def merge_into_target(
    client: bigquery.Client,
    staging_fqn: str,
    target_fqn: str,
    merge_keys: list[str],
    columns: list[str],
    update_on_match: bool = False,
    max_bytes_billed: int = _MAX_BYTES_BILLED_DEFAULT,
) -> int:
    """
    staging → target MERGE.

    Args:
        merge_keys: MERGE ON 조건에 사용할 컬럼 목록
        columns: INSERT 컬럼 목록
        update_on_match: True이면 MATCHED 시 UPDATE, False이면 NOT MATCHED만 INSERT
        max_bytes_billed: 비용 안전장치

    Returns:
        변경된 row 수 (삽입 + 수정)
    """
    on_clause = " AND ".join(f"T.{k} = S.{k}" for k in merge_keys)
    col_list = ", ".join(columns)
    s_col_list = ", ".join(f"S.{c}" for c in columns)

    if update_on_match:
        update_set = ", ".join(
            f"T.{c} = S.{c}" for c in columns if c not in merge_keys
        )
        when_matched = f"WHEN MATCHED THEN UPDATE SET {update_set}"
    else:
        when_matched = ""

    query = f"""
    MERGE `{target_fqn}` T
    USING `{staging_fqn}` S
    ON {on_clause}
    {when_matched}
    WHEN NOT MATCHED THEN
      INSERT ({col_list})
      VALUES ({s_col_list})
    """

    cfg = bigquery.QueryJobConfig(
        maximum_bytes_billed=max_bytes_billed,
        use_query_cache=False,
    )
    job = client.query(query, job_config=cfg)
    job.result()
    affected = job.num_dml_affected_rows or 0
    logger.info(f"  MERGE 완료: {affected:,}행 변경 → {target_fqn}")
    return affected


# ──────────────────────────────────────────────
# validation
# ──────────────────────────────────────────────

def validate_after_load(
    client: bigquery.Client,
    table_fqn: str,
    date_col: str,
    country_col: str | None,
    expected_min_date: date,
    expected_max_date: date,
    run_id: str,
    partition_filter: str | None = None,
) -> dict[str, Any]:
    """
    적재 후 자동 검증.

    Returns:
        결과 dict. "passed" 키가 False이면 workflow 실패 처리.
    """
    issues: list[str] = []

    where = f"WHERE {partition_filter}" if partition_filter else ""
    q = f"""
    SELECT
      COUNT(*) AS total_rows,
      MIN({date_col}) AS min_date,
      MAX({date_col}) AS max_date,
      {f'COUNT(DISTINCT {country_col}) AS n_countries,' if country_col else ''}
      COUNTIF({date_col} IS NULL) AS null_date_rows,
      COUNTIF(CAST({date_col} AS DATE) > CURRENT_DATE()) AS future_date_rows
    FROM `{table_fqn}`
    {where}
    """
    try:
        rows = list(client.query(q))
        r = rows[0]
        result: dict[str, Any] = {
            "table": table_fqn,
            "run_id": run_id,
            "checked_at": datetime.utcnow().isoformat(),
            "total_rows": int(r["total_rows"]),
            "min_date": str(r["min_date"]),
            "max_date": str(r["max_date"]),
            "null_date_rows": int(r["null_date_rows"]),
            "future_date_rows": int(r["future_date_rows"]),
        }
        if country_col and "n_countries" in r.keys():
            result["n_countries"] = int(r["n_countries"])

        # 검증 규칙
        if result["null_date_rows"] > 0:
            issues.append(f"null date rows: {result['null_date_rows']}")
        if result["future_date_rows"] > 0:
            issues.append(f"future date rows: {result['future_date_rows']}")
        if result["total_rows"] == 0:
            issues.append("적재 row 0건")

        max_loaded = date.fromisoformat(str(r["max_date"])[:10]) if r["max_date"] else None
        if max_loaded and max_loaded < expected_min_date:
            issues.append(f"MAX(date)={max_loaded} < 기대 최솟값={expected_min_date}")

        result["issues"] = issues
        result["passed"] = len(issues) == 0

        if issues:
            logger.error(f"  검증 실패: {issues}")
        else:
            logger.info(f"  검증 통과: {result['total_rows']:,}행, max={result['max_date']}")

        return result

    except Exception as e:
        logger.error(f"  검증 쿼리 실패: {e}")
        return {"table": table_fqn, "run_id": run_id, "passed": False, "issues": [str(e)]}


def generate_run_id() -> str:
    return uuid.uuid4().hex[:12]
