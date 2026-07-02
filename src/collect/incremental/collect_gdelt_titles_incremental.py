"""
GDELT GKG 기사 정보(gdelt_titles) 증분 수집기.

두 가지 수집 모드:

[date 모드 (기존, backfill)]
  - start_date, end_date 기준 월 단위 수집
  - BQ MAX(date) - overlap_days 자동 계산

[timestamp 모드 (15분 증분)]
  - GKG DATE 필드 기준 timestamp window 수집
  - GKG DATE: 14자리 정수 YYYYMMDDHHMMSS (시분초 있음)
  - start_timestamp, end_timestamp 파라미터 필요
  - watermark 갱신은 호출자(run_incremental)가 담당

MERGE key: (date, iso3, url) — partition pruning 포함
MERGE 방식: UPDATE + INSERT (v2tone 등 수정치 반영)

적재 흐름:
  gdelt-bq.gdeltv2.gkg_partitioned (공개 BQ)
  → 날짜/timestamp 범위 + 58개국 필터
  → staging table
  → validation
  → MERGE INTO gdelt_titles ON (date, iso3, url)
  → staging drop
"""

from datetime import date, datetime, timedelta, timezone

from google.cloud import bigquery
from google.cloud.bigquery import SchemaField

from ..collect_gdelt_titles_gkg import (
    GKG_TABLE,
    TARGET_DATASET,
    TARGET_TABLE,
    _dry_run_gb,
)
from ..config import COUNTRIES
from ..utils import date_range_months, get_logger
from .bigquery_io import (
    drop_table,
    generate_run_id,
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

MERGE_KEYS = ["date", "iso3", "url"]
GDELT_TITLES_SCHEMA = [
    SchemaField("date", "DATE", mode="REQUIRED"),
    SchemaField("iso3", "STRING", mode="REQUIRED"),
    SchemaField("title", "STRING"),
    SchemaField("url", "STRING", mode="REQUIRED"),
    SchemaField("domain", "STRING"),
    SchemaField("language", "STRING"),
    SchemaField("v2tone_avg", "FLOAT"),
    SchemaField("v2themes", "STRING"),
    SchemaField("v2persons", "STRING"),
]
ALL_COLUMNS = [f.name for f in GDELT_TITLES_SCHEMA]

# GKG DATE cross-day lag: 날짜 경계 포함을 위해 파티션 1일 여유
_GKG_PARTITION_LOOKBACK_DAYS = 1


def _build_staging_select_query(
    fips_codes: list[str],
    fips_to_iso3: dict[str, str],
    m_start: date,
    m_end: date,
    staging_fqn: str,
) -> str:
    """
    날짜 단위 GKG → staging table INSERT 쿼리 (date 모드).
    기존 _build_insert_query 구조를 재사용하되 target을 staging으로 변경.
    """
    fips_array = ", ".join(f"'{c}'" for c in fips_codes)
    like_or = " OR ".join(f"V2Locations LIKE '%#{c}#%'" for c in fips_codes)
    map_cases = "\n          ".join(
        f"WHEN '{f}' THEN '{i}'" for f, i in fips_to_iso3.items()
    )

    return f"""
    INSERT INTO `{staging_fqn}`
      (date, iso3, title, url, domain, language, v2tone_avg, v2themes, v2persons)
    WITH gkg AS (
      SELECT
        DATE,
        DocumentIdentifier,
        SourceCommonName,
        V2Tone,
        TranslationInfo,
        Extras,
        V2Themes,
        V2Persons,
        ARRAY(
          SELECT DISTINCT SPLIT(loc, '#')[SAFE_OFFSET(2)] AS fips
          FROM UNNEST(SPLIT(V2Locations, ';')) AS loc
          WHERE SPLIT(loc, '#')[SAFE_OFFSET(2)] IN ({fips_array})
        ) AS matched_fips
      FROM `{GKG_TABLE}`
      WHERE _PARTITIONTIME BETWEEN TIMESTAMP('{m_start}') AND TIMESTAMP('{m_end}')
        AND CAST(SUBSTR(CAST(DATE AS STRING), 1, 8) AS INT64)
            BETWEEN {m_start:%Y%m%d} AND {m_end:%Y%m%d}
        AND V2Locations IS NOT NULL
        AND DocumentIdentifier IS NOT NULL
        AND ({like_or})
    ),
    exploded AS (
      SELECT
        PARSE_DATE('%Y%m%d', SUBSTR(CAST(g.DATE AS STRING), 1, 8)) AS date,
        CASE fips
          {map_cases}
        END AS iso3,
        NULLIF(TRIM(REGEXP_EXTRACT(g.Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>')), '') AS title,
        g.DocumentIdentifier AS url,
        g.SourceCommonName AS domain,
        COALESCE(LOWER(REGEXP_EXTRACT(g.TranslationInfo, r'srclc:([a-zA-Z]+)')), 'eng') AS language,
        SAFE_CAST(SPLIT(g.V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) AS v2tone_avg,
        g.V2Themes AS v2themes,
        g.V2Persons AS v2persons
      FROM gkg AS g, UNNEST(g.matched_fips) AS fips
    ),
    deduped AS (
      SELECT
        date, iso3, title, url, domain, language, v2tone_avg, v2themes, v2persons,
        ROW_NUMBER() OVER (PARTITION BY date, iso3, url ORDER BY date) AS rn
      FROM exploded
      WHERE iso3 IS NOT NULL
    )
    SELECT date, iso3, title, url, domain, language, v2tone_avg, v2themes, v2persons
    FROM deduped
    WHERE rn = 1
    """


def _build_staging_select_query_ts(
    fips_codes: list[str],
    fips_to_iso3: dict[str, str],
    start_ts: datetime,
    end_ts: datetime,
    staging_fqn: str,
) -> str:
    """
    GKG DATE 기반 timestamp window INSERT 쿼리 (timestamp 모드).

    GKG DATE 컬럼: 14자리 정수 YYYYMMDDHHMMSS.
    _PARTITIONTIME: 하루 전 ~ 당일 (cross-day 경계 대응).
    """
    partition_start = (start_ts.date() - timedelta(days=_GKG_PARTITION_LOOKBACK_DAYS))
    partition_end = end_ts.date()
    start_int = dateadded_int(start_ts)
    end_int = dateadded_int(end_ts)

    fips_array = ", ".join(f"'{c}'" for c in fips_codes)
    like_or = " OR ".join(f"V2Locations LIKE '%#{c}#%'" for c in fips_codes)
    map_cases = "\n          ".join(
        f"WHEN '{f}' THEN '{i}'" for f, i in fips_to_iso3.items()
    )

    return f"""
    INSERT INTO `{staging_fqn}`
      (date, iso3, title, url, domain, language, v2tone_avg, v2themes, v2persons)
    WITH gkg AS (
      SELECT
        DATE,
        DocumentIdentifier,
        SourceCommonName,
        V2Tone,
        TranslationInfo,
        Extras,
        V2Themes,
        V2Persons,
        ARRAY(
          SELECT DISTINCT SPLIT(loc, '#')[SAFE_OFFSET(2)] AS fips
          FROM UNNEST(SPLIT(V2Locations, ';')) AS loc
          WHERE SPLIT(loc, '#')[SAFE_OFFSET(2)] IN ({fips_array})
        ) AS matched_fips
      FROM `{GKG_TABLE}`
      WHERE _PARTITIONTIME BETWEEN TIMESTAMP('{partition_start}') AND TIMESTAMP('{partition_end}')
        AND DATE BETWEEN {start_int} AND {end_int}
        AND V2Locations IS NOT NULL
        AND DocumentIdentifier IS NOT NULL
        AND ({like_or})
    ),
    exploded AS (
      SELECT
        PARSE_DATE('%Y%m%d', SUBSTR(CAST(g.DATE AS STRING), 1, 8)) AS date,
        CASE fips
          {map_cases}
        END AS iso3,
        NULLIF(TRIM(REGEXP_EXTRACT(g.Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>')), '') AS title,
        g.DocumentIdentifier AS url,
        g.SourceCommonName AS domain,
        COALESCE(LOWER(REGEXP_EXTRACT(g.TranslationInfo, r'srclc:([a-zA-Z]+)')), 'eng') AS language,
        SAFE_CAST(SPLIT(g.V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) AS v2tone_avg,
        g.V2Themes AS v2themes,
        g.V2Persons AS v2persons
      FROM gkg AS g, UNNEST(g.matched_fips) AS fips
    ),
    deduped AS (
      SELECT
        date, iso3, title, url, domain, language, v2tone_avg, v2themes, v2persons,
        ROW_NUMBER() OVER (PARTITION BY date, iso3, url ORDER BY date) AS rn
      FROM exploded
      WHERE iso3 IS NOT NULL
    )
    SELECT date, iso3, title, url, domain, language, v2tone_avg, v2themes, v2persons
    FROM deduped
    WHERE rn = 1
    """


def _create_empty_staging(client: bigquery.Client, staging_fqn: str) -> None:
    """staging table을 스키마로 미리 생성."""
    tbl = bigquery.Table(staging_fqn, schema=GDELT_TITLES_SCHEMA)
    client.create_table(tbl, exists_ok=True)
    logger.info(f"  staging table 생성: {staging_fqn}")


def _get_fips_mapping() -> tuple[list[str], dict[str, str]]:
    """설정에서 FIPS 코드 목록과 FIPS→ISO3 매핑 반환."""
    fips_codes: list[str] = []
    fips_to_iso3: dict[str, str] = {}
    for c in COUNTRIES:
        codes = c["gdelt"] if isinstance(c["gdelt"], list) else [c["gdelt"]]
        for fips in codes:
            fips_codes.append(fips)
            fips_to_iso3[fips] = c["iso3"]
    return fips_codes, fips_to_iso3


def run_gdelt_titles_incremental(
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
    GDELT GKG 기사 정보 증분 수집 + BQ 적재.

    timestamp 모드: start_timestamp, end_timestamp 둘 다 지정.
    date 모드: 기존 방식.

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
    staging_fqn = f"{project_id}.{TARGET_DATASET}._staging_{TARGET_TABLE}_{run_id}"

    last_date = require_max_date_from_bq(client, target_fqn, "date")
    try:
        start, end = compute_collection_window(
            last_date, "gdelt_titles", overlap_days, forced_start, forced_end
        )
    except ValueError as e:
        logger.warning(str(e))
        return {"source": "gdelt_titles", "run_id": run_id, "skipped": True, "reason": str(e)}

    log_state_summary("gdelt_titles", last_date, start, end)

    fips_codes, fips_to_iso3 = _get_fips_mapping()
    months = date_range_months(start, end)
    total_scanned_gb = 0.0

    if dry_run:
        for idx, (m_start, m_end) in enumerate(months, 1):
            dummy_fqn = "dry_run_placeholder"
            insert_q = _build_staging_select_query(
                fips_codes, fips_to_iso3, m_start, m_end, dummy_fqn
            )
            try:
                gb = _dry_run_gb(client, insert_q.replace(dummy_fqn, target_fqn))
                total_scanned_gb += gb
                logger.info(f"  [{idx}/{len(months)}] {m_start:%Y-%m}: 예상 {gb:.2f} GB")
            except Exception as e:
                logger.warning(f"  [{idx}/{len(months)}] {m_start:%Y-%m} dry-run 실패: {e}")
        logger.info(f"  총 예상 스캔: {total_scanned_gb:.2f} GB")
        return {
            "source": "gdelt_titles",
            "run_id": run_id,
            "dry_run": True,
            "passed": True,
            "mode": "date",
            "start": str(start),
            "end": str(end),
            "last_bq_date": str(last_date),
            "estimated_gb": round(total_scanned_gb, 2),
        }

    _create_empty_staging(client, staging_fqn)
    try:
        total_inserted_staging = 0
        for idx, (m_start, m_end) in enumerate(months, 1):
            insert_q = _build_staging_select_query(
                fips_codes, fips_to_iso3, m_start, m_end, staging_fqn
            )
            try:
                gb = _dry_run_gb(client, insert_q.replace(staging_fqn, target_fqn))
                total_scanned_gb += gb
            except Exception as e:
                logger.warning(f"  [{idx}/{len(months)}] {m_start:%Y-%m} dry-run 실패: {e}")
                gb = 0.0

            if gb > max_gb_per_query:
                logger.error(
                    f"  [{idx}/{len(months)}] {m_start:%Y-%m} 스캔 한도 초과 "
                    f"({gb:.1f} GB > {max_gb_per_query} GB). 건너뜀."
                )
                continue

            logger.info(f"  [{idx}/{len(months)}] {m_start:%Y-%m} INSERT → staging ({gb:.2f} GB)...")
            try:
                job = client.query(insert_q)
                job.result()
                rows = job.num_dml_affected_rows or 0
                total_inserted_staging += rows
                logger.info(f"    {m_start:%Y-%m}: {rows:,}행 → staging")
            except Exception as e:
                logger.error(f"  [{idx}/{len(months)}] {m_start:%Y-%m} staging INSERT 실패: {e}")
                raise

        logger.info(f"  staging 총 적재: {total_inserted_staging:,}행")

        result = _validate_staging_and_merge(
            client=client,
            project_id=project_id,
            target_fqn=target_fqn,
            staging_fqn=staging_fqn,
            run_id=run_id,
            start=start,
            end=end,
        )
        result.update({
            "mode": "date",
            "last_bq_date": str(last_date),
            "rows_staged": total_inserted_staging,
            "estimated_gb": round(total_scanned_gb, 2),
        })
        return result

    except Exception as e:
        logger.error(f"  gdelt_titles 수집 실패: {e}")
        raise
    finally:
        drop_table(client, staging_fqn)


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
    GKG DATE 기반 15분 증분 수집.

    GKG DATE 필드(14자리 YYYYMMDDHHMMSS)로 timestamp window 필터링.
    단일 쿼리 (월 단위 분할 불필요 — 15분 window는 매우 작음).
    MERGE key (date, iso3, url)로 중복 없이 idempotent 적재.
    """
    staging_fqn = f"{project_id}.{TARGET_DATASET}._staging_{TARGET_TABLE}_{run_id}"

    fips_codes, fips_to_iso3 = _get_fips_mapping()

    # dry-run 스캔량 확인
    dummy_insert_q = _build_staging_select_query_ts(
        fips_codes, fips_to_iso3, start_ts, end_ts, "dry_run_placeholder"
    )
    try:
        gb = _dry_run_gb(client, dummy_insert_q.replace("dry_run_placeholder", target_fqn))
        logger.info(f"  [timestamp] GKG DATE window 예상 스캔: {gb:.3f} GB")
    except Exception as e:
        logger.warning(f"  [timestamp] dry-run 실패: {e}")
        gb = 0.0

    if gb > max_gb_per_query:
        msg = f"스캔 한도 초과 ({gb:.1f} GB > {max_gb_per_query} GB)"
        logger.error(f"  {msg}")
        return {"source": "gdelt_titles", "run_id": run_id, "passed": False, "error": msg}

    if dry_run:
        return {
            "source": "gdelt_titles",
            "run_id": run_id,
            "dry_run": True,
            "passed": True,
            "mode": "timestamp",
            "start_timestamp": start_ts.isoformat(),
            "end_timestamp": end_ts.isoformat(),
            "estimated_gb": round(gb, 3),
        }

    _create_empty_staging(client, staging_fqn)
    try:
        insert_q = _build_staging_select_query_ts(
            fips_codes, fips_to_iso3, start_ts, end_ts, staging_fqn
        )
        logger.info(f"  [timestamp] INSERT → staging ...")
        try:
            job = client.query(insert_q)
            job.result()
            staged_rows = job.num_dml_affected_rows or 0
            logger.info(f"  [timestamp] {staged_rows:,}행 → staging")
        except Exception as e:
            logger.error(f"  [timestamp] staging INSERT 실패: {e}")
            raise

        start_date = (start_ts.date() - timedelta(days=_GKG_PARTITION_LOOKBACK_DAYS))
        end_date = end_ts.date()

        result = _validate_staging_and_merge(
            client=client,
            project_id=project_id,
            target_fqn=target_fqn,
            staging_fqn=staging_fqn,
            run_id=run_id,
            start=start_date,
            end=end_date,
        )
        result.update({
            "mode": "timestamp",
            "start_timestamp": start_ts.isoformat(),
            "end_timestamp": end_ts.isoformat(),
            "rows_staged": staged_rows,
            "estimated_gb": round(gb, 3),
        })
        return result

    except Exception as e:
        logger.error(f"  gdelt_titles timestamp 수집 실패: {e}")
        raise
    finally:
        drop_table(client, staging_fqn)


def _validate_staging_and_merge(
    client: bigquery.Client,
    project_id: str,
    target_fqn: str,
    staging_fqn: str,
    run_id: str,
    start: date,
    end: date,
) -> dict:
    """staging 검증 → MERGE → post-validation."""
    stg_count = list(client.query(f"SELECT COUNT(*) as n FROM `{staging_fqn}`"))[0]["n"]
    if stg_count == 0:
        logger.warning("  staging이 비어 있음. 원천에 신규 데이터 없음.")
        return {
            "source": "gdelt_titles",
            "run_id": run_id,
            "start": str(start),
            "end": str(end),
            "rows_collected": 0,
            "rows_merged": 0,
            "passed": True,
        }

    null_row = list(client.query(f"""
    SELECT
      COUNTIF(date IS NULL) as null_date,
      COUNTIF(iso3 IS NULL) as null_iso3,
      COUNTIF(url IS NULL) as null_url
    FROM `{staging_fqn}`
    """))[0]
    issues = []
    if null_row["null_date"] > 0:
        issues.append(f"null date: {null_row['null_date']}")
    if null_row["null_iso3"] > 0:
        issues.append(f"null iso3: {null_row['null_iso3']}")
    if null_row["null_url"] > 0:
        issues.append(f"null url: {null_row['null_url']}")

    dup_count = list(client.query(f"""
    SELECT COUNT(*) as dupes FROM (
      SELECT date, iso3, url, COUNT(*) as cnt FROM `{staging_fqn}`
      GROUP BY date, iso3, url HAVING cnt > 1
    )
    """))[0]["dupes"]
    if dup_count > 0:
        issues.append(f"merge key (date,iso3,url) 중복: {dup_count}")

    if issues:
        raise ValueError(f"staging validation 실패: {issues}")

    logger.info(f"  staging validation 통과: {stg_count:,}행, null=0, dup=0")

    partition_filter = f"T.date BETWEEN DATE('{start}') AND DATE('{end}')"
    affected = merge_into_target(
        client,
        staging_fqn=staging_fqn,
        target_fqn=target_fqn,
        merge_keys=MERGE_KEYS,
        columns=ALL_COLUMNS,
        update_on_match=True,
        target_partition_filter=partition_filter,
    )

    validation = validate_after_load(
        client, target_fqn, "date", "iso3",
        expected_min_date=start,
        expected_max_date=end,
        run_id=run_id,
        partition_filter=f"date >= DATE('{start}')",
    )

    return {
        "source": "gdelt_titles",
        "run_id": run_id,
        "start": str(start),
        "end": str(end),
        "rows_collected": stg_count,
        "rows_merged": affected,
        "validation": validation,
        "passed": validation.get("passed", False),
    }
