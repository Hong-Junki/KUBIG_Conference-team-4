"""
GDELT GKG 기사 정보(gdelt_titles) 증분 수집기.

기존 collect_gdelt_titles_gkg.py의 쿼리 생성 로직을 재사용한다.
DELETE+INSERT 방식 대신 staging → MERGE 방식으로 안전하게 적재한다.

MERGE key: (iso3, url)
  - url 단독: 1,404,287건 중복 (동일 기사가 여러 국가에 매핑되므로 부적절)
  - iso3 + url: 중복 0건 (완전 unique)
  → iso3 + url이 MERGE key

MERGE 방식: UPDATE + INSERT (upsert)
  - v2tone_avg, v2themes, v2persons는 GKG 처리 후 수정될 수 있음
  - date는 원천 기준이므로 MATCHED 시에도 UPDATE 허용

적재 흐름:
  gdelt-bq.gdeltv2.gkg_partitioned (공개 BQ)
  → 날짜 범위 + 58개국 필터
  → title, url, domain, language, v2tone_avg, v2themes, v2persons 추출
  → staging table
  → validation
  → MERGE INTO gdelt_titles ON (iso3, url)
  → staging drop
"""

import os
from datetime import date

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
    require_max_date_from_bq,
    log_state_summary,
)

logger = get_logger(__name__)

MERGE_KEYS = ["iso3", "url"]
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


def _build_staging_select_query(
    fips_codes: list[str],
    fips_to_iso3: dict[str, str],
    m_start: date,
    m_end: date,
    staging_fqn: str,
) -> str:
    """
    GKG → staging table INSERT 쿼리.
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
        ROW_NUMBER() OVER (PARTITION BY iso3, url ORDER BY date) AS rn
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


def run_gdelt_titles_incremental(
    project_id: str,
    overlap_days: int | None = None,
    forced_start: date | None = None,
    forced_end: date | None = None,
    dry_run: bool = False,
    max_gb_per_query: float = 100.0,
    run_id: str | None = None,
) -> dict:
    """
    GDELT GKG 기사 정보 증분 수집 + BQ 적재.

    Returns:
        결과 요약 dict.
    """
    run_id = run_id or generate_run_id()
    client = bigquery.Client(project=project_id)
    target_fqn = f"{project_id}.{TARGET_DATASET}.{TARGET_TABLE}"
    staging_fqn = f"{project_id}.{TARGET_DATASET}._staging_{TARGET_TABLE}_{run_id}"

    # 1. 수집 기간 결정 — MAX(date) 조회 실패 시 즉시 RuntimeError (fallback 없음)
    last_date = require_max_date_from_bq(client, target_fqn, "date")

    try:
        start, end = compute_collection_window(
            last_date, "gdelt_titles", overlap_days, forced_start, forced_end
        )
    except ValueError as e:
        logger.warning(str(e))
        return {"source": "gdelt_titles", "run_id": run_id, "skipped": True, "reason": str(e)}

    log_state_summary("gdelt_titles", last_date, start, end)

    # FIPS 코드 매핑
    fips_codes: list[str] = []
    fips_to_iso3: dict[str, str] = {}
    for c in COUNTRIES:
        codes = c["gdelt"] if isinstance(c["gdelt"], list) else [c["gdelt"]]
        for fips in codes:
            fips_codes.append(fips)
            fips_to_iso3[fips] = c["iso3"]

    months = date_range_months(start, end)
    total_scanned_gb = 0.0

    # 2. dry-run 모드: 스캔량만 계산
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
            "start": str(start),
            "end": str(end),
            "last_bq_date": str(last_date),
            "estimated_gb": round(total_scanned_gb, 2),
        }

    # 3. staging 생성
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

        # 4. validation (staging 기준)
        stg_rows_q = f"SELECT COUNT(*) as n FROM `{staging_fqn}`"
        stg_count = list(client.query(stg_rows_q))[0]["n"]
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

        # 필수 컬럼 null 검증
        null_check_q = f"""
        SELECT
          COUNTIF(date IS NULL) as null_date,
          COUNTIF(iso3 IS NULL) as null_iso3,
          COUNTIF(url IS NULL) as null_url
        FROM `{staging_fqn}`
        """
        null_row = list(client.query(null_check_q))[0]
        issues = []
        if null_row["null_date"] > 0:
            issues.append(f"null date: {null_row['null_date']}")
        if null_row["null_iso3"] > 0:
            issues.append(f"null iso3: {null_row['null_iso3']}")
        if null_row["null_url"] > 0:
            issues.append(f"null url: {null_row['null_url']}")

        # merge key duplicate 검증
        dup_q = f"""
        SELECT COUNT(*) as dupes FROM (
          SELECT iso3, url, COUNT(*) as cnt FROM `{staging_fqn}`
          GROUP BY iso3, url HAVING cnt > 1
        )
        """
        dup_count = list(client.query(dup_q))[0]["dupes"]
        if dup_count > 0:
            issues.append(f"merge key (iso3,url) 중복: {dup_count}")

        if issues:
            raise ValueError(f"staging validation 실패: {issues}")

        logger.info(f"  staging validation 통과: {stg_count:,}행, null=0, dup=0")

        # 5. MERGE into target
        affected = merge_into_target(
            client,
            staging_fqn=staging_fqn,
            target_fqn=target_fqn,
            merge_keys=MERGE_KEYS,
            columns=ALL_COLUMNS,
            update_on_match=True,  # v2tone 등 수정치 반영
        )

        # 6. post-write validation
        partition_filter = f"date >= DATE('{start}')"
        validation = validate_after_load(
            client, target_fqn, "date", "iso3",
            expected_min_date=start,
            expected_max_date=end,
            run_id=run_id,
            partition_filter=partition_filter,
        )

        return {
            "source": "gdelt_titles",
            "run_id": run_id,
            "start": str(start),
            "end": str(end),
            "last_bq_date": str(last_date),
            "rows_staged": total_inserted_staging,
            "rows_merged": affected,
            "estimated_gb": round(total_scanned_gb, 2),
            "validation": validation,
            "passed": validation.get("passed", False),
        }

    except Exception as e:
        logger.error(f"  gdelt_titles 수집 실패: {e}")
        raise
    finally:
        drop_table(client, staging_fqn)
