"""GDELT 2.0 GKG 기반 기사 제목 수집 (BigQuery → BigQuery 직접 적재)

목적
  - DOC 2.0 API throttle 회피 + 노트북 디스크 부담 0 + 4명 작업 결과 자동 통합
  - SELECT 결과를 노트북 parquet 으로 다운로드하지 않고 같은 GCP 프로젝트의
    BigQuery 테이블에 INSERT (네트워크 무료, capacitor 압축)

데이터 흐름
  gdelt-bq.gdeltv2.gkg_partitioned (공개 데이터)
    → REGEXP_EXTRACT + UNNEST + FIPS→ISO3 매핑 (SQL 안에서 변환)
    → conflict-early-warning.conflict_ew.gdelt_titles (INSERT)

대상 테이블 스키마 (PARTITION BY MONTH(date), CLUSTER BY iso3)
  date         DATE      — UTC, DATE 앞 8자리
  iso3         STRING    — 3-letter (V2Locations FIPS UNNEST 후 매핑)
  title        STRING    — Extras<PAGE_TITLE> (2019-09-23 이전 NULL)
  url          STRING    — DocumentIdentifier (dedup 키)
  domain       STRING    — SourceCommonName
  language     STRING    — TranslationInfo srclc (없으면 'eng')
  v2tone_avg   FLOAT64   — V2Tone 첫 필드
  v2themes     STRING    — raw (snappy 압축으로 평균 ~170B/행)
  v2persons    STRING    — raw

체크포인트
  input/raw/gdelt_titles/.ckpt_titles_gkg.json (월 단위, 노트북 로컬)

월 단위 idempotency
  같은 월 재실행 시 해당 월·담당 ISO3 데이터를 먼저 DELETE 후 INSERT.
  분담은 월 단위로 사람별 다르게 잡으므로 동시 실행 충돌 없음.

CLI
  python -m src.collect.collect_gdelt_titles_gkg --start 2020-01-01 --end 2021-12-31
  python -m src.collect.collect_gdelt_titles_gkg --start 2024-01-01 --end 2026-05-29 --countries UKR SYR
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

from google.cloud import bigquery

from .config import COUNTRIES, COUNTRY_BY_ISO3
from .utils import Checkpoint, date_range_months, get_logger, retry

logger = get_logger(__name__)

RAW_DIR = Path("input/raw/gdelt_titles")
CKPT_PATH = RAW_DIR / ".ckpt_titles_gkg.json"

GKG_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"
TARGET_PROJECT = "conflict-early-warning"
TARGET_DATASET = "conflict_ew"
TARGET_TABLE = "gdelt_titles"
TARGET_FQN = f"{TARGET_PROJECT}.{TARGET_DATASET}.{TARGET_TABLE}"

DEFAULT_MAX_GB_PER_QUERY = 50.0


def _build_insert_query(
    fips_codes: list[str],
    fips_to_iso3: dict[str, str],
    m_start: date,
    m_end: date,
) -> str:
    """월 단위 INSERT 쿼리. FIPS UNNEST + ISO3 매핑 + dedup 모두 SQL 안에서 처리."""
    fips_array = ", ".join(f"'{c}'" for c in fips_codes)
    like_or = " OR ".join(f"V2Locations LIKE '%#{c}#%'" for c in fips_codes)
    map_cases = "\n          ".join(
        f"WHEN '{f}' THEN '{i}'" for f, i in fips_to_iso3.items()
    )

    return f"""
    INSERT INTO `{TARGET_FQN}`
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


def _build_delete_query(iso3_list: list[str], m_start: date, m_end: date) -> str:
    """월·담당 ISO3 한정 DELETE — INSERT 전 idempotency 보장.

    partitioned table 의 해당 월 partition 만 스캔 → 비용 미미.
    """
    iso3_array = ", ".join(f"'{i}'" for i in iso3_list)
    return f"""
    DELETE FROM `{TARGET_FQN}`
    WHERE date BETWEEN DATE('{m_start}') AND DATE('{m_end}')
      AND iso3 IN ({iso3_array})
    """


def _build_count_query(iso3_list: list[str], m_start: date, m_end: date) -> str:
    iso3_array = ", ".join(f"'{i}'" for i in iso3_list)
    return f"""
    SELECT
      COUNT(*) AS total,
      COUNTIF(title IS NOT NULL) AS with_title,
      COUNTIF(v2themes IS NOT NULL) AS with_themes,
      COUNT(DISTINCT iso3) AS n_iso3
    FROM `{TARGET_FQN}`
    WHERE date BETWEEN DATE('{m_start}') AND DATE('{m_end}')
      AND iso3 IN ({iso3_array})
    """


def _dry_run_gb(client: bigquery.Client, query: str) -> float:
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False),
    )
    return job.total_bytes_processed / 1e9


@retry(max_attempts=3, base_delay=10.0, exceptions=(Exception,))
def _run_query(client: bigquery.Client, query: str) -> bigquery.QueryJob:
    job = client.query(query)
    job.result()
    return job


def collect_titles_gkg(
    start: date,
    end: date,
    countries: list[dict] | None = None,
    max_gb_per_query: float = DEFAULT_MAX_GB_PER_QUERY,
) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(CKPT_PATH)
    client = bigquery.Client(project=TARGET_PROJECT)

    target = countries or COUNTRIES
    fips_codes: list[str] = []
    fips_to_iso3: dict[str, str] = {}
    iso3_list: list[str] = []
    for c in target:
        codes = c["gdelt"] if isinstance(c["gdelt"], list) else [c["gdelt"]]
        for fips in codes:
            fips_codes.append(fips)
            fips_to_iso3[fips] = c["iso3"]
        iso3_list.append(c["iso3"])

    months = date_range_months(start, end)
    total = len(months)
    logger.info(
        f"GKG → BigQuery 적재 시작: {start} ~ {end} ({total}개월) × {len(target)}개국 "
        f"(FIPS {len(fips_codes)}개) → {TARGET_FQN}"
    )

    total_scanned_gb = 0.0
    total_inserted = 0

    for idx, (m_start, m_end) in enumerate(months, 1):
        key = f"gkg_{m_start:%Y%m}"
        if ckpt.is_done(key):
            logger.info(f"[{idx}/{total}] {m_start:%Y-%m} — 이미 완료, 건너뜀")
            continue

        insert_q = _build_insert_query(fips_codes, fips_to_iso3, m_start, m_end)
        try:
            gb = _dry_run_gb(client, insert_q)
        except Exception as e:
            logger.error(f"[{idx}/{total}] {m_start:%Y-%m} dry_run 실패: {e}")
            continue

        if gb > max_gb_per_query:
            logger.error(
                f"[{idx}/{total}] {m_start:%Y-%m} 스캔량 {gb:.1f} GB > 한도 "
                f"{max_gb_per_query} GB. 스킵."
            )
            continue

        # 1) idempotent DELETE (해당 월·담당 ISO3)
        delete_q = _build_delete_query(iso3_list, m_start, m_end)
        try:
            del_job = _run_query(client, delete_q)
            del_rows = del_job.num_dml_affected_rows or 0
            if del_rows:
                logger.info(f"[{idx}/{total}] {m_start:%Y-%m} 기존 {del_rows:,}행 DELETE (재실행)")
        except Exception as e:
            logger.error(f"[{idx}/{total}] {m_start:%Y-%m} DELETE 실패: {e}")
            continue

        # 2) INSERT
        logger.info(f"[{idx}/{total}] {m_start:%Y-%m} INSERT 실행 중 ({gb:.2f} GB scan)...")
        try:
            ins_job = _run_query(client, insert_q)
        except Exception as e:
            logger.error(f"[{idx}/{total}] {m_start:%Y-%m} INSERT 실패: {e}")
            continue
        total_scanned_gb += gb
        inserted = ins_job.num_dml_affected_rows or 0
        total_inserted += inserted

        # 3) 검증 카운트
        count_row = next(iter(client.query(_build_count_query(iso3_list, m_start, m_end))))
        logger.info(
            f"  {m_start:%Y-%m}: 적재 {inserted:,}행 / "
            f"제목 {count_row.with_title:,} ({count_row.with_title/max(count_row.total,1)*100:.1f}%) / "
            f"테마 {count_row.with_themes:,} / 국가 {count_row.n_iso3}개"
        )

        ckpt.mark_done(
            key,
            {
                "inserted": int(inserted),
                "with_title": int(count_row.with_title),
                "with_themes": int(count_row.with_themes),
                "n_iso3": int(count_row.n_iso3),
                "gb_scanned": round(gb, 2),
            },
        )

    logger.info(
        f"GKG → BigQuery 적재 완료: 총 스캔 {total_scanned_gb:.1f} GB / "
        f"신규 적재 {total_inserted:,}행 → {TARGET_FQN}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GDELT GKG 기반 기사 제목 수집 (BigQuery → BigQuery)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        required=True,
        help="수집 시작일 YYYY-MM-DD (포함)",
    )
    p.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        required=True,
        help="수집 종료일 YYYY-MM-DD (포함)",
    )
    p.add_argument(
        "--countries",
        nargs="+",
        metavar="ISO3",
        help="수집할 국가 ISO3 코드 (예: UKR SYR). 미지정 시 전체 58개국",
    )
    p.add_argument(
        "--max-gb-per-query",
        type=float,
        default=DEFAULT_MAX_GB_PER_QUERY,
        help=f"월별 쿼리 스캔량 한도 GB (기본 {DEFAULT_MAX_GB_PER_QUERY})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.end < args.start:
        raise SystemExit(f"--end({args.end})가 --start({args.start})보다 빠릅니다")

    if args.countries:
        target = [COUNTRY_BY_ISO3[c] for c in args.countries if c in COUNTRY_BY_ISO3]
        missing = set(args.countries) - {c["iso3"] for c in target}
        if missing:
            logger.warning(f"미지원 ISO3 무시: {sorted(missing)}")
        if not target:
            raise SystemExit("유효한 --countries 가 없습니다")
    else:
        target = None

    collect_titles_gkg(
        start=args.start,
        end=args.end,
        countries=target,
        max_gb_per_query=args.max_gb_per_query,
    )


if __name__ == "__main__":
    main()
