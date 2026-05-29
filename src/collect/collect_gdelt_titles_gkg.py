"""GDELT 2.0 GKG 기반 기사 제목 수집 (BigQuery)

목적
  - DOC 2.0 API 가 throttle(429) 로 비현실적(~30일 소요) → BigQuery GKG 우회
  - 기존 events 수집과 동일한 BigQuery 인프라 재사용
  - 실시간 호환: GKG 도 15분 주기 갱신 (events/mentions/gkg 동시 릴리즈)

데이터 소스
  - gdelt-bq.gdeltv2.gkg_partitioned (파티션 필터 적용, 스캔량 절감)
  - 커버리지: 2015-02-17 ~ 현재 (DOC API 2015-02-19 보다 2일 빠른 시작)

PoC 결과 (SYR 2024-03-01~07, 2000건 샘플; 2026-05-29)
  - PAGE_TITLE 추출률 99.15%
  - DocumentIdentifier / V2Tone 100%
  - V2Themes 94%, V2Persons 61%
  - 1주일 1국가 스캔 5.78 GB → 12년 전체 약 3.6 TB ≈ $18

저장 구조 (DOC 스크립트와 동일 — 다운스트림 호환)
  input/raw/gdelt_titles/
    {iso3}/{YYYY-MM}.parquet
    .ckpt_titles_gkg.json

스키마(parquet)
  date          : YYYY-MM-DD (UTC, DATE 앞 8자리)
  iso3          : 3-letter country code (V2Locations 매칭 → 국가별 행 복제)
  title         : 기사 제목 (Extras<PAGE_TITLE> html.unescape)
  url           : DocumentIdentifier (중복 제거 키)
  domain        : SourceCommonName
  language      : TranslationInfo srclc (예: ara, eng; 없으면 'eng')
  sourcecountry : (GKG 미제공) — 빈 문자열
  seendate      : DATE (YYYYMMDDHHMMSS → ISO8601 UTC)
  v2themes      : V2Themes 원본 (LLM 입력 보조)
  v2tone_avg    : V2Tone 첫 필드 (tone score)

CLI
  python -m src.collect.collect_gdelt_titles_gkg --start 2015-02-19 --end 2026-05-29
  python -m src.collect.collect_gdelt_titles_gkg --start 2024-01-01 --end 2024-12-31 --countries UKR SYR

운영 메모
  1. 월 단위 분할 쿼리 + 체크포인트 (월 단위 idempotent)
  2. 쿼리당 dry_run 비용 확인 → max_gb_per_query 초과 시 스킵
  3. V2Locations 의 모든 매칭 FIPS 를 UNNEST → iso3 별 행 분리
  4. 기존 DOC 스크립트와 폴더 / 스키마 호환 (이미 수집된 DOC parquet 와 url 중복 제거)
"""

from __future__ import annotations

import argparse
import html
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from .config import COUNTRIES, COUNTRY_BY_ISO3
from .utils import Checkpoint, date_range_months, get_logger, retry

logger = get_logger(__name__)

RAW_DIR = Path("input/raw/gdelt_titles")
CKPT_PATH = RAW_DIR / ".ckpt_titles_gkg.json"
GKG_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"

DEFAULT_MAX_GB_PER_QUERY = 50.0

PAGE_TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.DOTALL)
SRCLC_RE = re.compile(r"srclc:([a-z]+)", re.IGNORECASE)

KEEP_COLS = [
    "date",
    "iso3",
    "title",
    "url",
    "domain",
    "language",
    "sourcecountry",
    "seendate",
    "v2themes",
    "v2tone_avg",
]


def _build_query(fips_codes: list[str], start: date, end: date) -> str:
    """월 단위 GKG 쿼리. V2Locations 의 FIPS 코드 추출 → 매칭 국가별 행 분리.

    end 는 month_end (포함). _PARTITIONTIME 은 [start, end+1day) 반열림.
    """
    fips_array = ", ".join(f"'{c}'" for c in fips_codes)
    like_or = " OR ".join(f"V2Locations LIKE '%#{c}#%'" for c in fips_codes)

    return f"""
    WITH gkg AS (
      SELECT
        DATE,
        DocumentIdentifier,
        SourceCommonName,
        V2Themes,
        V2Tone,
        TranslationInfo,
        Extras,
        ARRAY(
          SELECT DISTINCT SPLIT(loc, '#')[SAFE_OFFSET(2)] AS fips
          FROM UNNEST(SPLIT(V2Locations, ';')) AS loc
          WHERE SPLIT(loc, '#')[SAFE_OFFSET(2)] IN ({fips_array})
        ) AS matched_fips
      FROM `{GKG_TABLE}`
      WHERE _PARTITIONTIME BETWEEN TIMESTAMP('{start}') AND TIMESTAMP('{end}')
        AND CAST(SUBSTR(CAST(DATE AS STRING), 1, 8) AS INT64)
            BETWEEN {start:%Y%m%d} AND {end:%Y%m%d}
        AND V2Locations IS NOT NULL
        AND ({like_or})
    )
    SELECT
      DATE,
      DocumentIdentifier,
      SourceCommonName,
      V2Themes,
      V2Tone,
      TranslationInfo,
      Extras,
      fips
    FROM gkg, UNNEST(matched_fips) AS fips
    WHERE ARRAY_LENGTH(matched_fips) > 0
    """


def _dry_run_gb(client: bigquery.Client, query: str) -> float:
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False),
    )
    return job.total_bytes_processed / 1e9


@retry(max_attempts=3, base_delay=10.0, exceptions=(Exception,))
def _run_query(client: bigquery.Client, query: str) -> pd.DataFrame:
    return client.query(query).to_dataframe()


def _extract_page_title(extras: str | None) -> str | None:
    if not extras:
        return None
    m = PAGE_TITLE_RE.search(extras)
    if not m:
        return None
    title = m.group(1).strip()
    if not title:
        return None
    return html.unescape(title)


def _extract_language(translation_info: str | None) -> str:
    if not translation_info:
        return "eng"
    m = SRCLC_RE.search(translation_info)
    return m.group(1).lower() if m else "eng"


def _extract_tone_avg(v2tone: str | None) -> float | None:
    if not v2tone:
        return None
    first = v2tone.split(",", 1)[0].strip()
    try:
        return float(first)
    except ValueError:
        return None


def _transform(df_raw: pd.DataFrame, fips_to_iso3: dict[str, str]) -> pd.DataFrame:
    """BigQuery 결과 → 다운스트림 호환 스키마로 변환."""
    if df_raw.empty:
        return pd.DataFrame(columns=KEEP_COLS)

    out = pd.DataFrame()
    out["iso3"] = df_raw["fips"].map(fips_to_iso3)
    out = out.dropna(subset=["iso3"])
    df_raw = df_raw.loc[out.index]

    date_str = df_raw["DATE"].astype(str).str.zfill(14)
    out["date"] = (
        date_str.str.slice(0, 4) + "-" + date_str.str.slice(4, 6) + "-" + date_str.str.slice(6, 8)
    )
    out["title"] = df_raw["Extras"].apply(_extract_page_title)
    out["url"] = df_raw["DocumentIdentifier"]
    out["domain"] = df_raw["SourceCommonName"]
    out["language"] = df_raw["TranslationInfo"].apply(_extract_language)
    out["sourcecountry"] = ""
    out["seendate"] = (
        date_str.str.slice(0, 4) + "-" + date_str.str.slice(4, 6) + "-" + date_str.str.slice(6, 8)
        + "T" + date_str.str.slice(8, 10) + ":" + date_str.str.slice(10, 12) + ":"
        + date_str.str.slice(12, 14) + "Z"
    )
    out["v2themes"] = df_raw["V2Themes"]
    out["v2tone_avg"] = df_raw["V2Tone"].apply(_extract_tone_avg)

    out = out.dropna(subset=["title", "url"])
    out = out.drop_duplicates(subset=["iso3", "url"], keep="first")
    return out[KEEP_COLS].reset_index(drop=True)


def _save_month_parquet(iso3: str, year_month: str, new_rows: pd.DataFrame) -> int:
    """월 단위 parquet 에 append (url 중복 제거). 총 행 수 반환."""
    if new_rows.empty:
        return 0
    out_dir = RAW_DIR / iso3
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year_month}.parquet"

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        merged = pd.concat([existing, new_rows], ignore_index=True).drop_duplicates(
            subset=["url"], keep="last"
        )
    else:
        merged = new_rows
    merged.to_parquet(out_path, index=False)
    return len(merged)


def collect_titles_gkg(
    start: date,
    end: date,
    countries: list[dict] | None = None,
    max_gb_per_query: float = DEFAULT_MAX_GB_PER_QUERY,
) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(CKPT_PATH)
    client = bigquery.Client()

    target = countries or COUNTRIES
    fips_codes: list[str] = []
    fips_to_iso3: dict[str, str] = {}
    for c in target:
        codes = c["gdelt"] if isinstance(c["gdelt"], list) else [c["gdelt"]]
        for fips in codes:
            fips_codes.append(fips)
            fips_to_iso3[fips] = c["iso3"]

    months = date_range_months(start, end)
    total = len(months)
    logger.info(
        f"GKG 수집 시작: {start} ~ {end} ({total}개월) × {len(target)}개국 (FIPS {len(fips_codes)}개)"
    )

    total_scanned_gb = 0.0
    total_rows = 0

    for idx, (m_start, m_end) in enumerate(months, 1):
        key = f"gkg_{m_start:%Y%m}"
        if ckpt.is_done(key):
            logger.info(f"[{idx}/{total}] {m_start:%Y-%m} — 이미 완료, 건너뜀")
            continue

        query = _build_query(fips_codes, m_start, m_end)

        try:
            gb = _dry_run_gb(client, query)
        except Exception as e:
            logger.error(f"[{idx}/{total}] {m_start:%Y-%m} dry_run 실패: {e}")
            continue

        if gb > max_gb_per_query:
            logger.error(
                f"[{idx}/{total}] {m_start:%Y-%m} 스캔량 {gb:.1f} GB > 한도 {max_gb_per_query} GB. 스킵."
            )
            continue

        logger.info(f"[{idx}/{total}] {m_start:%Y-%m} 쿼리 실행 중 ({gb:.2f} GB)...")
        df_raw = _run_query(client, query)
        total_scanned_gb += gb
        logger.info(f"  수신 {len(df_raw):,} 행 → 변환 중...")

        df = _transform(df_raw, fips_to_iso3)
        if df.empty:
            ckpt.mark_done(key, {"rows_in": len(df_raw), "rows_out": 0, "gb_scanned": round(gb, 2)})
            continue

        year_month = m_start.strftime("%Y-%m")
        saved_total = 0
        for iso3, group in df.groupby("iso3"):
            n_total = _save_month_parquet(iso3, year_month, group)
            saved_total += n_total

        total_rows += len(df)
        ckpt.mark_done(
            key,
            {
                "rows_in": len(df_raw),
                "rows_out": int(len(df)),
                "saved_month_total": int(saved_total),
                "gb_scanned": round(gb, 2),
            },
        )
        logger.info(
            f"  {m_start:%Y-%m}: 변환 {len(df):,}행 / {df['iso3'].nunique()}개국 저장"
        )

    logger.info(
        f"GKG 수집 완료: 총 스캔 {total_scanned_gb:.1f} GB / 신규 변환 {total_rows:,}행"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GDELT GKG 기반 기사 제목 수집 (BigQuery)",
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
