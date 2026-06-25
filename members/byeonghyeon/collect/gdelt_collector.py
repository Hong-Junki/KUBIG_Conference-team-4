"""
GDELT 수집기: BigQuery(과거) + DOC 2.0 API(실시간) 이중 경로
"""

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from .config import (
    COLLECT_END,
    COLLECT_START,
    COUNTRIES,
    COUNTRY_BY_GDELT,
    GDELT_BQ_TABLE,
    GDELT_DOC_KEYWORDS,
    GDELT_FIELDS,
)
from .utils import Checkpoint, date_range_months, get_logger, retry

logger = get_logger(__name__)

RAW_DIR = Path("input/raw/gdelt")
CKPT_PATH = RAW_DIR / ".ckpt_historical.json"

# BigQuery에서 가져올 컬럼
BQ_SELECT_FIELDS = [
    "GLOBALEVENTID",
    "SQLDATE",
    "ActionGeo_CountryCode",
    "EventCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumArticles",
    "AvgTone",
]


# ──────────────────────────────────────────────
# BigQuery (과거 데이터)
# ──────────────────────────────────────────────

def _get_bq_client() -> bigquery.Client:
    return bigquery.Client()


def _build_query(
    fips_codes: list[str],
    start: date,
    end: date,
) -> str:
    """특정 국가 + 기간의 GDELT 쿼리 생성."""
    fields = ", ".join(BQ_SELECT_FIELDS)
    fips_str = ", ".join(f"'{c}'" for c in fips_codes)
    return f"""
    SELECT {fields}
    FROM `{GDELT_BQ_TABLE}`
    WHERE _PARTITIONTIME BETWEEN TIMESTAMP('{start}') AND TIMESTAMP('{end}')
      AND CAST(SQLDATE AS STRING) BETWEEN '{start:%Y%m%d}' AND '{end:%Y%m%d}'
      AND ActionGeo_CountryCode IN ({fips_str})
    """


def dry_run_query(query: str) -> float:
    """쿼리 스캔량 GB 반환. 실행 전 반드시 호출."""
    client = _get_bq_client()
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(query, job_config=job_config)
    gb = job.total_bytes_processed / 1e9
    logger.info(f"예상 스캔량: {gb:.2f} GB")
    return gb


@retry(max_attempts=3, base_delay=10.0, exceptions=(Exception,))
def _run_bq_query(query: str) -> pd.DataFrame:
    """BigQuery 쿼리 실행 후 DataFrame 반환."""
    client = _get_bq_client()
    job_config = bigquery.QueryJobConfig(use_query_cache=True)
    df = client.query(query, job_config=job_config).to_dataframe()
    return df


def _parse_sqldate(df: pd.DataFrame) -> pd.DataFrame:
    """SQLDATE(int) → event_date(datetime UTC) 변환."""
    if df.empty:
        return df
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["SQLDATE"].astype(str), format="%Y%m%d", utc=True)
    return df


def _map_fips_to_iso3(df: pd.DataFrame) -> pd.DataFrame:
    """ActionGeo_CountryCode(FIPS) → iso3 매핑 추가. 다중 FIPS 지원."""
    if df.empty:
        return df
    df = df.copy()
    df["iso3"] = df["ActionGeo_CountryCode"].map(
        {code: c["iso3"] for c in COUNTRIES
         for code in (c["gdelt"] if isinstance(c["gdelt"], list) else [c["gdelt"]])}
    )
    return df


def collect_historical_bq(
    start: date = COLLECT_START,
    end: date = COLLECT_END,
    countries: list[dict] | None = None,
    max_gb_per_query: float = 50.0,
) -> None:
    """
    BigQuery로 과거 GDELT 이벤트 수집.

    전략: 월 단위로 쿼리 분할 → 국가별 parquet 저장.
    체크포인트: 월 단위로 완료 추적.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(CKPT_PATH)

    target = countries or COUNTRIES
    fips_codes = []
    for c in target:
        codes = c["gdelt"] if isinstance(c["gdelt"], list) else [c["gdelt"]]
        fips_codes.extend(codes)
    months = date_range_months(start, end)
    total = len(months)

    for idx, (m_start, m_end) in enumerate(months, 1):
        key = f"bq_{m_start:%Y%m}"

        if ckpt.is_done(key):
            logger.info(f"[{idx}/{total}] {m_start:%Y-%m} — 이미 완료, 건너뜀")
            continue

        query = _build_query(fips_codes, m_start, m_end)

        # dry_run 비용 확인
        gb = dry_run_query(query)
        if gb > max_gb_per_query:
            logger.error(f"[{idx}/{total}] {m_start:%Y-%m} 스캔량 {gb:.1f}GB > 한도 {max_gb_per_query}GB. 건너뜀.")
            continue

        logger.info(f"[{idx}/{total}] {m_start:%Y-%m} 수집 중 ({gb:.2f} GB)...")
        df = _run_bq_query(query)
        df = _parse_sqldate(df)
        df = _map_fips_to_iso3(df)

        # 매핑 안 되는 국가 코드 제거 (대상 국가 외)
        df = df.dropna(subset=["iso3"])

        # 국가별로 분리 저장
        saved_count = 0
        for iso3, group in df.groupby("iso3"):
            out_path = RAW_DIR / f"{iso3}.parquet"
            if out_path.exists():
                existing = pd.read_parquet(out_path)
                group = pd.concat([existing, group]).drop_duplicates("GLOBALEVENTID")
            group.to_parquet(out_path, index=False)
            saved_count += len(group)

        ckpt.mark_done(key, {"rows": len(df), "saved": saved_count, "gb_scanned": round(gb, 2)})
        logger.info(f"  {m_start:%Y-%m}: {len(df)}건 수집, {saved_count}건 저장")


# ──────────────────────────────────────────────
# DOC 2.0 API (실시간 데이터)
# ──────────────────────────────────────────────

@retry(max_attempts=3, base_delay=5.0, exceptions=(Exception,))
def _gdeltdoc_timeline(mode: str, keyword: str, country_fips: str,
                       start: str, end: str) -> pd.DataFrame:
    """gdeltdoc 라이브러리로 timeline 수집."""
    from gdeltdoc import GdeltDoc, Filters

    gd = GdeltDoc()
    f = Filters(
        keyword=keyword,
        country=country_fips,
        start_date=start,
        end_date=end,
    )
    df = gd.timeline_search(mode, f)
    return df


def collect_recent_doc(
    days: int = 90,
    countries: list[dict] | None = None,
) -> None:
    """
    DOC 2.0 API로 최근 데이터 수집 (최대 3개월).
    timelinevolraw(기사 수) + timelinetone(톤) 두 모드 수집.

    저장: input/raw/gdelt/{iso3}_doc_vol.parquet, {iso3}_doc_tone.parquet
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    end_date = date.today()
    start_date = end_date - timedelta(days=min(days, 89))  # DOC 2.0 최대 ~3개월

    target = countries or COUNTRIES
    total = len(target)

    for idx, country in enumerate(target, 1):
        name = country["name"]
        iso3 = country["iso3"]
        fips_raw = country["gdelt"]
        fips_list = fips_raw if isinstance(fips_raw, list) else [fips_raw]

        # 특수문자 포함 국가명 → 검색 친화적 키워드로 대체
        keyword = GDELT_DOC_KEYWORDS.get(iso3, name)

        logger.info(f"[{idx}/{total}] {name} DOC 2.0 수집 중 (keyword={keyword})...")

        try:
            # 다중 FIPS 코드 각각 수집 후 병합
            vol_frames = []
            tone_frames = []

            for fips in fips_list:
                vol_df = _gdeltdoc_timeline(
                    "timelinevolraw", keyword, fips,
                    start_date.isoformat(), end_date.isoformat(),
                )
                if not vol_df.empty:
                    vol_frames.append(vol_df)

                tone_df = _gdeltdoc_timeline(
                    "timelinetone", keyword, fips,
                    start_date.isoformat(), end_date.isoformat(),
                )
                if not tone_df.empty:
                    tone_frames.append(tone_df)

                if len(fips_list) > 1:
                    time.sleep(1.0)

            if vol_frames:
                vol_merged = pd.concat(vol_frames).drop_duplicates()
                vol_path = RAW_DIR / f"{iso3}_doc_vol.parquet"
                vol_merged.to_parquet(vol_path, index=False)
                logger.info(f"  {name}: vol {len(vol_merged)}행 → {vol_path}")

            if tone_frames:
                tone_merged = pd.concat(tone_frames).drop_duplicates()
                tone_path = RAW_DIR / f"{iso3}_doc_tone.parquet"
                tone_merged.to_parquet(tone_path, index=False)
                logger.info(f"  {name}: tone {len(tone_merged)}행 → {tone_path}")

        except Exception as e:
            logger.error(f"  {name} DOC 2.0 수집 실패: {e}")

        time.sleep(1.0)  # API 예의
