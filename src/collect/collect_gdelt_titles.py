"""
GDELT 2.0 기사 제목 수집 (DOC 2.0 ArtList API)

목적: 국가별 일자별 뉴스 기사 제목 수집 → LLM 기반 피처 추출(Track C)의 입력으로 사용

데이터 소스
  - GDELT 2.0 DOC API (ArtList mode) — gdeltdoc 라이브러리 사용
  - 커버리지: 2015-02-19 ~ 현재 (15분 단위 인덱스)

저장 구조
  input/raw/gdelt_titles/
    {iso3}/{YYYY-MM}.parquet   # 국가×월 단위 파편화 (append + dedup by url)
    .ckpt_titles.json          # 체크포인트 (국가+일자 단위)

스키마(parquet)
  date          : YYYY-MM-DD (UTC, 수집 대상일)
  iso3          : 3-letter country code
  title         : 기사 제목
  url           : 기사 URL (중복 제거 키)
  domain        : 출처 도메인 (예: bbc.com)
  language      : 언어 코드
  sourcecountry : GDELT가 분류한 출처 국가 (FIPS)
  seendate      : GDELT가 처음 본 시각 (UTC, ISO8601)

CLI
  python -m src.collect.collect_gdelt_titles --start 2024-01-01 --end 2024-12-31
  python -m src.collect.collect_gdelt_titles --start 2024-01-01 --end 2024-01-07 --countries UKR SYR
  python -m src.collect.collect_gdelt_titles --start 2024-01-01 --end 2024-01-07 --max-per-day 100

팀 작업 분담 (확정 2026-05-29) — 자세한 안내는 proc/plan/gdelt_titles_collect_guide.md
  - 팀원1:  --start 2015-02-19 --end 2015-12-31  &  --start 2022-01-01 --end 2023-12-31
  - 팀원2:  --start 2016-01-01 --end 2017-12-31
  - 본인 :  --start 2018-01-01 --end 2019-12-31  &  --start 2024-01-01 --end 2026-05-29
  - 팀원4:  --start 2020-01-01 --end 2021-12-31
  (2014-01-01 ~ 2015-02-18 은 GDELT 2.0 커버리지 밖이라 별도 처리)

운영 메모
  1. 같은 명령으로 재실행하면 체크포인트로 자동 재개 (idempotent)
  2. 기본 maxrecords=250/국가/일 — 다운스트림 LLM 처리비용 고려한 캡
  3. API 호출 간 1초 대기 (예의)
  4. 실패 시 5회 지수 백오프 재시도
  5. 모든 timestamp는 UTC
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from .config import COUNTRIES, COUNTRY_BY_ISO3, GDELT_DOC_KEYWORDS
from .utils import Checkpoint, get_logger, retry

logger = get_logger(__name__)

RAW_DIR = Path("input/raw/gdelt_titles")
CKPT_PATH = RAW_DIR / ".ckpt_titles.json"

# GDELT DOC 2.0 API 직접 호출 (gdeltdoc은 timeout 미지원 → connection reset 시 90초 hang)
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "conflict-early-warning/0.1 (KUBIG team, research)"

DEFAULT_MAX_RECORDS_PER_DAY = 250
HTTP_TIMEOUT_SECONDS = 30.0

# 차단 방지: 호출 간 5초 + 0~3초 jitter (실측: 3초 간격에서도 429 발생 사례 있음)
API_DELAY_MIN = 5.0
API_DELAY_JITTER = 3.0

# 429/503 등 throttle 응답 시 long backoff
THROTTLE_BACKOFF_SECONDS = 60.0

KEEP_COLS = [
    "date",
    "iso3",
    "title",
    "url",
    "domain",
    "language",
    "sourcecountry",
    "seendate",
]


class GdeltThrottled(Exception):
    """GDELT 측 throttle 신호 (429/503/연속 reset). long backoff용."""


def _polite_sleep() -> None:
    time.sleep(API_DELAY_MIN + random.random() * API_DELAY_JITTER)


@retry(max_attempts=5, base_delay=5.0, exceptions=(Exception,))
def _fetch_titles_for_day(
    country_fips: str,
    keyword: str,
    day: date,
    max_records: int,
) -> pd.DataFrame:
    """단일 국가, 단일 날짜의 기사 목록 수집 (DOC 2.0 ArtList 직접 호출).

    GDELT DOC API는 [start, end) 반열림 구간이므로 end_date에 다음 날 지정.
    timestamp 포맷: YYYYMMDDHHMMSS (UTC).
    """
    start_ts = day.strftime("%Y%m%d") + "000000"
    end_ts = (day + timedelta(days=1)).strftime("%Y%m%d") + "000000"

    query_parts = [f'"{keyword}"', f"sourcecountry:{country_fips}"]
    query = " ".join(query_parts)

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max_records),
        "startdatetime": start_ts,
        "enddatetime": end_ts,
        "sort": "DateDesc",
    }
    headers = {"User-Agent": USER_AGENT}

    resp = requests.get(
        GDELT_DOC_URL, params=params, headers=headers, timeout=HTTP_TIMEOUT_SECONDS
    )

    if resp.status_code in (429, 503):
        logger.warning(
            f"GDELT throttle({resp.status_code}) — {THROTTLE_BACKOFF_SECONDS:.0f}초 대기"
        )
        time.sleep(THROTTLE_BACKOFF_SECONDS)
        raise GdeltThrottled(f"HTTP {resp.status_code}")

    resp.raise_for_status()

    # GDELT는 잘못된 쿼리에도 200 + text/html 로 답하는 경우가 있음
    ctype = resp.headers.get("content-type", "")
    if "json" not in ctype:
        # 빈 결과 또는 쿼리 오류 — 빈 DataFrame 반환
        return pd.DataFrame()

    try:
        data = resp.json()
    except ValueError:
        return pd.DataFrame()

    articles = data.get("articles", [])
    if not articles:
        return pd.DataFrame()
    return pd.DataFrame(articles)


def _save_month_parquet(
    iso3: str,
    year_month: str,
    new_rows: pd.DataFrame,
) -> int:
    """월 단위 parquet에 append (url 기준 중복 제거). 저장 후 총 행 수 반환."""
    out_dir = RAW_DIR / iso3
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year_month}.parquet"

    if new_rows.empty:
        return 0

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        merged = pd.concat([existing, new_rows], ignore_index=True).drop_duplicates(
            subset=["url"], keep="last"
        )
    else:
        merged = new_rows

    merged.to_parquet(out_path, index=False)
    return len(merged)


def collect_titles(
    start: date,
    end: date,
    countries: list[dict] | None = None,
    max_records: int = DEFAULT_MAX_RECORDS_PER_DAY,
) -> None:
    """국가×일자 단위로 GDELT 기사 제목 수집.

    Args:
        start: 시작일 (포함)
        end: 종료일 (포함)
        countries: 대상 국가 dict 리스트. None이면 COUNTRIES 전체
        max_records: 국가×일자당 최대 기사 수 (DOC API 캡은 250)
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(CKPT_PATH)

    target = countries or COUNTRIES
    total_days = (end - start).days + 1
    total_jobs = len(target) * total_days
    job_idx = 0
    skipped = 0
    empty_days = 0

    logger.info(
        f"수집 시작: {start} ~ {end} ({total_days}일) × {len(target)}개국 = {total_jobs}작업"
    )
    logger.info(f"  국가/일자당 최대 {max_records}건, 저장: {RAW_DIR}/")

    cur = start
    while cur <= end:
        for country in target:
            job_idx += 1
            iso3 = country["iso3"]
            name = country["name"]
            fips_raw = country["gdelt"]
            fips_list = fips_raw if isinstance(fips_raw, list) else [fips_raw]
            keyword = GDELT_DOC_KEYWORDS.get(iso3, name)

            ckpt_key = f"{iso3}_{cur:%Y%m%d}"
            if ckpt.is_done(ckpt_key):
                skipped += 1
                continue

            day_frames = []
            try:
                for fips in fips_list:
                    df = _fetch_titles_for_day(fips, keyword, cur, max_records)
                    if df is not None and not df.empty:
                        df = df.copy()
                        df["iso3"] = iso3
                        df["date"] = cur.strftime("%Y-%m-%d")
                        day_frames.append(df)
                    _polite_sleep()
            except Exception as e:
                logger.error(
                    f"  [{job_idx}/{total_jobs}] {iso3} {cur}: 실패(스킵) — {e}"
                )
                continue

            if day_frames:
                combined = pd.concat(day_frames, ignore_index=True).drop_duplicates(
                    subset=["url"], keep="first"
                )
                cols = [c for c in KEEP_COLS if c in combined.columns]
                combined = combined[cols]

                year_month = cur.strftime("%Y-%m")
                total_in_month = _save_month_parquet(iso3, year_month, combined)
                ckpt.mark_done(
                    ckpt_key,
                    {"rows_new": len(combined), "month_total": total_in_month},
                )

                if job_idx % 100 == 0 or len(combined) >= max_records:
                    logger.info(
                        f"  [{job_idx}/{total_jobs}] {iso3} {cur}: "
                        f"+{len(combined)}건 (월 {year_month} 누적 {total_in_month})"
                    )
            else:
                empty_days += 1
                ckpt.mark_done(ckpt_key, {"rows_new": 0, "month_total": 0})

        cur += timedelta(days=1)

    logger.info(
        f"수집 완료: {start} ~ {end} | 완료 {job_idx} (스킵 {skipped}, 빈 일자 {empty_days})"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GDELT 기사 제목 수집기 (DOC 2.0 ArtList)",
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
        "--max-per-day",
        type=int,
        default=DEFAULT_MAX_RECORDS_PER_DAY,
        help=f"국가×일자당 최대 기사 수 (기본 {DEFAULT_MAX_RECORDS_PER_DAY}, DOC API 캡 250)",
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
            raise SystemExit("유효한 --countries가 없습니다")
    else:
        target = None

    collect_titles(
        start=args.start,
        end=args.end,
        countries=target,
        max_records=args.max_per_day,
    )


if __name__ == "__main__":
    main()
