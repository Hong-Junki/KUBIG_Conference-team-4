"""
경제지표 수집기: yfinance(VIX/WTI/Gold/DXY) + fredapi(STLFSI4)
"""

import ssl
from datetime import date
from pathlib import Path

import certifi
import pandas as pd
import yfinance as yf

from .config import (
    COLLECT_END,
    COLLECT_START,
    FRED_API_KEY,
    FRED_SERIES,
    YFINANCE_TICKERS,
)
from .utils import get_logger, retry

# macOS Python 3.10 SSL 인증서 패치 (알려진 함정 #17)
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

logger = get_logger(__name__)

RAW_DIR = Path("input/raw/economic")
OUT_PATH = RAW_DIR / "indicators.parquet"


# ──────────────────────────────────────────────
# yfinance
# ──────────────────────────────────────────────

@retry(max_attempts=5, base_delay=3.0, exceptions=(Exception,))
def _fetch_yfinance(name: str, ticker: str, start: date, end: date) -> pd.Series:
    """단일 티커 일간 종가 수집. 인덱스는 UTC DatetimeIndex."""
    t = yf.Ticker(ticker)
    df = t.history(start=start.isoformat(), end=end.isoformat())
    if df.empty:
        logger.warning(f"  {name}({ticker}): 데이터 없음")
        return pd.Series(dtype=float, name=name)

    # timezone → UTC 변환 (티커별로 다름: VIX=Chicago, 나머지=New York)
    close = df["Close"].copy()
    close.index = close.index.tz_convert("UTC").normalize()
    close.name = name
    return close


def collect_yfinance(
    start: date = COLLECT_START,
    end: date = COLLECT_END,
) -> pd.DataFrame:
    """
    모든 yfinance 티커 수집 후 날짜 인덱스 기준 병합.

    Returns:
        날짜(UTC, tz-naive) × 지표명 DataFrame
    """
    series_list = []
    for name, ticker in YFINANCE_TICKERS.items():
        logger.info(f"  yfinance 수집: {name} ({ticker})")
        s = _fetch_yfinance(name, ticker, start, end)
        if not s.empty:
            series_list.append(s)

    if not series_list:
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1)
    # tz-naive UTC로 통일 (저장/병합 편의)
    df.index = df.index.tz_localize(None)
    df.index.name = "date"
    return df


# ──────────────────────────────────────────────
# fredapi
# ──────────────────────────────────────────────

@retry(max_attempts=3, base_delay=5.0, exceptions=(Exception,))
def _fetch_fred(series_id: str, start: date, end: date) -> pd.Series:
    """FRED 시계열 수집. 인덱스는 tz-naive DatetimeIndex."""
    from fredapi import Fred

    fred = Fred(api_key=FRED_API_KEY)
    s = fred.get_series(
        series_id,
        observation_start=start.isoformat(),
        observation_end=end.isoformat(),
    )
    s.name = series_id
    s.index = pd.to_datetime(s.index)  # tz-naive 유지
    s.index.name = "date"
    return s


def collect_fred(
    start: date = COLLECT_START,
    end: date = COLLECT_END,
) -> pd.DataFrame:
    """
    모든 FRED 시리즈 수집.
    주간 데이터(STLFSI4)는 그대로 저장 — ffill은 feature_builder에서 처리.

    Returns:
        날짜(tz-naive) × 시리즈명 DataFrame
    """
    series_list = []
    for name, series_id in FRED_SERIES.items():
        logger.info(f"  fredapi 수집: {name} ({series_id})")
        s = _fetch_fred(series_id, start, end)
        if not s.empty:
            series_list.append(s)

    if not series_list:
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1)
    df.index.name = "date"
    return df


# ──────────────────────────────────────────────
# 통합 수집
# ──────────────────────────────────────────────

def collect_historical(
    start: date = COLLECT_START,
    end: date = COLLECT_END,
) -> None:
    """
    yfinance + fredapi 전체 수집 후 단일 parquet 저장.
    저장 경로: input/raw/economic/indicators.parquet

    컬럼: date(index), VIX, WTI, Gold, DXY, STLFSI4
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"경제지표 수집 시작: {start} ~ {end}")

    df_yf = collect_yfinance(start, end)
    df_fred = collect_fred(start, end)

    if df_yf.empty and df_fred.empty:
        logger.error("수집된 경제지표 없음")
        return

    # 날짜 인덱스 기준 외부 조인 병합
    frames = [f for f in [df_yf, df_fred] if not f.empty]
    df = frames[0].join(frames[1], how="outer") if len(frames) == 2 else frames[0]
    df = df.sort_index()

    # 기존 파일이 있으면 병합 (서로 다른 기간 분담 수집을 누적)
    if OUT_PATH.exists():
        df_old = pd.read_parquet(OUT_PATH)
        df = pd.concat([df_old, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()

    df.to_parquet(OUT_PATH)
    logger.info(f"경제지표 저장 완료: {len(df)}일 × {len(df.columns)}개 지표 → {OUT_PATH}")
    logger.info(f"  컬럼: {list(df.columns)}")
    logger.info(f"  기간: {df.index.min().date()} ~ {df.index.max().date()}")


def collect_recent(days: int = 30) -> None:
    """
    최근 N일 증분 수집. 기존 parquet에 병합.
    """
    from datetime import timedelta

    end = date.today()
    start = end - timedelta(days=days)
    logger.info(f"경제지표 증분 수집: {start} ~ {end}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df_yf = collect_yfinance(start, end)
    df_fred = collect_fred(start, end)

    frames = [f for f in [df_yf, df_fred] if not f.empty]
    if not frames:
        logger.warning("증분 수집 데이터 없음")
        return

    df_new = frames[0].join(frames[1], how="outer") if len(frames) == 2 else frames[0]

    if OUT_PATH.exists():
        df_old = pd.read_parquet(OUT_PATH)
        df_new = pd.concat([df_old, df_new])
        df_new = df_new[~df_new.index.duplicated(keep="last")]  # 최신값 우선

    df_new = df_new.sort_index()
    df_new.to_parquet(OUT_PATH)
    logger.info(f"경제지표 증분 저장 완료: 총 {len(df_new)}일 → {OUT_PATH}")
