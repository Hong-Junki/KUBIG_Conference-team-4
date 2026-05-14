"""
피처 빌더: 국가x일 피처 테이블 생성

피처 그룹:
  1. ACLED 피처 — 7일 lag 적용 필수 (주간 업데이트, 미래 정보 누수 방지)
     - 이벤트 수/사상자 rolling window (7d, 14d, 30d)
     - 이벤트 유형별 비율
     - 행위자 유형 분포
  2. GDELT 피처 — 당일 데이터 사용 가능 (15분 단위 업데이트)
     - GoldsteinScale/AvgTone/NumMentions 집계 (1d, 7d, 14d, 30d)
     - QuadClass 비율 (verbal/material cooperation/conflict)
  3. 경제 피처 — 글로벌 지표 (국가 공통)
     - VIX/WTI/Gold/DXY 일간값 + 변화율
     - STLFSI4 (ffill된 일간값)

입력: input/processed/{acled,gdelt,economic}/
출력: input/processed/features/features.parquet

스키마: date(UTC), country(iso3), + 피처 컬럼들
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.collect.config import COLLECT_END, COLLECT_START, COUNTRIES

PROCESSED_DIR = Path("input/processed")
FEATURES_DIR = PROCESSED_DIR / "features"

# ACLED lag: 최소 7일 (주간 업데이트 → 미래 정보 유입 방지)
ACLED_LAG_DAYS = 7

# Rolling window 크기
WINDOWS = [7, 14, 30]


# ──────────────────────────────────────────────
# ACLED 피처
# ───────────────────────────���──────────────────

def _build_acled_features(
    iso3: str,
    date_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    단일 국가의 ACLED 기반 피처 생성.

    7일 lag 적용: date 기준으로 date-7일 이전 데이터만 사용.
    """
    acled_path = PROCESSED_DIR / "acled" / f"{iso3}.parquet"

    # 빈 피처 프레임 (ACLED 없는 국가용)
    empty_cols = {}
    for w in WINDOWS:
        empty_cols[f"acled_event_count_{w}d"] = 0.0
        empty_cols[f"acled_fatalities_{w}d"] = 0.0
        empty_cols[f"acled_fatalities_max_{w}d"] = 0.0
    for etype in ["battles", "explosions", "vac"]:
        empty_cols[f"acled_ratio_{etype}"] = 0.0
    for atype in range(1, 9):
        empty_cols[f"acled_actor_type_{atype}_ratio"] = 0.0

    if not acled_path.exists():
        df_empty = pd.DataFrame({"date": date_range})
        for col, val in empty_cols.items():
            df_empty[col] = val
        df_empty["acled_missing_mask"] = 1.0
        return df_empty

    df = pd.read_parquet(acled_path)
    df["event_date"] = pd.to_datetime(df["event_date"], utc=True).dt.normalize()
    # 국가별 ACLED coverage 시작일: 7d lag 적용된 rolling 윈도우에 실제 이벤트가
    # 하나라도 들어오는 가장 이른 date를 결측 경계로 사용
    first_event_date = df["event_date"].min()
    coverage_start = first_event_date + pd.Timedelta(days=ACLED_LAG_DAYS)

    # 일별 집계
    daily_counts = df.groupby("event_date").agg(
        event_count=("event_id_cnty", "count"),
        fatalities_sum=("fatalities", "sum"),
        fatalities_max=("fatalities", "max"),
    ).reindex(date_range, fill_value=0)

    # 이벤트 유형별 일간 수
    etype_map = {
        "battles": "Battles",
        "explosions": "Explosions/Remote violence",
        "vac": "Violence against civilians",
    }
    daily_etypes = {}
    for short, full in etype_map.items():
        daily_etypes[short] = df[df["event_type"] == full].groupby(
            "event_date"
        ).size().reindex(date_range, fill_value=0)

    # 행위자 유형 일간 수
    daily_actors = {}
    for atype in range(1, 9):
        mask = (df["inter1"] == atype) | (df["inter2"] == atype)
        daily_actors[atype] = df[mask].groupby("event_date").size().reindex(
            date_range, fill_value=0
        )

    # 피처 계산 (7일 lag 적용: shift(ACLED_LAG_DAYS) 후 rolling)
    features = pd.DataFrame({"date": date_range})

    for w in WINDOWS:
        shifted_count = daily_counts["event_count"].shift(ACLED_LAG_DAYS)
        shifted_fat_sum = daily_counts["fatalities_sum"].shift(ACLED_LAG_DAYS)
        shifted_fat_max = daily_counts["fatalities_max"].shift(ACLED_LAG_DAYS)

        features[f"acled_event_count_{w}d"] = shifted_count.rolling(
            w, min_periods=1
        ).sum().values
        features[f"acled_fatalities_{w}d"] = shifted_fat_sum.rolling(
            w, min_periods=1
        ).sum().values
        features[f"acled_fatalities_max_{w}d"] = shifted_fat_max.rolling(
            w, min_periods=1
        ).max().values

    # 이벤트 유형 비율 (30일 윈도우, lag 적용)
    total_30d = daily_counts["event_count"].shift(ACLED_LAG_DAYS).rolling(
        30, min_periods=1
    ).sum()
    for short in etype_map:
        count_30d = daily_etypes[short].shift(ACLED_LAG_DAYS).rolling(
            30, min_periods=1
        ).sum()
        features[f"acled_ratio_{short}"] = np.where(
            total_30d > 0, count_30d / total_30d, 0.0
        )

    # 행위자 유형 비율 (30일 윈도우, lag 적용)
    for atype in range(1, 9):
        count_30d = daily_actors[atype].shift(ACLED_LAG_DAYS).rolling(
            30, min_periods=1
        ).sum()
        features[f"acled_actor_type_{atype}_ratio"] = np.where(
            total_30d > 0, count_30d / total_30d, 0.0
        )

    # ACLED 결측 마스크: 해당 일자의 lag 7d 윈도우가 첫 이벤트 이전이면 1
    features["acled_missing_mask"] = (
        date_range < coverage_start
    ).astype(float)

    return features


# ───────────────��──────────────────────────────
# GDELT 피처
# ──────────────────────────────────────────────

def _build_gdelt_features(
    iso3: str,
    date_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    단일 국가의 GDELT 기반 피처 생성.

    GDELT는 15분 단위 업데이트 → lag 불필요.
    단, 당일 데이터는 피처 계산에서 제외 (CLAUDE.md 알려진 함정 #6).
    → shift(1)로 전일까지만 사용.
    """
    gdelt_path = PROCESSED_DIR / "gdelt" / f"{iso3}.parquet"

    empty_cols = {}
    for w in WINDOWS:
        empty_cols[f"gdelt_goldstein_mean_{w}d"] = 0.0
        empty_cols[f"gdelt_goldstein_std_{w}d"] = 0.0
        empty_cols[f"gdelt_tone_mean_{w}d"] = 0.0
        empty_cols[f"gdelt_mentions_sum_{w}d"] = 0.0
        empty_cols[f"gdelt_event_count_{w}d"] = 0.0
    for qc in range(1, 5):
        empty_cols[f"gdelt_quadclass_{qc}_ratio"] = 0.0

    if not gdelt_path.exists():
        df_empty = pd.DataFrame({"date": date_range})
        for col, val in empty_cols.items():
            df_empty[col] = val
        return df_empty

    df = pd.read_parquet(gdelt_path)
    df["event_date"] = pd.to_datetime(df["event_date"], utc=True).dt.normalize()

    # 일별 집계
    daily = df.groupby("event_date").agg(
        goldstein_mean=("GoldsteinScale", "mean"),
        goldstein_std=("GoldsteinScale", "std"),
        tone_mean=("AvgTone", "mean"),
        mentions_sum=("NumMentions", "sum"),
        event_count=("GLOBALEVENTID", "count"),
    ).reindex(date_range)

    daily["goldstein_std"] = daily["goldstein_std"].fillna(0)
    daily["event_count"] = daily["event_count"].fillna(0)
    daily["mentions_sum"] = daily["mentions_sum"].fillna(0)

    # QuadClass별 일간 수
    daily_qc = {}
    for qc in range(1, 5):
        daily_qc[qc] = df[df["QuadClass"] == qc].groupby(
            "event_date"
        ).size().reindex(date_range, fill_value=0)

    # 피처 계산 (shift(1): 당일 제외, 전일까지)
    features = pd.DataFrame({"date": date_range})

    for w in WINDOWS:
        shifted_goldstein = daily["goldstein_mean"].shift(1)
        shifted_goldstein_std = daily["goldstein_std"].shift(1)
        shifted_tone = daily["tone_mean"].shift(1)
        shifted_mentions = daily["mentions_sum"].shift(1)
        shifted_count = daily["event_count"].shift(1)

        features[f"gdelt_goldstein_mean_{w}d"] = shifted_goldstein.rolling(
            w, min_periods=1
        ).mean().values
        features[f"gdelt_goldstein_std_{w}d"] = shifted_goldstein_std.rolling(
            w, min_periods=1
        ).mean().values
        features[f"gdelt_tone_mean_{w}d"] = shifted_tone.rolling(
            w, min_periods=1
        ).mean().values
        features[f"gdelt_mentions_sum_{w}d"] = shifted_mentions.rolling(
            w, min_periods=1
        ).sum().values
        features[f"gdelt_event_count_{w}d"] = shifted_count.rolling(
            w, min_periods=1
        ).sum().values

    # QuadClass 비율 (30일 윈도우)
    total_30d = daily["event_count"].shift(1).rolling(30, min_periods=1).sum()
    for qc in range(1, 5):
        count_30d = daily_qc[qc].shift(1).rolling(30, min_periods=1).sum()
        features[f"gdelt_quadclass_{qc}_ratio"] = np.where(
            total_30d > 0, count_30d / total_30d, 0.0
        )

    return features


# ���─────────────────────────────────────────────
# 경제 피처
# ───��──────────────────────────────────────────

def _build_economic_features(date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    글로벌 경제지표 피처 (국가 공통).

    VIX/WTI/Gold/DXY: 일간값 + 변화율(1d, 7d)
    STLFSI4: 일간값 (ffill 적용됨)
    """
    econ_path = PROCESSED_DIR / "economic" / "indicators.parquet"

    indicators = ["VIX", "WTI", "Gold", "DXY", "STLFSI4"]
    empty_cols = {}
    for ind in indicators:
        empty_cols[f"econ_{ind.lower()}"] = np.nan
        empty_cols[f"econ_{ind.lower()}_pct_1d"] = 0.0
        empty_cols[f"econ_{ind.lower()}_pct_7d"] = 0.0

    if not econ_path.exists():
        df_empty = pd.DataFrame({"date": date_range})
        for col, val in empty_cols.items():
            df_empty[col] = val
        return df_empty

    df = pd.read_parquet(econ_path)
    df.index = pd.to_datetime(df.index, utc=True).normalize()
    df = df.reindex(date_range)
    # 수집 시작 전 날짜(신년 연휴 등) 결측 → bfill 후 ffill
    df = df.bfill().ffill()

    features = pd.DataFrame({"date": date_range})

    for ind in indicators:
        col_lower = ind.lower()
        series = df[ind] if ind in df.columns else pd.Series(np.nan, index=date_range)

        # shift(1): 당일 종가 미확정 방지 (시장 마감 전 예측 시)
        shifted = series.shift(1)

        features[f"econ_{col_lower}"] = shifted.values
        features[f"econ_{col_lower}_pct_1d"] = shifted.pct_change(1).values
        features[f"econ_{col_lower}_pct_7d"] = shifted.pct_change(7).values

    return features


# ──────────────���───────────────────────────────
# 통합 피처 테이블
# ──────────────────────────────────────────────

def build_features(
    start: date = COLLECT_START,
    end: date = COLLECT_END,
    countries: list[dict] | None = None,
) -> pd.DataFrame:
    """
    전체 국가x일 피처 테이블 생성.

    Returns:
        DataFrame: date, country, + 피처 컬럼들
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    target = countries or COUNTRIES

    date_range = pd.date_range(start=start, end=end, freq="D", tz="UTC")
    econ_features = _build_economic_features(date_range)

    all_features = []

    for i, country in enumerate(target):
        iso3 = country["iso3"]
        name = country["name"]
        print(f"  [{i+1}/{len(target)}] {name} ({iso3}) 피처 생성 중...")

        acled_feat = _build_acled_features(iso3, date_range)
        gdelt_feat = _build_gdelt_features(iso3, date_range)

        # 경제 피처는 국가 공통
        df_country = acled_feat.copy()
        df_country["country"] = iso3

        # GDELT 피처 병합 (date 기준)
        gdelt_cols = [c for c in gdelt_feat.columns if c != "date"]
        for col in gdelt_cols:
            df_country[col] = gdelt_feat[col].values

        # 경제 피처 병합
        econ_cols = [c for c in econ_features.columns if c != "date"]
        for col in econ_cols:
            df_country[col] = econ_features[col].values

        all_features.append(df_country)

    df_all = pd.concat(all_features, ignore_index=True)

    # NaN → 0 (결측 피처)
    feature_cols = [c for c in df_all.columns if c not in ("date", "country")]
    df_all[feature_cols] = df_all[feature_cols].fillna(0)

    # 저장
    out_path = FEATURES_DIR / "features.parquet"
    df_all.to_parquet(out_path, index=False)

    print(f"\n피처 테이블 생성 완료:")
    print(f"  shape: {df_all.shape}")
    print(f"  기간: {df_all['date'].min().date()} ~ {df_all['date'].max().date()}")
    print(f"  국가: {df_all['country'].nunique()}개")
    print(f"  피처 수: {len(feature_cols)}개")
    print(f"  저장: {out_path}")

    return df_all


if __name__ == "__main__":
    build_features()
