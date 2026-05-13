"""
라벨 생성: ACLED 이벤트 → 국가x일 이진 라벨 (3종)

라벨 정의:
  y           = 해당 국가 다음 3일 내 무력충돌 1건 이상 (continuation 포함)
  y_onset     = 지난 14일 평화(분쟁 이벤트 0건) AND 다음 3일 분쟁 1건 이상 (엄격 onset)
  y_escalation = y_onset OR (다음 3일 fatalities > 지난 14일 평균 + 2σ) (onset + 급격 악화)

  대상 이벤트: Battles, Explosions/Remote violence, Violence against civilians

주 타겟(학술 정석):
  - y_escalation: 조기경보 시스템의 본연 목표(새 분쟁 + escalation)
  - y: persistence baseline 비교용

입력: input/processed/acled/{iso3}.parquet
출력: input/processed/labels/labels.parquet

스키마:
  - date: datetime64[ns, UTC]
  - country: str (iso3)
  - y: int — continuation 포함 (기존 정의, 유지)
  - y_onset: int — 엄격 onset (희소, 보조)
  - y_escalation: int — 주 학습 타겟 (onset + 급격 악화)
  - fatalities_next3d: int
  - event_count_next3d: int
  - past14d_event_count: int  — 진단용
  - past14d_fatalities_mean: float — 진단용
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.collect.config import COLLECT_END, COLLECT_START, COUNTRIES

PROCESSED_ACLED_DIR = Path("input/processed/acled")
LABELS_DIR = Path("input/processed/labels")

# 라벨 대상 이벤트 유형 (무력충돌만)
LABEL_EVENT_TYPES = {
    "Battles",
    "Explosions/Remote violence",
    "Violence against civilians",
}

# 예측 윈도우: 기준일 다음 날부터 3일 (24-72시간)
FORECAST_WINDOW_DAYS = 3

# ONSET 정의: 지난 N일 평화 조건
PEACE_LOOKBACK_DAYS = 14

# ESCALATION 정의: 다음 3일 fatalities > 지난 14일 평균 + N σ
ESCALATION_SIGMA = 2.0


def build_labels(
    start: date = COLLECT_START,
    end: date = COLLECT_END,
    countries: list[dict] | None = None,
) -> pd.DataFrame:
    """
    국가x일 이진 라벨(3종) 생성.

    Returns:
        DataFrame: date, country, y, y_onset, y_escalation,
                   fatalities_next3d, event_count_next3d,
                   past14d_event_count, past14d_fatalities_mean
    """
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    target = countries or COUNTRIES

    date_range = pd.date_range(start=start, end=end, freq="D", tz="UTC")

    all_labels = []

    for country in target:
        iso3 = country["iso3"]
        acled_path = PROCESSED_ACLED_DIR / f"{iso3}.parquet"

        if not acled_path.exists():
            print(f"  {iso3}: ACLED 파일 없음 — 전체 y=0으로 채움")
            df_country = pd.DataFrame({
                "date": date_range,
                "country": iso3,
                "y": 0,
                "y_onset": 0,
                "y_escalation": 0,
                "fatalities_next3d": 0,
                "event_count_next3d": 0,
                "past14d_event_count": 0,
                "past14d_fatalities_mean": 0.0,
            })
            all_labels.append(df_country)
            continue

        # ACLED 이벤트 로드 (라벨 대상 유형만)
        df_acled = pd.read_parquet(acled_path)
        df_acled = df_acled[df_acled["event_type"].isin(LABEL_EVENT_TYPES)].copy()
        df_acled["event_date"] = pd.to_datetime(df_acled["event_date"], utc=True)
        df_acled["event_date_only"] = df_acled["event_date"].dt.normalize()

        # 일별 집계
        daily = df_acled.groupby("event_date_only").agg(
            event_count=("event_id_cnty", "count"),
            fatalities=("fatalities", "sum"),
        ).reindex(date_range, fill_value=0)

        # ── 향후 3일 윈도우 (y, y_onset, y_escalation 공통) ──
        event_count_next3d = daily["event_count"].shift(-1).rolling(
            window=FORECAST_WINDOW_DAYS, min_periods=1
        ).sum().fillna(0).astype(int)

        fatalities_next3d = daily["fatalities"].shift(-1).rolling(
            window=FORECAST_WINDOW_DAYS, min_periods=1
        ).sum().fillna(0).astype(int)

        # ── 과거 14일 윈도우 (onset/escalation 판정용) ──
        # 기준일 포함 직전 14일: shift(1) 로 기준일 제외 후 rolling(14)
        past14d_event_count = daily["event_count"].shift(1).rolling(
            window=PEACE_LOOKBACK_DAYS, min_periods=1
        ).sum().fillna(0).astype(int)

        past14d_fatalities_mean = daily["fatalities"].shift(1).rolling(
            window=PEACE_LOOKBACK_DAYS, min_periods=1
        ).mean().fillna(0.0)

        past14d_fatalities_std = daily["fatalities"].shift(1).rolling(
            window=PEACE_LOOKBACK_DAYS, min_periods=2
        ).std().fillna(0.0)

        # ── 라벨 산출 ──
        y = (event_count_next3d > 0).astype(int)

        # y_onset: 지난 14일 평화(이벤트 0) AND 다음 3일 분쟁
        y_onset = ((past14d_event_count == 0) & (event_count_next3d > 0)).astype(int)

        # y_escalation:
        #   (1) onset 포함
        #   (2) 다음 3일 fatalities 평균 > 지난 14일 평균 + 2σ
        #       단 지난 14일 평균이 0이고 다음 3일 사상자 발생 시도 escalation
        #       (분산 0으로 threshold 계산 불가 방지)
        next3d_daily_mean = fatalities_next3d / FORECAST_WINDOW_DAYS
        threshold = past14d_fatalities_mean + ESCALATION_SIGMA * past14d_fatalities_std
        # 과거 평균·분산 모두 0이면 any fatality를 escalation으로 간주
        baseline_zero = (past14d_fatalities_mean == 0) & (past14d_fatalities_std == 0)
        esc_spike = np.where(
            baseline_zero,
            fatalities_next3d > 0,
            next3d_daily_mean > threshold,
        ).astype(bool)

        y_escalation = (y_onset.astype(bool) | esc_spike).astype(int)

        df_country = pd.DataFrame({
            "date": date_range,
            "country": iso3,
            "y": y.values,
            "y_onset": y_onset.values,
            "y_escalation": y_escalation,
            "fatalities_next3d": fatalities_next3d.values,
            "event_count_next3d": event_count_next3d.values,
            "past14d_event_count": past14d_event_count.values,
            "past14d_fatalities_mean": past14d_fatalities_mean.values,
        })
        all_labels.append(df_country)

    df_labels = pd.concat(all_labels, ignore_index=True)

    # 마지막 3일 제거 (미래 데이터 필요)
    cutoff = pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=FORECAST_WINDOW_DAYS)
    df_labels = df_labels[df_labels["date"] <= cutoff]

    out_path = LABELS_DIR / "labels.parquet"
    df_labels.to_parquet(out_path, index=False)

    # 통계 출력
    total = len(df_labels)
    pos_y = df_labels["y"].sum()
    pos_onset = df_labels["y_onset"].sum()
    pos_esc = df_labels["y_escalation"].sum()
    print(f"라벨 생성 완료: {total}행")
    print(f"  y            : {pos_y:>6} 양성 ({pos_y/total*100:5.2f}%) — continuation 포함")
    print(f"  y_onset      : {pos_onset:>6} 양성 ({pos_onset/total*100:5.2f}%) — 엄격 onset")
    print(f"  y_escalation : {pos_esc:>6} 양성 ({pos_esc/total*100:5.2f}%) — onset + 급격 악화")
    print(f"  기간: {df_labels['date'].min().date()} ~ {df_labels['date'].max().date()}")
    print(f"  국가: {df_labels['country'].nunique()}개")
    print(f"  저장: {out_path}")

    return df_labels


if __name__ == "__main__":
    build_labels()
