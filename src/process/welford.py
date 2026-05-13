"""
Welford 온라인 z-score 유틸.

배치 학습용: pandas rolling mean/std (compute_rolling_zscore)
실시간 추론용: WelfordState 온라인 업데이트
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 배치: pandas rolling z-score
# ──────────────────────────────────────────────

def compute_rolling_zscore(
    series: pd.Series,
    window: int = 90,
    min_periods: int = 30,
) -> pd.Series:
    """
    pandas rolling window z-score.
    (value − rolling_mean) / rolling_std, NaN → 0.

    주의: 그룹별로 호출해야 함 (국가별 독립 rolling).
    """
    rolling = series.rolling(window=window, min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.fillna(0.0)


def zscore_to_score(z: pd.Series, clip_z: float = 3.0) -> pd.Series:
    """
    z-score → sigmoid → 0-100 점수.
    z=0 → 50, z=3 → ~95, z=-3 → ~5.
    clip_z 이상/이하는 sigmoid로 자연 포화.
    """
    z_clipped = z.clip(-clip_z, clip_z)
    sigmoid = 1.0 / (1.0 + np.exp(-z_clipped))
    return (sigmoid * 100).round(2)


def batch_rolling_score(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "country",
    window: int = 90,
    min_periods: int = 30,
    clip_z: float = 3.0,
) -> pd.Series:
    """
    국가별 그룹 rolling z-score → sigmoid 0-100 변환.
    df는 date 순으로 정렬되어 있어야 함.
    """
    z = df.groupby(group_col, group_keys=False)[value_col].apply(
        lambda s: compute_rolling_zscore(s, window=window, min_periods=min_periods)
    )
    return zscore_to_score(z)


# ──────────────────────────────────────────────
# 실시간 온라인: WelfordState
# ──────────────────────────────────────────────

@dataclass
class WelfordState:
    """
    Welford 온라인 알고리즘 — 단일 스트림의 평균/분산을 O(1) 업데이트.
    국가별 인스턴스를 유지해 실시간 z-score 산출.
    """
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0  # sum of squared deviations

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / (self.n - 1) if self.n >= 2 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def zscore(self, value: float) -> float:
        if self.n < 2 or self.std == 0:
            return 0.0
        return (value - self.mean) / self.std

    def zscore_to_score(self, value: float, clip_z: float = 3.0) -> float:
        z = max(-clip_z, min(clip_z, self.zscore(value)))
        return round(1.0 / (1.0 + math.exp(-z)) * 100, 2)
