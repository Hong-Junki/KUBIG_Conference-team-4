"""
U/C/S/I 서브스코어 산출.

입력: input/processed/features/features.parquet
출력: output/scores/subscores.parquet (date, country, U, C, S, I)

데이터 제약 반영:
  - 우리 ACLED는 Battles/Explosions/VAC만 수집 (Protests/Riots 없음)
  - U (Unrest): acled_ratio_vac (민간인 피해 비율) z-score — 소요 proxy
  - C (Conflict): acled_fatalities_7d z-score — 무력 강도 직접 지표
  - S (Security): gdelt_quadclass_4_ratio + inverted goldstein_mean_7d 조합
  - I (Information): gdelt_mentions_sum_7d vs mentions_sum_30d 스파이크 비율

참고: risk-score.md §3
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .welford import batch_rolling_score, zscore_to_score, compute_rolling_zscore

FEATURES_PATH = Path("input/processed/features/features.parquet")
OUTPUT_PATH = Path("output/scores/subscores.parquet")


def _compute_U(df: pd.DataFrame) -> pd.Series:
    """
    U (Unrest): 민간인 피해 비율(acled_ratio_vac) rolling z-score.
    Protests/Riots 데이터 없어 VAC ratio를 unrest proxy로 사용.
    """
    return batch_rolling_score(df, "acled_ratio_vac")


def _compute_C(df: pd.DataFrame) -> pd.Series:
    """C (Conflict): 7일 사상자 rolling z-score."""
    return batch_rolling_score(df, "acled_fatalities_7d")


def _compute_S(df: pd.DataFrame) -> pd.Series:
    """
    S (Security): GDELT QuadClass4 비율(높을수록 대립) + Goldstein mean 역수(낮을수록 불안)
    0.6 × normalized(QuadClass4_ratio) + 0.4 × normalized(−Goldstein_mean)
    """
    q4 = batch_rolling_score(df, "gdelt_quadclass_4_ratio")

    # Goldstein mean 부호 반전 — 낮을수록(적대적) 점수 높음
    df = df.copy()
    df["gdelt_goldstein_inverted"] = -df["gdelt_goldstein_mean_7d"]
    g_inv = batch_rolling_score(df, "gdelt_goldstein_inverted")

    return (0.6 * q4 + 0.4 * g_inv).round(2)


def _compute_I(df: pd.DataFrame) -> pd.Series:
    """
    I (Information): 7일 뉴스 언급량 rolling z-score.
    국가별 90일 baseline 대비 최근 7일 언급량 이상 여부.
    U/C/S와 동일한 z-score → sigmoid 0-100 방식으로 range 통일.
    """
    return batch_rolling_score(df, "gdelt_mentions_sum_7d")


def build_subscores(
    features_path: Path = FEATURES_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    U/C/S/I 서브스코어 산출 후 parquet 저장.
    """
    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["country", "date"]).reset_index(drop=True)

    out = pd.DataFrame({
        "date": df["date"],
        "country": df["country"],
        "U": _compute_U(df),
        "C": _compute_C(df),
        "S": _compute_S(df),
        "I": _compute_I(df),
    })
    out["C_state"] = ((out["U"] + out["C"] + out["S"] + out["I"]) / 4).round(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"[subscore_builder] {len(out)} rows 저장 → {output_path}")
    print("서브스코어 통계:")
    for col in ["U", "C", "S", "I", "C_state"]:
        print(f"  {col}: mean={out[col].mean():.1f}, std={out[col].std():.1f}, "
              f"p5={out[col].quantile(0.05):.1f}, p95={out[col].quantile(0.95):.1f}")
    return out


if __name__ == "__main__":
    build_subscores()
