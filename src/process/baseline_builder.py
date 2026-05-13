"""
국가별 Baseline 점수 B 산출.

공식: B = 100 × rank_pctile(log1p(mean_annual_fatalities_5y))
데이터: input/raw_merged/acled/ 전체 기간 (수집된 범위 사용)
출력: input/processed/features/baseline_scores.parquet (iso3, country_name, B_score)
"""

from __future__ import annotations

from pathlib import Path

import glob
import numpy as np
import pandas as pd


ACLED_RAW_DIR = Path("input/raw_merged/acled")
OUTPUT_PATH = Path("input/processed/features/baseline_scores.parquet")

CONFLICT_EVENT_TYPES = {
    "Battles",
    "Explosions/Remote violence",
    "Violence against civilians",
}


def build_baseline_scores() -> pd.DataFrame:
    """
    58개국 각각의 연평균 사상자(log1p)를 percentile rank로 변환해
    0-100 baseline 점수를 산출한다.
    """
    records = []
    for fpath in sorted(ACLED_RAW_DIR.glob("*.parquet")):
        iso3 = fpath.stem
        df = pd.read_parquet(fpath, columns=["event_date", "event_type", "fatalities"])
        df = df[df["event_type"].isin(CONFLICT_EVENT_TYPES)].copy()
        df["event_date"] = pd.to_datetime(df["event_date"], utc=True)
        df["year"] = df["event_date"].dt.year

        if df.empty:
            mean_annual = 0.0
        else:
            mean_annual = df.groupby("year")["fatalities"].sum().mean()

        records.append({"iso3": iso3, "mean_annual_fatalities": mean_annual})

    result = pd.DataFrame(records)
    result["log_fatalities"] = np.log1p(result["mean_annual_fatalities"])

    # percentile rank 0-100 (같은 값이면 평균 rank)
    result["B_score"] = result["log_fatalities"].rank(pct=True) * 100
    result["B_score"] = result["B_score"].round(2)

    result = result[["iso3", "mean_annual_fatalities", "log_fatalities", "B_score"]]
    result = result.sort_values("B_score", ascending=False).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"[baseline_builder] {len(result)}개국 B 점수 저장 → {OUTPUT_PATH}")
    print(result[["iso3", "B_score"]].head(10).to_string(index=False))
    return result


if __name__ == "__main__":
    build_baseline_scores()
