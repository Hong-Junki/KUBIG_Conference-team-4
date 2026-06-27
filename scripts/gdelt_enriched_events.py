"""#61 (C) GDELT 이벤트 강화 피처.

gdelt_processed_events(3억, 국가-일)에서 기존 country GDELT 집계 너머의 폭력-특화 신호 추출.
GDELT는 실시간 소스라 lag 불필요(당일 사용 OK), rolling은 trailing(미래 미포함).

산출 (date, country):
  gdelt2_quad4_1d/7d/30d        QuadClass=4 물리적 충돌 이벤트 수
  gdelt2_quad4_accel            7d/30d 비율(충돌 가속)
  gdelt2_violent_7d/30d         EventRootCode 18/19/20(assault/fight/mass violence)
  gdelt2_violent_accel
  gdelt2_gold_min_7d            7d 내 최저 Goldstein(가장 충돌적 이벤트)
  gdelt2_quad4_share_7d         quad4/total 비율(7d)
  gdelt2_quad4_mentions_7d      충돌 이벤트 보도량(7d)

출력: input/processed/features/gdelt_enriched_events.parquet (build_dataset auto-join)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv(".env", override=True)
import os
GCP_PROJECT = os.getenv("GCP_PROJECT", "conflict-ew-mvp-20260604")
BQ_DATASET = os.getenv("BQ_DATASET", "conflict_ew")
SRC = f"{GCP_PROJECT}.{BQ_DATASET}.gdelt_processed_events"
OUT = Path("input/processed/features/gdelt_enriched_events.parquet")
EPS = 1e-6

import sys  # noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.collect.config import COUNTRIES  # noqa: E402


def main() -> None:
    iso3s = [c["iso3"] for c in COUNTRIES]
    client = bigquery.Client(project=GCP_PROJECT)
    q = f"""
    SELECT iso3, DATE(event_date) AS date,
      COUNTIF(QuadClass = 4) AS quad4,
      COUNTIF(EventRootCode IN ('18','19','20')) AS violent,
      COUNT(*) AS total,
      MIN(GoldsteinScale) AS gold_min,
      SUM(IF(QuadClass = 4, NumMentions, 0)) AS quad4_mentions
    FROM `{SRC}`
    WHERE iso3 IN ({','.join(repr(c) for c in iso3s)})
    GROUP BY iso3, date
    """
    job = client.query(q)
    daily = job.result().to_dataframe(create_bqstorage_client=False)
    print(f"스캔 {job.total_bytes_processed/1e9:.1f}GB (${job.total_bytes_processed/1e12*6.25:.3f}), daily {len(daily):,}행")
    daily["date"] = pd.to_datetime(daily["date"], utc=True)

    out_parts = []
    for country, g in daily.groupby("iso3", sort=False):
        g = g.sort_values("date").set_index("date")
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D", tz="UTC")
        g = g.reindex(idx, fill_value=0)
        q4 = g["quad4"]; vi = g["violent"]; tot = g["total"]
        q4_7 = q4.rolling(7, min_periods=1).sum()
        q4_30 = q4.rolling(30, min_periods=1).sum()
        vi_7 = vi.rolling(7, min_periods=1).sum()
        vi_30 = vi.rolling(30, min_periods=1).sum()
        rec = pd.DataFrame({
            "date": idx, "country": country,
            "gdelt2_quad4_1d": q4.values.astype(np.float32),
            "gdelt2_quad4_7d": q4_7.values.astype(np.float32),
            "gdelt2_quad4_30d": q4_30.values.astype(np.float32),
            "gdelt2_quad4_accel": ((q4_7 / 7) / ((q4_30 / 30) + EPS)).values.astype(np.float32),
            "gdelt2_violent_7d": vi_7.values.astype(np.float32),
            "gdelt2_violent_30d": vi_30.values.astype(np.float32),
            "gdelt2_violent_accel": ((vi_7 / 7) / ((vi_30 / 30) + EPS)).values.astype(np.float32),
            "gdelt2_gold_min_7d": g["gold_min"].rolling(7, min_periods=1).min().values.astype(np.float32),
            "gdelt2_quad4_share_7d": (q4_7 / (tot.rolling(7, min_periods=1).sum() + EPS)).values.astype(np.float32),
            "gdelt2_quad4_mentions_7d": g["quad4_mentions"].rolling(7, min_periods=1).sum().values.astype(np.float32),
        })
        out_parts.append(rec)

    out = pd.concat(out_parts, ignore_index=True)
    feat = [c for c in out.columns if c.startswith("gdelt2_")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"저장: {OUT}  shape={out.shape}, 피처 {len(feat)}개")


if __name__ == "__main__":
    main()
