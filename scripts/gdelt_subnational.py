"""#62 (A) 하위국가(sub-national) GDELT 피처.

공개 GDELT events_partitioned의 ActionGeo_ADM1Code로 국지 분쟁 신호 복원.
country-day 뉴스/이벤트 평균이 놓치는 "어느 주(州)가 튀는가"를 잡아 ACLED 지리 차원 근사.
material conflict(QuadClass=4) 이벤트만 사용.

산출 (date, country):
  gdelt_sub_n_adm1_1d / _7d / _accel   충돌 ADM1 수(지리 폭) + 가속(확산)
  gdelt_sub_maxadm1_7d                 최핫 주 강도
  gdelt_sub_concentration_7d           Herfindahl(1=한 주 집중, 낮을수록 분산)
  gdelt_sub_maxshare_7d                최대 주 점유율

출력: input/processed/features/gdelt_subnational.parquet (build_dataset auto-join)
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

warnings.filterwarnings("ignore")
load_dotenv(".env", override=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.collect.config import COUNTRIES, COUNTRY_BY_GDELT  # noqa: E402

GCP_PROJECT = os.getenv("GCP_PROJECT", "conflict-ew-mvp-20260604")
PUBLIC = "gdelt-bq.gdeltv2.events_partitioned"
OUT = Path("input/processed/features/gdelt_subnational.parquet")
EPS = 1e-6


def main() -> None:
    fips = []
    for c in COUNTRIES:
        g = c["gdelt"]
        fips += g if isinstance(g, list) else [g]
    fips = sorted(set(fips))

    client = bigquery.Client(project=GCP_PROJECT)
    q = f"""
    WITH ev AS (
      SELECT ActionGeo_CountryCode AS fips, SQLDATE, ActionGeo_ADM1Code AS adm1
      FROM `{PUBLIC}`
      WHERE ActionGeo_CountryCode IN ({','.join(repr(f) for f in fips)})
        AND QuadClass = 4 AND ActionGeo_ADM1Code IS NOT NULL AND ActionGeo_ADM1Code != ''
        AND SQLDATE BETWEEN 20140101 AND 20261231
    ),
    adm1_daily AS (
      SELECT fips, SQLDATE, adm1, COUNT(*) AS cnt FROM ev GROUP BY fips, SQLDATE, adm1
    )
    SELECT fips, SQLDATE, COUNT(*) AS n_adm1, MAX(cnt) AS max_adm1,
           SUM(cnt) AS total, SUM(cnt*cnt) AS sumsq
    FROM adm1_daily GROUP BY fips, SQLDATE
    """
    job = client.query(q)
    df = job.result().to_dataframe(create_bqstorage_client=False)
    print(f"스캔 {job.total_bytes_processed/1e9:.1f}GB (${job.total_bytes_processed/1e12*6.25:.3f}), {len(df):,}행")

    df["date"] = pd.to_datetime(df["SQLDATE"].astype(str), format="%Y%m%d", utc=True)
    df["country"] = df["fips"].map(lambda f: COUNTRY_BY_GDELT[f]["iso3"] if f in COUNTRY_BY_GDELT else None)
    df = df.dropna(subset=["country"])
    # 다중 FIPS 국가(팔레스타인 WE/GZ) → iso3 단위 재집계
    cd = df.groupby(["country", "date"], as_index=False).agg(
        n_adm1=("n_adm1", "sum"), max_adm1=("max_adm1", "max"),
        total=("total", "sum"), sumsq=("sumsq", "sum"))
    cd["concentration"] = cd["sumsq"] / (cd["total"] ** 2 + EPS)
    cd["maxshare"] = cd["max_adm1"] / (cd["total"] + EPS)

    out_parts = []
    for country, g in cd.groupby("country", sort=False):
        g = g.sort_values("date").set_index("date")
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D", tz="UTC")
        g = g.reindex(idx, fill_value=0)
        n = g["n_adm1"]
        n7 = n.rolling(7, min_periods=1).mean(); n30 = n.rolling(30, min_periods=1).mean()
        rec = pd.DataFrame({
            "date": idx, "country": country,
            "gdelt_sub_n_adm1_1d": n.values.astype(np.float32),
            "gdelt_sub_n_adm1_7d": n7.values.astype(np.float32),
            "gdelt_sub_n_adm1_accel": (n7 / (n30 + EPS)).values.astype(np.float32),
            "gdelt_sub_maxadm1_7d": g["max_adm1"].rolling(7, min_periods=1).mean().values.astype(np.float32),
            "gdelt_sub_concentration_7d": g["concentration"].rolling(7, min_periods=1).mean().values.astype(np.float32),
            "gdelt_sub_maxshare_7d": g["maxshare"].rolling(7, min_periods=1).mean().values.astype(np.float32),
        })
        out_parts.append(rec)

    out = pd.concat(out_parts, ignore_index=True)
    feat = [c for c in out.columns if c.startswith("gdelt_sub_")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"저장: {OUT}  shape={out.shape}, 피처 {len(feat)}개")


if __name__ == "__main__":
    main()
