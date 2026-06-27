"""Phase1 (#64) GDELT 이벤트 = ACLED 미러 피처.

공개 GDELT events_partitioned(CAMEO)에서 ACLED의 자기회귀 구조를 실시간 재현.
ACLED 3유형(Battles/Explosions·Remote/VAC) + 에스컬레이션 사다리 + 행위자 dyad + 강도.
GDELT 실시간이라 당일 사용 OK, rolling은 trailing(causal), 모든 피처 real-time 산출 가능.

산출 (date, country) ~24피처, prefix gdev_:
  유형: battles/remote/vac × 1d/7d/30d/accel
  사다리: protest/massviol 7d/30d, assault 7d
  행위자: gov/reb/civ_target 7d
  강도: goldneg 7d/30d/accel, conf_mentions 7d
  이상: battles_z90 (당일 vs 90d baseline z-score)

사용법:
  python scripts/gdelt_acled_mirror.py --dry-run   # 비용 추정
  python scripts/gdelt_acled_mirror.py
"""

from __future__ import annotations

import argparse
import os
_SI = int((os.environ.get('SERVE_START') or '2014-01-01').replace('-',''))
_EI = int((os.environ.get('SERVE_END') or '2026-12-31').replace('-',''))
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
OUT = Path("input/processed/features/gdelt_acled_mirror.parquet")
EPS = 1e-6


def build_sql(fips):
    fin = ",".join(repr(f) for f in fips)
    return f"""
    WITH ev AS (
      SELECT ActionGeo_CountryCode AS fips, SQLDATE,
        EventRootCode AS root, EventCode AS code, QuadClass AS qc,
        GoldsteinScale AS gold, NumMentions AS mentions,
        Actor1Type1Code AS a1, Actor2Type1Code AS a2
      FROM `{PUBLIC}`
      WHERE ActionGeo_CountryCode IN ({fin}) AND SQLDATE BETWEEN {_SI} AND {_EI}
    )
    SELECT fips, SQLDATE,
      COUNTIF(root='19') AS battles,
      COUNTIF(STARTS_WITH(code,'195') OR root='20') AS remote,
      COUNTIF(a2='CVL' AND root IN ('18','19','20')) AS vac,
      COUNTIF(root='14') AS protest,
      COUNTIF(root='18') AS assault,
      COUNTIF(root='20') AS massviol,
      COUNTIF(a1 IN ('GOV','MIL','COP') OR a2 IN ('GOV','MIL','COP')) AS gov,
      COUNTIF(a1 IN ('REB','INS','SEP') OR a2 IN ('REB','INS','SEP')) AS reb,
      COUNTIF(a2='CVL') AS civ_target,
      SUM(IF(gold<0, -gold, 0)) AS goldneg,
      SUM(IF(qc=4, mentions, 0)) AS conf_mentions
    FROM ev GROUP BY fips, SQLDATE
    """


def roll(s, w, fn="sum"):
    r = s.rolling(w, min_periods=1)
    return getattr(r, fn)()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    fips = []
    for c in COUNTRIES:
        g = c["gdelt"]; fips += g if isinstance(g, list) else [g]
    fips = sorted(set(fips))
    client = bigquery.Client(project=GCP_PROJECT)
    sql = build_sql(fips)
    if args.dry_run:
        j = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
        gb = j.total_bytes_processed / 1e9
        print(f"[dry-run] 예상 스캔 {gb:.1f}GB (~${gb/1000*6.25:.2f})")
        return
    job = client.query(sql)
    daily = job.result().to_dataframe(create_bqstorage_client=False)
    print(f"스캔 {job.total_bytes_processed/1e9:.1f}GB (${job.total_bytes_processed/1e12*6.25:.3f}), {len(daily):,}행")
    daily["date"] = pd.to_datetime(daily["SQLDATE"].astype(str), format="%Y%m%d", utc=True)
    daily["country"] = daily["fips"].map(lambda f: COUNTRY_BY_GDELT[f]["iso3"] if f in COUNTRY_BY_GDELT else None)
    daily = daily.dropna(subset=["country"])
    base = ["battles", "remote", "vac", "protest", "assault", "massviol",
            "gov", "reb", "civ_target", "goldneg", "conf_mentions"]
    cd = daily.groupby(["country", "date"], as_index=False)[base].sum()

    out_parts = []
    for country, g in cd.groupby("country", sort=False):
        g = g.sort_values("date").set_index("date")
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D", tz="UTC")
        g = g.reindex(idx, fill_value=0)
        rec = {"date": idx, "country": country}
        for sig in ["battles", "remote", "vac", "goldneg"]:
            s = g[sig]; s7 = roll(s, 7); s30 = roll(s, 30)
            rec[f"gdev_{sig}_1d"] = s.values.astype(np.float32)
            rec[f"gdev_{sig}_7d"] = s7.values.astype(np.float32)
            rec[f"gdev_{sig}_30d"] = s30.values.astype(np.float32)
            rec[f"gdev_{sig}_accel"] = ((s7 / 7) / ((s30 / 30) + EPS)).values.astype(np.float32)
        for sig in ["protest", "massviol"]:
            rec[f"gdev_{sig}_7d"] = roll(g[sig], 7).values.astype(np.float32)
            rec[f"gdev_{sig}_30d"] = roll(g[sig], 30).values.astype(np.float32)
        for sig in ["assault", "gov", "reb", "civ_target", "conf_mentions"]:
            rec[f"gdev_{sig}_7d"] = roll(g[sig], 7).values.astype(np.float32)
        # battles z-score vs 90d baseline
        b = g["battles"]; m90 = roll(b, 90, "mean"); sd90 = b.rolling(90, min_periods=5).std().fillna(0)
        rec["gdev_battles_z90"] = ((b - m90) / (sd90 + EPS)).values.astype(np.float32)
        out_parts.append(pd.DataFrame(rec))

    out = pd.concat(out_parts, ignore_index=True)
    feat = [c for c in out.columns if c.startswith("gdev_")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"저장: {OUT}  shape={out.shape}, 피처 {len(feat)}개")


if __name__ == "__main__":
    main()
