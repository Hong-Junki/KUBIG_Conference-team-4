"""
GKG 제목 추출 (BigQuery → 로컬 parquet, top-K=200 per country-day)

전략:
  - post-2019-09-23 (PAGE_TITLE 가용 구간)
  - title NOT NULL AND != ''
  - 동일 (date, iso3, title) → COUNT(DISTINCT domain) 으로 popularity score
  - ROW_NUMBER() OVER (PARTITION BY date, iso3 ORDER BY domain_count DESC) → top 200

출력:
  input/processed/gkg_titles/extracted.parquet
  output/gkg_embeddings/extract_stats.json
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

try:
    from dotenv import load_dotenv; load_dotenv(".env", override=True)
except Exception:
    pass
# GCP 자격증명은 GOOGLE_APPLICATION_CREDENTIALS 환경변수(.env)에서 주입

TOP_K = 200
START_DATE = date(2019, 9, 23)
END_DATE = date(2026, 5, 29)

OUT_PARQUET = Path("input/processed/gkg_titles/extracted.parquet")
STATS_PATH = Path("output/gkg_embeddings/extract_stats.json")

from src.collect.config import COUNTRIES, GCP_PROJECT, GDELT_TITLES_FQN  # noqa: E402

TARGET_TABLE = GDELT_TITLES_FQN


def build_query(iso3_list: list[str]) -> str:
    iso3_array = ", ".join(f"'{i}'" for i in iso3_list)
    return f"""
    WITH grouped AS (
      SELECT
        date,
        iso3,
        title,
        COUNT(*) AS occurrences,
        COUNT(DISTINCT domain) AS domain_count,
        ANY_VALUE(language) AS language,
        AVG(v2tone_avg) AS v2tone_avg
      FROM `{TARGET_TABLE}`
      WHERE date BETWEEN DATE('{START_DATE}') AND DATE('{END_DATE}')
        AND iso3 IN ({iso3_array})
        AND title IS NOT NULL
        AND title != ''
        AND LENGTH(title) >= 10
      GROUP BY date, iso3, title
    ),
    ranked AS (
      SELECT *, ROW_NUMBER() OVER (
        PARTITION BY date, iso3
        ORDER BY domain_count DESC, occurrences DESC, title
      ) AS rn
      FROM grouped
    )
    SELECT date, iso3, title, domain_count, occurrences, language, v2tone_avg
    FROM ranked
    WHERE rn <= {TOP_K}
    """


def main() -> None:
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    iso3_list = [c["iso3"] for c in COUNTRIES]
    print(f"국가 수: {len(iso3_list)} / 기간: {START_DATE} ~ {END_DATE} / top-K: {TOP_K}")

    client = bigquery.Client(project=GCP_PROJECT)
    sql = build_query(iso3_list)

    # Dry-run 비용 추정
    dry_job = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    bytes_processed = dry_job.total_bytes_processed
    est_cost = bytes_processed / 1e12 * 5.0
    print(f"BigQuery scan 추정: {bytes_processed/1e9:.2f} GB / ~${est_cost:.2f}")

    # 실행
    print("쿼리 실행 중 (수 분 소요)...")
    job = client.query(sql)
    df = job.to_dataframe(create_bqstorage_client=True)
    print(f"행 수: {len(df):,}")

    # 타입 정규화
    df["date"] = pd.to_datetime(df["date"])
    df["title"] = df["title"].astype(str)
    df["iso3"] = df["iso3"].astype(str)

    # title 평균 char 통계
    title_chars = df["title"].str.len()
    print(f"title chars — avg {title_chars.mean():.1f} / p50 {title_chars.median():.0f} / p95 {title_chars.quantile(0.95):.0f} / max {title_chars.max()}")

    # 저장
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"저장 완료 → {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1e6:.1f} MB)")

    # 통계
    stats = {
        "total_rows": int(len(df)),
        "n_countries": int(df["iso3"].nunique()),
        "n_dates": int(df["date"].nunique()),
        "title_chars_avg": float(title_chars.mean()),
        "title_chars_p50": float(title_chars.median()),
        "title_chars_p95": float(title_chars.quantile(0.95)),
        "title_chars_max": int(title_chars.max()),
        "bq_bytes_processed": int(bytes_processed),
        "bq_cost_est_usd": float(est_cost),
        "start_date": str(START_DATE),
        "end_date": str(END_DATE),
        "top_k": TOP_K,
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"통계 → {STATS_PATH}")


if __name__ == "__main__":
    main()
