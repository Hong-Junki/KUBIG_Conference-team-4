"""BQ엔 적재됐으나 country-day 집계(parquet)에서 누락된 (iso3, year) 보충.

배경:
  05_sync_embed 가 BQ 적재 성공 후 partial agg 저장이 실패해도 done 으로 마킹
  (주석: "agg 는 BQ 에서 재집계 가능"). 그러나 finalize_agg 는 partial 파일만
  읽으므로, partial 누락 청크는 gkg_embeddings.parquet 에서 빠진다.
  (관측: state done 894 vs partial 860, 18개 (iso3,year) 가 parquet 결측)

동작:
  1. BQ gkg_embeddings 의 (iso3, year) 커버리지 vs 로컬 parquet 커버리지 대조
  2. "BQ엔 있는데 parquet엔 없는" (iso3, year) 자동 탐지
  3. 해당 청크만 BQ에서 끌어와 country-day 평균 → agg_partials/{iso3}_{year}_refill.parquet
  4. (이후) python scripts/gkg_embed/05_sync_embed.py --finalize-agg 로 전체 재병합

사용법:
  python scripts/gkg_embed/09_refill_missing_agg.py            # 탐지 + 보충
  python scripts/gkg_embed/09_refill_missing_agg.py --dry-run  # 탐지만
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

warnings.filterwarnings("ignore")
load_dotenv(".env", override=True)
try:
    from dotenv import load_dotenv; load_dotenv(".env", override=True)
except Exception:
    pass
# GCP 자격증명은 GOOGLE_APPLICATION_CREDENTIALS 환경변수(.env)에서 주입

GCP_PROJECT = os.getenv("GCP_PROJECT", "conflict-ew-mvp-20260604")
BQ_DATASET = os.getenv("BQ_DATASET", "conflict_ew")
BQ_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.gkg_embeddings"
AGG_PATH = Path("input/processed/features/gkg_embeddings.parquet")
AGG_PARTIAL_DIR = Path("output/gkg_embeddings/agg_partials")
EMBEDDING_DIM = 1536


def detect_missing(client: bigquery.Client) -> list[tuple[str, int]]:
    """BQ엔 있는데 parquet엔 없는 (iso3, year) 목록."""
    q = f"""
    SELECT iso3, EXTRACT(YEAR FROM date) AS yr, COUNT(*) AS n
    FROM `{BQ_TABLE}`
    GROUP BY iso3, yr
    """
    bq = client.query(q).result().to_dataframe()
    bq_set = {(r["iso3"], int(r["yr"])) for _, r in bq.iterrows()}

    emb = pd.read_parquet(AGG_PATH, columns=["date", "country"])
    emb["date"] = pd.to_datetime(emb["date"], utc=True)
    emb["yr"] = emb["date"].dt.year
    pq_set = {(r["country"], int(r["yr"])) for _, r in emb[["country", "yr"]].drop_duplicates().iterrows()}

    missing = sorted(bq_set - pq_set)
    return missing


def refill_chunk(client: bigquery.Client, iso3: str, year: int) -> int:
    """한 (iso3, year) BQ 임베딩 → country-day 평균 partial 저장."""
    q = f"""
    SELECT date, iso3, embedding
    FROM `{BQ_TABLE}`
    WHERE iso3 = '{iso3}' AND EXTRACT(YEAR FROM date) = {year}
    """
    df = client.query(q).result().to_dataframe(create_bqstorage_client=True)
    if df.empty:
        return 0
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    emb_mat = np.stack(df["embedding"].values).astype(np.float32)

    df["__row"] = np.arange(len(df))
    grouped = df.groupby(["iso3", "date"])["__row"].apply(list).reset_index()
    means = np.zeros((len(grouped), EMBEDDING_DIM), dtype=np.float32)
    counts = np.zeros(len(grouped), dtype=np.int32)
    for i, idxs in enumerate(grouped["__row"]):
        means[i] = emb_mat[idxs].mean(axis=0)
        counts[i] = len(idxs)

    out = pd.DataFrame({
        "date": grouped["date"].values,
        "country": grouped["iso3"].values,
        "gkg_emb_n_titles_1d": counts,
    })
    for d in range(EMBEDDING_DIM):
        out[f"gkg_emb_{d}"] = means[:, d]

    AGG_PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(AGG_PARTIAL_DIR / f"{iso3}_{year}_refill.parquet", index=False)
    return len(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = bigquery.Client(project=GCP_PROJECT)
    missing = detect_missing(client)
    print(f"BQ엔 있는데 parquet엔 없는 (iso3, year): {len(missing)}개")
    for iso3, year in missing:
        print(f"  {iso3}_{year}")

    if args.dry_run or not missing:
        print("dry-run 또는 보충 대상 없음 → 종료")
        return

    print("\n=== 보충 집계 시작 ===")
    total_cd = 0
    for i, (iso3, year) in enumerate(missing, 1):
        n = refill_chunk(client, iso3, year)
        total_cd += n
        print(f"  [{i}/{len(missing)}] {iso3}_{year}: {n} country-days 집계 저장")
    print(f"\n총 {total_cd} country-day partial 저장 완료")
    print("다음: python scripts/gkg_embed/05_sync_embed.py --finalize-agg")


if __name__ == "__main__":
    main()
