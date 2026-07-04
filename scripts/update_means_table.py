"""일배치: 새 날 GKG 임베딩 → BQ `gkg_embeddings_means` 테이블 갱신.

15분 스코어 잡(컨테이너 features/score)은 이 테이블에서 country-day 평균 임베딩을 읽는다.
임베딩 자체는 비싸서 15분마다가 아니라 하루 1회(또는 몇 회) 여기서 갱신한다.

흐름: 01_extract(윈도우) → 05_sync_embed(증분, EMBED_KEY_SUFFIX) → finalize
      → 윈도우 means 를 gkg_embeddings_means 에 upsert(겹치는 date 삭제 후 append, 재실행 안전).

사용(Cloud Run Job, 일 1회):
  python scripts/update_means_table.py --start 2026-07-03 --end 2026-07-04 --suffix inc20260704
env: GCP_PROJECT / BQ_DATASET / OPENAI_API_KEY / GOOGLE_APPLICATION_CREDENTIALS.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env", override=True)
from google.cloud import bigquery  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GCP_PROJECT = os.getenv("GCP_PROJECT", "conflict-ew-mvp-20260604")
BQ_DATASET = os.getenv("BQ_DATASET", "conflict_ew")
TBL = f"{GCP_PROJECT}.{BQ_DATASET}.gkg_embeddings_means"
MEANS_PARQUET = ROOT / "input/processed/features/gkg_embeddings.parquet"
EMB_COLS = [f"gkg_emb_{d}" for d in range(1536)]


def _run(cmd: list[str], env: dict) -> None:
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def main(start: str, end: str, suffix: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT),
           "SERVE_START": start, "SERVE_END": end, "EMBED_KEY_SUFFIX": suffix}
    print(f"[1] 증분 임베딩 {start}~{end} (suffix={suffix})", flush=True)
    _run([sys.executable, "scripts/gkg_embed/01_extract.py"], env)
    _run([sys.executable, "scripts/gkg_embed/05_sync_embed.py"], env)
    _run([sys.executable, "scripts/gkg_embed/05_sync_embed.py", "--finalize-agg"], env)

    print("[2] 윈도우 means → gkg_embeddings_means upsert (배열 컬럼)", flush=True)
    df = pd.read_parquet(MEANS_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    w = df[(df["date"] >= start) & (df["date"] <= end)]
    out = pd.DataFrame({
        "date": w["date"].dt.date,
        "country": w["country"].astype(str),
        "gkg_emb_n_titles_1d": w["gkg_emb_n_titles_1d"].astype("int64"),
        "embedding": w[EMB_COLS].to_numpy(np.float32).tolist(),
    })
    c = bigquery.Client(project=GCP_PROJECT)
    c.query(f"DELETE FROM `{TBL}` WHERE date BETWEEN DATE('{start}') AND DATE('{end}')").result()
    c.load_table_from_dataframe(out, TBL, job_config=bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=[bigquery.SchemaField("date", "DATE"),
                bigquery.SchemaField("country", "STRING"),
                bigquery.SchemaField("gkg_emb_n_titles_1d", "INT64"),
                bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED")])).result()
    print(f"✅ upsert {len(out)}행 ({start}~{end}) → {TBL}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--suffix", required=True, help="EMBED_KEY_SUFFIX (유니크, 예: inc20260704)")
    a = ap.parse_args()
    main(a.start, a.end, a.suffix)
