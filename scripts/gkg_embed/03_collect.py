"""
GKG Batch 결과 수집 (OpenAI Batch API → BigQuery + per-cd 집계 parquet)

전략 (디스크 절약):
  - 한 청크씩 처리: 다운로드 → 파싱 → BigQuery 적재 + 로컬 per-cd 평균 append → 임시파일 삭제
  - title-level 임베딩 = BigQuery 영속 저장 (`conflict-early-warning.conflict_ew.gkg_embeddings`)
  - per (country, date) 평균 + count = 로컬 parquet (input/processed/features/gkg_embeddings.parquet)
  - JSONL 입력 파일은 청크 성공 후 자동 삭제

BigQuery 스키마:
  date DATE, iso3 STRING, title STRING, embedding ARRAY<FLOAT64>

CLI:
  python scripts/gkg_embed/03_collect.py             # 1회 폴링 후 종료
  python scripts/gkg_embed/03_collect.py --watch     # 30분 간격 폴링 지속
  python scripts/gkg_embed/03_collect.py --keep-jsonl  # JSONL 자동 삭제 비활성
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from openai import OpenAI

load_dotenv(".env", override=True)

try:
    from dotenv import load_dotenv; load_dotenv(".env", override=True)
except Exception:
    pass
# GCP 자격증명은 GOOGLE_APPLICATION_CREDENTIALS 환경변수(.env)에서 주입

PRICE_PER_1M_TOK_BATCH = 0.01
EMBEDDING_DIM = 1536

# 02_submit.py 와 동일해야 함 — chunk 분할 로직
MAX_INPUTS_PER_BATCH = 49_000
MAX_TOKENS_PER_TITLE = 256
MAX_TOKENS_PER_BATCH = 2_500_000

STATE_PATH = Path("output/gkg_embeddings/state.json")
BATCH_DIR = Path("output/gkg_embeddings/batches")
AGG_PATH = Path("input/processed/features/gkg_embeddings.parquet")
EXTRACTED_PARQUET = Path("input/processed/gkg_titles/extracted.parquet")

GCP_PROJECT = os.getenv("GCP_PROJECT", "conflict-ew-mvp-20260604")
BQ_DATASET = os.getenv("BQ_DATASET", "conflict_ew")
BQ_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.gkg_embeddings"


def chunk_titles(df: pd.DataFrame) -> dict:
    """02_submit.py 의 chunk_titles 와 동일 — line-cap + token-cap union.

    base 키와 _t{N} sub-chunk 키를 모두 포함하므로 state 의 어느 상태든 cover.
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    base_chunks = {}
    for (iso3, year), group in df.groupby(["iso3", "year"], sort=True):
        group = group.reset_index(drop=True)
        n = len(group)
        base_key = f"{iso3}_{year}"
        if n <= MAX_INPUTS_PER_BATCH:
            base_chunks[base_key] = group
            continue
        n_sub = (n + MAX_INPUTS_PER_BATCH - 1) // MAX_INPUTS_PER_BATCH
        sub_size = (n + n_sub - 1) // n_sub
        for i in range(n_sub):
            suffix = chr(ord("a") + i)
            sub_key = f"{base_key}_{suffix}"
            base_chunks[sub_key] = group.iloc[i * sub_size : (i + 1) * sub_size].reset_index(drop=True)

    out = {}
    for key, group in base_chunks.items():
        out[key] = group
        n = len(group)
        char_sum = group["title"].str.len().sum()
        tokens_est = int(min(char_sum / 3.5, n * MAX_TOKENS_PER_TITLE))
        if tokens_est <= MAX_TOKENS_PER_BATCH:
            continue
        n_sub = (tokens_est + MAX_TOKENS_PER_BATCH - 1) // MAX_TOKENS_PER_BATCH
        sub_size = (n + n_sub - 1) // n_sub
        for j in range(n_sub):
            new_key = f"{key}_t{j+1}"
            out[new_key] = group.iloc[j * sub_size : (j + 1) * sub_size].reset_index(drop=True)
    return out


def load_state() -> dict:
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def ensure_bq_table(client: bigquery.Client) -> None:
    """BQ 테이블 없으면 생성. 기존 테이블이 비어있고 schema가 REQUIRED면 NULLABLE로 재생성."""
    try:
        existing = client.get_table(BQ_TABLE)
        # 기존 테이블이 비었고 REQUIRED 모드면 NULLABLE로 재생성 (pandas load 호환)
        if existing.num_rows == 0 and any(f.mode == "REQUIRED" for f in existing.schema):
            print(f"  기존 빈 테이블 (REQUIRED schema) drop 후 재생성")
            client.delete_table(BQ_TABLE)
        else:
            return
    except Exception:
        pass
    schema = [
        bigquery.SchemaField("date", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("iso3", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("title", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        bigquery.SchemaField("chunk_key", "STRING", mode="NULLABLE"),
    ]
    table = bigquery.Table(BQ_TABLE, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.MONTH, field="date"
    )
    table.clustering_fields = ["iso3"]
    client.create_table(table)
    print(f"  BQ 테이블 생성: {BQ_TABLE}")


def upload_chunk_to_bq(
    client: bigquery.Client,
    df: pd.DataFrame,
    chunk_key: str,
) -> int:
    """청크 DataFrame → BQ append. 반환: 적재 행 수."""
    upload_df = pd.DataFrame({
        "date": pd.to_datetime(df["date"]).dt.date,
        "iso3": df["iso3"].astype(str),
        "title": df["title"].astype(str),
        "embedding": df["embedding"].apply(lambda v: list(v) if v is not None else []),
        "chunk_key": chunk_key,
    })
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=[
            bigquery.SchemaField("date", "DATE"),
            bigquery.SchemaField("iso3", "STRING"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
            bigquery.SchemaField("chunk_key", "STRING"),
        ],
    )
    job = client.load_table_from_dataframe(upload_df, BQ_TABLE, job_config=job_config)
    job.result()  # block until done
    return len(upload_df)


def append_chunk_to_agg(df: pd.DataFrame) -> None:
    """per (country, date) 평균 + count 를 누적 parquet 에 append."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    emb_mat = np.stack(df["embedding"].values).astype(np.float32)

    df["__row"] = np.arange(len(df))
    grouped = df.groupby(["iso3", "date"])["__row"].apply(list).reset_index()

    means = np.zeros((len(grouped), EMBEDDING_DIM), dtype=np.float32)
    counts = np.zeros(len(grouped), dtype=np.int32)
    for i, idxs in enumerate(grouped["__row"]):
        means[i] = emb_mat[idxs].mean(axis=0)
        counts[i] = len(idxs)

    chunk_agg = pd.DataFrame({
        "date": grouped["date"].values,
        "country": grouped["iso3"].values,
        "gkg_emb_n_titles_1d": counts,
    })
    for d in range(EMBEDDING_DIM):
        chunk_agg[f"gkg_emb_{d}"] = means[:, d]

    AGG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if AGG_PATH.exists():
        existing = pd.read_parquet(AGG_PATH)
        merged = pd.concat([existing, chunk_agg], ignore_index=True)
        # 동일 (country, date) 중복 처리: 가중 평균 (n_titles 가중)
        # 같은 키가 여러 청크에서 나올 수 없으므로(연도별 분리) 단순 concat OK
        # 안전장치로 dedup
        merged = merged.drop_duplicates(subset=["country", "date"], keep="last")
    else:
        merged = chunk_agg
    merged.to_parquet(AGG_PATH, index=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="30분 간격 폴링 지속")
    ap.add_argument("--interval", type=int, default=1800)
    ap.add_argument("--keep-jsonl", action="store_true", help="JSONL 입력 파일 보존")
    return ap.parse_args()


def collect_once(
    openai_client: OpenAI,
    bq_client: bigquery.Client,
    state: dict,
    all_chunks: dict,
    keep_jsonl: bool,
) -> bool:
    submitted = [(k, v) for k, v in state["chunks"].items() if v["status"] == "submitted"]
    if not submitted:
        print("submitted 상태 청크 없음.")
        return False

    any_progress = False
    for key, meta in submitted:
        batch_id = meta["batch_id"]
        try:
            batch = openai_client.batches.retrieve(batch_id)
        except Exception as e:
            print(f"[{key}] retrieve 실패: {e}")
            continue
        status = batch.status
        rc = batch.request_counts
        print(f"[{key}] batch={batch_id[:24]}.. status={status} ({rc.completed}/{rc.total})")

        if status == "completed":
            output_file_id = batch.output_file_id
            content = None
            for attempt in range(1, 5):
                try:
                    raw = openai_client.files.content(output_file_id).read()
                except Exception as e:
                    print(f"  download attempt {attempt}/4 실패: {e}")
                    time.sleep(min(30 * 2 ** (attempt - 1), 240))
                    continue
                # 응답이 HTML 에러 페이지(Cloudflare 504 등)일 수 있어 검증
                head = raw[:200].lstrip()
                if head.startswith(b"<") or b"<html" in head.lower():
                    snippet = head[:120].decode("utf-8", errors="replace")
                    print(f"  download attempt {attempt}/4: HTML 에러 응답 (Cloudflare?) — '{snippet}...'")
                    time.sleep(min(30 * 2 ** (attempt - 1), 240))
                    continue
                content = raw
                break
            if content is None:
                print(f"  [SKIP] {key}: 다운로드 4회 모두 실패. 다음 폴링에서 재시도.")
                continue

            lines = content.decode("utf-8").strip().split("\n")
            del content

            chunk_df = all_chunks.get(key)
            if chunk_df is None:
                print(f"  [SKIP] {key}: all_chunks 에 없음 (state 마이그레이션 누락?)")
                continue

            records = []
            actual_tokens = 0
            for line in lines:
                obj = json.loads(line)
                cid = obj["custom_id"]
                offset = int(cid.rsplit("__", 1)[1])
                body = obj.get("response", {}).get("body", {})
                embeddings = body.get("data", [])
                actual_tokens += body.get("usage", {}).get("total_tokens", 0)
                for i, emb_obj in enumerate(embeddings):
                    row_idx = offset + i
                    if row_idx >= len(chunk_df):
                        continue
                    records.append({
                        "date": chunk_df.iloc[row_idx]["date"],
                        "iso3": chunk_df.iloc[row_idx]["iso3"],
                        "title": chunk_df.iloc[row_idx]["title"],
                        "embedding": np.array(emb_obj["embedding"], dtype=np.float32),
                    })
            del lines

            if not records:
                print(f"  [WARN] {key}: 임베딩 파싱 결과 0건")
                continue

            chunk_emb_df = pd.DataFrame(records)
            del records

            # 1) BigQuery 적재
            try:
                n_loaded = upload_chunk_to_bq(bq_client, chunk_emb_df, key)
                print(f"  BQ 적재: {n_loaded:,} 행")
            except Exception as e:
                print(f"  [ERROR] BQ 적재 실패: {e}")
                continue

            # 2) per-cd 집계 append
            append_chunk_to_agg(chunk_emb_df)
            n_cd = chunk_emb_df.groupby(["iso3", chunk_emb_df["date"].dt.normalize()]).size().shape[0]
            print(f"  per-cd 집계 append: {n_cd} country-days")
            del chunk_emb_df

            cost_actual = actual_tokens / 1_000_000 * PRICE_PER_1M_TOK_BATCH
            print(f"  실제 토큰: {actual_tokens:,} / ${cost_actual:.4f}")

            state["chunks"][key]["status"] = "done"
            state["chunks"][key]["tokens_actual"] = actual_tokens
            state["chunks"][key]["cost_actual"] = round(cost_actual, 4)
            state["chunks"][key]["completed_at"] = datetime.utcnow().isoformat()
            state["chunks"][key]["bq_loaded"] = True

            # 3) JSONL 입력 파일 정리
            if not keep_jsonl:
                jsonl_path = BATCH_DIR / f"{key}.jsonl"
                if jsonl_path.exists():
                    sz = jsonl_path.stat().st_size / 1e6
                    jsonl_path.unlink()
                    print(f"  JSONL 삭제: {jsonl_path.name} ({sz:.1f} MB)")

            save_state(state)
            any_progress = True

        elif status in ("failed", "expired", "cancelled"):
            print(f"  [FAIL] batch → {status}. pending 으로 복구.")
            state["chunks"][key]["status"] = "pending"
            state["chunks"][key]["batch_id"] = None
            save_state(state)
            any_progress = True

    state["total_cost_actual"] = round(
        sum(c["cost_actual"] for c in state["chunks"].values() if c.get("cost_actual")), 4
    )
    save_state(state)
    return any_progress


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    openai_client = OpenAI(api_key=api_key)
    bq_client = bigquery.Client(project=GCP_PROJECT)
    ensure_bq_table(bq_client)

    df_all = pd.read_parquet(EXTRACTED_PARQUET)
    print(f"추출 parquet: {len(df_all):,} 행")

    all_chunks = chunk_titles(df_all)
    print(f"chunk_titles() 결과: {len(all_chunks)} 청크 (sub-chunk 포함)")
    del df_all

    state = load_state()
    n_pending = sum(1 for v in state["chunks"].values() if v["status"] == "pending")
    n_submitted = sum(1 for v in state["chunks"].values() if v["status"] == "submitted")
    n_done = sum(1 for v in state["chunks"].values() if v["status"] == "done")
    n_failed = sum(1 for v in state["chunks"].values() if v["status"] == "failed")
    print(f"청크 상태: pending={n_pending}, submitted={n_submitted}, done={n_done}, failed={n_failed}")
    print(f"누적 실제 비용: ${state['total_cost_actual']:.4f}")

    if args.watch:
        while True:
            print(f"\n=== 폴링 {datetime.utcnow().isoformat()} ===")
            collect_once(openai_client, bq_client, state, all_chunks, args.keep_jsonl)
            remaining_sub = sum(1 for v in state["chunks"].values() if v["status"] == "submitted")
            remaining_pen = sum(1 for v in state["chunks"].values() if v["status"] == "pending")
            if remaining_sub == 0 and remaining_pen == 0:
                print("submitted + pending 모두 0 — 전체 완료.")
                break
            print(f"  남은 submitted: {remaining_sub}, pending: {remaining_pen}. {args.interval}s 대기.")
            time.sleep(args.interval)
    else:
        collect_once(openai_client, bq_client, state, all_chunks, args.keep_jsonl)


if __name__ == "__main__":
    main()
