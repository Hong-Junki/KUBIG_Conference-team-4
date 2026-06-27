"""GKG 제목 동기(Synchronous) 임베딩 — 속도 최우선.

Batch API 폐기. text-embedding-3-small 동기 endpoint 를 ThreadPoolExecutor 로
병렬 호출. 한 chunk(=한 iso3,year) 의 titles 를 N개씩 묶어 worker 풀로 처리,
chunk 완료 시 BigQuery + per-cd 집계 + state.json 즉시 마킹 (resumable).

기존 state.json 호환: done/submitted(Batch 기 처리분) 건너뛰고 pending 만 처리.

가격:
  text-embedding-3-small 동기 API = $0.02 / 1M tok (Batch 50% 할인 미적용)

CLI:
  python scripts/gkg_embed/05_sync_embed.py                       # 전체 pending
  python scripts/gkg_embed/05_sync_embed.py --workers 16
  python scripts/gkg_embed/05_sync_embed.py --titles-per-req 100
  python scripts/gkg_embed/05_sync_embed.py --limit 1             # 1-chunk 실측
  python scripts/gkg_embed/05_sync_embed.py --dry-run             # 큐 분석만
  python scripts/gkg_embed/05_sync_embed.py --finalize-agg        # 부분 agg 병합만
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from google.cloud import bigquery
from openai import OpenAI

load_dotenv(".env", override=True)

try:
    from dotenv import load_dotenv; load_dotenv(".env", override=True)
except Exception:
    pass
# GCP 자격증명은 GOOGLE_APPLICATION_CREDENTIALS 환경변수(.env)에서 주입

MODEL = "text-embedding-3-small"
PRICE_PER_1M_TOK_SYNC = 0.02
EMBEDDING_DIM = 1536
MAX_TOKENS_PER_TITLE = 256

EXTRACTED_PARQUET = Path("input/processed/gkg_titles/extracted.parquet")
STATE_PATH = Path("output/gkg_embeddings/state.json")
AGG_PATH = Path("input/processed/features/gkg_embeddings.parquet")
AGG_PARTIAL_DIR = Path("output/gkg_embeddings/agg_partials")
GCP_PROJECT = os.getenv("GCP_PROJECT", "conflict-ew-mvp-20260604")
BQ_DATASET = os.getenv("BQ_DATASET", "conflict_ew")
BQ_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.gkg_embeddings"

# 02/03 분할 상수와 동일 — state key 호환 필수
MAX_INPUTS_PER_BATCH = 49_000
MAX_TOKENS_PER_BATCH = 2_500_000


# -----------------------------
# chunk 분할 (02/03 과 동일 union 로직)
# -----------------------------
def chunk_titles(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    base: dict[str, pd.DataFrame] = {}
    for (iso3, year), group in df.groupby(["iso3", "year"], sort=True):
        group = group.reset_index(drop=True)
        n = len(group)
        base_key = f"{iso3}_{year}"
        if n <= MAX_INPUTS_PER_BATCH:
            base[base_key] = group
            continue
        n_sub = (n + MAX_INPUTS_PER_BATCH - 1) // MAX_INPUTS_PER_BATCH
        sub_size = (n + n_sub - 1) // n_sub
        for i in range(n_sub):
            sub_key = f"{base_key}_{chr(ord('a') + i)}"
            base[sub_key] = group.iloc[i * sub_size : (i + 1) * sub_size].reset_index(drop=True)

    out: dict[str, pd.DataFrame] = {}
    for key, group in base.items():
        out[key] = group
        n = len(group)
        char_sum = group["title"].str.len().sum()
        tokens_est = int(min(char_sum / 3.5, n * MAX_TOKENS_PER_TITLE))
        if tokens_est <= MAX_TOKENS_PER_BATCH:
            continue
        n_sub = (tokens_est + MAX_TOKENS_PER_BATCH - 1) // MAX_TOKENS_PER_BATCH
        sub_size = (n + n_sub - 1) // n_sub
        for j in range(n_sub):
            out[f"{key}_t{j+1}"] = group.iloc[j * sub_size : (j + 1) * sub_size].reset_index(drop=True)
    return out


# -----------------------------
# state IO (atomic write)
# -----------------------------
_state_lock = threading.Lock()


def load_state() -> dict:
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    with _state_lock:
        tmp = STATE_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        tmp.replace(STATE_PATH)


# -----------------------------
# BigQuery
# -----------------------------
def ensure_bq_table(client: bigquery.Client) -> None:
    try:
        existing = client.get_table(BQ_TABLE)
        if existing.num_rows == 0 and any(f.mode == "REQUIRED" for f in existing.schema):
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


def upload_chunk_to_bq(client: bigquery.Client, df: pd.DataFrame, chunk_key: str) -> int:
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
    job.result()
    return len(upload_df)


# per-chunk per-cd 집계를 partial 파일로 즉시 저장 (resumable + 빠름).
# 모든 chunk 처리 후 finalize_agg() 가 partial 들을 concat → AGG_PATH.
def save_chunk_partial_agg(df: pd.DataFrame, chunk_key: str) -> None:
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

    out = pd.DataFrame({
        "date": grouped["date"].values,
        "country": grouped["iso3"].values,
        "gkg_emb_n_titles_1d": counts,
    })
    for d in range(EMBEDDING_DIM):
        out[f"gkg_emb_{d}"] = means[:, d]

    AGG_PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(AGG_PARTIAL_DIR / f"{chunk_key}.parquet", index=False)


def finalize_agg() -> None:
    """partial parquet 들을 (country,date) 가중평균으로 병합 → AGG_PATH."""
    files = sorted(AGG_PARTIAL_DIR.glob("*.parquet"))
    if not files:
        print("[finalize_agg] partial 없음.", flush=True)
        return
    print(f"[finalize_agg] {len(files)} partial 로드", flush=True)

    parts = [pd.read_parquet(f) for f in files]
    combined = pd.concat(parts, ignore_index=True)
    print(f"[finalize_agg] concat rows={len(combined):,}", flush=True)

    # 같은 (country,date) 가 여러 partial 에 있을 수 있음 → n_titles 가중평균
    emb_cols = [c for c in combined.columns if c.startswith("gkg_emb_") and c != "gkg_emb_n_titles_1d"]
    n_col = "gkg_emb_n_titles_1d"

    def _wmean(g: pd.DataFrame) -> pd.Series:
        w = g[n_col].values
        wsum = w.sum()
        out = {n_col: int(wsum)}
        # 1536 emb cols 가중평균
        vals = g[emb_cols].values  # (k, 1536)
        wm = (vals * w[:, None]).sum(axis=0) / wsum
        for c, v in zip(emb_cols, wm.astype(np.float32)):
            out[c] = v
        return pd.Series(out)

    agg = combined.groupby(["country", "date"], as_index=False).apply(_wmean).reset_index(drop=True)
    # column order
    agg = agg[["date", "country", n_col] + emb_cols]

    AGG_PATH.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(AGG_PATH, index=False)
    print(f"[finalize_agg] 저장: {AGG_PATH} ({len(agg):,} country-days)", flush=True)


# -----------------------------
# Embeddings call with backoff
# -----------------------------
class RateLimitState:
    """전역 429 카운터. 연속 다발 429 시 전체 worker sleep."""

    def __init__(self):
        self.lock = threading.Lock()
        self.consecutive_429 = 0
        self.global_sleep_until = 0.0
        self.total_429 = 0

    def hit_429(self):
        with self.lock:
            self.consecutive_429 += 1
            self.total_429 += 1
            if self.consecutive_429 >= 3:
                self.global_sleep_until = time.time() + 30.0
                self.consecutive_429 = 0

    def hit_ok(self):
        with self.lock:
            if self.consecutive_429 > 0:
                self.consecutive_429 = max(0, self.consecutive_429 - 1)

    def wait_if_throttled(self):
        with self.lock:
            wait = self.global_sleep_until - time.time()
        if wait > 0:
            time.sleep(wait)


def embed_request(
    client: OpenAI,
    titles: list[str],
    rl: RateLimitState,
    max_attempts: int = 6,
) -> tuple[list[list[float]], int]:
    """한 동기 요청. 429/5xx 자동 지수 백오프."""
    rl.wait_if_throttled()
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.embeddings.create(model=MODEL, input=titles, encoding_format="float")
            rl.hit_ok()
            embs = [d.embedding for d in resp.data]
            usage = resp.usage.total_tokens if resp.usage else 0
            return embs, usage
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "rate" in msg:
                rl.hit_429()
                sleep = min(2 ** attempt, 60) + attempt * 0.5
            elif any(c in msg for c in ("500", "502", "503", "504", "timeout", "connection")):
                sleep = min(2 ** attempt, 30)
            else:
                sleep = min(2 ** attempt, 20)
            time.sleep(sleep)
    raise RuntimeError(f"embed_request 실패 {max_attempts}회: {last_err}")


# -----------------------------
# Progress
# -----------------------------
class Progress:
    def __init__(self, total_chunks: int, total_tokens_est: int):
        self.total_chunks = total_chunks
        self.total_tokens_est = max(total_tokens_est, 1)
        self.done_chunks = 0
        self.done_titles = 0
        self.done_tokens = 0
        self.start = time.time()
        self.lock = threading.Lock()
        self.stop_flag = False

    def add(self, titles: int, tokens: int):
        with self.lock:
            self.done_titles += titles
            self.done_tokens += tokens

    def add_chunk(self):
        with self.lock:
            self.done_chunks += 1

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = time.time() - self.start
            tps = self.done_tokens / elapsed if elapsed > 0 else 0
            remain = max(self.total_tokens_est - self.done_tokens, 0)
            eta = remain / tps if tps > 0 else float("inf")
            return {
                "chunks": f"{self.done_chunks}/{self.total_chunks}",
                "titles": self.done_titles,
                "tokens": self.done_tokens,
                "tok_per_sec": tps,
                "elapsed_h": elapsed / 3600,
                "eta_h": eta / 3600,
                "cost_so_far": self.done_tokens / 1_000_000 * PRICE_PER_1M_TOK_SYNC,
            }


def progress_reporter(p: Progress, rl: RateLimitState, interval: float = 5.0):
    while not p.stop_flag:
        time.sleep(interval)
        s = p.snapshot()
        print(
            f"[progress] chunk={s['chunks']} titles={s['titles']:,} "
            f"tok={s['tokens']/1e6:.2f}M tok/s={s['tok_per_sec']:.0f} "
            f"elapsed={s['elapsed_h']:.2f}h eta={s['eta_h']:.2f}h "
            f"cost=${s['cost_so_far']:.2f} total_429={rl.total_429}",
            flush=True,
        )


# -----------------------------
# Chunk 처리
# -----------------------------
def process_chunk(
    key: str,
    df: pd.DataFrame,
    enc: tiktoken.Encoding,
    client: OpenAI,
    rl: RateLimitState,
    workers: int,
    titles_per_req: int,
    progress: Progress,
) -> tuple[pd.DataFrame, int]:
    titles = df["title"].tolist()
    truncated: list[str] = []
    for t in titles:
        toks = enc.encode(t[:1500])[:MAX_TOKENS_PER_TITLE]
        truncated.append(enc.decode(toks))

    batches: list[tuple[int, list[str]]] = [
        (i, truncated[i : i + titles_per_req])
        for i in range(0, len(truncated), titles_per_req)
    ]

    embeddings: list[list[float] | None] = [None] * len(truncated)
    total_tokens = 0
    tokens_lock = threading.Lock()
    failed_batches = 0

    def worker(item):
        offset, batch_titles = item
        embs, usage = embed_request(client, batch_titles, rl)
        return offset, embs, usage

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, b) for b in batches]
        for fut in as_completed(futures):
            try:
                offset, embs, usage = fut.result()
            except Exception as e:
                failed_batches += 1
                print(f"  [{key}] batch 실패: {e}", flush=True)
                continue
            for j, e in enumerate(embs):
                embeddings[offset + j] = e
            with tokens_lock:
                total_tokens += usage
            progress.add(titles=len(embs), tokens=usage)

    if failed_batches > 0:
        raise RuntimeError(f"{failed_batches}/{len(batches)} batch 실패")

    result = pd.DataFrame({
        "date": df["date"].values,
        "iso3": df["iso3"].values,
        "title": df["title"].values,
        "embedding": [
            np.asarray(e, dtype=np.float32) if e is not None else None for e in embeddings
        ],
    })
    missing = result["embedding"].isna().sum()
    if missing:
        # 정상 흐름에선 발생 안 함 (실패 시 위에서 raise)
        print(f"  [{key}] WARN: {missing} 행 embedding 누락 → drop", flush=True)
        result = result.dropna(subset=["embedding"]).reset_index(drop=True)
    return result, total_tokens


# -----------------------------
# main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8, help="동시 worker 수")
    ap.add_argument("--titles-per-req", type=int, default=100, help="한 요청당 titles")
    ap.add_argument("--limit", type=int, default=0, help="처리할 chunk 수 (0=전체)")
    ap.add_argument("--dry-run", action="store_true", help="큐 분석만")
    ap.add_argument("--progress-interval", type=float, default=5.0)
    ap.add_argument("--finalize-agg", action="store_true",
                    help="기 저장된 partial agg 만 병합 → AGG_PATH 후 종료")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.finalize_agg:
        finalize_agg()
        return

    print(f"[init] 추출 parquet 로드: {EXTRACTED_PARQUET}", flush=True)
    df_all = pd.read_parquet(EXTRACTED_PARQUET)
    print(f"[init] 총 {len(df_all):,} 행", flush=True)

    chunks = chunk_titles(df_all)
    print(f"[init] chunk_titles: {len(chunks)} 청크", flush=True)
    del df_all

    state = load_state()
    pending = [(k, v) for k, v in state["chunks"].items() if v["status"] == "pending"]
    submitted = [k for k, v in state["chunks"].items() if v["status"] == "submitted"]
    done = [k for k, v in state["chunks"].items() if v["status"] == "done"]
    failed = [k for k, v in state["chunks"].items() if v["status"] == "failed"]
    print(
        f"[init] state: done={len(done)} submitted={len(submitted)} "
        f"pending={len(pending)} failed={len(failed)}",
        flush=True,
    )

    # 큰 chunk 먼저 (worker pool 활용도 ↑, 마지막에 작은 chunk 정리)
    pending.sort(key=lambda kv: kv[1].get("tokens_est", 0), reverse=True)
    if args.limit > 0:
        pending = pending[: args.limit]

    total_tok_est = sum(v.get("tokens_est", 0) for _, v in pending)
    print(f"[init] 처리 대상: {len(pending)} chunks, est {total_tok_est/1e6:.1f}M tok, "
          f"예상 비용 ${total_tok_est/1_000_000 * PRICE_PER_1M_TOK_SYNC:.2f}", flush=True)
    print(f"[init] 설정: workers={args.workers}, titles/req={args.titles_per_req}, "
          f"progress interval={args.progress_interval}s", flush=True)

    if args.dry_run:
        print("--dry-run: 종료", flush=True)
        return

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY (또는 LLM_API_KEY) 미설정")
    client = OpenAI(api_key=api_key)
    bq_client = bigquery.Client(project=GCP_PROJECT)
    ensure_bq_table(bq_client)
    enc = tiktoken.get_encoding("cl100k_base")
    rl = RateLimitState()

    progress = Progress(total_chunks=len(pending), total_tokens_est=total_tok_est)
    reporter = threading.Thread(
        target=progress_reporter, args=(progress, rl, args.progress_interval), daemon=True
    )
    reporter.start()

    n_ok = 0
    n_err = 0
    for i, (key, meta) in enumerate(pending, 1):
        df = chunks.get(key)
        if df is None:
            print(f"[{i}/{len(pending)}] {key}: chunks 에 없음 → skip", flush=True)
            continue

        t0 = time.time()
        print(
            f"\n[{i}/{len(pending)}] {key} 시작: {len(df):,} titles, "
            f"est {meta.get('tokens_est',0)/1e6:.2f}M tok",
            flush=True,
        )
        try:
            emb_df, total_tokens = process_chunk(
                key, df, enc, client, rl,
                workers=args.workers,
                titles_per_req=args.titles_per_req,
                progress=progress,
            )
        except Exception as e:
            print(f"  [{key}] process_chunk 실패: {e}", flush=True)
            state["chunks"][key]["status"] = "failed"
            state["chunks"][key]["last_error"] = str(e)[:300]
            save_state(state)
            n_err += 1
            continue

        try:
            n_loaded = upload_chunk_to_bq(bq_client, emb_df, key)
            print(f"  [{key}] BQ 적재: {n_loaded:,} 행", flush=True)
        except Exception as e:
            print(f"  [{key}] BQ 적재 실패: {e} → failed 마킹", flush=True)
            state["chunks"][key]["status"] = "failed"
            state["chunks"][key]["last_error"] = f"bq: {str(e)[:300]}"
            save_state(state)
            n_err += 1
            continue

        try:
            save_chunk_partial_agg(emb_df, key)
        except Exception as e:
            # agg 는 BQ 에서 재집계 가능하므로 done 유지
            print(f"  [{key}] partial agg 저장 실패 (BQ 적재는 완료): {e}", flush=True)

        cost = total_tokens / 1_000_000 * PRICE_PER_1M_TOK_SYNC
        elapsed = time.time() - t0
        state["chunks"][key]["status"] = "done"
        state["chunks"][key]["tokens_actual"] = total_tokens
        state["chunks"][key]["cost_actual"] = round(cost, 4)
        state["chunks"][key]["completed_at"] = datetime.utcnow().isoformat()
        state["chunks"][key]["bq_loaded"] = True
        state["chunks"][key]["sync_processed"] = True
        state["total_cost_actual"] = round(
            sum(c["cost_actual"] for c in state["chunks"].values() if c.get("cost_actual")), 4
        )
        save_state(state)
        progress.add_chunk()
        n_ok += 1
        print(
            f"  [{key}] DONE: {total_tokens:,} tok / ${cost:.4f} / "
            f"{elapsed:.1f}s ({total_tokens/elapsed:.0f} tok/s)",
            flush=True,
        )

    progress.stop_flag = True
    time.sleep(0.5)
    s = progress.snapshot()
    print(f"\n=== 최종 ===", flush=True)
    print(f"  성공: {n_ok} / 실패: {n_err} / 처리 대상: {len(pending)}", flush=True)
    print(f"  처리 토큰: {s['tokens']/1e6:.2f}M / 비용: ${s['cost_so_far']:.4f}", flush=True)
    print(f"  소요: {s['elapsed_h']:.2f}h, 평균 throughput: {s['tok_per_sec']:.0f} tok/s", flush=True)
    print(f"  total_429: {rl.total_429}", flush=True)
    print(f"\n다음: python scripts/gkg_embed/05_sync_embed.py --finalize-agg", flush=True)


if __name__ == "__main__":
    main()
