"""
GKG 제목 임베딩 Batch 제출 (OpenAI text-embedding-3-small Batch API)

설계:
  1) 추출된 parquet 을 (iso3, year) 청크로 분할
  2) 각 청크를 JSONL 파일로 변환 (요청당 N개 title)
  3) Batch API 제출, batch_id 저장
  4) state.json 에 청크별 상태 저장 (resumable)
  5) 예산 가드: 예상 누적 비용이 BUDGET_HARD_CAP 초과 시 중단

상태 파일 구조:
  output/gkg_embeddings/state.json
    {
      "chunks": {
        "UKR_2019": {"status": "pending|submitted|done|failed",
                     "tokens_est": int, "cost_est": float,
                     "batch_id": str|null, "tokens_actual": int|null,
                     "cost_actual": float|null, "n_titles": int,
                     "submitted_at": str|null, "completed_at": str|null},
        ...
      },
      "total_cost_est": float,
      "total_cost_actual": float,
      "budget_cap": float
    }

CLI:
  python scripts/gkg_embed/02_submit.py              # 다음 청크 1개 제출
  python scripts/gkg_embed/02_submit.py --all        # 예산 한도까지 전부 제출
  python scripts/gkg_embed/02_submit.py --dry-run    # 토큰 추정만, 제출 없음
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env", override=True)

# OpenAI text-embedding-3-small 가격 (Batch API 50% 할인 적용 가정)
MODEL = "text-embedding-3-small"
PRICE_PER_1M_TOK_BATCH = 0.01  # $0.01 per 1M tokens (Batch 할인 적용)
PRICE_PER_1M_TOK_STD = 0.02

# 예산 가드 (USD) — 누적 예상 cost 가 이 값 넘으면 중단
BUDGET_HARD_CAP = 45.0  # 총 한도 $50, $5 safety margin
BUDGET_SOFT_WARN = 35.0

# Batch API 한도 (OpenAI text-embedding-3-small)
MAX_TITLES_PER_REQUEST = 50  # 한 JSONL 라인당 title 수
MAX_INPUTS_PER_BATCH = 49_000  # OpenAI 한도 50_000, 1_000 safety margin
MAX_REQUESTS_PER_BATCH = MAX_INPUTS_PER_BATCH // MAX_TITLES_PER_REQUEST  # 980
MAX_TOKENS_PER_TITLE = 256  # text-embedding-3-small 입력 truncate
# 조직 enqueue cap 3M tokens — 안전 마진 500k 적용.
# 단일 batch 가 이를 넘으면 OpenAI 가 processing 시점에 failed 처리 → 영구 거절.
MAX_TOKENS_PER_BATCH = 2_500_000

# 입출력 경로
EXTRACTED_PARQUET = Path("input/processed/gkg_titles/extracted.parquet")
STATE_PATH = Path("output/gkg_embeddings/state.json")
BATCH_DIR = Path("output/gkg_embeddings/batches")


def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        "chunks": {},
        "total_cost_est": 0.0,
        "total_cost_actual": 0.0,
        "budget_cap": BUDGET_HARD_CAP,
    }


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _chunk_titles_line_only(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """기존 line-cap 분할 (호환용). 키 형식: IND_2025 또는 IND_2025_a/_b/..."""
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    chunks = {}
    for (iso3, year), group in df.groupby(["iso3", "year"], sort=True):
        group = group.reset_index(drop=True)
        n = len(group)
        base_key = f"{iso3}_{year}"
        if n <= MAX_INPUTS_PER_BATCH:
            chunks[base_key] = group
            continue
        n_sub = (n + MAX_INPUTS_PER_BATCH - 1) // MAX_INPUTS_PER_BATCH
        sub_size = (n + n_sub - 1) // n_sub
        for i in range(n_sub):
            suffix = chr(ord("a") + i)
            sub_key = f"{base_key}_{suffix}"
            chunks[sub_key] = group.iloc[i * sub_size : (i + 1) * sub_size].reset_index(drop=True)
    return chunks


def chunk_titles(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """(iso3, year) 청크 분할 → line-cap + token-cap 동시 적용.

    1단계: 기존 line-cap (49k) 분할 → IND_2025 또는 IND_2025_a/_b
    2단계: 각 청크의 토큰 추정이 MAX_TOKENS_PER_BATCH 초과 시 추가 분할 →
           IND_2025_t1/_t2 또는 IND_2025_a_t1/_a_t2
    base 키도 함께 보존 (done/submitted 상태와의 호환).
    """
    base = _chunk_titles_line_only(df)
    out: dict[str, pd.DataFrame] = {}
    for key, group in base.items():
        out[key] = group  # 기존 키 보존 (compat)
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


def _is_token_subchunk(key: str) -> tuple[bool, str]:
    """key 가 `_t{N}` 형식 sub-chunk 인지 판정. (yes, parent_key)."""
    parts = key.rsplit("_t", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return True, parts[0]
    return False, key


def estimate_tokens(titles: list[str], enc) -> int:
    """tiktoken 으로 정확한 토큰 카운트 (truncate 256 적용)."""
    total = 0
    for t in titles:
        toks = enc.encode(t[:1500])  # char-level pre-truncate (대략 500 토큰)
        total += min(len(toks), MAX_TOKENS_PER_TITLE)
    return total


def build_batch_jsonl(chunk_key: str, df: pd.DataFrame, enc) -> tuple[Path, int, int]:
    """JSONL 배치 파일 작성. 반환: (path, n_titles, total_tokens_est)."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    titles = df["title"].tolist()
    # truncate to MAX_TOKENS_PER_TITLE
    truncated = []
    total_tokens = 0
    for t in titles:
        toks = enc.encode(t[:1500])[:MAX_TOKENS_PER_TITLE]
        truncated.append(enc.decode(toks))
        total_tokens += len(toks)

    # 요청당 50 title 단위로 묶음
    requests = []
    for i in range(0, len(truncated), MAX_TITLES_PER_REQUEST):
        batch_titles = truncated[i : i + MAX_TITLES_PER_REQUEST]
        requests.append(
            {
                "custom_id": f"{chunk_key}__{i}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": MODEL,
                    "input": batch_titles,
                    "encoding_format": "float",
                },
            }
        )

    if len(requests) > MAX_REQUESTS_PER_BATCH:
        raise RuntimeError(
            f"{chunk_key}: 요청 수 {len(requests)} > {MAX_REQUESTS_PER_BATCH}. "
            "청크를 더 작게 쪼개야 함."
        )

    path = BATCH_DIR / f"{chunk_key}.jsonl"
    with open(path, "w") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    return path, len(titles), total_tokens


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="예산 한도까지 전부 제출")
    ap.add_argument("--dry-run", action="store_true", help="토큰 추정만, 제출 없음")
    ap.add_argument("--limit", type=int, default=1, help="--all 미지정 시 제출할 청크 개수")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print(f"추출 parquet 로드: {EXTRACTED_PARQUET}")
    df = pd.read_parquet(EXTRACTED_PARQUET)
    print(f"총 {len(df):,} 행")

    chunks = chunk_titles(df)
    print(f"청크 수: {len(chunks)} (iso3 × year)")

    state = load_state()
    enc = tiktoken.get_encoding("cl100k_base")  # text-embedding-3-small 토크나이저

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    client = None if args.dry_run else OpenAI(api_key=api_key)

    # 청크별 토큰/비용 사전 추정 (state 에 미기록인 청크만)
    # _t{N} sub-chunk 는 부모 키가 done/submitted 면 skip (이미 임베딩 완료)
    parent_done_or_sub = {
        k for k, v in state["chunks"].items() if v["status"] in ("done", "submitted")
    }
    for key, group in chunks.items():
        if key in state["chunks"]:
            continue
        is_tsub, parent = _is_token_subchunk(key)
        if is_tsub and parent in parent_done_or_sub:
            continue  # 부모가 이미 처리됨 — 토큰 sub-chunk 추가 불필요
        n_titles = len(group)
        char_sum = group["title"].str.len().sum()
        tokens_est = int(min(char_sum / 3.5, n_titles * MAX_TOKENS_PER_TITLE))
        cost_est = tokens_est / 1_000_000 * PRICE_PER_1M_TOK_BATCH
        state["chunks"][key] = {
            "status": "pending",
            "n_titles": int(n_titles),
            "tokens_est": tokens_est,
            "cost_est": round(cost_est, 4),
            "batch_id": None,
            "tokens_actual": None,
            "cost_actual": None,
            "submitted_at": None,
            "completed_at": None,
        }

    # 마이그레이션: pending 중 oversized 단일 키 삭제 (새 _t{N} 로 대체됨)
    removed = []
    for key in list(state["chunks"].keys()):
        meta = state["chunks"][key]
        if meta["status"] != "pending":
            continue
        is_tsub, _ = _is_token_subchunk(key)
        if is_tsub:
            continue  # _t{N} 자체는 항상 ≤ MAX_TOKENS_PER_BATCH
        if meta.get("tokens_est", 0) > MAX_TOKENS_PER_BATCH:
            removed.append(key)
            del state["chunks"][key]
            p = BATCH_DIR / f"{key}.jsonl"
            if p.exists():
                p.unlink()
    if removed:
        print(f"[마이그레이션] oversized pending {len(removed)}개 삭제 → _t{{N}} sub-chunk 로 대체됨")

    state["total_cost_est"] = round(sum(c["cost_est"] for c in state["chunks"].values()), 4)
    save_state(state)

    print(f"\n=== 사전 추정 (state.json 기반) ===")
    print(f"  청크 총: {len(state['chunks'])}")
    print(f"  추정 누적 비용: ${state['total_cost_est']:.2f}")
    print(f"  예산 cap: ${BUDGET_HARD_CAP:.2f}")

    if state["total_cost_est"] > BUDGET_HARD_CAP:
        print(f"\n[경고] 전체 추정 비용 ${state['total_cost_est']:.2f} > 예산 ${BUDGET_HARD_CAP:.2f}")
        print("       부분 제출 + 나머지는 로컬(BGE-M3 Colab) 전환 필요.")

    if args.dry_run:
        print("\n--dry-run: 제출 없이 종료.")
        return

    # 제출 우선순위: 최신 데이터부터 (2024 → 2019). 신호 가치 큰 순.
    pending_chunks = sorted(
        [(k, v) for k, v in state["chunks"].items() if v["status"] == "pending"],
        key=lambda x: x[0].split("_")[1],
        reverse=True,
    )

    if not pending_chunks:
        print("\n제출 대기 청크 없음.")
        return

    limit = len(pending_chunks) if args.all else args.limit
    submitted_count = 0
    cumulative_actual = state["total_cost_actual"]
    cumulative_est = sum(
        c["cost_actual"] if c["cost_actual"] is not None else c["cost_est"]
        for c in state["chunks"].values()
        if c["status"] in ("submitted", "done")
    )

    for key, meta in pending_chunks:
        if submitted_count >= limit:
            break

        # 예산 가드
        projected = cumulative_est + meta["cost_est"] * 1.2  # 20% safety margin
        if projected > BUDGET_HARD_CAP:
            print(f"\n[예산 차단] 다음 청크 제출 시 ${projected:.2f} > ${BUDGET_HARD_CAP:.2f}")
            print(f"  남은 청크: {len([k for k,v in state['chunks'].items() if v['status']=='pending'])}")
            print(f"  → 로컬(BGE-M3 Colab) 전환 권장.")
            break

        print(f"\n[{key}] {meta['n_titles']:,} titles, 추정 {meta['tokens_est']:,} tok / ${meta['cost_est']:.4f}")
        group = chunks[key]

        # JSONL 작성
        jsonl_path, n_titles, tokens_exact = build_batch_jsonl(key, group, enc)
        cost_exact = tokens_exact / 1_000_000 * PRICE_PER_1M_TOK_BATCH
        print(f"  JSONL: {jsonl_path} ({jsonl_path.stat().st_size/1e6:.1f} MB)")
        print(f"  정확 토큰: {tokens_exact:,} / ${cost_exact:.4f}")

        state["chunks"][key]["tokens_est"] = tokens_exact
        state["chunks"][key]["cost_est"] = round(cost_exact, 4)

        # 재계산 후 예산 가드 한 번 더
        projected = cumulative_est + cost_exact * 1.2
        if projected > BUDGET_HARD_CAP:
            print(f"  [예산 차단] 정확 토큰 기준 ${projected:.2f} > ${BUDGET_HARD_CAP:.2f}")
            save_state(state)
            break

        # 업로드 + Batch 생성
        try:
            with open(jsonl_path, "rb") as f:
                upload = client.files.create(file=f, purpose="batch")
            batch = client.batches.create(
                input_file_id=upload.id,
                endpoint="/v1/embeddings",
                completion_window="24h",
                metadata={"chunk": key},
            )
            print(f"  batch_id: {batch.id} / file_id: {upload.id}")

            state["chunks"][key]["status"] = "submitted"
            state["chunks"][key]["batch_id"] = batch.id
            state["chunks"][key]["file_id"] = upload.id
            state["chunks"][key]["submitted_at"] = datetime.utcnow().isoformat()
            cumulative_est += cost_exact
            state["total_cost_est"] = round(sum(c["cost_est"] for c in state["chunks"].values()), 4)
            save_state(state)
            submitted_count += 1

            if cumulative_est > BUDGET_SOFT_WARN:
                print(f"  [SOFT WARN] 누적 ${cumulative_est:.2f} > ${BUDGET_SOFT_WARN:.2f}")

            # rate limit 회피
            time.sleep(1)
        except Exception as e:
            print(f"  [ERROR] 제출 실패: {e}")
            state["chunks"][key]["status"] = "failed"
            save_state(state)

    print(f"\n=== 제출 완료 ===")
    print(f"  이번 세션 제출: {submitted_count} 청크")
    print(f"  누적 추정 비용: ${cumulative_est:.4f}")
    submitted = [k for k, v in state["chunks"].items() if v["status"] == "submitted"]
    done = [k for k, v in state["chunks"].items() if v["status"] == "done"]
    pending = [k for k, v in state["chunks"].items() if v["status"] == "pending"]
    print(f"  상태: submitted={len(submitted)}, done={len(done)}, pending={len(pending)}")
    print(f"\n다음: python scripts/gkg_embed/03_collect.py")


if __name__ == "__main__":
    main()
