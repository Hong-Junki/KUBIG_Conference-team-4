"""서빙 오케스트레이터 — raw(BQ) → 419 model_input(BQ). 로컬 parquet 미경유(BQ↔메모리↔BQ).

흐름:
  1. raw 수집 파이프라인이 BQ에 적재: gdelt_processed_events / gdelt_titles / economic_daily.
  2. base 피처(gdelt_·econ_): bq_sources 로 BQ→메모리 DataFrame → feature_builder(save=False).
     (ACLED 는 aclfree라 0, build_dataset 단계서 drop)
  3. 그 외 그룹(BQ-SQL/Python 빌더, 매 run 최신 산출):
     - 01_extract+05_sync_embed → gkg_embeddings(평균) + BQ 제목임베딩  → 임베딩 파생 214
     - gkg_feature_builder → Track1 17 / 10_title_pooling → pool 87
     - gdelt_enriched_events → gdev / gdelt_subnational / gdelt_acled_mirror
  4. embedding_derived(214) + assemble(419 parity) → model_input
  5. gcr_(상대/동조 14)는 assemble 후 파생 → 합류
  6. BQ model_input 적재.

검증: 2(base BQ↔parquet) parity·4(assemble 419) parity 로컬 통과. 전체 run 은 실데이터 1회 검증.
env: GCP_PROJECT / BQ_DATASET / OPENAI_API_KEY.
"""
from __future__ import annotations
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "input/processed/features"
GCP_PROJECT = os.getenv("GCP_PROJECT", "conflict-ew-mvp-20260604")
BQ_DATASET = os.getenv("BQ_DATASET", "conflict_ew")
MODEL_INPUT_TBL = f"{GCP_PROJECT}.{BQ_DATASET}.model_input"

# base(gdelt_/econ_) 외 그룹을 만드는 BQ-SQL/Python 빌더 (매 run 최신 산출, 임시 parquet).
GROUP_BUILDERS = [
    [sys.executable, "scripts/gkg_embed/01_extract.py"],
    [sys.executable, "scripts/gkg_embed/05_sync_embed.py"],
    [sys.executable, "-m", "src.process.gkg_feature_builder"],
    [sys.executable, "scripts/gkg_embed/10_title_pooling.py"],
    [sys.executable, "scripts/gdelt_enriched_events.py"],
    [sys.executable, "scripts/gdelt_subnational.py"],
    [sys.executable, "scripts/gdelt_acled_mirror.py"],
]


def run_group_builders(env_extra: dict | None = None):
    env = {**os.environ, **(env_extra or {})}
    for cmd in GROUP_BUILDERS:
        print(f"  $ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def build_features(start: str, end: str, write_bq: bool = True,
                   groups_ready: bool = False, run_ts: str | None = None) -> pd.DataFrame:
    """BQ raw → 419 model_input. start/end: 'YYYY-MM-DD' (rolling lookback 포함해 넉넉히)."""
    sys.path.insert(0, str(ROOT))
    from src.process import feature_builder
    from src.serve import bq_sources
    from src.serve.embedding_features import load_artifacts, embedding_derived
    from src.serve.feature_pipeline import (assemble, JOIN_ORDER, EMB_DERIVED_FILES,
                                             GCR_FILE, add_gcr, _norm_date)

    print("[1] base 피처: BQ(gdelt_processed_events·economic_daily) → 메모리 → feature_builder")
    gdelt_by_iso3 = bq_sources.read_gdelt_events(start, end)
    econ_df = bq_sources.read_economic(start, end)
    base = feature_builder.build_features(
        date.fromisoformat(start), date.fromisoformat(end),
        gdelt_by_iso3=gdelt_by_iso3, econ_df=econ_df, save=False)

    if not groups_ready:
        print("[2] 그 외 그룹 빌더 실행(임베딩 적재 + Track1/pool/gdev/subnational/mirror)")
        run_group_builders({"SERVE_START": start, "SERVE_END": end})  # 윈도잉(전체스캔 방지)

    print("[3] 임베딩 파생 214 (저장 아티팩트 transform)")
    art = load_artifacts(ROOT)
    emb = _norm_date(pd.read_parquet(FEAT / "gkg_embeddings.parquet"))
    emb_derived = embedding_derived(emb, art)

    print("[4] build_dataset 조립 (left-join + acled drop)")
    groups, inserted = [], False
    for fn in JOIN_ORDER:
        if fn in EMB_DERIVED_FILES:
            if not inserted:
                groups.append(emb_derived); inserted = True
            continue
        if fn == GCR_FILE:
            continue  # gcr_ 는 조립 후 계산
        p = FEAT / fn
        if p.exists():
            groups.append(pd.read_parquet(p))
    model_input = assemble(base, groups, drop_acled=True)
    model_input = add_gcr(model_input, ROOT)   # gcr_(상대/동조 14) post-assembly 합류
    print(f"    model_input: {model_input.shape}")

    if write_bq:
        print(f"[5] BQ 적재 → {MODEL_INPUT_TBL}")
        _write_bq(model_input, run_ts)
    return model_input


def _write_bq(df: pd.DataFrame, run_ts: str | None):
    from google.cloud import bigquery
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    if run_ts is not None:
        out["run_ts"] = run_ts
    bigquery.Client(project=GCP_PROJECT).load_table_from_dataframe(
        out, MODEL_INPUT_TBL,
        job_config=bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND)).result()
    print(f"    적재 {len(out)}행 → {MODEL_INPUT_TBL}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (rolling lookback 포함)")
    ap.add_argument("--end", required=True)
    ap.add_argument("--run-ts", default=None)
    ap.add_argument("--groups-ready", action="store_true")
    ap.add_argument("--no-bq", action="store_true")
    args = ap.parse_args()
    df = build_features(args.start, args.end, write_bq=not args.no_bq,
                        groups_ready=args.groups_ready, run_ts=args.run_ts)
    print(f"\n완료. model_input {df.shape}")
