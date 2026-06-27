"""서빙 스코어링 잡 — BQ model_input → onset 점수 → BQ model_scores.

GitHub Actions/Cloud Run 스케줄에서 주기 실행:
  read model_input(최신 또는 윈도우) → score() → write model_scores
스코어링 담당이 model_scores 를 읽어 risk_score(0~100)+티어 계산 → Supabase → 대시보드.

env: GCP_PROJECT / BQ_DATASET. 추론 CPU. run_ts 는 호출측 주입(스크립트 시간 비결정 회피).
"""
from __future__ import annotations
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GCP_PROJECT = os.getenv("GCP_PROJECT", "conflict-ew-mvp-20260604")
BQ_DATASET = os.getenv("BQ_DATASET", "conflict_ew")
MODEL_INPUT_TBL = f"{GCP_PROJECT}.{BQ_DATASET}.model_input"
MODEL_SCORES_TBL = f"{GCP_PROJECT}.{BQ_DATASET}.model_scores"


def read_model_input(lookback_days: int = 60, target_date: str | None = None) -> pd.DataFrame:
    """BQ model_input 에서 최신 run + (윈도우=시퀀스 lookback) 읽기.
    시퀀스 윈도우(45일) 때문에 target_date 이전 lookback_days 도 함께 로드."""
    from google.cloud import bigquery
    client = bigquery.Client(project=GCP_PROJECT)
    where = ""
    if target_date:
        where = (f"WHERE date BETWEEN DATE_SUB(DATE '{target_date}', INTERVAL {lookback_days} DAY) "
                 f"AND DATE '{target_date}'")
    sql = f"""
      SELECT * EXCEPT(run_ts) FROM `{MODEL_INPUT_TBL}`
      {where}
      QUALIFY ROW_NUMBER() OVER (PARTITION BY country, date ORDER BY run_ts DESC) = 1
    """
    return client.query(sql).to_dataframe(create_bqstorage_client=False)


def write_model_scores(scored: pd.DataFrame, run_ts: str):
    """model_scores 적재 (country, date, run_ts, base_pred, onset_prob, calm_flag)."""
    from google.cloud import bigquery
    out = scored.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["run_ts"] = run_ts
    client = bigquery.Client(project=GCP_PROJECT)
    client.load_table_from_dataframe(
        out[["country", "date", "run_ts", "base_pred", "onset_prob", "calm_flag"]],
        MODEL_SCORES_TBL,
        job_config=bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND),
    ).result()
    print(f"  적재 {len(out)}행 → {MODEL_SCORES_TBL}")


def main(run_ts: str, target_date: str | None = None, model_dir: str | None = None):
    import sys; sys.path.insert(0, str(ROOT))
    from src.serve.score import score, MODEL_DIR
    print(f"[1] model_input 읽기 (target={target_date or '전체'})")
    df = read_model_input(target_date=target_date)
    print(f"    {df.shape}")
    print("[2] onset 스코어링")
    res = score(df, model_dir or MODEL_DIR, target_dates=[target_date] if target_date else None)
    print(f"    {len(res)}행 | onset_prob 평균 {res.onset_prob.mean():.3f} | calm {res.calm_flag.sum()}")
    print(f"[3] model_scores 적재")
    write_model_scores(res, run_ts)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ts", required=True, help="실행 타임스탬프 (ISO, 호출측 주입)")
    ap.add_argument("--target-date", default=None, help="스코어 대상일(없으면 전체)")
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args()
    main(args.run_ts, args.target_date, args.model_dir)
