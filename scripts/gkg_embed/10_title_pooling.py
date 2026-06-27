"""#52 제목 단위 임베딩 → anchor pooling 피처 (BQ SQL).

기존 country-day 평균 임베딩은 위험 제목 1건이 평균에 묻힌다. 여기서는 BQ의
제목 단위 임베딩(gkg_embeddings)을 16 anchor와 cosine(ML.DISTANCE) 계산 후,
country-day 단위로 극값/카운트 pooling 하여 ACLED event_count 구조를 재현한다.
추가로 ACLED가 못 잡는 직교 신호(테마 폭/동시발화/미디어 집중도)도 산출한다.

산출 피처 (date, country):
  ACLED 흉내 (16 anchor × 5 = 80):
    gkg_pool_<anchor>_max / _p95 / _p99           극값 (위험 제목 1건 보존)
    gkg_pool_<anchor>_cnt1d / _cnt7d              train p90 임계 초과 제목 수 (event_count 대응)
  직교 신호 (약 6):
    gkg_pool_n_anchors_hot / _n_anchors_hot_7d    동시에 뜬 테마 수 (위기 폭)
    gkg_pool_core_conflict_min                    armed_conflict ∧ civilian_casualties 동시발화
    gkg_pool_top3_max_mean                        상위 3 anchor 평균 (전반 강도)
    gkg_pool_conflict_share_1d / _7d              분쟁 관련 제목 비율 (미디어 집중도)

누수 차단:
  - cnt 임계값(thr_p90)은 train 기간(<= TRAIN_END)만으로 산출, 전 기간 동일 적용
  - 모든 7d 윈도우는 [t-6, t] (미래 미포함). GKG=실시간이라 당일 t 제목 사용은 합법

재사용 substrate:
  conflict_ew._title_cos (제목 × anchor cosine, 약 4억 행) — 신규 anchor 피처는 여기서 추가 비용 거의 없이 파생 가능

사용법:
  python scripts/gkg_embed/10_title_pooling.py --dry-run   # 비용 추정만
  python scripts/gkg_embed/10_title_pooling.py             # 전체 실행
  python scripts/gkg_embed/10_title_pooling.py --keep-temp # 임시테이블 보존
"""

from __future__ import annotations

import argparse
import os
_PSS, _PSE = os.environ.get('SERVE_START'), os.environ.get('SERVE_END')
_POOL_WHERE = f"WHERE e.date BETWEEN DATE('{_PSS}') AND DATE('{_PSE}')" if _PSS and _PSE else ''
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

warnings.filterwarnings("ignore")
load_dotenv(".env", override=True)

GCP_PROJECT = os.getenv("GCP_PROJECT", "conflict-ew-mvp-20260604")
BQ_DATASET = os.getenv("BQ_DATASET", "conflict_ew")
DS = f"{GCP_PROJECT}.{BQ_DATASET}"

EMB_TBL = f"{DS}.gkg_embeddings"
ANCHOR_TBL = f"{DS}.gkg_anchor_vectors"
TITLE_COS = f"{DS}._title_cos"
THRESH = f"{DS}._anchor_thresh"
POOL_ANCHOR = f"{DS}._pool_anchor"
POOL_ORTHO = f"{DS}._pool_ortho"

ANCHOR_NPZ = Path("input/processed/features/anchor_embeddings.npz")
OUT_PATH = Path("input/processed/features/gkg_emb_title_pool.parquet")

TRAIN_END = "2023-12-31"
CORE5 = ["armed_conflict", "civilian_casualties", "terrorism", "border_clash", "coup"]


def ensure_anchor_table(client: bigquery.Client) -> list[str]:
    """anchor_embeddings.npz → BQ gkg_anchor_vectors (anchor_id, vec)."""
    d = np.load(ANCHOR_NPZ, allow_pickle=True)
    anchors = d["anchors"].astype(float)  # (16, 1536)
    labels = [str(x) for x in d["labels"]]
    df = pd.DataFrame({"anchor_id": labels, "vec": [list(map(float, v)) for v in anchors]})
    jc = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema=[bigquery.SchemaField("anchor_id", "STRING"),
                bigquery.SchemaField("vec", "FLOAT64", mode="REPEATED")],
    )
    client.load_table_from_dataframe(df, ANCHOR_TBL, job_config=jc).result()
    print(f"  anchor 테이블 적재: {ANCHOR_TBL} ({len(df)} anchors)")
    return labels


def run(client: bigquery.Client, sql: str, label: str) -> bigquery.QueryJob:
    job = client.query(sql)
    job.result()
    gb = (job.total_bytes_processed or 0) / 1e9
    print(f"  [{label}] 완료 (스캔 {gb:.1f} GB, ${gb/1000*6.25:.3f})")
    return job


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Step A 비용 추정만")
    ap.add_argument("--keep-temp", action="store_true", help="임시테이블 보존")
    ap.add_argument("--skip-compute", action="store_true", help="계산 건너뛰고 기존 임시테이블에서 다운로드만")
    args = ap.parse_args()

    client = bigquery.Client(project=GCP_PROJECT)

    # Step A SQL (가장 비싼 단계)
    sql_title_cos = f"""
    CREATE OR REPLACE TABLE `{TITLE_COS}` AS
    SELECT e.date, e.iso3,
           FARM_FINGERPRINT(e.title) AS title_id,
           a.anchor_id,
           (1 - ML.DISTANCE(e.embedding, a.vec, 'COSINE')) AS cos
    FROM `{EMB_TBL}` e
    CROSS JOIN `{ANCHOR_TBL}` a
    {_POOL_WHERE}
    """

    if args.dry_run:
        # anchor 테이블이 있어야 dry-run 가능 → 없으면 만들고 진행
        try:
            client.get_table(ANCHOR_TBL)
        except Exception:
            print("[dry-run] anchor 테이블 생성")
            ensure_anchor_table(client)
        job = client.query(sql_title_cos, job_config=bigquery.QueryJobConfig(dry_run=True))
        gb = job.total_bytes_processed / 1e9
        print(f"[dry-run] Step A 예상 스캔: {gb:.1f} GB  (~${gb/1000*6.25:.2f})")
        return

    if args.skip_compute:
        print("[skip-compute] 계산 건너뜀 — 기존 임시테이블에서 다운로드만")
    else:
        print("[0/5] anchor 벡터 테이블")
        ensure_anchor_table(client)

        print("[1/5] _title_cos (제목 × anchor cosine, 비싼 단계)")
        run(client, sql_title_cos, "title_cos")

    print("[2/5] _anchor_thresh (train p90)")
    run(client, f"""
    CREATE OR REPLACE TABLE `{THRESH}` AS
    SELECT iso3, anchor_id, APPROX_QUANTILES(cos, 100)[OFFSET(90)] AS thr_p90
    FROM `{TITLE_COS}`
    WHERE date <= DATE '{TRAIN_END}'
    GROUP BY iso3, anchor_id
    """, "thresh")

    print("[3/5] _pool_anchor (일별 극값/카운트 + 7d)")
    run(client, f"""
    CREATE OR REPLACE TABLE `{POOL_ANCHOR}` AS
    WITH daily AS (
      SELECT c.date, c.iso3, c.anchor_id,
             MAX(c.cos) AS cos_max,
             APPROX_QUANTILES(c.cos,100)[OFFSET(95)] AS cos_p95,
             APPROX_QUANTILES(c.cos,100)[OFFSET(99)] AS cos_p99,
             COUNTIF(c.cos > t.thr_p90) AS cnt1d
      FROM `{TITLE_COS}` c
      JOIN `{THRESH}` t USING (iso3, anchor_id)
      GROUP BY c.date, c.iso3, c.anchor_id
    )
    SELECT *,
      SUM(cnt1d) OVER (PARTITION BY iso3, anchor_id ORDER BY UNIX_DATE(date)
                       RANGE BETWEEN 6 PRECEDING AND CURRENT ROW) AS cnt7d
    FROM daily
    """, "pool_anchor")

    print("[4/5] _pool_ortho (직교 신호)")
    core_in = ",".join(f"'{a}'" for a in CORE5)
    run(client, f"""
    CREATE OR REPLACE TABLE `{POOL_ORTHO}` AS
    WITH pa AS (
      SELECT p.date, p.iso3, p.anchor_id, p.cos_max, p.cos_p95, t.thr_p90,
             ROW_NUMBER() OVER (PARTITION BY p.date,p.iso3 ORDER BY p.cos_max DESC) AS rnk
      FROM `{POOL_ANCHOR}` p JOIN `{THRESH}` t USING (iso3, anchor_id)
    ),
    hot AS (
      SELECT date, iso3,
        COUNTIF(cos_p95 > thr_p90) AS n_anchors_hot,
        AVG(IF(rnk<=3, cos_max, NULL)) AS top3_max_mean,
        LEAST(MAX(IF(anchor_id='armed_conflict', cos_max, NULL)),
              MAX(IF(anchor_id='civilian_casualties', cos_max, NULL))) AS core_conflict_min
      FROM pa GROUP BY date, iso3
    ),
    title_flag AS (
      SELECT c.date, c.iso3, c.title_id,
        MAX(CASE WHEN c.anchor_id IN ({core_in}) AND c.cos > t.thr_p90 THEN 1 ELSE 0 END) AS is_conflict
      FROM `{TITLE_COS}` c JOIN `{THRESH}` t USING (iso3, anchor_id)
      GROUP BY c.date, c.iso3, c.title_id
    ),
    shr AS (
      SELECT date, iso3, AVG(is_conflict) AS conflict_share_1d, COUNT(*) AS n_titles
      FROM title_flag GROUP BY date, iso3
    ),
    j AS (
      SELECT h.date, h.iso3, h.n_anchors_hot, h.top3_max_mean, h.core_conflict_min,
             s.conflict_share_1d, s.n_titles
      FROM hot h JOIN shr s USING (date, iso3)
    )
    SELECT *,
      MAX(n_anchors_hot) OVER w AS n_anchors_hot_7d,
      AVG(conflict_share_1d) OVER w AS conflict_share_7d
    FROM j
    WINDOW w AS (PARTITION BY iso3 ORDER BY UNIX_DATE(date) RANGE BETWEEN 6 PRECEDING AND CURRENT ROW)
    """, "pool_ortho")

    print("[5/5] 다운로드 + pivot")
    pa = client.query(f"SELECT * FROM `{POOL_ANCHOR}`").to_dataframe(create_bqstorage_client=False)
    ortho = client.query(f"SELECT * FROM `{POOL_ORTHO}`").to_dataframe(create_bqstorage_client=False)
    print(f"  pool_anchor {len(pa):,}행, pool_ortho {len(ortho):,}행")

    metrics = {"cos_max": "max", "cos_p95": "p95", "cos_p99": "p99", "cnt1d": "cnt1d", "cnt7d": "cnt7d"}
    wide = pa.pivot_table(index=["date", "iso3"], columns="anchor_id",
                          values=list(metrics.keys()))
    wide.columns = [f"gkg_pool_{anchor}_{metrics[m]}" for m, anchor in wide.columns]
    wide = wide.reset_index()

    ortho = ortho.rename(columns={
        "n_anchors_hot": "gkg_pool_n_anchors_hot",
        "n_anchors_hot_7d": "gkg_pool_n_anchors_hot_7d",
        "core_conflict_min": "gkg_pool_core_conflict_min",
        "top3_max_mean": "gkg_pool_top3_max_mean",
        "conflict_share_1d": "gkg_pool_conflict_share_1d",
        "conflict_share_7d": "gkg_pool_conflict_share_7d",
        "n_titles": "gkg_pool_n_titles",
    })

    out = wide.merge(ortho, on=["date", "iso3"], how="outer")
    out = out.rename(columns={"iso3": "country"})
    out["date"] = pd.to_datetime(out["date"])
    feat_cols = [c for c in out.columns if c.startswith("gkg_pool_")]
    out[feat_cols] = out[feat_cols].fillna(0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"저장: {OUT_PATH}  shape={out.shape}, 피처 {len(feat_cols)}개")

    if not args.keep_temp:
        for t in [TITLE_COS, THRESH, POOL_ANCHOR, POOL_ORTHO]:
            client.delete_table(t, not_found_ok=True)
        print("  임시테이블 정리 완료")


if __name__ == "__main__":
    main()
