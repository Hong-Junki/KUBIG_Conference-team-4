"""
GKG 임베딩 → 일자×국가 피처 집계

전략:
  - input/processed/gkg_embeddings_raw/*.parquet 모두 로드
  - per (country, date): 평균 임베딩 (1536-dim)
  - 누수 차단: 당일 t 임베딩은 t 라벨에 직접 사용 가능 (제목은 당일 발생 뉴스 → t feature 로 인정)
    단 7d rolling mean 도 함께 생성 (gkg_emb_avg_7d_*) — features.parquet 패턴 일치
  - PCA 또는 차원 축소는 다음 단계에서 (1536-dim 전부 학습용으로 부담 → mean 후 64-dim 축소 고려)

출력:
  input/processed/features/gkg_embeddings.parquet
    columns: date, country, gkg_emb_{0..D-1}, gkg_emb_n_titles_1d, gkg_emb_n_titles_7d
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EMBEDDINGS_DIR = Path("input/processed/gkg_embeddings_raw")
OUT_PATH = Path("input/processed/features/gkg_embeddings.parquet")

EMBEDDING_DIM = 1536  # text-embedding-3-small


def main() -> None:
    files = sorted(EMBEDDINGS_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"임베딩 파일 없음: {EMBEDDINGS_DIR}")
    print(f"임베딩 파일 수: {len(files)}")

    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)
        dfs.append(df)
        print(f"  {fp.name}: {len(df):,} 행")
    df = pd.concat(dfs, ignore_index=True)
    print(f"총 {len(df):,} title 임베딩 로드")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # 임베딩 ndarray 화
    emb_mat = np.stack(df["embedding"].values).astype(np.float32)
    print(f"embedding matrix: {emb_mat.shape}")

    # per (iso3, date) 평균 + count
    df["__row"] = np.arange(len(df))
    grouped = df.groupby(["iso3", "date"])["__row"].apply(list).reset_index()
    print(f"고유 (country, date) 수: {len(grouped):,}")

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

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"저장 → {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")
    print(f"shape: {out.shape}")


if __name__ == "__main__":
    main()
