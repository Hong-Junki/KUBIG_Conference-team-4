"""GKG 임베딩 1536 → PCA 64 차원 축소.

목적:
  1536 차원 / 양성 4.29% 과적합 회피, 학습 시간 단축, 차원 폭발 방지.

데이터 누수 방지:
  PCA 는 unsupervised 지만, train period (~2023-12-31) 임베딩만으로 fit 후
  전체 기간 transform — 분포 shift 영향 차단.

출력:
  input/processed/features/gkg_embeddings_pca{N}.parquet
  output/models/gkg_pca/pca_{N}.pkl  (실시간 추론 시 동일 변환 적용용)

사용법:
  python scripts/gkg_embed/06_pca_reduce.py             # 기본 n_components=64
  python scripts/gkg_embed/06_pca_reduce.py --n 128
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SRC_PATH = Path("input/processed/features/gkg_embeddings.parquet")
OUT_DIR_FEAT = Path("input/processed/features")
OUT_DIR_MODEL = Path("output/models/gkg_pca")
TRAIN_END = pd.Timestamp("2023-12-31", tz="UTC")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="PCA 차원 (기본 64)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(SRC_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    emb_cols = [c for c in df.columns if c.startswith("gkg_emb_") and c != "gkg_emb_n_titles_1d"]
    assert len(emb_cols) == 1536, f"expected 1536 emb dims, got {len(emb_cols)}"
    print(f"입력: {df.shape}, 임베딩 차원: {len(emb_cols)}")

    train_mask = df["date"] <= TRAIN_END
    X_train = df.loc[train_mask, emb_cols].values.astype(np.float32)
    print(f"PCA fit 데이터: train period {train_mask.sum()} rows (≤ {TRAIN_END.date()})")

    pca = PCA(n_components=args.n, svd_solver="randomized", random_state=args.seed)
    pca.fit(X_train)
    cum_var = pca.explained_variance_ratio_.cumsum()
    print(f"PCA fit 완료. 누적 설명 분산: top1={cum_var[0]:.3f} / top10={cum_var[9]:.3f} / top{args.n}={cum_var[-1]:.3f}")

    X_all = df[emb_cols].values.astype(np.float32)
    X_reduced = pca.transform(X_all).astype(np.float32)
    print(f"transform 완료: {X_reduced.shape}")

    out_cols = [f"gkg_emb_pca_{i}" for i in range(args.n)]
    out_df = pd.DataFrame(X_reduced, columns=out_cols)
    out_df.insert(0, "country", df["country"].values)
    out_df.insert(0, "date", df["date"].values)
    out_df["gkg_emb_n_titles_1d"] = df["gkg_emb_n_titles_1d"].values

    out_path = OUT_DIR_FEAT / f"gkg_embeddings_pca{args.n}.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"저장: {out_path} ({out_df.shape})")

    OUT_DIR_MODEL.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR_MODEL / f"pca_{args.n}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"pca": pca, "emb_cols": emb_cols, "n_components": args.n, "seed": args.seed}, f)
    print(f"PCA 모델 저장: {model_path}")


if __name__ == "__main__":
    main()
