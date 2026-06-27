"""GKG 임베딩 보조 피처 생성 (B 강화 1차).

생성 피처:
  B1 시간차 (Δemb): country별 정렬 후 t vs t-3/t-7
    - gkg_emb_delta3_norm, gkg_emb_delta7_norm (L2 norm)
    - gkg_emb_cosdiss_3, gkg_emb_cosdiss_7 (1 - cosine_sim)
  B2 n_titles 시계열: country별 정렬 후
    - gkg_ntitles_delta3, gkg_ntitles_delta7
    - gkg_ntitles_max7d, gkg_ntitles_z30d
  B4 KMeans 거리 (k=30, PCA 64 데이터 사용)
    - gkg_cluster_dist_min, gkg_cluster_dist_mean
    - gkg_cluster_id (categorical→one-hot 30개 컬럼) ... LGBM split 효율 위해 distance 만 사용
    - gkg_cluster_dist_0 ~ gkg_cluster_dist_29 (각 centroid 까지 거리)

출력:
  input/processed/features/gkg_emb_aux.parquet  (date, country, B1+B2+B4 피처)
  output/models/gkg_pca/kmeans_30.pkl

사용법:
  python scripts/gkg_embed/07_aux_features.py
  python scripts/gkg_embed/07_aux_features.py --k 50
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

SRC_EMB_PATH = Path("input/processed/features/gkg_embeddings.parquet")
SRC_PCA_PATH = Path("input/processed/features/gkg_embeddings_pca64.parquet")
OUT_PATH = Path("input/processed/features/gkg_emb_aux.parquet")
KMEANS_DIR = Path("output/models/gkg_pca")
TRAIN_END = pd.Timestamp("2023-12-31", tz="UTC")


def compute_temporal_deltas(emb_df: pd.DataFrame, emb_cols: list[str]) -> pd.DataFrame:
    """country별 정렬 후 t vs t-3, t-7 비교 (B1 + B2 n_titles 시계열).

    NOTE: country-day 단위 임베딩이 매일 있는 게 아니므로(결측 일자 존재),
    먼저 (country, date) 로 정렬하고 country별 shift 사용.
    동일 country 내에서도 날짜 간격이 균일하지 않을 수 있어
    날짜를 인덱스로 reindex 후 처리하는 게 정확하지만, 1차 측정 우선이라 단순 shift 사용.
    """
    emb_df = emb_df.sort_values(["country", "date"]).reset_index(drop=True)
    out_records = []

    for country, g in emb_df.groupby("country", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        emb = g[emb_cols].values.astype(np.float32)
        n = len(g)

        rec = pd.DataFrame({"date": g["date"].values, "country": g["country"].values})

        for k in (3, 7):
            if n > k:
                diff = emb.copy()
                diff[k:] = emb[k:] - emb[:-k]
                diff[:k] = 0.0
                l2 = np.linalg.norm(diff, axis=1)
                rec[f"gkg_emb_delta{k}_norm"] = l2

                # cosine dissimilarity
                a = emb[k:]
                b = emb[:-k]
                na = np.linalg.norm(a, axis=1)
                nb = np.linalg.norm(b, axis=1)
                denom = np.clip(na * nb, 1e-12, None)
                cos = (a * b).sum(axis=1) / denom
                cosdiss = np.empty(n, dtype=np.float32)
                cosdiss[:k] = 0.0
                cosdiss[k:] = (1.0 - cos).astype(np.float32)
                rec[f"gkg_emb_cosdiss_{k}"] = cosdiss
            else:
                rec[f"gkg_emb_delta{k}_norm"] = 0.0
                rec[f"gkg_emb_cosdiss_{k}"] = 0.0

        # B2 n_titles 시계열
        n_titles = g["gkg_emb_n_titles_1d"].values.astype(np.float32)
        for k in (3, 7):
            if n > k:
                d = n_titles.copy()
                d[k:] = n_titles[k:] - n_titles[:-k]
                d[:k] = 0.0
                rec[f"gkg_ntitles_delta{k}"] = d
            else:
                rec[f"gkg_ntitles_delta{k}"] = 0.0
        # rolling max 7
        rmax = pd.Series(n_titles).rolling(7, min_periods=1).max().values.astype(np.float32)
        rec["gkg_ntitles_max7d"] = rmax
        # z-score 30
        rmean = pd.Series(n_titles).rolling(30, min_periods=5).mean().values.astype(np.float32)
        rstd = pd.Series(n_titles).rolling(30, min_periods=5).std().fillna(1.0).values.astype(np.float32)
        rstd = np.clip(rstd, 1e-3, None)
        z = ((n_titles - rmean) / rstd).astype(np.float32)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        rec["gkg_ntitles_z30d"] = z

        out_records.append(rec)

    return pd.concat(out_records, ignore_index=True)


def compute_kmeans_features(pca_df: pd.DataFrame, pca_cols: list[str], k: int, seed: int) -> tuple[pd.DataFrame, MiniBatchKMeans]:
    """PCA 64 데이터로 KMeans(k) 학습 → 전 row 의 각 centroid 까지 거리."""
    X = pca_df[pca_cols].values.astype(np.float32)
    train_mask = (pca_df["date"] <= TRAIN_END).values
    X_train = X[train_mask]
    print(f"  KMeans fit data: {X_train.shape}")

    km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=seed, n_init=5, max_iter=300)
    km.fit(X_train)

    distances = km.transform(X).astype(np.float32)  # (N, k)
    rec = pd.DataFrame({"date": pca_df["date"].values, "country": pca_df["country"].values})
    for i in range(k):
        rec[f"gkg_cluster_dist_{i}"] = distances[:, i]
    rec["gkg_cluster_dist_min"] = distances.min(axis=1)
    rec["gkg_cluster_dist_mean"] = distances.mean(axis=1)
    return rec, km


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30, help="KMeans 클러스터 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[1/3] 임베딩 로드 (raw 1536)")
    emb = pd.read_parquet(SRC_EMB_PATH)
    emb["date"] = pd.to_datetime(emb["date"], utc=True)
    emb_cols = [c for c in emb.columns if c.startswith("gkg_emb_") and c != "gkg_emb_n_titles_1d"]
    print(f"  shape={emb.shape}, dim={len(emb_cols)}")

    print(f"[2/3] B1 시간차 + B2 n_titles 시계열 계산")
    temporal = compute_temporal_deltas(emb, emb_cols)
    print(f"  temporal shape={temporal.shape}")

    print(f"[3/3] B4 KMeans(k={args.k}) on PCA 64")
    pca = pd.read_parquet(SRC_PCA_PATH)
    pca["date"] = pd.to_datetime(pca["date"], utc=True)
    pca_cols = [c for c in pca.columns if c.startswith("gkg_emb_pca_")]
    cluster, km = compute_kmeans_features(pca, pca_cols, k=args.k, seed=args.seed)

    # merge temporal + cluster
    aux = temporal.merge(cluster, on=["date", "country"], how="inner")
    print(f"  aux shape={aux.shape}")

    aux.to_parquet(OUT_PATH, index=False)
    print(f"저장: {OUT_PATH}")

    KMEANS_DIR.mkdir(parents=True, exist_ok=True)
    km_path = KMEANS_DIR / f"kmeans_{args.k}.pkl"
    with open(km_path, "wb") as f:
        pickle.dump({"kmeans": km, "pca_cols": pca_cols, "k": args.k, "seed": args.seed}, f)
    print(f"KMeans 모델 저장: {km_path}")


if __name__ == "__main__":
    main()
