"""#57 country baseline 임베딩 편차 (novelty, ACLED 밖 신호).

각 국가의 최근 30일(직전, 당일 제외) 평균 임베딩을 baseline 으로 두고,
당일 임베딩이 그 baseline 에서 얼마나 벗어났는지(L2 + cosine dissimilarity) 측정.
"평소와 다른 뉴스" = 신규 사건의 선행 신호. 만성 분쟁국(SDN/SYR)의 baseline
둔감 문제 보완 목표(2026-05-29 backtest 발견).

산출 (date, country):
  gkg_bdev_l2          당일 vs 직전30일평균 L2 거리
  gkg_bdev_cos         1 - cosine 유사도 (의미 이탈)
  gkg_bdev_l2_7dmean   L2 7d 평균
  gkg_bdev_accel       L2 7d/30d 비율 (이탈 가속)

누수 차단: baseline 은 직전 행들만(당일 제외, shift). 미래 미포함.
출력: input/processed/features/gkg_emb_baseline_dev.parquet (build_dataset auto-join)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("input/processed/features/gkg_embeddings.parquet")
OUT = Path("input/processed/features/gkg_emb_baseline_dev.parquet")
WIN, MINP, EPS = 30, 5, 1e-6


def main() -> None:
    df = pd.read_parquet(SRC)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    emb_cols = [c for c in df.columns if c.startswith("gkg_emb_") and c != "gkg_emb_n_titles_1d"]
    print(f"입력 {df.shape}, 임베딩 {len(emb_cols)}차원")

    out_parts = []
    for country, g in df.groupby("country", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        X = g[emb_cols].values.astype(np.float32)
        n = len(X)
        cs = np.vstack([np.zeros((1, X.shape[1]), dtype=np.float64), np.cumsum(X, axis=0)])
        dev_l2 = np.zeros(n, dtype=np.float32)
        dev_cos = np.zeros(n, dtype=np.float32)
        for i in range(n):
            lo = max(0, i - WIN)
            cnt = i - lo
            if cnt < MINP:
                continue
            base = (cs[i] - cs[lo]) / cnt
            diff = X[i] - base
            dev_l2[i] = np.sqrt(float((diff * diff).sum()))
            nb = float(np.linalg.norm(base)); nx = float(np.linalg.norm(X[i]))
            if nb > 0 and nx > 0:
                dev_cos[i] = 1.0 - float(X[i] @ base) / (nb * nx)
        s = pd.Series(dev_l2)
        rec = pd.DataFrame({
            "date": g["date"].values,
            "country": country,
            "gkg_bdev_l2": dev_l2,
            "gkg_bdev_cos": dev_cos,
            "gkg_bdev_l2_7dmean": s.rolling(7, min_periods=1).mean().values.astype(np.float32),
            "gkg_bdev_accel": (s.rolling(7, min_periods=1).mean()
                               / (s.rolling(30, min_periods=1).mean() + EPS)).values.astype(np.float32),
        })
        out_parts.append(rec)

    out = pd.concat(out_parts, ignore_index=True)
    feat = [c for c in out.columns if c.startswith("gkg_bdev_")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"저장: {OUT}  shape={out.shape}, 피처 {len(feat)}개")


if __name__ == "__main__":
    main()
