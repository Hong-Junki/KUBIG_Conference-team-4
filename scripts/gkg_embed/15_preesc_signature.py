"""#59 (E6) 사전-escalation 임베딩 시그니처 유사도 피처.

ACLED 흉내를 넘어선 ML-native 선행지표. "분쟁 터지기 직전의 뉴스는 어떻게 생겼나"를
train 데이터로 학습한다.

방법:
  1. train(<=2023-12-31)에서 각 (국가, t) 의 직전 14일 [t-14, t-1] 평균 임베딩 W 계산
  2. pos_sig = mean(W | y_escalation=1, train)   ← 사전징후 시그니처
     neg_sig = mean(W | y_escalation=0, train)   ← 평상시 baseline
     d = normalize(pos_sig - neg_sig)             ← "escalation 방향"
  3. 모든 (국가, 일) 의 당일/최근 임베딩을 d 에 투영 + pos_sig 와 cosine

산출 (date, country):
  gkg_preesc_proj_1d / _proj_7d     escalation 방향 투영 (당일 / 7d평균)
  gkg_preesc_cos_1d  / _cos_7d      pos_sig 와 cosine
  gkg_preesc_proj_accel            proj 7d/30d 비율

누수 차단: pos_sig/neg_sig/d 는 train 만으로 학습, 인과적(과거+당일) 적용.
출력: input/processed/features/gkg_emb_preesc.parquet (build_dataset auto-join)
      output/models/gkg_pca/preesc_signature.npz (실시간 재사용)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EMB_PATH = Path("input/processed/features/gkg_embeddings.parquet")
LABELS_PATH = Path("input/processed/labels/labels.parquet")
OUT = Path("input/processed/features/gkg_emb_preesc.parquet")
SIG_NPZ = Path("output/models/gkg_pca/preesc_signature.npz")
TRAIN_END = pd.Timestamp("2023-12-31", tz="UTC")
EPS = 1e-6


def trailing_mean(X: np.ndarray, k: int, include_current: bool) -> np.ndarray:
    n, d = X.shape
    cs = np.vstack([np.zeros((1, d)), np.cumsum(X.astype(np.float64), axis=0)])
    out = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        lo = max(0, i - k + 1) if include_current else max(0, i - k)
        hi = i + 1 if include_current else i
        cnt = hi - lo
        if cnt > 0:
            out[i] = (cs[hi] - cs[lo]) / cnt
    return out


def main() -> None:
    emb = pd.read_parquet(EMB_PATH)
    emb["date"] = pd.to_datetime(emb["date"], utc=True)
    lab = pd.read_parquet(LABELS_PATH)[["date", "country", "y_escalation"]]
    lab["date"] = pd.to_datetime(lab["date"], utc=True)
    df = emb.merge(lab, on=["date", "country"], how="left")
    emb_cols = [c for c in emb.columns if c.startswith("gkg_emb_") and c != "gkg_emb_n_titles_1d"]
    dim = len(emb_cols)
    print(f"입력 {df.shape}, dim {dim}")

    # Pass A: train W14 누적 → pos_sig / neg_sig
    pos_sum = np.zeros(dim); pos_cnt = 0
    neg_sum = np.zeros(dim); neg_cnt = 0
    per_country = {}
    for country, g in df.groupby("country", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        X = g[emb_cols].values.astype(np.float32)
        W14 = trailing_mean(X, 14, include_current=False)
        per_country[country] = (g, X)
        tr = (g["date"] <= TRAIN_END).values
        ye = g["y_escalation"].fillna(-1).values
        pos_m = tr & (ye == 1)
        neg_m = tr & (ye == 0)
        if pos_m.any():
            pos_sum += W14[pos_m].sum(axis=0); pos_cnt += int(pos_m.sum())
        if neg_m.any():
            neg_sum += W14[neg_m].sum(axis=0); neg_cnt += int(neg_m.sum())

    pos_sig = (pos_sum / max(pos_cnt, 1)).astype(np.float32)
    neg_sig = (neg_sum / max(neg_cnt, 1)).astype(np.float32)
    d_vec = pos_sig - neg_sig
    d_vec = d_vec / (np.linalg.norm(d_vec) + EPS)
    pos_n = np.linalg.norm(pos_sig) + EPS
    print(f"  pos events={pos_cnt}, neg events={neg_cnt}, ||pos-neg||={np.linalg.norm(pos_sig-neg_sig):.4f}")

    SIG_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SIG_NPZ, pos_sig=pos_sig, neg_sig=neg_sig, direction=d_vec)

    # Pass B: 피처
    out_parts = []
    for country, (g, X) in per_country.items():
        V1 = X
        V7 = trailing_mean(X, 7, include_current=True)
        proj1 = V1 @ d_vec
        proj7 = V7 @ d_vec
        cos1 = (V1 @ pos_sig) / (np.linalg.norm(V1, axis=1) * pos_n + EPS)
        cos7 = (V7 @ pos_sig) / (np.linalg.norm(V7, axis=1) * pos_n + EPS)
        proj7_s = pd.Series(proj7)
        accel = (proj7_s.rolling(7, min_periods=1).mean()
                 / (proj7_s.rolling(30, min_periods=1).mean().abs() + EPS)).values
        rec = pd.DataFrame({
            "date": g["date"].values, "country": country,
            "gkg_preesc_proj_1d": proj1.astype(np.float32),
            "gkg_preesc_proj_7d": proj7.astype(np.float32),
            "gkg_preesc_cos_1d": cos1.astype(np.float32),
            "gkg_preesc_cos_7d": cos7.astype(np.float32),
            "gkg_preesc_proj_accel": accel.astype(np.float32),
        })
        out_parts.append(rec)

    out = pd.concat(out_parts, ignore_index=True)
    feat = [c for c in out.columns if c.startswith("gkg_preesc_")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"저장: {OUT}  shape={out.shape}, 피처 {len(feat)}개")


if __name__ == "__main__":
    main()
