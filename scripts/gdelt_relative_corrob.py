"""신호형태 1+2: 국가-상대 정규화(z) + 소스 동조(corroboration).

분쟁신호는 '수준'이 아니라 '그 나라 평소 대비 변화'다. 핵심 신호를 국가별
trailing 90d causal z-score로 정규화 → 만성 분쟁국 포화 해소. 그리고 여러
독립 소스가 동시에 비정상인 정도(corroboration)를 직교 신호로 추가.

입력: full_pca16_aclfree.parquet (이미 모든 소스 신호 병합됨, 국가-일 grid)
출력: input/processed/features/gdelt_relative_corrob.parquet (build_dataset auto-join)
  gcr_<signal>_z        국가 90d 대비 z (causal, shift(1))
  gcr_corrob_count      오늘 z>1.5 인 conflict 소스 수
  gcr_corrob_strength   sum(max(z,0)) 전체 이상 강도
  gcr_corrob_count_7d   7d 내 최대 동조 수
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("input/processed/dataset/full_pca16_aclfree.parquet")
OUT = Path("input/processed/features/gdelt_relative_corrob.parquet")
EPS = 1e-6
WIN = 90

SIGNALS = [
    "gdev_battles_7d", "gdev_remote_7d", "gdev_vac_7d", "gdev_goldneg_7d",
    "gdev_protest_7d", "gdev_massviol_7d", "gdev_conf_mentions_7d",
    "gdelt_sub_n_adm1_7d", "gkg_anchor_cos_armed_conflict",
    "gkg_anchor_cos_civilian_casualties", "gkg_emb_n_titles_1d",
]
# corroboration 용 conflict-direction 부분집합 (z 높을수록 위험)
CORROB = ["gdev_battles_7d", "gdev_vac_7d", "gdev_goldneg_7d", "gdev_protest_7d",
          "gdelt_sub_n_adm1_7d", "gkg_anchor_cos_armed_conflict", "gkg_emb_n_titles_1d"]


def compute_gcr(df: pd.DataFrame) -> pd.DataFrame:
    """조립된 데이터셋(date,country,+SIGNALS) → gcr_ 14피처 (국가 90d causal z + 동조).
    배치(main)·서빙(build_model_input) 공유. 입력에 SIGNALS 컬럼이 있어야 함(조립 후)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    present = [s for s in SIGNALS if s in df.columns]
    out_parts = []
    for country, g in df.groupby("country", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        rec = pd.DataFrame({"date": g["date"].values, "country": country})
        zmap = {}
        for s in present:
            x = g[s].astype(float)
            mean = x.shift(1).rolling(WIN, min_periods=30).mean()
            std = x.shift(1).rolling(WIN, min_periods=30).std()
            z = ((x - mean) / (std + EPS)).fillna(0.0).clip(-8, 8)
            rec[f"gcr_{s}_z"] = z.values.astype(np.float32)
            zmap[s] = z.values
        cz = np.vstack([zmap[s] for s in CORROB if s in zmap])  # (k, n)
        count = (cz > 1.5).sum(axis=0).astype(np.float32)
        strength = np.clip(cz, 0, None).sum(axis=0).astype(np.float32)
        rec["gcr_corrob_count"] = count
        rec["gcr_corrob_strength"] = strength
        rec["gcr_corrob_count_7d"] = pd.Series(count).rolling(7, min_periods=1).max().values.astype(np.float32)
        out_parts.append(rec)
    return pd.concat(out_parts, ignore_index=True)


def main() -> None:
    df = pd.read_parquet(SRC, columns=["date", "country"] + SIGNALS)
    present = [s for s in SIGNALS if s in df.columns]
    print(f"신호 {len(present)}/{len(SIGNALS)}개 발견: {present}")
    out = compute_gcr(df)
    feat = [c for c in out.columns if c.startswith("gcr_")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"저장: {OUT}  shape={out.shape}, 피처 {len(feat)}개")


if __name__ == "__main__":
    main()
