"""#53 anchor cosine(country-day 평균) 시계열 피처.

기존 gkg_emb_anchors.parquet(16 anchor cosine, country-day 평균)를 시계열화.
각 anchor 별로 7d/30d 평균·표준편차 + 가속도(7d/30d 비율) → ACLED rolling stat 대응.
#52(제목 단위 극값/카운트, 1d·7d)와 상보: 여기는 30d·std·가속도 담당.

산출 (date, country):
  gkg_anchT_<anchor>_r7m / _r7s / _r30m / _r30s / _accel   (16 × 5 = 80)
    r7m/r30m: 7d/30d 평균,  r7s/r30s: 7d/30d 표준편차,  accel: r7m/(r30m+eps)

누수 차단:
  rolling은 trailing(과거+당일, 미래 미포함). 당일 t 제목은 t 시점 가용(GKG=실시간).
  결측일(뉴스 없음)은 0 fill = "그날 신호 없음"(ACLED no-event 대응).

출력: input/processed/features/gkg_emb_anchors_temporal.parquet (build_dataset auto-join)

사용법:
  python scripts/gkg_embed/11_anchor_temporal.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("input/processed/features/gkg_emb_anchors.parquet")
OUT = Path("input/processed/features/gkg_emb_anchors_temporal.parquet")
EPS = 1e-6


def main() -> None:
    df = pd.read_parquet(SRC)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    anchor_cols = [c for c in df.columns if c.startswith("gkg_anchor_cos_")]
    print(f"입력: {df.shape}, anchor {len(anchor_cols)}개")

    out_parts = []
    for country, g in df.groupby("country", sort=False):
        g = g.sort_values("date").set_index("date")
        # 일 단위 grid 재색인 (결측일 0) → 정확한 7d/30d 윈도우
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D", tz="UTC")
        g = g[anchor_cols].reindex(full_idx, fill_value=0.0)

        rec = pd.DataFrame(index=g.index)
        for c in anchor_cols:
            a = c.replace("gkg_anchor_cos_", "")
            s = g[c]
            r7m = s.rolling(7, min_periods=1).mean()
            r30m = s.rolling(30, min_periods=1).mean()
            rec[f"gkg_anchT_{a}_r7m"] = r7m.astype(np.float32)
            rec[f"gkg_anchT_{a}_r7s"] = s.rolling(7, min_periods=1).std().fillna(0).astype(np.float32)
            rec[f"gkg_anchT_{a}_r30m"] = r30m.astype(np.float32)
            rec[f"gkg_anchT_{a}_r30s"] = s.rolling(30, min_periods=1).std().fillna(0).astype(np.float32)
            rec[f"gkg_anchT_{a}_accel"] = (r7m / (r30m + EPS)).astype(np.float32)

        rec["country"] = country
        rec = rec.reset_index().rename(columns={"index": "date"})
        out_parts.append(rec)

    out = pd.concat(out_parts, ignore_index=True)
    feat_cols = [c for c in out.columns if c.startswith("gkg_anchT_")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"저장: {OUT}  shape={out.shape}, 피처 {len(feat_cols)}개")


if __name__ == "__main__":
    main()
