"""#55 위험 anchor 가중합 시계열 (단일 위험 강도 score).

분쟁성 핵심 anchor 5종 cosine 평균 → 단일 "위험 의미 강도" 스칼라 → 시계열화.
해석 쉬운 단일 위험 score (대시보드 서브스코어로도 활용 가능).

산출 (date, country):
  gkg_risk_1d, gkg_risk_7dmean, gkg_risk_7dmax, gkg_risk_30dmean, gkg_risk_accel  (5)
출력: input/processed/features/gkg_emb_riskscore.parquet (build_dataset auto-join)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("input/processed/features/gkg_emb_anchors.parquet")
OUT = Path("input/processed/features/gkg_emb_riskscore.parquet")
RISK5 = ["armed_conflict", "civilian_casualties", "coup", "border_clash", "terrorism"]
EPS = 1e-6


def main() -> None:
    df = pd.read_parquet(SRC)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    risk_cols = [f"gkg_anchor_cos_{a}" for a in RISK5]
    df["__risk"] = df[risk_cols].mean(axis=1)
    print(f"입력 {df.shape}, 위험 anchor {len(risk_cols)}종")

    out_parts = []
    for country, g in df.groupby("country", sort=False):
        g = g.sort_values("date").set_index("date")[["__risk"]]
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D", tz="UTC")
        r = g["__risk"].reindex(full_idx, fill_value=0.0)
        m7 = r.rolling(7, min_periods=1).mean()
        m30 = r.rolling(30, min_periods=1).mean()
        rec = pd.DataFrame({
            "date": full_idx,
            "country": country,
            "gkg_risk_1d": r.values.astype(np.float32),
            "gkg_risk_7dmean": m7.values.astype(np.float32),
            "gkg_risk_7dmax": r.rolling(7, min_periods=1).max().values.astype(np.float32),
            "gkg_risk_30dmean": m30.values.astype(np.float32),
            "gkg_risk_accel": (m7 / (m30 + EPS)).values.astype(np.float32),
        })
        out_parts.append(rec)

    out = pd.concat(out_parts, ignore_index=True)
    feat = [c for c in out.columns if c.startswith("gkg_risk_")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"저장: {OUT}  shape={out.shape}, 피처 {len(feat)}개")


if __name__ == "__main__":
    main()
