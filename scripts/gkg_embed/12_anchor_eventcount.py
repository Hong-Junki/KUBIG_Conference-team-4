"""#54 anchor cosine(평균) 이벤트 카운트 피처.

국가별 train(<=2023-12-31) anchor cosine p90을 임계값으로, 그 초과일을 "이벤트"로
보고 1d/7d/30d 합산 → ACLED event_count_{1,7,30}d 의 평균-임베딩 버전.
#52(제목 단위 카운트, 1d/7d)와 보완: 여기는 일(day) 단위 + 30d 지속성.

산출 (date, country): gkg_evt_<anchor>_1d / _7d / _30d  (16 × 3 = 48)
누수 차단: 임계값 train만, rolling trailing, 결측일 0(no-event).
출력: input/processed/features/gkg_emb_anchors_evtcnt.parquet (build_dataset auto-join)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("input/processed/features/gkg_emb_anchors.parquet")
OUT = Path("input/processed/features/gkg_emb_anchors_evtcnt.parquet")
TRAIN_END = pd.Timestamp("2023-12-31", tz="UTC")


def main() -> None:
    df = pd.read_parquet(SRC)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    anchor_cols = [c for c in df.columns if c.startswith("gkg_anchor_cos_")]

    # 국가 × anchor train p90 임계값
    thr = df[df["date"] <= TRAIN_END].groupby("country")[anchor_cols].quantile(0.90)
    print(f"입력 {df.shape}, 임계값 테이블 {thr.shape}")

    out_parts = []
    for country, g in df.groupby("country", sort=False):
        g = g.sort_values("date").set_index("date")
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D", tz="UTC")
        g = g[anchor_cols].reindex(full_idx, fill_value=0.0)
        rec = pd.DataFrame(index=g.index)
        for c in anchor_cols:
            a = c.replace("gkg_anchor_cos_", "")
            t = thr.loc[country, c] if country in thr.index else 0.0
            ev = (g[c] > t).astype(np.float32)
            rec[f"gkg_evt_{a}_1d"] = ev
            rec[f"gkg_evt_{a}_7d"] = ev.rolling(7, min_periods=1).sum().astype(np.float32)
            rec[f"gkg_evt_{a}_30d"] = ev.rolling(30, min_periods=1).sum().astype(np.float32)
        rec["country"] = country
        out_parts.append(rec.reset_index().rename(columns={"index": "date"}))

    out = pd.concat(out_parts, ignore_index=True)
    feat = [c for c in out.columns if c.startswith("gkg_evt_")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"저장: {OUT}  shape={out.shape}, 피처 {len(feat)}개")


if __name__ == "__main__":
    main()
