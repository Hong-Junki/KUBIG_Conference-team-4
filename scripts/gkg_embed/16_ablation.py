"""#58 피처 블록 통제 ablation (ACLED-free).

동일 고정 하이퍼파라미터 LGBM으로 피처 블록 조합만 바꿔 공정 비교.
목적: (1) 어느 블록이 효과/희석인지 분리 (2) 0.1에 가장 가까운 조합 탐색.

블록(GKG):
  CORE  비-GKG (GDELT events + 경제)        ← 항상 포함
  T1    Track1 통계
  EMB   PCA16 임베딩
  AUX   임베딩 보조(Δ/n_titles/KMeans)
  ANC   anchor cosine(평균)
  POOL  #52 제목 단위 pooling
  TEMP  #53 anchor 시계열
  EVT   #54 이벤트카운트
  RISK  #55 위험score
  BDEV  #57 baseline편차
  PRE   E6 사전징후 시그니처

사용법: python scripts/gkg_embed/16_ablation.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

DATA = Path("input/processed/dataset/full_pca16_aclfree.parquet")
TARGET = "y_escalation"
PERSIST_AP = 0.0354  # test persistence baseline PR-AUC (고정 기준)
LABEL_META = ["y", "y_onset", "y_escalation", "fatalities_next3d", "event_count_next3d",
              "past14d_event_count", "past14d_fatalities_mean"]
SPLIT = {"train_end": date(2023, 12, 31), "val_end": date(2024, 6, 30), "test_end": date(2025, 3, 28)}

PREFIX = {
    "POOL": ("gkg_pool_",), "TEMP": ("gkg_anchT_",), "EVT": ("gkg_evt_",),
    "RISK": ("gkg_risk_",), "BDEV": ("gkg_bdev_",), "PRE": ("gkg_preesc_",),
    "ANC": ("gkg_anchor_cos_",), "EMB": ("gkg_emb_pca_",),
    "AUX": ("gkg_emb_delta", "gkg_emb_cosdiss", "gkg_ntitles_", "gkg_cluster_",
            "gkg_emb_n_titles", "gkg_emb_missing_mask"),
    "GC2": ("gdelt2_",), "GSUB": ("gdelt_sub_",),
}


def assign_block(col: str) -> str:
    for blk, prefs in PREFIX.items():
        if col.startswith(prefs):
            return blk
    if col.startswith("gkg_") or col.startswith("page_title"):
        return "T1"
    return "CORE"


def precision_at_top(y, p, frac):
    k = max(1, int(len(y) * frac))
    idx = np.argsort(p)[::-1][:k]
    return float(y[idx].mean())


def train_eval(Xtr, ytr, Xva, yva, Xte, yte):
    # 공정 비교: 고정 트리 수(early-stop 미사용) + 정규화로 과적합 억제.
    # 모든 구성이 동일 학습량 → 피처 블록 효과만 분리.
    params = dict(objective="binary", n_estimators=200, learning_rate=0.03,
                  num_leaves=31, min_child_samples=80, scale_pos_weight=8.0,
                  subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0, reg_alpha=0.5,
                  random_state=42, n_jobs=-1, verbose=-1)
    m = lgb.LGBMClassifier(**params)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return {
        "PR-AUC": average_precision_score(yte, p),
        "gain": average_precision_score(yte, p) - PERSIST_AP,
        "P@1%": precision_at_top(yte, p, 0.01),
        "P@5%": precision_at_top(yte, p, 0.05),
        "n_feat": Xtr.shape[1],
        "best_iter": params["n_estimators"],
    }


def main():
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"])
    feat_cols = [c for c in df.columns if c not in (["date", "country"] + LABEL_META)]
    blocks: dict[str, list[str]] = {}
    for c in feat_cols:
        blocks.setdefault(assign_block(c), []).append(c)
    print("블록별 컬럼 수:", {k: len(v) for k, v in sorted(blocks.items())})

    d = df["date"].dt.date
    tr = (d <= SPLIT["train_end"]); va = (d > SPLIT["train_end"]) & (d <= SPLIT["val_end"])
    te = (d > SPLIT["val_end"]) & (d <= SPLIT["test_end"])
    y = df[TARGET].values

    LEAN = ["CORE", "T1", "POOL", "PRE", "BDEV"]
    configs = {
        "core_only":      ["CORE"],
        "lean(prev best)": LEAN,
        "core+C":         ["CORE", "GC2"],
        "core+A":         ["CORE", "GSUB"],
        "core+A+C":       ["CORE", "GC2", "GSUB"],
        "lean+C":         LEAN + ["GC2"],
        "lean+A":         LEAN + ["GSUB"],
        "lean+A+C":       LEAN + ["GC2", "GSUB"],
        "ALL+A+C":        list(blocks.keys()),
    }

    rows = []
    for name, blks in configs.items():
        cols = [c for b in blks for c in blocks.get(b, [])]
        cols = [c for c in cols if c in feat_cols]
        if not cols:
            continue
        r = train_eval(df.loc[tr, cols].values, y[tr], df.loc[va, cols].values, y[va],
                       df.loc[te, cols].values, y[te])
        r["config"] = name
        rows.append(r)
        print(f"  [{name:22s}] PR-AUC={r['PR-AUC']:.4f} gain={r['gain']:+.4f} "
              f"P@1%={r['P@1%']:.3f} P@5%={r['P@5%']:.3f} (feat {r['n_feat']}, iter {r['best_iter']})")

    res = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)
    out = Path("output/evaluation/ablation_gkg_blocks.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\n=== 정렬 결과 (PR-AUC 내림차순) ===")
    print(res[["config", "PR-AUC", "gain", "P@1%", "P@5%", "n_feat"]].to_string(index=False))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
