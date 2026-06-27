"""트리 base learner(LGBM+XGB)의 폴드별 val OOF 예측 생성 — 스태킹용.

cv_harness 와 동일한 폴드/피처(lean+A+C+GDEV)로 F2/F3 val 예측 저장.
torch 미import (macOS libomp segfault 회피). 출력 행은 GRU OOF 와 (country,date)로 정렬됨.

사용법: python scripts/base_trees_oof.py
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import lightgbm as lgb
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

DATA = "input/processed/dataset/full_pca16_aclfree.parquet"
OUT = "output/aclfree_010/trees_oof.parquet"
TARGET = "y_escalation"
LABEL_META = ["y", "y_onset", "y_escalation", "fatalities_next3d", "event_count_next3d",
              "past14d_event_count", "past14d_fatalities_mean"]
FOLDS = [("F2", date(2023, 12, 31), date(2024, 1, 1), date(2024, 6, 30)),
         ("F3", date(2024, 6, 30), date(2024, 7, 1), date(2024, 12, 31))]

PREFIX = {"POOL": ("gkg_pool_",), "PRE": ("gkg_preesc_",), "BDEV": ("gkg_bdev_",),
          "GC2": ("gdelt2_",), "GSUB": ("gdelt_sub_",), "GDEV": ("gdev_",),
          "ANC": ("gkg_anchor_cos_",), "EMB": ("gkg_emb_pca_",), "TEMP": ("gkg_anchT_",),
          "EVT": ("gkg_evt_",), "RISK": ("gkg_risk_",),
          "AUX": ("gkg_emb_delta", "gkg_emb_cosdiss", "gkg_ntitles_", "gkg_cluster_",
                  "gkg_emb_n_titles", "gkg_emb_missing_mask")}
KEEP = {"CORE", "T1", "POOL", "PRE", "BDEV", "GC2", "GSUB", "GDEV"}

# tune_cv.py 결과 (clean CV 0.0737, 2026-06-16).
LGBM_PARAMS = dict(objective="binary", n_estimators=459, learning_rate=0.0136, num_leaves=43,
                   min_child_samples=91, scale_pos_weight=22.0899, subsample=0.909,
                   colsample_bytree=0.8063, reg_lambda=1.3014, reg_alpha=2.7105)


def assign(c):
    for b, p in PREFIX.items():
        if c.startswith(p):
            return b
    return "T1" if (c.startswith("gkg_") or c.startswith("page_title")) else "CORE"


def main():
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    y = df[TARGET].values
    cols = [c for c in df.columns if c not in (["date", "country"] + LABEL_META) and assign(c) in KEEP]
    print(f"피처 {len(cols)}개 (lean+A+C+GDEV)")
    d = df["date"].dt.date

    rows = []
    for name, te, vs, ve in FOLDS:
        tr = (d <= te).values
        va = ((d >= vs) & (d <= ve)).values
        Xtr, ytr = df.loc[tr, cols].values, y[tr]
        Xva = df.loc[va, cols].values

        ml = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=42, n_jobs=4, verbose=-1)
        ml.fit(Xtr, ytr)
        p_lgbm = ml.predict_proba(Xva)[:, 1]

        if HAS_XGB:
            # 튜닝본(n_est=690 등)은 OOF에서 과적합(0.064) → 기본 robust 파라미터가 0.0734로 우수
            pos = ytr.mean()
            mx = xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=5,
                                   subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0,
                                   min_child_weight=5, scale_pos_weight=(1 - pos) / max(pos, 1e-6),
                                   eval_metric="aucpr", tree_method="hist", n_jobs=4, random_state=42)
            mx.fit(Xtr, ytr)
            p_xgb = mx.predict_proba(Xva)[:, 1]
        else:
            p_xgb = p_lgbm

        ap_l = average_precision_score(y[va], p_lgbm)
        ap_x = average_precision_score(y[va], p_xgb)
        print(f"  [{name}] LGBM PR-AUC={ap_l:.4f}  XGB PR-AUC={ap_x:.4f}")
        rows.append(pd.DataFrame({"country": df.loc[va, "country"].values,
                                  "date": df.loc[va, "date"].values, "fold": name,
                                  "y": y[va], "p_lgbm": p_lgbm, "p_xgb": p_xgb}))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT, index=False)
    clean_l = np.mean([average_precision_score(g["y"], g["p_lgbm"]) for _, g in out.groupby("fold")])
    clean_x = np.mean([average_precision_score(g["y"], g["p_xgb"]) for _, g in out.groupby("fold")])
    print(f"\nclean CV  LGBM={clean_l:.4f}  XGB={clean_x:.4f}")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
