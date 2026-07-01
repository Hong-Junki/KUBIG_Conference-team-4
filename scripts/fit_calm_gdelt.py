"""실시간 GDELT-calm 게이트 캘리브레이션 → output/models/onset_prod/calm_gdelt.pkl.

onset 경보는 calm 국가(활성 무력충돌 없음)에만 의미가 있다. 학습/평가는 ACLED
`past14d_event_count==0` 으로 calm 을 정의했지만, 실시간엔 ACLED 가 보고지연으로 stale →
calm_flag 이 전부 1 로 붕괴(2026 구간 실측). 그래서 calm 을 실시간 가용한 GDELT 로 근사한다.

핵심 교훈(실험 결과):
  - GDELT '절대 카운트'(violent_30d 등)는 in-sample AUC 0.73 으로 좋아 보이나, GDELT 볼륨이
    연도별로 폭증해 2022 학습 모델이 2025-26 엔 거의 전부 non-calm(calm율 0.002)으로 붕괴.
  - → 드리프트/볼륨 '무관'한 비율(share)·tone 피처만 사용한다. 절대 카운트는 금지.

선정 피처(전부 실시간 model_input 에 존재):
  gkg_theme_kill_ratio_7d   (살상 테마 비율)
  gdelt_quadclass_4_ratio   (물리적 충돌 이벤트 비율)
  gdelt_tone_mean_14d       (평균 톤; 높을수록 평온)

방법: train(<=2022) ACLED-calm 라벨로 표준화 로지스틱 fit → val(2023~24) AUC 검증,
임계값은 val ACLED-calm 율(0.164)에 calm 예측율을 정렬(P(calm)의 (1-rate) 분위수).
산출 아티팩트는 score.py `_calm_flag` 가 로드: calm_flag = 1 if sigmoid(coef·z+intercept) >= threshold.

사용: python scripts/fit_calm_gdelt.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA = "input/processed/dataset/full_pca16_aclfree.parquet"
OUT = Path("output/models/onset_prod/calm_gdelt.pkl")
FEATS = ["gkg_theme_kill_ratio_7d", "gdelt_quadclass_4_ratio", "gdelt_tone_mean_14d"]
TRAIN_END = "2022-12-31"
VAL = ("2023-01-01", "2024-12-31")


def main() -> None:
    df = pd.read_parquet(DATA, columns=["country", "date", "past14d_event_count"] + FEATS)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["calm"] = (df["past14d_event_count"].fillna(0) == 0).astype(int)  # ACLED-calm (학습 기준)

    tr = df[df["date"] <= TRAIN_END]
    va = df[(df["date"] >= VAL[0]) & (df["date"] <= VAL[1])]

    Xtr = tr[FEATS].fillna(0).values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    lr = LogisticRegression(max_iter=3000).fit((Xtr - mu) / sd, tr["calm"])

    pv = lr.predict_proba((va[FEATS].fillna(0).values.astype(float) - mu) / sd)[:, 1]
    auc = roc_auc_score(va["calm"], pv)
    target_rate = float(va["calm"].mean())
    thr = float(np.quantile(pv, 1 - target_rate))  # 예측 calm율 = val ACLED-calm율 정렬

    art = {
        "type": "logistic_calm", "feats": FEATS,
        "mu": mu.astype(float), "sd": sd.astype(float),
        "coef": lr.coef_.ravel().astype(float), "intercept": float(lr.intercept_[0]),
        "threshold": thr, "train_end": TRAIN_END, "val_auc": float(auc),
        "target_calm_rate": target_rate,
        "note": ("실시간 GDELT-calm 근사(ACLED past14d_event_count==0 대체). "
                 "드리프트/볼륨 무관 비율·tone 피처만(절대카운트는 GDELT 볼륨 증가로 serving서 붕괴)."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(art, open(OUT, "wb"))
    print(f"저장: {OUT}")
    print(f"  feats={FEATS}")
    print(f"  val AUC={auc:.3f} | threshold={thr:.4f} | target_calm_rate={target_rate:.3f}")
    print(f"  coef={art['coef']} intercept={art['intercept']:.3f}")

    # 최근 데이터 비-degenerate 점검(있으면)
    rec = df[df["date"] >= "2025-11-01"]
    if len(rec):
        pr = lr.predict_proba((rec[FEATS].fillna(0).values.astype(float) - mu) / sd)[:, 1]
        print(f"  [sanity] 최근(2025-11+) {len(rec)}행 예측 calm율 = {(pr >= thr).mean():.3f} "
              f"(전부 1/0 이면 실패)")


if __name__ == "__main__":
    main()
