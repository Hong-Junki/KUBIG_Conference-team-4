# D 카테고리 — 스태킹 프로토타입 경량 절제 분석

> **실험**: `stacking_tree_only_12y` | **작성일**: 2026-05-23
> **데이터**: 58개국 | train 2014-2023 (211,816행) | val 2024-01~06 (10,556행)

---

## 0. 주의사항 (모든 수치 해석 시 필독)

| 주의 | 설명 |
|------|------|
| **val early stopping** | Final LightGBM·XGBoost 모두 val로 early stopping → val 지표가 소폭 낙관적 |
| **meta C 선택** | LogReg meta-learner의 C 하이퍼파라미터도 val PR-AUC 기준으로 선택 → val 지표 낙관 편향 누적 |
| **Isotonic ECE ≈ 0** | val(10,556행)에서 학습·평가 → 자기 자신에 과적합. 고유 확률값 31개뿐(raw 10,554개 대비). test에서의 보정 성능은 불명 |
| **tree-only** | Level 0: LightGBM + XGBoost 2종만. LSTM은 C담당 파일 수령 후 추가 예정 |
| **OOF 설계** | 2014-2017은 OOF 예측 대상에서 제외(F1 학습셋으로만 사용). Meta 학습: 2018-2023 (127,078행) |

---

## 1. 모델 비교 — 전체 지표 (val set)

> 양성률: 0.0407 | top-5% = 528행 | top-10% = 1056행

| 모델 | PR-AUC | P@5% | P@10% | R@5% | R@10% | R@P≥.10 | R@P≥.20 | R@P≥.30 | Brier | ECE |
|------|--------|------|-------|------|-------|---------|---------|---------|-------|-----|
| LightGBM+SE 12y | 0.2596 | 0.2708 | 0.1809 | 0.3326 | 0.4442 | 0.7070 | 0.4326 | 0.3093 | 0.1443 | 0.2796 |
| XGBoost+SE 12y | 0.2534 | 0.2500 | 0.1695 | 0.3070 | 0.4163 | 0.7488 | 0.3535 | 0.2628 | 0.1314 | 0.2663 |
| Stacking (raw) | 0.2656 | 0.2614 | 0.1790 | 0.3209 | 0.4395 | 0.7419 | 0.3977 | 0.2837 | 0.3135 | 0.5071 |
| Stacking (Platt) | 0.2656 | 0.2614 | 0.1790 | 0.3209 | 0.4395 | 0.7419 | 0.3977 | 0.2837 | 0.0359 | 0.0074 |
| Stacking (Isotonic) | 0.2535 | 0.2746 | 0.1875 | 0.3372 | 0.4605 | 0.7349 | 0.3535 | 0.2558 | 0.0339 | 0.0000 |

**핵심 관찰**:

- **PR-AUC**: Stacking Platt (0.2656) > LightGBM (0.2596) > XGBoost (0.2534) ≈ Isotonic (0.2535)
- **P@top5%**: LightGBM (0.2708) > Isotonic (0.2746) > Stacking Platt (0.2614) > XGBoost (0.2500)
  - *Isotonic의 높은 P@5%는 val 과적합 산물일 가능성 — test에서 신뢰 불가*
- **P@top10%**: Isotonic (0.1875) > LightGBM (0.1809) ≈ Stacking (0.1790) > XGBoost (0.1695)
- **보정**: Platt ECE 0.0074 ← 운영 추천. raw ECE 0.5071 ← 사용 불가. Isotonic ECE ≈ 0 ← val 과적합
- **단일 모델 기준 (mask0_only LightGBM+SE)**: 0.1741 → 스태킹 Platt **+0.0915 (+52.6%) 개선**

---

## 2. top-5% Alert Set 다양성 분석

top-5% = 528개 알림 기준.

| 쌍 | 겹치는 알림 수 | Jaccard 유사도 |
|---|---|---|
| LGBM∩XGB | 391/528 | 0.588 |
| LGBM∩Stack(Platt) | 449/528 | 0.740 |
| XGB∩Stack(Platt) | 466/528 | 0.790 |

| 항목 | 건수 | 해석 |
|------|------|------|
| LGBM-only (스태킹에 없는 LGBM 알림) | 79 | 스태킹이 놓친 LGBM 고유 신호 |
| XGB-only  (스태킹에 없는 XGB 알림) | 62 | 스태킹이 놓친 XGB 고유 신호 |
| Stack-only (두 단독 모델 모두 없음) | 4 | 앙상블 효과로만 포착된 신호 |

**LightGBM vs XGBoost 확률 상관**: Pearson r = 0.8847 / Spearman ρ = 0.8729

> 두 모델 확률 상관이 0.88로 높아 앙상블 다양성이 제한적임. LGBM∩XGB Jaccard = 0.588 —
> 즉 top-5% 중 약 26%는 한 모델에서만 잡힘. LSTM 추가 시 다양성 대폭 증가 기대.

---

## 3. Platt 보정 구간별 검사

| 구간 | n | 실제 양성률 | Platt 평균확률 | 오차 |
|------|---|------------|--------------|------|
| [0.0,0.1) | 9,284 | 0.0235 | 0.0261 | 0.0026 |
| [0.1,0.2) | 1,147 | 0.1282 | 0.1410 | 0.0128 |
| [0.2,0.3) | 125 | 0.5200 | 0.2099 | 0.3101 ⚠ |

> [0.2-0.3) 구간 125행에서 실제 양성률 0.52 vs Platt 예측 0.21 → 고확률 구간에서 심한 과소추정.
> 표본이 125개로 적어 통계적 불안정성이 주 원인일 가능성 높음. 단, 고위험 국가 알람 정확도에 직접 영향.

---

## 4. Isotonic ECE=0 해석

- Isotonic calibrator가 val 10,556행을 학습 후 동일 val로 평가 → 사실상 암기(memorization)
- 고유 확률값 수: raw 10,554개 → Isotonic 31개
- 비선형 단조 보정이 val 데이터의 구간 분포에 완벽히 맞춰짐
- **test set에서의 ECE는 반드시 직접 측정해야 함** (현재 test y_true 없음)
- 운영 추천: **Platt**. test 평가 가능 시점에 Isotonic vs Platt 재비교 필요.

---

## 5. 현재 D 산출물 정리

| 파일 | 용도 | 권장 여부 |
|------|------|----------|
| `predictions__stacking_tree_only_12y_platt__D_byeonghyeon.csv` | **D 카테고리 1차 제출** | ✅ 권장 |
| `predictions__stacking_tree_only_12y_isotonic__D_byeonghyeon.csv` | 참고용 (val 과보정 의심) | ⚠️ 보류 |
| `predictions__stacking_tree_only_12y_raw__D_byeonghyeon.csv` | 보정 없음, 운영 불가 | ❌ 제출 불가 |

---

## 6. 다음 heavy ablation 권고 순서

| 우선순위 | 실험 | 이유 | 예상 비용 |
|----------|------|------|----------|
| 1 | **mask0-only** (`stacking_tree_only_12y_mask0`) | 현재 full-train에는 mask=1(29,262행)이 포함되어 SE=0 강제. 레퍼런스 단일 모델(PR-AUC=0.1741)과 동일 조건으로 공정 비교 | 중간 (OOF×12 + final×2) |
| 2 | **SE include vs exclude** | macis_se_score가 55→54 피처로 빠질 때 PR-AUC 변화량 = SE 기여도 정량화. 현재 스태킹에서 SE 의존도 불명 | 중간 |
| 3 | **train start 2016/2018** | ACLED 결측 구간(2014-2017 일부국) 제거 효과. mask0와 달리 행수 자체를 줄이는 실험 | 중간 (OOF 구조 변경 필요) |
| 4 | **LSTM Level 0 추가** | 시퀀스 다양성으로 스태킹 앙상블 다양성 대폭 증가 기대. LGBM∩XGB Jaccard=0.588이므로 이질적 신호 필요 | 높음 (C담당 파일 수령 선행) |

> **즉시 실행 가능**: mask0-only (스크립트 2줄 변경만 필요)
> **C담당 대기 필요**: LSTM 추가
