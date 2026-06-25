# 실험 B: ACLED-free Stacking Baseline 결과

**실험명**: `stacking_acled_free_baseline`  
**실행일**: 2026-06-03  
**Reference val PR-AUC**: 0.2714 (기존 full model, 비교 참고용)  

> ⚠️ val 지표는 (1) final base model early stopping, (2) meta C 탐색,
> (3) Platt calibration에 val을 모두 사용하므로 낙관적 추정임.

---

## 피처 구성

| 카테고리 | 피처 수 |
|----------|---------|
| GDELT events | 19 |
| Economic     | 15 |
| Country      | 1 |
| **합계**     | **35** |

제거됨: ACLED raw 20개 + acled_missing_mask + macis_se_score

---

## val 지표 비교

| 모델 | PR-AUC | P@5% | R@P≥.10 | R@P≥.20 | Brier | ECE | delta vs Reference |
|------|--------|------|---------|---------|-------|-----|--------------------|
| LightGBM (ACLED-free) | 0.0601 | 0.0682 | 0.0581 | 0.0070 | 0.1217 | 0.2738 | — |
| XGBoost (ACLED-free) | 0.0546 | 0.0568 | 0.0047 | 0.0047 | 0.2282 | 0.4242 | — |
| Stacking (raw) | 0.0564 | 0.0758 | 0.0047 | 0.0000 | 0.3038 | 0.5142 | — |
| Stacking (Platt) | 0.0564 | 0.0758 | 0.0047 | 0.0000 | 0.0390 | 0.0000 | -0.2150 ↓ |
| Stacking (Isotonic) | 0.0563 | 0.0928 | 0.0047 | 0.0000 | 0.0388 | 0.0000 | — |

---

## OOF fold 요약

| fold | pred_year | n_train | n_pred | pos_rate | LGBM PR-AUC | XGB PR-AUC |
|------|-----------|---------|--------|----------|-------------|------------|
| F1 | 2018 | 84,738 | 21,170 | 0.0514 | 0.0765 | 0.0690 |
| F2 | 2019 | 105,908 | 21,170 | 0.0461 | 0.0612 | 0.0601 |
| F3 | 2020 | 127,078 | 21,228 | 0.0468 | 0.0514 | 0.0498 |
| F4 | 2021 | 148,306 | 21,170 | 0.0456 | 0.0568 | 0.0578 |
| F5 | 2022 | 169,476 | 21,170 | 0.0480 | 0.0591 | 0.0554 |
| F6 | 2023 | 190,646 | 21,170 | 0.0441 | 0.0582 | 0.0528 |

---

## Meta LogReg

선택된 C: **10.0**  

---

## 실험 B 기준 결과 (C/D 비교 기준점)

**Stacking Platt PR-AUC (val) = 0.0564** — 이 수치가 실험 C/D의 validation 비교 기준점이다.

- 모든 지표는 **validation set (2024-01-01 ~ 2024-06-30) 기준**이다.
- **test set(2024-07-01 ~ 2025-03-28)은 아직 평가하지 않았다.**  
  test 평가는 B/C/D 실험을 validation으로 비교해 최종 feature 구성이 결정된 뒤, 마지막에 한 번만 수행한다.

---

## 해석 주의사항

1. Reference (0.2714)는 ACLED + macis_se_score를 포함한 결과이므로 직접 비교 대상이 아님.
2. 실험 C (B + GDELT titles)가 B 대비 **+0.003 이상 (≥ 0.0594)** 이면 titles 피처 채택 기준 충족.
3. Isotonic ECE ≈ 0은 val 과적합 (val 셋 크기 작음). Platt 채택.
4. val 지표 낙관적 추정 주의: early stopping / meta C 탐색 / Platt calibration 모두 val 사용.

*생성: 2026-06-03*