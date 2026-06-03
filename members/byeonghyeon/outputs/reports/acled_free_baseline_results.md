# 실험 B: ACLED-free Stacking Baseline 결과

**실험명**: `stacking_acled_free_baseline_smoke`  
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
| XGBoost (ACLED-free) | 0.0524 | 0.0549 | 0.0140 | 0.0093 | 0.2388 | 0.4433 | — |
| Stacking (raw) | 0.0603 | 0.0682 | 0.0581 | 0.0070 | 0.1785 | 0.3686 | — |
| Stacking (Platt) | 0.0603 | 0.0682 | 0.0581 | 0.0070 | 0.0389 | 0.0000 | -0.2111 ↓ |
| Stacking (Isotonic) | 0.0608 | 0.0928 | 0.0581 | 0.0070 | 0.0387 | 0.0000 | — |

---

## OOF fold 요약

| fold | pred_year | n_train | n_pred | pos_rate | LGBM PR-AUC | XGB PR-AUC |
|------|-----------|---------|--------|----------|-------------|------------|
| F6 | 2023 | 190,646 | 21,170 | 0.0441 | 0.0592 | 0.0560 |

---

## Meta LogReg

선택된 C: **1.0**  

---

## 해석 주의사항

1. Reference (0.2714)는 ACLED + macis_se_score를 포함한 결과이므로 직접 비교 대상이 아님.
2. 실험 C (B + GDELT titles)가 B 대비 +0.003 이상이면 titles 피처 채택 기준 충족.
3. Isotonic ECE ≈ 0은 val 과적합 (val 셋 크기 작음). Platt 채택.

*생성: 2026-06-03*