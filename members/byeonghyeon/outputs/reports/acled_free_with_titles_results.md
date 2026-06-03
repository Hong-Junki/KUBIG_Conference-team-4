# 실험 C: ACLED-free + GDELT Title Features 결과

**실험명**: `stacking_acled_free_with_titles_smoke`  
**실행일**: 2026-06-03  
**B baseline val PR-AUC**: 0.0564 (비교 기준)  
**채택 기준**: Stacking Platt PR-AUC ≥ 0.0594 (B + 0.003)  

> ⚠️ val 지표는 (1) final base model early stopping, (2) meta C 탐색,
> (3) Platt calibration에 val을 모두 사용하므로 낙관적 추정임.
> test는 아직 평가하지 않았다.

---

## 실험 C 채택 여부

**Stacking Platt PR-AUC (val) = 0.0633**

✅ **채택** — B baseline(0.0564) 대비 +0.0069 ↑

---

## 피처 구성

| 카테고리 | 피처 수 |
|----------|---------|
| GDELT events (B)    | 19 |
| Economic (B)        | 15 |
| Country (B)         | 1 |
| B 합계              | 35 |
| GDELT title (신규)  | 21 |
| coverage_mask (신규)| 1 |
| **C 합계**          | **57** |

---

## val 지표 비교 (B vs C)

| 모델 | C PR-AUC | B PR-AUC | delta | P@5% | R@P≥.10 | Brier | ECE |
|------|---------|---------|-------|------|---------|-------|-----|
| LightGBM | 0.0613 | 0.0601 | +0.0012 ↑ | 0.0758 | 0.0767 | 0.1514 | 0.3199 |
| XGBoost | 0.0599 | 0.0546 | +0.0053 ↑ | 0.0720 | 0.0070 | 0.2198 | 0.4176 |
| Stacking (raw) | 0.0633 | 0.0564 | +0.0069 ↑ | 0.0814 | 0.0465 | 0.2109 | 0.4092 |
| Stacking (Platt) ★ | 0.0633 | 0.0564 | +0.0069 ↑ | 0.0814 | 0.0465 | 0.0388 | 0.0000 |
| Stacking (Isotonic) | 0.0629 | 0.0563 | +0.0066 ↑ | 0.0928 | 0.0395 | 0.0387 | 0.0000 |

---

## OOF fold 요약

| fold | pred_year | n_train | n_pred | pos_rate | LGBM PR-AUC | XGB PR-AUC |
|------|-----------|---------|--------|----------|-------------|------------|
| F6 | 2023 | 190,646 | 21,170 | 0.0441 | 0.0574 | 0.0612 |

---

## Meta LogReg

선택된 C: **0.1**  

---

## 해석 주의사항

1. B baseline(0.0564)과의 비교가 핵심. Reference(0.2714)는 ACLED+SE 포함 결과로 직접 비교 대상 아님.
2. 채택 기준: B + 0.003 = 0.0594. 이를 초과하면 실험 D (themes/persons)로 진행.
3. Isotonic ECE ≈ 0은 val 과적합 (val 셋 크기 작음). Platt 채택.
4. test는 아직 평가하지 않았다. B/C/D 비교 후 최종 모델 결정 시 1회 평가.

*생성: 2026-06-03*