# 실험 F0: Safe ACLED Lag + B feature 결과

**실험명**: `stacking_safe_acled_f0`  
**실행일**: 2026-06-04  
**B baseline val PR-AUC**: 0.0564 (ACLED-free, 비교 기준)  
**C baseline val PR-AUC**: 0.0653 (ACLED-free + title, 참고)  
**채택 기준**: Stacking Platt PR-AUC ≥ 0.0594 (B + 0.003)  

> ⚠️ val 지표는 (1) final base model early stopping, (2) meta C 탐색,
> (3) Platt calibration에 val을 모두 사용하므로 낙관적 추정임.
> test는 아직 평가하지 않았다.

---

## 실험 F0 채택 여부

**Stacking Platt PR-AUC (val) = 0.0996**

| 비교 기준 | 기준값 | F0 PR-AUC | delta | 판정 |
|-----------|--------|-----------|-------|------|
| B baseline (ACLED-free) | 0.0564 | 0.0996 | +0.0432 ↑ | ✅ F0-B 달성 |
| C baseline (ACLED-free+title) | 0.0653 | 0.0996 | +0.0343 ↑ | — |
| 채택 기준 (B+0.003) | 0.0594 | 0.0996 | — | ✅ **채택** |

---

## 피처 구성

| 카테고리 | 피처 수 |
|----------|---------|
| GDELT events (B) | 19 |
| Economic (B)     | 15 |
| Country (B)      | 1 |
| B 합계           | 35 |
| safe ACLED lag (신규) | 15 |
| **F0 총계**      | **50** |

safe ACLED lag feature 설계:  
- shift(7) 후 rolling → t일 feature는 최대 t-7일 ACLED만 참조  
- label window (t+1~t+3) 와 feature window (~t-7) 사이 **8일 gap** 확보  
- macis_se_score 및 기존 acled_* feature 완전 제외  

---

## val 지표 비교 (B / C / F0)

| 모델 | F0 PR-AUC | B PR-AUC | F0-B delta | C PR-AUC | F0-C delta | P@5% | R@P≥.10 | Brier | ECE |
|------|-----------|----------|------------|----------|------------|------|---------|-------|-----|
| LightGBM | 0.0984 | 0.0601 | +0.0383 ↑ | 0.0601 | +0.0383 ↑ | 0.1439 | 0.3093 | 0.0997 | 0.2303 |
| XGBoost | 0.0933 | 0.0546 | +0.0387 ↑ | 0.0546 | +0.0387 ↑ | 0.1231 | 0.2279 | 0.2149 | 0.4150 |
| Stacking (raw) | 0.0996 | 0.0564 | +0.0432 ↑ | 0.0564 | +0.0432 ↑ | 0.1439 | 0.3209 | 0.3062 | 0.5171 |
| Stacking (Platt) ★ | 0.0996 | 0.0564 | +0.0432 ↑ | 0.0653 | +0.0343 ↑ | 0.1439 | 0.3209 | 0.0386 | 0.0000 |
| Stacking (Isotonic) | 0.0972 | 0.0563 | +0.0409 ↑ | 0.0563 | +0.0409 ↑ | 0.1515 | 0.3047 | 0.0379 | 0.0000 |

---

## OOF fold 요약

| fold | pred_year | n_train | n_pred | pos_rate | LGBM PR-AUC | XGB PR-AUC |
|------|-----------|---------|--------|----------|-------------|------------|
| F1 | 2018 | 84,738 | 21,170 | 0.0514 | 0.0765 | 0.0673 |
| F2 | 2019 | 105,908 | 21,170 | 0.0461 | 0.0612 | 0.0603 |
| F3 | 2020 | 127,078 | 21,228 | 0.0468 | 0.0514 | 0.0474 |
| F4 | 2021 | 148,306 | 21,170 | 0.0456 | 0.0568 | 0.0590 |
| F5 | 2022 | 169,476 | 21,170 | 0.0480 | 0.0591 | 0.0575 |
| F6 | 2023 | 190,646 | 21,170 | 0.0441 | 0.0765 | 0.0731 |

---

## Meta LogReg

선택된 C: **0.1**  

---

## 해석 주의사항

1. 채택 기준: B + 0.003 = 0.0594. 이를 초과하면 safe ACLED lag feature 효과 확인.
2. C(0.0653) 대비 F0 성능 비교: safe ACLED만 추가했을 때의 기여도 측정.
3. Isotonic ECE ≈ 0은 val 과적합 (val 셋 크기 작음). Platt 채택.
4. macis_se_score와 기존 acled_* feature는 완전 제외됨.
5. safe ACLED feature는 shift(7)+rolling으로 leakage-free 보장.

## test set 평가 정책

> **test set은 최종 feature/model 구조가 확정된 시점에 딱 한 번만 평가한다.**  
> 현재는 F 실험 비교 단계이며, test set은 아직 평가하지 않았다.  
> val 지표로만 실험 방향을 결정하고, 최종 모델 선택 후 test PR-AUC를 1회 측정한다.  

*생성: 2026-06-04*