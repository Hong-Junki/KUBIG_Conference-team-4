# Clean Validation: F0/F1/F2 비교 결과

**실험명**: `cleanval_compare`  
**실행일**: 2026-06-04  
**validation 기준 최종 후보**: **F2_clean** (Stacking Platt PR-AUC = 0.1027)  

## Split 설계

| Split | 기간 | 역할 |
|-------|------|------|
| train_fit | 2014-01-01 ~ 2022-12-31 | base model 학습, OOF 생성 |
| tune_cal  | 2023-01-01 ~ 2023-12-31 | early stopping, meta C 선택, calibration fit |
| val_eval  | 2024-01-01 ~ 2024-06-30 | **순수 평가 전용** (어떤 학습에도 사용 안 함) |
| test      | 2024-07-01 ~             | 아직 평가하지 않음 |

> val_eval은 early stopping, meta C 선택, calibration fit 어디에도 사용하지 않았다.  
> 따라서 ECE/Brier/PR-AUC는 기존 F0/F1 스크립트보다 덜 낙관적인 추정이다.  
> 단, 절대 수치는 여전히 test에서 달라질 수 있다.

---

## Feature 구성

| 실험 | feature 수 | 구성 |
|------|-----------|------|
| F0_clean | 50 | B 35 + safe ACLED 15 |
| F1_clean | 72 | F0 + GDELT title 21 + coverage_mask 1 |
| F2_clean | 94 | F1 + GDELT theme/person 22 |

---

## 실험별 val_eval 성능

### F0_clean (feature 50개)

| 모델 | PR-AUC | P@5% | R@P≥.10 | Brier | ECE |
|------|--------|------|---------|-------|-----|
| LightGBM | 0.0795 | 0.1117 | 0.1581 | 0.1182 | 0.2593 |
| XGBoost | 0.0735 | 0.1061 | 0.2791 | 0.1995 | 0.3866 |
| Stacking (raw) | 0.0781 | 0.1098 | 0.1977 | 0.3006 | 0.5115 |
| Stacking (Platt) ★ | 0.0781 | 0.1098 | 0.1977 | 0.0387 | 0.0010 |
| Stacking (Isotonic) | 0.0740 | 0.1117 | 0.1395 | 0.0386 | 0.0041 |

Meta LogReg C: 0.01

### F1_clean (feature 72개)

| 모델 | PR-AUC | P@5% | R@P≥.10 | Brier | ECE |
|------|--------|------|---------|-------|-----|
| LightGBM | 0.0831 | 0.1155 | 0.1628 | 0.1309 | 0.2691 |
| XGBoost | 0.0885 | 0.1269 | 0.2442 | 0.1837 | 0.3599 |
| Stacking (raw) | 0.0836 | 0.1061 | 0.1674 | 0.3171 | 0.5263 |
| Stacking (Platt) ★ | 0.0836 | 0.1061 | 0.1674 | 0.0385 | 0.0013 |
| Stacking (Isotonic) | 0.0747 | 0.0985 | 0.1674 | 0.0385 | 0.0041 |

Meta LogReg C: 0.1

### F2_clean (feature 94개)

| 모델 | PR-AUC | P@5% | R@P≥.10 | Brier | ECE |
|------|--------|------|---------|-------|-----|
| LightGBM | 0.0975 | 0.1307 | 0.2209 | 0.1316 | 0.2740 |
| XGBoost | 0.0909 | 0.1193 | 0.3000 | 0.1706 | 0.3381 |
| Stacking (raw) | 0.1027 | 0.1269 | 0.2860 | 0.3192 | 0.5283 |
| Stacking (Platt) ★ | 0.1027 | 0.1269 | 0.2860 | 0.0384 | 0.0022 |
| Stacking (Isotonic) | 0.0882 | 0.1231 | 0.2209 | 0.0384 | 0.0018 |

Meta LogReg C: 1.0

---

## OOF fold 요약 (train_fit 내부)

### F0_clean
| fold | pred_year | n_train | n_pred | pos_rate | LGBM PR-AUC | XGB PR-AUC |
|------|-----------|---------|--------|----------|-------------|------------|
| OOF_F1 | 2018 | 84,738 | 21,170 | 0.0514 | 0.0765 | 0.0673 |
| OOF_F2 | 2019 | 105,908 | 21,170 | 0.0461 | 0.0612 | 0.0603 |
| OOF_F3 | 2020 | 127,078 | 21,228 | 0.0468 | 0.0514 | 0.0474 |
| OOF_F4 | 2021 | 148,306 | 21,170 | 0.0456 | 0.0568 | 0.0590 |
| OOF_F5 | 2022 | 169,476 | 21,170 | 0.0480 | 0.0591 | 0.0575 |

### F1_clean
| fold | pred_year | n_train | n_pred | pos_rate | LGBM PR-AUC | XGB PR-AUC |
|------|-----------|---------|--------|----------|-------------|------------|
| OOF_F1 | 2018 | 84,738 | 21,170 | 0.0514 | 0.0809 | 0.0642 |
| OOF_F2 | 2019 | 105,908 | 21,170 | 0.0461 | 0.0613 | 0.0620 |
| OOF_F3 | 2020 | 127,078 | 21,228 | 0.0468 | 0.0539 | 0.0503 |
| OOF_F4 | 2021 | 148,306 | 21,170 | 0.0456 | 0.0603 | 0.0617 |
| OOF_F5 | 2022 | 169,476 | 21,170 | 0.0480 | 0.0582 | 0.0568 |

### F2_clean
| fold | pred_year | n_train | n_pred | pos_rate | LGBM PR-AUC | XGB PR-AUC |
|------|-----------|---------|--------|----------|-------------|------------|
| OOF_F1 | 2018 | 84,738 | 21,170 | 0.0514 | 0.0786 | 0.0656 |
| OOF_F2 | 2019 | 105,908 | 21,170 | 0.0461 | 0.0634 | 0.0632 |
| OOF_F3 | 2020 | 127,078 | 21,228 | 0.0468 | 0.0538 | 0.0508 |
| OOF_F4 | 2021 | 148,306 | 21,170 | 0.0456 | 0.0583 | 0.0621 |
| OOF_F5 | 2022 | 169,476 | 21,170 | 0.0480 | 0.0608 | 0.0599 |

---

## 실험 간 Stacking Platt PR-AUC 비교

| 실험 | feature 수 | Stacking Platt PR-AUC | vs B(0.0564) | vs C(0.0653) | 판정 |
|------|-----------|----------------------|-------------|-------------|------|
| B baseline (ACLED-free) | 35 | 0.0564 | — | — | 비교 기준 |
| C baseline (ACLED-free+title) | 57 | 0.0653 | +0.0089 ↑ | — | 비교 기준 |
| **F0_clean** | 50 | **0.0781** | +0.0217 ↑ | +0.0128 ↑ | ✅ 채택 (기준≥0.0594) |
| **F1_clean** | 72 | **0.0836** | +0.0272 ↑ | +0.0183 ↑ | ✅ 채택 (기준≥0.0811) |
| **F2_clean** | 94 | **0.1027** | +0.0463 ↑ | +0.0374 ↑ | ✅ 채택 (기준≥0.0866) |

### 단계별 개선폭 (Stacking Platt, val_eval 기준)

- F0_clean           : 0.0781
- F1_clean           : 0.0836  (F1-F0 delta: +0.0055 ↑)
- F2_clean           : 0.1027  (F2-F1 delta: +0.0190 ↑)

---

## ECE/Brier 해석

- calibration fit: tune_cal (2023) 기준  
- calibration 평가: val_eval (2024-H1) 기준  
- **기존 스크립트(F0/F1)와 달리 fit set ≠ eval set이므로 ECE/Brier가 실제 calibration 품질을 반영함**  
- 다만 Platt calibration은 단조 변환이므로 PR-AUC는 raw/Platt 간 거의 동일  
- 모델 선택 기준: PR-AUC (rank 기반, calibration 무관)

---

## test set 평가 정책

> **test set은 최종 feature/model 구조가 확정된 시점에 딱 한 번만 평가한다.**  
> 현재는 F0/F1/F2 비교 단계이며, test set은 아직 평가하지 않았다.  
> val_eval 지표로 방향을 결정하고, 최종 모델 선택 후 test PR-AUC를 1회 측정한다.  

---

## 기존 F0/F1 결과와 cleanval 결과 비교

기존 F0/F1 스크립트는 val set(2024-H1)을 early stopping, meta C 선택, calibration fit, 평가에 모두 사용했다. 이로 인해:
- val PR-AUC가 낙관적으로 추정됨 (fit=eval 구조)
- ECE ≈ 0은 수학적으로 당연한 결과 (calibration이 val에 완벽히 fit됨), 실제 calibration 품질 반영 아님

| 실험 | 기존 스크립트 PR-AUC | cleanval PR-AUC | 차이 |
|------|---------------------|----------------|------|
| F0 / F0_clean | 0.0996 | **0.0781** | -0.0215 |
| F1 / F1_clean | 0.1160 | **0.0836** | -0.0324 |

cleanval에서 수치가 낮아진 것은 **공정한 평가 구조** 덕분이다. 방향성(F1 > F0 > B)은 두 결과에서 동일하다.

---

## 최종 모델 후보 결론

### F2_clean을 validation 기준 최종 후보로 결정

| 근거 | 내용 |
|------|------|
| F2 Stacking Platt PR-AUC | **0.1027** — 세 실험 중 최고 |
| F2 - F1 delta | **+0.0191** — theme/person 피처의 기여 확인 |
| F2 - F0 delta | **+0.0246** — safe ACLED 위에서 GDELT 조합 효과 |
| 기준 충족 | F1 + 0.003 = 0.0866 기준 ✅ 채택 |

### 각 feature group의 기여 요약

| feature group 추가 | PR-AUC | 개선폭 |
|-------------------|--------|--------|
| B (GDELT events + economic + country) | 0.0564 | — |
| + safe ACLED lag (F0_clean) | 0.0781 | +0.0217 |
| + GDELT title (F1_clean) | 0.0836 | +0.0055 |
| + GDELT theme/person (F2_clean) | **0.1027** | **+0.0191** |

- **safe ACLED lag**: 과거 분쟁 기록이 y_escalation 예측에 가장 강력한 신호임을 확인
- **GDELT title**: safe ACLED 위에서도 미디어 보도 신호가 독립적으로 기여
- **GDELT theme/person**: 테마/인물 구조가 가장 큰 추가 개선 제공

### 주의사항

1. **F2_clean은 validation 기준 최종 후보**이지, test 성능이 확인된 최종 모델은 아니다.
2. calibration(Platt)은 tune_cal(2023) fit → val_eval(2024-H1) eval 구조이므로 기존보다 해석 가능하지만, 최종 일반화 성능은 test 1회 평가로 확인해야 한다.
3. val_eval(2024-H1)도 특정 시점의 단면이므로, test(2024-H2 ~ 2025-Q1)와 다른 분쟁 패턴을 포함할 수 있다.

---

## 다음 단계

1. **F2_clean feature importance 분석** — 94개 feature 중 기여도 상위 피처 파악
2. **feature group ablation** — 각 group(safe ACLED / title / TP)의 기여도 정량화
3. **최종 후보 확정 후 test 1회 평가** — val 지표로 최종 모델 결정 시 test PR-AUC 측정

*생성: 2026-06-04*