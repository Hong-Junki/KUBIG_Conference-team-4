# F2_clean Feature Group Ablation 결과

**분석일**: 2026-06-04  
**F2_full baseline (val_eval Stacking Platt PR-AUC)**: 0.1027  
**F2_full feature 수**: 94개  

## Split (cleanval 동일)

| Split | 기간 | 역할 |
|-------|------|------|
| train_fit | 2014–2022 | base model 학습, OOF |
| tune_cal  | 2023      | early stopping, meta C, calibration fit |
| val_eval  | 2024-H1   | 순수 평가 전용 |
| test      | 미사용    | 평가 금지 |

---

## Ablation 요약 (PR-AUC 하락폭 순)

| 실험 | feature 수 | Stacking Platt PR-AUC | F2_full delta |
|------|-----------|----------------------|--------------|
| **F2_full (baseline)** | 94 | **0.1027** | — |
| **F2_no_safe_acled** | 79 | **0.0589** | **-0.0438 ↓** |
| **F2_no_country** | 93 | **0.0833** | **-0.0194 ↓** |
| **F2_no_gdelt_theme_person** | 72 | **0.0836** | **-0.0191 ↓** |
| **F2_no_gdelt_title** | 72 | **0.0861** | **-0.0166 ↓** |
| F2_no_economic | 79 | 0.0987 | -0.0040 ↓ |
| F2_no_gdelt_events | 75 | 0.1001 | -0.0026 ↓ |

---

## Ablation별 상세 지표

| 실험 | LGB PR-AUC | XGB PR-AUC | Platt PR-AUC | P@5% | R@P≥.10 | Brier | ECE |
|------|-----------|-----------|-------------|------|---------|-------|-----|
| F2_no_safe_acled | 0.0617 | 0.0524 | 0.0589 | 0.0701 | 0.0093 | 0.0389 | 0.0017 |
| F2_no_country | 0.0752 | 0.0808 | 0.0833 | 0.1231 | 0.2837 | 0.0387 | 0.0024 |
| F2_no_gdelt_theme_person | 0.0831 | 0.0885 | 0.0836 | 0.1061 | 0.1674 | 0.0385 | 0.0013 |
| F2_no_gdelt_title | 0.0832 | 0.0733 | 0.0861 | 0.1231 | 0.2140 | 0.0385 | 0.0002 |
| F2_no_economic | 0.0929 | 0.0869 | 0.0987 | 0.1174 | 0.3093 | 0.0384 | 0.0051 |
| F2_no_gdelt_events | 0.0991 | 0.0768 | 0.1001 | 0.1136 | 0.3279 | 0.0385 | 0.0005 |

---

## Feature Group 기여도 순위

> PR-AUC 하락폭이 클수록 해당 group의 기여도가 높음

| 순위 | Group (제거된 것) | PR-AUC 하락 | 기여도 해석 |
|------|-----------------|------------|------------|
| 1 | **no_safe_acled** | -0.0438 ↓ | 핵심 — 제거 시 대폭 하락 |
| 2 | **no_country** | -0.0194 ↓ | 핵심 — 제거 시 대폭 하락 |
| 3 | **no_gdelt_theme_person** | -0.0191 ↓ | 핵심 — 제거 시 대폭 하락 |
| 4 | **no_gdelt_title** | -0.0166 ↓ | 핵심 — 제거 시 대폭 하락 |
| 5 | **no_economic** | -0.0040 ↓ | 부분적 — 소폭 하락 |
| 6 | **no_gdelt_events** | -0.0026 ↓ | 부분적 — 소폭 하락 |

---

## 핵심 해석 요약

### 실제 기여도 순위 (ablation 기준)

feature importance(LGB gain)와 ablation 결과가 **다르게 나온 점이 중요하다.** LGB gain은 모델 내 분기 기여를 반영하지만, 실제 예측력 기여는 ablation이 더 직접적으로 측정한다.

| Group | LGB gain % | Ablation 하락 | 해석 |
|-------|-----------|--------------|------|
| safe_acled | 4.3% | **-0.0438** | gain이 낮아도 실제 기여 1위 — ACLED 신호가 모델 전체를 지지 |
| country | 24.6% | -0.0194 | gain 1위지만 ablation은 2위 — 국가 고정효과 반영 |
| gdelt_theme_person | 11.5% | -0.0191 | F2 개선의 직접 원인 |
| gdelt_title | 15.2% | -0.0166 | 단독보다 조합 내 시너지가 큼 |
| economic | 23.2% | -0.0040 | gain과 달리 ablation에서 낮음 — 다른 feature로 대체 가능 |
| gdelt_events | 21.2% | -0.0026 | ablation에서 가장 낮음 — safe_acled/title 등이 커버 |

---

## 질문별 해석

**Q1. country 제거 시 성능이 얼마나 떨어지는가?**
→ country 제거 시 -0.0194 하락. 실제로 기여하지만, 이는 **국가별 baseline risk** (일부 국가가 구조적으로 분쟁 위험이 높음)를 모델이 흡수하는 효과다. feature importance가 24.6%로 높지만 ablation에서는 safe_acled보다 낮다. 해석 시 주의 필요.  
  (제거 전: 0.1027, 제거 후: 0.0833, delta: -0.0194 ↓)

**Q2. theme/person 제거 시 F2 개선분이 사라지는가?**
→ theme/person 제거 시 0.0836으로 하락 — **F1_clean(0.0836)과 정확히 동일**. F2-F1 delta +0.0191이 theme/person group에서 비롯된 것임을 ablation으로 직접 확인.  
  (제거 후: 0.0836, delta: -0.0191 ↓)

**Q3. title 제거 시 F1 효과가 사라지는가?**
→ title 제거 시 -0.0166 하락 — F1-F0 단계 개선(+0.0055)보다 크다. 이는 title이 safe_acled, theme/person과 조합될 때 **단독보다 더 큰 시너지**를 내기 때문이다. title은 F2 구성에서 유효하다.  
  (제거 후: 0.0861, delta: -0.0166 ↓)

**Q4. safe ACLED 제거 시 모델이 여전히 괜찮은가?**
→ safe ACLED 제거 시 0.0589로 폭락 — **6개 group 중 압도적 1위 하락폭**. LGB gain(4.3%)이 낮았던 것과 대조적으로, ablation은 safe ACLED가 모델 전체의 기반 신호임을 보여준다. GDELT/경제 feature들이 ACLED 없이는 효과적으로 작동하지 못한다.  
  (제거 후: 0.0589, delta: -0.0438 ↓)

**Q5. economic 제거 시 성능이 크게 떨어지는가?**
→ economic 제거 시 -0.0040 소폭 하락. LGB gain(23.2%)과 달리 ablation에서는 낮은 기여도를 보인다. 경제 지표(VIX, WTI, Gold 등)가 포함되는 것이 좋지만, 다른 feature들이 상당 부분 커버한다.  
  (제거 후: 0.0987, delta: -0.0040 ↓)

**Q6. gdelt_events 제거 시 어떻게 되는가?**
→ gdelt_events 제거 시 -0.0026으로 가장 작은 하락. LGB gain(21.2%)에도 불구하고 ablation 효과는 미미하다. safe_acled와 title/theme 등이 이미 충분한 분쟁 관련 신호를 제공하므로, gdelt_events의 정보는 중복될 가능성이 높다.  
  (제거 후: 0.1001, delta: -0.0026 ↓)

---

## test set 평가 정책

> **test set은 아직 평가하지 않았다.**  
> 현재는 F2_clean ablation 분석 단계이며, 최종 feature/model 구조 확정 후  
> test PR-AUC를 딱 한 번만 측정한다.  

*생성: 2026-06-04*