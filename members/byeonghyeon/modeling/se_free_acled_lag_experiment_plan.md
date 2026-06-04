# SE-free + Safe ACLED Lag 실험 계획 (Experiment F)

**작성일**: 2026-06-04

---

## 1. 지금까지 실험 결과 요약

### ACLED-free 실험 (구버전 val 사용)

| 실험 | feature 구성 | feature 수 | Stacking Platt PR-AUC | 판정 |
|------|-------------|-----------|----------------------|------|
| B | GDELT events + economic + country | 35 | 0.0564 | baseline |
| C | B + GDELT title/tone/count/domain/language | 57 | 0.0653 | 현재 ACLED-free 최고 |
| C2 | C + 3d rolling/spike/country-normalized derived | 76 | 0.0643 | ❌ 미채택 |
| D | C + GDELT v2themes/v2persons | 79 | 0.0633 | ❌ 미채택 |

### F 실험 (clean validation: val_eval 순수 평가 전용)

| 실험 | feature 구성 | feature 수 | Stacking Platt PR-AUC (cleanval) | 판정 |
|------|-------------|-----------|----------------------------------|------|
| F0_clean | B + safe ACLED lag | 50 | 0.0781 | ✅ 채택 (≥0.0594) |
| F1_clean | F0 + GDELT title + mask | 72 | 0.0836 | ✅ 채택 (≥0.0811) |
| **F2_clean** | F1 + GDELT theme/person | **94** | **0.1027** | ✅ **채택 (≥0.0866) — 현재 최고** |

> cleanval 기준: train_fit(2014~2022) 학습, tune_cal(2023) early stop/meta C/calibration, val_eval(2024-H1) 순수 평가  
> test set은 아직 평가하지 않았다. 최종 모델 확정 후 1회만 평가한다.

---

## 2. 왜 ACLED-free만으로는 성능이 낮은가

C(0.0653)는 B(0.0564) 대비 개선되었지만, 절대 성능은 낮다.

원인 분석:
1. **y_escalation label 자체가 ACLED 기반**: label은 t+1~t+3일 ACLED event count와 fatalities를 직접 사용해 생성된다. 따라서 과거 ACLED 패턴은 강력한 예측 신호임에도 실험에서 완전히 제외되었다.
2. **GDELT 대체 한계**: GDELT title/tone/theme 피처는 미디어 보도 기반이며, 실제 분쟁 사건을 직접 측정하지 않는다. 분쟁 escalation 예측에서 과거 분쟁 기록(ACLED)의 정보량이 훨씬 크다.
3. **C2/D에서 추가 피처 효과 없음**: 파생 피처(C2)나 테마/인물 집계(D) 모두 stacking 기준으로 C보다 낮아, 단순 피처 확장으로는 한계에 도달했다.

---

## 3. 왜 SE score(macis_se_score)를 계속 제외하는가

- `macis_se_score`는 생성 시점이 불명확하다. train-only로 만들어졌을 경우 val/test 예측 시 데이터가 없거나 훈련 데이터를 참조해 leakage가 발생할 수 있다.
- 운영 환경에서 실시간으로 사용 가능한지 검증되지 않았다.
- 보수적 원칙: leakage 가능성이 조금이라도 있으면 제외한다.

---

## 4. 왜 과거 ACLED lag feature는 leakage가 아닌가

```
타임라인:
  ... [ACLED feature window: ~ t-7] ... [gap 8일] ... [label window: t+1 ~ t+3]
```

- `y_escalation(t)` label은 t+1~t+3 미래 정보를 사용해 만들어진다.
- safe ACLED feature는 shift(7) 후 rolling으로, t일 기준 최대 t-7일 ACLED만 사용한다.
- feature 최신일(t-7)과 label 최초일(t+1) 사이에 **8일 gap**이 존재한다.
- 현실에서도 "과거 7일 전까지의 분쟁 기록"은 예측 시점에 관측 가능하다 (ACLED publication lag ~7일 가정).
- 따라서 과거 lag ACLED feature는 leakage가 아닌 적법한 예측 변수다.

---

## 5. 실험 F 설계

### 공통 원칙
- macis_se_score: 모든 F 실험에서 제외
- safe ACLED feature: 모두 `_lag7` 접미사로 lag 명시
- test set: F 실험 전체 완료 후 채택 모델에 대해 1회만 평가

### F0: Safe ACLED + GDELT events + economic + country (SE-free baseline with ACLED)

| 항목 | 내용 |
|------|------|
| 목적 | safe ACLED lag feature만 추가했을 때 baseline 성능 |
| feature 수 | 35 (B) + 15 (safe ACLED) = **50개** |
| 비교 기준 | B(0.0564) 대비 개선 여부 |
| 채택 기준 | Stacking Platt PR-AUC ≥ 0.0594 (B + 0.003) |

**F0 feature 구성 (50개)**

- B: GDELT events 19 + economic 15 + country 1 = 35
- safe ACLED: count/fatalities 9 + event type ratio 3 + actor ratio 2 + missing_mask 1 = 15
  - actor ratio 2개: state_forces(inter=1), external_forces(inter=8) — 실제 데이터 존재 확인
  - 제외: rebel(2)/political_militia(3) — 58개국 데이터에 없음
  - 보류: civilians(7) — VAC ratio와 중복 가능성

### F1: F0 + C title features

| 항목 | 내용 |
|------|------|
| 목적 | safe ACLED + GDELT title 조합 성능 |
| feature 수 | 50 (F0) + 21 (C title) + 1 (coverage_mask) = **72개** |
| 비교 기준 | C(0.0653) 대비 개선 여부 |
| 채택 기준 | Stacking Platt PR-AUC ≥ 0.0683 (C + 0.003) |

**F1 feature 구성 (72개)**

- F0 50개
- C: GDELT title 1d/7d 21 + coverage_mask 1 = 22

### F2: F1 + D theme/person features

| 항목 | 내용 |
|------|------|
| 목적 | safe ACLED + C title + D theme/person 풀 조합 |
| feature 수 | 72 (F1) + 22 (D) = **94개** |
| 비교 기준 | F1 대비 개선 여부 |
| 채택 기준 | Stacking Platt PR-AUC ≥ F1 + 0.003 |

### 실험 순서 및 의사결정

```
F0 실행
  ├─ F0 ≥ 0.0594 → F1 진행
  └─ F0 < 0.0594 → safe ACLED feature 효과 없음, 피처 검토 후 재설계

F1 실행
  ├─ F1 ≥ 0.0683 (C + 0.003) → ✅ F1 채택, test 평가 진행
  ├─ F1 ≥ C(0.0653) → F2 진행 (F1이 C보다 좋으나 기준 미달)
  └─ F1 < C(0.0653) → F2 진행 또는 C로 복귀 검토

F2 실행 (필요 시)
  ├─ F2 ≥ F1 + 0.003 → ✅ F2 채택, test 평가
  └─ F2 < F1 + 0.003 → F1 유지 또는 C 복귀
```

---

## 6. Validation 비교 기준 요약

| 실험 | feature 수 | 채택 기준 |
|------|-----------|---------|
| F0 | 50 | ≥ 0.0594 (B + 0.003) |
| F1 | 72 | ≥ 0.0683 (C + 0.003) |
| F2 | 94 | ≥ F1 + 0.003 |

---

## 7. Test Set 평가 정책

> **test set은 최종 feature/model 구조가 확정된 시점에 딱 한 번만 평가한다.**

- val 지표만으로 실험 방향을 결정한다
- F 실험 중간 단계에서 test를 보지 않는다
- 채택 모델이 결정되면 그때 1회 test PR-AUC를 측정한다
- val이 낙관적 추정임을 항상 감안한다 (final model early stopping, meta C 탐색, Platt calibration 모두 val 사용)

---

## 8. 예상되는 성능 개선 가능성

**낙관적 시나리오**: F1이 0.0683 이상
- safe ACLED lag feature는 과거 분쟁 패턴의 직접적인 신호이므로 모델에 강력한 정보 추가
- ACLED 기반 label과 직접적인 상관관계 존재 (과거 분쟁 있는 국가에서 미래 분쟁 높음)
- C title feature는 이미 효과 입증됨

**보수적 시나리오**: F0 > B이지만 F1이 C에 못 미칠 수 있음
- ACLED 피처 추가가 overfitting을 유발할 수 있음 (특히 XGBoost)
- 현재 stacking이 피처 수 증가에 취약한 경향 관찰됨 (C2, D 모두 하락)
- 이 경우 LightGBM 단독 또는 피처 선별 ablation이 필요

---

## 9. 주의할 점

1. **safe ACLED feature 생성 전 dry-run 필수**: `--dry-run` 옵션으로 ACLED 파일 존재 확인 후 실제 실행
2. **OOF fold 설계 동일 유지**: B/C/C2/D와 동일한 expanding-window 6-fold (F1~F6)
3. **하이퍼파라미터 동결**: F 실험에서 LGB/XGB 하이퍼파라미터는 B/C와 동일하게 유지. 튜닝은 최종 채택 모델 확정 후 별도 진행
4. **actor type 코드 4~8 보류**: inter code 1, 8만 존재 확인. 향후 추가 여부는 feature importance 확인 후 결정

---

## 10. F2_clean 최종 후보 결정 및 다음 단계

**F2_clean (94개)을 validation 기준 최종 후보로 결정.**

### 다음 단계

| 단계 | 작업 | 상태 |
|------|------|------|
| 1 | F2_clean feature importance 분석 | 미수행 |
| 2 | feature group 기여도 ablation | 미수행 |
| 3 | 최종 후보 확정 | F2_clean (val 기준) |
| 4 | **test 1회 평가** | **미수행 — F2_clean 확정 후 진행** |

### test set 평가 정책

> **test set은 F2_clean으로 최종 모델을 확정한 뒤 딱 한 번만 평가한다.**  
> val_eval(cleanval)에서 F2_clean이 최고임을 확인했으므로, feature importance 분석 후 test를 진행한다.  
> test 평가 전에 feature group ablation을 통해 F2_clean이 최적인지 재검토할 수 있다.
4. **actor type ratio 코드 검증**: processed ACLED의 inter1/inter2 int 코드(1~8)가 ACLED codebook과 일치하는지 최초 실행 후 feature distribution으로 확인 필요
5. **inter4~8 보류**: 현재 builder에서 actor type 1/2/3만 포함. 4~8 (Identity militia, Rioters, Protesters, Civilians, External forces)은 예측력 검증 후 추가 여부 결정
