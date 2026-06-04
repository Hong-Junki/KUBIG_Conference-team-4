# Safe ACLED Lag Feature Builder 실행 가이드

## 목적

processed ACLED event 데이터에서 **publication lag 7일을 명시적으로 반영**한  
leakage-safe ACLED lag feature를 생성한다.

> t일 예측에 사용하는 모든 feature는 **최대 t-7일까지의 ACLED 데이터**만 사용한다.

---

## 왜 ACLED feature를 다시 포함하는가

ACLED-free 실험(B/C/C2/D)에서 최고 val PR-AUC는 C = 0.0653에 불과하다.  
`y_escalation` label 자체가 ACLED event count / fatalities로 만들어지므로,  
과거 분쟁 기록은 강력한 예측 신호임에도 불구하고 완전히 제외되었다.

"과거 7일 전까지의 분쟁 기록"은 현실에서도 예측 시점에 관측 가능하므로 (ACLED publication lag ~7일),  
lag을 명시적으로 적용하면 leakage 없이 ACLED 정보를 활용할 수 있다.

---

## 왜 기존 ACLED feature를 그대로 믿지 않고 새로 만드는가

기존 `feature_builder.py`는 기술적으로 `shift(7)` 후 rolling을 적용하여 안전하다.  
그럼에도 새 builder를 별도로 작성하는 이유:

1. **이름에 lag 명시**: `acled_event_count_7d`는 lag이 명시되지 않아 혼란스럽다. 새 feature는 `_lag7` 접미사로 lag을 이름에 명시한다.
2. **실험 독립성**: 팀 pipeline 변경에 영향을 받지 않는 독립 parquet를 실험 재현에 사용한다.
3. **macis_se_score 완전 분리**: 팀 pipeline에서 SE score가 어디서 합쳐지는지 불명확하다. 새 builder는 ACLED-only이다.
4. **actor type 코드 매핑 명시**: processed ACLED의 int 코드를 소스에 주석으로 명확히 기록한다.

---

## y_escalation이 ACLED 기반이어도 과거 lag ACLED feature는 왜 leakage가 아닌가

```
타임라인:
  ... [feature window: ~ t-7] .... 8일 gap .... [label window: t+1 ~ t+3]
```

- `y_escalation(t)`는 t+1~t+3 미래 ACLED 정보로 만든다.
- safe ACLED feature는 최대 t-7일 정보만 사용한다.
- **feature 최신일(t-7)과 label 최초일(t+1) 사이에 8일 gap** → leakage 없음.
- "과거의 분쟁 기록"은 예측 변수로 적법하다.

---

## 7일 publication lag 가정

ACLED는 주간(weekly) 업데이트 방식으로 데이터를 배포한다.  
따라서 t일 기준 실제로 사용 가능한 ACLED 정보는 최대 t-7일까지다.  
이 가정을 `shift(7)` 으로 명시적으로 구현한다.

---

## Feature window와 Label window 분리 구조

| 항목 | 범위 | 비고 |
|------|------|------|
| Feature 최신일 | t - 7 | shift(7) 적용 |
| Feature 7d 시작 | t - 13 | rolling(7) |
| Feature 14d 시작 | t - 20 | rolling(14) |
| Feature 30d 시작 | t - 36 | rolling(30) |
| Label 최초일 | t + 1 | shift(-1) |
| Label 마지막일 | t + 3 | rolling(3) |
| **Gap** | **8일** | ✅ leakage-free |

---

## macis_se_score를 계속 제외하는 이유

- 생성 시점이 불명확 (train-only 생성 의심)
- val/test 예측 시 데이터 가용성 불확실
- leakage 가능성이 조금이라도 있으면 제외 (보수적 원칙)

---

## future label column을 제외하는 이유

`event_count_next3d`, `fatalities_next3d`, `y`, `y_onset`, `y_escalation`은  
t+1~t+3 미래 정보를 직접 포함한다. 이를 feature로 사용하면 완전한 label leakage다.

---

## 입력 / 출력

| 항목 | 경로 |
|------|------|
| 입력 | `conflict-early-warning/input/processed/acled/{iso3}.parquet` |
| 출력 | `members/byeonghyeon/input/processed/acled_safe/safe_acled_lag_features.parquet` |

출력 파일은 `.gitignore`로 추적되지 않음 (재생성 가능).

---

## 생성 Feature (총 15개)

### count / fatalities (9개)

| 피처명 | 계산 방식 | t일 참조 기간 |
|--------|-----------|--------------|
| `safe_acled_event_count_7d_lag7` | shift(7).rolling(7).sum | t-13 ~ t-7 |
| `safe_acled_event_count_14d_lag7` | shift(7).rolling(14).sum | t-20 ~ t-7 |
| `safe_acled_event_count_30d_lag7` | shift(7).rolling(30).sum | t-36 ~ t-7 |
| `safe_acled_fatalities_7d_lag7` | shift(7).rolling(7).sum | t-13 ~ t-7 |
| `safe_acled_fatalities_14d_lag7` | shift(7).rolling(14).sum | t-20 ~ t-7 |
| `safe_acled_fatalities_30d_lag7` | shift(7).rolling(30).sum | t-36 ~ t-7 |
| `safe_acled_fatalities_max_7d_lag7` | shift(7).rolling(7).max | t-13 ~ t-7 |
| `safe_acled_fatalities_max_14d_lag7` | shift(7).rolling(14).max | t-20 ~ t-7 |
| `safe_acled_fatalities_max_30d_lag7` | shift(7).rolling(30).max | t-36 ~ t-7 |

### event type ratio (3개, 30d 기준)

| 피처명 | 분모 | 분자 |
|--------|------|------|
| `safe_acled_ratio_battles_lag7` | all events 30d | Battles 30d |
| `safe_acled_ratio_explosions_lag7` | all events 30d | Explosions/Remote violence 30d |
| `safe_acled_ratio_vac_lag7` | all events 30d | Violence against civilians 30d |

### actor type ratio (2개, 30d 기준)

전체 58개국 processed ACLED 실제 검증 결과:
- `inter1` 실제 코드: **[0, 1, 8]** 만 존재
- `inter2` 실제 코드: **[0, 1, 7, 8]** 만 존재
- 코드 2(Rebel), 3(Political militia), 4~6은 데이터에 없음 → 해당 ratio는 항상 0

| 피처명 | ACLED inter code | 의미 | 데이터 존재 |
|--------|-----------------|------|------------|
| `safe_acled_ratio_state_forces_lag7` | 1 | State forces 관여 비율 | ✅ inter1/inter2 모두 |
| `safe_acled_ratio_external_forces_lag7` | 8 | External/Other forces 관여 비율 | ✅ inter1/inter2 모두 |

> **보류: Civilians (inter code 7)**  
> inter2에만 등장하며, event type ratio에 `safe_acled_ratio_vac_lag7`(Violence against civilians)이  
> 이미 존재하므로 중복 가능성을 피해 이번 버전에서 제외.
>
> **제외: Rebel(2), Political militia(3), Identity militia(4), Rioters(5), Protesters(6)**  
> 전체 58개국 데이터에 존재하지 않음 → 항상 0인 무의미한 feature.

### 결측 마스크 (1개)

| 피처명 | 의미 |
|--------|------|
| `safe_acled_missing_mask` | 해당 국가 첫 이벤트일 + 7일 이전 = 1 (ACLED 커버리지 없음) |

---

## 실행 방법

### 1. Dry-run (입력 파일 확인만, 처리 안 함)

```bash
cd <KUBIG_Conference-team-4 루트>
python members/byeonghyeon/modeling/build_safe_acled_lag_features.py --dry-run
```

### 2. 실제 실행

```bash
python members/byeonghyeon/modeling/build_safe_acled_lag_features.py
```

### 3. 기간 지정

```bash
python members/byeonghyeon/modeling/build_safe_acled_lag_features.py \
  --start 2014-01-01 --end 2025-03-31
```

---

## 실험 F 계획

| 실험 | feature 구성 | 총 feature 수 | 채택 기준 |
|------|-------------|--------------|---------|
| **F0** | B 35 + safe ACLED 15 | **50** | ≥ 0.0594 (B + 0.003) |
| **F1** | F0 50 + C title 21 + coverage_mask 1 | **72** | ≥ 0.0683 (C + 0.003) |
| **F2** | F1 72 + D theme/person 22 | **94** | ≥ F1 + 0.003 |

자세한 실험 계획: `se_free_acled_lag_experiment_plan.md`  
leakage 검토 상세: `acled_lag_leakage_review.md`
