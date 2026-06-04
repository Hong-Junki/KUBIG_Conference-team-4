# ACLED Lag Leakage Review

**작성일**: 2026-06-04  
**검토 대상**: `src/process/label_builder.py`, `src/process/feature_builder.py`

---

## 1. Label 정의 요약

`label_builder.py` 기준 `y_escalation(t)` 정의:

```
y_escalation(t) = y_onset(t) OR esc_spike(t)

y_onset(t)   = (past14d_event_count(t) == 0) AND (event_count_next3d(t) > 0)
esc_spike(t) = fatalities_next3d(t) / 3  >  past14d_fatalities_mean(t) + 2 * past14d_fatalities_std(t)
               (과거 기준 0이면 fatalities_next3d > 0 이면 spike로 처리)
```

### 윈도우 정의

| 변수 | 계산 방식 | 참조 기간 |
|------|-----------|-----------|
| `event_count_next3d(t)` | shift(-1).rolling(3).sum | **t+1 ~ t+3** (미래 3일) |
| `fatalities_next3d(t)` | shift(-1).rolling(3).sum | **t+1 ~ t+3** (미래 3일) |
| `past14d_event_count(t)` | shift(1).rolling(14).sum | t-14 ~ t-1 (과거 14일) |
| `past14d_fatalities_mean(t)` | shift(1).rolling(14).mean | t-14 ~ t-1 |
| `past14d_fatalities_std(t)` | shift(1).rolling(14).std | t-14 ~ t-1 |

### 핵심 확인: label window

> **y_escalation(t)의 label window = t+1 ~ t+3**

label은 t 시점에서 3일 후 미래 분쟁 escalation을 예측하는 이진 타겟이다.  
y, y_onset, y_escalation 모두 미래 3일(t+1~t+3) 데이터를 직접 사용하여 계산된다.

### y / y_onset / y_escalation 차이

| label | 정의 | 양성 조건 |
|-------|------|-----------|
| `y` | continuation 포함 — 기존 분쟁 여부 무관 | event_count_next3d > 0 |
| `y_onset` | 엄격 onset — 과거 14일 평화 상태에서 새 분쟁 발생 | past14d_event_count == 0 AND event_count_next3d > 0 |
| `y_escalation` | onset + 급격 악화 — **주 학습 타겟** | y_onset OR (fatalities 급증) |

실험 타겟: `y_escalation`

---

## 2. 기존 ACLED Feature 생성 로직 점검

`feature_builder.py` → `_build_acled_features()` 분석.

### 핵심 코드

```python
ACLED_LAG_DAYS = 7  # 명시적 lag 상수

shifted_count = daily_counts["event_count"].shift(ACLED_LAG_DAYS)  # shift(7)
features[f"acled_event_count_{w}d"] = shifted_count.rolling(w, min_periods=1).sum().values
```

### Rolling 방식: shift 후 rolling

`shift(7)` 후 `rolling(w)` 이므로:

| feature | t일 기준 참조 기간 |
|---------|-------------------|
| `acled_event_count_7d[t]` | daily[t-13 ~ t-7] (7일 합계, 최신일 t-7) |
| `acled_event_count_14d[t]` | daily[t-20 ~ t-7] (14일 합계, 최신일 t-7) |
| `acled_event_count_30d[t]` | daily[t-36 ~ t-7] (30일 합계, 최신일 t-7) |
| `acled_fatalities_*` | 동일 패턴 |
| `acled_ratio_*` | shift(7).rolling(30) → daily[t-36 ~ t-7] |
| `acled_actor_type_*_ratio` | 동일 패턴 |

### Feature window vs Label window 비교

```
t일 기준:
  feature window 최신일: t - 7   (shift(7) 적용)
  label window  최초일: t + 1

gap = (t+1) - (t-7) = 8일 ✅
```

**기존 feature_builder.py는 leakage-safe하다.**  
`ACLED_LAG_DAYS = 7`이 명시적으로 정의되어 있고, shift → rolling 순서도 올바르다.

---

## 3. 기존 ACLED Feature를 그대로 쓰지 않는 이유

기술적으로 안전하더라도 아래 이유로 새 safe builder를 별도 작성한다:

1. **재현성 명시**: 기존 builder는 팀 전체 feature table을 생성하는 파이프라인의 일부이며, 인자나 파이프라인 변경이 있을 경우 결과가 달라질 수 있다. 독립 builder는 이 실험 전용 재현성을 보장한다.

2. **feature 이름에 lag 명시**: `acled_event_count_7d`라는 이름은 "7일 rolling count"를 의미하지만, 실제로는 shift(7) 후 rolling이므로 "7일 rolling, 최신일 t-7"이다. 새 feature는 `_lag7` 접미사로 lag을 이름에 명시한다.

3. **actor type 코드 검증 필요**: 기존 builder에서 `inter1/inter2` 코드(1~8)는 processed ACLED에서 int64로 저장된다. raw ACLED에서는 문자열이었다가 processed에서 numeric으로 변환되었는데, 이 매핑이 ACLED codebook과 일치하는지 명시적으로 검증한 기록이 없다. 새 builder는 코드 매핑을 소스에 주석으로 명시한다.

4. **macis_se_score 완전 분리**: 기존 팀 pipeline에서 macis_se_score가 어디서 합쳐지는지 불명확하다. 새 builder는 ACLED-only이며 SE score를 일절 포함하지 않는다.

5. **실험 통제**: 이번 실험 F는 "safe ACLED lag feature를 추가했을 때 성능이 올라가는가"를 검증한다. 기존 pipeline 전체가 아닌 특정 feature set만 추가/제거하려면 독립 parquet가 필요하다.

---

## 4. Feature Window vs Label Window 비교표

```
타임라인:
──────────────────────────────────────────────────────────────────▶ t
  ...  t-36  t-30  t-20  t-14  t-13  t-7   t   t+1  t+2  t+3
                                      │     │        │         │
             acled_30d 시작           │     │   label window   │
                   acled_14d 시작 ────┘     │                 │
                         acled_7d 시작ᵍ     │                 │
                                            │ ←─ gap 8일 ──→  │
                                            t         label end│
                                       feature       (t+1~t+3) │
                                       최신일(t-7)             │
```

| 항목 | 범위 |
|------|------|
| Feature 최신일 (shift 7) | t - 7 |
| Label 최초일 | t + 1 |
| **Gap** | **8일 이상 ✅** |
| Feature 30d 시작일 | t - 36 |
| Feature 14d 시작일 | t - 20 |
| Feature 7d 시작일 | t - 13 |

---

## 5. Safe ACLED Feature를 새로 만드는 이유 요약

- `y_escalation`이 ACLED 기반 label이더라도, **과거 ACLED lag feature**는 leakage가 아니다.  
  feature window 최신일(t-7)과 label window 최초일(t+1) 사이에 **8일 gap**이 존재한다.
- "과거의 분쟁 기록"은 현실에서도 관측 가능하므로 예측 변수로 적합하다.
- 문제가 되는 것은 macis_se_score처럼 **생성 시점이 불명확**하거나 **미래 정보를 포함**할 수 있는 변수다.

---

## 6. Leakage 방지 체크리스트

| 항목 | 확인 결과 |
|------|-----------|
| feature window 최신일 ≤ t-7 | ✅ shift(7) 적용 |
| label window = t+1 ~ t+3 | ✅ label_builder 코드 확인 |
| feature-label gap ≥ 1일 | ✅ 8일 gap |
| macis_se_score 제외 | ✅ builder에 포함 안 함 |
| event_count_next3d 제외 | ✅ future label |
| fatalities_next3d 제외 | ✅ future label |
| y / y_onset / y_escalation 제외 | ✅ label columns |
| 문자열 원본(actor name 등) 저장 안 함 | ✅ 집계 count만 저장 |
| date dtype UTC-aware | ✅ tz_localize("UTC") |
| nullable Int64 → int64 변환 | ✅ 변환 로직 포함 |
| actor type 코드 실증 검증 | ✅ 전체 58개국 inter 코드 확인: [0,1,8]/[0,1,7,8] |
| 코드 2/3 (Rebel/Political militia) 제외 | ✅ 데이터에 존재하지 않아 제거 |
| External forces(8) 포함 | ✅ 실제 존재 확인 후 추가 |
| Civilians(7) 보류 | ✅ VAC ratio와 중복 가능성으로 이번 버전 제외 |
