# ACLED-Free Problem Redefinition Plan
## GDELT + 경제지표 기반 Media-Risk / Conflict-Attention Escalation 예측

**작성일**: 2026-06-04  
**작성자**: 김병현  
**상태**: 설계/조사 단계 — 모델 학습/BigQuery 쿼리 실행 없음

---

## 1. 방향 전환 배경

기존 y_escalation 문제:
- ACLED(Armed Conflict Location & Event Data) 기반 사망자/이벤트 spike를 label로 사용
- ACLED 자체가 7일 publication lag를 가지며, feature/label 모두에 의존
- 새로운 GCP 계정 전환 시점을 기회로 ACLED 완전 배제 결정

**새로운 방향**:
- ACLED feature/label 모두 완전 제거
- GDELT(Global Database of Events, Language and Tone) + 경제지표만 사용
- "실제 충돌 예측" → **"미디어 기반 분쟁 attention 상승 예측"** 문제로 전환

---

## 2. 현재 데이터 현황 파악

### 2-1. GDELT GKG 타이틀 테이블 (이미 BQ 수집 완료)

**BQ 테이블**: `conflict-early-warning.conflict_ew.gdelt_titles`  
**파티션**: MONTH(date), CLUSTER BY iso3  
**커버리지**: 2015-02-17 ~ 2025-03-31  
**국가 수**: 58개국

**저장된 raw 컬럼**:

| raw 컬럼 | 타입 | 설명 | 주의 |
|---------|------|------|------|
| `title` | STRING | 기사 제목 `<PAGE_TITLE>` | **2019-09-23 이전 NULL** |
| `url` | STRING | DocumentIdentifier | dedup 키 |
| `domain` | STRING | SourceCommonName | 출처 도메인 |
| `language` | STRING | TranslationInfo srclc | 없으면 'eng' |
| `v2tone_avg` | FLOAT64 | V2Tone 첫 번째 필드 | 2015-02-17부터 유효 |
| `v2themes` | STRING | raw V2Themes (;구분) | 2015-02-17부터 유효 |
| `v2persons` | STRING | raw V2Persons (;구분) | 2015-02-17부터 유효 |
| `date` | DATE | 파티션 키 | |
| `iso3` | STRING | 클러스터 키 | |

**아직 수집하지 않은 GKG raw 컬럼 후보**:

| 컬럼 | 설명 | 활용 가능성 |
|------|------|-----------|
| `v2organizations` | 언급 조직 (정부/군/NGO 등) | 높음 — 주요 행위자 분석 |
| `v2locations` | 언급 지명/좌표 | 중간 — 분쟁 지역 집중도 |
| `v2enhancedthemes` | 상세 테마 계층 (CRIME_*, ENV_* 등) | 높음 — 더 세밀한 테마 분류 |
| `V2Tone` (full 6-field) | tone, positivity, negativity, polarity, ARD, WDSD | 중간 — polarity, ARD 추가 가능 |
| `SRCLC` | 원문 언어 코드 (source language) | 낮음 — language와 중복 |
| `SharingImage` | 이미지 URL 포함 여부 | 낮음 |

### 2-2. GDELT Title Features (parquet 이미 생성 완료)

**경로**: `input/processed/gdelt_titles/gdelt_title_features.parquet`  
**shape**: (214,257, 23) = 58국 × ~3,694일  
**커버리지**: 2015-02-17 ~ 2025-03-31  
**coverage_mask**: date < 2015-02-17이면 학습 스크립트에서 1로 처리 (parquet에 없음)

**21개 feature (+ date, country)**:

```
1d features (10개):
  gdelt_title_count_1d           하루 기사 건수
  gdelt_title_nonnull_count_1d   title 비결측 건수 (2019-09 이후만 유효)
  gdelt_title_tone_mean_1d       평균 V2Tone
  gdelt_title_tone_std_1d        V2Tone 표준편차
  gdelt_title_tone_min_1d        최저 V2Tone
  gdelt_title_negative_count_1d  tone < -5 기사 수
  gdelt_title_positive_count_1d  tone > +2 기사 수
  gdelt_title_eng_count_1d       영어 기사 수
  gdelt_title_domain_diversity_1d 고유 도메인 수 (출처 다양성)
  gdelt_title_lang_diversity_1d  고유 언어 수

7d rolling features (11개):
  gdelt_title_count_7d
  gdelt_title_nonnull_count_7d
  gdelt_title_negative_count_7d
  gdelt_title_positive_count_7d
  gdelt_title_eng_count_7d
  gdelt_title_tone_mean_7d
  gdelt_title_tone_std_7d
  gdelt_title_tone_min_7d
  gdelt_title_domain_diversity_7d
  gdelt_title_lang_diversity_7d
  gdelt_title_tone_trend_7d      tone_mean_1d - 7일 전 tone_mean (추세)
```

**BQ 쿼리 비용**: 약 82.8 GB / 월 1TB 무료 내 → $0 (단, 재실행 시 재계산)

### 2-3. GDELT Theme/Person Features (parquet 이미 생성 완료)

**경로**: `input/processed/gdelt_titles/gdelt_theme_person_features.parquet`  
**shape**: (214,257, 24) = 58국 × ~3,694일  
**커버리지**: 2015-02-17 ~ 2025-03-31

**22개 feature (+ date, country)**:

```
1d features (11개):
  gdelt_theme_nonnull_count_1d     v2themes 비결측 기사 수
  gdelt_person_nonnull_count_1d    v2persons 비결측 기사 수
  gdelt_theme_count_1d             총 테마 토큰 수 (';' 구분)
  gdelt_person_count_1d            총 인물 토큰 수
  gdelt_theme_conflict_count_1d    CONFLICT/MILITARY_ATTACK/ARMED/WAR 테마 기사 수
  gdelt_theme_protest_count_1d     PROTEST/RIOT/CIVIL_UNREST/STRIKE/DEMONSTRATION 기사 수
  gdelt_theme_military_count_1d    MILITARY 테마 기사 수
  gdelt_theme_refugee_count_1d     REFUGEE/DISPLACED/ASYLUM/HUMANITARIAN 기사 수
  gdelt_theme_sanction_count_1d    SANCTION/EMBARGO 테마 기사 수
  gdelt_theme_government_count_1d  GOV_/ELECTION/COUP/CEASEFIRE/BLOCKADE 기사 수
  gdelt_person_density_1d          기사당 평균 인물 토큰 수

7d rolling features (11개):
  위 모든 1d feature의 7d rolling sum/avg
```

**BQ 쿼리 비용**: 약 1,282 GB (~$6.26) — 재실행 시 비용 발생 주의

### 2-4. GDELT Events Features (train.parquet에 이미 포함)

**BQ 테이블**: `gdelt-bq.gdeltv2.events_partitioned`  
**수집 컬럼**: GLOBALEVENTID, SQLDATE, ActionGeo_CountryCode, EventCode, EventRootCode, QuadClass, GoldsteinScale, NumMentions, NumArticles, AvgTone

**19개 feature (이미 train.parquet에 포함)**:

```
Goldstein Scale (평화-전쟁 스케일, -10~+10):
  gdelt_goldstein_mean_7d / 14d / 30d
  gdelt_goldstein_std_7d / 14d / 30d

AvgTone (보도 톤):
  gdelt_tone_mean_7d / 14d / 30d

NumMentions (총 언급 수):
  gdelt_mentions_sum_7d / 14d / 30d

Event Count:
  gdelt_event_count_7d / 14d / 30d

QuadClass 비율 (4분류):
  gdelt_quadclass_1_ratio  (Verbal Cooperation)
  gdelt_quadclass_2_ratio  (Material Cooperation)
  gdelt_quadclass_3_ratio  (Verbal Conflict)
  gdelt_quadclass_4_ratio  (Material Conflict)
```

**수집했지만 아직 feature로 변환 안 된 Events 컬럼**:
- `EventCode` (2-4자리 CAMEO 코드) — 세부 이벤트 유형 (현재 root code만 사용)
- `NumArticles` — 기사 수 (NumMentions과 별개)
- Actor 관련 컬럼 — 현재 미수집 (Actor1Code, Actor2Code, Actor1Name 등)

### 2-5. 경제지표 Features (train.parquet에 이미 포함)

**수집 방법**: yfinance + FRED API

**15개 feature**:

```
VIX (글로벌 변동성 지수):
  econ_vix, econ_vix_pct_1d, econ_vix_pct_7d

WTI 원유 가격:
  econ_wti, econ_wti_pct_1d, econ_wti_pct_7d

Gold 금 가격:
  econ_gold, econ_gold_pct_1d, econ_gold_pct_7d

DXY 달러 인덱스:
  econ_dxy, econ_dxy_pct_1d, econ_dxy_pct_7d

STLFSI4 (St. Louis 금융스트레스지수):
  econ_stlfsi4, econ_stlfsi4_pct_1d, econ_stlfsi4_pct_7d
```

---

## 3. BigQuery Raw Column 현황 정리

### 이미 읽어서 feature로 변환한 컬럼

| 소스 | raw 컬럼 | 변환된 feature |
|------|---------|--------------|
| gdelt_titles | v2tone_avg | tone_mean, tone_std, tone_min, negative_count, positive_count |
| gdelt_titles | v2themes | conflict/protest/military/refugee/sanction/government count |
| gdelt_titles | v2persons | person_count, person_density |
| gdelt_titles | language | eng_count, lang_diversity |
| gdelt_titles | domain | domain_diversity |
| gdelt_titles | title | nonnull_count만 (텍스트 미활용) |
| events | GoldsteinScale | goldstein_mean, goldstein_std |
| events | AvgTone | tone_mean |
| events | NumMentions | mentions_sum |
| events | QuadClass | quadclass_1~4_ratio |
| events | SQLDATE | event_count |

### 읽었지만 아직 활용 안 된 컬럼

| 소스 | 컬럼 | 활용 가능성 |
|------|------|-----------|
| gdelt_titles | title 텍스트 | NLP 가능 (2019-09 이후), 현재 미활용 |
| gdelt_titles | url | 중복 제거 외 미활용 |
| events | EventCode (2-4자리) | 세부 이벤트 분류 가능 |
| events | NumArticles | NumMentions과 비교 가능 |

### 아직 수집하지 않은 유용한 GKG 컬럼 후보

| 컬럼 | 추정 BQ 비용 | 활용 가치 | 추천 순위 |
|------|------------|---------|---------|
| `v2organizations` | ~동급 (~1TB) | 행위자 분석 (군/정부/NGO) | ⭐⭐⭐ |
| `v2enhancedthemes` | ~동급 | 상세 테마 (GKG 2.0 분류) | ⭐⭐⭐ |
| `V2Tone` full (6 fields) | ~동급 | ARD(절대적 감성 편차), polarity | ⭐⭐ |
| `v2locations` | ~동급 | 지리적 집중도 | ⭐ |

---

## 4. ACLED-Free Target 후보 설계

> **전제**: ACLED 데이터는 label 생성에도 절대 사용하지 않는다.  
> **핵심 질문**: GDELT 기반 target은 "실제 conflict"가 아니라 "미디어 주목도/보도 패턴의 변화"를 예측한다.

### Target A: GDELT Conflict-Theme Volume Spike (권장 ⭐⭐⭐)

```
정의:
  y_conflict_theme_spike(t) = 1 if
    gdelt_theme_conflict_count_3d(t+1~t+3) / 3
    > k × rolling_mean(gdelt_theme_conflict_count, past 14d)
    AND gdelt_theme_conflict_count_3d(t+1~t+3) >= min_abs_threshold

  k = 2.0~3.0 (탐색 필요)
  min_abs_threshold = 3~5 기사 (데이터 기반 결정)
```

| 항목 | 내용 |
|------|------|
| 필요 raw 컬럼 | v2themes (이미 수집) |
| label window | t+1 ~ t+3 (feature window ≤ t-7) → 8일 gap |
| 장점 | 직접 분쟁 관련 보도 spike 포착. 해석 명확 |
| 단점 | 미디어 agenda-setting에 종속 (실제 충돌 ≠ 보도 급증) |
| leakage 위험 | v2themes가 real-time이므로 publication lag 없음 → feature window 관리 중요 |
| 해석 | "향후 3일 내 분쟁 관련 보도가 급증할 국가" |
| 대시보드 | "미디어 분쟁 관심도 급상승 경보" |

### Target B: V2Tone Negative Spike (권장 ⭐⭐⭐)

```
정의:
  y_tone_spike(t) = 1 if
    mean(v2tone_avg, t+1~t+3) < tone_threshold_low
    AND mean(v2tone_avg, t+1~t+3) < rolling_min(v2tone_avg, past 30d) × 0.8

  tone_threshold_low = -5.0 (탐색 필요)
```

| 항목 | 내용 |
|------|------|
| 필요 raw 컬럼 | v2tone_avg (이미 수집) |
| label window | t+1 ~ t+3 |
| 장점 | 가장 안정적으로 수집된 컬럼. 크로스-국가 비교 용이 |
| 단점 | 부정적 보도 ≠ 분쟁 (선거, 경제위기, 재해도 음수 tone) |
| leakage 위험 | 낮음 — tone은 이미 집계값 |
| 해석 | "향후 3일 내 해당 국가 보도 tone이 극도로 부정적으로 변할 것" |
| 대시보드 | "미디어 부정 보도 급증 경보" |

### Target C: Multi-Signal Composite Media-Risk Spike (권장 ⭐⭐)

```
정의:
  composite_score(t) =
    w1 * conflict_theme_z(t) +
    w2 * negative_tone_z(t) +
    w3 * protest_theme_z(t) +
    w4 * military_theme_z(t)
    (z-score 기준, 국가별 rolling 30d normalization)

  y_composite_spike(t) = 1 if composite_score(t+1~t+3) > threshold
```

| 항목 | 내용 |
|------|------|
| 필요 raw 컬럼 | v2themes, v2tone_avg (이미 수집) |
| label window | t+1 ~ t+3 |
| 장점 | 단일 신호 노이즈 줄임. 다면적 분쟁 포착 |
| 단점 | 가중치 w1~w4 결정 근거 필요. 설명력 복잡 |
| leakage 위험 | 중간 — z-score 계산에 미래 포함 주의 |
| 해석 | "미디어가 다각도로 분쟁 신호를 동시에 급증시킬 국가" |
| 대시보드 | "복합 미디어 위험 지수 상승" |

### Target D: Article Volume Spike (보조 후보)

```
정의:
  y_volume_spike(t) = 1 if
    count_3d(t+1~t+3) / 3 > k × rolling_mean(count, past 14d)
    AND count_3d >= min_abs_count
```

| 항목 | 내용 |
|------|------|
| 장점 | 가장 단순. 구현 쉬움 |
| 단점 | 실제 분쟁과 무관한 이벤트(선거, 정상회담)도 포착. 예측 가치 낮음 |
| 해석 | "미디어 보도량이 갑자기 급증할 국가" |
| 대시보드 | 단독 사용 부적합. 다른 target의 보조 feature로 활용 권장 |

### Target E: Protest + Refugee Theme Onset (온셋 탐지)

```
정의:
  y_protest_onset(t) = 1 if
    protest_or_refugee_count_3d(t+1~t+3) > 0
    AND protest_or_refugee_count_7d(t-7~t) == 0  (완전 평화 → 갑작스런 등장)
```

| 항목 | 내용 |
|------|------|
| 장점 | y_onset과 개념 유사. 조기 경보에 특화 |
| 단점 | positive rate 매우 낮을 수 있음. 국가별 편차 큼 |
| 해석 | "완전히 조용했다가 갑자기 시위/난민 보도가 시작" |
| 대시보드 | 온셋 경보로 의미 있음 |

### Target F: Goldstein Scale Rapid Deterioration

```
정의:
  y_goldstein_drop(t) = 1 if
    mean(GoldsteinScale, t+1~t+3) - mean(GoldsteinScale, t-7~t) < -3.0
```

| 항목 | 내용 |
|------|------|
| 필요 데이터 | gdelt_goldstein_mean (이미 train.parquet에 있음) |
| 장점 | 기존 feature를 label로 전환 가능. 추가 BigQuery 비용 없음 |
| 단점 | Goldstein scale이 보도 이벤트 기반 → GDELT events 수집 의존 |
| 해석 | "7일 내 보도되는 이벤트의 갈등 강도가 급격히 악화" |
| 대시보드 | "갈등 이벤트 악화 경보" |

---

## 5. 권장 ACLED-Free Target 후보 2~3개

### 1순위: Target A (Conflict-Theme Volume Spike)

**이유**:
- v2themes의 CONFLICT/MILITARY_ATTACK/WAR 테마는 GDELT가 자동 분류한 실질적 분쟁 신호
- 이미 BQ에서 수집 완료 → 추가 쿼리 비용 없음
- 국가별 rolling 정규화로 만성 분쟁국(ISR, AFG)의 포화 문제를 일부 해소 가능
- 운영 시점에 실시간 재현 가능 (GDELT는 15분 업데이트)

**구현 단계**:
1. parquet에서 gdelt_theme_conflict_count 3d future rolling 계산
2. 국가별 past 14d rolling mean과 비교
3. k, min_abs_threshold 탐색 (목표 positive rate: 5~15%)

### 2순위: Target B (V2Tone Negative Spike) 또는 F (Goldstein Drop)

**이유**:
- V2Tone은 GKG table의 핵심 신호. 모든 기사에 대해 계산됨
- Goldstein은 이미 train.parquet에 있어 즉시 구현 가능 (추가 BQ 비용 zero)
- 두 신호 중 Target A와 낮은 상관관계를 가지는 쪽이 유용

### 3순위: Target C (Composite Score) — 추후 검토

- A + B를 각각 검증한 뒤 앙상블 label로 발전 가능
- 단독으로 시작하기보다 A/B 검증 후 결합 방식으로 접근 권장

---

## 6. ACLED-Free Feature Set 설계

### 사용 가능 feature (기존 + 신규 후보)

#### A. GDELT GKG Title/Tone (이미 있음, 21개)

```
1d/7d tone, count, domain_diversity, lang_diversity, negative/positive_count
tone_trend_7d (추세)
```

#### B. GDELT GKG Theme/Person (이미 있음, 22개)

```
conflict/protest/military/refugee/sanction/government count (1d/7d)
person_density (1d/7d)
```

#### C. GDELT Events (이미 있음, 19개)

```
goldstein_mean/std (7d/14d/30d)
tone_mean (7d/14d/30d)
mentions_sum (7d/14d/30d)
event_count (7d/14d/30d)
quadclass_1~4_ratio
```

#### D. 경제지표 (이미 있음, 15개)

```
vix, wti, gold, dxy, stlfsi4 (각 level + pct_1d + pct_7d)
```

#### E. 신규 후보 (BigQuery 추가 수집 필요)

```
v2organizations count → 행위자 다양성 (정부/군/국제기구/NGO)
v2enhancedthemes → 세부 분류 (CRIME_VIOLENCE, ENV_*, ECON_* 등)
V2Tone ARD (절대 감성 편차) → tone 극화 정도
conflict_theme_ratio = conflict_count / total_count → 비율 feature
negative_tone_ratio → 부정 기사 비율
```

#### F. 파생 feature (추가 계산 가능)

```
tone_volatility = tone_std_7d (이미 있음)
theme_diversity = 테마 종류 수 (v2enhancedthemes에서 계산)
conflict_acceleration = conflict_count_3d / conflict_count_14d (비율)
gdelt_event_count_ratio_quadclass34 = (QC3+QC4)/(QC1+QC2+QC3+QC4)
```

#### 제외 대상 (ACLED 관련)

```
❌ acled_event_count_*, acled_fatalities_*, acled_ratio_*
❌ safe_acled_event_count_*, safe_acled_fatalities_*, safe_acled_ratio_*
❌ enhanced_safe_acled_*
❌ y, y_onset, y_escalation
❌ fatalities_next3d, event_count_next3d, past14d_event_count, past14d_fatalities_mean
❌ macis_se_score
```

---

## 7. LGBM / XGBoost / LSTM 모델링 설계

### 7-1. Tabular 모델 (LGBM / XGBoost)

**Feature matrix**: (date × country) 단위, ~58국 × n일

```
Feature group       | 수   | 출처
--------------------|------|-----------------------------
GDELT title/tone    | 21   | gdelt_title_features.parquet
GDELT theme/person  | 22   | gdelt_theme_person_features.parquet
GDELT events        | 19   | train.parquet (기존 GDELT 컬럼)
경제지표             | 15   | train.parquet (econ_* 컬럼)
country (cat)       |  1   | ISO3 문자열
시간 feature        |  3~5 | month, quarter, day_of_week 등
계                  | ~81  |
```

**Target alignment**:

```
t일 feature → y(t+1~t+3) 예측
feature window: 최대 t (당일까지 포함 가능, GDELT는 real-time)
label window: t+1 ~ t+3
gap: 1일 이상 (ACLED처럼 7일 lag 불필요 — GDELT는 즉시 반영)

주의: label을 GDELT 기반으로 만들면 feature-label 간 style shift 위험이 낮지만
  "미래 GDELT로 미래 GDELT를 예측"하는 구조이므로
  특징 변수의 rolling window가 label과 겹치지 않도록 명확히 구분해야 함
```

**Cleanval split 유지 가능 여부**:

```
train_fit : 2015-02-17 ~ 2022-12-31  (GDELT coverage 시작)
tune_cal  : 2023-01-01 ~ 2023-12-31
val_eval  : 2024-01-01 ~ 2024-06-30
test      : 2024-07-01 ~             (평가 금지)

2014-01-01 ~ 2015-02-17: GDELT GKG 커버리지 없음
  → coverage_mask=1로 처리하거나 해당 기간 완전 제외
  → train_fit 시작을 2015-02-17로 조정 권장
```

### 7-2. LSTM 시퀀스 모델

**Dataset 구조**:

```
X: (n_samples, lookback, n_features) — country별 시계열
y: (n_samples,) — binary target at t+1~t+3

lookback 후보: 30일, 60일, 90일
  → GDELT coverage 시작 2015-02-17 + lookback = 실제 학습 시작일
  → lookback=60이면 train 시작 2015-04-18

Feature alignment:
  X[i, :, :] = features from t-lookback+1 to t
  y[i]       = target at t+1~t+3
  gap 없음 (X의 마지막 행이 t, y는 t+1~t+3)

Missing 처리:
  GDELT 커버리지 없는 날 → 0 패딩 또는 마스크
  경제지표 weekend/holiday → ffill
  coverage_mask feature 추가 권장
```

**Country 단위 분리 vs 전체 pooling**:

```
옵션 A: 전체 pooling (권장 시작점)
  - 58국 전체를 하나의 dataset으로 학습
  - country를 feature로 포함 (embedding 또는 one-hot)
  - 샘플 수: 58 × ~3,000일 = ~174,000

옵션 B: Country-specific LSTM
  - 국가별 개별 모델
  - 샘플 수 부족 위험 (1개국 = ~3,000일)
  - 추후 검토
```

---

## 8. 결론 및 핵심 질문

### 8-1. 문제 정의 변화

| 항목 | 기존 (y_escalation) | 신규 (ACLED-free) |
|------|-------------------|-----------------|
| Label 소스 | ACLED 사망자/이벤트 | GDELT 보도 패턴 |
| 예측 대상 | 실제 무력 충돌 급증 | 미디어 분쟁 attention 급증 |
| Feature 소스 | ACLED + GDELT + 경제 | GDELT + 경제 |
| Publication lag | ACLED 7일 | GDELT 없음 (15분 업데이트) |
| 해석 | "이 나라에서 충돌이 일어날 것" | "이 나라 분쟁 관련 보도가 급증할 것" |

**핵심 경고**: 새 target은 실제 conflict escalation이 아니라 **미디어 보도 패턴의 변화**를 예측한다. "보도가 많다 ≠ 실제 충돌이 있다"는 점을 대시보드에 명확히 표시해야 한다.

### 8-2. GDELT 기반 target 설계의 최대 위험

1. **Agenda-Setting Bias**: 미디어가 주목하는 국가와 실제로 위험한 국가가 다를 수 있음 (ISR 과대, 내륙 아프리카 과소)
2. **Autocorrelation**: GDELT feature와 GDELT label이 같은 소스 → 모델이 "오늘 많이 보도된 나라가 내일도 많이 보도될 것"을 학습할 위험
3. **Coverage Gap**: 2015-02-17 이전 GKG 커버리지 없음 → 학습 데이터 6년 단축
4. **Language Bias**: 영어 보도 중심 → 프랑스어권/아랍어권 국가 과소 대표
5. **Circular Label**: label 생성에 쓰는 컬럼과 feature 컬럼이 동일 소스 → 공식적으로 leakage는 아니지만 해석상 주의 필요

### 8-3. 추가로 BigQuery에서 가져와야 할 컬럼

| 우선순위 | 컬럼 | 비용 | 이유 |
|---------|------|------|------|
| ⭐⭐⭐ | v2organizations | ~1TB (~$0 무료 내 / 월 1TB 리셋 후) | 행위자 다양성 |
| ⭐⭐⭐ | V2Tone full (ARD, polarity) | 기존 쿼리 수정으로 추가 비용 미미 | 감성 극화 |
| ⭐⭐ | v2enhancedthemes | ~1TB | 더 세밀한 테마 |
| ⭐ | EventCode 2-4자리 | 기존 events 쿼리 수정 | 세부 이벤트 분류 |

### 8-4. 모델링 전 팀원들과 합의해야 할 핵심 질문

```
Q1. 새 target을 "미디어 분쟁 관심도 상승"으로 명확히 재정의하는 데 동의하는가?
    → 대시보드 해석 방식이 근본적으로 달라짐

Q2. GDELT coverage 시작일(2015-02-17)로 train 시작을 늦추는 데 동의하는가?
    → 기존 2014-01-01 시작 대비 ~1.1년 데이터 손실

Q3. 새 positive rate는 몇 %를 목표로 하는가?
    → 기존 y_escalation 4.07% → 신규 target은 5~15% 수준 목표 권장

Q4. label window는 3일을 유지할 것인가, 7일로 확장할 것인가?
    → GDELT는 즉시 반영되므로 3일이 더 현실적

Q5. LGBM/XGBoost 단독 vs LSTM 포함 stacking 중 무엇을 먼저 구현할 것인가?
    → tabular 먼저 검증 후 LSTM 추가 권장

Q6. v2organizations 등 추가 컬럼 수집을 위한 BigQuery 비용(~$6)을 승인하는가?
    → theme/person 쿼리와 동급 비용 예상

Q7. 경제지표를 계속 포함할 것인가 (글로벌 지표 → 국가별 연결 불명확)?
    → 분리 실험 권장 (econ_* ablation)

Q8. 새 target 기준 PR-AUC 채택 threshold를 어떻게 설정할 것인가?
    → positive rate가 달라지므로 Lift@top5% 기준 사용 권장
```

---

## 9. 다음 단계 제안 (우선순위 순)

```
[1] 팀 합의 (Q1~Q8)
[2] Target A 정의 확정 → positive rate 탐색 (BQ 없이 parquet로 가능)
[3] ACLED-free feature matrix 구성 (기존 parquet 병합으로 즉시 가능)
[4] LGBM baseline 학습 및 검증 (BigQuery 없이 기존 parquet만으로)
[5] v2organizations / V2Tone full 추가 수집 (BQ 비용 승인 후)
[6] LSTM 시퀀스 모델 추가 (tabular 검증 후)
```

---

*이 문서는 조사/설계 단계 결과물이다. BigQuery 쿼리 실행/모델 학습/test 평가는 수행하지 않았다.*  
*생성: 2026-06-04*
