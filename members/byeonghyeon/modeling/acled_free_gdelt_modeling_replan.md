# ACLED-Free GDELT 운영 모델 재설계 계획
## GDELT + 경제지표 기반 Conflict-Attention Escalation Prediction

**작성일**: 2026-06-04  
**작성자**: 김병현  
**상태**: 설계/조사 단계 — 모델 학습/BigQuery 쿼리 실행 없음

---

## 1. 왜 ACLED-Free 운영 모델이 필요한가

### 운영 환경 제약

| 항목 | ACLED | GDELT |
|------|-------|-------|
| 업데이트 주기 | 주 1회 (weekly) | 15분 (near real-time) |
| API 가용성 | 제한적 (계약 필요) | 무료 (BigQuery / DOC 2.0) |
| Lag | 7일 publication lag | 사실상 없음 |
| 실시간 서빙 가능 | **불가** | **가능** |

대시보드는 매일 오전 자동 업데이트를 목표로 한다. ACLED는 주 1회 업데이트되므로 실시간 운영에서 사용 불가. **운영 모델의 feature는 GDELT + 경제지표만으로 구성해야 한다.**

---

## 2. y_escalation을 target으로 계속 써도 되는 이유

```
훈련/평가 시:  y_escalation = ACLED 기반 label
운영 예측 시:  feature = GDELT + 경제지표 (ACLED 없음)
              target(예측 대상) = y_escalation (과거 정답 학습)
```

**y_escalation은 학습 신호(정답 레이블)로만 쓰이고, 운영 시점의 input feature로는 쓰이지 않는다.** 따라서 ACLED 데이터에 학습 레이블 의존성이 있어도 무방하다. 이는 "label로서의 ACLED"와 "feature로서의 ACLED"를 명확히 분리하는 것이다.

---

## 3. ACLED Feature를 Input으로 쓰면 안 되는 이유

| 문제 | 내용 |
|------|------|
| 운영 불가 | 대시보드 서빙 시점에 최신 ACLED 없음 |
| Lag 불일치 | safe_acled shift(7) 적용해도 운영 재현 불가 |
| 데이터 접근 | ACLED API 제한 / 비용 발생 |
| 예측 신뢰성 | ACLED 없는 환경에서 성능 급락 가능 |

**금지 feature 목록**:
```
❌ acled_event_count_*, acled_fatalities_*, acled_ratio_*
❌ acled_actor_type_*_ratio, acled_missing_mask
❌ safe_acled_event_count_*, safe_acled_fatalities_*, safe_acled_ratio_*
❌ safe_acled_missing_mask
❌ enhanced_safe_acled_*
```

---

## 4. 기존 GDELT Title Feature 정리

### 4-1. Parquet 현황

| 항목 | 값 |
|------|-----|
| 경로 | `input/processed/gdelt_titles/gdelt_title_features.parquet` |
| Shape | (214,257, 23) |
| Coverage | 2015-02-17 ~ 2025-03-31 |
| 국가 수 | 58개국 |
| BQ 소스 | `conflict-early-warning.conflict_ew.gdelt_titles` |

### 4-2. Feature 목록 (21개 + date/country)

**1d Feature (10개)**:
```
gdelt_title_count_1d           하루 기사 건수
gdelt_title_nonnull_count_1d   title 비결측 건수 (2019-09-23 이후만 유효)
gdelt_title_tone_mean_1d       평균 V2Tone
gdelt_title_tone_std_1d        V2Tone 표준편차
gdelt_title_tone_min_1d        최저 V2Tone
gdelt_title_negative_count_1d  tone < -5 기사 수
gdelt_title_positive_count_1d  tone > +2 기사 수
gdelt_title_eng_count_1d       영어 기사 수
gdelt_title_domain_diversity_1d 고유 도메인 수 (출처 다양성)
gdelt_title_lang_diversity_1d  고유 언어 수
```

**7d Rolling Feature (11개)**:
```
gdelt_title_count_7d           7일 기사 수 합계
gdelt_title_nonnull_count_7d   7일 title 비결측 합계
gdelt_title_negative_count_7d  7일 부정 기사 수
gdelt_title_positive_count_7d  7일 긍정 기사 수
gdelt_title_eng_count_7d       7일 영어 기사 수
gdelt_title_tone_mean_7d       7일 평균 톤
gdelt_title_tone_std_7d        7일 톤 표준편차 평균
gdelt_title_tone_min_7d        7일 최저 톤
gdelt_title_domain_diversity_7d 7일 도메인 다양성 평균
gdelt_title_lang_diversity_7d  7일 언어 다양성 평균
gdelt_title_tone_trend_7d      tone_mean_1d - LAG(7일) 톤 추세
```

### 4-3. Coverage 및 결측 처리

| 항목 | 내용 |
|------|------|
| Coverage 시작 | 2015-02-17 (그 이전 데이터 없음) |
| coverage_mask | date < 2015-02-17이면 1 (학습 스크립트에서 직접 설정) |
| title 텍스트 | 2019-09-23 이전 NULL (v2tone_avg/v2themes는 유효) |
| 결측 처리 | left-join 시 fillna(0) |

### 4-4. 기존 실험 기여도

| 실험 | Feature 추가 | Stacking Platt PR-AUC | 기여 |
|------|------------|---------------------|------|
| B baseline | 없음 | 0.0564 | 기준 |
| **C** | GDELT title 21개 + coverage_mask | **0.0653** | **+0.0089** ✅ |

→ **GDELT title feature는 ACLED-free 환경에서 가장 큰 단일 기여 feature group**

---

## 5. 기존 GDELT Theme/Person Feature 정리

### 5-1. Parquet 현황

| 항목 | 값 |
|------|-----|
| 경로 | `input/processed/gdelt_titles/gdelt_theme_person_features.parquet` |
| Shape | (214,257, 24) |
| Coverage | 2015-02-17 ~ 2025-03-31 |
| 국가 수 | 58개국 |
| BQ 소스 | `conflict-early-warning.conflict_ew.gdelt_titles` |
| BQ 쿼리 비용 | ~1,282 GB (~$6.26) — 재실행 시 비용 발생 주의 |

### 5-2. Feature 목록 (22개 + date/country)

**v2themes 기반 (1d/7d)**:
```
gdelt_theme_nonnull_count_1d/7d   v2themes 비결측 기사 수
gdelt_theme_count_1d/7d           총 테마 토큰 수 (';' 구분)
gdelt_theme_conflict_count_1d/7d  CONFLICT/MILITARY_ATTACK/ARMED/WAR 테마
gdelt_theme_protest_count_1d/7d   PROTEST/RIOT/CIVIL_UNREST/STRIKE/DEMO 테마
gdelt_theme_military_count_1d/7d  MILITARY 테마
gdelt_theme_refugee_count_1d/7d   REFUGEE/DISPLACED/ASYLUM/HUMANITARIAN 테마
gdelt_theme_sanction_count_1d/7d  SANCTION/EMBARGO 테마
gdelt_theme_government_count_1d/7d GOV_/ELECTION/COUP/CEASEFIRE 테마
```

**v2persons 기반 (1d/7d)**:
```
gdelt_person_nonnull_count_1d/7d  v2persons 비결측 기사 수
gdelt_person_count_1d/7d          총 인물 토큰 수
gdelt_person_density_1d/7d        기사당 평균 인물 토큰 수
```

### 5-3. 기존 실험 기여도 분석

| 실험 | Feature 수 | Stacking Platt PR-AUC | vs C |
|------|-----------|---------------------|------|
| C | 57 | 0.0653 | 기준 |
| **D (C + theme/person)** | 79 | **0.0633** | **-0.0020** ❌ |

#### D가 C보다 낮게 나온 원인

1. **OOF 과적합 위험**: feature 57→79로 증가하며 OOF 분산 증가
2. **Meta learner 불균형**: meta C=10 선택 → LightGBM에 과도하게 치우침, XGB 기여도 낮아짐
3. **Val set 낙관적 추정**: 기존 B/C/D 실험은 cleanval 구조가 아니라 early stopping/C선택/calibration이 모두 val에 사용 → 실제 성능보다 낙관적
4. **Theme feature 노이즈**: regex 패턴 매칭 → 오탐 가능성 (MILITARY 테마 = 군사 보도 ≠ 분쟁)

#### F2_clean에서 theme/person이 크게 기여한 이유

- **cleanval split 적용**: val_eval이 early stopping/calibration에서 완전히 분리됨 → 더 신뢰도 높은 평가
- **F1→F2 delta**: +0.0191 (group ablation 재확인)
- **ACLED + GDELT 조합**: ACLED가 주는 구조적 신호 위에 GDELT theme/person이 보완적 역할
- **ACLED-free 환경에서는 주역**: ACLED 없으면 GDELT theme/person이 분쟁 신호의 주요 소스가 됨

---

## 6. 기존 BigQuery 쿼리에서 사용한 GDELT Column 정리

### 6-1. gdelt_titles (Custom BQ Table - GKG 기반)

**테이블**: `conflict-early-warning.conflict_ew.gdelt_titles`  
**파티션**: MONTH(date), **클러스터**: iso3

| Raw 컬럼 | 사용 여부 | 현재 feature |
|---------|---------|------------|
| `v2tone_avg` | ✅ 사용 | tone_mean/std/min/neg/pos count |
| `v2themes` | ✅ 사용 | conflict/protest/military/refugee/sanction/government count |
| `v2persons` | ✅ 사용 | person_count, person_density |
| `language` | ✅ 사용 | eng_count, lang_diversity |
| `domain` | ✅ 사용 | domain_diversity |
| `title` | ⚠️ 부분 | nonnull_count만 (텍스트 미활용, 2019-09 이후만 유효) |
| `url` | ❌ 미사용 | dedup 키로만 활용 |
| `date` | ✅ 파티션 키 | - |
| `iso3` | ✅ 클러스터 키 | country 매핑 |

### 6-2. gdelt-bq.gdeltv2.events_partitioned (GDELT Events)

**테이블**: `gdelt-bq.gdeltv2.events_partitioned`

| Raw 컬럼 | 사용 여부 | 현재 feature |
|---------|---------|------------|
| `GoldsteinScale` | ✅ | goldstein_mean/std (7d/14d/30d) |
| `AvgTone` | ✅ | tone_mean (7d/14d/30d) |
| `NumMentions` | ✅ | mentions_sum (7d/14d/30d) |
| `QuadClass` | ✅ | quadclass_1~4_ratio |
| `SQLDATE` | ✅ | event_count (7d/14d/30d) |
| `ActionGeo_CountryCode` | ✅ | country 필터 (FIPS → ISO3 변환) |
| `EventCode` | ⚠️ 부분 | root code만 (2자리), 세부코드 미활용 |
| `EventRootCode` | ✅ | 내부 집계 용도 |
| `NumArticles` | ❌ 미사용 | NumMentions와 별개 |
| `NumSources` | ❌ 미사용 | 소스 다양성 측정 가능 |
| `Actor1Code/Name` | ❌ 미수집 | 행위자 분석 불가 |
| `Actor2Code/Name` | ❌ 미수집 | - |
| `EventBaseCode` | ❌ 미사용 | - |

---

## 7. 추가로 활용 가능한 GDELT Column 후보

### 7-1. gdelt_titles (GKG) — 추가 수집 후보

| 컬럼 | 타입 | 활용 가치 | 추정 BQ 비용 | 추천 |
|------|------|---------|------------|------|
| `v2organizations` | STRING (;구분) | 행위자 다양성 (군/정부/NGO) | ~기존 theme/person 수준 | ⭐⭐⭐ |
| `V2Tone` full (6 field) | STRING | ARD(절대적 감성편차), polarity | 기존 쿼리 수정으로 추가 가능 | ⭐⭐⭐ |
| `v2enhancedthemes` | STRING (;구분) | GKG 2.0 상세 계층 (CRIME_*, ENV_*) | ~기존 theme/person 수준 | ⭐⭐ |
| `v2locations` | STRING | 분쟁 지역 지명 집중도 | 높음 | ⭐ |
| `SRCLC` | STRING | 원문 언어 (language와 중복 일부) | 낮음 | ⭐ |

**V2Tone full 6-field 분해**:
```
V2Tone = "tone,positivity,negativity,polarity,ARD,WDSD"
현재:  v2tone_avg = tone (첫 번째 필드만)
추가:  ARD (Absolute Reference Distance) → 감성 극화 정도
       polarity → 감성 양극화 측정
```

### 7-2. GDELT Events — 추가 활용 후보

| 컬럼/파생 | 활용 방식 | 현재 상태 |
|---------|---------|---------|
| `EventCode` 세부 (3~4자리) | Verbal vs Material conflict 비율 | root code만 사용 중 |
| `NumArticles` | mentions 대비 article 비율 → 증폭도 | 미사용 |
| `NumSources` | 소스 다양성 (domain diversity 보완) | 미수집 |
| `Actor1Type` / `Actor2Type` | 행위자 유형 (정부/군/반군) | 미수집 |
| QuadClass 비율 (3+4)/(1+2+3+4) | 갈등 비율 | 계산 가능 |
| Goldstein 14→7d slope | 추세 계산 | parquet에서 가능 |

---

## 8. ACLED-Free Feature Set 후보 (O0/O1/O2/O3)

### O0: GDELT Events + 경제지표 + Country (35개)

기존 B baseline과 동일. 이미 train.parquet에 포함.

```
GDELT Events (19개):
  gdelt_goldstein_mean/std × {7d, 14d, 30d}    6개
  gdelt_tone_mean × {7d, 14d, 30d}             3개
  gdelt_mentions_sum × {7d, 14d, 30d}          3개
  gdelt_event_count × {7d, 14d, 30d}           3개
  gdelt_quadclass_{1,2,3,4}_ratio              4개

경제지표 (15개):
  econ_{vix,wti,gold,dxy,stlfsi4} × {level, pct_1d, pct_7d}

Country (1개)
```

**기존 결과**: Stacking Platt PR-AUC = 0.0564 (구 val 기준, cleanval 미적용)

### O1: O0 + GDELT Title Features (57개 = O0 35 + title 21 + coverage_mask 1)

기존 C와 동일 feature set. gdelt_title_features.parquet 있음.

```
추가 feature:
  GDELT title 1d (10개): count, nonnull_count, tone_mean, tone_std, tone_min,
                          negative_count, positive_count, eng_count,
                          domain_diversity, lang_diversity
  GDELT title 7d (11개): 위 rolling version + tone_trend_7d
  coverage_mask (1개):   date < 2015-02-17이면 1
```

**기존 결과**: Stacking Platt PR-AUC = 0.0653 (구 val 기준)  
**cleanval 재실험 필요**: 기존 B/C/D는 F6 fold (2023 val) + 기존 val 셋 사용 → cleanval과 다름

### O2: O1 + GDELT Theme/Person Features (79개 = O1 57 + theme/person 22)

기존 D와 동일 feature set. gdelt_theme_person_features.parquet 있음.

```
추가 feature:
  theme 1d (7개): conflict, protest, military, refugee, sanction, government count
                  + nonnull_count, theme_count, person_nonnull_count
  theme 7d (7개): 위 rolling version
  person 1d (2개): person_count, person_density
  person 7d (2개): rolling version
  (합계 22개)
```

**기존 결과**: Stacking Platt PR-AUC = 0.0633 (구 val 기준, C보다 낮음)  
**재실험 이유**: cleanval split + 개선된 meta learner 설정으로 재평가 필요

### O3: O2 + 추가 BigQuery Column 기반 확장 (예상 ~90개+)

BigQuery 재수집 필요. 현재는 계획 단계.

```
추가 후보:
  v2organizations 기반: org_count_1d/7d, org_diversity_1d/7d         4개
  V2Tone full: ARD_mean_1d/7d, polarity_mean_1d/7d                   4개
  파생: conflict_theme_ratio_1d/7d (conflict_count/total_count)       2개
        neg_pos_ratio_1d/7d (negative/positive count)                 2개
        quadclass_conflict_ratio (QC3+QC4 / total)                   1개
        goldstein_7d_slope (trend)                                    1개
  합계: ~14개 추가
```

---

## 9. LGBM / XGBoost / LSTM 모델링 계획

### 9-1. 각 모델의 강점

| 모델 | 적합한 Feature | 장점 | 단점 |
|------|-------------|------|------|
| **LightGBM** | tabular aggregation, categorical(country) | 빠른 학습, categorical 직접 처리 | 시계열 패턴 미반영 |
| **XGBoost** | tabular aggregation, numeric | 안정적 성능 | categorical 처리 불리 |
| **LSTM** | country별 daily sequence | 시계열 패턴 포착, lag 효과 | 학습 비용, 데이터 요구량 |
| **Stacking** | LGBM/XGB/LSTM score | 다양성 결합 | meta 과적합 위험 |

### 9-2. Feature별 모델 배분 전략

```
LGBM/XGB (tabular):
  → O0~O3 전체 feature matrix (date×country 단위 집계값)
  → country = categorical feature (LGBM: category dtype, XGB: label encoding)
  → 경제지표 = global signal (country 불문 동일값)

LSTM (sequence):
  → country별 daily time-series 구성
  → lookback window: 30일 또는 60일
  → feature: GDELT daily aggregations (count, tone, theme count 등)
  → 경제지표 sequence도 포함
  → 최종 hidden state → classification head → P(y_escalation=1)

Stacking meta:
  → LGBM score + XGB score + LSTM score → LogisticRegression meta
  → meta 학습: OOF predictions (train_fit 내부)
  → meta C 선택: tune_cal 기준
  → calibration: tune_cal fit, val_eval 평가
```

---

## 10. LSTM용 Sequence Dataset 설계

### 10-1. Dataset 형태

```python
# Country별 date 정렬
# Shape: (n_samples, lookback, n_features)

X[i, :, :] = feature matrix for country c, days [t-lookback+1 ... t]
y[i]        = y_escalation at time t

# n_samples = Σ_country max(0, n_days - lookback)
# n_features = GDELT daily features + economic indicators
```

### 10-2. Lookback Window 후보

| Lookback | 장점 | 단점 | 권장 |
|---------|------|------|------|
| 14일 | 빠른 학습, 최근 패턴 집중 | 장기 추세 미반영 | 2순위 |
| 30일 | 월간 패턴 포착, 균형 | - | **1순위** |
| 60일 | 장기 trend 포착 | 학습 데이터 감소 | 3순위 |

### 10-3. Feature Sequence 구성

```
daily features per country per day (약 20~30개):
  gdelt_event_count_1d       (daily GDELT event count)
  gdelt_goldstein_mean_1d    (당일 Goldstein 평균)
  gdelt_tone_mean_1d         (당일 tone 평균)
  gdelt_title_count_1d       (당일 기사 수)
  gdelt_title_tone_mean_1d   (당일 title tone)
  gdelt_title_negative_count_1d
  gdelt_theme_conflict_count_1d
  gdelt_theme_protest_count_1d
  gdelt_theme_military_count_1d
  gdelt_person_density_1d
  coverage_mask              (GDELT 커버리지 마스크)
  econ_vix / econ_wti / econ_gold / econ_dxy  (global, country 무관)
  ...
```

### 10-4. GDELT Coverage 시작일 처리

```
GDELT GKG coverage: 2015-02-17부터
lookback=30일 시:
  train_fit 학습 시작 가능 날짜: 2015-03-18 (2015-02-17 + 30일)
  2014-01-01 ~ 2015-03-17: LSTM 학습에서 제외 (coverage 부족)
  coverage_mask feature로 보완 또는 해당 구간 완전 제외

권장:
  LGBM/XGB: coverage_mask=1 행 유지 (mask feature로 정보 전달)
  LSTM: coverage_mask=1 구간 완전 제외 (시퀀스 오염 방지)
```

### 10-5. Train/Val/Test Split (cleanval 유지)

```
train_fit  : 2015-02-17 ~ 2022-12-31  (LSTM: 2015-03-18 ~ 2022-12-31)
tune_cal   : 2023-01-01 ~ 2023-12-31
val_eval   : 2024-01-01 ~ 2024-06-30
test       : 2024-07-01 ~             (평가 금지)
```

---

## 11. Clean Validation 실험 계획

### 주요 변경점: 기존 B/C/D vs 신규 O0~O2

| 항목 | 기존 B/C/D | 신규 O0~O2 |
|------|-----------|-----------|
| Val 구조 | 기존 val 셋 (조기종료/C/calibration 모두 사용) | cleanval (tune_cal로 분리) |
| OOF 최종 fold | F6 (train≤2022 → pred 2023) | F5 (train≤2021 → pred 2022) |
| Val 기간 | 기존 dataset val (2014~) | val_eval 2024-H1 |
| 평가 신뢰도 | 낙관적 (leak 일부) | 보수적 (leak 없음) |

### 실험 순서 (단계별)

```
1단계 — Tabular (빠른 검증):
  O0_clean: GDELT events + 경제지표 + country (35개)
  O1_clean: O0 + GDELT title (57개)
  O2_clean: O1 + GDELT theme/person (79개)

2단계 — O3 (BigQuery 재수집 필요, 팀 합의 후):
  O3_clean: O2 + v2organizations/V2Tone full/derived (90개+)

3단계 — LSTM (선택):
  O2_LSTM: O2 feature를 sequence로 재구성 + LSTM 단독
  O2_stacking: O2 LGBM + O2 XGB + O2 LSTM → stacking

4단계 — Stacking:
  best_O_stacking: 가장 좋은 O feature set으로 LGBM+XGB+LSTM stacking
```

### 채택 기준 (제안)

```
ACLED-free 모델의 새 기준:
  PR-AUC >= O1_clean + 0.003  (각 단계 누적)
  또는 Lift@top5% >= 현재 최고 × 1.05 이상

ACLED 포함 F2_clean(0.1027)과의 직접 비교는 권장하지 않음
  → ACLED feature 제거 자체가 성능 하락이므로 별도 기준 필요
```

---

## 12. 누수 방지 체크리스트

### 12-1. 절대 금지 Feature

```
❌ ACLED 관련 (운영 불가):
   acled_*, safe_acled_*, enhanced_safe_acled_*

❌ Label/Future 관련 (직접 누수):
   y, y_onset, y_escalation
   fatalities_next3d, event_count_next3d
   past14d_event_count, past14d_fatalities_mean

❌ 기타 금지:
   macis_se_score
   future_*, next_*
```

### 12-2. GDELT Feature의 Publication Lag 검토

| 데이터 소스 | Lag | Feature window 조정 필요? |
|-----------|-----|------------------------|
| GDELT Events (gdelt-bq) | 사실상 없음 (자동수집) | ✅ 당일 feature 사용 가능 |
| GDELT GKG (gdelt_titles) | 사실상 없음 (15분 업데이트) | ✅ 당일 feature 사용 가능 |
| 경제지표 (yfinance) | 당일 장 마감 후 | ✅ 당일 feature 사용 가능 |

**중요**: ACLED의 7일 publication lag와 달리, GDELT는 거의 실시간이므로 shift(7) 등의 lag 처리가 필요 없다. 단, t일 예측 시 t일의 GDELT feature를 쓰는 것이 운영에서 실제로 가능한지 확인 필요.

```
운영 시나리오:
  - 매일 오전 6시 실행
  - t = 어제 날짜
  - feature: t일(어제)까지의 GDELT/경제지표
  - target: 오늘(t+1)~모레(t+3)의 escalation 가능성
  → GDELT t일 feature 사용 가능 ✅
```

### 12-3. GDELT Feature-Label 겹침 검토

```
주의사항:
  y_escalation(t) = ACLED 기반 t+1~t+3 spike
  gdelt_theme_conflict_count_1d(t) = t일의 GDELT 보도

  → feature(t)가 label(t+1~t+3)을 예측: 정상 (미래 정보 아님)
  → 단, label 계산에 GDELT를 쓴다면 순환 문제 발생
     y_escalation은 ACLED 기반이므로 이 문제 없음 ✅
```

### 12-4. Test Set 정책

```
- test set(2024-07-01~)은 최종 후보 1개 확정 후 1회만 평가
- O0/O1/O2 비교는 val_eval(2024-H1)로만 수행
- test 결과를 보고 모델 선택 금지
```

---

## 13. 다음 단계 (우선순위 순)

```
[1] O0_clean 스크립트 작성 및 실행
    - 기존 run_stacking_acled_free_with_titles.py를 cleanval split으로 수정
    - ACLED feature 완전 제거 확인
    - PR-AUC baseline 재측정

[2] O1_clean 실행
    - O0 + gdelt_title_features.parquet merge
    - coverage_mask 적용
    - C 기존 결과(0.0653)와 cleanval 결과 비교

[3] O2_clean 실행
    - O1 + gdelt_theme_person_features.parquet merge
    - 기존 D 결과(0.0633)와 cleanval 결과 비교
    - 개선된 meta learner 설정 (class_weight balanced, C 재탐색)

[4] LSTM 검토 (O2 이후)
    - sequence dataset 구성 코드 작성
    - lookback=30 우선 실험
    - LGBM/XGB/LSTM 3-way stacking

[5] O3 (BigQuery 재수집 - 팀 합의 후)
    - v2organizations, V2Tone ARD/polarity 추가
    - 비용 ~$6 승인 필요
```

---

## 14. 현재 실험 진행 현황 정리

| 실험 | 구조 | Val 방식 | PR-AUC | 상태 |
|------|------|---------|--------|------|
| B | GDELT events + econ (35개) | 구 val | 0.0564 | ✅ 완료 |
| C | B + title (57개) | 구 val | 0.0653 | ✅ 완료 |
| C2 | C + derived title | 구 val | 0.0643 | ❌ 미채택 |
| D | C + theme/person (79개) | 구 val | 0.0633 | ❌ 미채택 |
| F0_clean | B + safe_ACLED (50개) | cleanval | 0.0781 | ✅ (ACLED 포함) |
| F1_clean | F0 + title (72개) | cleanval | 0.0836 | ✅ (ACLED 포함) |
| F2_clean | F1 + theme/person (94개) | cleanval | 0.1027 | ✅ **ACLED 포함 상한** |
| **O0_clean** | B 35개 | **cleanval** | **미실행** | ⬜ 다음 |
| **O1_clean** | O0 + title | **cleanval** | **미실행** | ⬜ 다음 |
| **O2_clean** | O1 + theme/person | **cleanval** | **미실행** | ⬜ 다음 |

**핵심**: B/C/D는 구 val 방식(낙관적)이므로 cleanval로 재측정 필요. F2_clean(0.1027)은 ACLED 포함 upper-bound로 참고값이고, **ACLED-free 운영 모델의 실질적 baseline은 O0_clean부터 새로 측정해야 한다.**

---

*이 문서는 조사/설계 단계 결과물이다. BigQuery 쿼리 실행/모델 학습/test 평가는 수행하지 않았다.*  
*생성: 2026-06-04*
