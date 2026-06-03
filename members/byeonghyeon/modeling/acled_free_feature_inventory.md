# ACLED-free Feature Inventory
**작성자**: byeonghyeon  
**작성일**: 2026-06-03  
**목적**: 실험 B/C/D를 위한 실제 학습 데이터 컬럼 전체 분류 및 ACLED-free feature_cols 확정

---

## 1. 실제 학습 데이터 파일 탐색 결과

### 1-1. 파일 위치

팀 레포(`KUBIG_Conference-team-4/`)에는 학습용 parquet 파일이 없다.  
실제 전처리 완료된 데이터는 **개인 레포**에 있다.

```
개인 레포: conflict-early-warning/
  input/processed/dataset/
    train.parquet   ← 211,816행 × 64컬럼
    val.parquet     ← 10,556행  × 64컬럼
    test.parquet    ← 15,718행  × 64컬럼
    full.parquet    ← 전체 합본
    full_se.parquet ← SE 포함 버전 (별도)
  input/processed/features/
    features.parquet    ← 피처만 (라벨 없음)
    se_scores.parquet   ← SE 피처만 (별도 파일로 관리)
  output/macis_12y/
    se_scores.parquet   ← stacking script가 직접 읽는 SE 파일
```

### 1-2. train/val/test 파일 기본 정보

| 파일 | 행수 | 컬럼수 | 기간 | y_escalation 양성률 |
|------|------|--------|------|-------------------|
| `train.parquet` | 211,816 | 64 | 2014-01-01 ~ 2023-12-31 | 4.29% |
| `val.parquet` | 10,556 | 64 | 2024-01-01 ~ 2024-06-30 | 4.07% |
| `test.parquet` | 15,718 | 64 | 2024-07-01 ~ 2025-03-28 | 4.06% |

- `date` 컬럼 타입: `datetime64[ns, UTC]`
- `country` 컬럼 타입: `object` (ISO3)

### 1-3. macis_se_score 위치

`train.parquet`에 포함되지 않는다.  
stacking script 실행 시 `output/macis_12y/se_scores.parquet`에서 left-merge된다.

```
se_scores.parquet
  shape: 206,929행 × 3컬럼 (iso3, date, se_score)
  date range: 2014-01-05 ~ 2025-03-31
  null 수: 1,682건 (0.81%)
```

---

## 2. train.parquet 전체 64컬럼 inventory

### 2-1. KEY 컬럼 (피처 아님, 항상 보존)

```
date       datetime64[ns, UTC]  학습 기준일 (UTC)
country    object               국가 ISO3
```

### 2-2. remove_label_or_future (7개) — 항상 제거, 실험 설정과 무관

```
y_escalation          int64    ← 타깃. 절대 피처에 포함 불가
y                     int64    ← label_meta
y_onset               int64    ← label_meta
fatalities_next3d     int64    ← 미래 3일 사상자 (직접 leakage)
event_count_next3d    int64    ← 미래 3일 이벤트 수 (직접 leakage)
past14d_event_count   int64    ← ACLED 과거 14일 이벤트 (label_meta, ACLED 파생)
past14d_fatalities_mean float64 ← ACLED 과거 14일 사상자 평균 (label_meta, ACLED 파생)
```

### 2-3. remove_acled_raw (20개) — ACLED-free에서 제거

```
acled_event_count_7d      float64
acled_event_count_14d     float64
acled_event_count_30d     float64
acled_fatalities_7d       float64
acled_fatalities_14d      float64
acled_fatalities_30d      float64
acled_fatalities_max_7d   float64
acled_fatalities_max_14d  float64
acled_fatalities_max_30d  float64
acled_ratio_battles       float64
acled_ratio_explosions    float64
acled_ratio_vac           float64
acled_actor_type_1_ratio  float64
acled_actor_type_2_ratio  float64
acled_actor_type_3_ratio  float64
acled_actor_type_4_ratio  float64
acled_actor_type_5_ratio  float64
acled_actor_type_6_ratio  float64
acled_actor_type_7_ratio  float64
acled_actor_type_8_ratio  float64
```

### 2-4. remove_acled_derived (1개) — ACLED-free에서 제거

```
acled_missing_mask    float64   ← ACLED coverage 기반 파생
```

### 2-5. remove_se_score (1개) — ACLED-free에서 제거 (parquet 외부)

```
macis_se_score    float64   ← train.parquet에 없음. stacking script에서 merge됨.
                              leakage 미확인 → ACLED-free 실험에서 merge 자체를 건너뜀.
```

### 2-6. gdelt_events_feature (19개) — ACLED-free에서 유지

```
gdelt_goldstein_mean_7d   float64
gdelt_goldstein_mean_14d  float64
gdelt_goldstein_mean_30d  float64
gdelt_goldstein_std_7d    float64
gdelt_goldstein_std_14d   float64
gdelt_goldstein_std_30d   float64
gdelt_tone_mean_7d        float64
gdelt_tone_mean_14d       float64
gdelt_tone_mean_30d       float64
gdelt_mentions_sum_7d     float64
gdelt_mentions_sum_14d    float64
gdelt_mentions_sum_30d    float64
gdelt_event_count_7d      float64
gdelt_event_count_14d     float64
gdelt_event_count_30d     float64
gdelt_quadclass_1_ratio   float64
gdelt_quadclass_2_ratio   float64
gdelt_quadclass_3_ratio   float64
gdelt_quadclass_4_ratio   float64
```

> 모두 `shift(1)` 적용 (전일까지 데이터) — `feature_builder.py` 확인 완료 ✅

### 2-7. economic_feature (15개) — ACLED-free에서 유지

```
econ_vix         float64
econ_vix_pct_1d  float64
econ_vix_pct_7d  float64
econ_wti         float64
econ_wti_pct_1d  float64
econ_wti_pct_7d  float64
econ_gold        float64
econ_gold_pct_1d float64
econ_gold_pct_7d float64
econ_dxy         float64
econ_dxy_pct_1d  float64
econ_dxy_pct_7d  float64
econ_stlfsi4     float64
econ_stlfsi4_pct_1d float64
econ_stlfsi4_pct_7d float64
```

> `shift(1)` 적용 (당일 종가 미확정 방지) — `feature_builder.py` 확인 완료 ✅

### 2-8. country_or_time_feature (1개) — ACLED-free에서 유지

```
country    object    ← ISO3. LightGBM: categorical, XGBoost: label-encode
```

> 시간 피처(day_of_week, month 등): `feature_builder.py`에 미구현 → 현재 parquet에 없음  
> Telegram 피처: 미통합 → 없음

### 2-9. gdelt_title_feature (0개, 신규 추가 예정)

```
(실험 C에서 추가)
gdelt_title_count_1d
gdelt_title_tone_mean_1d
gdelt_title_count_7d
gdelt_title_tone_mean_7d
gdelt_title_count_14d
gdelt_title_tone_mean_14d
... (Section 5 참조)
```

---

## 3. 피처 수 요약

| 실험 | 피처 소스 | 피처 수 |
|------|----------|---------|
| Reference (기존 full model) | 20 ACLED + 1 mask + 19 GDELT + 15 econ + 1 country = 56 (parquet) + 1 macis_se_score (외부) | **57** |
| **B (ACLED-free baseline)** | 19 GDELT + 15 econ + 1 country (parquet only, no SE merge) | **35** |
| **C (B + GDELT titles)** | B + titles 집계 피처 ~12개 | **~47** |
| **D (C + themes/persons)** | C + GKG 키워드 피처 ~6개 | **~53** |

---

## 4. 실험 B feature_cols 정의 (확정)

```python
# 실험 B에서 stacking script의 feature selection 로직
LABEL_META_COLS = [
    "y", "y_onset", "y_escalation",
    "fatalities_next3d", "event_count_next3d",
    "past14d_event_count", "past14d_fatalities_mean",
]

ACLED_REMOVE_COLS = [
    "acled_event_count_7d",    "acled_event_count_14d",  "acled_event_count_30d",
    "acled_fatalities_7d",     "acled_fatalities_14d",   "acled_fatalities_30d",
    "acled_fatalities_max_7d", "acled_fatalities_max_14d","acled_fatalities_max_30d",
    "acled_ratio_battles",     "acled_ratio_explosions",  "acled_ratio_vac",
    *[f"acled_actor_type_{i}_ratio" for i in range(1, 9)],
    "acled_missing_mask",
]

ALWAYS_EXCLUDE = ["date"] + ACLED_REMOVE_COLS

# feature_cols = 모든 컬럼 - LABEL_META - ALWAYS_EXCLUDE
# 결과: 35개 컬럼 (GDELT 19 + econ 15 + country 1)

# SE merge 건너뜀 (macis_se_score 미사용)
# train, val, test = pd.read_parquet(...)  # SE merge 없이 직접 로드
```

**실험 B feature_cols 확정 목록 (35개)**:
```python
FEATURE_COLS_B = [
    # GDELT events (19개)
    'gdelt_goldstein_mean_7d',  'gdelt_goldstein_mean_14d', 'gdelt_goldstein_mean_30d',
    'gdelt_goldstein_std_7d',   'gdelt_goldstein_std_14d',  'gdelt_goldstein_std_30d',
    'gdelt_tone_mean_7d',       'gdelt_tone_mean_14d',      'gdelt_tone_mean_30d',
    'gdelt_mentions_sum_7d',    'gdelt_mentions_sum_14d',   'gdelt_mentions_sum_30d',
    'gdelt_event_count_7d',     'gdelt_event_count_14d',    'gdelt_event_count_30d',
    'gdelt_quadclass_1_ratio',  'gdelt_quadclass_2_ratio',
    'gdelt_quadclass_3_ratio',  'gdelt_quadclass_4_ratio',
    # economic (15개)
    'econ_vix',     'econ_vix_pct_1d',     'econ_vix_pct_7d',
    'econ_wti',     'econ_wti_pct_1d',     'econ_wti_pct_7d',
    'econ_gold',    'econ_gold_pct_1d',    'econ_gold_pct_7d',
    'econ_dxy',     'econ_dxy_pct_1d',     'econ_dxy_pct_7d',
    'econ_stlfsi4', 'econ_stlfsi4_pct_1d', 'econ_stlfsi4_pct_7d',
    # country (1개)
    'country',
]
```

---

## 5. 실험 C/D에서 추가될 GDELT title/GKG 피처

### 5-1. 실험 C 추가 피처 (gdelt_title_feature, ~12개)

BQ `conflict-early-warning.conflict_ew.gdelt_titles` 집계 기반:

```python
GDELT_TITLE_COLS_C = [
    # 1일 집계
    'gdelt_title_count_1d',
    'gdelt_title_tone_mean_1d',
    'gdelt_title_tone_std_1d',
    'gdelt_title_tone_min_1d',
    'gdelt_title_negative_count_1d',
    'gdelt_title_eng_count_1d',
    'gdelt_title_domain_diversity_1d',
    # 7일 rolling
    'gdelt_title_count_7d',
    'gdelt_title_tone_mean_7d',
    # 14일 rolling
    'gdelt_title_count_14d',
    'gdelt_title_tone_mean_14d',
    # 추세
    'gdelt_title_tone_trend_7d',  # tone_1d - tone_8d_lag
]
```

**Coverage gap 처리**: 2014-01-01 ~ 2015-02-16 기간은 BQ 데이터 없음 → 0 채움

### 5-2. 실험 D 추가 피처 (gdelt_gkg_feature, ~6개, BQ v2themes/v2persons 기반)

```python
GDELT_GKG_COLS_D = [
    # v2themes 기반 키워드 카운트
    'gdelt_theme_conflict_count_1d',   # CONFLICT, MILITARY_ATTACK 포함
    'gdelt_theme_protest_count_1d',    # PROTEST, RIOT 포함
    'gdelt_theme_refugee_count_1d',    # REFUGEES, DISPLACED 포함
    'gdelt_theme_sanction_count_1d',   # ECON_SANCTIONS, UN_SANCTIONS 포함
    # v2persons 기반
    'gdelt_persons_named_count_1d',    # persons 비결측 기사 수
    # 7일 rolling
    'gdelt_theme_conflict_count_7d',
]
```

**v2themes 파싱 방식**: `SPLIT(v2themes, ';')` → 각 원소에서 `,` 앞 부분만 추출 → 키워드 매칭

---

## 6. BLOCK-2 해결 결론

| 항목 | 결과 |
|------|------|
| train.parquet 위치 | 개인 레포(`conflict-early-warning/input/processed/dataset/`) |
| 팀 레포 parquet 존재 여부 | ❌ 없음 (전처리 파이프라인 재실행 필요 또는 개인 레포 경로 사용) |
| 실제 컬럼 수 | 64개 (key 2 + label 7 + ACLED 21 + GDELT 19 + econ 15 + country 1) |
| ACLED-free B feature_cols 확정 | ✅ 35개 (GDELT 19 + econ 15 + country 1) |
| macis_se_score 위치 | train.parquet 외부 (별도 merge) → B/C/D에서 merge 자체 건너뜀 |
| Telegram 피처 | ❌ 현재 미통합 |
| 시간 피처 | ❌ 현재 미구현 |
| 코드 카테고리 분류 불일치 | 없음. 모든 컬럼이 예상 범주와 일치 ✅ |

**BLOCK-2 완료** — 실험 B 스크립트 작성 가능 상태

---

## 7. 실험 B 구현 위치 요약

변경 대상: 기존 `model/run_stacking_d_with_mask_feature_ablation.py` 복사 후 수정

| 수정 항목 | 내용 |
|----------|------|
| SE 로드/merge 제거 | `se_df = pd.read_parquet(SE_PATH)` 및 3개 `merge_se()` 호출 삭제 |
| ALWAYS_EXCLUDE 수정 | ACLED 컬럼 21개 추가 (Section 4 `ACLED_REMOVE_COLS`) |
| EXPERIMENT 이름 변경 | `"stacking_acled_free_baseline"` |
| 출력 파일 경로 | 자동으로 EXPERIMENT 이름 기반 생성됨 |
| 실행 경로 | 개인 레포 루트에서 실행 (데이터가 거기 있음) |

---

## 8. 미해결 리스크

| 항목 | 리스크 | 대응 |
|------|--------|------|
| 팀 레포에 parquet 없음 | 실험을 개인 레포 기준으로 실행해야 함 | 팀 레포 README에 데이터 위치 명시 권장 |
| 2014 titles 데이터 없음 | GDELT titles 피처 초기 구간 0 채움 | coverage_mask 피처 추가 검토 |
| gdelt_tone_mean (기존) vs gdelt_title_tone_mean_* (신규) 중복 | 상관 높을 경우 모델 해석 복잡 | 상관 분석 후 판단 |
| se_scores.parquet leakage | 학습 범위 미확인 | B/C/D 모두 SE 제거로 우회 ✅ |
