# ACLED-free 실험 준비 상태 점검
**작성자**: byeonghyeon  
**작성일**: 2026-06-03  
**목적**: 실험 B/C/D 시작 전 실행 가능 여부 전체 점검

---

## 1. 실험 구조 요약

| 실험 | 피처 구성 | 상태 |
|------|----------|------|
| **B** (ACLED-free baseline) | GDELT events + 경제 + country | 🟡 구현 필요 |
| **C** (B + GDELT titles) | B + titles 집계 피처 | 🟡 데이터 확인 필요 |
| **D** (C + themes/persons) | C + GKG 키워드 피처 | 🔴 BQ 확인 후 결정 |
| **Reference** (기존 full model) | ACLED + SE + full 57개 | ✅ 기존 결과, 재실험 불필요 |

**핵심 비교**: B → C (GDELT titles 피처의 독립 기여)

---

## 2. BigQuery gdelt_titles 스키마 확인

### 2-1. 현재 로컬 백업 스키마 확인 결과

```
파일: input/raw/gdelt_titles/AFG/2022-01.parquet
shape: (80,963행, 9컬럼)

컬럼              타입         설명
-----------      ---------    --------------------------------
date             object/date  보도 날짜
iso3             string       국가 ISO3
title            string       기사 제목 (nullable)
url              string       기사 URL
domain           string       출처 도메인
language         string       언어 코드 (eng, ara, rus, ...)
sourcecountry    string       출처 국가 (nullable)
seendate         string/ts    GDELT 수집 시각
v2tone_avg       float64      기사 톤 점수 (mean ≈ -2.8, range -28 ~ +16)
```

### 2-2. themes / persons 컬럼 존재 여부

```
로컬 백업 기준: ❌ themes 없음, ❌ persons 없음
→ 실험 D는 현재 로컬 데이터로 불가
→ BigQuery 테이블에 추가 컬럼이 있을 수 있음 — BQ 직접 확인 필요
```

### 2-3. BigQuery 확인 SQL (실행 필요)

```sql
-- [REQUIRED-1] 기간 및 규모 확인
SELECT
  MIN(date)             AS min_date,
  MAX(date)             AS max_date,
  COUNT(*)              AS total_rows,
  COUNT(DISTINCT iso3)  AS n_countries
FROM `conflict-early-warning.conflict_ew.gdelt_titles`;

-- [REQUIRED-2] 컬럼 목록 및 themes/persons 존재 여부
SELECT column_name, data_type
FROM `conflict-early-warning.conflict_ew.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'gdelt_titles';

-- [REQUIRED-3] 연도별 row 수 (2014 이전 커버 여부 확인)
SELECT
  EXTRACT(YEAR FROM date) AS year,
  COUNT(*) AS rows,
  COUNT(DISTINCT iso3) AS n_countries
FROM `conflict-early-warning.conflict_ew.gdelt_titles`
GROUP BY year ORDER BY year;

-- [OPTIONAL-4] 국가별 coverage 요약
SELECT iso3, MIN(date) AS first, MAX(date) AS last, COUNT(*) AS rows
FROM `conflict-early-warning.conflict_ew.gdelt_titles`
GROUP BY iso3 ORDER BY iso3;
```

### 2-4. BQ 확인 결과에 따른 분기

| BQ 결과 | 실험 진행 방향 |
|---------|--------------|
| 2014 이전 데이터 있음 | 실험 C 전 기간(2014~2025) 진행 가능 |
| 2022 이후만 있음 | 실험 C를 val/test 기간만으로 제한하거나 실험 축소 |
| themes 컬럼 있음 | 실험 D 진행 가능 |
| themes 컬럼 없음 | 실험 D 후속 GKG 수집 과제로 분리 |

---

## 3. 기존 학습 데이터 컬럼 inventory

### 3-1. 확인 방법

```python
# 팀 레포 루트에서 실행
import pandas as pd
train = pd.read_parquet("input/processed/dataset/train.parquet")
print("shape:", train.shape)
print("columns:", sorted(train.columns.tolist()))
```

> ⚠️ `input/processed/dataset/train.parquet`이 로컬에 없을 경우  
> `src/process/build_dataset.py`를 실행해 생성해야 함 (전처리 파이프라인 전체 필요).

### 3-2. 코드에서 추론한 예상 컬럼 목록

`src/process/feature_builder.py` + `src/process/label_builder.py` 분석 결과:

#### 키 컬럼
```
date, country
```

#### 타깃 / label_meta (학습에 사용 불가)
```
y_escalation         ← 타깃
y                    ← label_meta
y_onset              ← label_meta
fatalities_next3d    ← 미래 leakage
event_count_next3d   ← 미래 leakage
past14d_event_count  ← ACLED 파생 (label_meta)
past14d_fatalities_mean  ← ACLED 파생 (label_meta)
```

#### ACLED raw feature (20개) — ACLED-free에서 제거
```
acled_event_count_7d      acled_event_count_14d      acled_event_count_30d
acled_fatalities_7d       acled_fatalities_14d       acled_fatalities_30d
acled_fatalities_max_7d   acled_fatalities_max_14d   acled_fatalities_max_30d
acled_ratio_battles       acled_ratio_explosions     acled_ratio_vac
acled_actor_type_1_ratio  acled_actor_type_2_ratio   acled_actor_type_3_ratio
acled_actor_type_4_ratio  acled_actor_type_5_ratio   acled_actor_type_6_ratio
acled_actor_type_7_ratio  acled_actor_type_8_ratio
```

#### ACLED derived feature (1개) — ACLED-free에서 제거
```
acled_missing_mask
```

#### SE score (stacking script에서 별도 merge) — ACLED-free에서 제거
```
macis_se_score   (parquet에는 없고, stacking script 실행 중 merge됨)
```

#### GDELT events feature (~19개) — 유지
```
gdelt_goldstein_mean_7d   gdelt_goldstein_mean_14d   gdelt_goldstein_mean_30d
gdelt_goldstein_std_7d    gdelt_goldstein_std_14d    gdelt_goldstein_std_30d
gdelt_tone_mean_7d        gdelt_tone_mean_14d        gdelt_tone_mean_30d
gdelt_mentions_sum_7d     gdelt_mentions_sum_14d     gdelt_mentions_sum_30d
gdelt_event_count_7d      gdelt_event_count_14d      gdelt_event_count_30d
gdelt_quadclass_1_ratio   gdelt_quadclass_2_ratio
gdelt_quadclass_3_ratio   gdelt_quadclass_4_ratio
```

> `feature_builder.py`에서 shift(1) 적용 확인 ✅ (전일까지 데이터)

#### economic feature (15개) — 유지
```
econ_vix     econ_vix_pct_1d     econ_vix_pct_7d
econ_wti     econ_wti_pct_1d     econ_wti_pct_7d
econ_gold    econ_gold_pct_1d    econ_gold_pct_7d
econ_dxy     econ_dxy_pct_1d     econ_dxy_pct_7d
econ_stlfsi4 econ_stlfsi4_pct_1d econ_stlfsi4_pct_7d
```

> `feature_builder.py`에서 shift(1) 적용 확인 ✅

#### Telegram feature — 현재 미통합
```
없음 (Telegram/ 폴더 존재하나 feature_builder.py에 미포함)
```

---

## 4. ACLED-free 피처 분류 요약

| 카테고리 | 컬럼 | 실험 B | 실험 C | 실험 D |
|----------|------|--------|--------|--------|
| `remove_acled_raw` | acled_event_count_*, acled_fatalities_*, acled_ratio_*, acled_actor_type_* | ❌ | ❌ | ❌ |
| `remove_acled_derived` | acled_missing_mask | ❌ | ❌ | ❌ |
| `remove_se_score` | macis_se_score | ❌ | ❌ | ❌ |
| `remove_label_or_future` | y, y_onset, y_escalation, fatalities_next3d, event_count_next3d, past14d_* | ❌ | ❌ | ❌ |
| `gdelt_events_feature` | gdelt_* | ✅ | ✅ | ✅ |
| `economic_feature` | econ_* | ✅ | ✅ | ✅ |
| `country_or_time_feature` | country | ✅ | ✅ | ✅ |
| `gdelt_title_feature` (신규) | gdelt_title_* | ❌ | ✅ | ✅ |
| `gdelt_gkg_feature` (조건부) | gdelt_theme_*, gdelt_persons_* | ❌ | ❌ | ✅ |

---

## 5. ACLED-free baseline 구현에 필요한 코드 수정 위치

### 5-1. dataset 재생성 여부

**불필요** — 기존 `train/val/test.parquet` 그대로 사용.  
ACLED 컬럼이 parquet에 있어도, stacking script에서 feature selection 시 제외 가능.

### 5-2. 수정 방식: 신규 스크립트 작성 (기존 수정 최소화)

기존 `model/run_stacking_d_with_mask_feature_ablation.py`를 복사해  
`run_stacking_acled_free_baseline.py`로 만들고, 아래 두 가지만 변경:

**변경 1: SE merge 제거**
```python
# 제거 (SE 로드 및 merge 전체 삭제)
# se_df = pd.read_parquet(SE_PATH)
# train = merge_se(train, se_df, "train")
# val   = merge_se(val,   se_df, "val")
# test  = merge_se(test,  se_df, "test")

# 대신: 직접 로드
train = pd.read_parquet(TRAIN_PATH)
val   = pd.read_parquet(VAL_PATH)
test  = pd.read_parquet(TEST_PATH)
```

**변경 2: ALWAYS_EXCLUDE에 ACLED 컬럼 추가**
```python
ACLED_REMOVE_COLS = [
    "acled_event_count_7d",   "acled_event_count_14d",  "acled_event_count_30d",
    "acled_fatalities_7d",    "acled_fatalities_14d",   "acled_fatalities_30d",
    "acled_fatalities_max_7d","acled_fatalities_max_14d","acled_fatalities_max_30d",
    "acled_ratio_battles",    "acled_ratio_explosions",  "acled_ratio_vac",
    *[f"acled_actor_type_{i}_ratio" for i in range(1, 9)],
    "acled_missing_mask",
]
ALWAYS_EXCLUDE = [DATE_COL] + ACLED_REMOVE_COLS
```

**변경 3: EXPERIMENT 이름 변경**
```python
EXPERIMENT = "stacking_acled_free_baseline"
```

### 5-3. 실험 C 추가 작업

1. `src/process/gdelt_titles_feature_builder.py` 신규 작성
2. BQ 또는 로컬 parquet에서 집계 → `gdelt_titles_features.parquet`
3. `run_stacking_acled_free_with_titles.py`에서 merge 후 사용

---

## 6. 실행 순서 (다음에 실제로 할 일)

```
[확인] Step 0: 전제 조건 (코드 작성 전)
  □ 0-1. BQ SQL [REQUIRED-1] ~ [REQUIRED-3] 실행
         → MIN(date), themes/persons 컬럼 존재 여부 확인
  □ 0-2. 팀 레포에서 train.parquet 읽어 실제 컬럼 목록 확인
         python3 -c "import pandas as pd; df=pd.read_parquet('input/processed/dataset/train.parquet'); print(df.columns.tolist())"

[구현] Step 1: 실험 B
  □ 1-1. run_stacking_acled_free_baseline.py 작성 (Section 5-2 기준)
  □ 1-2. 실행 → val PR-AUC, P@5%, ECE 확인

[구현] Step 2: GDELT titles 피처 생성
  □ 2-1. BQ 쿼리 실행 → gdelt_titles_daily.parquet 저장
  □ 2-2. rolling 계산 → gdelt_titles_features.parquet 저장

[구현] Step 3: 실험 C
  □ 3-1. run_stacking_acled_free_with_titles.py 작성
  □ 3-2. C-count, C-tone 단독 실험
  □ 3-3. C-full 종합 실험
  □ 3-4. B vs C delta 보고

[조건부] Step 4: 실험 D
  □ themes/persons 있으면 진행, 없으면 GKG 수집 과제 등록
```

---

## 7. 미해결 리스크

| 항목 | 리스크 수준 | 설명 | 대응 |
|------|-----------|------|------|
| BQ gdelt_titles 2014 이전 없음 | 🔴 HIGH | train 기간 커버 불가 → 실험 C 대폭 축소 | BQ 확인이 최우선 |
| themes/persons 컬럼 없음 | 🟡 MEDIUM | D 불가 → 후속 GKG 수집 과제 | 실험 D 제외 후 B/C만 진행 |
| train.parquet 로컬 없음 | 🟡 MEDIUM | 컬럼 확인 및 실험 B 실행 불가 | 전처리 파이프라인 재실행 필요 |
| B 성능이 매우 낮음 (<0.05) | 🟡 MEDIUM | 실험 구조 재검토 필요 | B 실행 후 판단 |
| GDELT titles vs events 중복 | 🟢 LOW | v2tone과 gdelt_tone_mean 상관 | 상관 확인 후 중복 시 titles 쪽 제거 가능 |
| macis_se_score leakage | 🟢 LOW (우회됨) | ACLED-free에서 SE 제거로 우회 | B/C/D 모두 SE 미사용으로 해결 |

---

## 8. 현재 블로킹 항목

실험을 시작하기 전에 아래 두 가지가 반드시 해결되어야 한다:

```
[BLOCK-1] BQ gdelt_titles 기간 범위 확인
  → 실험 C가 2014부터 가능한지 결정. 불가하면 실험 축소 필요.

[BLOCK-2] train.parquet 실제 컬럼 확인
  → 코드에서 추론한 컬럼 목록과 실제 컬럼이 일치하는지 검증.
     컬럼명이 다르면 ALWAYS_EXCLUDE 목록 수정 필요.
```

---

*이 문서는 실험 가능 여부 점검 보고서입니다. 학습 실행 및 코드 수정은 블로킹 항목 해결 후 진행합니다.*
