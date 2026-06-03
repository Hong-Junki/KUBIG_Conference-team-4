# ACLED-free 실험 준비 상태 점검
**작성자**: byeonghyeon  
**작성일**: 2026-06-03  
**목적**: 실험 B/C/D 시작 전 실행 가능 여부 전체 점검

---

## 1. 실험 구조 요약

| 실험 | 피처 구성 | 상태 |
|------|----------|------|
| **B** (ACLED-free baseline) | GDELT events + 경제 + country | 🟡 구현 필요 |
| **C** (B + GDELT titles) | B + titles count/tone 집계 피처 | 🟢 데이터 확인 완료 (2015~) |
| **D** (C + themes/persons) | C + v2themes/v2persons 키워드 피처 | 🟢 진행 가능 (91.7% / 64.3% fill) |
| **Reference** (기존 full model) | ACLED + SE + full 57개 | ✅ 기존 결과, 재실험 불필요 |

**핵심 비교**: B → C (GDELT titles 피처의 독립 기여)

---

## 2. BigQuery gdelt_titles 스키마 확인 결과 ✅ BLOCK-1 완료

**확인 일시**: 2026-06-03  
**인증**: `conflict-early-warning-4672e791d960.json` (서비스 계정)

### 2-1. 기간 및 규모

| 항목 | 값 |
|------|-----|
| MIN(date) | **2015-02-17** |
| MAX(date) | 2026-05-29 |
| 전체 행수 | **859,303,212** (8억 5천만 행) |
| 국가 수 | **58개국** (train/val/test와 동일) |

> ⚠️ **중요**: 데이터가 **2015-02-17**부터 시작된다.  
> train 시작일(2014-01-01)~2015-02-16 약 13.5개월이 완전 결측.  
> 이 기간 GDELT titles 피처는 0으로 채워지며, OOF F1(2018 예측)에 영향.

### 2-2. BQ 실제 컬럼 목록

```
컬럼명          타입       nullable   설명
-----------    --------   --------   --------------------------------
date           DATE       NOT NULL   보도 날짜
iso3           STRING     NOT NULL   국가 ISO3
title          STRING     nullable   기사 제목
url            STRING     NOT NULL   기사 URL
domain         STRING     nullable   출처 도메인
language       STRING     nullable   언어 코드
v2tone_avg     FLOAT64    nullable   기사 톤 점수
v2themes       STRING     nullable   GKG 테마 목록 (세미콜론 구분)  ← 실험 D 가능
v2persons      STRING     nullable   GKG 인물 목록 (세미콜론 구분)  ← 실험 D 가능
```

**로컬 백업과의 차이**:
- BQ에 있고 로컬에 없음: `v2themes`, `v2persons`
- 로컬에 있고 BQ에 없음: `sourcecountry`, `seendate`

### 2-3. themes / persons 컬럼 fill 비율

| 컬럼 | non-null 행수 | fill 비율 |
|------|-------------|----------|
| `v2themes` | 787,990,601 | **91.7%** → 실험 D 가능 ✅ |
| `v2persons` | 552,570,230 | **64.3%** → 실험 D 가능 ✅ |

### 2-4. v2themes / v2persons 포맷

GDELT GKG 형식 — `테마명,문자위치;테마명,문자위치;...`

```
v2themes 예시:
  CEASEFIRE,65;CEASEFIRE,146;BLOCKADE,4133;SEIGE,4133;TAX_FNCACT_LEADER,14...

v2persons 예시:
  Donald Trump,858;Scott Bessent,1867;Ali Khamenei,1476;Masoud Pezeshkian,6144...
```

→ **집계 방식**: 세미콜론으로 split → 콤마 앞 테마명만 추출 → 키워드 매칭  
→ 위치 숫자(예: `,65`)는 기사 내 character offset이며, count 집계와 무관

### 2-5. 연도별 row 수

| 연도 | row_count | n_countries |
|------|-----------|-------------|
| 2015 | 94,911,369 | 58 |
| 2016 | 135,291,743 | 58 |
| 2017 | 114,059,222 | 58 |
| 2018 | 91,668,772 | 58 |
| 2019 | 67,902,081 | 58 |
| 2020 | 50,776,207 | 58 |
| 2021 | 46,223,097 | 58 |
| 2022 | 52,015,805 | 58 |
| 2023 | 66,579,262 | 58 |
| 2024 | 62,090,091 | 58 |
| 2025 | 53,656,192 | 58 |
| 2026 | 24,129,371 | 58 |

> 2015년이 가장 많고(1.35억), 2021년이 가장 적음(4.6천만).  
> 2016년 이후부터 모든 연도에 58개국이 존재.

### 2-6. 국가별 row 수 (상위 10 / 하위 10)

**상위 10** (보도량 많음)

| iso3 | row_count | first_date | last_date |
|------|-----------|------------|-----------|
| RUS | 125,944,321 | 2015-02-18 | 2026-05-29 |
| TUR | 71,874,358 | 2015-02-17 | 2026-05-29 |
| IND | 70,810,641 | 2015-02-17 | 2026-05-29 |
| UKR | 66,078,474 | 2015-02-18 | 2026-05-29 |
| MEX | 48,318,445 | 2015-02-18 | 2026-05-29 |
| ISR | 45,952,006 | 2015-02-17 | 2026-05-29 |
| EGY | 35,073,578 | 2015-02-18 | 2026-05-29 |
| SYR | 33,721,264 | 2015-02-18 | 2026-05-29 |
| IDN | 32,166,829 | 2015-02-18 | 2026-05-29 |
| IRN | 30,160,070 | 2015-02-18 | 2026-05-29 |

**하위 10** (보도량 적음)

| iso3 | row_count | first_date | last_date |
|------|-----------|------------|-----------|
| MOZ | 1,410,801 | 2015-02-18 | 2026-05-29 |
| SSD | 1,401,147 | 2015-02-18 | 2026-05-29 |
| TJK | 1,275,569 | 2015-02-18 | 2026-05-29 |
| MDG | 1,046,129 | 2015-02-18 | 2026-05-29 |
| ERI | 968,845 | 2015-02-18 | 2026-05-29 |
| COD | 877,548 | 2015-02-18 | 2026-05-29 |
| TGO | 794,561 | 2015-02-18 | 2026-05-29 |
| SLE | 768,929 | 2015-02-18 | 2026-05-29 |
| CAF | 671,282 | 2015-02-18 | 2026-05-29 |
| GNB | 314,009 | 2015-02-18 | 2026-05-29 |

> 최하위 GNB(기니비사우) 31만 건 vs 최상위 RUS 1.26억 건 — 400배 차이.  
> 보도량 편차가 크므로 집계 피처는 **절대값 + 비율 모두** 사용 필요.

### 2-7. BLOCK-1 해결 결론

| 항목 | 결과 |
|------|------|
| 데이터 시작일 | 2015-02-17 (2014 없음 → 0 채움 처리 필요) |
| v2themes 컬럼 | ✅ 있음 (91.7% fill) |
| v2persons 컬럼 | ✅ 있음 (64.3% fill) |
| 실험 C 가능 여부 | ✅ 가능 (2015-2025 데이터 충분) |
| 실험 D 가능 여부 | ✅ 가능 (themes/persons 모두 존재) |
| 2014 결측 처리 | → 0 채움 (기사 0건으로 간주) |

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
| 2014 데이터 없음 (2015-02-17 시작) | 🟡 MEDIUM | 2014~2015-02-16 titles 피처 = 0. OOF F1(train ≤2017)에서 초기 연도 신호 약함 | 0 채움 처리. 필요 시 OOF 기간 2015 이후로 조정 가능 |
| 국가별 보도량 400배 편차 | 🟡 MEDIUM | GNB(31만) vs RUS(1.26억). 절대값 피처가 국가 크기 편향을 학습할 수 있음 | 절대값 + 국가별 z-score 또는 비율 피처 병행 |
| v2themes/v2persons 파싱 | 🟡 MEDIUM | 세미콜론 split → 콤마 앞 추출 필요. BQ 문자열 처리 비용 큼 | BQ에서 일별 집계 후 로컬 저장. 전체 재처리는 1회만 |
| train.parquet 로컬 없음 | 🟡 MEDIUM | 컬럼 확인 및 실험 B 실행 불가 | 전처리 파이프라인 재실행 필요 (BLOCK-2) |
| GDELT titles v2tone vs events tone 중복 | 🟢 LOW | v2tone_avg와 기존 gdelt_tone_mean 상관 가능성 | 상관관계 확인 후 중복 시 titles 쪽 조정 |
| macis_se_score leakage | 🟢 LOW (우회됨) | ACLED-free에서 SE 제거로 우회 | B/C/D 모두 SE 미사용 |

---

## 8. 블로킹 항목 상태

| 항목 | 상태 |
|------|------|
| **BLOCK-1**: BQ gdelt_titles 기간/스키마 확인 | ✅ **완료** — 2015-02-17 시작, v2themes/v2persons 확인 |
| **BLOCK-2**: train.parquet 실제 컬럼 확인 | ✅ **완료** — 개인 레포에서 확인 (64컬럼, ACLED-free 35개 확정) |

### BLOCK-2 결과 요약

```
파일 위치: conflict-early-warning/input/processed/dataset/train.parquet
  shape: (211,816 × 64)
  실제 컬럼: key 2 + label 7 + ACLED 21 + GDELT 19 + econ 15 + country 1 = 64
  ACLED-free feature_cols (B): 35개 확정 (GDELT 19 + econ 15 + country 1)
  macis_se_score: train.parquet에 없음 (stacking script에서 외부 merge)
  Telegram/시간 피처: 없음
```

> 상세 컬럼 분류 → `acled_free_feature_inventory.md` 참조

두 블로킹 항목이 모두 해결되었다. **실험 B 스크립트 작성 가능 상태.**

---

*이 문서는 실험 가능 여부 점검 보고서입니다. 두 BLOCK이 완료되어 실험 B 구현 단계로 진행 가능합니다.*
