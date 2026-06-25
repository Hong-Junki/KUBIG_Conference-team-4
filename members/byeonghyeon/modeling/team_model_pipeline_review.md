# 팀 레포 모델 파이프라인 리뷰
**작성자**: 김병현  
**작성일**: 2026-06-03  
**브랜치**: `add-byeonghyeon-subtree`  
**목적**: ACLED-free baseline + GDELT title 피처 실험 전, 팀 레포 파이프라인 구조 및 피처 분류 정리

---

## 실험 핵심 방향

> **ACLED 기반 정보와 macis_se_score 없이,  
> GDELT title/themes/persons 피처만으로 분쟁 조기경보 성능이 얼마나 확보되는가?**

기존 팀 final stacking model(val PR-AUC=0.2714)은 ACLED 기반 피처와 macis_se_score를 포함하며,  
SE leakage 가능성이 팀 내부에서도 미확인 상태로 지적된 바 있다.  
따라서 **이 모델은 참고용 기존 결과로만 기록하고, 실제 실험의 baseline으로 사용하지 않는다.**

실제 실험 baseline = **B. ACLED-free baseline**

---

## 1. 파이프라인 전체 흐름

```
[수집]           src/collect/
  ACLED          → input/raw/acled/{iso3}.parquet         (주 1회)
  GDELT events   → input/raw/gdelt/{iso3}.parquet         (15분 단위 BQ)
  GDELT titles   → input/raw/gdelt_titles/{iso3}/{ym}.parquet  (BQ 직적재)
  경제지표        → input/raw/economic/indicators.parquet  (일 1회)

[병합·전처리]    src/process/
  merge_raw      → input/raw_merged/
  preprocess     → input/processed/acled|gdelt|economic/
  feature_builder→ input/processed/features/features.parquet  (54 피처)
  label_builder  → input/processed/labels/labels.parquet
  build_dataset  → input/processed/dataset/train|val|test.parquet

[모델링]         model/
  run_stacking_d_with_mask_feature_ablation.py  ← 기존 참고용
  (신규) run_stacking_acled_free_baseline.py     ← 실험 B 예정
  (신규) run_stacking_acled_free_with_titles.py  ← 실험 C 예정
```

---

## 2. 데이터셋 Key 구조

| 항목 | 값 |
|------|-----|
| 조인 키 | `date` (UTC datetime), `country` (ISO3 3자리) |
| 타깃 컬럼 | `y_escalation` — 다음 3일 내 분쟁 급격 악화 |
| 양성률 | train 4.29% / val 4.07% / test 4.06% |
| 총 행수 | train 211,816 / val 10,556 / test 15,718 |
| 국가 수 | 58개국 |
| 데이터 범위 | 2014-01-01 ~ 2025-03-28 |

### 시간 기반 Split

```python
train_start: 2014-01-01,  train_end: 2023-12-31
val:   2024-01-01 ~ 2024-06-30
test:  2024-07-01 ~ 2025-03-28
```

---

## 3. 참고용 기존 모델 (실험 baseline 아님)

**실험명**: `stacking_tree_only_12y_with_mask_feature`  
**val Stacking Platt PR-AUC**: 0.2714

포함 피처:
- ACLED raw 20개 + acled_missing_mask 1개 → **ACLED-free 실험에서 전부 제거**
- macis_se_score 1개 → **ACLED-free 실험에서 제거** (leakage 미확인)
- GDELT events 17개 → 유지
- 경제지표 15개 → 유지
- country 1개 → 유지

> 이 수치는 ACLED와 SE를 포함한 결과이므로, ACLED-free 실험 결과와 직접 비교하지 않는다.

---

## 4. 피처 소스별 분류

코드 기준(`src/process/feature_builder.py`) 실제 컬럼 목록:

### 제거: ACLED raw feature (20개)

```
acled_event_count_7d, acled_event_count_14d, acled_event_count_30d
acled_fatalities_7d, acled_fatalities_14d, acled_fatalities_30d
acled_fatalities_max_7d, acled_fatalities_max_14d, acled_fatalities_max_30d
acled_ratio_battles, acled_ratio_explosions, acled_ratio_vac
acled_actor_type_1_ratio, acled_actor_type_2_ratio, acled_actor_type_3_ratio
acled_actor_type_4_ratio, acled_actor_type_5_ratio, acled_actor_type_6_ratio
acled_actor_type_7_ratio, acled_actor_type_8_ratio
```

### 제거: ACLED derived feature (1개)

```
acled_missing_mask    ← ACLED coverage 기반 파생값
```

### 제거: SE score feature (1개)

```
macis_se_score        ← LSTM AE 재구성 오차, leakage 미확인
```

### 제거: label/future leakage feature (7개, LABEL_META_COLS)

```
y, y_onset, y_escalation
fatalities_next3d, event_count_next3d
past14d_event_count, past14d_fatalities_mean
```

### 유지: GDELT events feature (~19개, 코드 기준)

```
gdelt_goldstein_mean_7d,  gdelt_goldstein_mean_14d,  gdelt_goldstein_mean_30d
gdelt_goldstein_std_7d,   gdelt_goldstein_std_14d,   gdelt_goldstein_std_30d
gdelt_tone_mean_7d,       gdelt_tone_mean_14d,       gdelt_tone_mean_30d
gdelt_mentions_sum_7d,    gdelt_mentions_sum_14d,    gdelt_mentions_sum_30d
gdelt_event_count_7d,     gdelt_event_count_14d,     gdelt_event_count_30d
gdelt_quadclass_1_ratio,  gdelt_quadclass_2_ratio
gdelt_quadclass_3_ratio,  gdelt_quadclass_4_ratio
```

> GDELT features는 shift(1) 적용 — 전일까지 데이터만 사용 (당일 제외)

### 유지: economic feature (15개, 코드 기준)

```
econ_vix,     econ_vix_pct_1d,     econ_vix_pct_7d
econ_wti,     econ_wti_pct_1d,     econ_wti_pct_7d
econ_gold,    econ_gold_pct_1d,    econ_gold_pct_7d
econ_dxy,     econ_dxy_pct_1d,     econ_dxy_pct_7d
econ_stlfsi4, econ_stlfsi4_pct_1d, econ_stlfsi4_pct_7d
```

### 유지: country/time feature (1개)

```
country      ← LightGBM categorical, XGBoost label-encoded
```

> 시간 피처(day_of_week 등): `feature_builder.py`에 구현 없음 → 현재 미포함

### Telegram feature: 현재 미적용

> `Telegram/` 폴더 존재하나 `feature_builder.py`에 통합 안 됨 → 현재 실험에서 제외

---

## 5. ACLED-free baseline에서의 피처 수 (train.parquet 실제 확인 기준)

| 카테고리 | 기존 full | ACLED-free 제거 | 실험 B 유지 |
|----------|----------|----------------|------------|
| ACLED raw (parquet) | 20 | -20 | 0 |
| acled_missing_mask (parquet) | 1 | -1 | 0 |
| macis_se_score (외부 merge) | 1 | -1 (merge 건너뜀) | 0 |
| GDELT events (parquet) | 19 | 0 | **19** |
| economic (parquet) | 15 | 0 | **15** |
| country (parquet) | 1 | 0 | **1** |
| **합계** | **57** | **-22** | **35** |

> train.parquet 직접 확인 완료 ✅ (211,816행 × 64컬럼)  
> 상세 컬럼 목록 → `acled_free_feature_inventory.md` 참조

---

## 6. ACLED-free 구현에 필요한 수정

**dataset 재생성 불필요** — `train/val/test.parquet`는 그대로 사용.  
ACLED 컬럼이 parquet에 존재하더라도, 스태킹 스크립트 내 `ALWAYS_EXCLUDE`에 추가하면  
`get_feature_cols()` 로직이 자동으로 필터링한다.

```python
# 기존 (run_stacking_d_with_mask_feature_ablation.py)
ALWAYS_EXCLUDE = [DATE_COL]   # macis_se_score는 merge 후 포함됨

# 실험 B용 수정
ACLED_COLS = [
    "acled_event_count_7d",  "acled_event_count_14d",  "acled_event_count_30d",
    "acled_fatalities_7d",   "acled_fatalities_14d",   "acled_fatalities_30d",
    "acled_fatalities_max_7d","acled_fatalities_max_14d","acled_fatalities_max_30d",
    "acled_ratio_battles",   "acled_ratio_explosions", "acled_ratio_vac",
    *[f"acled_actor_type_{i}_ratio" for i in range(1, 9)],
    "acled_missing_mask",
]
ALWAYS_EXCLUDE = [DATE_COL] + ACLED_COLS
# SE 로드 단계 건너뜀 (merge_se 호출 제거)
```

SE merge 단계도 제거해야 한다:
```python
# 제거: train = merge_se(train, se_df, "train")
# train, val, test를 SE 없이 사용
```

---

## 7. GDELT titles 데이터 현황 (BQ 확인 완료)

**BigQuery `conflict-early-warning.conflict_ew.gdelt_titles`** (2026-06-03 확인):

```
기간: 2015-02-17 ~ 2026-05-29  (2014 없음 ⚠️)
행수: 859,303,212
국가: 58개국
컬럼: date, iso3, title, url, domain, language, v2tone_avg, v2themes, v2persons
  v2themes  fill: 91.7%  → 실험 D 가능 ✅
  v2persons fill: 64.3%  → 실험 D 가능 ✅
```

로컬 백업 `input/raw/gdelt_titles/{iso3}/{YYYY-MM}.parquet`는 2022-01~만 존재하며  
BQ 스키마와 다름 (sourcecountry/seendate 있음, v2themes/v2persons 없음).  
실험 C/D는 BQ 테이블 기준으로 진행.

---

## 8. 리스크 최종 상태

| 항목 | 상태 | 비고 |
|------|------|------|
| train/val/test.parquet 실제 컬럼 목록 | ✅ 확인 완료 | 64컬럼, ACLED-free 35개 확정 |
| BQ gdelt_titles 기간 범위 | ✅ 확인 완료 | 2015-02-17 시작 (2014 없음) |
| BQ v2themes/v2persons 존재 | ✅ 확인 완료 | 91.7% / 64.3% fill → 실험 D 가능 |
| macis_se_score leakage | ⚠️ 미확인 | B/C/D 모두 SE 미사용으로 우회 |
| GDELT events shift(1) lag | ✅ 확인 완료 | feature_builder.py 명시 |
| 2014~2015-02-16 titles 없음 | ⚠️ 주의 필요 | 해당 구간 0 채움. coverage_mask 피처 검토 |
| 팀 레포에 parquet 없음 | ⚠️ 주의 필요 | 실험은 개인 레포 기준으로 실행 필요 |
