# full.parquet BigQuery 업로드 전 감사 문서

**작성일**: 2026-06-05  
**작성자**: 김병현  
**상태**: 감사/설계 단계 — 아직 BigQuery 업로드 미실행

---

## 1. full.parquet 기본 정보

| 항목 | 값 |
|------|-----|
| 경로 | `conflict-early-warning/input/processed/dataset/full.parquet` |
| 파일 크기 | 42MB |
| row 수 | 259,260 |
| column 수 | **64개** |
| row_groups | 1 |
| date min | 2014-01-01 UTC |
| date max | 2026-03-28 UTC |
| country distinct | 58개 |
| country-date 중복 | 0 (정상) |

---

## 2. target positive rate

| 컬럼 | positive rate | positive rows |
|------|--------------|---------------|
| y | 54.10% | 140,255 |
| y_onset | 0.94% | 2,436 |
| y_escalation | 3.91% | 10,147 |

---

## 3. 전체 64개 컬럼 목록 (순서 보존)

| # | column | dtype |
|---|--------|-------|
| 00 | date | timestamp[ns, tz=UTC] |
| 01 | acled_event_count_7d | double |
| 02 | acled_fatalities_7d | double |
| 03 | acled_fatalities_max_7d | double |
| 04 | acled_event_count_14d | double |
| 05 | acled_fatalities_14d | double |
| 06 | acled_fatalities_max_14d | double |
| 07 | acled_event_count_30d | double |
| 08 | acled_fatalities_30d | double |
| 09 | acled_fatalities_max_30d | double |
| 10 | acled_ratio_battles | double |
| 11 | acled_ratio_explosions | double |
| 12 | acled_ratio_vac | double |
| 13 | acled_actor_type_1_ratio | double |
| 14 | acled_actor_type_2_ratio | double |
| 15 | acled_actor_type_3_ratio | double |
| 16 | acled_actor_type_4_ratio | double |
| 17 | acled_actor_type_5_ratio | double |
| 18 | acled_actor_type_6_ratio | double |
| 19 | acled_actor_type_7_ratio | double |
| 20 | acled_actor_type_8_ratio | double |
| 21 | acled_missing_mask | double |
| 22 | country | string |
| 23 | gdelt_goldstein_mean_7d | double |
| 24 | gdelt_goldstein_std_7d | double |
| 25 | gdelt_tone_mean_7d | double |
| 26 | gdelt_mentions_sum_7d | double |
| 27 | gdelt_event_count_7d | double |
| 28 | gdelt_goldstein_mean_14d | double |
| 29 | gdelt_goldstein_std_14d | double |
| 30 | gdelt_tone_mean_14d | double |
| 31 | gdelt_mentions_sum_14d | double |
| 32 | gdelt_event_count_14d | double |
| 33 | gdelt_goldstein_mean_30d | double |
| 34 | gdelt_goldstein_std_30d | double |
| 35 | gdelt_tone_mean_30d | double |
| 36 | gdelt_mentions_sum_36d | double |
| 37 | gdelt_event_count_30d | double |
| 38 | gdelt_quadclass_1_ratio | double |
| 39 | gdelt_quadclass_2_ratio | double |
| 40 | gdelt_quadclass_3_ratio | double |
| 41 | gdelt_quadclass_4_ratio | double |
| 42 | econ_vix | double |
| 43 | econ_vix_pct_1d | double |
| 44 | econ_vix_pct_7d | double |
| 45 | econ_wti | double |
| 46 | econ_wti_pct_1d | double |
| 47 | econ_wti_pct_7d | double |
| 48 | econ_gold | double |
| 49 | econ_gold_pct_1d | double |
| 50 | econ_gold_pct_7d | double |
| 51 | econ_dxy | double |
| 52 | econ_dxy_pct_1d | double |
| 53 | econ_dxy_pct_7d | double |
| 54 | econ_stlfsi4 | double |
| 55 | econ_stlfsi4_pct_1d | double |
| 56 | econ_stlfsi4_pct_7d | double |
| 57 | y | int64 |
| 58 | y_onset | int64 |
| 59 | y_escalation | int64 |
| 60 | fatalities_next3d | int64 |
| 61 | event_count_next3d | int64 |
| 62 | past14d_event_count | int64 |
| 63 | past14d_fatalities_mean | double |

---

## 4. 그룹별 컬럼 분류

### group 합계 검증

| group | 개수 |
|-------|------|
| id/date/country | 2 |
| target/label | 3 |
| label_generation/future | 4 |
| GDELT event features | 19 |
| economic features | 15 |
| ACLED raw features | 21 |
| safe_acled features | 0 |
| macis_se_score | 0 |
| GDELT title features | 0 |
| GDELT theme/person features | 0 |
| 기타 | 0 |
| **합계** | **64** ✅ |

---

### id/date/country (2개)
```
date, country
```

---

### target/label (3개)
```
y, y_onset, y_escalation
```

---

### label_generation/future columns (4개)
> ⚠️ 미래 정보 또는 label 생성 보조 변수. 운영 feature로 사용 금지. 대시보드 노출 금지.

```
fatalities_next3d
event_count_next3d
past14d_event_count
past14d_fatalities_mean
```

---

### GDELT event features (19개)
```
gdelt_goldstein_mean_7d
gdelt_goldstein_std_7d
gdelt_tone_mean_7d
gdelt_mentions_sum_7d
gdelt_event_count_7d
gdelt_goldstein_mean_14d
gdelt_goldstein_std_14d
gdelt_tone_mean_14d
gdelt_mentions_sum_14d
gdelt_event_count_14d
gdelt_goldstein_mean_30d
gdelt_goldstein_std_30d
gdelt_tone_mean_30d
gdelt_mentions_sum_30d
gdelt_event_count_30d
gdelt_quadclass_1_ratio
gdelt_quadclass_2_ratio
gdelt_quadclass_3_ratio
gdelt_quadclass_4_ratio
```

---

### economic features (15개)
```
econ_vix
econ_vix_pct_1d
econ_vix_pct_7d
econ_wti
econ_wti_pct_1d
econ_wti_pct_7d
econ_gold
econ_gold_pct_1d
econ_gold_pct_7d
econ_dxy
econ_dxy_pct_1d
econ_dxy_pct_7d
econ_stlfsi4
econ_stlfsi4_pct_1d
econ_stlfsi4_pct_7d
```

---

### ACLED raw features (21개)
> ⚠️ 운영 시 ACLED API 미가용. 운영 예측 파이프라인 feature로 사용 금지.

```
acled_event_count_7d
acled_fatalities_7d
acled_fatalities_max_7d
acled_event_count_14d
acled_fatalities_14d
acled_fatalities_max_14d
acled_event_count_30d
acled_fatalities_30d
acled_fatalities_max_30d
acled_ratio_battles
acled_ratio_explosions
acled_ratio_vac
acled_actor_type_1_ratio
acled_actor_type_2_ratio
acled_actor_type_3_ratio
acled_actor_type_4_ratio
acled_actor_type_5_ratio
acled_actor_type_6_ratio
acled_actor_type_7_ratio
acled_actor_type_8_ratio
acled_missing_mask
```

---

### safe_acled features (0개)
❌ 없음

### macis_se_score (0개)
❌ 없음

### GDELT title features (0개)
❌ 없음

### GDELT theme/person features (0개)
❌ 없음

---

## 5. ACLED-free feature 후보 (37개)

`modeling_acled_free_view`에 포함할 컬럼. feature 35개 + target 1개(y_escalation) + id 2개(date, country).

| 역할 | 컬럼 |
|------|------|
| id | date, country |
| GDELT event (19개) | gdelt_goldstein_mean_7d, gdelt_goldstein_std_7d, gdelt_tone_mean_7d, gdelt_mentions_sum_7d, gdelt_event_count_7d, gdelt_goldstein_mean_14d, gdelt_goldstein_std_14d, gdelt_tone_mean_14d, gdelt_mentions_sum_14d, gdelt_event_count_14d, gdelt_goldstein_mean_30d, gdelt_goldstein_std_30d, gdelt_tone_mean_30d, gdelt_mentions_sum_30d, gdelt_event_count_30d, gdelt_quadclass_1_ratio, gdelt_quadclass_2_ratio, gdelt_quadclass_3_ratio, gdelt_quadclass_4_ratio |
| econ (15개) | econ_vix, econ_vix_pct_1d, econ_vix_pct_7d, econ_wti, econ_wti_pct_1d, econ_wti_pct_7d, econ_gold, econ_gold_pct_1d, econ_gold_pct_7d, econ_dxy, econ_dxy_pct_1d, econ_dxy_pct_7d, econ_stlfsi4, econ_stlfsi4_pct_1d, econ_stlfsi4_pct_7d |
| target | y_escalation |

**제외 이유별 목록:**
- acled_* (21개): 운영 시 미가용
- y, y_onset (2개): label — 예측 대상이므로 feature 아님
- fatalities_next3d, event_count_next3d (2개): 미래 정보
- past14d_event_count, past14d_fatalities_mean (2개): label 생성 보조 변수

---

## 6. BigQuery 추천 구조

### 구조

```
conflict_ew.modeling_full_dataset   ← full.parquet 64개 컬럼 전체
    ↓ (BigQuery View)
conflict_ew.modeling_acled_free_view ← 37개 컬럼 (ACLED·label-gen 제외)
```

### 적절성 판단: ✅ 추천

| 항목 | 판단 |
|------|------|
| 재현성 | modeling_full_dataset에 ACLED 포함 전체 보존 → F/B/C/D 계열 실험 재현 가능 |
| 운영 모델 기준 | View로 ACLED/미래 컬럼 차단 → O0/O1/O2 개발 시 실수 방지 |
| 비용 | View는 저장 비용 없음 (full_dataset 조회 시에만 쿼리 비용 발생) |
| 확장성 | gdelt_title/theme/person 컬럼은 full_dataset에 없으므로 O1/O2용 feature는 별도 BigQuery 집계 필요 |

### 테이블 설정 (완료)

| 테이블/뷰 | partition | clustering | 상태 |
|-----------|-----------|------------|------|
| modeling_full_dataset | MONTH on `date` | `country` | **생성 완료** |
| modeling_acled_free_view | (view, 설정 불필요) | — | **생성 완료** |

### View DDL (실제 적용)

```sql
SELECT
  date, country,
  gdelt_goldstein_mean_7d, gdelt_goldstein_std_7d, gdelt_tone_mean_7d,
  gdelt_mentions_sum_7d, gdelt_event_count_7d,
  gdelt_goldstein_mean_14d, gdelt_goldstein_std_14d, gdelt_tone_mean_14d,
  gdelt_mentions_sum_14d, gdelt_event_count_14d,
  gdelt_goldstein_mean_30d, gdelt_goldstein_std_30d, gdelt_tone_mean_30d,
  gdelt_mentions_sum_30d, gdelt_event_count_30d,
  gdelt_quadclass_1_ratio, gdelt_quadclass_2_ratio,
  gdelt_quadclass_3_ratio, gdelt_quadclass_4_ratio,
  econ_vix, econ_vix_pct_1d, econ_vix_pct_7d,
  econ_wti, econ_wti_pct_1d, econ_wti_pct_7d,
  econ_gold, econ_gold_pct_1d, econ_gold_pct_7d,
  econ_dxy, econ_dxy_pct_1d, econ_dxy_pct_7d,
  econ_stlfsi4, econ_stlfsi4_pct_1d, econ_stlfsi4_pct_7d,
  y_escalation
FROM `conflict-ew-mvp-20260604.conflict_ew.modeling_full_dataset`
```

---

## 7. 업로드 및 View 생성 결과 (2026-06-05 완료)

### modeling_full_dataset 검증

| 항목 | 결과 | 기대 | 일치 |
|------|------|------|------|
| row count | 259,260 | 259,260 | ✅ |
| column count | 64 | 64 | ✅ |
| country distinct | 58 | 58 | ✅ |
| min date | 2014-01-01 UTC | 2014-01-01 | ✅ |
| max date | 2026-03-28 UTC | 2026-03-28 | ✅ |
| null date | 0 | 0 | ✅ |
| null country | 0 | 0 | ✅ |
| partitioning | MONTH on date | MONTH | ✅ |
| clustering | country | country | ✅ |

### modeling_acled_free_view 검증

| 항목 | 결과 | 기대 | 일치 |
|------|------|------|------|
| row count | 259,260 | 259,260 | ✅ |
| column count | 37 | 37 | ✅ |
| country distinct | 58 | 58 | ✅ |
| min date | 2014-01-01 UTC | — | — |
| max date | 2026-03-28 UTC | — | — |
| y_escalation positive rate | 0.0391 (3.91%) | — | — |
| acled_* in view | 없음 | 없음 | ✅ |
| future/past14d cols in view | 없음 | 없음 | ✅ |
| y, y_onset in view | 없음 | 없음 | ✅ |

---

## 8. 대시보드/운영에서 노출하면 안 되는 컬럼

| 컬럼 | 이유 |
|------|------|
| `fatalities_next3d`, `event_count_next3d` | 미래 정보 (t+1~t+3 집계) |
| `past14d_event_count`, `past14d_fatalities_mean` | label 생성 보조 변수 |
| `acled_*` (21개) | 운영 시 ACLED API 미가용 |
| `y`, `y_onset`, `y_escalation` | 정답 레이블 — 예측 대상, 대시보드 input 불가 |

---

## 9. 현재 BigQuery 상태 (2026-06-05 기준)

| 리소스 | 이름 | 상태 |
|--------|------|------|
| event-level table | `gdelt_processed_events` | 존재 (309,533,500 rows) |
| event-level test table | `gdelt_processed_events_load_test` | 존재 (31,768,110 rows) |
| modeling table | `modeling_full_dataset` | **생성 완료** (259,260 rows, 64 cols) |
| modeling view | `modeling_acled_free_view` | **생성 완료** (37 cols, ACLED 제외) |
| 모델 학습 | — | **미실행** |

---

## 10. 다음 단계

```
[1] O0_bq_clean 실험 (아래 섹션 참고)

[2] gdelt_title/theme/person feature 추가 (O1/O2 실험용)
    - gdelt_processed_events에서 SQL 집계 필요
    - 또는 기존 feature parquet를 별도 BigQuery 테이블로 업로드
```

---

## 11. 다음 실험: O0_bq_clean

| 항목 | 내용 |
|------|------|
| 실험명 | O0_bq_clean |
| 입력 | `conflict-ew-mvp-20260604.conflict_ew.modeling_acled_free_view` |
| feature | GDELT event 19개 + econ 15개 + country (원핫 또는 label encode) |
| target | `y_escalation` |
| train_fit | 2014-01-01 ~ 2022-12-31 |
| tune_cal | 2023-01-01 ~ 2023-12-31 |
| val_eval | 2024-01-01 ~ 2024-06-30 |
| test | 2024-07-01~ — **미평가** (최종 모델 확정 후 1회만) |
| 목적 | 로컬 `O0_clean` 결과와 BigQuery view 기반 결과가 재현되는지 확인 |
| 비교 기준 | 기존 local O0_clean Stacking Platt PR-AUC **0.0583** |
| 모델 | LGBM + XGBoost stacking, meta: Logistic Regression (Platt) |
| 평가 지표 | PR-AUC (val_eval 기준) |

**주의:**
- test set은 아직 평가하지 않는다
- val_eval 기준으로만 비교 → BigQuery 적재가 결과에 영향 없는지 확인

---

*모델 학습/test set 평가는 수행하지 않았다.*  
*최초 생성: 2026-06-05 / 최종 수정: 2026-06-05 (O0_bq_clean 계획 추가, test table 삭제, 작성자명 통일)*
