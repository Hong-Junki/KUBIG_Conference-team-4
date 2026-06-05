# GDELT Title 피처 실험 계획 (ACLED-free 기준)
**작성자**: 김병현  
**작성일**: 2026-06-03  
**핵심 질문**: ACLED과 macis_se_score 없이, 새 GDELT title/themes/persons 피처만으로 성능을 얼마나 확보할 수 있는가?

---

## 1. 배경 및 동기

기존 팀 final stacking model(val PR-AUC=0.2714)은 ACLED 기반 피처 20개와  
macis_se_score에 크게 의존한다. 팀 내부에서도 SE leakage 가능성이 미확인 상태로 지적된 바 있다.

따라서 **기존 full model은 참고용 기존 결과로만 기록하고, 실험의 baseline으로 사용하지 않는다.**

**실제 실험 구조는 B / C / D 세 가지로 구성한다.**

---

## 2. 실험 구조

### Reference: 기존 full model (참고용 기존 결과, 실험 baseline 아님)

- ACLED 20개 + macis_se_score + acled_missing_mask + GDELT + 경제 + country = 57개 피처
- val Stacking Platt PR-AUC = 0.2714
- SE leakage 미확인 → 이번 실험 결과와 직접 비교하지 않음

---

### B. ACLED-free baseline ← **실제 실험 baseline**

- **제거**: ACLED raw 20개 + acled_missing_mask + macis_se_score
- **유지**: GDELT events ~19개 + 경제지표 15개 + country
- **피처 수**: 약 35개
- **목적**: ACLED 없이 달성 가능한 성능 수준 확인
- **이 결과가 C와의 비교 기준점**

---

### C. ACLED-free + GDELT title/tone aggregate features ← **핵심 실험**

- **B의 피처 set에 GDELT titles 집계 피처 추가**
- 새 BigQuery 테이블 `conflict-early-warning.conflict_ew.gdelt_titles`에서 country-date 단위 집계
- **핵심 비교: C val PR-AUC − B val PR-AUC = GDELT titles 피처의 실질 기여**
- 개선 임계값: **+0.003 PR-AUC** (기존 ablation 실험과 동일 기준)

---

### D. ACLED-free + GDELT title/tone + themes/persons features ✅ 진행 가능

- **BQ 확인 완료**: v2themes(91.7% fill), v2persons(64.3% fill) 존재 확인
- 실험 C 결과가 B 대비 의미 있는 개선을 보일 경우 이어서 진행
- v2themes 파싱: `;` split → `,` 앞 테마명 추출 → 분쟁 관련 키워드 카운트

---

### 실험 비교 구조

```
Reference  (full, ACLED 포함)   → 참고용 기존 결과만. 비교 근거 아님.
B          (ACLED-free)          → 실제 baseline
C          (B + GDELT titles)    → 핵심 실험
D          (C + themes/persons)  → 조건부 확장 실험

핵심 비교:  B → C  (GDELT titles 피처의 기여)
보조 비교:  C → D  (themes/persons 추가 기여, 조건부)
```

---

## 3. BigQuery 테이블 스키마 ✅ 확인 완료 (2026-06-03)

**실제 BQ 스키마** (`INFORMATION_SCHEMA.COLUMNS` 직접 조회):

```
컬럼명       타입      nullable  설명
-----------  -------   --------  ------------------------------------------------
date         DATE      NOT NULL  보도 날짜 (UTC)
iso3         STRING    NOT NULL  대상 국가 ISO3
title        STRING    nullable  기사 제목
url          STRING    NOT NULL  기사 URL
domain       STRING    nullable  출처 도메인
language     STRING    nullable  언어 코드 (eng, ara, fra, rus, ...)
v2tone_avg   FLOAT64   nullable  기사 톤 점수 (음수=부정적)
v2themes     STRING    nullable  GKG 테마 목록 (`;` 구분, `,`로 위치 포함)
v2persons    STRING    nullable  GKG 인물 목록 (`;` 구분, `,`로 위치 포함)
```

**로컬 백업과 차이**: BQ는 sourcecountry/seendate 없음, v2themes/v2persons 있음

**기간 및 규모**:
- MIN(date): **2015-02-17** (2014 없음 — 해당 구간 0 처리 필요)
- MAX(date): 2026-05-29
- 총 행수: **859,303,212** / 58개국
- v2themes fill: **91.7%** → 실험 D 가능 ✅
- v2persons fill: **64.3%** → 실험 D 가능 ✅

**v2themes/v2persons 포맷**: `테마명,문자위치;테마명,문자위치;...`
- 숫자는 기사 내 character offset (count 아님)
- 집계: `;` split → `,` 앞 테마명만 추출 → 키워드 매칭

---

## 4. BigQuery 확인 SQL (실험 전 반드시 실행)

```sql
-- 4-1. 기간/규모 확인
SELECT
  MIN(date)             AS min_date,
  MAX(date)             AS max_date,
  COUNT(*)              AS total_rows,
  COUNT(DISTINCT iso3)  AS n_countries
FROM `conflict-early-warning.conflict_ew.gdelt_titles`;

-- 4-2. 컬럼 목록 확인 (themes/persons 존재 여부)
SELECT column_name, data_type
FROM `conflict-early-warning.conflict_ew.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'gdelt_titles';

-- 4-3. 연도별 row 수
SELECT
  EXTRACT(YEAR FROM date) AS year,
  COUNT(*) AS rows,
  COUNT(DISTINCT iso3) AS n_countries
FROM `conflict-early-warning.conflict_ew.gdelt_titles`
GROUP BY year
ORDER BY year;

-- 4-4. 국가별 coverage 요약
SELECT
  iso3,
  MIN(date) AS first_date,
  MAX(date) AS last_date,
  COUNT(*) AS total_rows
FROM `conflict-early-warning.conflict_ew.gdelt_titles`
GROUP BY iso3
ORDER BY iso3;
```

---

## 5. 파이프라인 결합 위치

```
실험 B:
  기존 train/val/test.parquet 그대로 사용
  stacking script에서 ALWAYS_EXCLUDE에 ACLED 컬럼 추가 + SE merge 제거
  → 새 스크립트: run_stacking_acled_free_baseline.py

실험 C:
  신규 모듈: src/process/gdelt_titles_feature_builder.py
    BQ 또는 로컬 parquet에서 일별 집계
    → input/processed/gdelt_titles/gdelt_titles_features.parquet

  B의 feature set에 left-merge:
    on=['date', 'country'], how='left', 결측 → 0
  → 새 스크립트: run_stacking_acled_free_with_titles.py

실험 D (조건부):
  C에 themes/persons 기반 집계 피처 추가 merge
```

---

## 6. GDELT titles 집계 SQL (실험 C용)

### 6-1. 일별 국가 기사 집계

```sql
SELECT
  date,
  iso3                                    AS country,

  COUNT(*)                                AS gdelt_title_count_1d,
  COUNTIF(title IS NOT NULL)             AS gdelt_title_nonnull_count_1d,
  COUNTIF(language = 'eng')             AS gdelt_title_eng_count_1d,

  AVG(v2tone_avg)                         AS gdelt_title_tone_mean_1d,
  STDDEV(v2tone_avg)                      AS gdelt_title_tone_std_1d,
  MIN(v2tone_avg)                         AS gdelt_title_tone_min_1d,
  COUNTIF(v2tone_avg < -5)                AS gdelt_title_negative_count_1d,
  COUNTIF(v2tone_avg > 2)                 AS gdelt_title_positive_count_1d,

  COUNT(DISTINCT domain)                  AS gdelt_title_domain_diversity_1d,
  COUNT(DISTINCT language)                AS gdelt_title_lang_diversity_1d

FROM `conflict-early-warning.conflict_ew.gdelt_titles`
WHERE date BETWEEN '2014-01-01' AND '2025-03-31'
GROUP BY date, iso3
ORDER BY iso3, date
```

### 6-2. Rolling window 피처 (7일/14일)

```sql
WITH daily AS (
  SELECT
    date, iso3,
    COUNT(*) AS count_1d,
    AVG(v2tone_avg) AS tone_1d
  FROM `conflict-early-warning.conflict_ew.gdelt_titles`
  WHERE date BETWEEN '2014-01-01' AND '2025-03-31'
  GROUP BY date, iso3
)
SELECT
  date, iso3,
  count_1d AS gdelt_title_count_1d,
  tone_1d  AS gdelt_title_tone_mean_1d,

  SUM(count_1d) OVER (
    PARTITION BY iso3 ORDER BY date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW   -- 7일, 과거 방향
  ) AS gdelt_title_count_7d,

  AVG(tone_1d) OVER (
    PARTITION BY iso3 ORDER BY date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS gdelt_title_tone_mean_7d,

  SUM(count_1d) OVER (
    PARTITION BY iso3 ORDER BY date
    ROWS BETWEEN 13 PRECEDING AND CURRENT ROW  -- 14일
  ) AS gdelt_title_count_14d,

  AVG(tone_1d) OVER (
    PARTITION BY iso3 ORDER BY date
    ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
  ) AS gdelt_title_tone_mean_14d

FROM daily
ORDER BY iso3, date
```

---

## 7. GDELT titles 피처 후보

### 7-1. 실험 C (기본, v2tone_avg + count 기반)

| 피처명 | 설명 |
|--------|------|
| `gdelt_title_count_1d` | 하루 기사 건수 |
| `gdelt_title_tone_mean_1d` | 평균 톤 |
| `gdelt_title_tone_std_1d` | 톤 표준편차 |
| `gdelt_title_tone_min_1d` | 최저 톤 |
| `gdelt_title_negative_count_1d` | 톤<-5 기사 수 |
| `gdelt_title_eng_count_1d` | 영어 기사 수 |
| `gdelt_title_domain_diversity_1d` | 고유 도메인 수 |
| `gdelt_title_count_7d` | 7일 기사 수 합계 |
| `gdelt_title_tone_mean_7d` | 7일 평균 톤 |
| `gdelt_title_count_14d` | 14일 기사 수 합계 |
| `gdelt_title_tone_mean_14d` | 14일 평균 톤 |
| `gdelt_title_tone_trend_7d` | 톤 7일 변화량 (t − t-7) |

### 7-2. 실험 D (조건부, themes/persons 컬럼 있을 경우)

| 피처명 | 키워드 |
|--------|--------|
| `gdelt_theme_conflict_count_1d` | CONFLICT, MILITARY_ATTACK, WEAPONS |
| `gdelt_theme_protest_count_1d` | PROTEST, RIOT |
| `gdelt_theme_refugee_count_1d` | REFUGEES, DISPLACED_PERSONS |
| `gdelt_theme_sanction_count_1d` | ECON_SANCTIONS, UN_SANCTIONS |
| `gdelt_persons_named_count_1d` | persons 비결측 건수 |

---

## 8. Leakage 방지 체크리스트

| 규칙 | 확인 방법 |
|------|----------|
| t일 feature ← t일까지 GDELT 데이터만 | Rolling: `ROWS BETWEEN N PRECEDING AND CURRENT ROW` |
| t+1 이후 기사 미포함 | future FOLLOWING 금지 |
| GDELT events는 shift(1) 적용 | `feature_builder.py` 확인 완료 ✅ |
| titles 피처도 shift(1) 적용 여부 결정 필요 | `gdelt_titles_feature_builder.py` 구현 시 결정 |
| Scaler fit → train 기준만 | count/평균 피처 → scaler 불필요. 결측 → 0 |
| macis_se_score 완전 제거 | B, C, D 모두 SE 미사용 |

---

## 9. Ablation 실험 계획

### Phase 0: B baseline 수립

| 실험 | 피처 | 목적 |
|------|------|------|
| **B** | GDELT events + 경제 + country (~35개) | 실제 baseline 확립 |

→ B 성능 수준 확인 후 Phase 1 진행 여부 결정  
(PR-AUC < 0.05이면 모델 구조 재검토)

### Phase 1: GDELT titles 기본 추가 (C 구성)

| 실험 | 추가 피처 | 목적 |
|------|----------|------|
| `C-count` | `gdelt_title_count_7d` 1개 | 보도량 단독 기여 |
| `C-tone` | `gdelt_title_tone_mean_7d` 1개 | 톤 단독 기여 |
| `C-basic` | count + tone 1d+7d 4개 | 기본 결합 효과 |
| `C-full` | 1d + 7d + 14d 피처 ~12개 | 종합 추가 |

→ B 대비 +0.003 이상 → C 개선 판정, 최선 구성 선정

### Phase 2: Rolling window 비교 (Phase 1 최선 기반)

| 실험 | 윈도우 | 내용 |
|------|--------|------|
| `C-1d` | 1일 | 당일 신호 |
| `C-7d` | 7일 | 주간 |
| `C-14d` | 14일 | 2주 |
| `C-7d-14d` | 7d+14d | 다중 |

### Phase 3: D 실험 (themes/persons 컬럼 있을 경우만)

| 실험 | 추가 피처 |
|------|----------|
| `D-conflict-theme` | conflict/protest theme count |
| `D-all-themes` | 전체 테마 그룹 |
| `D-persons` | persons count |

---

## 10. 성능 비교 지표

| 지표 | 설명 | C 개선 판정 기준 |
|------|------|----------------|
| **Stacking Platt PR-AUC** | 주지표 | B 대비 **+0.003** 이상 |
| P@top5% | 상위 5% 정밀도 | B 대비 절댓값 기록 |
| ECE | 보정 오차 | 낮을수록 좋음 |
| Brier Score | 확률 정확도 | 낮을수록 좋음 |

> **기존 Reference 모델(0.2714)을 C의 목표로 삼지 않는다.**  
> C의 목표는 오직 B 대비 개선이다.

---

## 11. 예상 리스크

| 항목 | 리스크 | 대응 |
|------|--------|------|
| BQ 2014 이전 없음 | train 대부분 결측 → 실험 불가 | BQ 범위 먼저 확인 |
| titles 데이터 결측 | GDELT 보도 없는 국가-일 → 0 처리 적절성 | 결측 비율 확인 후 판단 |
| themes/persons 없음 | D 불가 | 후속 GKG 수집 과제로 분리 |
| B 성능이 너무 낮음 | C 개선 의미 약화 | B 결과 확인 후 C 진행 여부 결정 |
| GDELT titles vs events 중복 | v2tone이 gdelt_tone_mean과 상관 높을 수 있음 | Pearson 상관 확인 후 중복 시 조정 |

---

## 12. 구현 순서

```
Step 0: 전제 조건 확인 (코드/실험 전에)
  0-1. BQ gdelt_titles 기간 범위 및 themes/persons 컬럼 확인 (Section 4 SQL)
  0-2. train.parquet 실제 컬럼 목록 확인 (readiness_check.md 참조)
  0-3. ACLED-free feature_cols 목록 확정

Step 1: 실험 B 구현 및 실행
  1-1. run_stacking_acled_free_baseline.py 작성
       (기존 stacking 스크립트에서 SE merge 제거 + ACLED 컬럼 ALWAYS_EXCLUDE 추가)
  1-2. B 실행 → val PR-AUC 확인

Step 2: GDELT titles 피처 생성 모듈 작성
  2-1. src/process/gdelt_titles_feature_builder.py 작성
  2-2. BQ 또는 로컬 parquet에서 집계 → gdelt_titles_features.parquet 저장

Step 3: 실험 C 실행 (Phase 1 순서)
  3-1. C-count, C-tone 단독 실험
  3-2. C-full 종합 실험
  3-3. B vs C delta 보고

Step 4: Phase 2, 3 (C 개선 확인 후)
  4-1. Rolling window 비교
  4-2. D 실험 (themes/persons 있을 경우)
```

---

*이 문서는 실험 계획서입니다. 코드 수정 및 학습 실행은 Step 0 확인 후 진행합니다.*
