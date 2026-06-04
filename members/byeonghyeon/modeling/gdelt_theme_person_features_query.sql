-- ============================================================
-- GDELT Theme & Person Feature Builder (Experiment D)
-- 기간: 2014-01-01 ~ 2025-03-31
-- 생성일: 2026-06-03
--
-- BigQuery Dry Run 결과 (2026-06-03 측정):
--   처리 예상 바이트: 1,376,773,987,919 bytes
--   처리 예상 크기 : 1282.22 GB  (1.2522 TB)
--   예상 비용 (온디맨드 $5/TB): ~$6.26
--
-- C title feature 쿼리 대비: 1282 GB / 83 GB = ~15.5x
-- 비용 차이 이유: v2themes, v2persons가 긴 문자열 컬럼이라 스캔 비용 큼
--
-- 기간 축소(2015-02-17~) 효과 확인:
--   2014-01-01 ~ 2025-03-31: 1,282.22 GB  ($6.26)
--   2015-02-17 ~ 2025-03-31: 1,282.22 GB  ($6.26)  ← 동일
--   → 차이 없음 (2015-02-17 이전 행이 BQ에 없어 파티션 자동 skip)
--   → 기간 축소로 비용 절감 불가. 현재 쿼리가 이미 최소 비용.
--
-- BigQuery Console에서 비용 재확인 방법:
--   1. https://console.cloud.google.com/bigquery (프로젝트: conflict-early-warning)
--   2. 이 SQL 붙여넣기 -> 실행 버튼 드롭다운 -> Dry Run
-- ============================================================

-- ============================================================
-- GDELT Theme & Person Feature Builder (Experiment D)
-- Raw 테이블 스캔: 1회 (daily CTE에서만)
-- v2themes / v2persons 기반 country × date 집계 + 7일 rolling
-- 기간: 2014-01-01 ~ 2025-03-31
-- ============================================================

WITH daily AS (
  -- ① raw 테이블 1회 스캔 + country × date 집계
  SELECT
    date,
    iso3                                          AS country,

    -- 커버리지
    COUNTIF(v2themes IS NOT NULL)                 AS gdelt_theme_nonnull_count_1d,
    COUNTIF(v2persons IS NOT NULL)                AS gdelt_person_nonnull_count_1d,

    -- 토큰 수: ARRAY_LENGTH(SPLIT(col, ';')) — offset 숫자 미포함, 테마/인물 이름 count만
    SUM(CASE WHEN v2themes  IS NOT NULL
             THEN ARRAY_LENGTH(SPLIT(v2themes,  ';')) ELSE 0 END) AS gdelt_theme_count_1d,
    SUM(CASE WHEN v2persons IS NOT NULL
             THEN ARRAY_LENGTH(SPLIT(v2persons, ';')) ELSE 0 END) AS gdelt_person_count_1d,

    -- 테마 키워드 그룹별 기사 수
    -- 패턴: (?:^|;)KEYWORD — 세미콜론 토큰 시작에서 매칭
    COUNTIF(REGEXP_CONTAINS(IFNULL(v2themes, ''), r'(?:^|;)(?:CONFLICT|MILITARY_ATTACK|ARMED|BATTLE|WAR)'))  AS gdelt_theme_conflict_count_1d,
    COUNTIF(REGEXP_CONTAINS(IFNULL(v2themes, ''), r'(?:^|;)(?:PROTEST|RIOT|CIVIL_UNREST|STRIKE_ACTION|DEMONSTRATION)'))  AS gdelt_theme_protest_count_1d,
    COUNTIF(REGEXP_CONTAINS(IFNULL(v2themes, ''), r'(?:^|;)MILITARY'))  AS gdelt_theme_military_count_1d,
    COUNTIF(REGEXP_CONTAINS(IFNULL(v2themes, ''), r'(?:^|;)(?:REFUGEE|DISPLACED|ASYLUM|HUMANITARIAN)'))  AS gdelt_theme_refugee_count_1d,
    COUNTIF(REGEXP_CONTAINS(IFNULL(v2themes, ''), r'(?:^|;)(?:SANCTION|EMBARGO)'))  AS gdelt_theme_sanction_count_1d,
    COUNTIF(REGEXP_CONTAINS(IFNULL(v2themes, ''), r'(?:^|;)(?:GOV_|ELECTION|COUP|CEASEFIRE|BLOCKADE|SEIGE)'))  AS gdelt_theme_government_count_1d,

    -- 인물 밀도: 기사당 평균 인물 토큰 수 (v2persons 있는 기사만)
    AVG(CASE WHEN v2persons IS NOT NULL
             THEN ARRAY_LENGTH(SPLIT(v2persons, ';'))
             ELSE NULL END)                       AS gdelt_person_density_1d

  FROM `conflict-early-warning.conflict_ew.gdelt_titles`
  WHERE date BETWEEN '2014-01-01' AND '2025-03-31'
  GROUP BY date, iso3
),

with_rolling AS (
  -- ② daily 집계 결과에만 window function 적용 (raw 테이블 재스캔 없음)
  -- ROWS BETWEEN 6 PRECEDING AND CURRENT ROW: 과거 방향, 당일 포함 (최대 7행)
  SELECT
    date,
    country,

    -- 1d 피처 (pass-through)
    gdelt_theme_nonnull_count_1d,
    gdelt_person_nonnull_count_1d,
    gdelt_theme_count_1d,
    gdelt_person_count_1d,
    gdelt_theme_conflict_count_1d,
    gdelt_theme_protest_count_1d,
    gdelt_theme_military_count_1d,
    gdelt_theme_refugee_count_1d,
    gdelt_theme_sanction_count_1d,
    gdelt_theme_government_count_1d,
    gdelt_person_density_1d,

    -- 7일 rolling sum (count 계열)
    SUM(gdelt_theme_nonnull_count_1d)   OVER w7   AS gdelt_theme_nonnull_count_7d,
    SUM(gdelt_person_nonnull_count_1d)  OVER w7   AS gdelt_person_nonnull_count_7d,
    SUM(gdelt_theme_count_1d)           OVER w7   AS gdelt_theme_count_7d,
    SUM(gdelt_person_count_1d)          OVER w7   AS gdelt_person_count_7d,
    SUM(gdelt_theme_conflict_count_1d)  OVER w7   AS gdelt_theme_conflict_count_7d,
    SUM(gdelt_theme_protest_count_1d)   OVER w7   AS gdelt_theme_protest_count_7d,
    SUM(gdelt_theme_military_count_1d)  OVER w7   AS gdelt_theme_military_count_7d,
    SUM(gdelt_theme_refugee_count_1d)   OVER w7   AS gdelt_theme_refugee_count_7d,
    SUM(gdelt_theme_sanction_count_1d)  OVER w7   AS gdelt_theme_sanction_count_7d,
    SUM(gdelt_theme_government_count_1d) OVER w7  AS gdelt_theme_government_count_7d,

    -- 7일 rolling mean (밀도 계열)
    AVG(gdelt_person_density_1d)        OVER w7   AS gdelt_person_density_7d

  FROM daily
  WINDOW w7 AS (
    PARTITION BY country
    ORDER BY date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  )
)

SELECT *
FROM with_rolling
ORDER BY country, date
