-- ============================================================
-- GDELT Title Feature Builder (Experiment C)
-- Raw 테이블 스캔: 1회 (daily CTE에서만)
-- country × date 단위 집계 + 7일 rolling
-- 기간: 2014-01-01 ~ 2025-03-31
--
-- BigQuery Dry Run 결과 (2026-06-03 측정):
--   처리 예상 바이트: 88,905,534,988 bytes (~82.80 GB / ~0.081 TB)
--   예상 비용 (온디맨드 $5/TB): ~$0.40
--   무료 한도: 1TB/월 — 이 쿼리는 한도 미초과
--
-- BigQuery Console에서 비용 재확인 방법:
--   1. https://console.cloud.google.com/bigquery (프로젝트: conflict-early-warning)
--   2. 이 SQL 붙여넣기
--   3. 실행 버튼 옆 드롭다운 → "Dry Run" 클릭
--   4. 우측 상단 처리 예상 바이트 확인
-- ============================================================

WITH daily AS (
  -- ① raw 테이블 1회 스캔 + country × date 집계
  SELECT
    date,
    iso3                                         AS country,

    -- 보도량
    COUNT(*)                                     AS gdelt_title_count_1d,
    COUNTIF(title IS NOT NULL)                  AS gdelt_title_nonnull_count_1d,

    -- 톤 집계
    AVG(v2tone_avg)                              AS gdelt_title_tone_mean_1d,
    STDDEV(v2tone_avg)                           AS gdelt_title_tone_std_1d,
    MIN(v2tone_avg)                              AS gdelt_title_tone_min_1d,
    COUNTIF(v2tone_avg < -5)                     AS gdelt_title_negative_count_1d,
    COUNTIF(v2tone_avg > 2)                      AS gdelt_title_positive_count_1d,

    -- 언어/출처 다양성
    COUNTIF(language = 'eng')                   AS gdelt_title_eng_count_1d,
    COUNT(DISTINCT domain)                       AS gdelt_title_domain_diversity_1d,
    COUNT(DISTINCT language)                     AS gdelt_title_lang_diversity_1d

  FROM `conflict-early-warning.conflict_ew.gdelt_titles`
  WHERE date BETWEEN '2014-01-01' AND '2025-03-31'
  GROUP BY date, iso3
),

with_rolling AS (
  -- ② daily 집계 결과에만 window function 적용 (raw 테이블 재스캔 없음)
  -- ROWS BETWEEN 6 PRECEDING AND CURRENT ROW: 과거 방향, 당일 포함 (최대 7행)
  -- LAG(7): 7일 전 행 참조 (과거 방향)
  SELECT
    date,
    country,

    -- 1d 피처 (그대로 pass-through)
    gdelt_title_count_1d,
    gdelt_title_nonnull_count_1d,
    gdelt_title_tone_mean_1d,
    gdelt_title_tone_std_1d,
    gdelt_title_tone_min_1d,
    gdelt_title_negative_count_1d,
    gdelt_title_positive_count_1d,
    gdelt_title_eng_count_1d,
    gdelt_title_domain_diversity_1d,
    gdelt_title_lang_diversity_1d,

    -- 7일 rolling sum (보도량 계열)
    SUM(gdelt_title_count_1d)           OVER w7  AS gdelt_title_count_7d,
    SUM(gdelt_title_nonnull_count_1d)   OVER w7  AS gdelt_title_nonnull_count_7d,
    SUM(gdelt_title_negative_count_1d)  OVER w7  AS gdelt_title_negative_count_7d,
    SUM(gdelt_title_positive_count_1d)  OVER w7  AS gdelt_title_positive_count_7d,
    SUM(gdelt_title_eng_count_1d)       OVER w7  AS gdelt_title_eng_count_7d,

    -- 7일 rolling mean / min (톤/다양성 계열)
    AVG(gdelt_title_tone_mean_1d)       OVER w7  AS gdelt_title_tone_mean_7d,
    AVG(gdelt_title_tone_std_1d)        OVER w7  AS gdelt_title_tone_std_7d,
    MIN(gdelt_title_tone_min_1d)        OVER w7  AS gdelt_title_tone_min_7d,
    AVG(gdelt_title_domain_diversity_1d) OVER w7 AS gdelt_title_domain_diversity_7d,
    AVG(gdelt_title_lang_diversity_1d)  OVER w7  AS gdelt_title_lang_diversity_7d,

    -- 7일 톤 추세: 당일 평균 − 7일 전 평균 (NULL if < 7 rows exist)
    gdelt_title_tone_mean_1d
      - LAG(gdelt_title_tone_mean_1d, 7) OVER (
          PARTITION BY country ORDER BY date
        )                                        AS gdelt_title_tone_trend_7d

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
