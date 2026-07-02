-- pipeline_watermarks: GDELT 계열 15분 증분 수집 watermark 테이블
--
-- 적용 방법:
--   PROJECT_ID=conflict-ew-mvp-20260604
--   bq query --use_legacy_sql=false --project_id=$PROJECT_ID \
--     "$(sed "s/{PROJECT_ID}/$PROJECT_ID/g" migrations/create_pipeline_watermarks.sql)"
--
-- 또는 bq mk로 직접 생성:
--   bq mk --table \
--     $PROJECT_ID:conflict_ew.pipeline_watermarks \
--     source:STRING,last_success_at:TIMESTAMP,updated_at:TIMESTAMP,run_id:STRING
--
-- 적용 대상 source:
--   gdelt        — DATEADDED 기준 15분 증분 watermark
--   gdelt_titles — GKG DATE 기준 15분 증분 watermark
--
-- economic은 날짜 단위 수집이므로 이 테이블을 사용하지 않음.
-- BQ MAX(date) 기반 state.py compute_collection_window 로직 유지.

CREATE TABLE IF NOT EXISTS `{PROJECT_ID}.conflict_ew.pipeline_watermarks` (
  source        STRING    NOT NULL
    OPTIONS(description='수집 소스: gdelt, gdelt_titles'),
  last_success_at TIMESTAMP NOT NULL
    OPTIONS(description='마지막 성공 수집의 end_timestamp (UTC). 다음 실행의 start 기준.'),
  updated_at    TIMESTAMP NOT NULL
    OPTIONS(description='이 행이 마지막으로 갱신된 UTC 시각'),
  run_id        STRING
    OPTIONS(description='마지막 성공 수집의 run_id (GitHub Actions run_id 또는 uuid)')
)
OPTIONS(
  description='GDELT 계열 15분 증분 수집 watermark. 실패 시 갱신하지 않음.',
  labels=[("managed_by", "pipeline"), ("env", "prod")]
);
