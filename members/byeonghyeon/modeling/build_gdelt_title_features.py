"""
GDELT Title Feature Builder — Experiment C
==========================================
BigQuery 테이블 `conflict-early-warning.conflict_ew.gdelt_titles`를 조회해
country × date 단위 GDELT title/tone/count/domain/language 집계 피처를 생성한다.

생성 피처 (총 21개):
  1d 집계 (10개):
    gdelt_title_count_1d           하루 기사 건수
    gdelt_title_nonnull_count_1d   title 비결측 건수
    gdelt_title_tone_mean_1d       평균 톤 점수
    gdelt_title_tone_std_1d        톤 표준편차
    gdelt_title_tone_min_1d        최저 톤
    gdelt_title_negative_count_1d  톤 < -5 기사 수
    gdelt_title_positive_count_1d  톤 > +2 기사 수
    gdelt_title_eng_count_1d       영어 기사 수
    gdelt_title_domain_diversity_1d 고유 도메인 수
    gdelt_title_lang_diversity_1d  고유 언어 수

  7일 rolling (11개):
    gdelt_title_count_7d           7일 기사 수 합계
    gdelt_title_nonnull_count_7d   7일 title 비결측 합계
    gdelt_title_tone_mean_7d       7일 평균 톤
    gdelt_title_tone_std_7d        7일 톤 표준편차 평균
    gdelt_title_tone_min_7d        7일 최저 톤
    gdelt_title_negative_count_7d  7일 부정 기사 수 합계
    gdelt_title_positive_count_7d  7일 긍정 기사 수 합계
    gdelt_title_eng_count_7d       7일 영어 기사 수 합계
    gdelt_title_domain_diversity_7d 7일 도메인 다양성 평균
    gdelt_title_lang_diversity_7d  7일 언어 다양성 평균
    gdelt_title_tone_trend_7d      톤 추세 (tone_mean_1d - 7일 전 tone_mean)

NOTE — coverage_mask 설계:
  이 parquet에는 BQ에서 최소 1건이라도 보도된 (country, date) 행만 포함된다.
  보도가 없는 날 또는 2015-02-17 이전 날짜는 parquet에 행이 없다.
  학습 스크립트에서 train/val/test와 left-join 후:
    - 결측 피처 → 0 채움
    - gdelt_title_coverage_mask = (date < BQ_MIN_DATE) → 학습 스크립트에서 직접 설정
  coverage_mask를 이 parquet에 포함하지 않는 이유:
    BQ `daily` CTE가 데이터 없는 날을 반환하지 않으므로, SQL에서 계산하면 항상 0이 됨.

NOTE — rolling window + missing dates:
  `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW`는 실제 존재하는 행 기준으로 작동한다.
  보도가 전혀 없는 날은 daily CTE에 행이 없으므로 rolling 윈도우에서 제외된다.
  따라서 `count_7d`는 "직전 7개 보도일의 합계"이지 "직전 7 달력일의 합계"가 아니다.
  2015 이후 대부분의 국가-일은 보도가 있으므로 실용적 영향은 제한적이다.

Leakage 방지:
  - 모든 rolling: ROWS BETWEEN 6 PRECEDING AND CURRENT ROW (과거 방향, 당일 포함)
  - LAG(7): 7일 전 값 참조 (과거 방향)
  - 당일 데이터까지 포함 허용 (GDELT 15분 단위 업데이트, 예측 시점 이전 이미 가용)

실행:
  # SQL + 비용 안내 출력 (BQ 쿼리 실행 없음)
  python members/byeonghyeon/modeling/build_gdelt_title_features.py --dry-run-sql

  # 실제 BQ 쿼리 실행 후 parquet 저장
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json \\
    python members/byeonghyeon/modeling/build_gdelt_title_features.py

  # 기간 지정
  python members/byeonghyeon/modeling/build_gdelt_title_features.py \\
    --start 2014-01-01 --end 2025-03-31

  # 출력 경로 지정
  python members/byeonghyeon/modeling/build_gdelt_title_features.py \\
    --output /path/to/gdelt_title_features.parquet
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_SUBTREE_ROOT = _SCRIPT_DIR.parent  # members/byeonghyeon/

# 기본 출력 경로
DEFAULT_OUTPUT = _SUBTREE_ROOT / "input" / "processed" / "gdelt_titles" / "gdelt_title_features.parquet"

# BQ 프로젝트
BQ_PROJECT = "conflict-early-warning"
BQ_TABLE   = f"{BQ_PROJECT}.conflict_ew.gdelt_titles"

# BQ 인증 — 환경변수 GOOGLE_APPLICATION_CREDENTIALS 미설정 시 개인 레포에서 *.json 탐색
# 우선순위: 환경변수 > 개인 레포 루트의 첫 번째 서비스 계정 JSON
_PERSONAL_REPO = _SUBTREE_ROOT.parent.parent.parent / "conflict-early-warning"

def _find_default_cred() -> Path:
    """개인 레포 루트에서 서비스 계정 JSON을 탐색한다. 없으면 빈 Path 반환."""
    if not _PERSONAL_REPO.exists():
        return Path("")
    candidates = sorted(_PERSONAL_REPO.glob("*.json"))
    # package.json, tsconfig.json 등 제외 (service account JSON은 보통 project-id-hash.json 형태)
    sa_candidates = [p for p in candidates if len(p.stem) > 20]
    return sa_candidates[0] if sa_candidates else Path("")

# Coverage gap: BQ 데이터는 2015-02-17부터 시작.
# NOTE: coverage_mask는 이 parquet에 포함하지 않는다.
#       학습 스크립트에서 left-join 후 직접 설정:
#         df["gdelt_title_coverage_mask"] = (df["date"] < pd.Timestamp(BQ_MIN_DATE, tz="UTC")).astype(int)
BQ_MIN_DATE = "2015-02-17"

# 기본 학습 데이터 기간 (train+val+test 전체)
DEFAULT_START = "2014-01-01"
DEFAULT_END   = "2025-03-31"


# ─────────────────────────────────────────────────────────────────────────────
# 공개 상수 (학습 스크립트에서 import 가능)
# ─────────────────────────────────────────────────────────────────────────────

GDELT_TITLE_FEATURE_COLS = [
    # 1d 집계 (10개)
    "gdelt_title_count_1d",
    "gdelt_title_nonnull_count_1d",
    "gdelt_title_tone_mean_1d",
    "gdelt_title_tone_std_1d",
    "gdelt_title_tone_min_1d",
    "gdelt_title_negative_count_1d",
    "gdelt_title_positive_count_1d",
    "gdelt_title_eng_count_1d",
    "gdelt_title_domain_diversity_1d",
    "gdelt_title_lang_diversity_1d",
    # 7일 rolling (11개)
    "gdelt_title_count_7d",
    "gdelt_title_nonnull_count_7d",
    "gdelt_title_tone_mean_7d",
    "gdelt_title_tone_std_7d",
    "gdelt_title_tone_min_7d",
    "gdelt_title_negative_count_7d",
    "gdelt_title_positive_count_7d",
    "gdelt_title_eng_count_7d",
    "gdelt_title_domain_diversity_7d",
    "gdelt_title_lang_diversity_7d",
    "gdelt_title_tone_trend_7d",
]
# coverage_mask는 학습 스크립트에서 직접 설정 (위 목록에 포함하지 않음)
# GDELT_TITLE_COVERAGE_MASK_COL = "gdelt_title_coverage_mask"


# ─────────────────────────────────────────────────────────────────────────────
# SQL 생성
# ─────────────────────────────────────────────────────────────────────────────

def build_sql(start_date: str, end_date: str) -> str:
    """
    country × date 단위 1d 집계 + 7일 rolling 피처 생성 SQL.

    구조:
      1. daily CTE: raw 테이블을 1회 스캔해 country × date 단위로 집계 (GROUP BY)
      2. with_rolling CTE: daily 결과에 window function 적용 (추가 raw 스캔 없음)

    BQ 파티셔닝 참고:
      gdelt_titles 테이블이 date 컬럼으로 파티셔닝된 경우 WHERE date BETWEEN ... 이
      파티션 pruning을 자동으로 적용한다. 파티셔닝 여부는 BQ Console에서 확인 가능.

    Rolling window 주의:
      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW 는 실제 존재하는 행 기준.
      보도 없는 날은 daily CTE에 행이 없으므로 7d rolling이
      "직전 7개 보도일의 값"이지 "직전 7 달력일의 값"이 아닐 수 있다.
    """
    return f"""
-- ============================================================
-- GDELT Title Feature Builder (Experiment C)
-- Raw 테이블 스캔: 1회 (daily CTE에서만)
-- country × date 단위 집계 + 7일 rolling
-- 기간: {start_date} ~ {end_date}
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

  FROM `{BQ_TABLE}`
  WHERE date BETWEEN '{start_date}' AND '{end_date}'
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
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# BQ 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_bq_query(sql: str, output_path: Path) -> None:
    try:
        from google.cloud import bigquery
    except ImportError:
        print("[오류] google-cloud-bigquery 미설치.")
        print("  pip install google-cloud-bigquery pyarrow")
        sys.exit(1)

    # 인증 설정: 환경변수 우선 → 개인 레포 자동 탐색
    cred_path_str = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if cred_path_str:
        cred_path = Path(cred_path_str)
        print(f"  인증 키 (환경변수): {cred_path}")
    else:
        cred_path = _find_default_cred()
        if cred_path and cred_path.exists():
            print(f"  인증 키 (자동 탐색): {cred_path}")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)
        else:
            print("[오류] BQ 인증 키를 찾을 수 없습니다.")
            print("  다음 중 하나로 설정하세요:")
            print("  1. export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json")
            print(f"  2. 서비스 계정 JSON을 {_PERSONAL_REPO}/ 에 위치")
            sys.exit(1)

    if not cred_path.exists():
        print(f"[오류] 인증 키 파일이 없습니다: {cred_path}")
        sys.exit(1)

    import time
    import pandas as pd

    client = bigquery.Client(project=BQ_PROJECT)

    print(f"\n[BQ] 쿼리 실행 중 — {BQ_TABLE}")
    print("  예상 소요 시간: 3~10분")
    t0 = time.time()
    df = client.query(sql).to_dataframe()
    elapsed = time.time() - t0

    print(f"  완료: {len(df):,}행 × {len(df.columns)}컬럼  ({elapsed:.1f}초)")
    print(f"  date range: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  국가 수: {df['country'].nunique()}")

    # tone_trend_7d는 처음 7행에서 LAG 결측 발생 → 0 채움
    n_null = df[GDELT_TITLE_FEATURE_COLS].isnull().sum().sum()
    if n_null > 0:
        print(f"  결측값 {n_null:,}개 → 0 채움 (tone_trend_7d 초기 LAG 등)")
        df[GDELT_TITLE_FEATURE_COLS] = df[GDELT_TITLE_FEATURE_COLS].fillna(0)
    else:
        print("  결측값 없음 ✅")

    # date: BQ date32[day] → UTC-aware datetime64[ns, UTC]
    # train.parquet가 datetime64[ns, UTC]이므로 merge 호환성을 위해 tz를 맞춤
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")

    # Int64 (nullable int) → int64: LightGBM/XGBoost 호환성 및 parquet 재현성
    int64_cols = [c for c in df.columns if str(df[c].dtype) == "Int64"]
    if int64_cols:
        df[int64_cols] = df[int64_cols].astype("int64")

    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n  저장 완료: {output_path}  ({size_mb:.1f} MB)")
    print(f"  총 행수  : {len(df):,}  (BQ 데이터 있는 country×date만 포함)")
    print()
    print("  ── coverage_mask 안내 ─────────────────────────────────────────")
    print(f"  이 parquet에는 {BQ_MIN_DATE} 이전 행이 없습니다.")
    print("  학습 스크립트에서 left-join 후 다음과 같이 coverage_mask를 설정하세요:")
    print(f'    df["gdelt_title_coverage_mask"] = (df["date"] < pd.Timestamp("{BQ_MIN_DATE}", tz="UTC")).astype(int)')
    print("  ────────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# 출력 크기 추정
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_result_rows(start: str, end: str) -> str:
    """BQ 결과 행수 추정. raw 행수(8.6억)가 아닌 집계 결과(국가×일수) 기준."""
    try:
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d")
        total_days = (e - s).days
        bq_start   = datetime.strptime(BQ_MIN_DATE, "%Y-%m-%d")
        bq_days    = max(0, (e - bq_start).days)  # 실제 데이터 있는 일수
        result_rows = bq_days * 58
        raw_rows_est = int(total_days / 365.25 * 7e7)
        return (
            f"결과 행수 약 {result_rows:,} ({bq_days}일 × 58개국) | "
            f"raw 스캔 약 {raw_rows_est:,}행"
        )
    except Exception:
        return "추정 불가"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BigQuery에서 GDELT title 피처를 생성해 parquet으로 저장합니다."
    )
    parser.add_argument(
        "--dry-run-sql", action="store_true",
        help="SQL과 비용 확인 안내를 출력하고 BQ 쿼리는 실행하지 않습니다."
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"출력 parquet 경로 (기본: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--start", type=str, default=DEFAULT_START,
        help=f"집계 시작일 YYYY-MM-DD (기본: {DEFAULT_START})"
    )
    parser.add_argument(
        "--end", type=str, default=DEFAULT_END,
        help=f"집계 종료일 YYYY-MM-DD (기본: {DEFAULT_END})"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    sql         = build_sql(args.start, args.end)
    row_est     = _estimate_result_rows(args.start, args.end)

    print("=" * 70)
    print("  GDELT Title Feature Builder — Experiment C")
    print(f"  BQ 테이블 : {BQ_TABLE}")
    print(f"  기간      : {args.start} ~ {args.end}")
    print(f"  행수 추정 : {row_est}")
    print(f"  출력      : {output_path}")
    print(f"  피처 수   : {len(GDELT_TITLE_FEATURE_COLS)}개 (+ date, country 키)")
    print(f"  raw 스캔  : 1회 (daily CTE에서만, window function은 집계 결과만 읽음)")
    print("=" * 70)

    if args.dry_run_sql:
        print("\n[--dry-run-sql] SQL을 출력합니다. BQ 쿼리는 실행되지 않습니다.\n")
        print("-" * 70)
        print(sql)
        print("-" * 70)
        print(f"\n생성될 피처 목록 ({len(GDELT_TITLE_FEATURE_COLS)}개):")
        for col in GDELT_TITLE_FEATURE_COLS:
            print(f"  {col}")
        print()
        print("─" * 70)
        print("  BQ 실행 전 비용 확인 방법:")
        print()
        print("  1. BQ Console (https://console.cloud.google.com/bigquery)")
        print(f"     → 프로젝트: {BQ_PROJECT}")
        print("     → 위 SQL을 붙여넣고 [실행] 버튼 오른쪽 삼각형 아이콘 클릭")
        print("     → 'Dry Run' 선택 → 처리 바이트 수 확인")
        print()
        print("  2. bq CLI:")
        print(f"     bq query --dry_run --use_legacy_sql=false '<위 SQL>'")
        print()
        print("  BQ 요금: 온디맨드 기준 $5 / TB (2024년 기준)")
        print(f"  gdelt_titles 전체({BQ_MIN_DATE}~) 약 860억 행 → 추정 수십~수백 GB 스캔")
        print("─" * 70)
        return

    print(f"\n[주의] BQ 쿼리를 실행합니다. 요금이 발생할 수 있습니다.")
    print(f"  BQ 인증 환경변수 GOOGLE_APPLICATION_CREDENTIALS 를 먼저 설정하세요.")
    print()

    run_bq_query(sql, output_path)

    print()
    print("=" * 70)
    print("  완료. 다음 단계:")
    print(f"  1. parquet 확인  : {output_path}")
    print("  2. coverage_mask : 학습 스크립트에서 left-join 후 직접 설정")
    print("  3. 실험 C 학습   : run_stacking_acled_free_with_titles.py 실행")
    print("=" * 70)


if __name__ == "__main__":
    main()
