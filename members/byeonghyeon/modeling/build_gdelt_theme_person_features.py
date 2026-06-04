"""
GDELT Theme & Person Feature Builder — Experiment D
====================================================
BigQuery 테이블 `conflict-early-warning.conflict_ew.gdelt_titles`의
v2themes, v2persons 컬럼을 활용해 country × date 단위 테마/인물 집계 피처를 생성한다.

v2themes / v2persons 포맷:
  v2themes  예: "CEASEFIRE,65;CONFLICT,146;BLOCKADE,4133;LEADER,14"
  v2persons 예: "Donald Trump,858;Ali Khamenei,1476;Masoud Pezeshkian,6144"
  → 세미콜론(;) 구분 / 각 토큰은 "이름_또는_테마,문자위치" 형태
  → 숫자(문자위치)는 count가 아닌 offset — 사용하지 않음
  → 이름/테마 문자열 자체는 저장하지 않음; 집계 count만 저장

생성 피처 (총 22개):
  1d 집계 (11개):
    gdelt_theme_nonnull_count_1d   v2themes 비결측 기사 수
    gdelt_person_nonnull_count_1d  v2persons 비결측 기사 수
    gdelt_theme_count_1d           전체 theme 토큰 수 합계
    gdelt_person_count_1d          전체 person 토큰 수 합계
    gdelt_theme_conflict_count_1d  충돌/분쟁 테마 포함 기사 수
    gdelt_theme_protest_count_1d   시위/폭동 테마 포함 기사 수
    gdelt_theme_military_count_1d  군사 테마 포함 기사 수
    gdelt_theme_refugee_count_1d   난민/이재민 테마 포함 기사 수
    gdelt_theme_sanction_count_1d  제재/금수 테마 포함 기사 수
    gdelt_theme_government_count_1d 정치/정부 테마 포함 기사 수
    gdelt_person_density_1d        기사 1건당 평균 인물 언급 수

  7일 rolling (11개):
    위 11개의 7일 rolling sum (밀도 피처는 rolling mean)
    gdelt_theme_nonnull_count_7d, gdelt_person_nonnull_count_7d,
    gdelt_theme_count_7d, gdelt_person_count_7d,
    gdelt_theme_conflict_count_7d, gdelt_theme_protest_count_7d,
    gdelt_theme_military_count_7d, gdelt_theme_refugee_count_7d,
    gdelt_theme_sanction_count_7d, gdelt_theme_government_count_7d,
    gdelt_person_density_7d

테마 키워드 그룹 (GDELT GKG 테마명 기준):
  conflict    : CONFLICT | MILITARY_ATTACK | ARMED | BATTLE | WAR
  protest     : PROTEST | RIOT | CIVIL_UNREST | STRIKE_ACTION | DEMONSTRATION
  military    : (?:^|;)MILITARY  (군사 관련 전체)
  refugee     : REFUGEE | DISPLACED | ASYLUM | HUMANITARIAN
  sanction    : SANCTION | EMBARGO
  government  : GOV_ | ELECTION | COUP | CEASEFIRE | BLOCKADE | SEIGE

  키워드는 세미콜론 토큰의 시작에서 매칭 ((?:^|;)KEYWORD)하여
  부분 매칭 오탐을 최소화한다.
  단, 일부 테마(GOV_, MILITARY 등)는 prefix 매칭을 의도함.

NOTE — coverage_mask:
  이 parquet에는 BQ 데이터 없는 날(2015-02-17 이전 등)의 행이 없다.
  학습 스크립트에서 left-join 후 결측 → 0 채움.
  gdelt_title_coverage_mask는 C 스크립트에서 이미 설정되어 있으므로
  이 parquet에서 별도로 관리하지 않는다.

NOTE — rolling window:
  ROWS BETWEEN 6 PRECEDING AND CURRENT ROW는 실제 행 기준.
  보도 없는 날은 daily CTE에서 제외되므로 7d rolling이
  "직전 7 달력일" 합계가 아닌 "직전 7 보도일" 합계일 수 있다.

Leakage 방지:
  - ROWS BETWEEN 6 PRECEDING AND CURRENT ROW (과거 방향, 당일 포함)
  - 당일 GDELT 데이터 사용 허용 (15분 단위 업데이트)

실행:
  # SQL + 비용 안내 출력 (BQ 실행 없음)
  python members/byeonghyeon/modeling/build_gdelt_theme_person_features.py --dry-run-sql

  # SQL을 파일로 저장
  python members/byeonghyeon/modeling/build_gdelt_theme_person_features.py --save-sql

  # 실제 BQ 쿼리 실행
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json \\
    python members/byeonghyeon/modeling/build_gdelt_theme_person_features.py

  # 기간 지정
  python members/byeonghyeon/modeling/build_gdelt_theme_person_features.py \\
    --start 2014-01-01 --end 2025-03-31
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_SUBTREE_ROOT = _SCRIPT_DIR.parent  # members/byeonghyeon/

DEFAULT_OUTPUT = (_SUBTREE_ROOT / "input" / "processed" / "gdelt_titles"
                  / "gdelt_theme_person_features.parquet")
SQL_SAVE_PATH  = _SCRIPT_DIR / "gdelt_theme_person_features_query.sql"

BQ_PROJECT = "conflict-early-warning"
BQ_TABLE   = f"{BQ_PROJECT}.conflict_ew.gdelt_titles"

_PERSONAL_REPO = _SUBTREE_ROOT.parent.parent.parent / "conflict-early-warning"

def _find_default_cred() -> Path:
    if not _PERSONAL_REPO.exists():
        return Path("")
    candidates = sorted(_PERSONAL_REPO.glob("*.json"))
    sa = [p for p in candidates if len(p.stem) > 20]
    return sa[0] if sa else Path("")

BQ_MIN_DATE   = "2015-02-17"
DEFAULT_START = "2014-01-01"
DEFAULT_END   = "2025-03-31"


# ─────────────────────────────────────────────────────────────────────────────
# 공개 상수 (학습 스크립트에서 import 가능)
# ─────────────────────────────────────────────────────────────────────────────

GDELT_THEME_PERSON_FEATURE_COLS = [
    # 1d 집계 (11개)
    "gdelt_theme_nonnull_count_1d",
    "gdelt_person_nonnull_count_1d",
    "gdelt_theme_count_1d",
    "gdelt_person_count_1d",
    "gdelt_theme_conflict_count_1d",
    "gdelt_theme_protest_count_1d",
    "gdelt_theme_military_count_1d",
    "gdelt_theme_refugee_count_1d",
    "gdelt_theme_sanction_count_1d",
    "gdelt_theme_government_count_1d",
    "gdelt_person_density_1d",
    # 7일 rolling (11개)
    "gdelt_theme_nonnull_count_7d",
    "gdelt_person_nonnull_count_7d",
    "gdelt_theme_count_7d",
    "gdelt_person_count_7d",
    "gdelt_theme_conflict_count_7d",
    "gdelt_theme_protest_count_7d",
    "gdelt_theme_military_count_7d",
    "gdelt_theme_refugee_count_7d",
    "gdelt_theme_sanction_count_7d",
    "gdelt_theme_government_count_7d",
    "gdelt_person_density_7d",
]


# ─────────────────────────────────────────────────────────────────────────────
# 테마 키워드 정의
# ─────────────────────────────────────────────────────────────────────────────
# v2themes 포맷: "THEME,offset;THEME,offset;..."
# (?:^|;)KEYWORD 패턴으로 세미콜론 토큰 시작에서 매칭 → 부분 오탐 최소화
# GDELT GKG 테마명은 대문자+밑줄 형식이므로 대소문자 구분 가능

THEME_PATTERNS = {
    # 충돌·분쟁: CONFLICT, MILITARY_ATTACK, ARMED_CONFLICT, BATTLE, WAR 등
    "conflict":    r"(?:^|;)(?:CONFLICT|MILITARY_ATTACK|ARMED|BATTLE|WAR)",
    # 시위·폭동: PROTEST, RIOT, CIVIL_UNREST, STRIKE, DEMONSTRATION 등
    "protest":     r"(?:^|;)(?:PROTEST|RIOT|CIVIL_UNREST|STRIKE_ACTION|DEMONSTRATION)",
    # 군사: MILITARY 로 시작하는 모든 테마 (MILITARY_DEPLOYMENT, MILITARY_BUILDUP 등 포함)
    "military":    r"(?:^|;)MILITARY",
    # 난민·인도주의: REFUGEE, DISPLACED, ASYLUM, HUMANITARIAN 등
    "refugee":     r"(?:^|;)(?:REFUGEE|DISPLACED|ASYLUM|HUMANITARIAN)",
    # 제재·금수: SANCTION, EMBARGO, ECON_SANCTION, UN_SANCTION 등
    "sanction":    r"(?:^|;)(?:SANCTION|EMBARGO)",
    # 정치·정부: GOV_ 계열, ELECTION, COUP, CEASEFIRE, BLOCKADE, SEIGE 등
    "government":  r"(?:^|;)(?:GOV_|ELECTION|COUP|CEASEFIRE|BLOCKADE|SEIGE)",
}


# ─────────────────────────────────────────────────────────────────────────────
# SQL 생성
# ─────────────────────────────────────────────────────────────────────────────

def build_sql(start_date: str, end_date: str) -> str:
    """
    v2themes / v2persons 기반 country × date 집계 + 7일 rolling SQL.

    v2themes 처리:
      - ARRAY_LENGTH(SPLIT(v2themes, ';')): 전체 토큰 수 (THEME,offset 쌍)
      - COUNTIF(REGEXP_CONTAINS(v2themes, pattern)): 특정 테마 포함 기사 수
      - (?:^|;)KEYWORD 패턴: 토큰 시작 매칭 (부분 오탐 최소화)

    v2persons 처리:
      - ARRAY_LENGTH(SPLIT(v2persons, ';')): 전체 인물 토큰 수
      - AVG(...): 기사당 평균 인물 언급 수 (person density)
      - 인물 이름 문자열은 저장하지 않음

    Raw 테이블 스캔: 1회 (daily CTE에서만)
    Rolling: ROWS BETWEEN 6 PRECEDING AND CURRENT ROW (과거 방향, 당일 포함)
    """
    # 키워드 COUNTIF 조각 생성 (각 행마다 쉼표 포함)
    def theme_countif(key):
        return (f"    COUNTIF(REGEXP_CONTAINS(IFNULL(v2themes, ''), r'{THEME_PATTERNS[key]}'))"
                f"  AS gdelt_theme_{key}_count_1d")

    # 각 COUNTIF 행 사이에 쉼표, 마지막 행에도 쉼표 (다음 AVG 앞)
    theme_countifs = ",\n".join(theme_countif(k) for k in THEME_PATTERNS) + ","

    return f"""-- ============================================================
-- GDELT Theme & Person Feature Builder (Experiment D)
-- Raw 테이블 스캔: 1회 (daily CTE에서만)
-- v2themes / v2persons 기반 country × date 집계 + 7일 rolling
-- 기간: {start_date} ~ {end_date}
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
{theme_countifs}

    -- 인물 밀도: 기사당 평균 인물 토큰 수 (v2persons 있는 기사만)
    AVG(CASE WHEN v2persons IS NOT NULL
             THEN ARRAY_LENGTH(SPLIT(v2persons, ';'))
             ELSE NULL END)                       AS gdelt_person_density_1d

  FROM `{BQ_TABLE}`
  WHERE date BETWEEN '{start_date}' AND '{end_date}'
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
ORDER BY country, date""".strip()


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

    import time
    import pandas as pd

    # 인증 설정
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
            print("  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json")
            sys.exit(1)

    if not cred_path.exists():
        print(f"[오류] 인증 키 파일이 없습니다: {cred_path}")
        sys.exit(1)

    client = bigquery.Client(project=BQ_PROJECT)

    print(f"\n[BQ] 쿼리 실행 중 — {BQ_TABLE}")
    print("  REGEXP_CONTAINS + ARRAY_LENGTH 집계로 C보다 다소 시간이 걸릴 수 있습니다.")
    print("  예상 소요 시간: 5~15분")

    t0 = time.time()
    df = client.query(sql).to_dataframe()
    elapsed = time.time() - t0

    print(f"  완료: {len(df):,}행 × {len(df.columns)}컬럼  ({elapsed:.1f}초)")
    print(f"  date range: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  국가 수: {df['country'].nunique()}")

    # 날짜 → UTC-aware
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")

    # 결측 처리
    feat_cols = GDELT_THEME_PERSON_FEATURE_COLS
    n_null = df[feat_cols].isnull().sum().sum()
    if n_null > 0:
        print(f"  결측값 {n_null:,}개 → 0 채움 (초기 rolling LAG 등)")
        df[feat_cols] = df[feat_cols].fillna(0)
    else:
        print("  결측값 없음 ✅")

    # Int64 nullable → int64 / float64 정규화
    for col in df.columns:
        if str(df[col].dtype) == "Int64":
            df[col] = df[col].astype("int64")

    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n  저장 완료: {output_path}  ({size_mb:.1f} MB)")
    print(f"  총 행수  : {len(df):,}  (BQ 데이터 있는 country×date만 포함)")
    print()
    print("  ── coverage_mask 안내 ─────────────────────────────────────────")
    print(f"  이 parquet에는 {BQ_MIN_DATE} 이전 행이 없습니다.")
    print("  학습 스크립트에서 gdelt_title_coverage_mask 재사용 또는")
    print(f'  df["gdelt_title_coverage_mask"] = (df["date"] < pd.Timestamp("{BQ_MIN_DATE}", tz="UTC")).astype(int)')
    print("  ────────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# 예상 크기/비용 추정
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_info(start: str, end: str) -> str:
    try:
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d")
        total_days = (e - s).days
        bq_start   = datetime.strptime(BQ_MIN_DATE, "%Y-%m-%d")
        bq_days    = max(0, (e - bq_start).days)
        result_rows = bq_days * 58
        raw_est     = int(total_days / 365.25 * 7e7)
        return (f"결과 행수 약 {result_rows:,} ({bq_days}일 × 58개국) | "
                f"raw 스캔 약 {raw_est:,}행")
    except Exception:
        return "추정 불가"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BigQuery v2themes/v2persons 집계 피처를 생성해 parquet로 저장합니다."
    )
    parser.add_argument("--dry-run-sql", action="store_true",
                        help="SQL + 비용 안내를 출력하고 BQ 쿼리는 실행하지 않습니다.")
    parser.add_argument("--save-sql", action="store_true",
                        help=f"SQL을 파일로 저장합니다 ({SQL_SAVE_PATH}). BQ 쿼리는 실행하지 않습니다.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"출력 parquet 경로 (기본: {DEFAULT_OUTPUT})")
    parser.add_argument("--start", type=str, default=DEFAULT_START,
                        help=f"집계 시작일 YYYY-MM-DD (기본: {DEFAULT_START})")
    parser.add_argument("--end", type=str, default=DEFAULT_END,
                        help=f"집계 종료일 YYYY-MM-DD (기본: {DEFAULT_END})")
    args = parser.parse_args()

    output_path = Path(args.output)
    sql         = build_sql(args.start, args.end)
    info        = _estimate_info(args.start, args.end)

    print("=" * 70)
    print("  GDELT Theme & Person Feature Builder — Experiment D")
    print(f"  BQ 테이블 : {BQ_TABLE}")
    print(f"  기간      : {args.start} ~ {args.end}")
    print(f"  행수 추정 : {info}")
    print(f"  출력      : {output_path}")
    print(f"  피처 수   : {len(GDELT_THEME_PERSON_FEATURE_COLS)}개 (+ date, country 키)")
    print(f"  raw 스캔  : 1회 (daily CTE에서만)")
    print("=" * 70)

    # SQL 파일 저장
    if args.save_sql:
        header = (
            f"-- GDELT Theme & Person Feature Builder (Experiment D)\n"
            f"-- 기간: {args.start} ~ {args.end}\n"
            f"-- 생성일: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"--\n"
            f"-- BigQuery Dry Run으로 비용 확인 방법:\n"
            f"--   1. BQ Console: https://console.cloud.google.com/bigquery\n"
            f"--      → 프로젝트: {BQ_PROJECT}\n"
            f"--      → 이 SQL 붙여넣기 → 실행 버튼 드롭다운 → Dry Run\n"
            f"--   2. Python client: dry_run=True\n"
            f"-- BQ 요금: 온디맨드 $5/TB\n"
            f"-- 주의: v2themes REGEXP_CONTAINS 연산으로 C 쿼리보다 다소 비쌀 수 있음\n"
            f"--\n\n"
        )
        SQL_SAVE_PATH.write_text(header + sql, encoding="utf-8")
        print(f"\n  SQL 파일 저장: {SQL_SAVE_PATH}")
        return

    if args.dry_run_sql:
        print("\n[--dry-run-sql] SQL을 출력합니다. BQ 쿼리는 실행되지 않습니다.\n")
        print("-" * 70)
        print(sql)
        print("-" * 70)
        print(f"\n생성될 피처 목록 ({len(GDELT_THEME_PERSON_FEATURE_COLS)}개):")
        for col in GDELT_THEME_PERSON_FEATURE_COLS:
            print(f"  {col}")
        print()
        print("─" * 70)
        print("  테마 키워드 정의:")
        for key, pattern in THEME_PATTERNS.items():
            print(f"    {key:<12} : {pattern}")
        print()
        print("  BQ 실행 전 비용 확인 방법:")
        print()
        print("  1. BQ Console (https://console.cloud.google.com/bigquery)")
        print(f"     → 프로젝트: {BQ_PROJECT}")
        print("     → 위 SQL을 붙여넣고 [실행] 버튼 드롭다운 → 'Dry Run'")
        print()
        print("  2. Python (추천):")
        print("     from google.cloud import bigquery")
        print(f"     client = bigquery.Client(project='{BQ_PROJECT}')")
        print("     job = client.query(SQL, job_config=bigquery.QueryJobConfig(dry_run=True))")
        print("     print(f'{job.total_bytes_processed / 1e9:.2f} GB')")
        print()
        print("  BQ 요금: 온디맨드 $5/TB")
        print("  C 쿼리(~$0.40) 대비 REGEXP_CONTAINS 추가로 다소 비쌀 수 있음")
        print("─" * 70)
        return

    print(f"\n[주의] BQ 쿼리를 실행합니다. 요금이 발생할 수 있습니다.")
    print(f"  GOOGLE_APPLICATION_CREDENTIALS 환경변수를 먼저 설정하세요.")
    print()

    run_bq_query(sql, output_path)

    print()
    print("=" * 70)
    print("  완료. 다음 단계:")
    print(f"  1. parquet 확인    : {output_path}")
    print("  2. 실험 D 학습     : run_stacking_acled_free_with_titles_and_themes.py 실행")
    print("=" * 70)


if __name__ == "__main__":
    main()
