"""
GKG 피처 빌더 (Track 1, 통계 기반): BigQuery 집계 → parquet 적재

입력 (BigQuery)
  conflict-early-warning.conflict_ew.gdelt_titles  (8.6B 행, 2015-02-17~2026-05-29)
출력
  input/processed/features/gkg_features.parquet  (국가x일자 grid, 17컬럼)

설계: docs/feature-engineering.md §5 참조.
  Group A (4): gkg_article_count_1d/7d, gkg_unique_domain_count_7d, gkg_language_diversity_7d
  Group B (11): V2Themes prefix 7개 (kill/protest/terror/military/refugee/armedconflict 7d + armedconflict 1d)
                V2Tone p10 7d & 1d, V2Persons unique_count_7d & top1_share_7d
  메타 (2): gkg_missing_mask, page_title_available_flag

누수 차단
  모든 윈도우 ROWS BETWEEN N PRECEDING AND 1 PRECEDING (당일 t 제외).
  일자 grid는 GENERATE_DATE_ARRAY 로 강제 채움 (빠진 날 = mask=1, 피처 0).

CLI
  python -m src.process.gkg_feature_builder --dry-run         # 비용 추정만
  python -m src.process.gkg_feature_builder                   # 실행 후 parquet 저장
  python -m src.process.gkg_feature_builder --output gkg_features_v2.parquet \\
       --themes KILL PROTEST TERROR                           # ablation variant 집계
"""

from __future__ import annotations

import os

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from src.collect.config import COUNTRIES, GCP_PROJECT, GDELT_TITLES_FQN

PROCESSED_FEATURES_DIR = Path("input/processed/features")
TARGET_TABLE = GDELT_TITLES_FQN

# 기본 집계 범위 (gdelt_titles 가용 구간 전체)
DEFAULT_START = date.fromisoformat(os.environ['SERVE_START']) if os.environ.get('SERVE_START') else date(2015, 2, 17)
DEFAULT_END = date.fromisoformat(os.environ['SERVE_END']) if os.environ.get('SERVE_END') else date(2026, 5, 29)

# Group B V2Themes prefix 6개 (기본)
DEFAULT_THEME_PREFIXES = ["KILL", "PROTEST", "TERROR", "MILITARY", "REFUGEE", "ARMEDCONFLICT"]

# PAGE_TITLE 가용 시작일 (메타 컬럼용)
PAGE_TITLE_AVAILABLE_FROM = date(2019, 9, 23)


def _theme_match_clause(prefix: str) -> str:
    """v2themes 컬럼이 ;-separated 일 때 ^prefix or ;prefix 로 시작하는 token 매칭."""
    return f"REGEXP_CONTAINS(v2themes, r'(^|;){prefix}[^;]*')"


def build_query(
    start: date,
    end: date,
    iso3_list: list[str],
    theme_prefixes: list[str],
) -> str:
    """Track 1 GKG 피처 집계 쿼리.

    구조:
      1) daily_raw CTE: 일자 x iso3 raw 집계 (gdelt_titles 1회 스캔)
         - article_count, unique_domain_count, language_diversity
         - v2tone_avg 의 percentile (APPROX_QUANTILES)
         - 각 theme prefix 별 매칭 기사 수
         - persons 통계 (daily unique, daily top1, daily total)
      2) grid CTE: GENERATE_DATE_ARRAY x iso3 (모든 (iso3, date) 행 보장)
      3) joined CTE: grid LEFT JOIN daily_raw → 결측일은 NULL
      4) windowed: ROWS BETWEEN N PRECEDING AND 1 PRECEDING 으로 1d/7d 집계
    """
    iso3_array = ", ".join(f"'{i}'" for i in iso3_list)
    theme_array = ", ".join(f"'{t}'" for t in theme_prefixes)

    # Group B 의 theme prefix 7d ratio 컬럼
    theme_7d_cols = []
    theme_daily_cols = []
    for prefix in theme_prefixes:
        col = f"theme_{prefix.lower()}_articles_daily"
        theme_daily_cols.append(
            f"      COUNTIF({_theme_match_clause(prefix)}) AS {col}"
        )
        theme_7d_cols.append(
            f"    SAFE_DIVIDE("
            f"SUM(j.{col}) OVER w7, "
            f"NULLIF(SUM(j.article_count_daily) OVER w7, 0)"
            f") AS gkg_theme_{prefix.lower()}_ratio_7d"
        )
    # ARMEDCONFLICT 1d ratio (추가)
    armedconflict_1d = (
        f"    SAFE_DIVIDE("
        f"SUM(j.theme_armedconflict_articles_daily) OVER w1, "
        f"NULLIF(SUM(j.article_count_daily) OVER w1, 0)"
        f") AS gkg_theme_armedconflict_ratio_1d"
    )

    return f"""
    -- Track 1 GKG 피처 집계 (Group A 4 + Group B 11 + 메타 2)
    WITH persons_flat AS (
      -- v2persons 를 UNNEST → 일자×iso3×person 별 article 수
      SELECT
        date, iso3,
        TRIM(SPLIT(person_token, ',')[SAFE_OFFSET(0)]) AS person,
        COUNT(*) AS articles_with_person
      FROM `{TARGET_TABLE}`,
           UNNEST(SPLIT(IFNULL(v2persons, ''), ';')) AS person_token
      WHERE date BETWEEN DATE('{start}') AND DATE('{end}')
        AND iso3 IN ({iso3_array})
        AND person_token IS NOT NULL
        AND person_token != ''
      GROUP BY date, iso3, person
    ),
    persons_daily AS (
      SELECT
        date, iso3,
        COUNT(DISTINCT person) AS persons_unique_daily,
        MAX(articles_with_person) AS persons_top1_articles_daily,
        SUM(articles_with_person) AS persons_total_articles_daily
      FROM persons_flat
      WHERE person IS NOT NULL AND person != ''
      GROUP BY date, iso3
    ),
    daily_raw AS (
      -- gdelt_titles 본체 일자×iso3 집계 (1회 스캔)
      SELECT
        date,
        iso3,
        COUNT(*) AS article_count_daily,
        COUNT(DISTINCT domain) AS unique_domain_daily,
        COUNT(DISTINCT language) AS language_diversity_daily,
        APPROX_QUANTILES(v2tone_avg, 100)[SAFE_OFFSET(10)] AS tone_p10_daily,
{(',' + chr(10)).join(theme_daily_cols)}
      FROM `{TARGET_TABLE}`
      WHERE date BETWEEN DATE('{start}') AND DATE('{end}')
        AND iso3 IN ({iso3_array})
      GROUP BY date, iso3
    ),
    grid AS (
      -- 모든 (iso3, date) 행 보장 — 결측일 명시
      SELECT iso3, date
      FROM UNNEST([{iso3_array}]) AS iso3
      CROSS JOIN UNNEST(GENERATE_DATE_ARRAY(DATE('{start}'), DATE('{end}'))) AS date
    ),
    joined AS (
      SELECT
        g.iso3,
        g.date,
        COALESCE(d.article_count_daily, 0) AS article_count_daily,
        COALESCE(d.unique_domain_daily, 0) AS unique_domain_daily,
        COALESCE(d.language_diversity_daily, 0) AS language_diversity_daily,
        d.tone_p10_daily,
{chr(10).join(f"        COALESCE(d.theme_{p.lower()}_articles_daily, 0) AS theme_{p.lower()}_articles_daily," for p in theme_prefixes)}
        COALESCE(p.persons_unique_daily, 0) AS persons_unique_daily,
        COALESCE(p.persons_top1_articles_daily, 0) AS persons_top1_articles_daily,
        COALESCE(p.persons_total_articles_daily, 0) AS persons_total_articles_daily,
        CASE WHEN d.article_count_daily IS NULL OR d.article_count_daily = 0 THEN 1 ELSE 0 END
          AS gkg_missing_mask_daily
      FROM grid g
      LEFT JOIN daily_raw d ON g.iso3 = d.iso3 AND g.date = d.date
      LEFT JOIN persons_daily p ON g.iso3 = p.iso3 AND g.date = p.date
    )
    SELECT
      j.iso3 AS country,
      j.date,
      -- Group A (4)
      SUM(j.article_count_daily) OVER w1 AS gkg_article_count_1d,
      SUM(j.article_count_daily) OVER w7 AS gkg_article_count_7d,
      AVG(j.unique_domain_daily) OVER w7 AS gkg_unique_domain_count_7d,
      AVG(j.language_diversity_daily) OVER w7 AS gkg_language_diversity_7d,
      -- Group B: V2Themes prefix 7d ratios + ARMEDCONFLICT 1d
{','.join(chr(10) + c for c in theme_7d_cols)},
{armedconflict_1d},
      -- Group B: V2Tone p10
      AVG(j.tone_p10_daily) OVER w7 AS gkg_tone_p10_7d,
      AVG(j.tone_p10_daily) OVER w1 AS gkg_tone_p10_1d,
      -- Group B: V2Persons
      AVG(j.persons_unique_daily) OVER w7 AS gkg_persons_unique_count_7d,
      SAFE_DIVIDE(
        SUM(j.persons_top1_articles_daily) OVER w7,
        NULLIF(SUM(j.persons_total_articles_daily) OVER w7, 0)
      ) AS gkg_persons_top1_share_7d,
      -- 메타 (2)
      j.gkg_missing_mask_daily AS gkg_missing_mask,
      CASE WHEN j.date >= DATE('{PAGE_TITLE_AVAILABLE_FROM}') THEN 1 ELSE 0 END
        AS page_title_available_flag
    FROM joined j
    WINDOW
      w1 AS (PARTITION BY j.iso3 ORDER BY j.date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING),
      w7 AS (PARTITION BY j.iso3 ORDER BY j.date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)
    ORDER BY j.iso3, j.date
    """


def dry_run_estimate(client: bigquery.Client, query: str) -> tuple[float, float]:
    """Dry-run 으로 스캔 GB + 예상 비용($) 추정. BigQuery on-demand $6.25/TB 기준."""
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False),
    )
    gb = job.total_bytes_processed / 1e9
    cost_usd = gb / 1024 * 6.25
    return gb, cost_usd


def run_and_save(
    client: bigquery.Client,
    query: str,
    output_path: Path,
    max_bytes_gb: float,
) -> pd.DataFrame:
    """실제 실행 + parquet 저장. max_bytes_gb 한도 초과 시 자동 거부."""
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=int(max_bytes_gb * 1e9),
        use_query_cache=True,
    )
    job = client.query(query, job_config=job_config)
    df = job.to_dataframe(progress_bar_type="tqdm")
    # date 컬럼: db_dtypes.DateDtype → 표준 datetime64[ns] (parquet 읽기 호환성)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--start",
        type=lambda s: date.fromisoformat(s),
        default=DEFAULT_START,
        help=f"집계 시작일 (기본 {DEFAULT_START})",
    )
    p.add_argument(
        "--end",
        type=lambda s: date.fromisoformat(s),
        default=DEFAULT_END,
        help=f"집계 종료일 (기본 {DEFAULT_END})",
    )
    p.add_argument(
        "--themes",
        nargs="+",
        default=DEFAULT_THEME_PREFIXES,
        help=f"Theme prefix subset (기본 {DEFAULT_THEME_PREFIXES})",
    )
    p.add_argument(
        "--output",
        type=str,
        default="gkg_features.parquet",
        help="출력 parquet 파일명 (input/processed/features/ 안에 저장)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="실행 안 하고 스캔 GB 비용 추정만",
    )
    p.add_argument(
        "--max-bytes-gb",
        type=float,
        default=4000.0,
        help="BigQuery maximum_bytes_billed 한도 (GB, 기본 4000=4TB)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    iso3_list = sorted({c["iso3"] for c in COUNTRIES})
    print(f"대상 국가: {len(iso3_list)}개 ({iso3_list[:5]} ...)")
    print(f"기간: {args.start} ~ {args.end} ({(args.end - args.start).days + 1}일)")
    print(f"Theme prefix: {args.themes}")

    client = bigquery.Client(project=GCP_PROJECT)
    query = build_query(
        start=args.start, end=args.end, iso3_list=iso3_list, theme_prefixes=args.themes
    )

    gb, cost = dry_run_estimate(client, query)
    print(f"\n[dry-run] 스캔 예상: {gb:,.2f} GB ({gb/1024:.2f} TB)")
    print(f"[dry-run] 예상 비용: ${cost:.2f} (on-demand $6.25/TB 기준)")

    if args.dry_run:
        print("\n--dry-run 모드 — 실제 실행 안 함. 종료.")
        return

    if gb > args.max_bytes_gb:
        raise SystemExit(
            f"스캔량 {gb:.1f} GB > 한도 {args.max_bytes_gb} GB. 중단.\n"
            f"실행하려면 --max-bytes-gb {gb + 100:.0f} 이상으로 재실행."
        )

    output_path = PROCESSED_FEATURES_DIR / args.output
    print(f"\n실제 실행 → {output_path}")
    df = run_and_save(client, query, output_path, args.max_bytes_gb)
    print(f"저장 완료: {len(df):,}행 × {df.shape[1]}컬럼")
    print(f"\n샘플:\n{df.head()}")
    print(f"\nNaN 비율:\n{df.isna().mean().round(4)}")


if __name__ == "__main__":
    main()
