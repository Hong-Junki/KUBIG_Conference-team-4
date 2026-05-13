"""
과거 데이터 일괄 수집 오케스트레이터

실행:
    python -m src.collect.run_historical [--sources acled gdelt economic] [--dry-run]

옵션:
    --sources   수집할 소스 목록 (기본값: 전체). 예: --sources acled economic
    --dry-run   GDELT BigQuery 스캔량 확인만 수행 (실제 수집 안 함)
    --start     수집 시작일 (YYYY-MM-DD, 기본값: config.COLLECT_START)
    --end       수집 종료일 (YYYY-MM-DD, 기본값: config.COLLECT_END)
"""

import argparse
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

from .config import COLLECT_END, COLLECT_START
from .utils import get_logger

logger = get_logger(__name__)

VALID_SOURCES = ["acled", "gdelt", "economic"]


# ──────────────────────────────────────────────
# 개별 소스 수집 래퍼
# ──────────────────────────────────────────────

def run_acled(start: date, end: date) -> bool:
    """ACLED 과거 수집. 성공 여부 반환."""
    try:
        from .acled_collector import collect_historical
        logger.info("=" * 60)
        logger.info("ACLED 수집 시작")
        logger.info("=" * 60)
        collect_historical(start=start, end=end)
        logger.info("ACLED 수집 완료")
        return True
    except Exception as e:
        logger.error(f"ACLED 수집 실패: {e}")
        logger.debug(traceback.format_exc())
        return False


def run_gdelt(start: date, end: date, dry_run: bool = False) -> bool:
    """GDELT BigQuery 과거 수집. dry_run=True이면 비용 확인만."""
    try:
        from .gdelt_collector import collect_historical_bq, _build_query, dry_run_query
        from .config import COUNTRIES

        logger.info("=" * 60)
        logger.info(f"GDELT BigQuery {'dry-run' if dry_run else '수집'} 시작")
        logger.info("=" * 60)

        if dry_run:
            # 전체 기간 단일 쿼리로 스캔량 추정
            fips_codes = []
            for c in COUNTRIES:
                codes = c["gdelt"] if isinstance(c["gdelt"], list) else [c["gdelt"]]
                fips_codes.extend(codes)
            from .utils import date_range_months
            months = date_range_months(start, end)
            total_gb = 0.0
            logger.info(f"  {len(months)}개 월 쿼리 dry-run 시작...")
            for m_start, m_end in months:
                query = _build_query(fips_codes, m_start, m_end)
                gb = dry_run_query(query)
                total_gb += gb
            logger.info(f"GDELT 전체 예상 스캔량: {total_gb:.2f} GB")
            logger.info(f"  BigQuery 무료 한도: 월 1TB (1000 GB)")
            logger.info(f"  예상 비용: 약 ${max(0, total_gb - 1000) * 0.005:.2f} (초과분 $5/TB)")
            return True

        collect_historical_bq(start=start, end=end)
        logger.info("GDELT 수집 완료")
        return True
    except Exception as e:
        logger.error(f"GDELT 수집 실패: {e}")
        logger.debug(traceback.format_exc())
        return False


def run_economic(start: date, end: date) -> bool:
    """경제지표 과거 수집. 성공 여부 반환."""
    try:
        from .economic_collector import collect_historical
        logger.info("=" * 60)
        logger.info("경제지표 수집 시작")
        logger.info("=" * 60)
        collect_historical(start=start, end=end)
        logger.info("경제지표 수집 완료")
        return True
    except Exception as e:
        logger.error(f"경제지표 수집 실패: {e}")
        logger.debug(traceback.format_exc())
        return False


# ──────────────────────────────────────────────
# 수집 결과 검증
# ──────────────────────────────────────────────

def validate_collection() -> None:
    """수집된 파일 존재 여부 및 기본 커버리지 확인."""
    import pandas as pd

    logger.info("=" * 60)
    logger.info("수집 결과 검증")
    logger.info("=" * 60)

    from .config import COUNTRIES

    issues = []

    # ACLED
    acled_dir = Path("input/raw/acled")
    acled_files = list(acled_dir.glob("*.parquet")) if acled_dir.exists() else []
    acled_iso3s = {f.stem for f in acled_files if not f.stem.startswith(".")}
    expected_iso3s = {c["iso3"] for c in COUNTRIES}
    missing_acled = expected_iso3s - acled_iso3s
    logger.info(f"ACLED: {len(acled_iso3s)}/{len(expected_iso3s)}개국 파일 존재")
    if missing_acled:
        logger.warning(f"  누락 국가: {sorted(missing_acled)}")
        issues.append(f"ACLED 누락 {len(missing_acled)}개국")

    for f in sorted(acled_files):
        try:
            df = pd.read_parquet(f)
            min_date = df["event_date"].min()
            max_date = df["event_date"].max()
            logger.info(f"  {f.stem}: {len(df)}건, {min_date} ~ {max_date}")
        except Exception as e:
            logger.warning(f"  {f.stem}: 읽기 실패 — {e}")
            issues.append(f"ACLED {f.stem} 읽기 실패")

    # GDELT
    gdelt_dir = Path("input/raw/gdelt")
    gdelt_files = list(gdelt_dir.glob("???.parquet")) if gdelt_dir.exists() else []
    gdelt_iso3s = {f.stem for f in gdelt_files}
    missing_gdelt = expected_iso3s - gdelt_iso3s
    logger.info(f"GDELT BQ: {len(gdelt_iso3s)}/{len(expected_iso3s)}개국 파일 존재")
    if missing_gdelt:
        logger.warning(f"  누락 국가: {sorted(missing_gdelt)}")
        issues.append(f"GDELT 누락 {len(missing_gdelt)}개국")

    for f in sorted(gdelt_files):
        try:
            df = pd.read_parquet(f)
            min_date = df["event_date"].min()
            max_date = df["event_date"].max()
            logger.info(f"  {f.stem}: {len(df)}건, {min_date} ~ {max_date}")
        except Exception as e:
            logger.warning(f"  {f.stem}: 읽기 실패 — {e}")
            issues.append(f"GDELT {f.stem} 읽기 실패")

    # 경제지표
    econ_path = Path("input/raw/economic/indicators.parquet")
    if econ_path.exists():
        try:
            df = pd.read_parquet(econ_path)
            logger.info(f"경제지표: {len(df)}일 × {len(df.columns)}개 컬럼")
            logger.info(f"  컬럼: {list(df.columns)}")
            logger.info(f"  기간: {df.index.min().date()} ~ {df.index.max().date()}")
            missing_cols = {"VIX", "WTI", "Gold", "DXY", "STLFSI4"} - set(df.columns)
            if missing_cols:
                logger.warning(f"  누락 컬럼: {missing_cols}")
                issues.append(f"경제지표 누락 컬럼: {missing_cols}")
        except Exception as e:
            logger.warning(f"경제지표: 읽기 실패 — {e}")
            issues.append("경제지표 읽기 실패")
    else:
        logger.warning("경제지표: 파일 없음")
        issues.append("경제지표 파일 없음")

    # 결과 요약
    logger.info("-" * 60)
    if issues:
        logger.warning(f"검증 이슈 {len(issues)}건:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("검증 완료 — 이슈 없음")


# ──────────────────────────────────────────────
# CLI 진입점
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="과거 데이터 일괄 수집 오케스트레이터",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=VALID_SOURCES,
        default=VALID_SOURCES,
        metavar="SOURCE",
        help=f"수집할 소스 목록 (선택: {VALID_SOURCES}). 기본값: 전체",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="GDELT BigQuery 스캔량 확인만 수행 (실제 수집 안 함)",
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=COLLECT_START,
        help=f"수집 시작일 YYYY-MM-DD (기본값: {COLLECT_START})",
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=COLLECT_END,
        help=f"수집 종료일 YYYY-MM-DD (기본값: {COLLECT_END})",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="수집 없이 기존 파일 검증만 수행",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("과거 데이터 수집 오케스트레이터 시작")
    logger.info(f"  기간: {args.start} ~ {args.end}")
    logger.info(f"  소스: {args.sources}")
    if args.dry_run:
        logger.info("  모드: dry-run (GDELT 스캔량 확인만)")
    logger.info("=" * 60)

    if args.validate_only:
        validate_collection()
        return

    results = {}

    if "acled" in args.sources and not args.dry_run:
        results["acled"] = run_acled(args.start, args.end)

    if "gdelt" in args.sources:
        results["gdelt"] = run_gdelt(args.start, args.end, dry_run=args.dry_run)

    if "economic" in args.sources and not args.dry_run:
        results["economic"] = run_economic(args.start, args.end)

    # 수집 후 자동 검증 (dry-run 제외)
    if not args.dry_run and results:
        validate_collection()

    # 결과 요약
    logger.info("=" * 60)
    logger.info("수집 결과 요약")
    for source, ok in results.items():
        status = "성공" if ok else "실패"
        logger.info(f"  {source}: {status}")

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        logger.error(f"실패한 소스: {failed}")
        sys.exit(1)
    else:
        logger.info("모든 소스 수집 완료")


if __name__ == "__main__":
    main()
