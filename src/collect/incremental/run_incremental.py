"""
증분 수집 CLI 오케스트레이터.

담당 범위:
  외부 데이터 → raw BigQuery 테이블 적재 까지.

  대상 테이블:
    - conflict-ew-mvp-20260604.conflict_ew.gdelt_processed_events
    - conflict-ew-mvp-20260604.conflict_ew.economic_daily
    - conflict-ew-mvp-20260604.conflict_ew.gdelt_titles

  담당하지 않는 것:
    modeling_full_dataset, modeling_acled_free_view,
    feature 생성, embedding, 모델 추론 → 다른 담당자 워크플로우

수집 모드:
  [date 모드 (기본)]
    - BQ MAX(date) - overlap_days 기반 날짜 범위 수집
    - economic, 수동 backfill, reconciliation에 사용
    - --reconcile-days N: 오늘부터 N일 전 ~ 어제 범위를 강제 지정 (GDELT 지연 보정)

  [timestamp 모드]
    - pipeline_watermarks 테이블의 watermark 기반 15분 증분 수집
    - GDELT: DATEADDED 필터 (14자리 YYYYMMDDHHMMSS)
    - GKG Titles: DATE 필터 (14자리 YYYYMMDDHHMMSS)
    - 성공 시에만 watermark 갱신

실행 예:
  # timestamp 모드 (15분 schedule)
  python -m src.collect.incremental.run_incremental \\
    --sources gdelt gdelt_titles --mode timestamp --overlap-minutes 30

  # 날짜 backfill
  python -m src.collect.incremental.run_incremental \\
    --sources gdelt --start 2026-04-01 --end 2026-04-10

  # GDELT 7일 reconciliation
  python -m src.collect.incremental.run_incremental \\
    --sources gdelt --reconcile-days 7

  # BQ 현황 확인
  python -m src.collect.incremental.run_incremental --validate-only

필수 환경변수:
  GCP_PROJECT_ID                 — GCP 프로젝트 ID
  GOOGLE_APPLICATION_CREDENTIALS — 서비스 계정 JSON (또는 ADC)
  FRED_API_KEY                   — FRED API 키 (economic 소스 사용 시)
"""

import argparse
import json
import os
import sys
import traceback
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ..utils import get_logger
from .bigquery_io import generate_run_id
from .state import (
    compute_timestamp_window,
    get_watermark_with_fallback,
    log_timestamp_window,
    set_watermark,
)

logger = get_logger(__name__)

VALID_SOURCES = ["gdelt", "economic", "gdelt_titles"]
_DEFAULT_PROJECT = "conflict-ew-mvp-20260604"

RAW_TARGETS = {
    "gdelt":        "conflict_ew.gdelt_processed_events",
    "economic":     "conflict_ew.economic_daily",
    "gdelt_titles": "conflict_ew.gdelt_titles",
}

# timestamp 모드를 지원하는 소스 (pipeline_watermarks 사용)
_TIMESTAMP_MODE_SOURCES = frozenset({"gdelt", "gdelt_titles"})

# timestamp 모드에서 사용하는 target 날짜 컬럼 (watermark fallback용)
_TARGET_DATE_COLS = {
    "gdelt":        "event_date",
    "gdelt_titles": "date",
}


def _get_project_id(args: argparse.Namespace | None = None) -> str:
    pid = getattr(args, "project_id", None) or os.getenv("GCP_PROJECT_ID", _DEFAULT_PROJECT)
    logger.info(f"GCP_PROJECT_ID: {pid}")
    return pid


def _check_credentials() -> bool:
    cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred:
        p = Path(cred)
        if not p.exists():
            logger.error(f"GOOGLE_APPLICATION_CREDENTIALS 파일 없음: {cred}")
            return False
        logger.info(f"서비스 계정 파일 확인: {p.name}")
        return True
    try:
        import google.auth
        google.auth.default()
        logger.info("Google ADC 인증 사용 (Workload Identity 또는 gcloud auth)")
        return True
    except Exception as e:
        logger.error(f"GCP 인증 없음: {e}")
        return False


# ──────────────────────────────────────────────
# validate-only
# ──────────────────────────────────────────────

def validate_only(project_id: str) -> None:
    from google.cloud import bigquery
    from .state import get_max_date_from_bq, get_watermark

    client = bigquery.Client(project=project_id)
    checks = [
        (f"{project_id}.conflict_ew.gdelt_processed_events", "event_date", "event_date"),
        (f"{project_id}.conflict_ew.economic_daily", "date", None),
        (f"{project_id}.conflict_ew.gdelt_titles", "date", None),
    ]
    logger.info("=== BQ target table 현황 (읽기 전용) ===")
    for fqn, date_col, partition_col in checks:
        max_dt = get_max_date_from_bq(client, fqn, date_col, partition_col)
        logger.info(f"  {fqn.split('.')[-1]}: MAX({date_col}) = {max_dt}")

    logger.info("=== pipeline_watermarks 현황 ===")
    for source in ["gdelt", "gdelt_titles"]:
        try:
            wm = get_watermark(client, project_id, source)
            logger.info(f"  {source}: {wm.isoformat() if wm else 'None (최초 실행)'}")
        except RuntimeError as e:
            logger.warning(f"  {source}: {e}")


# ──────────────────────────────────────────────
# validate-permissions
# ──────────────────────────────────────────────

def validate_permissions(project_id: str) -> bool:
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    run_id = uuid.uuid4().hex[:8]
    test_table = f"{project_id}.conflict_ew._permission_test_{run_id}"
    results: dict[str, str] = {}
    all_pass = True

    def check(key: str, fn):
        nonlocal all_pass
        try:
            fn()
            results[key] = "PASS"
        except Exception as e:
            results[key] = f"FAIL: {e}"
            all_pass = False

    check("BigQuery job 생성", lambda: list(client.query("SELECT 1")))
    check("dataset 조회", lambda: client.get_dataset(f"{project_id}.conflict_ew"))

    for tbl in ["gdelt_processed_events", "economic_daily", "gdelt_titles"]:
        check(f"target schema: {tbl}", lambda t=tbl: client.get_table(f"{project_id}.conflict_ew.{t}"))

    for tbl, col in [
        ("gdelt_processed_events", "event_date"),
        ("economic_daily", "date"),
        ("gdelt_titles", "date"),
    ]:
        check(
            f"target MAX(date): {tbl}",
            lambda t=tbl, c=col: list(client.query(
                f"SELECT MAX({c}) as m FROM `{project_id}.conflict_ew.{t}`"
            )),
        )

    # watermark 테이블 확인 (없으면 경고만)
    try:
        list(client.query(
            f"SELECT source FROM `{project_id}.conflict_ew.pipeline_watermarks` LIMIT 1"
        ))
        results["pipeline_watermarks 조회"] = "PASS"
    except Exception as e:
        results["pipeline_watermarks 조회"] = f"WARNING (migration 미적용?): {e}"

    schema = [bigquery.SchemaField("test_col", "STRING")]
    check("staging create", lambda: client.create_table(
        bigquery.Table(test_table, schema=schema), exists_ok=True
    ))

    def _write():
        errors = client.insert_rows_json(test_table, [{"test_col": "hello"}])
        if errors:
            raise Exception(str(errors))
    check("staging write", _write)

    def _merge():
        q = f"""
        MERGE `{test_table}` T
        USING (SELECT 'world' AS test_col) S
        ON T.test_col = S.test_col
        WHEN NOT MATCHED THEN INSERT (test_col) VALUES (S.test_col)
        """
        list(client.query(q))
    check("staging MERGE", _merge)

    check("staging delete", lambda: client.delete_table(test_table, not_found_ok=True))

    logger.info("=== 권한 검증 결과 ===")
    for k, v in results.items():
        status = "✓" if v == "PASS" else ("△" if "WARNING" in v else "✗")
        logger.info(f"  {status} {k}: {v}")

    if not all_pass:
        logger.error("권한 검증 실패. 다음 IAM 역할이 필요합니다:")
        logger.error("  프로젝트: roles/bigquery.jobUser")
        logger.error("  conflict_ew 데이터셋: roles/bigquery.dataEditor")
    else:
        logger.info("모든 권한 검증 PASS")

    return all_pass


# ──────────────────────────────────────────────
# 소스별 수집 래퍼
# ──────────────────────────────────────────────

def run_gdelt(
    project_id: str,
    args: argparse.Namespace,
    run_id: str,
    start_timestamp: datetime | None = None,
    end_timestamp: datetime | None = None,
) -> dict:
    try:
        from .collect_gdelt_incremental import run_gdelt_incremental
        return run_gdelt_incremental(
            project_id=project_id,
            overlap_days=args.overlap_days,
            forced_start=args.start,
            forced_end=args.end,
            dry_run=args.dry_run,
            max_gb_per_query=args.max_gb_per_query,
            run_id=run_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            overlap_minutes=args.overlap_minutes,
        )
    except Exception as e:
        logger.error(f"GDELT 수집 실패: {e}")
        logger.debug(traceback.format_exc())
        return {"source": "gdelt", "run_id": run_id, "passed": False, "error": str(e)}


def run_economic(project_id: str, args: argparse.Namespace, run_id: str) -> dict:
    try:
        from .collect_economic_incremental import run_economic_incremental
        return run_economic_incremental(
            project_id=project_id,
            overlap_days=args.overlap_days,
            forced_start=args.start,
            forced_end=args.end,
            dry_run=args.dry_run,
            run_id=run_id,
        )
    except Exception as e:
        logger.error(f"경제지표 수집 실패: {e}")
        logger.debug(traceback.format_exc())
        return {"source": "economic", "run_id": run_id, "passed": False, "error": str(e)}


def run_gdelt_titles(
    project_id: str,
    args: argparse.Namespace,
    run_id: str,
    start_timestamp: datetime | None = None,
    end_timestamp: datetime | None = None,
) -> dict:
    try:
        from .collect_gdelt_titles_incremental import run_gdelt_titles_incremental
        return run_gdelt_titles_incremental(
            project_id=project_id,
            overlap_days=args.overlap_days,
            forced_start=args.start,
            forced_end=args.end,
            dry_run=args.dry_run,
            max_gb_per_query=args.max_gb_per_query,
            run_id=run_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            overlap_minutes=args.overlap_minutes,
        )
    except Exception as e:
        logger.error(f"GDELT Titles 수집 실패: {e}")
        logger.debug(traceback.format_exc())
        return {"source": "gdelt_titles", "run_id": run_id, "passed": False, "error": str(e)}


# ──────────────────────────────────────────────
# timestamp 모드 워크플로우
# ──────────────────────────────────────────────

def _run_timestamp_sources(
    project_id: str,
    args: argparse.Namespace,
    run_id: str,
    sources: list[str],
) -> dict[str, dict]:
    """
    timestamp 모드 수집 오케스트레이션.

    소스별로 watermark 조회 → window 계산 → 수집 → 성공 시 watermark 갱신.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    results: dict[str, dict] = {}

    source_runners = {
        "gdelt": run_gdelt,
        "gdelt_titles": run_gdelt_titles,
    }

    for source in sources:
        if source not in _TIMESTAMP_MODE_SOURCES:
            logger.warning(f"[{source}] timestamp 모드 미지원 소스. 건너뜀.")
            results[source] = {
                "source": source, "run_id": run_id,
                "skipped": True, "reason": "timestamp 모드 미지원 소스",
            }
            continue

        logger.info("─" * 40)
        logger.info(f"[{source}] timestamp 모드 수집 시작")

        # 1. watermark 조회 (최초 실행 시 target MAX(date) fallback)
        target_fqn = f"{project_id}.{RAW_TARGETS[source]}"
        date_col = _TARGET_DATE_COLS[source]
        try:
            watermark_before = get_watermark_with_fallback(
                client, project_id, source, target_fqn, date_col
            )
        except RuntimeError as e:
            logger.error(str(e))
            results[source] = {
                "source": source, "run_id": run_id,
                "passed": False, "error": str(e),
            }
            continue

        # 2. timestamp window 계산
        start_ts, end_ts = compute_timestamp_window(
            watermark_before, overlap_minutes=args.overlap_minutes
        )
        log_timestamp_window(source, watermark_before, start_ts, end_ts, args.overlap_minutes)

        # 3. 수집
        runner = source_runners[source]
        result = runner(
            project_id, args, run_id,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
        )

        # 4. 성공 시에만 watermark 갱신
        if result.get("passed") and not args.dry_run:
            try:
                set_watermark(client, project_id, source, end_ts, run_id)
                result["watermark_after"] = end_ts.isoformat()
                logger.info(f"[{source}] watermark 갱신: {end_ts.isoformat()}")
            except Exception as e:
                logger.error(f"[{source}] watermark 갱신 실패 (수집은 성공): {e}")
                result["watermark_update_error"] = str(e)
        elif not result.get("passed"):
            logger.warning(
                f"[{source}] 수집 실패 → watermark 유지: "
                f"{watermark_before.isoformat() if watermark_before else 'None'}"
            )

        result["watermark_before"] = watermark_before.isoformat() if watermark_before else None
        results[source] = result

    return results


# ──────────────────────────────────────────────
# CLI 인자
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="증분 원천 데이터 수집 오케스트레이터 (raw BigQuery 적재까지)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--sources",
        nargs="+",
        choices=VALID_SOURCES,
        default=VALID_SOURCES,
        metavar="SOURCE",
        help=f"수집할 소스 목록. 기본값: 전체",
    )
    p.add_argument(
        "--mode",
        choices=["date", "timestamp"],
        default="date",
        help="수집 모드. date=날짜 단위(기본), timestamp=15분 watermark 증분",
    )
    # ── 날짜 단위 (date 모드) ────────────────
    p.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="수집 시작일 YYYY-MM-DD (date 모드). 미지정 시 BQ MAX(date) - overlap_days",
    )
    p.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="수집 종료일 YYYY-MM-DD (date 모드). 미지정 시 어제",
    )
    p.add_argument(
        "--overlap-days",
        type=int,
        default=None,
        metavar="N",
        help="날짜 overlap 일수 (date 모드). 기본: gdelt=3, economic=10, gdelt_titles=3",
    )
    p.add_argument(
        "--reconcile-days",
        type=int,
        default=None,
        metavar="N",
        help=(
            "보정 일수 (date 모드). 오늘부터 N일 전 ~ 어제 범위를 강제 지정. "
            "--start/--end보다 우선. 예: --reconcile-days 7"
        ),
    )
    # ── timestamp 모드 ───────────────────────
    p.add_argument(
        "--overlap-minutes",
        type=int,
        default=30,
        metavar="N",
        help="watermark overlap 분 (timestamp 모드). 기본 30분",
    )
    # ── 공통 ────────────────────────────────
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="스캔량 확인만 수행. 실제 BQ 쓰기 없음.",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="수집 없이 BQ 현황 조회만 수행.",
    )
    p.add_argument(
        "--validate-permissions",
        action="store_true",
        help="임시 테이블로 권한 검증. raw table 변경 없음.",
    )
    p.add_argument(
        "--max-bytes-billed",
        type=int,
        default=200 * 1024**3,
        help="BQ MERGE 최대 청구 바이트 (기본 200GB)",
    )
    p.add_argument(
        "--max-gb-per-query",
        type=float,
        default=100.0,
        help="월 단위 쿼리 최대 스캔 GB (기본 100GB)",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="실행 ID (미지정 시 자동 생성)",
    )
    p.add_argument(
        "--project-id",
        type=str,
        default=None,
        help=f"GCP 프로젝트 ID (기본값: GCP_PROJECT_ID env 또는 {_DEFAULT_PROJECT})",
    )
    p.add_argument(
        "--schedule-cron",
        type=str,
        default="manual",
        help="실행된 cron 표현식 (로그·추적용). 예: '*/15 * * * *'",
    )
    return p.parse_args()


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    run_id = args.run_id or generate_run_id()
    project_id = _get_project_id(args)

    # ── 특수 모드 ──────────────────────────
    if args.validate_only:
        if not _check_credentials():
            sys.exit(1)
        validate_only(project_id)
        return

    if args.validate_permissions:
        if not _check_credentials():
            sys.exit(1)
        passed = validate_permissions(project_id)
        sys.exit(0 if passed else 1)

    # ── reconcile-days → forced_start/forced_end 변환 ──
    if args.reconcile_days is not None:
        today = date.today()
        reconcile_end = today - timedelta(days=1)
        reconcile_start = today - timedelta(days=args.reconcile_days)
        logger.info(
            f"--reconcile-days {args.reconcile_days}: "
            f"강제 날짜 범위 {reconcile_start} ~ {reconcile_end}"
        )
        args.start = reconcile_start
        args.end = reconcile_end
        args.mode = "date"  # reconcile은 항상 date 모드

    # ── 로그 헤더 ──────────────────────────
    logger.info("=" * 60)
    logger.info("증분 원천 데이터 수집 시작")
    logger.info(f"  event          : {args.schedule_cron}")
    logger.info(f"  mode           : {args.mode}")
    logger.info(f"  sources        : {args.sources}")
    logger.info(f"  run_id         : {run_id}")
    logger.info(f"  project        : {project_id}")
    if args.mode == "date":
        logger.info(f"  start          : {args.start or '(BQ MAX-overlap)'}")
        logger.info(f"  end            : {args.end or '(어제)'}")
        logger.info(f"  overlap_days   : {args.overlap_days or '(소스별 기본값)'}")
        if args.reconcile_days:
            logger.info(f"  reconcile_days : {args.reconcile_days}")
    else:
        logger.info(f"  overlap_minutes: {args.overlap_minutes}")
    logger.info(f"  dry_run        : {args.dry_run}")
    for src in args.sources:
        logger.info(f"  target         : {src} → {project_id}.{RAW_TARGETS[src]}")
    logger.info("=" * 60)

    if not _check_credentials():
        sys.exit(1)

    if not args.dry_run:
        logger.info("권한 사전 검증 중...")
        if not validate_permissions(project_id):
            logger.error("권한 검증 실패. 수집을 시작하지 않습니다.")
            sys.exit(1)

    results: dict[str, dict] = {}

    # ── timestamp 모드 ──────────────────────
    if args.mode == "timestamp":
        ts_sources = [s for s in args.sources if s in _TIMESTAMP_MODE_SOURCES]
        non_ts = [s for s in args.sources if s not in _TIMESTAMP_MODE_SOURCES]
        if non_ts:
            logger.warning(f"timestamp 모드에서 제외된 소스 (미지원): {non_ts}")
        results.update(_run_timestamp_sources(project_id, args, run_id, ts_sources))

    # ── date 모드 ───────────────────────────
    else:
        source_runners = {
            "gdelt": lambda: run_gdelt(project_id, args, run_id),
            "economic": lambda: run_economic(project_id, args, run_id),
            "gdelt_titles": lambda: run_gdelt_titles(project_id, args, run_id),
        }
        for source in args.sources:
            logger.info("─" * 40)
            logger.info(f"{source} 수집 시작 (date 모드)")
            results[source] = source_runners[source]()

    # ── 결과 요약 ───────────────────────────
    logger.info("=" * 60)
    logger.info("수집 결과 요약")

    failed_sources = []
    for source, r in results.items():
        target = f"{project_id}.{RAW_TARGETS[source]}"
        if r.get("skipped"):
            logger.info(f"  {source}: 건너뜀 — {r.get('reason')}")
        elif r.get("dry_run"):
            logger.info(
                f"  {source}: dry-run 완료 "
                f"({r.get('estimated_gb', 0):.3f} GB 예상, "
                f"mode={r.get('mode','?')})"
            )
        elif r.get("passed"):
            rows = r.get("rows_merged") or r.get("new_rows", 0)
            wm_after = r.get("watermark_after", "-")
            logger.info(
                f"  {source}: 성공 ({rows:,}행 변경, "
                f"mode={r.get('mode','?')}, "
                f"watermark_after={wm_after}, "
                f"target={target})"
            )
        else:
            err = r.get("error") or r.get("validation", {}).get("issues")
            wm_before = r.get("watermark_before", "-")
            logger.error(
                f"  {source}: 실패 — {err} "
                f"(watermark 유지: {wm_before})"
            )
            failed_sources.append(source)

    if failed_sources:
        logger.error(f"실패한 소스: {failed_sources}")

    # JSON 로그 저장
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"incremental_{run_id}.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "project": project_id,
                "mode": args.mode,
                "schedule_cron": args.schedule_cron,
                "results": results,
            },
            f, indent=2, default=str,
        )
    logger.info(f"JSON 로그: {log_path}")

    sys.exit(1 if failed_sources else 0)


if __name__ == "__main__":
    main()
