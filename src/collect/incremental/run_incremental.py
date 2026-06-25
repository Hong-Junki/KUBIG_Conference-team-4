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

실행 예:
  # BQ MAX(date) 현황 확인만 (읽기 전용)
  python -m src.collect.incremental.run_incremental --validate-only

  # 권한 검증 (실제 create/write/merge/delete 테스트, raw table 변경 없음)
  python -m src.collect.incremental.run_incremental --validate-permissions

  # dry-run (스캔량 확인만, BQ 쓰기 없음)
  python -m src.collect.incremental.run_incremental \\
    --sources gdelt economic gdelt_titles --dry-run

  # 실제 증분 수집
  python -m src.collect.incremental.run_incremental \\
    --sources gdelt economic gdelt_titles

  # 날짜 강제 지정
  python -m src.collect.incremental.run_incremental \\
    --sources gdelt --start 2026-04-01 --end 2026-04-10

필수 환경변수:
  GCP_PROJECT_ID                 — GCP 프로젝트 ID (기본값: conflict-ew-mvp-20260604)
  GOOGLE_APPLICATION_CREDENTIALS — 서비스 계정 JSON 로컬 경로 또는 ADC 사용
  FRED_API_KEY                   — FRED API 키 (economic 소스 사용 시)
"""

import argparse
import json
import os
import sys
import traceback
import uuid
from datetime import date, datetime
from pathlib import Path

from ..utils import get_logger
from .bigquery_io import generate_run_id

logger = get_logger(__name__)

VALID_SOURCES = ["gdelt", "economic", "gdelt_titles"]
_DEFAULT_PROJECT = "conflict-ew-mvp-20260604"

# raw target table 정의 (이 목록 외 테이블은 절대 수정하지 않는다)
RAW_TARGETS = {
    "gdelt":        "conflict_ew.gdelt_processed_events",
    "economic":     "conflict_ew.economic_daily",
    "gdelt_titles": "conflict_ew.gdelt_titles",
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
        logger.error("GOOGLE_APPLICATION_CREDENTIALS 또는 gcloud auth application-default login 필요")
        return False


# ──────────────────────────────────────────────
# validate-only: BQ 현황 읽기 전용
# ──────────────────────────────────────────────

def validate_only(project_id: str) -> None:
    from google.cloud import bigquery
    from .state import get_max_date_from_bq

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


# ──────────────────────────────────────────────
# validate-permissions: 실제 권한 테스트
# ──────────────────────────────────────────────

def validate_permissions(project_id: str) -> bool:
    """
    임시 테이블로 create/write/merge/delete 권한을 검증한다.
    raw target 테이블은 변경하지 않는다.

    Returns:
        True if all passed, False otherwise.
    """
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

    # BQ job 생성
    check("BigQuery job 생성", lambda: list(client.query("SELECT 1")))

    # dataset 조회
    check("dataset 조회", lambda: client.get_dataset(f"{project_id}.conflict_ew"))

    # target schema 조회
    for tbl in ["gdelt_processed_events", "economic_daily", "gdelt_titles"]:
        check(f"target schema: {tbl}", lambda t=tbl: client.get_table(f"{project_id}.conflict_ew.{t}"))

    # target MAX(date)
    for tbl, col in [("gdelt_processed_events", "event_date"), ("economic_daily", "date"), ("gdelt_titles", "date")]:
        check(
            f"target MAX(date): {tbl}",
            lambda t=tbl, c=col: list(client.query(f"SELECT MAX({c}) as m FROM `{project_id}.conflict_ew.{t}`")),
        )

    # staging create
    schema = [bigquery.SchemaField("test_col", "STRING")]
    check("staging create", lambda: client.create_table(bigquery.Table(test_table, schema=schema), exists_ok=True))

    # staging write
    def _write():
        errors = client.insert_rows_json(test_table, [{"test_col": "hello"}])
        if errors:
            raise Exception(str(errors))
    check("staging write", _write)

    # staging MERGE
    def _merge():
        q = f"""
        MERGE `{test_table}` T
        USING (SELECT 'world' AS test_col) S
        ON T.test_col = S.test_col
        WHEN NOT MATCHED THEN INSERT (test_col) VALUES (S.test_col)
        """
        list(client.query(q))
    check("staging MERGE", _merge)

    # staging delete
    check("staging delete", lambda: client.delete_table(test_table, not_found_ok=True))

    logger.info("=== 권한 검증 결과 ===")
    for k, v in results.items():
        status = "✓" if v == "PASS" else "✗"
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

def run_gdelt(project_id: str, args: argparse.Namespace, run_id: str) -> dict:
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


def run_gdelt_titles(project_id: str, args: argparse.Namespace, run_id: str) -> dict:
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
        )
    except Exception as e:
        logger.error(f"GDELT Titles 수집 실패: {e}")
        logger.debug(traceback.format_exc())
        return {"source": "gdelt_titles", "run_id": run_id, "passed": False, "error": str(e)}


# ──────────────────────────────────────────────
# CLI
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
        help=f"수집할 소스 목록 ({VALID_SOURCES}). 기본값: 전체",
    )
    p.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="수집 시작일 YYYY-MM-DD. 미지정 시 BQ MAX(date) - overlap_days",
    )
    p.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="수집 종료일 YYYY-MM-DD. 미지정 시 어제 날짜",
    )
    p.add_argument(
        "--overlap-days",
        type=int,
        default=None,
        metavar="N",
        help="overlap 일수. 미지정 시 소스별 기본값 (gdelt=3, economic=10, gdelt_titles=3)",
    )
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
        help="임시 테이블로 create/write/merge/delete 권한 검증. raw table 변경 없음.",
    )
    p.add_argument(
        "--max-bytes-billed",
        type=int,
        default=200 * 1024**3,
        help="BQ MERGE 쿼리 최대 청구 바이트 (기본 200GB)",
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or generate_run_id()
    project_id = _get_project_id(args)

    # ── 특수 모드 ──────────────────────────────

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

    # ── 일반 수집 모드 ─────────────────────────

    logger.info("=" * 60)
    logger.info("증분 원천 데이터 수집 시작")
    logger.info("  담당 범위: raw BigQuery 테이블 적재까지")
    logger.info("  대상 테이블:")
    for src in args.sources:
        logger.info(f"    {src}: {project_id}.{RAW_TARGETS[src]}")
    logger.info(f"  run_id  : {run_id}")
    logger.info(f"  project : {project_id}")
    logger.info(f"  sources : {args.sources}")
    logger.info(f"  start   : {args.start or '(BQ MAX-overlap)'}")
    logger.info(f"  end     : {args.end or '(어제)'}")
    logger.info(f"  dry_run : {args.dry_run}")
    logger.info("=" * 60)

    if not _check_credentials():
        sys.exit(1)

    # dry-run이 아닌 실제 수집에서는 권한 검증 선행
    if not args.dry_run:
        logger.info("권한 사전 검증 중...")
        if not validate_permissions(project_id):
            logger.error("권한 검증 실패. 수집을 시작하지 않습니다.")
            sys.exit(1)

    results: dict[str, dict] = {}
    source_runners = {
        "gdelt": run_gdelt,
        "economic": run_economic,
        "gdelt_titles": run_gdelt_titles,
    }

    for source in args.sources:
        logger.info("─" * 40)
        logger.info(f"{source} 수집 시작")
        results[source] = source_runners[source](project_id, args, run_id)

    # ── 결과 요약 ──────────────────────────────
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
                f"({r.get('estimated_gb', 0):.2f} GB 예상, "
                f"target MAX={r.get('last_bq_date', 'N/A')})"
            )
        elif r.get("passed"):
            rows = r.get("rows_merged") or r.get("new_rows", 0)
            logger.info(
                f"  {source}: 성공 ({rows:,}행 변경, "
                f"{r.get('start')} ~ {r.get('end')}, "
                f"target={target})"
            )
        else:
            err = r.get("error") or r.get("validation", {}).get("issues")
            logger.error(f"  {source}: 실패 — {err}")
            failed_sources.append(source)

    if failed_sources:
        logger.error(f"실패한 소스: {failed_sources}")
        logger.info("(성공한 소스는 rollback하지 않음 — 부분 성공 상태)")

    # JSON 로그 저장
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"incremental_{run_id}.json"
    with open(log_path, "w") as f:
        json.dump(
            {"run_id": run_id, "project": project_id, "results": results},
            f, indent=2, default=str,
        )
    logger.info(f"JSON 로그: {log_path}")

    sys.exit(1 if failed_sources else 0)


if __name__ == "__main__":
    main()
