"""
경제지표 증분 수집기.

기존 economic_collector.py의 수집 함수를 재사용하고,
결과를 economic_daily 테이블에 MERGE로 idempotent하게 적재한다.

경제지표 특성:
  - yfinance: 거래일(월~금)만 존재. 주말·휴일은 원천에 없음.
  - STLFSI4: 주간 발표 (매주 목요일). 로우 데이터에도 주간 간격.
  - 수정치: 과거 값 갱신 가능 → update_on_match=True 사용.

적재 방식:
  1. yfinance + FRED 수집 → DataFrame
  2. 파생 지표 계산 (pct_change, volatility_proxy)
  3. staging table 적재
  4. MERGE INTO economic_daily (date 기반, MATCHED 시 UPDATE)
  5. staging 삭제
"""

from datetime import date, timedelta

import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery import SchemaField

from ..config import FRED_SERIES, YFINANCE_TICKERS
from ..economic_collector import collect_fred, collect_yfinance
from ..utils import get_logger
from .bigquery_io import (
    drop_table,
    generate_run_id,
    load_dataframe_to_staging,
    merge_into_target,
    validate_after_load,
)
from .state import (
    compute_collection_window,
    require_max_date_from_bq,
    log_state_summary,
)

logger = get_logger(__name__)

TARGET_DATASET = "conflict_ew"
TARGET_TABLE = "economic_daily"
MERGE_KEY = ["date"]

ECONOMIC_SCHEMA = [
    SchemaField("date", "DATE"),
    SchemaField("VIX", "FLOAT"),
    SchemaField("WTI", "FLOAT"),
    SchemaField("Gold", "FLOAT"),
    SchemaField("DXY", "FLOAT"),
    SchemaField("STLFSI4", "FLOAT"),
    SchemaField("VIX_pct_change", "FLOAT"),
    SchemaField("WTI_pct_change", "FLOAT"),
    SchemaField("Gold_pct_change", "FLOAT"),
    SchemaField("DXY_pct_change", "FLOAT"),
    SchemaField("STLFSI4_pct_change", "FLOAT"),
    SchemaField("econ_volatility_proxy", "FLOAT"),
]
ALL_COLUMNS = [f.name for f in ECONOMIC_SCHEMA]


def _compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    파생 컬럼 계산. 기존 modeling_full_dataset의 econ 컬럼 계산 방식 반영.

    주의: 증분 수집 시 이전 데이터가 없으므로 pct_change 계산에 첫 행은 NaN이 됨.
    이는 BQ MERGE 후 이전 데이터와의 join을 통해 feature_builder가 처리해야 함.
    """
    df = df.copy()
    raw_cols = ["VIX", "WTI", "Gold", "DXY", "STLFSI4"]

    for col in raw_cols:
        if col in df.columns:
            df[f"{col}_pct_change"] = df[col].pct_change() * 100
        else:
            df[f"{col}_pct_change"] = float("nan")

    # volatility proxy: |VIX 1d pct change| * |WTI 1d pct change| (부호 없음)
    if "VIX_pct_change" in df.columns and "WTI_pct_change" in df.columns:
        df["econ_volatility_proxy"] = (
            df["VIX_pct_change"].abs() * df["WTI_pct_change"].abs()
        ).pow(0.5)
    else:
        df["econ_volatility_proxy"] = float("nan")

    return df


def _build_economic_df(start: date, end: date) -> pd.DataFrame | None:
    """yfinance + FRED 수집 후 병합 + 파생 컬럼 계산."""
    df_yf = collect_yfinance(start, end)
    df_fred = collect_fred(start, end)

    frames = [f for f in [df_yf, df_fred] if not f.empty]
    if not frames:
        logger.warning("  경제지표: 수집 데이터 없음")
        return None

    df = frames[0].join(frames[1], how="outer") if len(frames) == 2 else frames[0]
    df = df.sort_index()
    df = df[(df.index.date >= start) & (df.index.date <= end)]

    if df.empty:
        return None

    df = _compute_derived_columns(df)
    df.index = pd.to_datetime(df.index).normalize()
    df = df.reset_index()
    df.rename(columns={"date": "date"}, inplace=True)
    df["date"] = df["date"].dt.date

    # 컬럼 순서 정리
    available = [c for c in ALL_COLUMNS if c in df.columns]
    df = df[available]

    return df


def run_economic_incremental(
    project_id: str,
    overlap_days: int | None = None,
    forced_start: date | None = None,
    forced_end: date | None = None,
    dry_run: bool = False,
    run_id: str | None = None,
) -> dict:
    """
    경제지표 증분 수집 + BQ 적재.

    Returns:
        결과 요약 dict.
    """
    run_id = run_id or generate_run_id()
    client = bigquery.Client(project=project_id)
    target_fqn = f"{project_id}.{TARGET_DATASET}.{TARGET_TABLE}"

    # 1. 수집 기간 결정 — MAX(date) 조회 실패 시 즉시 RuntimeError (fallback 없음)
    last_date = require_max_date_from_bq(client, target_fqn, "date")
    try:
        start, end = compute_collection_window(
            last_date, "economic", overlap_days, forced_start, forced_end
        )
    except ValueError as e:
        logger.warning(str(e))
        return {"source": "economic", "run_id": run_id, "skipped": True, "reason": str(e)}

    log_state_summary("economic", last_date, start, end)

    if dry_run:
        logger.info("  [dry-run] 경제지표 수집 생략 (API 기반, BQ 스캔 없음)")
        return {
            "source": "economic",
            "run_id": run_id,
            "dry_run": True,
            "start": str(start),
            "end": str(end),
            "estimated_gb": 0.0,
        }

    # 2. 수집
    df = _build_economic_df(start, end)
    if df is None:
        logger.warning("  수집된 경제지표 없음 (원천에 신규 데이터 없음)")
        return {
            "source": "economic",
            "run_id": run_id,
            "start": str(start),
            "end": str(end),
            "new_rows": 0,
            "passed": True,
        }

    logger.info(f"  수집 완료: {len(df):,}행 ({start} ~ {end})")

    # 3. staging → MERGE
    available_schema = [f for f in ECONOMIC_SCHEMA if f.name in df.columns]
    staging_fqn = load_dataframe_to_staging(
        client, df, project_id, TARGET_DATASET,
        TARGET_TABLE, run_id, schema=available_schema,
    )
    available_cols = [f.name for f in available_schema]

    try:
        affected = merge_into_target(
            client,
            staging_fqn=staging_fqn,
            target_fqn=target_fqn,
            merge_keys=MERGE_KEY,
            columns=available_cols,
            update_on_match=True,  # 수정치 반영
        )
    finally:
        drop_table(client, staging_fqn)

    # 4. 검증
    validation = validate_after_load(
        client, target_fqn, "date", None,
        expected_min_date=start,
        expected_max_date=end,
        run_id=run_id,
    )

    return {
        "source": "economic",
        "run_id": run_id,
        "start": str(start),
        "end": str(end),
        "rows_collected": len(df),
        "rows_merged": affected,
        "validation": validation,
        "passed": validation.get("passed", False),
    }
