"""
Safe ACLED Lag Feature Builder — Experiment F (SE-free, ACLED-lag)
==================================================================
processed ACLED event 데이터에서 publication lag 7일을 명시적으로 반영한
leakage-safe ACLED lag feature를 생성한다.

핵심 원칙:
  t일 예측에 사용하는 모든 feature는 최대 t-7일까지의 ACLED 데이터만 사용한다.
  label window (t+1 ~ t+3) 와 feature window (~ t-7) 사이에 최소 8일 gap이 보장된다.

구현 방식: Method A — shift(7) 후 rolling
  1. processed ACLED 이벤트를 country-date 단위 daily table로 집계
  2. 각 daily series에 .shift(7) 적용
     → shifted[t] = daily[t-7]  (즉 t일 feature는 t-7일 이전 정보만 참조)
  3. shift된 값에 rolling(7 / 14 / 30) 적용
     → acled_event_count_7d_lag7[t]  = sum(daily[t-13 ~ t-7])
     → acled_event_count_14d_lag7[t] = sum(daily[t-20 ~ t-7])
     → acled_event_count_30d_lag7[t] = sum(daily[t-36 ~ t-7])
  4. 이 방식은 기존 feature_builder.py(ACLED_LAG_DAYS=7)와 동일한 논리이며,
     본 builder는 재현성과 leakage-free 보장을 명시적으로 문서화하는 목적으로 작성됨.

입력:
  conflict-early-warning/input/processed/acled/{iso3}.parquet
  (국가별 이벤트 레코드, event_type / fatalities / inter1 / inter2 포함)

출력:
  members/byeonghyeon/input/processed/acled_safe/safe_acled_lag_features.parquet
  (gitignore로 추적 안 됨 — 재생성 가능)

생성 feature (총 15개):
  count / fatalities 계열 (9개):
    safe_acled_event_count_7d_lag7
    safe_acled_event_count_14d_lag7
    safe_acled_event_count_30d_lag7
    safe_acled_fatalities_7d_lag7
    safe_acled_fatalities_14d_lag7
    safe_acled_fatalities_30d_lag7
    safe_acled_fatalities_max_7d_lag7
    safe_acled_fatalities_max_14d_lag7
    safe_acled_fatalities_max_30d_lag7

  event type ratio 계열 (3개, 30d 기준 + lag7):
    safe_acled_ratio_battles_lag7
    safe_acled_ratio_explosions_lag7
    safe_acled_ratio_vac_lag7

  actor type ratio 계열 (2개, 30d 기준 + lag7):
    safe_acled_ratio_state_forces_lag7    (inter1 또는 inter2 == 1)
    safe_acled_ratio_external_forces_lag7 (inter1 또는 inter2 == 8)
    [코드 2/3/4~6은 데이터 없음, 코드 7(Civilians)은 VAC ratio와 중복으로 보류]

  결측 마스크 (1개):
    safe_acled_missing_mask

  총 key column: date, country
  총 feature column: 15개

제외:
  - macis_se_score: 절대 사용 안 함
  - event_count_next3d, fatalities_next3d: future label, 절대 포함 안 함
  - y, y_onset, y_escalation: label, 절대 포함 안 함
  - 이름/문자열 원본 (actor1, actor2, admin1 등): 저장 안 함

실행:
  cd <KUBIG_Conference-team-4 루트>

  # 입력 파일 존재 확인 (실제 처리 안 함)
  python members/byeonghyeon/modeling/build_safe_acled_lag_features.py --dry-run

  # 실제 실행
  python members/byeonghyeon/modeling/build_safe_acled_lag_features.py

  # 기간 지정
  python members/byeonghyeon/modeling/build_safe_acled_lag_features.py \\
    --start 2014-01-01 --end 2025-03-31
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_SUBTREE_ROOT = _SCRIPT_DIR.parent           # members/byeonghyeon/
_TEAM_REPO    = _SUBTREE_ROOT.parent.parent  # KUBIG_Conference-team-4/

# conflict-early-warning 개인 레포 (processed ACLED 위치)
_PERSONAL_REPO = _TEAM_REPO.parent / "conflict-early-warning"

ACLED_PROCESSED_DIR = _PERSONAL_REPO / "input" / "processed" / "acled"

DEFAULT_OUTPUT = (_SUBTREE_ROOT / "input" / "processed" / "acled_safe"
                  / "safe_acled_lag_features.parquet")

DEFAULT_START = "2014-01-01"
DEFAULT_END   = "2025-03-31"

# Publication lag: 7일 (ACLED는 주간 업데이트 기준)
ACLED_LAG_DAYS = 7

# Rolling window 크기
WINDOWS = [7, 14, 30]

EPS = 1e-6  # ratio 분모 0 방지


# ─────────────────────────────────────────────────────────────────────────────
# 공개 상수 (학습 스크립트에서 import 가능)
# ─────────────────────────────────────────────────────────────────────────────

SAFE_ACLED_FEATURE_COLS = [
    # count / fatalities 계열 (9개)
    "safe_acled_event_count_7d_lag7",
    "safe_acled_event_count_14d_lag7",
    "safe_acled_event_count_30d_lag7",
    "safe_acled_fatalities_7d_lag7",
    "safe_acled_fatalities_14d_lag7",
    "safe_acled_fatalities_30d_lag7",
    "safe_acled_fatalities_max_7d_lag7",
    "safe_acled_fatalities_max_14d_lag7",
    "safe_acled_fatalities_max_30d_lag7",
    # event type ratio 계열 (3개)
    "safe_acled_ratio_battles_lag7",
    "safe_acled_ratio_explosions_lag7",
    "safe_acled_ratio_vac_lag7",
    # actor type ratio 계열 (2개, 실제 데이터 기반)
    # 전체 58개국 inter1/inter2 실제 코드: [0,1,8] / [0,1,7,8]
    # 코드 2(Rebel)/3(Political militia)는 데이터에 없어 제외
    # 코드 7(Civilians)은 inter2에만 등장하며 vac ratio와 중복 가능성으로 보류
    "safe_acled_ratio_state_forces_lag7",
    "safe_acled_ratio_external_forces_lag7",
    # 결측 마스크
    "safe_acled_missing_mask",
]

# ACLED inter code → 피처 suffix 매핑 (processed parquet 기준 int64 코드)
# ACLED codebook: 1=State forces, 2=Rebel groups, 3=Political militias,
#                 4=Identity militias, 5=Rioters, 6=Protesters, 7=Civilians,
#                 8=External/Other forces
#
# 실제 데이터 검증 결과 (전체 58개국):
#   inter1 실제 코드: [0, 1, 8]
#   inter2 실제 코드: [0, 1, 7, 8]
#   코드 2/3/4/5/6은 데이터에 존재하지 않음 → 해당 ratio는 항상 0이 되므로 제외
#   코드 7(Civilians)은 inter2에만 등장하며, event_type ratio에 이미 VAC ratio가
#   존재하므로 중복 가능성을 피해 보류
ACTOR_TYPE_MAP = {
    1: "state_forces",     # inter1/inter2 == 1: State forces (데이터 존재 확인 ✅)
    8: "external_forces",  # inter1/inter2 == 8: External/Other forces (데이터 존재 확인 ✅)
}

# 이벤트 유형 매핑
EVENT_TYPE_MAP = {
    "battles":    "Battles",
    "explosions": "Explosions/Remote violence",
    "vac":        "Violence against civilians",
}


# ─────────────────────────────────────────────────────────────────────────────
# 단일 국가 feature 생성
# ─────────────────────────────────────────────────────────────────────────────

def _build_country_features(iso3: str, date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    단일 국가 safe ACLED lag feature 생성.

    leakage-free 보장:
      - shift(ACLED_LAG_DAYS) 적용 후 rolling → t일 feature는 최대 t-7일 데이터만 참조
      - label window (t+1 ~ t+3)와 feature window (~ t-7) 사이 8일 이상 gap 확보
    """
    acled_path = ACLED_PROCESSED_DIR / f"{iso3}.parquet"

    # 빈 feature 프레임 (ACLED 없는 국가용)
    if not acled_path.exists():
        df_empty = pd.DataFrame({"date": date_range, "country": iso3})
        for col in SAFE_ACLED_FEATURE_COLS:
            df_empty[col] = 0.0
        df_empty["safe_acled_missing_mask"] = 1.0
        return df_empty

    # 이벤트 로드 (label_builder와 동일한 3개 event_type만 사용)
    df = pd.read_parquet(acled_path)
    df["event_date"] = pd.to_datetime(df["event_date"], utc=True).dt.normalize()

    # ── 일별 집계 ──────────────────────────────────────────────────────────────
    daily = df.groupby("event_date").agg(
        event_count=("event_id_cnty", "count"),
        fatalities_sum=("fatalities", "sum"),
        fatalities_max=("fatalities", "max"),
    ).reindex(date_range, fill_value=0)

    # 이벤트 유형별 일간 수
    daily_etype = {}
    for short, full in EVENT_TYPE_MAP.items():
        daily_etype[short] = (
            df[df["event_type"] == full]
            .groupby("event_date").size()
            .reindex(date_range, fill_value=0)
        )

    # 행위자 유형별 일간 수 (inter1 OR inter2 해당 코드)
    daily_actor = {}
    for code, name in ACTOR_TYPE_MAP.items():
        mask = (df["inter1"] == code) | (df["inter2"] == code)
        daily_actor[name] = (
            df[mask].groupby("event_date").size()
            .reindex(date_range, fill_value=0)
        )

    # ── Method A: shift(7) 후 rolling ─────────────────────────────────────────
    # shifted_x[t] = daily_x[t - 7]
    # rolling(w)[t] = sum/mean(daily_x[t-7-w+1 .. t-7])
    # → t일 feature는 최대 t-7일 데이터만 사용: leakage-free ✅

    feats = pd.DataFrame({"date": date_range, "country": iso3})

    # count / fatalities
    s_count    = daily["event_count"].shift(ACLED_LAG_DAYS)
    s_fat_sum  = daily["fatalities_sum"].shift(ACLED_LAG_DAYS)
    s_fat_max  = daily["fatalities_max"].shift(ACLED_LAG_DAYS)

    for w in WINDOWS:
        feats[f"safe_acled_event_count_{w}d_lag7"]    = s_count.rolling(w, min_periods=1).sum().values
        feats[f"safe_acled_fatalities_{w}d_lag7"]     = s_fat_sum.rolling(w, min_periods=1).sum().values
        feats[f"safe_acled_fatalities_max_{w}d_lag7"] = s_fat_max.rolling(w, min_periods=1).max().values

    # event type ratio (30d 기준 + lag7)
    total_30d = s_count.rolling(30, min_periods=1).sum()
    for short in EVENT_TYPE_MAP:
        s_etype = daily_etype[short].shift(ACLED_LAG_DAYS)
        count_30d = s_etype.rolling(30, min_periods=1).sum()
        feats[f"safe_acled_ratio_{short}_lag7"] = np.where(
            total_30d > EPS, count_30d / (total_30d + EPS), 0.0
        )

    # actor type ratio (30d 기준 + lag7)
    for code, name in ACTOR_TYPE_MAP.items():
        s_actor = daily_actor[name].shift(ACLED_LAG_DAYS)
        count_30d = s_actor.rolling(30, min_periods=1).sum()
        feats[f"safe_acled_ratio_{name}_lag7"] = np.where(
            total_30d > EPS, count_30d / (total_30d + EPS), 0.0
        )

    # 결측 마스크: 첫 이벤트일 + lag7 이전은 실제 커버리지 없음
    if len(df) > 0:
        first_event = df["event_date"].min()
        coverage_start = first_event + pd.Timedelta(days=ACLED_LAG_DAYS)
        feats["safe_acled_missing_mask"] = (date_range < coverage_start).astype(float)
    else:
        feats["safe_acled_missing_mask"] = 1.0

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# 국가 목록 탐색
# ─────────────────────────────────────────────────────────────────────────────

def _discover_countries() -> list[str]:
    """ACLED processed 디렉토리에서 iso3 목록을 자동 탐색."""
    if not ACLED_PROCESSED_DIR.exists():
        return []
    return sorted(p.stem for p in ACLED_PROCESSED_DIR.glob("*.parquet"))


# ─────────────────────────────────────────────────────────────────────────────
# 메인 빌드 함수
# ─────────────────────────────────────────────────────────────────────────────

def build_safe_acled_features(
    start: str = DEFAULT_START,
    end:   str = DEFAULT_END,
    output_path: Path = DEFAULT_OUTPUT,
    dry_run: bool = False,
) -> None:
    countries = _discover_countries()
    date_range = pd.date_range(start=start, end=end, freq="D", tz="UTC")

    print("=" * 70)
    print("  Safe ACLED Lag Feature Builder — Experiment F")
    print(f"  입력 디렉토리: {ACLED_PROCESSED_DIR}")
    print(f"  국가 수      : {len(countries)}개국")
    print(f"  기간         : {start} ~ {end}  ({len(date_range):,}일)")
    print(f"  publication lag: {ACLED_LAG_DAYS}일 (shift 후 rolling)")
    print(f"  출력         : {output_path}")
    print(f"  feature 수   : {len(SAFE_ACLED_FEATURE_COLS)}개")
    print("=" * 70)

    if not ACLED_PROCESSED_DIR.exists():
        print(f"\n[오류] ACLED 입력 디렉토리 없음: {ACLED_PROCESSED_DIR}")
        sys.exit(1)

    if len(countries) == 0:
        print(f"\n[오류] ACLED parquet 파일 없음: {ACLED_PROCESSED_DIR}")
        sys.exit(1)

    print(f"  국가 목록: {countries[:5]} ... ({len(countries)}개)")

    if dry_run:
        print("\n[--dry-run] 입력 파일 존재 확인만 합니다. 실제 처리는 하지 않습니다.\n")
        missing = [c for c in countries if not (ACLED_PROCESSED_DIR / f"{c}.parquet").exists()]
        if missing:
            print(f"  [경고] 누락 파일: {missing}")
        else:
            print(f"  모든 {len(countries)}개 국가 ACLED 파일 존재 ✅")
        print(f"\n  leakage-free 설계 요약:")
        print(f"    shift({ACLED_LAG_DAYS}) 후 rolling → t일 feature는 최대 t-{ACLED_LAG_DAYS}일 데이터만 사용")
        print(f"    label window (t+1 ~ t+3) 와 feature window (~ t-{ACLED_LAG_DAYS}) 사이 {ACLED_LAG_DAYS+1}일 gap ✅")
        print(f"\n  생성 예정 feature ({len(SAFE_ACLED_FEATURE_COLS)}개):")
        for col in SAFE_ACLED_FEATURE_COLS:
            print(f"    {col}")
        return

    # ── 실제 처리 ──────────────────────────────────────────────────────────────
    all_dfs = []
    total_null_before = 0

    for i, iso3 in enumerate(countries, 1):
        print(f"  [{i:2d}/{len(countries)}] {iso3} ...", end=" ", flush=True)
        df_c = _build_country_features(iso3, date_range)

        feat_cols = [c for c in SAFE_ACLED_FEATURE_COLS if c in df_c.columns]
        n_null = df_c[feat_cols].isnull().sum().sum()
        total_null_before += n_null

        # lag 초기 구간 결측 → 0 채움
        df_c[feat_cols] = df_c[feat_cols].fillna(0)

        missing_rows = int((df_c["safe_acled_missing_mask"] == 1.0).sum())
        print(f"null {n_null:,}→0 채움 | coverage_gap {missing_rows}행")
        all_dfs.append(df_c)

    df_all = pd.concat(all_dfs, ignore_index=True)

    # ── 타입 정규화 ────────────────────────────────────────────────────────────
    # date: UTC-aware 확인
    if df_all["date"].dt.tz is None:
        df_all["date"] = df_all["date"].dt.tz_localize("UTC")
    # nullable Int64 → int64/float64
    for col in df_all.columns:
        if str(df_all[col].dtype) == "Int64":
            df_all[col] = df_all[col].astype("int64")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1024 / 1024

    # ── 요약 출력 ─────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  완료")
    print(f"  shape      : {df_all.shape[0]:,}행 × {df_all.shape[1]}컬럼")
    print(f"  date range : {df_all['date'].min().date()} ~ {df_all['date'].max().date()}")
    print(f"  date dtype : {df_all['date'].dtype}  (UTC-aware ✅)")
    print(f"  country 수 : {df_all['country'].nunique()}")
    print(f"  null 수    : {df_all[SAFE_ACLED_FEATURE_COLS].isnull().sum().sum()} ✅")
    print(f"  파일 크기  : {size_mb:.1f} MB")
    print(f"  저장       : {output_path}")
    print()
    print("  leakage-free 확인:")
    print(f"    모든 feature는 shift({ACLED_LAG_DAYS}) 후 rolling으로 계산")
    print(f"    t일 feature 최대 참조일: t - {ACLED_LAG_DAYS}일")
    print(f"    label window: t+1 ~ t+3")
    print(f"    gap: {ACLED_LAG_DAYS + 1}일 이상 ✅")
    print()
    print("  다음 단계:")
    print("    python members/byeonghyeon/modeling/run_stacking_acled_safe_f0.py  (미작성)")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Safe ACLED lag feature를 생성해 parquet로 저장합니다. "
                    "모든 feature는 7일 publication lag를 명시적으로 반영합니다."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="입력 파일 존재 확인만 합니다. 실제 처리는 하지 않습니다.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"출력 parquet 경로 (기본: {DEFAULT_OUTPUT})")
    parser.add_argument("--start", type=str, default=DEFAULT_START,
                        help=f"시작일 YYYY-MM-DD (기본: {DEFAULT_START})")
    parser.add_argument("--end", type=str, default=DEFAULT_END,
                        help=f"종료일 YYYY-MM-DD (기본: {DEFAULT_END})")
    args = parser.parse_args()

    build_safe_acled_features(
        start=args.start,
        end=args.end,
        output_path=Path(args.output),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
