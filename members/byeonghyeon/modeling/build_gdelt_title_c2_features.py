"""
GDELT Title C2 Feature Builder — Experiment C2
===============================================
기존 `gdelt_title_features.parquet` (C 피처)를 읽어서
BigQuery 추가 비용 없이 파생 피처를 로컬에서 생성한다.

입력:
  members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_features.parquet
  (214,257행 × 23컬럼, C 실험용 GDELT title 피처)

출력:
  members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_c2_features.parquet

생성 피처 (총 19개):
  3일 rolling (9개):
    gdelt_title_count_3d             3일 기사 수 합계
    gdelt_title_nonnull_count_3d     3일 title 비결측 합계
    gdelt_title_negative_count_3d    3일 부정 기사 수 합계
    gdelt_title_positive_count_3d    3일 긍정 기사 수 합계
    gdelt_title_eng_count_3d         3일 영어 기사 수 합계
    gdelt_title_tone_mean_3d         3일 평균 톤
    gdelt_title_tone_min_3d          3일 최저 톤
    gdelt_title_domain_diversity_3d  3일 도메인 다양성 평균
    gdelt_title_lang_diversity_3d    3일 언어 다양성 평균

  spike feature (5개):
    gdelt_title_count_3d_vs_7d       최근 3일 일평균 / 7일 일평균 비율  (>1=급증)
    gdelt_title_count_3d_vs_14d      최근 3일 일평균 / 14일 일평균 비율
    gdelt_title_negative_3d_vs_7d    부정 기사 최근 3일 일평균 / 7일 일평균
    gdelt_title_negative_3d_vs_14d   부정 기사 최근 3일 일평균 / 14일 일평균
    gdelt_title_tone_drop_3d_vs_7d   톤 악화: tone_mean_7d − tone_mean_3d (양수=최근 더 부정)

  country-normalized feature (5개):
    gdelt_title_count_7d_country_z_90d
      (count_7d - 과거90일 rolling mean) / (과거90일 rolling std + ε)
    gdelt_title_negative_count_7d_country_z_90d
    gdelt_title_tone_mean_7d_country_z_90d
    gdelt_title_count_7d_country_ratio_90d
      count_7d / (과거90일 rolling mean + ε)
    gdelt_title_negative_count_7d_country_ratio_90d

Leakage 방지:
  - 모든 rolling: country별 date 정렬 후 과거 방향
  - 3d rolling: ROWS[-2:현재] (min_periods=1) — 당일 포함
  - 14d rolling: ROWS[-13:현재] (min_periods=1) — 당일 포함
  - 90d rolling (country z-score): ROWS[-89:현재] — 당일 포함, 미래 없음
  - sparse grid 주의: BQ parquet는 보도 있는 날만 포함.
    rolling은 실제 존재하는 행 기준으로 계산됨 (달력 기준 아님).

실행:
  python members/byeonghyeon/modeling/build_gdelt_title_c2_features.py

  # 입력/출력 경로 명시
  python members/byeonghyeon/modeling/build_gdelt_title_c2_features.py \\
    --input  /path/to/gdelt_title_features.parquet \\
    --output /path/to/gdelt_title_c2_features.parquet

  # 생성 결과 검증 (shape, nulls, 컬럼 목록)
  python members/byeonghyeon/modeling/build_gdelt_title_c2_features.py --verify
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_SUBTREE_ROOT = _SCRIPT_DIR.parent

_GDELT_DIR = _SUBTREE_ROOT / "input" / "processed" / "gdelt_titles"
DEFAULT_INPUT  = _GDELT_DIR / "gdelt_title_features.parquet"
DEFAULT_OUTPUT = _GDELT_DIR / "gdelt_title_c2_features.parquet"

# division by zero 방지용 epsilon
EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 공개 상수 (학습 스크립트에서 import 가능)
# ─────────────────────────────────────────────────────────────────────────────

GDELT_TITLE_C2_FEATURE_COLS = [
    # 3일 rolling (9개)
    "gdelt_title_count_3d",
    "gdelt_title_nonnull_count_3d",
    "gdelt_title_negative_count_3d",
    "gdelt_title_positive_count_3d",
    "gdelt_title_eng_count_3d",
    "gdelt_title_tone_mean_3d",
    "gdelt_title_tone_min_3d",
    "gdelt_title_domain_diversity_3d",
    "gdelt_title_lang_diversity_3d",
    # spike (5개)
    "gdelt_title_count_3d_vs_7d",
    "gdelt_title_count_3d_vs_14d",
    "gdelt_title_negative_3d_vs_7d",
    "gdelt_title_negative_3d_vs_14d",
    "gdelt_title_tone_drop_3d_vs_7d",
    # country-normalized (5개)
    "gdelt_title_count_7d_country_z_90d",
    "gdelt_title_negative_count_7d_country_z_90d",
    "gdelt_title_tone_mean_7d_country_z_90d",
    "gdelt_title_count_7d_country_ratio_90d",
    "gdelt_title_negative_count_7d_country_ratio_90d",
]


# ─────────────────────────────────────────────────────────────────────────────
# 피처 생성
# ─────────────────────────────────────────────────────────────────────────────

def _roll_sum(grp: pd.Series, window: int) -> pd.Series:
    """Country별 group 내부에서 과거 방향 rolling sum. 당일 포함, min_periods=1."""
    return grp.rolling(window, min_periods=1).sum()


def _roll_mean(grp: pd.Series, window: int) -> pd.Series:
    """Country별 group 내부에서 과거 방향 rolling mean. 당일 포함, min_periods=1."""
    return grp.rolling(window, min_periods=1).mean()


def _roll_min(grp: pd.Series, window: int) -> pd.Series:
    """Country별 group 내부에서 과거 방향 rolling min. 당일 포함, min_periods=1."""
    return grp.rolling(window, min_periods=1).min()


def build_c2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    기존 gdelt_title_features DataFrame에 C2 파생 피처를 추가해 반환한다.

    입력 df는 반드시 country 기준으로 date 오름차순 정렬되어 있어야 한다.
    모든 rolling은 groupby(country) 내에서 과거 방향으로만 계산한다.
    """
    print("  [1/4] 3일 rolling 피처 계산 중...")

    # ── 3일 rolling ────────────────────────────────────────────────────────────
    # sum: 보도량 계열 (count, nonnull, negative, positive, eng)
    for col_1d, col_3d in [
        ("gdelt_title_count_1d",           "gdelt_title_count_3d"),
        ("gdelt_title_nonnull_count_1d",    "gdelt_title_nonnull_count_3d"),
        ("gdelt_title_negative_count_1d",   "gdelt_title_negative_count_3d"),
        ("gdelt_title_positive_count_1d",   "gdelt_title_positive_count_3d"),
        ("gdelt_title_eng_count_1d",        "gdelt_title_eng_count_3d"),
    ]:
        df[col_3d] = df.groupby("country")[col_1d].transform(
            lambda x: _roll_sum(x, 3)
        )

    # mean: 평균값 계열 (tone_mean, domain_diversity, lang_diversity)
    for col_1d, col_3d in [
        ("gdelt_title_tone_mean_1d",        "gdelt_title_tone_mean_3d"),
        ("gdelt_title_domain_diversity_1d", "gdelt_title_domain_diversity_3d"),
        ("gdelt_title_lang_diversity_1d",   "gdelt_title_lang_diversity_3d"),
    ]:
        df[col_3d] = df.groupby("country")[col_1d].transform(
            lambda x: _roll_mean(x, 3)
        )

    # min: 최저 톤
    df["gdelt_title_tone_min_3d"] = df.groupby("country")["gdelt_title_tone_min_1d"].transform(
        lambda x: _roll_min(x, 3)
    )

    print("  [2/4] Spike feature (3d vs 7d/14d) 계산 중...")

    # ── 14일 rolling (spike 분모용, parquet에 없으므로 직접 계산) ─────────────
    df["_count_14d"]    = df.groupby("country")["gdelt_title_count_1d"].transform(
        lambda x: _roll_sum(x, 14)
    )
    df["_negative_14d"] = df.groupby("country")["gdelt_title_negative_count_1d"].transform(
        lambda x: _roll_sum(x, 14)
    )

    # ── spike feature ─────────────────────────────────────────────────────────
    # 일평균 비율: (count_3d/3) / (count_Nd/N)  — 1보다 크면 최근 급증
    # 분모 0 방지: + EPS
    df["gdelt_title_count_3d_vs_7d"] = (
        (df["gdelt_title_count_3d"]  / 3) /
        (df["gdelt_title_count_7d"]  / 7 + EPS)
    )
    df["gdelt_title_count_3d_vs_14d"] = (
        (df["gdelt_title_count_3d"]  / 3) /
        (df["_count_14d"]            / 14 + EPS)
    )
    df["gdelt_title_negative_3d_vs_7d"] = (
        (df["gdelt_title_negative_count_3d"] / 3) /
        (df["gdelt_title_negative_count_7d"] / 7 + EPS)
    )
    df["gdelt_title_negative_3d_vs_14d"] = (
        (df["gdelt_title_negative_count_3d"] / 3) /
        (df["_negative_14d"]                 / 14 + EPS)
    )
    # 톤 악화: 7일 평균 대비 최근 3일 평균 하락 (양수 = 최근 더 부정적)
    df["gdelt_title_tone_drop_3d_vs_7d"] = (
        df["gdelt_title_tone_mean_7d"] - df["gdelt_title_tone_mean_3d"]
    )

    # 임시 컬럼 제거
    df.drop(columns=["_count_14d", "_negative_14d"], inplace=True)

    print("  [3/4] Country-normalized feature (90d rolling z-score / ratio) 계산 중...")

    # ── country 내 90일 rolling mean / std ────────────────────────────────────
    # 당일 포함 과거 90일: min_periods=1 (초기 구간은 가용 데이터만 사용)
    # sparse grid이므로 90일 = 실제 존재하는 행 기준 최대 90개 행
    for base_col, suffix in [
        ("gdelt_title_count_7d",          "count_7d"),
        ("gdelt_title_negative_count_7d", "negative_count_7d"),
        ("gdelt_title_tone_mean_7d",      "tone_mean_7d"),
    ]:
        roll_mean_col = f"_rm90_{suffix}"
        roll_std_col  = f"_rs90_{suffix}"

        df[roll_mean_col] = df.groupby("country")[base_col].transform(
            lambda x: x.rolling(90, min_periods=1).mean()
        )
        df[roll_std_col] = df.groupby("country")[base_col].transform(
            lambda x: x.rolling(90, min_periods=1).std().fillna(0)
        )

        # z-score: (value - mean) / (std + ε)
        df[f"gdelt_title_{suffix}_country_z_90d"] = (
            (df[base_col] - df[roll_mean_col]) /
            (df[roll_std_col] + EPS)
        )

        # ratio: value / (mean + ε)
        if suffix in ("count_7d", "negative_count_7d"):
            df[f"gdelt_title_{suffix}_country_ratio_90d"] = (
                df[base_col] / (df[roll_mean_col] + EPS)
            )

        df.drop(columns=[roll_mean_col, roll_std_col], inplace=True)

    print("  [4/4] 피처 정리 및 검증 중...")

    # 최종 결과: date, country + C2 피처만 포함
    output_cols = ["date", "country"] + GDELT_TITLE_C2_FEATURE_COLS
    missing = [c for c in output_cols if c not in df.columns]
    if missing:
        print(f"  [오류] 누락된 컬럼: {missing}")
        sys.exit(1)

    return df[output_cols].copy()


# ─────────────────────────────────────────────────────────────────────────────
# 검증 출력
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, label: str = "") -> None:
    tag = f" ({label})" if label else ""
    print(f"\n{'='*60}")
    print(f"  요약{tag}")
    print(f"{'='*60}")
    print(f"  shape   : {df.shape[0]:,}행 × {df.shape[1]}컬럼")
    print(f"  date    : {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  country : {df['country'].nunique()}개국")

    null_total = df[GDELT_TITLE_C2_FEATURE_COLS].isnull().sum().sum()
    print(f"  null    : {null_total} {'✅' if null_total == 0 else '⚠️'}")

    print(f"\n  컬럼 목록 ({len(df.columns)}개):")
    feat_cols = [c for c in df.columns if c not in ("date", "country")]
    for c in feat_cols:
        mn  = df[c].mean()
        std = df[c].std()
        print(f"    {c:<48} mean={mn:8.4f}  std={std:8.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="gdelt_title_features.parquet에서 C2 파생 피처를 생성합니다."
    )
    parser.add_argument("--input",  type=str, default=str(DEFAULT_INPUT),
                        help=f"입력 parquet (기본: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"출력 parquet (기본: {DEFAULT_OUTPUT})")
    parser.add_argument("--verify", action="store_true",
                        help="출력 파일이 이미 존재하면 통계만 출력하고 종료")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    print("=" * 70)
    print("  GDELT Title C2 Feature Builder")
    print(f"  입력: {input_path}")
    print(f"  출력: {output_path}")
    print(f"  BigQuery 호출: 없음 (로컬 parquet 재사용)")
    print(f"  생성 피처: {len(GDELT_TITLE_C2_FEATURE_COLS)}개")
    print("=" * 70)

    # --verify: 기존 파일 통계 확인
    if args.verify and output_path.exists():
        print("\n[--verify] 기존 출력 파일 통계를 출력합니다.")
        df_out = pd.read_parquet(output_path)
        print_summary(df_out, label="기존 출력")
        return

    # 입력 파일 확인
    if not input_path.exists():
        print(f"\n[오류] 입력 파일 없음: {input_path}")
        print("  먼저 실행하세요:")
        print("    python members/byeonghyeon/modeling/build_gdelt_title_features.py")
        sys.exit(1)

    # 로드
    print("\n[Step 1] 입력 parquet 로드")
    df = pd.read_parquet(input_path)
    print(f"  로드 완료: {df.shape[0]:,}행 × {df.shape[1]}컬럼")
    print(f"  date range: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  countries : {df['country'].nunique()}개국")

    # country별 date 정렬 (rolling 전제 조건)
    print("\n[Step 2] country별 date 오름차순 정렬")
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    print("  정렬 완료")

    # C2 피처 생성
    print("\n[Step 3] C2 파생 피처 생성")
    df_c2 = build_c2_features(df)

    # 통계 출력
    print_summary(df_c2, label="출력 C2 피처")

    # 저장
    print(f"\n[Step 4] parquet 저장")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_c2.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  저장 완료: {output_path}  ({size_mb:.1f} MB)")
    print()
    print("=" * 70)
    print("  완료. 다음 단계:")
    print("    run_stacking_acled_free_with_titles_c2.py 실행 (미작성)")
    print("=" * 70)


if __name__ == "__main__":
    main()
