"""
전처리: 수집 데이터 → 정제된 parquet

1. 타임존 UTC 통일
2. 중복 제거 (GDELT 중복률 ~20%)
3. 결측 처리
4. 국가 코드 표준화

입력: input/raw_merged/{acled,gdelt,economic}/  (8년+ 합본)
출력: input/processed/{acled,gdelt,economic}/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("input/raw_merged")
PROCESSED_DIR = Path("input/processed")


# ──────────────────────────────────────────────
# ACLED 전처리
# ──────────────────────────────────────────────

def preprocess_acled(raw_dir: Path = RAW_DIR / "acled") -> dict[str, pd.DataFrame]:
    """
    ACLED 국가별 parquet → 정제.

    처리:
      - event_date UTC 확인
      - event_id_cnty 기준 중복 제거
      - fatalities 음수/결측 → 0
      - inter1/inter2 텍스트→숫자 매핑 (2024.09.26 형식 변경 대응)

    Returns:
        {iso3: DataFrame} dict
    """
    out_dir = PROCESSED_DIR / "acled"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    files = sorted(raw_dir.glob("*.parquet"))

    for f in files:
        iso3 = f.stem
        df = pd.read_parquet(f)

        if df.empty:
            continue

        # UTC 보장
        if not hasattr(df["event_date"].dt, "tz") or df["event_date"].dt.tz is None:
            df["event_date"] = pd.to_datetime(df["event_date"], utc=True)

        # 중복 제거
        before = len(df)
        df = df.drop_duplicates(subset=["event_id_cnty"], keep="last")
        dupes = before - len(df)
        if dupes > 0:
            print(f"  {iso3}: {dupes}건 중복 제거")

        # fatalities 정제
        df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0).astype(int)
        df.loc[df["fatalities"] < 0, "fatalities"] = 0

        # inter1/inter2: 텍스트 → 숫자 매핑 (2024.09.26 형식 변경)
        inter_map = {
            "State forces": 1, "Rebel groups": 2,
            "Political militias": 3, "Identity militias": 4,
            "Rioters": 5, "Protesters": 6,
            "Civilians": 7, "External/Other forces": 8,
        }
        for col in ["inter1", "inter2"]:
            if col in df.columns:
                # 이미 숫자이면 변환 불필요
                if df[col].dtype == object:
                    df[col] = df[col].map(inter_map).fillna(0).astype(int)
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # 날짜순 정렬
        df = df.sort_values("event_date").reset_index(drop=True)

        out_path = out_dir / f"{iso3}.parquet"
        df.to_parquet(out_path, index=False)
        results[iso3] = df

    print(f"ACLED 전처리 완료: {len(results)}개국")
    return results


# ──────────────────────────────────────────────
# GDELT 전처리
# ──────────────────────────────────────────────

def preprocess_gdelt(raw_dir: Path = RAW_DIR / "gdelt") -> dict[str, pd.DataFrame]:
    """
    GDELT 국가별 parquet → 정제.

    처리:
      - event_date UTC 확인
      - GLOBALEVENTID 기준 중복 제거 (~20% 중복)
      - GoldsteinScale 범위 검증 (-10 ~ +10)
      - AvgTone 범위 검증 (-100 ~ +100)
      - QuadClass 유효값 검증 (1-4)

    Returns:
        {iso3: DataFrame} dict
    """
    out_dir = PROCESSED_DIR / "gdelt"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    files = sorted(raw_dir.glob("*.parquet"))

    for f in files:
        iso3 = f.stem
        if iso3.startswith("."):
            continue

        df = pd.read_parquet(f)
        if df.empty:
            continue

        # UTC 보장
        if "event_date" in df.columns:
            if not hasattr(df["event_date"].dt, "tz") or df["event_date"].dt.tz is None:
                df["event_date"] = pd.to_datetime(df["event_date"], utc=True)

        # GLOBALEVENTID 기준 중복 제거
        before = len(df)
        df = df.drop_duplicates(subset=["GLOBALEVENTID"], keep="last")
        dupes = before - len(df)
        if dupes > 0:
            print(f"  {iso3}: {dupes}건 중복 제거 ({dupes/before*100:.1f}%)")

        # 수치 컬럼 정제
        df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
        df["AvgTone"] = pd.to_numeric(df["AvgTone"], errors="coerce")
        df["NumMentions"] = pd.to_numeric(df["NumMentions"], errors="coerce").fillna(0).astype(int)
        df["QuadClass"] = pd.to_numeric(df["QuadClass"], errors="coerce")

        # 범위 밖 → NaN
        df.loc[~df["GoldsteinScale"].between(-10, 10), "GoldsteinScale"] = np.nan
        df.loc[~df["AvgTone"].between(-100, 100), "AvgTone"] = np.nan
        df.loc[~df["QuadClass"].isin([1, 2, 3, 4]), "QuadClass"] = np.nan

        # 날짜순 정렬
        df = df.sort_values("event_date").reset_index(drop=True)

        out_path = out_dir / f"{iso3}.parquet"
        df.to_parquet(out_path, index=False)
        results[iso3] = df

    print(f"GDELT 전처리 완료: {len(results)}개국")
    return results


# ──────────────────────────────────────────────
# 경제지표 전처리
# ──────────────────────────────────────────────

def preprocess_economic(
    raw_path: Path = RAW_DIR / "economic" / "indicators.parquet",
) -> pd.DataFrame:
    """
    경제지표 parquet → 정제.

    처리:
      - 인덱스 UTC 변환 (yfinance timezone 혼재 대응)
      - STLFSI4 주간 → 일간 ffill
      - VIX/WTI/Gold/DXY 결측 ffill (공휴일 등)
      - 이상치 검증

    Returns:
        정제된 DataFrame (인덱스: date UTC, 컬럼: VIX/WTI/Gold/DXY/STLFSI4)
    """
    out_dir = PROCESSED_DIR / "economic"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(raw_path)

    # 인덱스 UTC 통일
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index = df.index.normalize()  # 시간 부분 제거 (날짜만)
    df.index.name = "date"

    # 전체 날짜 범위로 reindex (캘린더 일 기준)
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D", tz="UTC")
    df = df.reindex(full_range)
    df.index.name = "date"

    # STLFSI4: 주간 → 일간 ffill
    # 나머지 지표: 영업일 결측 ffill (주말/공휴일)
    df = df.ffill()

    # 수집 시작 이전 결측은 bfill
    df = df.bfill()

    out_path = out_dir / "indicators.parquet"
    df.to_parquet(out_path)
    print(f"경제지표 전처리 완료: {len(df)}일 × {len(df.columns)}지표 → {out_path}")

    return df


# ──────────────────────────────────────────────
# 전체 전처리 실행
# ──────────────────────────────────────────────

def preprocess_all() -> None:
    """모든 소스 전처리 실행."""
    print("=" * 60)
    print("전처리 시작")
    print("=" * 60)

    acled_dir = RAW_DIR / "acled"
    gdelt_dir = RAW_DIR / "gdelt"
    econ_path = RAW_DIR / "economic" / "indicators.parquet"

    if acled_dir.exists() and list(acled_dir.glob("*.parquet")):
        preprocess_acled()
    else:
        print("ACLED 원본 없음 — 건너뜀")

    if gdelt_dir.exists() and list(gdelt_dir.glob("*.parquet")):
        preprocess_gdelt()
    else:
        print("GDELT 원본 없음 — 건너뜀")

    if econ_path.exists():
        preprocess_economic()
    else:
        print("경제지표 원본 없음 — 건너뜀")

    print("=" * 60)
    print("전처리 완료")


if __name__ == "__main__":
    preprocess_all()
