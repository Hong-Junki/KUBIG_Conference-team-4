"""
피처 + 라벨 결합 → 최종 데이터셋 생성 + 시간 기반 split

입력:
  input/processed/features/features.parquet  — 54 피처 + (date, country)
  input/processed/labels/labels.parquet       — y, y_onset, y_escalation + 보조 컬럼

출력:
  input/processed/dataset/full.parquet   — 전체 (피처+라벨)
  input/processed/dataset/train.parquet  — ~2023-12-31
  input/processed/dataset/val.parquet    — 2024-01-01 ~ 2024-06-30
  input/processed/dataset/test.parquet   — 2024-07-01 ~

피처 미래 누수 방지를 위해 meta 컬럼은 train.py의 split_dataset에서 제외 처리됨.

사용법:
  python -m src.process.build_dataset
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

FEATURES_PATH = Path("input/processed/features/features.parquet")
LABELS_PATH = Path("input/processed/labels/labels.parquet")
DATASET_DIR = Path("input/processed/dataset")

# Split 경계 (model-spec.md와 일치). raw_merged 8년+ 합본 반영.
# test_end는 ACLED 라벨 마지막 가용일(2025-03-31 - 마지막 3일 제거 = 2025-03-28).
SPLIT = {
    "train_start": date(2014, 1, 1),
    "train_end":   date(2023, 12, 31),
    "val_end":     date(2024, 6, 30),
    "test_end":    date(2025, 3, 28),
}


def build_dataset() -> pd.DataFrame:
    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)

    # 날짜/국가 키 정규화
    features["date"] = pd.to_datetime(features["date"], utc=True)
    labels["date"] = pd.to_datetime(labels["date"], utc=True)

    # 피처 ↔ 라벨 inner join (라벨이 마지막 3일 제거되어 작음)
    dataset = features.merge(labels, on=["date", "country"], how="inner")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_DIR / "full.parquet", index=False)

    # 시간 기반 split
    d = dataset["date"].dt.date
    train = dataset[(d >= SPLIT["train_start"]) & (d <= SPLIT["train_end"])]
    val = dataset[(d > SPLIT["train_end"]) & (d <= SPLIT["val_end"])]
    test = dataset[(d > SPLIT["val_end"]) & (d <= SPLIT["test_end"])]

    train.to_parquet(DATASET_DIR / "train.parquet", index=False)
    val.to_parquet(DATASET_DIR / "val.parquet", index=False)
    test.to_parquet(DATASET_DIR / "test.parquet", index=False)

    # 통계
    print(f"=== 데이터셋 생성 완료 ===")
    print(f"  full : {len(dataset):>6}행  ({dataset['date'].min().date()} ~ {dataset['date'].max().date()})")
    print(f"  train: {len(train):>6}행  ~ {SPLIT['train_end']}")
    print(f"  val  : {len(val):>6}행  {SPLIT['train_end']} ~ {SPLIT['val_end']}")
    print(f"  test : {len(test):>6}행  > {SPLIT['val_end']}")

    print(f"\n  타겟별 양성 비율:")
    for tgt in ["y", "y_onset", "y_escalation"]:
        if tgt not in dataset.columns:
            continue
        tr = train[tgt].mean()
        va = val[tgt].mean()
        te = test[tgt].mean()
        print(f"    {tgt:>14}: train {tr*100:5.2f}% / val {va*100:5.2f}% / test {te*100:5.2f}%")

    return dataset


if __name__ == "__main__":
    build_dataset()
