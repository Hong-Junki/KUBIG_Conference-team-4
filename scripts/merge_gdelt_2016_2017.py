"""raw_merged/gdelt에 새로 수집한 2016-2017 GDELT 데이터를 주입.

- 기존 52개 결측 국가: 2016-2017 비어있음 → input/raw/gdelt에서 해당 구간 추가
- 기존 6개 합성 국가(ETH/IRQ/PSE/SDN/SYR/UKR): 합성 행(음수 GLOBALEVENTID) 제거 후 실데이터로 교체
- 나머지 기간(2014-2015, 2018-2026)은 raw_merged 그대로 보존
"""
from __future__ import annotations
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_MERGED = ROOT / "input/raw_merged/gdelt"
SRC_NEW = ROOT / "input/raw/gdelt"
BACKUP = ROOT / "input/raw_merged/gdelt_backup_pre1617merge"

START, END = 20160101, 20171231


def main() -> None:
    if BACKUP.exists():
        raise SystemExit(f"백업 폴더 이미 존재 — 중단: {BACKUP}")
    SRC_MERGED.rename(BACKUP)
    SRC_MERGED.mkdir()

    rows = []
    for f_old in sorted(BACKUP.glob("*.parquet")):
        iso3 = f_old.stem
        df_old = pd.read_parquet(f_old)
        f_new = SRC_NEW / f"{iso3}.parquet"

        if not f_new.exists():
            shutil.copy(f_old, SRC_MERGED / f"{iso3}.parquet")
            rows.append((iso3, len(df_old), len(df_old), 0, 0, 0, "no_new"))
            continue

        df_new = pd.read_parquet(f_new)
        mask_old = (df_old["SQLDATE"] >= START) & (df_old["SQLDATE"] <= END)
        mask_new = (df_new["SQLDATE"] >= START) & (df_new["SQLDATE"] <= END)
        n_old = int(mask_old.sum())
        n_new = int(mask_new.sum())

        df_keep = df_old[~mask_old]
        df_inject = df_new[mask_new]
        df_merged = pd.concat([df_keep, df_inject], ignore_index=True)
        df_merged = df_merged.sort_values(["SQLDATE", "GLOBALEVENTID"]).reset_index(drop=True)
        df_merged.to_parquet(SRC_MERGED / f"{iso3}.parquet")

        neg_left = int((df_merged["GLOBALEVENTID"] < 0).sum())
        rows.append((iso3, len(df_old), len(df_merged), n_old, n_new, neg_left, "merged"))

    res = pd.DataFrame(rows, columns=["iso3", "old_rows", "new_rows", "old_1617", "new_1617", "neg_id", "status"])
    print(res.to_string(index=False))
    print()
    print(f"합계 행수: old={res.old_rows.sum():,} → new={res.new_rows.sum():,}")
    print(f"결측 52개국 (old_1617=0) 중 new_1617>0: {(res[res.old_1617 == 0].new_1617 > 0).sum()}/52")
    print(f"합성 6개국 (old_1617>0) 중 교체 완료: {(res[res.old_1617 > 0].new_1617 > 0).sum()}/6")
    print(f"음수 GLOBALEVENTID 잔존 합계: {res.neg_id.sum():,}")


if __name__ == "__main__":
    main()
