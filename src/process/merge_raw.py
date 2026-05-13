"""
Raw 데이터 병합 스크립트
- input/raw/ (2018-2026) + input/input 2014-2015/ + input/raw_2016-2017/ + input/raw 2020-2021/ 통합
- 출력: input/raw_merged/ (같은 구조)

주의: 2016-2017 GDELT는 원본 이벤트 포맷과 비호환(6개국 집계본)이므로 GDELT는 2016-2017 구간 결측
"""
import os
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent.parent / "input"
OUT  = BASE / "raw_merged"

# ISO numeric → ISO3
COUNTRY_ISO = {
    4:"AFG",51:"ARM",31:"AZE",854:"BFA",50:"BGD",140:"CAF",384:"CIV",120:"CMR",
    180:"COD",170:"COL",12:"DZA",218:"ECU",818:"EGY",232:"ERI",231:"ETH",
    324:"GIN",624:"GNB",320:"GTM",340:"HND",332:"HTI",360:"IDN",356:"IND",
    364:"IRN",368:"IRQ",376:"ISR",404:"KEN",417:"KGZ",422:"LBN",434:"LBY",
    450:"MDG",484:"MEX",466:"MLI",104:"MMR",508:"MOZ",562:"NER",566:"NGA",
    586:"PAK",608:"PHL",275:"PSE",643:"RUS",682:"SAU",729:"SDN",686:"SEN",
    694:"SLE",706:"SOM",728:"SSD",760:"SYR",148:"TCD",768:"TGO",764:"THA",
    762:"TJK",788:"TUN",792:"TUR",800:"UGA",804:"UKR",862:"VEN",887:"YEM",716:"ZWE",
}
ALL_COUNTRIES = sorted(COUNTRY_ISO.values())


# ──────────────────────────────────────────────
# ACLED 병합
# 소스: 2014-2015 parquet / 2016-2017 CSV / 2018-2026 parquet
# ──────────────────────────────────────────────
def merge_acled():
    out_dir = OUT / "acled"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2016-2017 CSV → dict[iso3 → DataFrame]
    csv_df = pd.read_csv(BASE / "raw_2016-2017/acled_20160101_20171231.csv")
    csv_df["event_date"] = pd.to_datetime(csv_df["event_date"], utc=True)
    csv_df["iso3_col"] = csv_df["iso"].map(COUNTRY_ISO)
    csv_df = csv_df.dropna(subset=["iso3_col"])

    # 필요한 컬럼만 (raw/ parquet과 동일하게)
    keep = ["event_id_cnty","event_date","year","disorder_type","event_type",
            "sub_event_type","actor1","inter1","actor2","inter2","iso","country",
            "admin1","latitude","longitude","fatalities","timestamp"]
    csv_df = csv_df[[c for c in keep if c in csv_df.columns] + ["iso3_col"]]
    csv_by_country = dict(tuple(csv_df.groupby("iso3_col")))

    for iso3 in ALL_COUNTRIES:
        parts = []

        # 2014-2015 parquet
        p14 = BASE / f"input 2014-2015/raw_submit_2014_2015/acled/{iso3}.parquet"
        if p14.exists():
            df = pd.read_parquet(p14)
            if len(df) > 0:
                parts.append(df)

        # 2016-2017 CSV 조각
        if iso3 in csv_by_country:
            chunk = csv_by_country[iso3].drop(columns=["iso3_col"], errors="ignore").copy()
            # CSV와 parquet 간 타입 불일치 컬럼 → parquet 기준 object(str)로 통일
            for col in ["inter1", "inter2", "latitude", "longitude"]:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype(str)
            parts.append(chunk)

        # 2020-2021 parquet
        p20 = BASE / f"raw 2020-2021/acled/{iso3}.parquet"
        if p20.exists():
            df = pd.read_parquet(p20)
            if len(df) > 0:
                parts.append(df)

        # 2018-2026 parquet
        p18 = BASE / f"raw/acled/{iso3}.parquet"
        if p18.exists():
            parts.append(pd.read_parquet(p18))

        if not parts:
            # 빈 스켈레톤 저장
            empty = pd.DataFrame(columns=keep)
            empty.to_parquet(out_dir / f"{iso3}.parquet", index=False)
            continue

        merged = pd.concat(parts, ignore_index=True)
        merged = merged.drop_duplicates(subset=["event_id_cnty"])
        merged = merged.sort_values("event_date").reset_index(drop=True)
        merged.to_parquet(out_dir / f"{iso3}.parquet", index=False)

    print(f"[ACLED] 완료 — {len(ALL_COUNTRIES)}개국")


# ──────────────────────────────────────────────
# GDELT 2016-2017 CSV → 합성 이벤트 행 변환
#
# CSV는 일별 집계 데이터(avg_goldstein, avg_tone, mentions_sum, gdelt_event_count 등)
# feature_builder는 이벤트 단위 행을 기대하므로, 일별 집계를 QuadClass별 합성 행으로 전개.
#
# 변환 규칙 (1일 → 최대 3행):
#   - QC3 (Verbal Conflict): verbal_conflict_count개 행, NumMentions=mentions_sum/gdelt_event_count
#   - QC4 (Material Conflict): material_conflict_count개 행
#   - QC1 (Cooperation 합산): 나머지 행
#   각 행: GoldsteinScale=avg_goldstein, AvgTone=avg_tone
#
# 한계: goldstein_std=0 (원본 개별 이벤트값 없음)
#       event_count는 실제 이벤트 수와 동일하게 복원됨
# ──────────────────────────────────────────────
FIPS_TO_ISO3 = {"ET": "ETH", "GZ": "PSE", "IZ": "IRQ", "SU": "SDN", "SY": "SYR", "UP": "UKR"}

# 합성 GLOBALEVENTID 시작값 (실제 GDELT ID 범위 2B 이상 → 충돌 방지용 음수)
_SYNTH_ID_START = -1_000_000_000


def _csv_gdelt_to_events(csv_path: Path) -> dict[str, pd.DataFrame]:
    """2016-2017 GDELT CSV → {iso3: 합성 이벤트 DataFrame}"""
    df = pd.read_csv(csv_path)
    df["event_date"] = pd.to_datetime(df["event_date"], utc=True)
    df["iso3"] = df["country_fips"].map(FIPS_TO_ISO3)
    df = df.dropna(subset=["iso3"])

    result = {}
    synth_id = _SYNTH_ID_START

    for iso3, grp in df.groupby("iso3"):
        rows = []
        for _, r in grp.iterrows():
            total = int(r["gdelt_event_count"]) if r["gdelt_event_count"] > 0 else 1
            mentions_per = r["mentions_sum"] / total
            articles_per = r["articles_sum"] / total

            qc3 = max(0, int(r["verbal_conflict_count"]))
            qc4 = max(0, int(r["material_conflict_count"]))
            qc1 = max(0, total - qc3 - qc4)

            for qc, cnt in [(3, qc3), (4, qc4), (1, qc1)]:
                if cnt == 0:
                    continue
                rows.append({
                    "GLOBALEVENTID":   synth_id,
                    "SQLDATE":         int(r["event_date"].strftime("%Y%m%d")),
                    "ActionGeo_CountryCode": None,
                    "EventCode":       None,
                    "EventRootCode":   None,
                    "QuadClass":       qc,
                    "GoldsteinScale":  r["avg_goldstein"],
                    "NumMentions":     mentions_per * cnt,
                    "NumArticles":     articles_per * cnt,
                    "AvgTone":         r["avg_tone"],
                    "event_date":      r["event_date"],
                    "iso3":            iso3,
                    # cnt만큼 행이 생성됐음을 표시 — GLOBALEVENTID는 행마다 고유해야 함
                    "_cnt":            cnt,
                })
                synth_id -= 1  # 행 단위로 고유 ID 부여 (cnt별 반복 아님)

        if not rows:
            continue

        # 위 rows는 QuadClass별 1행씩 — event_count 복원을 위해 cnt만큼 확장
        expanded = []
        eid = _SYNTH_ID_START - len(result) * 10_000_000  # iso3별 오프셋
        for row in rows:
            cnt = row.pop("_cnt")
            mnpp = row["NumMentions"] / cnt
            anpp = row["NumArticles"] / cnt
            for _ in range(cnt):
                new_row = row.copy()
                new_row["GLOBALEVENTID"] = eid
                new_row["NumMentions"] = mnpp
                new_row["NumArticles"] = anpp
                expanded.append(new_row)
                eid -= 1

        result[iso3] = pd.DataFrame(expanded)

    return result


def merge_gdelt():
    out_dir = OUT / "gdelt"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2016-2017 CSV → 합성 이벤트
    synth_by_country = _csv_gdelt_to_events(
        BASE / "raw_2016-2017/gdelt_20160101_20171231.csv"
    )
    print(f"  2016-2017 합성 이벤트 생성: {sum(len(v) for v in synth_by_country.values()):,}건 "
          f"({list(synth_by_country.keys())})")

    for iso3 in ALL_COUNTRIES:
        parts = []

        # 2014-2015
        p14 = BASE / f"input 2014-2015/raw_submit_2014_2015/gdelt/{iso3}.parquet"
        if p14.exists():
            df = pd.read_parquet(p14)
            if len(df) > 0:
                parts.append(df)

        # 2016-2017 합성 이벤트 (6개국만 해당)
        if iso3 in synth_by_country:
            parts.append(synth_by_country[iso3])

        # 2020-2021
        p20 = BASE / f"raw 2020-2021/gdelt/{iso3}.parquet"
        if p20.exists():
            df = pd.read_parquet(p20)
            if len(df) > 0:
                parts.append(df)

        # 2018-2026
        p18 = BASE / f"raw/gdelt/{iso3}.parquet"
        if p18.exists():
            parts.append(pd.read_parquet(p18))

        if not parts:
            empty = pd.DataFrame(columns=[
                "GLOBALEVENTID","SQLDATE","ActionGeo_CountryCode","EventCode",
                "EventRootCode","QuadClass","GoldsteinScale","NumMentions",
                "NumArticles","AvgTone","event_date","iso3"
            ])
            empty.to_parquet(out_dir / f"{iso3}.parquet", index=False)
            continue

        merged = pd.concat(parts, ignore_index=True)
        merged = merged.drop_duplicates(subset=["GLOBALEVENTID"])
        merged = merged.sort_values("event_date").reset_index(drop=True)
        merged.to_parquet(out_dir / f"{iso3}.parquet", index=False)

    print("[GDELT] 완료")


# ──────────────────────────────────────────────
# Economic 병합
# 소스: 2014-2015 parquet / 2016-2017 CSV (pivot) / 2018-2026 parquet
# ──────────────────────────────────────────────
def merge_economic():
    out_dir = OUT / "economic"
    out_dir.mkdir(parents=True, exist_ok=True)

    ECON_MAP = {"vix":"VIX","wti":"WTI","gold":"Gold","dxy":"DXY","stlfsi4":"STLFSI4"}

    # 2014-2015
    df14 = pd.read_parquet(BASE / "input 2014-2015/raw_submit_2014_2015/economic/indicators.parquet")

    # 2016-2017 CSV → wide format
    df16_long = pd.read_csv(BASE / "raw_2016-2017/economics_20160101_20171231.csv")
    df16_long["indicator_mapped"] = df16_long["indicator"].map(ECON_MAP)
    df16_long = df16_long.dropna(subset=["indicator_mapped"])
    df16 = df16_long.pivot_table(index="date", columns="indicator_mapped", values="close", aggfunc="first")
    df16.index = pd.to_datetime(df16.index, utc=True).tz_localize(None)
    df16.index.name = "date"
    # 누락 컬럼 보충
    for col in ["VIX","WTI","Gold","DXY","STLFSI4"]:
        if col not in df16.columns:
            df16[col] = float("nan")
    df16 = df16[["VIX","WTI","Gold","DXY","STLFSI4"]]

    # 2020-2021
    df20 = pd.read_parquet(BASE / "raw 2020-2021/economic/indicators.parquet")

    # 2018-2026
    df18 = pd.read_parquet(BASE / "raw/economic/indicators.parquet")

    merged = pd.concat([df14, df16, df20, df18])
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    merged.to_parquet(out_dir / "indicators.parquet")
    print(f"[Economic] 완료 — {merged.index.min().date()} ~ {merged.index.max().date()}, {len(merged)}일")


if __name__ == "__main__":
    print("=== Raw 데이터 병합 시작 ===")
    merge_acled()
    merge_gdelt()
    merge_economic()
    print("=== 완료: input/raw_merged/ ===")
