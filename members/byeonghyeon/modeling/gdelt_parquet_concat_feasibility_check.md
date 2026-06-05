# GDELT Parquet Concat Feasibility Check

작성일: 2026-06-04  
범위: parquet metadata와 샘플 head만 확인. 전체 concat, 전체 row scan, duplicate/coverage 정밀 분석, BigQuery 업로드/쿼리 없음.

## 1. 목적

로컬 GDELT 국가별 parquet 58개 파일을 BigQuery의 단일 테이블로 올릴 수 있는지, “그대로 concat 가능한지”만 사전 점검한다.

판정 기준:

- 파일 수가 58개이고 파일명이 ISO3 형식인지
- parquet schema가 concat 가능한지
- dtype mismatch가 보정 가능한 수준인지
- key/date 컬럼이 존재하는지
- 샘플 head에서 파일명과 `iso3` 값이 일치하는지
- ACLED/safe_ACLED/SE 컬럼이 섞여 있지 않은지

## 2. 확인 대상 폴더

| 구분 | 경로 |
|---|---|
| raw_merged GDELT | `/Users/byeonghyeonkim/Desktop/공부/활동/KUBIG/26-1 Vibe Coding/conflict-early-warning/input/raw_merged/gdelt` |
| processed GDELT | `/Users/byeonghyeonkim/Desktop/공부/활동/KUBIG/26-1 Vibe Coding/conflict-early-warning/input/processed/gdelt` |

실행 중 Python 작업 확인:

- sandbox 내부 `ps aux | grep python | grep -v grep`는 권한 문제로 실패
- escalated 조회 결과, `raw_merged/gdelt` 또는 `processed/gdelt`를 읽는 오래 도는 Python 작업은 보이지 않았음
- 별도 `processed/acled` 관련 과거 조회 프로세스는 보였으나 이번 GDELT 감사 대상은 아님

## 3. raw_merged/gdelt 파일 수 / 크기 / row count metadata

| 항목 | 값 |
|---|---:|
| parquet 파일 수 | 58 |
| 파일명 ISO3 형식 | yes |
| expected 58개국 일치 | yes |
| 총 용량 | 3,532,358,102 bytes |
| 총 row 수 | 309,533,500 |
| 파일별 row min | 76,665 |
| 파일별 row median | 1,909,388 |
| 파일별 row max | 37,573,462 |
| row count 0 파일 | 없음 |

파일명 예시:

- first 5: `AFG.parquet`, `ARM.parquet`, `AZE.parquet`, `BFA.parquet`, `BGD.parquet`
- last 5: `UGA.parquet`, `UKR.parquet`, `VEN.parquet`, `YEM.parquet`, `ZWE.parquet`

row 수가 작은 파일:

| file | rows |
|---|---:|
| `GNB.parquet` | 76,665 |
| `TGO.parquet` | 228,486 |
| `MDG.parquet` | 264,185 |
| `CAF.parquet` | 273,791 |
| `ERI.parquet` | 275,787 |

## 4. processed/gdelt 파일 수 / 크기 / row count metadata

| 항목 | 값 |
|---|---:|
| parquet 파일 수 | 58 |
| 파일명 ISO3 형식 | yes |
| expected 58개국 일치 | yes |
| 총 용량 | 3,562,393,122 bytes |
| 총 row 수 | 309,533,500 |
| 파일별 row min | 76,665 |
| 파일별 row median | 1,909,388 |
| 파일별 row max | 37,573,462 |
| row count 0 파일 | 없음 |

파일명 예시:

- first 5: `AFG.parquet`, `ARM.parquet`, `AZE.parquet`, `BFA.parquet`, `BGD.parquet`
- last 5: `UGA.parquet`, `UKR.parquet`, `VEN.parquet`, `YEM.parquet`, `ZWE.parquet`

row 수가 작은 파일:

| file | rows |
|---|---:|
| `GNB.parquet` | 76,665 |
| `TGO.parquet` | 228,486 |
| `MDG.parquet` | 264,185 |
| `CAF.parquet` | 273,791 |
| `ERI.parquet` | 275,787 |

## 5. raw_merged/gdelt schema variant 결과

schema variant 수: 2

### Variant 1

- 파일 수: 52
- 대표 파일: `AFG.parquet`
- 파일 목록: `AFG`, `ARM`, `AZE`, `BFA`, `BGD`, `CAF`, `CIV`, `CMR`, `COD`, `COL`, `DZA`, `ECU`, `EGY`, `ERI`, `GIN`, `GNB`, `GTM`, `HND`, `HTI`, `IDN`, `IND`, `IRN`, `ISR`, `KEN`, `KGZ`, `LBN`, `LBY`, `MDG`, `MEX`, `MLI`, `MMR`, `MOZ`, `NER`, `NGA`, `PAK`, `PHL`, `RUS`, `SAU`, `SEN`, `SLE`, `SOM`, `SSD`, `TCD`, `TGO`, `THA`, `TJK`, `TUN`, `TUR`, `UGA`, `VEN`, `YEM`, `ZWE`

Schema:

| column | dtype |
|---|---|
| `GLOBALEVENTID` | int64 |
| `SQLDATE` | int64 |
| `ActionGeo_CountryCode` | string |
| `EventCode` | string |
| `EventRootCode` | string |
| `QuadClass` | int64 |
| `GoldsteinScale` | double |
| `NumMentions` | int64 |
| `NumArticles` | int64 |
| `AvgTone` | double |
| `event_date` | timestamp[ns, tz=UTC] |
| `iso3` | string |

### Variant 2

- 파일 수: 6
- 대표 파일: `ETH.parquet`
- 파일 목록: `ETH`, `IRQ`, `PSE`, `SDN`, `SYR`, `UKR`

Schema 차이:

| column | variant 1 | variant 2 |
|---|---|---|
| `NumMentions` | int64 | double |
| `NumArticles` | int64 | double |

그 외 column 이름과 순서는 동일하다.

## 6. processed/gdelt schema variant 결과

schema variant 수: 2

### Variant 1

- 파일 수: 52
- 대표 파일: `AFG.parquet`
- 파일 목록: `AFG`, `ARM`, `AZE`, `BFA`, `BGD`, `CAF`, `CIV`, `CMR`, `COD`, `COL`, `DZA`, `ECU`, `EGY`, `ERI`, `GIN`, `GNB`, `GTM`, `HND`, `HTI`, `IDN`, `IND`, `IRN`, `ISR`, `KEN`, `KGZ`, `LBN`, `LBY`, `MDG`, `MEX`, `MLI`, `MMR`, `MOZ`, `NER`, `NGA`, `PAK`, `PHL`, `RUS`, `SAU`, `SEN`, `SLE`, `SOM`, `SSD`, `TCD`, `TGO`, `THA`, `TJK`, `TUN`, `TUR`, `UGA`, `VEN`, `YEM`, `ZWE`

Schema:

| column | dtype |
|---|---|
| `GLOBALEVENTID` | int64 |
| `SQLDATE` | int64 |
| `ActionGeo_CountryCode` | string |
| `EventCode` | string |
| `EventRootCode` | string |
| `QuadClass` | int64 |
| `GoldsteinScale` | double |
| `NumMentions` | int64 |
| `NumArticles` | int64 |
| `AvgTone` | double |
| `event_date` | timestamp[ns, tz=UTC] |
| `iso3` | string |

### Variant 2

- 파일 수: 6
- 대표 파일: `ETH.parquet`
- 파일 목록: `ETH`, `IRQ`, `PSE`, `SDN`, `SYR`, `UKR`

Schema 차이:

| column | variant 1 | variant 2 |
|---|---|---|
| `NumArticles` | int64 | double |

`NumMentions`는 processed 쪽에서는 모든 파일이 int64다.

## 7. 샘플 파일 head 확인 결과

각 폴더에서 첫 번째, 중간, 마지막 파일만 head 5행을 읽었다.

샘플 파일:

- first: `AFG.parquet`
- middle: `MDG.parquet`
- last: `ZWE.parquet`

### raw_merged/gdelt

| sample | date column visible | country column visible | GLOBALEVENTID visible | NumMentions/NumArticles visible | file stem vs iso3 | head nulls |
|---|---|---|---|---|---|---|
| `AFG.parquet` | `event_date` | `iso3` | yes | yes | match (`AFG`) | none in checked cols |
| `MDG.parquet` | `event_date` | `iso3` | yes | yes | match (`MDG`) | none in checked cols |
| `ZWE.parquet` | `event_date` | `iso3` | yes | yes | match (`ZWE`) | none in checked cols |

### processed/gdelt

| sample | date column visible | country column visible | GLOBALEVENTID visible | NumMentions/NumArticles visible | file stem vs iso3 | head nulls |
|---|---|---|---|---|---|---|
| `AFG.parquet` | `event_date` | `iso3` | yes | yes | match (`AFG`) | none in checked cols |
| `MDG.parquet` | `event_date` | `iso3` | yes | yes | match (`MDG`) | none in checked cols |
| `ZWE.parquet` | `event_date` | `iso3` | yes | yes | match (`ZWE`) | none in checked cols |

샘플 확인 컬럼:

- `event_date`
- `iso3`
- `GLOBALEVENTID`
- `SQLDATE`
- `NumMentions`
- `NumArticles`
- `GoldsteinScale`
- `AvgTone`

## 8. dtype mismatch 컬럼

| folder | mismatch columns | detail |
|---|---|---|
| `raw_merged/gdelt` | `NumMentions`, `NumArticles` | 52 files int64, 6 files double |
| `processed/gdelt` | `NumArticles` | 52 files int64, 6 files double |

이 mismatch는 concat 불가능한 구조적 mismatch가 아니라 numeric dtype widening 문제다. concat/load 전에 `float64` 또는 BigQuery `FLOAT64`로 통일하면 된다.

## 9. concat 가능 여부 최종 판정

| folder | 판정 | 이유 |
|---|---|---|
| `raw_merged/gdelt` | B. dtype casting 후 concat 가능 | column 이름/순서 동일. `NumMentions`, `NumArticles` dtype만 int64/double mismatch |
| `processed/gdelt` | B. dtype casting 후 concat 가능 | column 이름/순서 동일. `NumArticles` dtype만 int64/double mismatch |

주의:

- “그대로 concat 가능”은 아니다. pyarrow/pandas/BigQuery load 방식에 따라 dtype mismatch가 문제를 만들 수 있으므로 schema를 명시하거나 사전 cast가 필요하다.
- 두 폴더 모두 daily aggregate feature가 아니라 event-level GDELT schema다. `processed/gdelt`도 현재 확인된 schema 기준으로는 `gdelt_daily_features`가 아니라 `gdelt_processed_events` 성격이다.

## 10. 필요한 보정 규칙

공통 보정:

1. column order를 아래 순서로 고정한다.
   - `GLOBALEVENTID`
   - `SQLDATE`
   - `ActionGeo_CountryCode`
   - `EventCode`
   - `EventRootCode`
   - `QuadClass`
   - `GoldsteinScale`
   - `NumMentions`
   - `NumArticles`
   - `AvgTone`
   - `event_date`
   - `iso3`
2. `event_date`를 timestamp 또는 date partition source로 사용한다.
3. `iso3`는 string으로 유지한다.
4. `source_file` 또는 `source_iso3_file` 컬럼 추가를 권장한다.
   - 필수는 아니지만 업로드 후 provenance/debug에 유리하다.
5. ACLED/safe_ACLED/macis_se_score 컬럼은 없음. 그대로 유지한다.

raw_merged 보정:

- `NumMentions`: int64/double mismatch -> `float64` 또는 BigQuery `FLOAT64`로 통일
- `NumArticles`: int64/double mismatch -> `float64` 또는 BigQuery `FLOAT64`로 통일

processed 보정:

- `NumArticles`: int64/double mismatch -> `float64` 또는 BigQuery `FLOAT64`로 통일
- `NumMentions`: int64 유지 가능. raw와 동일 스키마로 맞추려면 `FLOAT64`로 통일해도 됨

## 11. 추천 BigQuery 테이블명

| source folder | recommended table | note |
|---|---|---|
| `raw_merged/gdelt` | `project-7181e301-a6fb-46f2-871.conflict_ew.gdelt_raw_merged` | 원천/중간 merged event-level table |
| `processed/gdelt` | `project-7181e301-a6fb-46f2-871.conflict_ew.gdelt_processed_events` | processed지만 현재 schema는 event-level |

`processed/gdelt`를 `gdelt_daily_features`로 부르는 것은 비추천한다. 현재 파일에는 `date,country` daily aggregate feature가 아니라 `GLOBALEVENTID` 단위 event row가 들어 있다.

## 12. partition / clustering 추천

| table | partition | clustering |
|---|---|---|
| `gdelt_raw_merged` | `DATE(event_date)` | `iso3`, `EventRootCode`, `QuadClass` |
| `gdelt_processed_events` | `DATE(event_date)` | `iso3`, `EventRootCode`, `QuadClass` |

BigQuery load에서 `event_date`가 TIMESTAMP이면 ingestion schema에 따라 time partitioning field로 직접 지정 가능하다. DATE partition을 명확히 원하면 업로드 전 `date = DATE(event_date)` 컬럼을 projection으로 추가하는 방식을 고려한다.

## 13. 추천 업로드 방식

비교:

| 방식 | 장점 | 단점 | 현재 판단 |
|---|---|---|---|
| A. 로컬에서 하나의 parquet로 concat 후 `bq load` | 단일 파일 관리가 단순. schema cast를 확실히 적용 가능 | 3.5GB x 2 폴더, 3억 row라 로컬 concat 비용/시간/디스크 부담 큼 | 가능하지만 비추천 |
| B. GCS에 58개 parquet 업로드 후 wildcard load | 로컬 대형 concat 불필요. BigQuery load에 적합. 대용량/다파일에 안정적 | schema mismatch 대응을 위해 load schema 명시 필요. dtype mismatch가 자동 처리되는지 사전 테스트 필요 | 추천 |
| C. Python BigQuery client로 dataframe load | 변환 로직을 Python에서 세밀하게 제어 가능 | 3억 row를 dataframe으로 다루면 메모리/시간 부담 큼 | 비추천 |

추천:

- 1순위: GCS에 58개 parquet를 올리고 BigQuery wildcard load
- 단, `NumMentions`/`NumArticles` dtype mismatch가 있으므로 BigQuery schema를 명시하거나, 사전에 lightweight projection/cast parquet를 국가별로 다시 쓰는 방식을 고려한다.
- 한 파일로 local concat은 최종 수단이다.

## 14. 업로드 전 체크리스트

- [ ] 58개 parquet 파일 존재
- [ ] 모든 파일명이 ISO3 형식
- [ ] row count 0 파일 없음
- [ ] `event_date` 컬럼 존재
- [ ] `iso3` 컬럼 존재
- [ ] 샘플에서 파일명과 `iso3` 값 일치
- [ ] schema variant가 보정 가능한 수준
- [ ] raw: `NumMentions`, `NumArticles` dtype 통일 필요
- [ ] processed: `NumArticles` dtype 통일 필요
- [ ] ACLED/safe_ACLED/macis_se_score 컬럼 없음
- [ ] BigQuery에는 나라별 테이블 58개가 아니라 하나의 통합 테이블로 업로드
- [ ] partition by `event_date` 또는 `DATE(event_date)`
- [ ] cluster by `iso3`
- [ ] 실제 load 전 소규모 subset으로 schema load 테스트

## 15. 다음 단계

1. BigQuery/GCS 인증 상태를 확인한다.
2. raw와 processed 중 실제로 둘 다 필요한지 결정한다.
   - 둘의 row 수와 schema가 동일하므로 중복 보관 가능성이 있다.
   - 먼저 `processed/gdelt`가 raw 대비 어떤 처리 차이가 있는지 파일 생성 로직을 확인하는 것이 좋다.
3. GCS wildcard load를 기준으로 업로드 절차를 설계한다.
4. dtype mismatch를 BigQuery explicit schema로 처리할지, 국가별 cast parquet로 처리할지 결정한다.
5. 소규모 2~3개 파일로 dry-run/load test를 한 뒤 전체 58개 파일 load를 수행한다.

## bq load 초안

실행하지 말 것.

```bash
PROJECT_ID="project-7181e301-a6fb-46f2-871"
DATASET="conflict_ew"
BUCKET="gs://YOUR_BUCKET/gdelt"

bq load \
  --project_id="${PROJECT_ID}" \
  --source_format=PARQUET \
  --replace \
  --time_partitioning_field=event_date \
  --clustering_fields=iso3,EventRootCode,QuadClass \
  "${PROJECT_ID}:${DATASET}.gdelt_raw_merged" \
  "${BUCKET}/raw_merged/gdelt/*.parquet"

bq load \
  --project_id="${PROJECT_ID}" \
  --source_format=PARQUET \
  --replace \
  --time_partitioning_field=event_date \
  --clustering_fields=iso3,EventRootCode,QuadClass \
  "${PROJECT_ID}:${DATASET}.gdelt_processed_events" \
  "${BUCKET}/processed/gdelt/*.parquet"
```

dtype mismatch가 load에서 문제가 되면 explicit schema 또는 cast parquet를 사용한다.
