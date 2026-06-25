# GDELT Raw vs Processed Comparison

작성일: 2026-06-04  
범위: parquet metadata와 지정 4개국 head 1000행 비교만 수행. 전체 row scan, 전체 concat, BigQuery 업로드/쿼리, 모델 학습 없음.

## 1. 목적

`raw_merged/gdelt`와 `processed/gdelt`가 내용상 같은지 확인하고, BigQuery에 어떤 폴더를 우선 업로드할지 결정한다.

핵심 질문:

- 둘 다 올릴 필요가 있는가?
- 하나만 올린다면 raw와 processed 중 무엇이 모델링/운영용으로 더 적합한가?

## 2. 이전 concat 가능성 감사 요약

이전 감사 결과:

- 두 폴더 모두 58개 parquet
- 둘 다 event-level GDELT schema
- 둘 다 309,533,500 rows
- `raw_merged/gdelt` 총 용량: 약 3.53GB
- `processed/gdelt` 총 용량: 약 3.56GB
- 둘 다 `event_date`, `iso3`, `GLOBALEVENTID` 컬럼 존재
- 둘 다 ACLED/safe_ACLED/macis_se_score 컬럼 없음
- schema 차이는 numeric dtype mismatch 수준
- `processed/gdelt`도 daily feature가 아니라 event-level table
- concat 가능성 판정: dtype casting 후 concat 가능

## 3. 파일 목록/row count 비교

| 항목 | raw_merged/gdelt | processed/gdelt |
|---|---:|---:|
| parquet 파일 수 | 58 | 58 |
| 파일명 set | 동일 | 동일 |
| raw only 파일 | 없음 | - |
| processed only 파일 | - | 없음 |
| 총 row 수 | 309,533,500 | 309,533,500 |
| 파일별 row count 차이 | 없음 | 없음 |
| 파일 크기 차이 | 모든 58개 파일에서 차이 있음 | 모든 58개 파일에서 차이 있음 |

샘플 4개국 파일 크기:

| country | raw size | processed size | delta |
|---|---:|---:|---:|
| AFG | 71,433,173 | 72,256,867 | +823,694 |
| ETH | 18,068,371 | 18,465,245 | +396,874 |
| UKR | 273,366,143 | 274,438,037 | +1,071,894 |
| ZWE | 24,958,574 | 25,302,633 | +344,059 |

해석:

- row count는 완전히 동일하다.
- 파일 크기는 모두 processed가 더 크다.
- 저장 포맷/dtype/order 차이 또는 전처리 후 parquet encoding 차이가 있다.

## 4. schema 비교

두 폴더 모두 schema variant는 2개다.

공통 column:

| column | role |
|---|---|
| `GLOBALEVENTID` | event id |
| `SQLDATE` | GDELT integer date |
| `ActionGeo_CountryCode` | GDELT/FIPS country code |
| `EventCode` | GDELT event code |
| `EventRootCode` | GDELT root event code |
| `QuadClass` | GDELT quad class |
| `GoldsteinScale` | Goldstein score |
| `NumMentions` | mentions count |
| `NumArticles` | article count |
| `AvgTone` | average tone |
| `event_date` | UTC timestamp |
| `iso3` | ISO3 country |

Schema variants:

| folder | variant count | dtype mismatch |
|---|---:|---|
| raw_merged/gdelt | 2 | `NumMentions`: int64/double, `NumArticles`: int64/double |
| processed/gdelt | 2 | `NumArticles`: int64/double |

Cross-folder type comparison:

- `NumMentions` differs by folder-level type set.
  - raw: int64 and double
  - processed: int64
- `NumArticles` differs inside both folders.

No ACLED/safe_ACLED/macis_se_score columns were found in either folder.

## 5. 샘플 국가별 raw vs processed 비교

샘플 국가:

- AFG
- ETH
- UKR
- ZWE

각 국가에서 raw와 processed 각각 head 1000행만 읽었다.

### AFG

| check | result |
|---|---|
| row count | same: 6,254,046 |
| file size | processed +823,694 bytes |
| head1000 `GLOBALEVENTID` set | different |
| head1000 `event_date` min/max | same: 2014-02-18 ~ 2014-03-25 |
| `iso3` values | both `AFG` |
| equal columns | `SQLDATE`, `ActionGeo_CountryCode`, `event_date`, `iso3` |
| different columns | `GLOBALEVENTID`, `EventCode`, `EventRootCode`, `QuadClass`, `GoldsteinScale`, `NumMentions`, `NumArticles`, `AvgTone` |

### ETH

| check | result |
|---|---|
| row count | same: 1,579,650 |
| file size | processed +396,874 bytes |
| head1000 `GLOBALEVENTID` set | different |
| head1000 `event_date` min/max | same: 2014-01-04 ~ 2014-11-01 |
| `iso3` values | both `ETH` |
| equal columns | `SQLDATE`, `ActionGeo_CountryCode`, `event_date`, `iso3` |
| different columns | `GLOBALEVENTID`, `EventCode`, `EventRootCode`, `QuadClass`, `GoldsteinScale`, `NumMentions`, `NumArticles`, `AvgTone` |

### UKR

| check | result |
|---|---|
| row count | same: 23,934,414 |
| file size | processed +1,071,894 bytes |
| head1000 `GLOBALEVENTID` set | different |
| head1000 `event_date` min/max | same: 2014-01-07 ~ 2014-03-05 |
| `iso3` values | both `UKR` |
| equal columns | `SQLDATE`, `ActionGeo_CountryCode`, `event_date`, `iso3` |
| different columns | `GLOBALEVENTID`, `EventCode`, `EventRootCode`, `QuadClass`, `GoldsteinScale`, `NumMentions`, `NumArticles`, `AvgTone` |

### ZWE

| check | result |
|---|---|
| row count | same: 2,170,736 |
| file size | processed +344,059 bytes |
| head1000 `GLOBALEVENTID` set | different |
| head1000 `event_date` min/max | same: 2014-02-19 ~ 2014-05-03 |
| `iso3` values | both `ZWE` |
| equal columns | `SQLDATE`, `ActionGeo_CountryCode`, `event_date`, `iso3` |
| different columns | `GLOBALEVENTID`, `EventCode`, `EventRootCode`, `QuadClass`, `GoldsteinScale`, `NumMentions`, `NumArticles`, `AvgTone` |

## 6. 차이가 있는 컬럼

샘플 head 1000 기준 차이가 있었던 컬럼:

- `GLOBALEVENTID`
- `EventCode`
- `EventRootCode`
- `QuadClass`
- `GoldsteinScale`
- `NumMentions`
- `NumArticles`
- `AvgTone`

샘플 head 1000 기준 같았던 컬럼:

- `SQLDATE`
- `ActionGeo_CountryCode`
- `event_date`
- `iso3`

주의:

- 이 결과는 head 1000 비교다.
- 전체 row set이 다른지, 정렬 차이 때문에 head set이 다른지는 전체 scan 없이는 확정하지 않았다.
- 다만 생성 로직상 processed는 raw의 단순 복사본이 아니다.

## 7. raw와 processed의 관계 판정

판정: **내용 차이 있음: processed를 모델링용으로 우선 업로드**

근거:

1. 두 폴더의 파일명 set과 row count는 완전히 같다.
2. 샘플 4개국 head 1000에서 `GLOBALEVENTID` set이 모두 다르다.
3. 주요 event attribute 컬럼도 head 1000 기준 다르다.
4. `src/process/preprocess.py`의 `preprocess_gdelt`는 다음 처리를 수행한다.
   - `event_date` UTC 보장
   - `GLOBALEVENTID` 기준 중복 제거
   - `GoldsteinScale`, `AvgTone`, `NumMentions`, `QuadClass` numeric coercion
   - `GoldsteinScale`, `AvgTone`, `QuadClass` 범위 검증
   - `event_date` 기준 정렬
5. row count가 같은 것은 현재 raw에서 제거될 중복이 없었거나, merge 단계에서 이미 정리됐기 때문일 수 있다.
6. processed는 모델링 전처리 규칙이 적용된 산출물로 보는 것이 안전하다.

## 8. BigQuery 업로드 대상 추천

기본 추천:

- `processed/gdelt`만 업로드한다.
- raw와 processed를 둘 다 올리는 것은 기본적으로 비추천한다.

이유:

- 두 폴더 모두 3억 row, 3.5GB급으로 크다.
- row count와 coverage는 동일해 보인다.
- processed는 raw에 대한 모델링용 전처리 산출물이다.
- 둘 다 올리면 저장 비용과 테이블 관리 비용만 증가할 가능성이 높다.
- 원본 보존 목적이 아니라면 raw_merged까지 BigQuery에 둘 필요는 낮다.

## 9. 추천 테이블명

선택지:

| 선택 | table | 판단 |
|---|---|---|
| raw 업로드 | `conflict_ew.gdelt_raw_merged` | 원본성은 높지만 모델링용 우선순위는 낮음 |
| processed 업로드 | `conflict_ew.gdelt_processed_events` | 명확하지만 downstream 이름이 다소 길다 |
| 통합 이름 | `conflict_ew.gdelt_events` | 추천. 하나만 올릴 경우 가장 실용적 |

최종 추천:

- source: `processed/gdelt`
- BigQuery table: `project-7181e301-a6fb-46f2-871.conflict_ew.gdelt_events`

보조 metadata:

- table description에 `source_folder=processed/gdelt` 명시
- raw 보존 필요성이 생기면 별도 `gdelt_raw_merged`를 나중에 업로드

## 10. 왜 둘 다 올리지 않는지

둘 다 올리지 않는 이유:

1. 두 폴더 모두 동일 파일 수, 동일 row 수다.
2. 둘 다 event-level GDELT이며 daily aggregate feature가 아니다.
3. 저장 비용과 관리 복잡도가 두 배가 된다.
4. downstream 쿼리에서 raw/processed 중 어느 것을 써야 하는지 혼동이 생긴다.
5. 모델링/운영 목적에서는 전처리된 `processed/gdelt`가 더 적합하다.

둘 다 올릴 수 있는 예외:

- 원본 재현성/audit 목적이 강한 경우
- processed 생성 로직을 BigQuery에서 재검증해야 하는 경우
- raw와 processed 전체 event set 차이를 정밀 분석해야 하는 경우

현재 목적은 ACLED-free 운영 모델 재구축이므로 예외에 해당하지 않는다.

## 11. 다음 단계

1. `processed/gdelt`를 업로드 source로 확정한다.
2. 업로드 전 `NumArticles` dtype을 `FLOAT64`로 통일할지, BigQuery explicit schema로 처리할지 결정한다.
3. GCS wildcard load 방식을 기준으로 업로드 절차를 만든다.
4. table name은 `conflict_ew.gdelt_events`를 우선 사용한다.
5. partition은 `event_date`, clustering은 `iso3`, `EventRootCode`, `QuadClass`를 권장한다.
6. 업로드 후 row count와 schema만 검증한다. 모델 학습/test 평가는 별도 단계에서 수행한다.
