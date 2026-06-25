# GDELT processed parquet → BigQuery Direct Load Test Log

**작성일**: 2026-06-05  
**작성자**: 김병현  
**상태**: test load 성공 — 전체 58개 업로드 미실행

---

## 1. 목적

`processed/gdelt/` 하위 ISO3별 58개 parquet 파일을 BigQuery의 **단일 테이블 하나**로 통합 적재.  
국가별로 별도 테이블을 만들지 않고, iso3 컬럼으로 국가를 구분하는 방식.  
이번 세션에서는 AFG / ETH / UKR 3개 파일로 test load를 수행하고 성공 여부를 검증.

---

## 2. 인증 정보

| 항목 | 값 |
|------|-----|
| project_id | conflict-ew-mvp-20260604 |
| client_email | conflict-bigquery@conflict-ew-mvp-20260604.iam.gserviceaccount.com |
| key file 위치 | `/api/conflict-ew-mvp-20260604-4af3cecfb588.json` (repo 외부) |

---

## 3. dataset 확인 결과

| 항목 | 결과 |
|------|------|
| dataset | conflict_ew |
| 상태 | **이미 존재** (생성 불필요) |
| location | **US** |

---

## 4. test load 대상 파일

| 파일 | 원본 크기 | 원본 rows |
|------|----------|-----------|
| AFG.parquet | 69MB | 6,254,046 |
| ETH.parquet | 18MB | 1,579,650 |
| UKR.parquet | 262MB | 23,934,414 |

---

## 5. 테이블명 확정

| 역할 | 테이블명 |
|------|---------|
| test table | `conflict-ew-mvp-20260604.conflict_ew.gdelt_processed_events_load_test` |
| final table (예정) | `conflict-ew-mvp-20260604.conflict_ew.gdelt_processed_events` |

### 구테이블 삭제 이력

`gdelt_events_load_test` (구 test table) → **삭제 완료** (2026-06-05)  
이유: 테이블명이 `gdelt_processed_events` 기준 네이밍 규칙과 불일치.

---

## 6. 발생한 문제와 해결

### 문제 1: NumArticles dtype 불일치

| 파일 | NumArticles PyArrow 타입 |
|------|------------------------|
| AFG | int64 |
| ETH | double |
| UKR | double |

**해결**: 3개 파일 모두 NumArticles / NumMentions를 `float64`로 정규화 후 `/tmp/`에 임시 parquet 저장.  
`GLOBALEVENTID`, `SQLDATE`, `QuadClass` (pandas Int64 nullable) → `int64`로 명시 변환.

### 문제 2: event_date 나노초 → BigQuery INT64 오인식

BigQuery parquet import는 `timestamp[ns]`를 INT64로 해석.  
**해결**: `datetime64[us, UTC]` (마이크로초)로 변환 후 재저장.

### 문제 3: DAY 파티셔닝 4000 파티션 한도 초과

AFG 단독으로 distinct day = 4,421개 → 한도 4,000 초과.  
**해결**: `MONTH` 파티셔닝으로 변경 (2014-01 ~ 2026-03 = 약 147개월).

---

## 7. 정규화 규칙 (로컬 → BigQuery)

| 컬럼 | 정규화 후 PyArrow 타입 | BigQuery 타입 |
|------|----------------------|--------------|
| GLOBALEVENTID | int64 | INTEGER |
| SQLDATE | int64 | INTEGER |
| ActionGeo_CountryCode | string | STRING |
| EventCode | string | STRING |
| EventRootCode | string | STRING |
| QuadClass | int64 | INTEGER |
| GoldsteinScale | double | FLOAT |
| NumMentions | **double** | **FLOAT** |
| NumArticles | **double** | **FLOAT** |
| AvgTone | double | FLOAT |
| event_date | **timestamp[us, UTC]** | **TIMESTAMP** |
| iso3 | string | STRING |

임시 파일 저장 경로: `/tmp/{ISO3}_normalized.parquet` (repo 외부)

---

## 8. 최종 load 방식

AFG → 테이블 신규 생성 (`--replace`), ETH / UKR → 동일 테이블에 순차 append (`--noreplace`).

```bash
# AFG: 테이블 생성 (MONTH 파티셔닝, 클러스터링)
bq load \
  --source_format=PARQUET \
  --time_partitioning_field=event_date \
  --time_partitioning_type=MONTH \
  --clustering_fields=iso3,EventRootCode,QuadClass \
  --replace \
  conflict-ew-mvp-20260604:conflict_ew.gdelt_processed_events_load_test \
  /tmp/AFG_normalized.parquet

# ETH: append
bq load --source_format=PARQUET --noreplace \
  conflict-ew-mvp-20260604:conflict_ew.gdelt_processed_events_load_test \
  /tmp/ETH_normalized.parquet

# UKR: append
bq load --source_format=PARQUET --noreplace \
  conflict-ew-mvp-20260604:conflict_ew.gdelt_processed_events_load_test \
  /tmp/UKR_normalized.parquet
```

---

## 9. iso3 컬럼 보존 여부

**보존됨.** iso3 컬럼이 그대로 STRING 타입으로 유지되며, 테이블에서 국가 구분 가능.

---

## 10. load 성공 여부

**성공** (3개 파일 모두 단일 테이블로 적재 완료)

---

## 11. 검증 결과

### 기본 통계

| 항목 | 값 |
|------|-----|
| total row count | 31,768,110 |
| iso3 distinct count | 3 |
| null event_date | 0 |
| null iso3 | 0 |
| min event_date | 2014-01-04 00:00:00 UTC |
| max event_date | 2026-03-31 00:00:00 UTC |

### iso3별 row count

| iso3 | rows |
|------|------|
| AFG | 6,254,046 |
| ETH | 1,579,650 |
| UKR | 23,934,414 |
| **합계** | **31,768,110** |

로컬 합계(6,254,046 + 1,579,650 + 23,934,414 = 31,768,110)와 일치 ✅

### 최종 schema

| 컬럼 | BigQuery 타입 |
|------|-------------|
| GLOBALEVENTID | INTEGER |
| SQLDATE | INTEGER |
| ActionGeo_CountryCode | STRING |
| EventCode | STRING |
| EventRootCode | STRING |
| QuadClass | INTEGER |
| GoldsteinScale | FLOAT |
| NumMentions | **FLOAT** |
| NumArticles | **FLOAT** |
| AvgTone | FLOAT |
| event_date | TIMESTAMP |
| iso3 | STRING |

### 파티셔닝 / 클러스터링

| 항목 | 설정 |
|------|------|
| partitioning type | MONTH |
| partitioning field | event_date |
| clustering fields | iso3, EventRootCode, QuadClass |

---

## 12. 전체 58개 업로드 결과 (2026-06-05 완료)

**성공.** 58개 전체 단일 final table에 적재 완료.

### iso3별 row count

| iso3 | rows | iso3 | rows | iso3 | rows |
|------|------|------|------|------|------|
| AFG | 6,254,046 | IND | 29,721,852 | SAU | 7,040,990 |
| ARM | 2,901,093 | IRN | 12,426,415 | SDN | 2,699,672 |
| AZE | 2,711,545 | IRQ | 6,654,543 | SEN | 809,927 |
| BFA | 562,586 | ISR | 25,707,930 | SLE | 351,059 |
| BGD | 4,649,535 | KEN | 3,624,892 | SOM | 1,648,040 |
| CAF | 273,791 | KGZ | 746,428 | SSD | 630,499 |
| CIV | 463,436 | LBN | 3,825,981 | SYR | 12,326,445 |
| CMR | 733,482 | LBY | 2,692,613 | TCD | 561,812 |
| COD | 441,657 | MDG | 264,185 | TGO | 228,486 |
| COL | 4,858,788 | MEX | 12,546,994 | THA | 2,557,006 |
| DZA | 1,524,376 | MLI | 1,497,660 | TJK | 433,542 |
| ECU | 1,347,503 | MMR | 1,140,203 | TUN | 1,498,254 |
| EGY | 9,899,785 | MOZ | 498,700 | TUR | 15,053,770 |
| ERI | 275,787 | NER | 737,194 | UGA | 1,647,206 |
| ETH | 1,579,650 | NGA | 15,717,842 | UKR | 23,934,414 |
| GIN | 566,347 | PAK | 11,939,695 | VEN | 6,227,846 |
| GNB | 76,665 | PHL | 6,283,561 | YEM | 3,256,116 |
| GTM | 989,940 | PSE | 2,596,038 | ZWE | 2,170,736 |
| HND | 1,001,860 | RUS | 37,573,462 | | |
| HTI | 1,067,234 | | | **합계** | **309,533,500** |
| IDN | 8,082,386 | | | | |

### 최종 검증 결과

| 항목 | 값 | 기대 | 일치 |
|------|-----|------|------|
| total rows | 309,533,500 | 309,533,500 | ✅ |
| iso3 distinct | 58 | 58 | ✅ |
| null event_date | 0 | 0 | ✅ |
| null iso3 | 0 | 0 | ✅ |
| min event_date | 2014-01-01 UTC | — | — |
| max event_date | 2026-03-31 UTC | — | — |
| NumArticles 타입 | FLOAT | FLOAT64 | ✅ |
| NumMentions 타입 | FLOAT | FLOAT64 | ✅ |
| partitioning | MONTH on event_date | MONTH | ✅ |
| clustering | iso3, EventRootCode, QuadClass | 동일 | ✅ |

### 실행 방식

- 파일별 pandas read → NumArticles/NumMentions float64, event_date timestamp[us,UTC] 정규화
- /tmp/gdelt_bq_final_normalized/ 에 임시 저장 후 bq load, 처리 즉시 삭제
- AFG(첫 번째): `--replace` + MONTH 파티셔닝 + 클러스터링으로 테이블 생성
- 나머지 57개: `--noreplace` 순차 append
- 실패: 0개

### 비용
- BigQuery load job: 무료
- 저장: ~3.56GB × $0.02/GB/월 ≈ $0.07/월

---

## 13. 현재 상태 (2026-06-05 기준)

| 항목 | 상태 |
|------|------|
| gdelt_events_load_test (구 test table) | **삭제 완료** |
| gdelt_processed_events_load_test (신 test table) | **존재, 유지** |
| gdelt_processed_events (final table) | **생성 완료, 58개국 309,533,500 rows** |
| 전체 58개 upload | **완료** |
| 모델 학습 | **미실행** |

---

## 14. 다음 단계

```
[1] gdelt_processed_events를 feature engineering에 활용
    - BigQuery SQL로 GDELT feature 집계 (국가별 daily/weekly)
    - 기존 feature parquet와 join 또는 대체

[2] ACLED-free 운영 모델 O0_clean 실행
    - gdelt_processed_events 기반 feature로 baseline 측정

[3] test table(gdelt_processed_events_load_test) 삭제 여부 결정
    - 필요 없으면 bq rm -f -t 로 삭제
```

---

*BigQuery 쿼리 실행(load job 제외)/모델 학습/test set 평가는 수행하지 않았다.*  
*최초 생성: 2026-06-05 / 최종 수정: 2026-06-05 (전체 58개 업로드 완료)*
