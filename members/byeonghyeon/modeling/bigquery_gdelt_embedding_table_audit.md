# BigQuery GDELT Embedding Table Audit

작성일: 2026-06-04  
대상:

- `project-7181e301-a6fb-46f2-871.conflict_ew.gdelt_titles`
- `project-7181e301-a6fb-46f2-871.conflict_ew.gkg_embeddings`

## 1. BigQuery project/dataset/table 확인 결과

로컬 환경에서 `gcloud`와 `bq` CLI가 모두 발견되지 않았다.

```text
gcloud: command not found
bq: command not found
```

Python `google-cloud-bigquery` client는 설치되어 있어 metadata 호출을 시도했지만, Application Default Credentials가 없어 중단됐다.

```text
ERROR_TYPE DefaultCredentialsError
ERROR Your default credentials were not found.
```

따라서 현재 세션에서는 project id 유효성 확인 단계까지 도달하지 못했고, 다음 항목을 실제 확인하지 못했다.

- active account
- active project
- `project-7181e301-a6fb-46f2-871` 프로젝트 접근 가능 여부
- `conflict_ew` dataset 존재 여부
- `gdelt_titles`, `gkg_embeddings` table 존재 여부
- schema, row count metadata, table size, location, partitioning, clustering

필요한 값/도구:

- Google Cloud SDK 또는 `bq` CLI
- 인증된 Google account
- Application Default Credentials 또는 서비스 계정 key
- 접근 가능한 정확한 project id
- `conflict_ew` dataset read 권한
- `gdelt_titles`, `gkg_embeddings` table metadata read 권한

## 2. gdelt_titles schema 요약

미확인. `bq show --format=prettyjson project-7181e301-a6fb-46f2-871:conflict_ew.gdelt_titles` 실행이 필요하다.

기대 확인 컬럼:

- date column: `date` 또는 equivalent timestamp/date
- country column: `iso3` 또는 `country`
- title/text column: `title`
- url/document id: `url` 또는 문서 식별자
- language/domain/tone/theme/person 컬럼 존재 여부

## 3. gkg_embeddings schema 요약

미확인. `bq show --format=prettyjson project-7181e301-a6fb-46f2-871:conflict_ew.gkg_embeddings` 실행이 필요하다.

기대 확인 컬럼:

- date column
- country/iso3 column
- title/text/url/document id
- embedding column
- cosine/score column
- anchor id/name column

## 4. embedding이 title-level인지 country-day-level인지 판정

현재는 판정 불가.

판정 기준:

- title-level: `date`, `country/iso3`, `title` 또는 `url`, `embedding`이 row마다 존재
- country-day-level: `date`, `country/iso3` 조합당 1행이고 평균 embedding 또는 score만 존재
- aggregated cosine: `date`, `country/iso3`, anchor별 cosine 평균/max/p95/count가 존재

## 5. date/country/title/url/embedding column 존재 여부

미확인.

## 6. join key 가능성

미확인.

우선순위:

1. `url` 또는 GKG document id
2. `date + iso3 + url`
3. `date + iso3 + title`
4. `date + iso3` only는 title-level join key로는 불충분

## 7. 58개국 coverage 확인에 필요한 쿼리

실제 실행 금지. 먼저 dry-run으로 bytes를 확인한다.

```sql
SELECT
  COUNT(DISTINCT iso3) AS n_countries,
  ARRAY_AGG(DISTINCT iso3 ORDER BY iso3 LIMIT 100) AS countries
FROM `project-7181e301-a6fb-46f2-871.conflict_ew.gkg_embeddings`
WHERE date BETWEEN DATE '2014-01-01' AND DATE '2025-03-31';
```

country-date coverage:

```sql
SELECT
  date,
  COUNT(DISTINCT iso3) AS n_countries
FROM `project-7181e301-a6fb-46f2-871.conflict_ew.gkg_embeddings`
WHERE date BETWEEN DATE '2014-01-01' AND DATE '2025-03-31'
GROUP BY date;
```

## 8. date range 확인에 필요한 쿼리

실제 실행 금지. 먼저 partition metadata와 dry-run을 확인한다.

```sql
SELECT
  MIN(date) AS min_date,
  MAX(date) AS max_date,
  COUNT(*) AS n_rows
FROM `project-7181e301-a6fb-46f2-871.conflict_ew.gkg_embeddings`;
```

## 9. 예상 비용을 줄이기 위한 dry-run 예시

```bash
bq query \
  --use_legacy_sql=false \
  --dry_run \
  'SELECT COUNT(*) FROM `project-7181e301-a6fb-46f2-871.conflict_ew.gkg_embeddings`
   WHERE date BETWEEN DATE "2024-01-01" AND DATE "2024-01-31"'
```

metadata 중심 확인:

```bash
bq ls --project_id=project-7181e301-a6fb-46f2-871
bq ls project-7181e301-a6fb-46f2-871:conflict_ew
bq show --format=prettyjson project-7181e301-a6fb-46f2-871:conflict_ew.gdelt_titles
bq show --format=prettyjson project-7181e301-a6fb-46f2-871:conflict_ew.gkg_embeddings
```

## 10. E5a/E2/E1/E3/E4/E5b 가능 여부

현재는 schema 미확인으로 최종 판정 불가.

잠정 기준:

| Feature | 가능 조건 |
|---|---|
| E5a 제목 단위 극값 pooling | title-level embedding 또는 title-level cosine 필요 |
| E2 임계초과 이벤트 카운트 | title-level score/cosine 필요 |
| E1 평균 cosine 시계열화 | country-day mean cosine 또는 title-level cosine 집계 필요 |
| E3 단일 위험 강도 스칼라 | anchor별 cosine 생성 가능해야 함 |
| E4 국가별 baseline 편차 | country-day score 필요 |
| E5b 다국어 lexicon | title text 필요 |

## 11. 다운로드가 필요한 최소 column 후보

schema 확인 후 최소 컬럼만 export/download해야 한다.

title-level embedding인 경우:

- `date`
- `iso3` 또는 `country`
- `url` 또는 document id
- `title`
- `language`
- `embedding` 또는 `emb_0...emb_n`

cosine/anchor score를 BigQuery에서 만들거나 저장한 경우:

- `date`
- `iso3` 또는 `country`
- `url` 또는 document id
- anchor id/name
- cosine/score

country-day 집계로 충분한 경우:

- `date`
- `iso3` 또는 `country`
- anchor별 mean/max/p95/count_above_threshold

## 12. GDELT로 ACLED 따라잡기 우선순위

1. E5a 제목 단위 극값 pooling
2. E2 임계초과 이벤트 카운트
3. E1 평균 cosine 시계열화
4. E3 단일 위험 강도 스칼라
5. E4 국가별 baseline 편차
6. E5b 다국어 lexicon

## 13. 다음 단계

1. `gcloud` 또는 `bq` CLI가 있는 환경에서 project id 유효성을 먼저 확인한다.
2. `bq show`로 두 table schema, partitioning, clustering, row count metadata를 확인한다.
3. coverage/date range 쿼리는 실제 실행 전에 dry-run bytes를 확인한다.
4. `gkg_embeddings`가 title-level이면 E5a/E2를 우선 설계한다.
5. country-day 평균만 있으면 E5a/E2는 제한적이며, E1/E3/E4 중심으로 재설계한다.
