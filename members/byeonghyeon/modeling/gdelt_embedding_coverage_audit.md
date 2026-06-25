# GDELT Embedding Coverage Audit

작성일: 2026-06-04  
범위: 로컬 레포 파일 메타데이터/스키마 점검만 수행. BigQuery 쿼리, 신규 다운로드, 모델 학습, test 평가 없음.

## 1. 확인한 embedding 후보 파일 목록

검색 대상:
- `members/`
- `input/`
- `outputs/` 없음
- `data/` 없음
- `embeddings/` 없음
- `processed/` 없음

검색 키워드:
- `embedding`, `embed`, `title_embedding`, `gdelt_embedding`, `gkg_embedding`, `sentence`, `vector`, `cosine`, `e5`, `parquet`, `npy`, `pkl`, `csv`

명시적 embedding 산출물:
- 없음
- `.npy`, `.pkl` embedding/vector 파일 없음
- `embedding`, `embed`, `e5`, `cosine`, `vector`가 파일명 또는 컬럼명에 들어간 GDELT title embedding 산출물 없음

GDELT title 관련 후보:

| 파일 | 크기 | 형식 | row | column | 판정 |
|---|---:|---|---:|---:|---|
| `members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_features.parquet` | 16,910,161 bytes | parquet | 214,257 | 23 | country-day aggregated non-embedding feature |
| `members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_c2_features.parquet` | 25,381,008 bytes | parquet | 214,257 | 21 | country-day aggregated non-embedding feature |
| `members/byeonghyeon/input/processed/gdelt_titles/gdelt_theme_person_features.parquet` | 15,396,009 bytes | parquet | 214,257 | 24 | country-day aggregated non-embedding feature |
| `input/raw/gdelt_titles/{ISO3}/{YYYY-MM}.parquet` | 1,184,906,610 bytes total | parquet | 8,311,094 total | 9 | title-level raw text, not embedding |

## 2. 각 파일의 구조 요약

### `gdelt_title_features.parquet`

- columns 일부: `date`, `country`, `gdelt_title_count_1d`, `gdelt_title_nonnull_count_1d`, `gdelt_title_tone_mean_1d`, `gdelt_title_tone_std_1d`, `gdelt_title_tone_min_1d`, `gdelt_title_negative_count_1d`, `gdelt_title_positive_count_1d`, `gdelt_title_eng_count_1d`, `gdelt_title_domain_diversity_1d`, `gdelt_title_lang_diversity_1d`, `gdelt_title_count_7d`
- date column: 있음 (`date`)
- country column: 있음 (`country`)
- title/text column: 원문 없음. 집계 컬럼명에 `title`은 있으나 텍스트 컬럼은 아님
- embedding column: 없음
- embedding 형태: 해당 없음

### `gdelt_title_c2_features.parquet`

- columns 일부: `date`, `country`, `gdelt_title_count_3d`, `gdelt_title_nonnull_count_3d`, `gdelt_title_negative_count_3d`, `gdelt_title_positive_count_3d`, `gdelt_title_eng_count_3d`, `gdelt_title_tone_mean_3d`, `gdelt_title_tone_min_3d`, `gdelt_title_domain_diversity_3d`, `gdelt_title_lang_diversity_3d`, `gdelt_title_count_3d_vs_7d`
- date column: 있음 (`date`)
- country column: 있음 (`country`)
- title/text column: 원문 없음
- embedding column: 없음
- embedding 형태: 해당 없음

### `gdelt_theme_person_features.parquet`

- columns 일부: `date`, `country`, `gdelt_theme_nonnull_count_1d`, `gdelt_person_nonnull_count_1d`, `gdelt_theme_count_1d`, `gdelt_person_count_1d`, `gdelt_theme_conflict_count_1d`, `gdelt_theme_protest_count_1d`, `gdelt_theme_military_count_1d`
- date column: 있음 (`date`)
- country column: 있음 (`country`)
- title/text column: 없음
- embedding column: 없음
- embedding 형태: 해당 없음

### `input/raw/gdelt_titles/{ISO3}/{YYYY-MM}.parquet`

- total files: 116
- total rows: 8,311,094
- months: `2022-01`, `2022-02`
- columns: `date`, `iso3`, `title`, `url`, `domain`, `language`, `sourcecountry`, `seendate`, `v2tone_avg`
- date column: 있음 (`date`, `seendate`)
- country column: 있음 (`iso3`, `sourcecountry`)
- title/text column: 있음 (`title`)
- embedding column: 없음
- embedding 형태: 해당 없음

## 3. title-level인지 country-day 평균인지 판정

| 파일 | 분류 | 근거 |
|---|---|---|
| `gdelt_title_features.parquet` | 기타: country-day aggregated non-embedding feature | `date`, `country` 1행 단위. count/tone/domain/lang rolling feature만 있음 |
| `gdelt_title_c2_features.parquet` | 기타: country-day aggregated non-embedding feature | `date`, `country` 1행 단위. 3d/7d/14d 비율, country baseline z/ratio만 있음 |
| `gdelt_theme_person_features.parquet` | 기타: country-day aggregated non-embedding feature | `date`, `country` 1행 단위. theme/person count만 있음 |
| `input/raw/gdelt_titles/{ISO3}/{YYYY-MM}.parquet` | title-level raw text | `title`, `url`, `domain`, `language`가 남아 있으나 embedding/cosine 없음 |

현재 로컬에는 다음 분류에 해당하는 파일이 없음:
- title-level embedding
- country-day average embedding
- country-day aggregated cosine feature

## 4. date/country/title/embedding column 존재 여부

| 파일 | date | country | title/text | embedding | cosine/score |
|---|---|---|---|---|---|
| `gdelt_title_features.parquet` | 있음 | 있음 | 원문 없음 | 없음 | 없음 |
| `gdelt_title_c2_features.parquet` | 있음 | 있음 | 원문 없음 | 없음 | 없음 |
| `gdelt_theme_person_features.parquet` | 있음 | 있음 | 없음 | 없음 | 없음 |
| `input/raw/gdelt_titles/{ISO3}/{YYYY-MM}.parquet` | 있음 | 있음 | 있음 | 없음 | 없음 |

## 5. 58개국 coverage 확인 결과

expected 58 countries는 `members/byeonghyeon/input/processed/labels/y_escalation_7d_labels.parquet`의 country 목록 기준으로 비교했다.

expected countries:

```text
AFG, ARM, AZE, BFA, BGD, CAF, CIV, CMR, COD, COL, DZA, ECU, EGY, ERI, ETH, GIN, GNB, GTM, HND, HTI, IDN, IND, IRN, IRQ, ISR, KEN, KGZ, LBN, LBY, MDG, MEX, MLI, MMR, MOZ, NER, NGA, PAK, PHL, PSE, RUS, SAU, SDN, SEN, SLE, SOM, SSD, SYR, TCD, TGO, THA, TJK, TUN, TUR, UGA, UKR, VEN, YEM, ZWE
```

| 파일 | country unique | expected 58과 비교 | 완전히 빠진 country |
|---|---:|---|---|
| `gdelt_title_features.parquet` | 58 | 비교 가능, 전체 country 존재 | 없음 |
| `gdelt_title_c2_features.parquet` | 58 | 비교 가능, 전체 country 존재 | 없음 |
| `gdelt_theme_person_features.parquet` | 58 | 비교 가능, 전체 country 존재 | 없음 |
| `input/raw/gdelt_titles/{ISO3}/{YYYY-MM}.parquet` | 58 | 비교 가능, 전체 country directory 존재 | 없음 |

## 6. 기간 coverage 확인 결과

| 파일 | date min | date max | n_dates | 비고 |
|---|---|---|---:|---|
| `gdelt_title_features.parquet` | 2015-02-17 | 2025-03-31 | 3,695 | 2015-02-17만 5개국, 이후 3,694일은 58개국 |
| `gdelt_title_c2_features.parquet` | 2015-02-17 | 2025-03-31 | 3,695 | 동일 |
| `gdelt_theme_person_features.parquet` | 2015-02-17 | 2025-03-31 | 3,695 | 동일 |
| `input/raw/gdelt_titles/{ISO3}/{YYYY-MM}.parquet` | 파일명 기준 2022-01 | 파일명 기준 2022-02 | 2 months | 로컬 raw backup은 2022년 1-2월만 있음 |

date별 country 수 분포:

| 파일 | min | q25 | median | q75 | max |
|---|---:|---:|---:|---:|---:|
| `gdelt_title_features.parquet` | 5 | 58 | 58 | 58 | 58 |
| `gdelt_title_c2_features.parquet` | 5 | 58 | 58 | 58 | 58 |
| `gdelt_theme_person_features.parquet` | 5 | 58 | 58 | 58 | 58 |

country별 date 수 분포:

| 파일 | min | q25 | median | q75 | max |
|---|---:|---:|---:|---:|---:|
| `gdelt_title_features.parquet` | 3,694 | 3,694 | 3,694 | 3,694 | 3,695 |
| `gdelt_title_c2_features.parquet` | 3,694 | 3,694 | 3,694 | 3,694 | 3,695 |
| `gdelt_theme_person_features.parquet` | 3,694 | 3,694 | 3,694 | 3,694 | 3,695 |

시작일 `2015-02-17`에 존재하는 국가는 `IND`, `ISR`, `LBY`, `THA`, `TUR` 5개뿐이다. 나머지 53개국은 `2015-02-18`부터 존재한다.

## 7. missing country-date 문제

processed GDELT feature 3개:
- 실제 country-date pair: 214,257
- 관측 date range 기준 기대 pair: 58 countries x 3,695 dates = 214,310
- missing pair: 53
- missing ratio: 0.0247%
- 문제 위치: `2015-02-17` 하루에만 53개 country missing

raw GDELT title local backup:
- country-month pair: 116
- 기대 country-month pair: 58 countries x 2 months = 116
- month-level missing ratio: 0%
- 단, 기간 자체가 2022-01~2022-02로 제한되어 운영 feature 전체 기간에는 부족하다.

## 8. E5a/E2/E1/E3/E4/E5b 가능 여부

현재 로컬 파일 기준 판정:

| Feature | 가능 여부 | 이유 |
|---|---|---|
| E5a 제목 단위 극값 pooling | 불가 | title-level embedding 또는 title-level cosine 없음 |
| E2 임계초과 이벤트 카운트 | 불가 | title-level score/cosine 없음 |
| E1 평균 cosine 시계열화 | 불가 | country-day mean cosine 없음 |
| E3 단일 위험 강도 스칼라 | 불가 | anchor별 cosine feature 없음 |
| E4 국가별 baseline 편차 | 불가 | country-day embedding/cosine score 없음. 단, 기존 `gdelt_title_c2_features`에는 non-embedding count/tone 기반 country baseline feature가 있음 |
| E5b 다국어 lexicon | 제한적 가능 | raw title text는 2022-01~2022-02 로컬 backup에만 있음. 전체 운영 기간 적용에는 추가 원문 title 확보 필요 |

## 9. 추가 BigQuery 수집이 필요한지 여부

필요하다.

이유:
- 현재 로컬에는 GDELT title embedding 파일이 없다.
- title-level embedding, title-level cosine, country-day average embedding, country-day mean cosine 중 어느 것도 없다.
- processed GDELT title feature는 country-day count/tone/domain/lang 집계이며, embedding 기반 ACLED-like feature의 직접 재료가 아니다.
- raw title text는 로컬에 2022-01~2022-02만 있으므로 전체 train/operation 기간의 E5b 또는 embedding 생성에는 부족하다.

단, BigQuery를 바로 실행하기 전 확인할 것:
- 팀원이 만든 embedding 산출물이 레포 밖 로컬 경로, 공유 드라이브, DVC/artifact storage, 또는 gitignore된 디렉터리에 있는지 확인
- 파일명/컬럼명이 `embedding`이 아닌 `sentence`, `e5`, `vec`, `emb_0`, `dim_0`, `sim`, `anchor_score` 등으로 저장되었는지 확인
- 산출물이 title-level인지, 이미 country-day 집계된 것인지 명확히 확인

## 10. 다음 단계 제안

1. 팀원에게 embedding 산출물의 실제 경로와 스키마를 확인한다.
2. 최소 요구 스키마를 고정한다.
   - title-level: `date`, `country` 또는 `iso3`, `title` 또는 `url`, `embedding` 또는 `emb_0...emb_n`
   - cosine-level: `date`, `country`, `title` 또는 `url`, anchor별 cosine/score
   - country-day-level: `date`, `country`, anchor별 mean/max/p95/count_above_threshold
3. title-level embedding이 있으면 E5a/E2를 먼저 만든다.
4. country-day mean embedding만 있으면 E5a/E2는 포기하거나 제한적 feature로 분리하고, E1/E3/E4에 맞는 cosine score를 추가 생성한다.
5. title text가 없는 embedding이면 운영 feature로 바로 쓰기 전에 `date-country-title/url` join key 보존 여부를 확인한다.
6. 추가 BigQuery 수집이 필요하면, 모델 학습 전에 별도 수집/embedding 생성 계획서와 비용 추정을 먼저 작성한다.
