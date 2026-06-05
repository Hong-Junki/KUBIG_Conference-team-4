# BigQuery Upload Plan for ACLED-Free Model

작성일: 2026-06-04  
범위: 업로드 대상 설계와 검증 계획만 작성. 실제 BigQuery 업로드, BigQuery 쿼리, 모델 학습, test 평가 없음.

## 1. 목적

ACLED-free 운영 모델 재구축을 위해 로컬 parquet에서 BigQuery로 올릴 데이터를 feature, label, split 계층으로 분리한다.

핵심 원칙:

- `y_escalation`은 target으로 유지한다.
- ACLED는 target 생성/평가 기준으로만 사용한다.
- 운영 feature 테이블에는 ACLED 기반 feature를 넣지 않는다.
- label/source 컬럼과 운영 feature 컬럼을 물리적으로 분리한다.
- `macis_se_score`는 사용하지 않는다.
- `full.parquet`, `train.parquet`, `val.parquet`, `test.parquet`를 그대로 업로드하지 않는다.

## 2. 현재 BigQuery 상태

전달받은 현재 BigQuery 정보:

- project: `project-7181e301-a6fb-46f2-871`
- dataset: `conflict_ew`
- 확인된 테이블:
  - `gdelt_titles`
  - `gkg_embeddings`
- 경제지표는 이미 BigQuery에 올라가 있다고 들었으나, 실제 테이블명은 미확인

현재 로컬 세션에서는 `gcloud`/`bq` CLI와 ADC 인증이 없어 BigQuery metadata를 직접 확인하지 않았다. 따라서 경제지표 테이블은 업로드 후보에서 일단 제외하고, 다음 단계에서 `bq ls project-7181e301-a6fb-46f2-871:conflict_ew`로 실제 테이블명을 확인해야 한다.

## 3. 로컬 parquet 후보 목록

| path | size | rows | cols | date | country | label | ACLED/safe/SE | GDELT | economic | title/theme/person |
|---|---:|---:|---:|---|---|---|---|---:|---:|---:|
| `../conflict-early-warning/input/processed/dataset/full.parquet` | 43,948,914 | 259,260 | 64 | yes | yes | yes | ACLED 21 | 19 | 15 | 0 |
| `../conflict-early-warning/input/processed/dataset/train.parquet` | 35,992,085 | 211,816 | 64 | yes | yes | yes | ACLED 21 | 19 | 15 | 0 |
| `../conflict-early-warning/input/processed/dataset/val.parquet` | 2,148,777 | 10,556 | 64 | yes | yes | yes | ACLED 21 | 19 | 15 | 0 |
| `../conflict-early-warning/input/processed/dataset/test.parquet` | 3,137,044 | 15,718 | 64 | yes | yes | yes | ACLED 21 | 19 | 15 | 0 |
| `../conflict-early-warning/input/processed/dataset/full_se.parquet` | 14,219,233 | 68,614 | 64 | yes | yes | yes | ACLED 20 + `macis_se_score` | 19 | 15 | 0 |
| `members/byeonghyeon/input/processed/labels/y_escalation_7d_labels.parquet` | 510,727 | 259,260 | 6 | yes | yes | yes | no | 0 | 0 | 0 |
| `members/byeonghyeon/input/processed/acled_safe/safe_acled_lag_features.parquet` | 2,084,409 | 238,264 | 17 | yes | yes | no | `safe_acled_*` 15 | 0 | 0 | 0 |
| `members/byeonghyeon/input/processed/acled_safe/enhanced_safe_acled_features.parquet` | 7,505,919 | 238,264 | 40 | yes | yes | no | enhanced safe ACLED | 0 | 0 | 0 |
| `members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_features.parquet` | 16,910,161 | 214,257 | 23 | yes | yes | no | no | 21 | 0 | title 21 |
| `members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_c2_features.parquet` | 25,381,008 | 214,257 | 21 | yes | yes | no | no | 19 | 0 | title C2 19 |
| `members/byeonghyeon/input/processed/gdelt_titles/gdelt_theme_person_features.parquet` | 15,396,009 | 214,257 | 24 | yes | yes | no | no | 22 | 0 | theme/person 22 |

Coverage notes:

- `full.parquet`: 2014-01-01 ~ 2026-03-28, 58 countries, no duplicate `date,country`
- `train.parquet`: 2014-01-01 ~ 2023-12-31, 58 countries, no duplicate `date,country`
- `val.parquet`: 2024-01-01 ~ 2024-06-30, 58 countries, no duplicate `date,country`
- `test.parquet`: 2024-07-01 ~ 2025-03-28, 58 countries, no duplicate `date,country`
- GDELT title/theme files: 2015-02-17 ~ 2025-03-31, 58 countries, no duplicate `date,country`

## 4. 업로드해야 할 테이블

### A. 반드시 업로드할 테이블

1. `labels_daily`
2. `split_daily`
3. `gdelt_events_daily_features`
4. `modeling_country_day_panel`

### B. 이미 BigQuery에 있으면 업로드하지 않을 테이블

1. `economic_daily_features`

경제지표는 이미 BigQuery에 있다고 들었으므로 먼저 실제 테이블명을 확인한다. 이미 존재한다면 로컬에서 다시 업로드하지 않고 `date,country` 또는 날짜 기준 join 계획만 세운다. 존재하지 않으면 `full.parquet`에서 `econ_*` 15개만 projection해서 별도 업로드한다.

### C. 나중에 필요하면 업로드할 테이블

1. `gdelt_title_daily_features`
2. `gdelt_title_c2_daily_features`
3. `gdelt_theme_person_daily_features`
4. `gdelt_embedding_acled_like_features`

### D. 운영 feature로 업로드하면 안 되는 테이블

1. `acled_*` feature table
2. `safe_acled_*` feature table
3. enhanced safe ACLED feature table
4. `macis_se_score` 포함 table
5. label/future columns가 섞인 model feature table

## 5. 업로드하지 말아야 할 테이블

운영 feature dataset에는 아래 파일/컬럼을 그대로 올리지 않는다.

- `../conflict-early-warning/input/processed/dataset/full.parquet`
- `../conflict-early-warning/input/processed/dataset/train.parquet`
- `../conflict-early-warning/input/processed/dataset/val.parquet`
- `../conflict-early-warning/input/processed/dataset/test.parquet`
- `../conflict-early-warning/input/processed/dataset/full_se.parquet`
- `members/byeonghyeon/input/processed/acled_safe/safe_acled_lag_features.parquet`
- `members/byeonghyeon/input/processed/acled_safe/enhanced_safe_acled_features.parquet`

금지 컬럼 패턴:

- `acled_*`
- `safe_acled_*`
- `enhanced_safe_acled_*`
- `macis_se_score`
- `fatalities_next3d`
- `event_count_next3d`
- `future_*`
- `next*`
- `past14d_*` in feature tables
- `y`, `y_onset`, `y_escalation` in feature tables

## 6. 각 테이블 권장 schema

### `labels_daily`

목적: target과 label generation 관련 컬럼 보관. 운영 feature와 분리.

권장 schema:

| column | BigQuery type | mode | description |
|---|---|---|---|
| `date` | DATE | REQUIRED | country-day 기준 날짜 |
| `country` | STRING | REQUIRED | ISO3 |
| `y_escalation` | INT64 | REQUIRED | 운영 모델 target |
| `y_onset` | INT64 | NULLABLE | label diagnostic |
| `y` | INT64 | NULLABLE | 기존 label |
| `fatalities_next3d` | FLOAT64 또는 INT64 | NULLABLE | label generation only |
| `event_count_next3d` | FLOAT64 또는 INT64 | NULLABLE | label generation only |
| `past14d_event_count` | FLOAT64 또는 INT64 | NULLABLE | label generation only |
| `past14d_fatalities_mean` | FLOAT64 | NULLABLE | label generation only |

`members/byeonghyeon/input/processed/labels/y_escalation_7d_labels.parquet`를 별도 label variant로 올릴 경우:

- `y_escalation_3d`
- `y_escalation_7d`
- `future_7d_positive_count` (label generation only)
- `y_escalation_7d_missing_mask`

### `split_daily`

목적: clean validation split을 명시적으로 보관.

권장 schema:

| column | BigQuery type | mode | description |
|---|---|---|---|
| `date` | DATE | REQUIRED | country-day 기준 날짜 |
| `country` | STRING | REQUIRED | ISO3 |
| `split` | STRING | REQUIRED | `train_fit`, `tune_cal`, `val_eval`, `test` |

split 정의:

- `train_fit`: 2014-01-01 ~ 2022-12-31
- `tune_cal`: 2023-01-01 ~ 2023-12-31
- `val_eval`: 2024-01-01 ~ 2024-06-30
- `test`: 2024-07-01 이후

### `gdelt_events_daily_features`

목적: O0_clean 운영 feature의 GDELT events 계열만 보관.

권장 schema:

| column | BigQuery type |
|---|---|
| `date` | DATE |
| `country` | STRING |
| `gdelt_goldstein_mean_7d` | FLOAT64 |
| `gdelt_goldstein_mean_14d` | FLOAT64 |
| `gdelt_goldstein_mean_30d` | FLOAT64 |
| `gdelt_goldstein_std_7d` | FLOAT64 |
| `gdelt_goldstein_std_14d` | FLOAT64 |
| `gdelt_goldstein_std_30d` | FLOAT64 |
| `gdelt_tone_mean_7d` | FLOAT64 |
| `gdelt_tone_mean_14d` | FLOAT64 |
| `gdelt_tone_mean_30d` | FLOAT64 |
| `gdelt_mentions_sum_7d` | FLOAT64 |
| `gdelt_mentions_sum_14d` | FLOAT64 |
| `gdelt_mentions_sum_30d` | FLOAT64 |
| `gdelt_event_count_7d` | FLOAT64 |
| `gdelt_event_count_14d` | FLOAT64 |
| `gdelt_event_count_30d` | FLOAT64 |
| `gdelt_quadclass_1_ratio` | FLOAT64 |
| `gdelt_quadclass_2_ratio` | FLOAT64 |
| `gdelt_quadclass_3_ratio` | FLOAT64 |
| `gdelt_quadclass_4_ratio` | FLOAT64 |

### `economic_daily_features`

이미 BigQuery에 존재하면 업로드하지 않는다. 없을 때만 projection 업로드 후보.

권장 schema:

| column | BigQuery type |
|---|---|
| `date` | DATE |
| `country` | STRING |
| `econ_vix` | FLOAT64 |
| `econ_vix_pct_1d` | FLOAT64 |
| `econ_vix_pct_7d` | FLOAT64 |
| `econ_wti` | FLOAT64 |
| `econ_wti_pct_1d` | FLOAT64 |
| `econ_wti_pct_7d` | FLOAT64 |
| `econ_gold` | FLOAT64 |
| `econ_gold_pct_1d` | FLOAT64 |
| `econ_gold_pct_7d` | FLOAT64 |
| `econ_dxy` | FLOAT64 |
| `econ_dxy_pct_1d` | FLOAT64 |
| `econ_dxy_pct_7d` | FLOAT64 |
| `econ_stlfsi4` | FLOAT64 |
| `econ_stlfsi4_pct_1d` | FLOAT64 |
| `econ_stlfsi4_pct_7d` | FLOAT64 |

### `modeling_country_day_panel`

목적: modeling용 base key와 split을 제공하는 country-day panel. feature와 label은 별도 table에서 join한다.

권장 schema:

| column | BigQuery type | description |
|---|---|---|
| `date` | DATE | country-day 기준 날짜 |
| `country` | STRING | ISO3 |
| `split` | STRING | clean validation split |

선택적으로 BigQuery view를 따로 만든다.

- `modeling_country_day_panel_with_labels`: panel + labels
- `modeling_o0_features_view`: panel + GDELT events + economic
- label은 view에서 join 가능하지만, base feature table에는 넣지 않는다.

### `gdelt_title_daily_features`

나중에 O1 후보. `members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_features.parquet` 기반.

권장 schema:

- `date` DATE
- `country` STRING
- `gdelt_title_*` 21 columns FLOAT64/INT64
- optional `gdelt_title_coverage_mask` INT64

### `gdelt_theme_person_daily_features`

나중에 O2 후보. `members/byeonghyeon/input/processed/gdelt_titles/gdelt_theme_person_features.parquet` 기반.

권장 schema:

- `date` DATE
- `country` STRING
- `gdelt_theme_*`
- `gdelt_person_*`

### `gdelt_embedding_acled_like_features`

나중에 E feature 후보. 현재 로컬 projection source 없음. BigQuery `gkg_embeddings` schema 확인 후 설계 확정.

권장 최소 schema:

- `date` DATE
- `country` STRING
- anchor별 mean/max/p95 cosine
- threshold exceedance count
- title_count denominator
- coverage/missing mask

## 7. `full/train/val/test` parquet를 그대로 올리면 안 되는 이유

그대로 업로드하면 다음 문제가 생긴다.

1. 운영 feature와 label/source가 한 테이블에 섞인다.
2. `acled_*` 21개가 포함되어 운영 feature 누수 위험이 있다.
3. `y`, `y_onset`, `y_escalation`이 feature와 같은 row에 존재한다.
4. `fatalities_next3d`, `event_count_next3d`, `past14d_*`가 포함되어 future/label generation 컬럼이 feature table에 섞인다.
5. `full_se.parquet`는 `macis_se_score`를 포함한다.
6. 추후 BigQuery view 작성 시 `SELECT *`가 발생하면 금지 컬럼이 모델 입력으로 들어갈 위험이 높다.

따라서 원본 parquet를 그대로 올리지 말고 필요한 컬럼만 projection한 별도 parquet를 만든 뒤 업로드해야 한다.

## 8. projection parquet 생성 계획

실제 생성은 아직 하지 않는다. 다음 단계에서 별도 script로 생성한다.

권장 출력 경로:

```text
members/byeonghyeon/input/bq_upload/
  labels_daily.parquet
  split_daily.parquet
  gdelt_events_daily_features.parquet
  economic_daily_features.parquet        # BigQuery에 없을 때만
  modeling_country_day_panel.parquet
  gdelt_title_daily_features.parquet     # later
  gdelt_theme_person_daily_features.parquet # later
```

projection source:

- `labels_daily`: `../conflict-early-warning/input/processed/dataset/full.parquet`
- `split_daily`: `full.parquet`의 `date,country`에서 split rule 적용
- `gdelt_events_daily_features`: `full.parquet`에서 `date,country` + 19 GDELT event columns
- `economic_daily_features`: `full.parquet`에서 `date,country` + 15 `econ_*` columns, BigQuery에 없을 때만
- `modeling_country_day_panel`: `full.parquet`에서 `date,country` + split
- later title/theme: `members/byeonghyeon/input/processed/gdelt_titles/*.parquet`

## 9. 업로드 전 검증 체크리스트

공통:

- [ ] `date` dtype이 DATE로 변환 가능한지 확인
- [ ] `country`가 ISO3 문자열인지 확인
- [ ] country unique count = 58
- [ ] `date,country` 중복 없음
- [ ] row count가 기대 범위와 일치
- [ ] expected date range 확인
- [ ] null count 확인
- [ ] inf/-inf 없음
- [ ] schema와 BigQuery target schema 일치

feature table:

- [ ] `acled_*` 없음
- [ ] `safe_acled_*` 없음
- [ ] `enhanced_safe_acled_*` 없음
- [ ] `macis_se_score` 없음
- [ ] `y`, `y_onset`, `y_escalation` 없음
- [ ] `future`, `next`, `past14d` substring 컬럼 없음
- [ ] `gdelt_title_*`, `gdelt_theme_*`, `gdelt_person_*`는 O0 feature table에 없음
- [ ] embedding/cosine/vector 계열은 O0 feature table에 없음

label table:

- [ ] label generation only 컬럼이 description에 명시됨
- [ ] 운영 feature view에서 label columns가 직접 포함되지 않음

split table:

- [ ] `train_fit`, `tune_cal`, `val_eval`, `test` 값만 존재
- [ ] split boundary가 cleanval 기준과 일치

## 10. bq load 명령 초안

실제 실행하지 않는다.

```bash
PROJECT_ID="project-7181e301-a6fb-46f2-871"
DATASET="conflict_ew"

bq load \
  --project_id="${PROJECT_ID}" \
  --source_format=PARQUET \
  --time_partitioning_field=date \
  --clustering_fields=country \
  "${PROJECT_ID}:${DATASET}.labels_daily" \
  "members/byeonghyeon/input/bq_upload/labels_daily.parquet"

bq load \
  --project_id="${PROJECT_ID}" \
  --source_format=PARQUET \
  --time_partitioning_field=date \
  --clustering_fields=country \
  "${PROJECT_ID}:${DATASET}.split_daily" \
  "members/byeonghyeon/input/bq_upload/split_daily.parquet"

bq load \
  --project_id="${PROJECT_ID}" \
  --source_format=PARQUET \
  --time_partitioning_field=date \
  --clustering_fields=country \
  "${PROJECT_ID}:${DATASET}.gdelt_events_daily_features" \
  "members/byeonghyeon/input/bq_upload/gdelt_events_daily_features.parquet"

bq load \
  --project_id="${PROJECT_ID}" \
  --source_format=PARQUET \
  --time_partitioning_field=date \
  --clustering_fields=country \
  "${PROJECT_ID}:${DATASET}.modeling_country_day_panel" \
  "members/byeonghyeon/input/bq_upload/modeling_country_day_panel.parquet"
```

경제지표가 없을 때만:

```bash
bq load \
  --project_id="${PROJECT_ID}" \
  --source_format=PARQUET \
  --time_partitioning_field=date \
  --clustering_fields=country \
  "${PROJECT_ID}:${DATASET}.economic_daily_features" \
  "members/byeonghyeon/input/bq_upload/economic_daily_features.parquet"
```

## 11. Python 업로드 코드 초안

실제 실행하지 않는다.

```python
from google.cloud import bigquery

PROJECT_ID = "project-7181e301-a6fb-46f2-871"
DATASET = "conflict_ew"

client = bigquery.Client(project=PROJECT_ID)

job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.PARQUET,
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    time_partitioning=bigquery.TimePartitioning(field="date"),
    clustering_fields=["country"],
)

loads = {
    "labels_daily": "members/byeonghyeon/input/bq_upload/labels_daily.parquet",
    "split_daily": "members/byeonghyeon/input/bq_upload/split_daily.parquet",
    "gdelt_events_daily_features": "members/byeonghyeon/input/bq_upload/gdelt_events_daily_features.parquet",
    "modeling_country_day_panel": "members/byeonghyeon/input/bq_upload/modeling_country_day_panel.parquet",
}

for table, path in loads.items():
    table_id = f"{PROJECT_ID}.{DATASET}.{table}"
    with open(path, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_id, job_config=job_config)
    job.result()
    print(table_id, client.get_table(table_id).num_rows)
```

## 12. parquet export 초안

실제 실행하지 않는다.

```python
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path("../conflict-early-warning/input/processed/dataset/full.parquet")
OUT = Path("members/byeonghyeon/input/bq_upload")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(SRC)
df["date"] = pd.to_datetime(df["date"]).dt.date

gdelt_cols = [
    "gdelt_goldstein_mean_7d", "gdelt_goldstein_mean_14d", "gdelt_goldstein_mean_30d",
    "gdelt_goldstein_std_7d", "gdelt_goldstein_std_14d", "gdelt_goldstein_std_30d",
    "gdelt_tone_mean_7d", "gdelt_tone_mean_14d", "gdelt_tone_mean_30d",
    "gdelt_mentions_sum_7d", "gdelt_mentions_sum_14d", "gdelt_mentions_sum_30d",
    "gdelt_event_count_7d", "gdelt_event_count_14d", "gdelt_event_count_30d",
    "gdelt_quadclass_1_ratio", "gdelt_quadclass_2_ratio",
    "gdelt_quadclass_3_ratio", "gdelt_quadclass_4_ratio",
]
econ_cols = [c for c in df.columns if c.startswith("econ_")]
label_cols = [
    "y", "y_onset", "y_escalation",
    "fatalities_next3d", "event_count_next3d",
    "past14d_event_count", "past14d_fatalities_mean",
]

def assign_split(d):
    d = pd.Timestamp(d)
    if d <= pd.Timestamp("2022-12-31"):
        return "train_fit"
    if d <= pd.Timestamp("2023-12-31"):
        return "tune_cal"
    if d <= pd.Timestamp("2024-06-30"):
        return "val_eval"
    return "test"

split = df[["date", "country"]].copy()
split["split"] = split["date"].map(assign_split)

labels = df[["date", "country"] + label_cols].copy()
gdelt = df[["date", "country"] + gdelt_cols].copy()
panel = split.copy()

# BigQuery에 경제 테이블이 없을 때만 생성
econ = df[["date", "country"] + econ_cols].copy()

# validation examples
for name, part in {
    "labels_daily": labels,
    "split_daily": split,
    "gdelt_events_daily_features": gdelt,
    "modeling_country_day_panel": panel,
}.items():
    assert part.duplicated(["date", "country"]).sum() == 0, name
    assert part["country"].nunique() == 58, name

gdelt.to_parquet(OUT / "gdelt_events_daily_features.parquet", index=False)
labels.to_parquet(OUT / "labels_daily.parquet", index=False)
split.to_parquet(OUT / "split_daily.parquet", index=False)
panel.to_parquet(OUT / "modeling_country_day_panel.parquet", index=False)
```

## 13. 다음 단계

1. BigQuery 인증을 준비하고 `bq ls project-7181e301-a6fb-46f2-871:conflict_ew`로 경제지표 테이블명을 확인한다.
2. projection script를 별도 파일로 작성한다.
3. projection parquet를 `members/byeonghyeon/input/bq_upload/`에 생성한다.
4. 검증 체크리스트를 통과한 뒤에만 `bq load`를 실행한다.
5. 업로드 후 BigQuery에서 row count, schema, partitioning, clustering을 확인한다.
6. `modeling_o0_features_view`를 만들어 feature와 label 분리를 유지한 상태로 모델링 입력을 구성한다.
