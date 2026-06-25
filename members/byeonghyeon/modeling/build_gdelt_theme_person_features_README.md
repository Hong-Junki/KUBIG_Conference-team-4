# GDELT Theme & Person Feature Builder 실행 가이드

## 목적

BigQuery `conflict-early-warning.conflict_ew.gdelt_titles`의  
`v2themes`, `v2persons` 컬럼을 활용해 country × date 단위 테마/인물 집계 피처를 생성한다.  
실험 D (C feature set + themes/persons features) 학습에 사용한다.

---

## v2themes / v2persons 포맷

```
v2themes  : "CEASEFIRE,65;CONFLICT,146;BLOCKADE,4133;LEADER,14"
v2persons : "Donald Trump,858;Ali Khamenei,1476;Masoud Pezeshkian,6144"
```

- 세미콜론(`;`) 구분 / 각 토큰은 `이름_또는_테마,문자위치` 형태
- **숫자(문자위치 offset)는 count가 아닌 character position** — 사용하지 않음
- **이름/테마 문자열 자체는 저장하지 않음** — 집계 count만 저장

---

## 생성 피처 (22개)

### 1일 집계 (11개)

| 피처명 | 설명 |
|--------|------|
| `gdelt_theme_nonnull_count_1d` | v2themes 비결측 기사 수 |
| `gdelt_person_nonnull_count_1d` | v2persons 비결측 기사 수 |
| `gdelt_theme_count_1d` | 전체 theme 토큰 수 합계 (SPLIT by `;`) |
| `gdelt_person_count_1d` | 전체 person 토큰 수 합계 |
| `gdelt_theme_conflict_count_1d` | 충돌/분쟁 테마 포함 기사 수 |
| `gdelt_theme_protest_count_1d` | 시위/폭동 테마 포함 기사 수 |
| `gdelt_theme_military_count_1d` | 군사 테마 포함 기사 수 |
| `gdelt_theme_refugee_count_1d` | 난민/이재민 테마 포함 기사 수 |
| `gdelt_theme_sanction_count_1d` | 제재/금수 테마 포함 기사 수 |
| `gdelt_theme_government_count_1d` | 정치/정부 테마 포함 기사 수 |
| `gdelt_person_density_1d` | 기사 1건당 평균 인물 언급 수 |

### 7일 rolling (11개)

위 11개의 7일 rolling: count 계열은 `SUM`, 밀도 계열은 `AVG`

> **Rolling window 주의**: `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW`는 실제 존재하는 행 기준.  
> 보도 없는 날은 제외되므로 "7 달력일"이 아닌 "7 보도일" 합계/평균일 수 있다.

---

## 테마 키워드 정의

| 그룹 | 패턴 (`(?:^|;)` 토큰 시작 매칭) | 설명 |
|------|------|------|
| `conflict` | `CONFLICT\|MILITARY_ATTACK\|ARMED\|BATTLE\|WAR` | 충돌·분쟁 |
| `protest` | `PROTEST\|RIOT\|CIVIL_UNREST\|STRIKE_ACTION\|DEMONSTRATION` | 시위·폭동 |
| `military` | `MILITARY` (prefix) | 군사 전반 |
| `refugee` | `REFUGEE\|DISPLACED\|ASYLUM\|HUMANITARIAN` | 난민·인도주의 |
| `sanction` | `SANCTION\|EMBARGO` | 제재·금수 |
| `government` | `GOV_\|ELECTION\|COUP\|CEASEFIRE\|BLOCKADE\|SEIGE` | 정치·정부 |

키워드는 `(?:^|;)KEYWORD` 패턴으로 세미콜론 토큰 시작에서 매칭해 부분 오탐을 최소화한다.

---

## 전제 조건

```bash
pip install google-cloud-bigquery pyarrow pandas
```

BQ 인증:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
```

---

## 실행 방법

### Step 1: SQL 확인 + 비용 안내 (BQ 실행 없음)

```bash
cd /path/to/KUBIG_Conference-team-4

python members/byeonghyeon/modeling/build_gdelt_theme_person_features.py --dry-run-sql
```

출력 내용:
- SQL 전문
- 테마 키워드 정의
- BQ dry run으로 비용 확인하는 방법

### Step 2: SQL 파일로 저장 (BQ Console에서 dry run 확인용)

```bash
python members/byeonghyeon/modeling/build_gdelt_theme_person_features.py --save-sql
```

저장 위치: `members/byeonghyeon/modeling/gdelt_theme_person_features_query.sql`

### Step 3: Python dry run으로 예상 비용 확인 (권장)

```python
from google.cloud import bigquery
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/key.json"
client = bigquery.Client(project="conflict-early-warning")

SQL = open("gdelt_theme_person_features_query.sql").read()
job = client.query(SQL, job_config=bigquery.QueryJobConfig(dry_run=True))
gb = job.total_bytes_processed / 1024**3
print(f"{gb:.2f} GB  (~${gb/1024*5:.2f})")
```

> C 쿼리(~82 GB, ~$0.40) 대비 REGEXP_CONTAINS 연산이 추가되어 약간 더 비쌀 수 있음.  
> 그러나 동일한 테이블·기간을 스캔하므로 크게 차이나지 않을 것으로 예상.

### Step 4: 실제 BQ 쿼리 실행

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

cd /path/to/KUBIG_Conference-team-4
python members/byeonghyeon/modeling/build_gdelt_theme_person_features.py
```

기본 설정:
- **기간**: 2014-01-01 ~ 2025-03-31
- **출력**: `members/byeonghyeon/input/processed/gdelt_titles/gdelt_theme_person_features.parquet`
- **예상 소요 시간**: 5~15분 (REGEXP 연산으로 C보다 다소 오래 걸릴 수 있음)

---

## 출력 파일

```
members/byeonghyeon/input/processed/gdelt_titles/
└── gdelt_theme_person_features.parquet
      컬럼: date, country, [22개 피처]
      행수: 약 215,000행 (BQ 데이터 있는 country×date만)
      크기: 약 10~25 MB 예상
```

> **이 파일은 git에 커밋하지 않는다.** `members/byeonghyeon/input/` 경로는 gitignored 대상.

---

## 학습 스크립트에서의 사용 예시

```python
import pandas as pd
from build_gdelt_theme_person_features import GDELT_THEME_PERSON_FEATURE_COLS, BQ_MIN_DATE

TP_PARQUET = "members/byeonghyeon/input/processed/gdelt_titles/gdelt_theme_person_features.parquet"

# 1. 기존 C 피처 parquet도 로드
from build_gdelt_title_features import GDELT_TITLE_FEATURE_COLS
TITLE_PARQUET = "members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_features.parquet"

gdelt_title_df = pd.read_parquet(TITLE_PARQUET)
gdelt_tp_df    = pd.read_parquet(TP_PARQUET)

# 2. left-join (C + D 피처 동시 merge)
for df in [train, val, test]:
    df = df.merge(gdelt_title_df, on=["date", "country"], how="left")
    df = df.merge(gdelt_tp_df,    on=["date", "country"], how="left")

# 3. 결측 0 채움
all_gdelt_cols = GDELT_TITLE_FEATURE_COLS + GDELT_THEME_PERSON_FEATURE_COLS
for df in [train, val, test]:
    df[all_gdelt_cols] = df[all_gdelt_cols].fillna(0)

# 4. coverage_mask (C와 공유)
for df in [train, val, test]:
    df["gdelt_title_coverage_mask"] = (
        df["date"] < pd.Timestamp(BQ_MIN_DATE, tz="UTC")
    ).astype(int)

# 5. feature_cols = B 35 + title 21 + coverage_mask 1 + themes/persons 22 = 79
feature_cols_D = FEATURE_COLS_B + GDELT_TITLE_FEATURE_COLS + ["gdelt_title_coverage_mask"] + GDELT_THEME_PERSON_FEATURE_COLS
```

---

## 주의사항

1. **v2persons에서 이름 저장 없음**: 개인정보 및 GDPR 고려. count/density만 사용.
2. **BQ 비용**: dry run으로 확인 후 실행.
3. **이 파일은 git에 커밋하지 않는다**: 재생성 가능한 artifact.
4. **coverage_mask 재사용**: C에서 이미 설정된 `gdelt_title_coverage_mask`를 D에서도 재사용.
