# GDELT Title Feature Builder 실행 가이드

## 목적

BigQuery `conflict-early-warning.conflict_ew.gdelt_titles` 테이블에서  
country × date 단위 GDELT title/tone/count/domain/language 집계 피처를 생성해  
실험 C (ACLED-free + GDELT titles) 학습에 사용한다.

---

## 생성 피처 (21개)

### 1일 집계 (10개)
| 피처명 | 설명 |
|--------|------|
| `gdelt_title_count_1d` | 하루 기사 건수 |
| `gdelt_title_nonnull_count_1d` | title 비결측 건수 |
| `gdelt_title_tone_mean_1d` | 평균 톤 점수 (음수=부정) |
| `gdelt_title_tone_std_1d` | 톤 표준편차 |
| `gdelt_title_tone_min_1d` | 최저 톤 (극단적 부정 기사) |
| `gdelt_title_negative_count_1d` | 톤 < -5 기사 수 |
| `gdelt_title_positive_count_1d` | 톤 > +2 기사 수 |
| `gdelt_title_eng_count_1d` | 영어 기사 수 (국제 언론 노출도) |
| `gdelt_title_domain_diversity_1d` | 고유 도메인 수 |
| `gdelt_title_lang_diversity_1d` | 고유 언어 수 |

### 7일 rolling (11개)
| 피처명 | 계산 방식 |
|--------|----------|
| `gdelt_title_count_7d` | 7일 합계 |
| `gdelt_title_nonnull_count_7d` | 7일 합계 |
| `gdelt_title_tone_mean_7d` | 7일 평균 |
| `gdelt_title_tone_std_7d` | 7일 평균 |
| `gdelt_title_tone_min_7d` | 7일 최솟값 |
| `gdelt_title_negative_count_7d` | 7일 합계 |
| `gdelt_title_positive_count_7d` | 7일 합계 |
| `gdelt_title_eng_count_7d` | 7일 합계 |
| `gdelt_title_domain_diversity_7d` | 7일 평균 |
| `gdelt_title_lang_diversity_7d` | 7일 평균 |
| `gdelt_title_tone_trend_7d` | 당일 톤 평균 − 7일 전 톤 평균 (초기 7일은 NULL → 0) |

> **Rolling window 주의사항**  
> `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW`는 실제 존재하는 행 기준으로 작동한다.  
> BQ daily CTE는 보도가 있는 날만 포함하므로, 보도 없는 날이 존재하면  
> `count_7d`는 "직전 7 달력일의 합"이 아닌 "직전 7 보도일의 합"이 된다.  
> 2015 이후 대부분의 국가-일에서는 매일 보도가 있으므로 실용적 영향은 제한적이다.

### coverage_mask — 이 parquet에 포함되지 않음

`gdelt_title_coverage_mask`는 이 parquet에 포함되지 않는다.  
**학습 스크립트에서 left-join 후 직접 설정해야 한다.**

```python
import pandas as pd
BQ_MIN_DATE = "2015-02-17"

gdelt_df = pd.read_parquet("gdelt_title_features.parquet")
train = train.merge(gdelt_df, on=["date", "country"], how="left")

# 결측 피처 0 채움
from build_gdelt_title_features import GDELT_TITLE_FEATURE_COLS
train[GDELT_TITLE_FEATURE_COLS] = train[GDELT_TITLE_FEATURE_COLS].fillna(0)

# coverage_mask: BQ 데이터 없는 구간 표시
train["gdelt_title_coverage_mask"] = (
    train["date"] < pd.Timestamp(BQ_MIN_DATE, tz="UTC")
).astype(int)
```

이유: BQ daily CTE는 데이터 없는 날을 반환하지 않으므로, SQL에서 계산하면  
2015-02-17 이전 행이 결과에 없어 mask가 항상 0이 됨.  
Python에서 merge 후 `date` 비교로 설정하는 것이 올바른 방법이다.

---

## 전제 조건

### 패키지

```bash
pip install google-cloud-bigquery pyarrow pandas
```

### BigQuery 인증 (중요)

스크립트는 다음 순서로 인증을 처리한다:

**1순위 (권장): 환경변수**

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
python members/byeonghyeon/modeling/build_gdelt_title_features.py
```

**2순위 (자동 탐색): 개인 레포 루트의 서비스 계정 JSON**

환경변수가 설정되지 않으면, `conflict-early-warning/` 루트에서  
파일명 길이가 20자 이상인 `*.json` 파일을 자동으로 탐색해 사용한다.  
파일이 없으면 오류 후 종료.

> 파일명이 코드에 하드코딩되지 않으며, 환경변수를 통한 명시적 설정을 권장한다.

---

## 실행 방법

### Step 1: SQL 확인 + 비용 안내 출력 (BQ 실행 없음, 비용 없음)

```bash
cd /path/to/KUBIG_Conference-team-4

python members/byeonghyeon/modeling/build_gdelt_title_features.py --dry-run-sql
```

출력 내용:
- 생성될 SQL 전문
- 예상 스캔 데이터량
- BQ Console / bq CLI에서 Dry Run으로 비용 확인하는 방법

### Step 2: BQ Console에서 예상 비용 확인 (권장)

1. [BigQuery Console](https://console.cloud.google.com/bigquery) → 프로젝트 `conflict-early-warning`
2. Step 1에서 출력된 SQL을 붙여넣기
3. **실행 버튼 옆 드롭다운 → "Dry Run"** 클릭
4. 우측 상단에 처리 예상 바이트 수 확인

또는 `bq` CLI:
```bash
bq query --dry_run --use_legacy_sql=false "$(python members/byeonghyeon/modeling/build_gdelt_title_features.py --dry-run-sql 2>/dev/null | grep -A1000 'WITH daily')"
```

### Step 3: 실제 BQ 쿼리 실행

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

cd /path/to/KUBIG_Conference-team-4
python members/byeonghyeon/modeling/build_gdelt_title_features.py
```

기본 설정:
- **기간**: 2014-01-01 ~ 2025-03-31
- **출력**: `members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_features.parquet`
- **예상 소요 시간**: 3~10분

---

## 출력 파일

```
members/byeonghyeon/input/processed/gdelt_titles/
└── gdelt_title_features.parquet
      컬럼: date, country, [21개 피처]
      행수: 약 215,000행 (58개국 × ~3,700일, 보도 있는 날만)
      크기: 약 10~20 MB (parquet 압축 후)
```

> **참고**: BQ 데이터는 2015-02-17부터 시작하므로,  
> 2014-01-01~2015-02-16 구간은 parquet에 없다.  
> 학습 시 left-join 후 해당 날짜는 0으로 채워지고, coverage_mask로 표시.

> **이 파일은 git에 커밋하지 않는다.** `members/byeonghyeon/input/`는 gitignored 대상.

---

## SQL 구조 요약

```
raw 테이블 스캔: 1회
  ↓
daily CTE: country × date 집계 (WHERE date BETWEEN 필터 적용)
  ↓
with_rolling CTE: window function (집계 결과만 읽음, 추가 raw 스캔 없음)
  ↓
SELECT * FROM with_rolling
```

**BQ 파티셔닝**: `gdelt_titles` 테이블이 `date` 컬럼으로 파티셔닝된 경우  
`WHERE date BETWEEN '2014-01-01' AND '2025-03-31'` 필터가 자동으로  
파티션 pruning에 사용된다. 파티셔닝 여부는 BQ Console 테이블 스키마에서 확인.

---

## 학습 스크립트에서의 사용 예시

```python
import pandas as pd
from build_gdelt_title_features import GDELT_TITLE_FEATURE_COLS, BQ_MIN_DATE

GDELT_TITLE_PARQUET = "members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_features.parquet"

# 1. parquet 로드
gdelt_df = pd.read_parquet(GDELT_TITLE_PARQUET)

# 2. train/val/test에 left-join
train = train.merge(gdelt_df, on=["date", "country"], how="left")
val   = val.merge(gdelt_df, on=["date", "country"], how="left")
test  = test.merge(gdelt_df, on=["date", "country"], how="left")

# 3. 결측 피처 0 채움 (보도 없는 날 또는 coverage gap)
for df in [train, val, test]:
    df[GDELT_TITLE_FEATURE_COLS] = df[GDELT_TITLE_FEATURE_COLS].fillna(0)

# 4. coverage_mask 설정 (BQ 데이터 없는 구간)
for df in [train, val, test]:
    df["gdelt_title_coverage_mask"] = (
        df["date"] < pd.Timestamp(BQ_MIN_DATE, tz="UTC")
    ).astype(int)

# 5. feature_cols에 추가
feature_cols_C = FEATURE_COLS_B + GDELT_TITLE_FEATURE_COLS + ["gdelt_title_coverage_mask"]
```

---

## 주의사항

1. **BQ 비용**: 전체 기간 쿼리는 수십~수백 GB 스캔 발생.  
   실행 전 Step 2 (Dry Run)로 예상 비용을 반드시 확인하세요.

2. **인증은 환경변수로**: `GOOGLE_APPLICATION_CREDENTIALS` 환경변수 사용을 권장.  
   서비스 계정 키 경로를 코드에 직접 쓰지 마세요.

3. **이 parquet는 git에 커밋하지 않는다**: 대용량 + 재생성 가능한 artifact.

4. **coverage_mask는 학습 스크립트에서**: 위 "학습 스크립트에서의 사용 예시" 참조.

5. **Rolling window 제한**: 보도 없는 날은 7d rolling 윈도우에서 제외됨.  
   2015 이후 대부분의 국가-일에서는 매일 보도가 있으므로 실용적 영향 제한적.
