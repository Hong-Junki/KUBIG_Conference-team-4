# 파이프라인 구현 계획 문서

> **프로젝트**: KUBIG Conference Team 4 — 무력충돌 조기경보 대시보드
> **작성일**: 2026-05-26
> **참조 문서**: `docs/data_model_serving_pipeline.md`
> **목적**: 설계 문서를 파일 단위 구현 TODO로 쪼개고, 지금 만들 수 있는 것과 팀 결정 이후에 만들 것을 구분한다.
> **주의**: 이 문서는 구현 계획이다. 실제 코드 파일을 생성하거나 수정하지 않는다.

---

## 목차

1. [구현 범위 요약](#1-구현-범위-요약)
2. [추천 파일 구조](#2-추천-파일-구조)
3. [파일별 상세 설계](#3-파일별-상세-설계)
4. [구현 순서 제안](#4-구현-순서-제안)
5. [팀 회의에서 받아야 하는 입력값](#5-팀-회의에서-받아야-하는-입력값)
6. [구현 전 주의사항](#6-구현-전-주의사항)

---

## 1. 구현 범위 요약

### A. 지금 바로 skeleton으로 만들 수 있는 것

최종 모델과 DB 구성이 확정되지 않아도 인터페이스와 스키마를 먼저 고정할 수 있는 항목이다.

| 항목 | 파일 후보 | 비고 |
|------|----------|------|
| feature table schema 정의 | `features/feature_schema.py` | 컬럼명/타입 상수 정의 |
| mock feature table 생성 | `features/build_daily_features.py` | raw 없어도 shape 확인 가능한 mock 포함 |
| model_predictions schema 정의 | `models/predict.py` | 출력 컬럼 고정 |
| mock prediction 생성 | `models/predict.py` | 실제 모델 없어도 random prob 반환 가능 |
| dashboard_country_risk schema 정의 | `scoring/risk_score.py` | 출력 컬럼 고정 |
| risk scoring Candidate 1 (prob × 100) | `scoring/risk_score.py` | 가장 단순한 후보부터 구현 |
| dashboard CSV/JSON exporter skeleton | `serving/export_dashboard_data.py` | 파일 저장 경로/포맷 고정 |
| run_daily_pipeline.py 단계 연결 | `run_daily_pipeline.py` | 함수 호출 순서 고정, 실제 구현은 나중 |

### B. 팀 결정 이후 구현해야 하는 것

아래 항목은 팀 회의 또는 모델 담당자의 결과물이 나온 뒤에 채울 수 있다.

| 항목 | 결정 주체 | 비고 |
|------|----------|------|
| 최종 feature list와 column order | 모델 담당자 + feature builder 담당자 | `feature_list.json`으로 고정 |
| 최종 model artifact 로드 방식 | 모델 담당자 | `.pkl` / `.pt` 등 형식 확정 후 |
| 실제 `model.predict` 또는 `predict_proba` | 모델 담당자 | artifact 확정 후 교체 |
| DB upsert 방식 | 인프라 담당자 | PostgreSQL / SQLite 등 확정 후 |
| Docker container boundary | 인프라 담당자 | MVP 이후 단계 |
| `risk_level` threshold | 팀 회의 | 3단계 vs 4단계 포함 |
| `main_driver` 생성 방식 | 모델 담당자 | feature importance 기반 등 |
| `confidence_level` / `data_quality_flag` 컬럼명 | 팀 회의 | |

---

## 2. 추천 파일 구조

설계 문서(`data_model_serving_pipeline.md`)에서 정의한 인터페이스를 기준으로 아래 파일 구조를 제안한다.

```
project_root/
├── features/
│   ├── feature_schema.py          # 컬럼명/타입 상수 정의
│   └── build_daily_features.py    # raw → feature table 변환
├── models/
│   ├── model_loader.py            # artifact 로드
│   └── predict.py                 # feature table → model_predictions
├── scoring/
│   └── risk_score.py              # model_predictions → dashboard_country_risk
├── serving/
│   └── export_dashboard_data.py   # dashboard_country_risk → CSV/JSON/DB
└── run_daily_pipeline.py          # 전체 파이프라인 단계 연결
```

### 파일별 요약 표

| 파일 경로 | 역할 | 입력 | 출력 | 주요 함수 후보 | skeleton 가능 | 팀 결정 필요 |
|----------|------|------|------|--------------|-------------|-------------|
| `features/feature_schema.py` | 컬럼 상수 정의 | 없음 (상수 파일) | Python 상수/리스트 | `FEATURE_COLS`, `KEY_COLS` | **가능** | 최종 feature list |
| `features/build_daily_features.py` | raw → feature table | `input/raw/` parquet | `input/processed/features_country_daily.parquet` | `build_features()`, `build_mock_features()` | **부분 가능** | lag 기준, 결측 처리, ffill 방식 |
| `models/model_loader.py` | artifact 로드 | `artifacts/` | 로드된 모델 객체 | `load_model()`, `load_feature_list()` | **skeleton 가능** | artifact 형식 (.pkl/.pt) |
| `models/predict.py` | feature → 예측 | `features_country_daily.parquet` | `model_predictions.parquet` | `run_prediction()`, `mock_prediction()` | **mock 가능** | 실제 model.predict 구현 |
| `scoring/risk_score.py` | 예측 → risk score | `model_predictions.parquet` | `dashboard_country_risk` | `score_candidate1()`, `assign_risk_level()` | **Candidate 1 가능** | threshold, 단계 수, main_driver |
| `serving/export_dashboard_data.py` | 스키마 → 파일/DB | `dashboard_country_risk` DataFrame | CSV / JSON / DB | `export_csv()`, `export_json()`, `export_db()` | **CSV/JSON 가능** | DB schema, 저장 경로 확정 |
| `run_daily_pipeline.py` | 전체 실행 연결 | 없음 (CLI entry point) | 없음 (side effects) | `run_pipeline()`, `main()` | **순서 skeleton 가능** | 실패 fallback, 스케줄러 연결 |

---

## 3. 파일별 상세 설계

---

### features/feature_schema.py

**목적**

feature table에서 사용할 컬럼명, 타입, 그룹을 상수로 정의한다. `build_daily_features.py`와 `predict.py`가 이 파일을 import해서 컬럼 불일치를 방지한다.

**입력**

없음 (순수 상수 정의 파일)

**출력**

Python 상수, 리스트, 딕셔너리 — 다른 모듈에서 import해서 사용

**주요 상수/함수 후보**

```python
KEY_COLS = ["country", "date"]

ACLED_FEATURE_COLS = [
    "acled_event_count_lag7_7d",
    "acled_fatalities_lag7_30d",
    "acled_battle_count_lag7_14d",
    # ...
]

GDELT_HISTORICAL_FEATURE_COLS = [
    "gdelt_num_articles_1d",
    "gdelt_avg_tone_7d",
    "gdelt_goldstein_mean_7d",
    "gdelt_num_mentions_7d",
    # ...
]

GDELT_RECENT_FEATURE_COLS = [
    "gdelt_doc_volume_1d",
    "gdelt_doc_volume_zscore_30d",
    "gdelt_doc_tone_7d",
    # ...
]

ECONOMIC_FEATURE_COLS = [
    "VIX", "WTI", "Gold", "DXY", "STLFSI4",
    "VIX_7d_change", "WTI_7d_change",
    # ...
]

TIME_FEATURE_COLS = ["day_of_week", "month", "week_of_year"]

ALL_FEATURE_COLS = (
    ACLED_FEATURE_COLS
    + GDELT_HISTORICAL_FEATURE_COLS
    + GDELT_RECENT_FEATURE_COLS
    + ECONOMIC_FEATURE_COLS
    + TIME_FEATURE_COLS
)

FEATURE_TABLE_PATH = "input/processed/features_country_daily.parquet"
```

**아직 결정 필요한 것**

- 최종 feature list와 column order (모델 담당자가 `feature_list.json`으로 제공해야 함)
- GDELT recent feature를 모델 입력에 포함할지 여부
- lag 기준일 (7일 / 14일 / 30일 등)

**구현 시 주의점**

- 이 파일의 `ALL_FEATURE_COLS` 순서가 학습 시 사용한 순서와 반드시 일치해야 한다.
- 모델 artifact 확정 시 `feature_list.json`을 이 파일의 상수와 비교 검증하는 로직을 추가한다.
- 상수를 변경하면 `build_daily_features.py`와 `predict.py`에 동시 영향을 주므로 변경 시 주의한다.

---

### features/build_daily_features.py

**목적**

`input/raw/` 하위의 ACLED, GDELT, economic parquet 파일을 읽어 country-date 단위 feature table(`features_country_daily.parquet`)로 변환한다.

**입력 raw 경로**

```
input/raw/acled/{iso3}.parquet            # 국가별 이벤트 (이벤트 단위)
input/raw/gdelt/{iso3}.parquet            # 국가별 GDELT BQ 이벤트
input/raw/gdelt/{iso3}_doc_vol.parquet    # GDELT DOC 2.0 volume
input/raw/gdelt/{iso3}_doc_tone.parquet   # GDELT DOC 2.0 tone
input/raw/economic/indicators.parquet     # 글로벌 경제지표
```

**출력 feature table 경로**

```
input/processed/features_country_daily.parquet
  컬럼: country, date, [feature columns from feature_schema.py]
```

**주요 함수 후보**

```python
def build_acled_features(country: str, date_range: list) -> pd.DataFrame:
    # 이벤트 단위 → country-date 집계 + lag/shift
    ...

def build_gdelt_features(country: str, date_range: list) -> pd.DataFrame:
    # BQ raw + DOC 2.0 집계
    ...

def build_economic_features(date_range: list) -> pd.DataFrame:
    # STLFSI4 ffill 포함
    ...

def build_time_features(date_range: list) -> pd.DataFrame:
    # day_of_week, month, week_of_year
    ...

def build_features(start_date: str, end_date: str) -> pd.DataFrame:
    # 위 함수들을 호출해서 merge 후 저장
    ...

def build_mock_features(n_countries: int = 5, n_days: int = 30) -> pd.DataFrame:
    # 실제 raw 없이 schema 테스트용 mock 데이터 생성
    ...
```

**ACLED / GDELT / economic 처리 방향**

| 소스 | 처리 방향 |
|------|----------|
| ACLED | 이벤트 단위 → country-date 집계 → lag N일 feature 생성. ACLED 결측 국가는 `acled_missing_flag=1` 처리 |
| GDELT BQ | country-date 집계 → Goldstein Scale, AvgTone, NumArticles rolling |
| GDELT DOC 2.0 | volume zscore 등 정규화 처리 후 별도 컬럼 추가 |
| Economic | yfinance 지표는 일별 그대로 병합. STLFSI4는 **이 단계에서 일별 ffill 적용** |

**아직 결정 필요한 것**

- ACLED lag 기준 (7일 / 14일 / 30일)
- GDELT recent feature의 모델 입력 포함 여부
- 결측치 처리 방식 (zero-fill / ffill / indicator column)
- country-date 기준 cut-off 시각 (UTC 기준)
- 57개국 전체 처리 vs 특정 국가 범위 파라미터화

**구현 시 주의점**

- STLFSI4 ffill은 collect 단계가 아니라 이 파일에서 처리해야 한다. (`[확인됨]`)
- feature table의 `country` 컬럼은 반드시 ISO3 코드를 사용한다.
- `build_mock_features()`를 먼저 만들어 두면 모델 인터페이스 테스트를 raw 없이도 진행할 수 있다.

---

### models/model_loader.py

**목적**

모델 artifact 파일을 로드하고, 필요한 메타데이터(모델명, 버전, feature list)를 함께 반환한다. 최종 모델이 확정되기 전에는 mock 객체를 반환하는 skeleton으로 유지한다.

**artifact 후보 경로**

```
artifacts/final_model.pkl        # 직렬화된 모델 객체
artifacts/feature_list.json      # 학습 시 사용한 feature 이름과 순서
artifacts/model_metadata.json    # model_name, model_version, trained_date, target 등
```

**주요 함수 후보**

```python
def load_model(artifact_path: str = "artifacts/final_model.pkl"):
    # pkl 또는 pt 파일 로드
    # 아직 artifact 없으면 MockModel() 반환
    ...

def load_feature_list(path: str = "artifacts/feature_list.json") -> list:
    # feature_list.json 로드
    # 없으면 feature_schema.py의 ALL_FEATURE_COLS 반환
    ...

def load_model_metadata(path: str = "artifacts/model_metadata.json") -> dict:
    # model_name, model_version 등 반환
    # 없으면 {"model_name": "mock", "model_version": "0.0.0"} 반환
    ...
```

**최종 모델이 확정되기 전 skeleton 처리 방향**

- `artifacts/` 디렉토리 또는 artifact 파일이 없는 경우 `MockModel`을 반환하도록 분기한다.
- `MockModel`은 `predict_proba(X)` 호출 시 `np.random.uniform(0, 0.3, size=len(X))`를 반환한다.
- artifact가 생기면 `load_model()`의 반환 객체만 교체하면 이후 pipeline은 수정 없이 작동한다.

**아직 결정 필요한 것**

- artifact 파일 형식: `.pkl` (sklearn/joblib), `.pt` (PyTorch), 기타
- 다중 모델(stacking) 구조의 경우 artifact가 여러 파일일 수 있음 — 로드 방식 별도 설계 필요
- `artifacts/` 경로를 환경변수 또는 config로 관리할지 여부

---

### models/predict.py

**목적**

feature table을 읽어 model artifact로 예측을 생성하고, `model_predictions` schema로 저장한다. 이 파일의 출력 schema는 고정되어야 하며, 모델 내부 구조가 바뀌어도 출력 형식은 유지된다.

**입력 feature table**

```
input/processed/features_country_daily.parquet
  컬럼: country, date, [feature columns]
```

**출력 model_predictions**

```
outputs/predictions/model_predictions.parquet
  또는 DB table: model_predictions
```

출력 schema:

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `country` | VARCHAR(3) | ISO3 국가 코드 |
| `date` | DATE | 예측 대상일 |
| `model_name` | VARCHAR | 모델 식별자 |
| `model_version` | VARCHAR | 모델 버전 태그 |
| `predicted_probability` | FLOAT [0, 1] | 모델 출력 확률값 |
| `created_at` | TIMESTAMP | 예측 생성 시각 |

**주요 함수 후보**

```python
def run_prediction(
    feature_table_path: str,
    output_path: str,
    model=None,
    feature_list: list = None,
) -> pd.DataFrame:
    # feature table 로드 → feature_list 순서로 슬라이싱 → predict_proba → 저장
    ...

def mock_prediction(feature_table: pd.DataFrame) -> pd.DataFrame:
    # 실제 모델 없이 0~0.3 균일 분포 확률값 생성
    # 인터페이스 및 schema 테스트용
    ...
```

**mock prediction 또는 placeholder 처리 가능성**

- `model_loader.load_model()`이 `MockModel`을 반환할 때, `run_prediction()`은 `mock_prediction()`을 호출한다.
- `MockModel.predict_proba(X)` → random uniform 반환, 출력 schema는 동일하게 유지.
- 실제 artifact가 생기면 `model_loader.load_model()`만 교체하면 된다.

**아직 결정 필요한 것**

- 실제 `predict_proba` 호출 방식 (sklearn API / PyTorch forward / 커스텀 래퍼)
- 예측 대상 날짜 범위 파라미터 (전체 feature table vs 최근 N일)
- `model_predictions.parquet`를 덮어쓸지 append할지 (날짜 파티션 전략)

---

### scoring/risk_score.py

**목적**

`model_predictions`를 입력으로 받아 risk scoring 후보 중 하나를 적용하고, `dashboard_country_risk` schema로 변환한다. 이 파일이 두 테이블의 경계 역할을 한다.

**입력**

```
outputs/predictions/model_predictions.parquet
  컬럼: country, date, model_name, model_version, predicted_probability, created_at
```

**출력**

`dashboard_country_risk` schema (DataFrame 또는 저장 파일):

| 컬럼 | 생성 방식 |
|------|----------|
| `country`, `date`, `model_name`, `model_version`, `predicted_probability` | model_predictions에서 직접 |
| `temperature_score` | 선택한 Candidate 함수 적용 |
| `risk_level` | `temperature_score` → threshold 기반 레이블 |
| `delta_1d`, `delta_7d` | 전일/전주 대비 `temperature_score` 변화 |
| `rank_today` | 당일 전체 국가 중 위험 순위 |
| `confidence_level` / `data_quality_flag` | 데이터 품질/커버리지 플래그 |
| `main_driver_1/2/3` | Optional, feature importance 기반 |
| `updated_at` | 현재 시각 |

**Candidate 함수 후보**

```python
def score_candidate1(predicted_probability: pd.Series) -> pd.Series:
    # Probability-only: predicted_probability × 100
    return predicted_probability * 100

def score_candidate2(
    predicted_probability: pd.Series,
    news_signal: pd.Series,
    event_signal: pd.Series,
) -> pd.Series:
    # Composite: 0.75 × prob + 0.15 × news + 0.10 × event, × 100
    return (0.75 * predicted_probability + 0.15 * news_signal + 0.10 * event_signal) * 100

def score_candidate3(predicted_probability: pd.Series) -> pd.Series:
    # Rank-based: percentile rank × 100
    return predicted_probability.rank(pct=True) * 100
```

**risk_level 변환 함수 후보**

```python
def assign_risk_level(temperature_score: pd.Series, thresholds: dict) -> pd.Series:
    # thresholds 예: {"High": 60, "Medium": 30} (3단계 기준)
    # 또는 {"Critical": 80, "High": 60, "Moderate": 30} (4단계 기준)
    # threshold는 팀 회의에서 결정 후 주입
    ...
```

**confidence_level / data_quality_flag 처리 방향**

- ACLED 결측 국가(`acled_missing_flag=1`)는 `data_sparse`로 표기
- 최근 수집 실패 여부는 feature table의 NaN 비율로 판단
- 컬럼명(`confidence_level` vs `data_quality_flag`)은 팀 회의에서 결정

**아직 결정 필요한 것**

- 최종 scoring 후보 (Candidate 1/2/3 중 선택)
- `risk_level` 단계 수 (3단계 vs 4단계)
- `risk_level` threshold 구체적 수치
- `delta_1d`, `delta_7d` 계산을 위한 이전 날짜 데이터 보존 방식
- `main_driver` 생성 방식 (MVP에서는 None 처리)
- `confidence_level` 컬럼명 확정

---

### serving/export_dashboard_data.py

**목적**

`dashboard_country_risk` DataFrame을 CSV, JSON, 또는 DB에 저장해 대시보드가 읽을 수 있게 한다. MVP에서는 CSV/JSON을 우선 구현하고, DB는 추후 확장으로 분리한다.

**입력**

```
dashboard_country_risk DataFrame
  (scoring/risk_score.py의 출력)
```

**출력 후보**

```
CSV:  public/data/dashboard_country_risk.csv
JSON: public/data/dashboard_country_risk.json
DB:   dashboard_country_risk 테이블 (확장 시)
```

**주요 함수 후보**

```python
def export_csv(df: pd.DataFrame, path: str = "public/data/dashboard_country_risk.csv") -> None:
    # DataFrame → CSV 저장
    # 저장 전 path의 상위 디렉토리 생성
    ...

def export_json(df: pd.DataFrame, path: str = "public/data/dashboard_country_risk.json") -> None:
    # DataFrame → JSON (records orient) 저장
    ...

def export_db(df: pd.DataFrame, table: str, conn_str: str) -> None:
    # DataFrame → DB upsert
    # conn_str은 .env에서 로드, 이 파일에 직접 쓰지 않음
    # MVP에서는 미구현 placeholder로 두기
    ...

def export(df: pd.DataFrame, mode: str = "csv") -> None:
    # mode: "csv" / "json" / "db"
    # MVP 기본값: "csv"
    ...
```

**MVP에서는 CSV/JSON 우선, DB는 추후 확장**

- MVP에서는 `export_csv()`와 `export_json()`을 구현하고, `export_db()`는 `NotImplementedError`를 raise한다.
- DB 연결 문자열은 `.env`에서만 로드한다. 이 파일에 직접 쓰지 않는다.

**아직 결정 필요한 것**

- 저장 경로 (`public/data/` vs 다른 위치) 확정
- JSON orient 형식 (`records` vs `split` 등) 확정
- DB 사용 시 upsert 방식 (PK: `country + date + model_version`)
- 파일 덮어쓰기 vs 날짜별 아카이브 방식

---

### run_daily_pipeline.py

**목적**

`collect_recent → build_features → run_inference → run_risk_scoring → export_dashboard` 전체 파이프라인을 하나의 스크립트로 연결한다. 각 단계는 독립 모듈을 호출하는 방식으로 구성해 단계별 교체가 가능하게 한다.

**전체 실행 순서**

```
1. collect_recent     수집기별 collect_recent() 호출 → input/raw/ 갱신
2. build_features     features/build_daily_features.py → features_country_daily.parquet 갱신
3. run_inference      models/predict.py → model_predictions.parquet 저장
4. run_risk_scoring   scoring/risk_score.py → dashboard_country_risk DataFrame 생성
5. export_dashboard   serving/export_dashboard_data.py → CSV/JSON/DB 저장
```

**단계별 호출 함수 후보**

```python
def step_collect_recent() -> bool:
    # acled_collector.collect_recent()
    # gdelt_collector.collect_recent_doc()
    # economic_collector.collect_recent()
    ...

def step_build_features() -> bool:
    # features.build_daily_features.build_features()
    ...

def step_run_inference() -> bool:
    # models.predict.run_prediction()
    ...

def step_run_risk_scoring() -> bool:
    # scoring.risk_score.score() + assign_risk_level()
    ...

def step_export_dashboard() -> bool:
    # serving.export_dashboard_data.export()
    ...

def run_pipeline(stop_on_failure: bool = True) -> None:
    steps = [
        step_collect_recent,
        step_build_features,
        step_run_inference,
        step_run_risk_scoring,
        step_export_dashboard,
    ]
    for step in steps:
        success = step()
        if not success and stop_on_failure:
            logger.error(f"Step {step.__name__} failed. Stopping pipeline.")
            sys.exit(1)
```

**실패 시 중단/계속 여부**

- 기본값: `stop_on_failure=True` (한 단계 실패 시 이후 단계 실행 중단)
- `--continue-on-failure` 플래그로 전환 가능하게 설계
- 수집 실패의 경우에는 feature table에 이전 날짜 데이터가 남아 있으므로, 계속 진행 여부를 선택할 수 있게 한다.

**logging / checkpoint 적용 방향**

- `collect/utils.py`의 `get_logger()`를 import해서 동일한 로그 포맷 유지
- 각 단계 시작/완료/실패를 INFO/ERROR 레벨로 기록
- 단계별 checkpoint는 MVP에서는 단순 파일 존재 여부로 확인 (`model_predictions.parquet` 최신 여부 등)
- 추후 `collect/utils.py`의 `Checkpoint` 클래스를 단계 단위로 확장 가능

**아직 결정 필요한 것**

- CLI 인터페이스 (`argparse`): `--date`, `--sources`, `--dry-run` 등
- 스케줄러 연결 방식 (cron expression, GitHub Actions schedule 등)
- 수집 실패 시 fallback 정책 (이전 날짜 feature 재사용 vs 파이프라인 중단)
- daily pipeline과 historical backfill을 단일 entry point로 통합할지 분리할지

---

## 4. 구현 순서 제안

아래 순서로 진행하면 모델/DB/Docker 확정을 기다리지 않고 인터페이스와 흐름을 먼저 검증할 수 있다.

| 단계 | 작업 | 완료 기준 | 팀 결정 필요 여부 |
|------|------|----------|------------------|
| 1 | `features/feature_schema.py` 작성 | 컬럼 상수 정의 완료 | 최종 feature list 확정 전까지 예시 기반 |
| 2 | `features/build_daily_features.py`의 `build_mock_features()` 작성 | mock DataFrame 생성 가능 | 불필요 |
| 3 | `models/predict.py`의 `mock_prediction()` + 출력 schema 정의 | `model_predictions` schema로 저장 가능 | 불필요 |
| 4 | `scoring/risk_score.py`의 Candidate 1 + `assign_risk_level()` skeleton | `dashboard_country_risk` schema로 변환 가능 | threshold 미확정 상태로 dummy 처리 |
| 5 | `serving/export_dashboard_data.py`의 `export_csv()` / `export_json()` | CSV/JSON 파일 생성 가능 | 저장 경로 확정 필요 |
| 6 | `run_daily_pipeline.py` skeleton | mock 데이터로 전체 흐름 end-to-end 실행 가능 | fallback 정책 |
| 7 | 실제 raw 기반 `build_features()` 구현 | feature table 실제 생성 가능 | lag 기준, 결측 처리 방식 |
| 8 | 최종 model artifact 연결 | `model_loader.load_model()`이 실제 모델 반환 | 모델 담당자 artifact 제공 |
| 9 | risk_level threshold 적용 | 팀 회의 결과 반영 | threshold 확정 필요 |
| 10 | DB / Docker 연결 | `export_db()` 구현 + container 연결 | 인프라 담당자 결정 필요 |

---

## 5. 팀 회의에서 받아야 하는 입력값

아래 항목은 팀 회의 또는 담당자가 결과물을 제공해야 구현이 가능한 부분이다.

**모델 관련**

- [ ] 최종 모델 파일 형식 (`.pkl` / `.pt` / 기타) 및 저장 경로
- [ ] 최종 `feature_list.json` 제공 여부 (컬럼 이름 + 순서)
- [ ] 최종 모델 입력 feature column order (학습 시 기준)
- [ ] `predicted_probability` 컬럼명 확정 (변경 없이 유지 권장)

**Risk Scoring 관련**

- [ ] risk scoring 후보 중 MVP 적용 방식 (Candidate 1 / 2 / 3)
- [ ] `risk_level` 단계 수 (3단계 `Low/Medium/High` vs 4단계 `Low/Moderate/High/Critical`)
- [ ] `risk_level` threshold 구체적 수치 (temperature_score 기준)
- [ ] `confidence_level` 컬럼명을 `data_quality_flag`로 바꿀지 여부

**서빙 / 인프라 관련**

- [ ] dashboard output 저장 방식 (CSV / JSON / DB)
- [ ] DB 사용 시 table schema 및 upsert key 확정
- [ ] Docker container boundary (어떤 스크립트가 어떤 컨테이너에 들어가는지)
- [ ] daily batch 갱신 주기 (매일 / 매주 / 수동)
- [ ] 수집 실패 시 fallback 정책

---

## 6. 구현 전 주의사항

**git 관련**

- 현재 `git status`에 여러 untracked 파일(`outputs/predictions/`, `model/`, `modeling/` 등)이 있다. 내 작업 파일만 명시적으로 `git add {파일명}`으로 스테이징해야 한다.
- `git add .` 또는 `git add -A`는 절대 사용하지 않는다. 모델 담당자의 예측 파일, modeling 스크립트 등이 의도치 않게 포함될 수 있다.

**모듈 분리**

- `model/` 또는 `modeling/` 하위 파일은 모델 담당자가 관리한다. 이 파이프라인 작업에서 해당 파일을 수정하거나 참조 경로를 변경하지 않는다.
- 이 파이프라인(`features/`, `models/`, `scoring/`, `serving/`)은 `collect/`와 인터페이스(파일 경로, 컬럼명)로만 연결되며, collect 내부 코드를 수정하지 않는다.

**Credential 관련**

- API key, OAuth token, DB 연결 문자열은 `.env`에서만 로드한다. `config.py`나 구현 파일에 직접 쓰지 않는다.
- `export_db()`의 `conn_str`은 반드시 환경변수에서 주입받는 방식으로 설계한다.

**인터페이스 우선 원칙**

- 최종 모델이 확정되기 전에는 실제 inference를 구현하지 않는다. `MockModel`과 `mock_prediction()`으로 인터페이스와 schema를 먼저 고정한다.
- `model_predictions`와 `dashboard_country_risk`의 출력 schema는 모델이 바뀌어도 최대한 유지한다. 내부 구현이 달라져도 이 두 schema를 변경하지 않는 방향으로 설계한다.
- 구현 중에 schema를 바꿔야 할 경우, 반드시 `data_model_serving_pipeline.md`를 함께 업데이트한다.

---

## 참고 문서

| 문서 | 내용 |
|------|------|
| `docs/data_model_serving_pipeline.md` | 이 구현 계획의 기준이 되는 설계 문서 |
| `collect/config.py` | 57개국 매핑, 수집 기간, 소스 설정 |
| `collect/utils.py` | 로거, retry, Checkpoint 공통 유틸 |
| `collect/acled_collector.py` | ACLED raw 수집, collect_recent 인터페이스 |
| `collect/gdelt_collector.py` | GDELT BQ / DOC 2.0 수집 인터페이스 |
| `collect/economic_collector.py` | 경제지표 수집, STLFSI4 ffill 처리 위치 |
| `docs/eda-summary.md` | 데이터셋 분포, 양성률, 국가 커버리지 |
