# 파이프라인 사용 가이드

> **프로젝트**: KUBIG Conference Team 4 — 무력충돌 조기경보 대시보드
> **작성일**: 2026-05-26
> **대상**: 파이프라인을 처음 실행하는 팀원

---

## 목차

1. [이 문서의 목적](#1-이-문서의-목적)
2. [현재 pipeline 실행 방법](#2-현재-pipeline-실행-방법)
3. [실행 흐름](#3-실행-흐름)
4. [생성되는 주요 output](#4-생성되는-주요-output)
5. [현재 mock인 부분](#5-현재-mock인-부분)
6. [실제 모델 연결 방법](#6-실제-모델-연결-방법)
7. [대시보드 연결 방법](#7-대시보드-연결-방법)
8. [팀 회의에서 정해야 할 것](#8-팀-회의에서-정해야-할-것)
9. [git 주의사항](#9-git-주의사항)

---

## 1. 이 문서의 목적

이 문서는 팀원이 데이터-모델-대시보드 skeleton pipeline을 직접 실행하고 output 파일을 확인할 수 있도록 돕기 위한 실행 가이드다.

**현재 상태**

- 실제 모델 artifact(`.pkl`)와 DB가 아직 연결되지 않은 **mock 기반 pipeline**이다.
- `python run_daily_pipeline.py`만 실행하면 외부 API 호출 없이 전체 흐름이 동작한다.
- 최종 모델과 DB가 확정되면 두 파일(`models/model_loader.py`, `serving/export_dashboard_data.py`)만 교체하면 되도록 설계되어 있다.

설계 배경은 [`docs/data_model_serving_pipeline.md`](data_model_serving_pipeline.md)와 [`docs/data_model_pipeline_implementation_plan.md`](data_model_pipeline_implementation_plan.md)를 참고한다.

---

## 2. 현재 pipeline 실행 방법

### 기본 실행

```bash
python run_daily_pipeline.py
```

- 외부 API를 호출하지 않는다.
- `.env`, API key, BigQuery credential이 없어도 실행된다.
- mock feature table → mock prediction → risk scoring → CSV/JSON export까지 전체 흐름이 완료된다.

### 날짜 지정 실행

```bash
python run_daily_pipeline.py --as-of-date 2025-01-15
```

- 기본값은 오늘 날짜다. 특정 날짜를 지정하면 해당 날짜 기준으로 feature table을 생성한다.

### Risk scoring 방식 선택

```bash
# Candidate 1: predicted_probability × 100 (기본값)
python run_daily_pipeline.py --scoring-method probability_only

# Candidate 3: 국가 간 상대 분위 기반
python run_daily_pipeline.py --scoring-method rank_based
```

### Risk level 단계 변경

```bash
# 3단계: Low / Medium / High (기본값)
python run_daily_pipeline.py --risk-level-scheme three_level

# 4단계: Low / Moderate / High / Critical
python run_daily_pipeline.py --risk-level-scheme four_level
```

### 수집기 실행 포함 (주의 필요)

```bash
python run_daily_pipeline.py --run-collect
```

> **주의**: `--run-collect`는 ACLED OAuth, GDELT BigQuery, FRED API credential이 `.env`에 설정되어 있어야 한다. collect 환경이 갖춰지지 않은 상태에서는 사용하지 않는다. 기본 실행에서는 이 옵션을 생략한다.

---

## 3. 실행 흐름

```
python run_daily_pipeline.py
  │
  ▼ (선택) --run-collect
  collect_recent   ACLED / GDELT DOC 2.0 / economic 최신 데이터 수집
  │
  ▼ 1단계
  build_daily_features   feature table 생성
  │
  ▼ 2단계
  run_inference          model_predictions 생성
  │
  ▼ 3단계
  run_risk_scoring       dashboard_country_risk 생성
  │
  ▼ 4단계
  export_dashboard_data  public/data/ 로 CSV/JSON export
```

### 단계별 입출력

| 단계 | 읽는 파일 | 생성하는 파일 |
|------|----------|--------------|
| `build_daily_features` | `input/raw/acled/*.parquet` (있으면 참고), `collect/config.py` (국가 목록) | `input/processed/features_country_daily.parquet` |
| `run_inference` | `input/processed/features_country_daily.parquet`, `artifacts/final_model.pkl` (없으면 mock) | `outputs/predictions/model_predictions.parquet`, `outputs/predictions/model_predictions.csv` |
| `run_risk_scoring` | `outputs/predictions/model_predictions.parquet` | `outputs/dashboard/dashboard_country_risk.parquet`, `outputs/dashboard/dashboard_country_risk.csv` |
| `export_dashboard_data` | `outputs/dashboard/dashboard_country_risk.parquet` | `public/data/dashboard_country_risk.csv`, `public/data/dashboard_country_risk.json` |

각 단계는 이전 단계의 output을 입력으로 받는다. 한 단계가 실패하면 pipeline은 기본적으로 중단(`stop_on_failure=True`)된다.

---

## 4. 생성되는 주요 output

### 피처 테이블

| 파일 | 설명 |
|------|------|
| `input/processed/features_country_daily.parquet` | 국가-날짜 단위 feature table. 현재는 mock 값. Key columns: `country`(ISO3), `date` |

### 모델 예측

| 파일 | 설명 |
|------|------|
| `outputs/predictions/model_predictions.parquet` | 모델 원본 출력. `country`, `date`, `model_name`, `model_version`, `predicted_probability`, `created_at` |
| `outputs/predictions/model_predictions.csv` | 동일 내용의 CSV 버전 |

### 대시보드 위험 점수

| 파일 | 설명 |
|------|------|
| `outputs/dashboard/dashboard_country_risk.parquet` | risk scoring 적용 결과. 후처리된 전체 컬럼 포함 |
| `outputs/dashboard/dashboard_country_risk.csv` | 동일 내용의 CSV 버전 |

### 대시보드 공개 파일 (프론트엔드용)

| 파일 | 설명 |
|------|------|
| `public/data/dashboard_country_risk.csv` | 대시보드가 직접 읽는 CSV |
| `public/data/dashboard_country_risk.json` | 대시보드가 직접 읽는 JSON (records orientation) |

### dashboard_country_risk 컬럼 구조

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `country` | str (ISO3) | 국가 코드 |
| `date` | date | 예측 대상일 |
| `model_name` | str | 사용된 모델 식별자 |
| `model_version` | str | 모델 버전 태그 |
| `predicted_probability` | float [0,1] | 모델 원본 출력 확률 |
| `temperature_score` | float [0,100] | risk scoring 적용 결과 |
| `risk_level` | str | `Low` / `Medium` / `High` (또는 4단계) |
| `delta_1d` | float | 전일 대비 temperature_score 변화 |
| `delta_7d` | float | 전주 대비 temperature_score 변화 |
| `rank_today` | int | 당일 전체 국가 중 위험 순위 |
| `data_quality_flag` | str | 데이터 품질 힌트 (`ok` / `mock_or_unverified`) |
| `main_driver_1~3` | str | 주요 위험 요인 (현재 빈 문자열) |
| `updated_at` | timestamp | 생성 시각 |

---

## 5. 현재 mock인 부분

| 항목 | 현재 상태 | 실제 구현 조건 |
|------|----------|--------------|
| **feature table** | mock 난수 값 (deterministic seed 기반) | 최종 feature list 확정 + feature builder 구현 후 |
| **model prediction** | `artifacts/final_model.pkl`이 없으면 mock probability 생성 | 모델 담당자가 artifact 제공 후 자동 전환 |
| **risk scoring** | Candidate 1 (`predicted_probability × 100`) 기본값 | 팀 회의에서 방식 확정 후 `--scoring-method` 인자로 선택 |
| **risk_level threshold** | `Low < 30`, `30 ≤ Medium < 60`, `High ≥ 60` (임시 기준) | 팀 회의에서 threshold 확정 후 `scoring/risk_score.py` 상수 수정 |
| **data_quality_flag** | mock prediction 여부로 판단 (`mock_or_unverified` / `ok`) | 실제 데이터 커버리지 체크 로직으로 교체 |
| **main_driver** | 빈 문자열 | feature importance 기반 구현 (추후) |
| **DB export** | 미구현 (`NotImplementedError`) | DB schema + 연결 문자열 확정 후 구현 |

---

## 6. 실제 모델 연결 방법

모델 담당자가 아래 파일을 `artifacts/` 디렉토리에 제공하면 **pipeline 코드 수정 없이** 실제 inference로 전환된다.

### 제공할 파일

| 파일 | 내용 |
|------|------|
| `artifacts/final_model.pkl` | joblib 또는 pickle로 직렬화된 최종 모델 객체 |
| `artifacts/feature_list.json` | 학습 시 사용한 feature 이름과 순서 (`["col_a", "col_b", ...]`) |
| `artifacts/model_metadata.json` | 모델 식별 정보 (`{"model_name": "...", "model_version": "..."}`) |

### 전환 방식

`models/model_loader.py`의 `load_model_bundle()`은 `artifacts/` 디렉토리를 자동으로 확인한다.

- `final_model.pkl`이 있으면 → joblib/pickle로 로드 후 실제 `predict_proba` 실행
- `final_model.pkl`이 없으면 → WARNING 로그 출력 후 mock prediction으로 fallback
- `feature_list.json`이 있으면 → 해당 순서로 feature matrix 구성
- `feature_list.json`이 없으면 → feature table의 numeric 컬럼을 자동 추론 (순서 불일치 위험 있음)

### 확인 방법

```bash
python run_daily_pipeline.py
```

로그에서 아래 메시지가 나오면 실제 모델이 연결된 것이다:

```
[INFO] models.model_loader — Model bundle ready: <model_name> v<model_version>
```

아래 메시지가 나오면 mock 모드다:

```
[WARNING] models.model_loader — Model artifact not found: artifacts/final_model.pkl. Will use mock prediction.
[INFO] models.model_loader — No model artifact found — pipeline will run in mock-prediction mode.
```

---

## 7. 대시보드 연결 방법

### 현재 (CSV/JSON 기반)

대시보드는 아래 두 파일 중 하나를 읽으면 된다.

```
public/data/dashboard_country_risk.csv
public/data/dashboard_country_risk.json
```

- JSON은 `records` orientation이다. 각 행이 `{country, date, temperature_score, risk_level, ...}` 형태의 객체로 저장된다.
- 날짜/타임스탬프 컬럼은 ISO 문자열로 변환되어 있어 별도 파싱 없이 읽을 수 있다.
- pipeline을 다시 실행하면 파일이 덮어씌워진다.

### 추후 (DB 기반)

최종 DB schema와 연결 문자열이 확정되면 `serving/export_dashboard_data.py`의 `upsert_to_database_placeholder()`를 실제 DB upsert 구현으로 교체한다. 연결 문자열은 `.env`에서만 로드하며 코드에 직접 쓰지 않는다.

---

## 8. 팀 회의에서 정해야 할 것

아래 항목이 확정되어야 pipeline이 실 운영 모드로 전환된다.

**모델 / 아티팩트**
- [ ] 최종 model artifact 형식 (`.pkl` / `.pt` 등)과 `artifacts/`에 제공하는 파일명
- [ ] `feature_list.json` 제공 여부 — 없으면 feature 순서를 자동 추론하므로 불일치 위험 있음

**Risk Scoring**
- [ ] risk scoring 방식 확정 (`probability_only` / `rank_based` / composite)
- [ ] `risk_level` threshold 확정 (현재 Low < 30 / Medium < 60 / High ≥ 60은 임시 기준)
- [ ] `data_quality_flag` 컬럼명 유지 여부 (`confidence_level`로 바꿀지)

**서빙 / 인프라**
- [ ] dashboard output 저장 방식 확정 (CSV/JSON 유지 / DB 전환)
- [ ] DB 사용 시 table schema 및 upsert key 확정
- [ ] Docker에서 `python run_daily_pipeline.py`를 실행할 경우 container image, 환경변수 주입 방식, 실행 주기

---

## 9. git 주의사항

현재 repository에는 untracked 파일이 많다 (`outputs/`, `model/`, `modeling/` 등). 아래 규칙을 반드시 지킨다.

**금지**

```bash
git add .        # 절대 사용 금지
git add -A       # 절대 사용 금지
```

`outputs/predictions/`, `modeling/`, `model/` 등 모델 담당자 파일이 의도치 않게 포함될 수 있다.

**올바른 방법 — 파일을 명시적으로 지정**

```bash
# 파이프라인 관련 신규 파일만 add
git add run_daily_pipeline.py
git add features/
git add models/
git add scoring/
git add serving/
git add docs/pipeline_usage.md
git add docs/data_model_serving_pipeline.md
git add docs/data_model_pipeline_implementation_plan.md
```

**모델 담당자 파일은 분리해서 관리**

`model/`, `modeling/`, `outputs/predictions/val_predictions__*` 등은 이 pipeline과 별도로 모델 담당자가 add/commit한다.
