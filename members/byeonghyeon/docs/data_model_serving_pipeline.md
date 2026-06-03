# 데이터-모델-서빙 파이프라인 설계 문서

> **프로젝트**: KUBIG Conference Team 4 — 무력충돌 조기경보 대시보드
> **작성일**: 2026-05-26
> **상태**: MVP 설계 (인터페이스 기준점 문서)
> **표기 약속**: `[확인됨]` = 실제 파일에서 검증된 사실 / `[설계 필요]` = 구현 시 결정이 필요한 항목

---

## 목차

1. [문서 목적](#1-문서-목적)
2. [전체 파이프라인 개요](#2-전체-파이프라인-개요)
3. [현재 데이터 수집 구조](#3-현재-데이터-수집-구조)
   - 3.1 ACLED Collector
   - 3.2 GDELT Collector
   - 3.3 Economic Collector
   - 3.4 공통 유틸리티
4. [현재 오케스트레이션 구조](#4-현재-오케스트레이션-구조)
5. [Feature Table Schema 초안](#5-feature-table-schema-초안)
6. [Model Inference Interface 초안](#6-model-inference-interface-초안)
7. [Risk Scoring 후보](#7-risk-scoring-후보)
8. [Dashboard Output Schema 초안](#8-dashboard-output-schema-초안)
9. [파이프라인 실행 옵션 비교](#9-파이프라인-실행-옵션-비교)
10. [팀 회의에서 결정해야 할 사항](#10-팀-회의에서-결정해야-할-사항)
11. [이후 구현 단계](#11-이후-구현-단계)
12. [참고 파일](#12-참고-파일)

---

## 1. 문서 목적

이 문서는 **무력충돌 조기경보 대시보드**의 데이터-모델-서빙 파이프라인을 정리하기 위한 인터페이스 기준점 문서다.

**작성 배경**

모델, DB, Docker, 대시보드가 팀원별로 병렬 작업되는 상황에서 각 모듈의 입력/출력 계약을 먼저 고정하지 않으면, 이후 통합 단계에서 불필요한 수정 비용이 발생한다. 이 문서는 그 계약을 사전에 정의하는 것을 목적으로 한다.

**이 문서의 범위**

- 데이터 수집 모듈(`collect/`)의 실제 구조와 출력 스키마
- Feature Table 표준 스키마 초안
- Model Inference 인터페이스 초안 (모델 내부 구조 제외)
- Risk Scoring 후보안
- Dashboard Output 스키마 초안
- 파이프라인 실행 방식 비교

**이 문서의 범위 밖**

- 최종 모델은 아직 확정되지 않았다. 이 문서에서는 모델 구조, 성능 지표, ablation 결과를 다루지 않는다.
- 모델 내부 구조와 성능 지표는 모델 담당자 문서에서 별도로 관리한다.
- 이 문서는 모델을 black-box로 보고, 입력/출력 인터페이스만 정의한다.

---

## 2. 전체 파이프라인 개요

```
[외부 데이터 소스]
  ACLED REST API / GDELT BigQuery / GDELT DOC 2.0 / yfinance / FRED
         │
         ▼
[Data Collectors]  collect/ 모듈
  수집기별로 raw 데이터를 가져와 parquet으로 저장
         │
         ▼
[Raw Storage]  input/raw/
  수집기 출력 원본 파일 보관. 재가공 없이 저장.
         │
         ▼
[Feature Builder]  [설계 필요]
  raw → feature table 변환 (lag/shift, 집계, 결측 처리, ffill 등)
         │
         ▼
[Model Inference Interface]  [설계 필요]
  feature table → country-date별 predicted_probability 반환
  (최종 모델 확정 전까지 인터페이스만 정의)
         │
         ▼
[Risk Scoring]  [설계 필요]
  predicted_probability → temperature_score, risk_level 등 파생 컬럼 생성
         │
         ▼
[Dashboard Store]  [설계 필요]
  CSV / JSON / DB 중 선택. dashboard가 읽을 최종 스키마 고정.
         │
         ▼
[Frontend Dashboard]
  대시보드가 Dashboard Store를 읽어 지도/순위/트렌드 등 시각화
```

각 단계는 독립적으로 교체 가능하도록 경계를 유지한다.

- 모델이 바뀌어도 `dashboard_country_risk` output schema는 최대한 유지한다.
- 다만 feature builder의 feature list와 column order는 최종 모델 artifact에 맞춰 조정될 수 있다.
- 따라서 안정적으로 고정해야 할 핵심 계약은 `model_predictions`와 `dashboard_country_risk`의 output schema다.

---

## 3. 현재 데이터 수집 구조

### 3.1 ACLED Collector

**파일**: `collect/acled_collector.py` `[확인됨]`

#### 저장 경로

```
input/raw/acled/{iso3}.parquet         # 국가별 원시 이벤트 (누적)
input/raw/acled/.ckpt_historical.json  # 수집 체크포인트
```

#### 수집 함수

| 함수 | 역할 | 체크포인트 키 |
|------|------|--------------|
| `collect_historical(start, end, countries)` | 지정 기간 전체 수집 | `{iso3}_{start}_{end}` |
| `collect_recent(days=14)` | 최근 N일 증분 수집 | 별도 없음 |

- 인증: OAuth 2.0 `TokenManager` (access_token 24h, refresh_token 14d)
- 페이지네이션: REST API 자동 처리
- 기존 parquet과 병합 후 저장 (누적 방식)

#### 중복 제거 기준 `[확인됨]`

`event_id_cnty` 기준 dedup

#### 주요 컬럼 `[확인됨]`

`event_id_cnty`, `event_date`, `year`, `disorder_type`, `event_type`, `sub_event_type`,
`actor1`, `actor2`, `inter1`, `inter2`, `country`, `iso`, `admin1`,
`latitude`, `longitude`, `fatalities`, `timestamp`

대상 이벤트 유형: Battles / Explosions/Remote violence / Violence against civilians

#### Feature 사용 시 주의

- raw 파일에는 이벤트 단위로 저장된다. feature builder에서 country-date 단위로 집계 및 lag/shift 처리가 필요하다. `[설계 필요]`
- ACLED 데이터가 없는 국가-날짜는 결측으로 처리하거나 별도 마스킹 컬럼을 생성해야 한다. `[설계 필요]`

---

### 3.2 GDELT Collector

**파일**: `collect/gdelt_collector.py` `[확인됨]`

#### 이중 수집 경로

```
경로 A — BigQuery (역사적 데이터)
  저장: input/raw/gdelt/{iso3}.parquet
  체크포인트: input/raw/gdelt/.ckpt_historical.json  (키: bq_{YYYYMM})
  테이블: gdelt-bq.gdeltv2.events_partitioned

경로 B — DOC 2.0 API (최근 90일 이내)
  저장: input/raw/gdelt/{iso3}_doc_vol.parquet
        input/raw/gdelt/{iso3}_doc_tone.parquet
```

#### BigQuery 수집 `[확인됨]`

- `collect_historical_bq(start, end, max_gb_per_query=50.0)`: 월별 청크 처리
- `dry_run_query(query)`: 실행 전 스캔 GB 예측, 50GB 초과 시 건너뜀
- GDELT FIPS 코드 → ISO3 매핑으로 국가별 parquet 분리 저장
- 중복 제거 키: `GLOBALEVENTID`

#### DOC 2.0 API 수집 `[확인됨]`

- `collect_recent_doc(days=90)`: 최대 90일 범위, timelinevolraw + timelinetone 동시 수집

#### 주요 BQ 컬럼 `[확인됨]`

`GLOBALEVENTID`, `SQLDATE`, `ActionGeo_CountryCode`, `EventCode`, `EventRootCode`,
`QuadClass`, `GoldsteinScale`, `NumMentions`, `NumArticles`, `AvgTone`

#### 국가 매핑 `[확인됨]`

GDELT FIPS 코드 → ISO3 매핑 사용. 특수 케이스:
- 팔레스타인: FIPS `WE`(서안) + `GZ`(가자) 두 코드 매핑

#### Feature 사용 시 주의

- 기사량(NumArticles, NumMentions)은 국가·사건 규모에 따라 편향이 있으므로, feature builder에서 정규화 또는 상대값 변환이 필요하다. `[설계 필요]`
- GDELT recent(DOC 2.0)와 historical(BQ) 피처를 모델 입력에 함께 사용할지, 대시보드 briefing 전용으로만 사용할지 팀이 결정해야 한다. `[설계 필요]`

---

### 3.3 Economic Collector

**파일**: `collect/economic_collector.py` `[확인됨]`

#### 저장 경로 `[확인됨]`

```
input/raw/economic/indicators.parquet    # 글로벌 일별/주별 경제지표 (단일 파일)
```

#### 수집 지표 `[확인됨]`

| 소스 | 지표명 | 티커/ID | 빈도 |
|------|--------|---------|------|
| yfinance | VIX | `^VIX` | 일별 |
| yfinance | WTI | `CL=F` | 일별 |
| yfinance | Gold | `GC=F` | 일별 |
| yfinance | DXY | `DX-Y.NYB` | 일별 |
| FRED | STLFSI4 | `STLFSI4` | **주별** |

- yfinance: timezone → UTC 정규화, tz-naive 저장
- STLFSI4: 주별 원본 그대로 저장. **ffill은 feature_builder에서 처리해야 한다.** `[확인됨]`

#### 수집 함수 차이 `[확인됨]`

| 함수 | 역할 | 병합 방식 |
|------|------|----------|
| `collect_historical(start, end)` | 지정 기간 전체 수집 | outer join 후 저장 |
| `collect_recent(days=30)` | 최근 N일 수집 | 기존 parquet과 outer join, `keep="last"` (최신값 우선) |

---

### 3.4 공통 유틸리티

**파일**: `collect/utils.py` `[확인됨]`

#### 로깅

```python
get_logger(name)
# 콘솔 + logs/{name}.log 파일 동시 출력
```

#### 재시도 데코레이터

```python
@retry(max_attempts=5, base_delay=2.0, exceptions=(Exception,))
# 지수 백오프. 네트워크 오류, API 일시 장애 자동 대응.
```

#### 체크포인트

```python
Checkpoint(path)
  .is_done(key)           # 완료 여부 확인
  .mark_done(key, meta)   # 완료 기록 + 메타데이터 저장
  .reset(key)             # 특정 키 초기화
  .reset_all()            # 전체 초기화
```

JSON 파일 기반. 중단 후 재시작 시 완료된 국가/월 자동 건너뜀. feature builder와 inference 단계에도 동일한 패턴 적용을 권장한다. `[설계 필요]`

#### 날짜 분할 유틸리티

```python
date_range_months(start, end)
# → (month_start, month_end) 튜플 리스트
# BigQuery 월별 청크 분할에 사용
```

---

## 4. 현재 오케스트레이션 구조

**파일**: `collect/run_historical.py` `[확인됨]`

### CLI 인터페이스

```bash
python collect/run_historical.py \
  --sources acled gdelt economic \   # 수집할 소스 선택 (복수 가능)
  --start   2022-01-01 \             # 수집 시작일
  --end     2025-03-31 \             # 수집 종료일
  --dry-run                          # 실제 실행 없이 수집량 사전 확인
  --validate-only                    # 수집 건너뛰고 기존 파일 검증만 수행
```

유효 소스: `["acled", "gdelt", "economic"]` `[확인됨]`

### 실행 순서 `[확인됨]`

```
1. run_acled()
2. run_gdelt()
3. run_economic()
4. validate_collection()   # --dry-run이 아닐 때 자동 실행
```

순차 실행. 각 함수는 성공/실패 `bool` 반환. 검증 실패 시 `sys.exit(1)`.

### 검증 기준 `[확인됨]`

| 검증 대상 | 기준 |
|----------|------|
| ACLED | 57개 국가별 parquet 존재 여부 |
| GDELT | 57개 국가별 parquet 존재 여부 |
| Economic | `VIX`, `WTI`, `Gold`, `DXY`, `STLFSI4` 컬럼 존재 여부 |

### 향후 확장 지점 `[설계 필요]`

현재 `run_historical.py`는 대규모 초기 수집용이다. 일별 자동화를 위해 `run_daily_pipeline.py` 또는 동등한 스크립트를 별도로 작성하는 것이 적합하다. 해당 스크립트는 다음을 포함해야 한다:

- 최근 N일만 수집 (`collect_recent` 호출)
- feature table 갱신 트리거
- model inference 트리거
- dashboard output 갱신 트리거
- 실패 시 알림 또는 fallback 처리

---

## 5. Feature Table Schema 초안

> 최종 feature list는 모델이 확정된 이후에 고정된다. 이 섹션에서는 feature group과 key 컬럼 구조를 제안하는 것에 그친다.

### 파일 후보 `[설계 필요]`

```
input/processed/features_country_daily.parquet
```

### 필수 Key 컬럼

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `country` | VARCHAR(3) | ISO3 국가 코드 |
| `date` | DATE | 피처 기준일 (YYYY-MM-DD) |

### Feature Group 후보

| 그룹 | 설명 | 비고 |
|------|------|------|
| ACLED lag feature | 과거 N일 이벤트 수, 사망자 수, 이벤트 유형 비율 등 | lag 기준일 `[설계 필요]` |
| GDELT historical feature | Goldstein Scale, QuadClass, AvgTone 등 집계 | BQ raw 기반 |
| GDELT recent volume/tone | DOC 2.0 기반 기사량, 평균 톤 | 모델 입력 or briefing 전용 `[설계 필요]` |
| Economic level feature | VIX, WTI, Gold, DXY, STLFSI4 당일 값 | STLFSI4 ffill 적용 후 |
| Economic change/rolling feature | 전일 대비 변화율, N일 이동평균 등 | `[설계 필요]` |
| Time feature | month, week_of_year, day_of_week 등 | 계절성 처리 |
| Optional interaction feature | 지역-경제 상호작용 등 | 모델 담당자와 협의 |

### 예시 컬럼 후보

| 그룹 | 예시 컬럼 |
|------|----------|
| ACLED lag | `acled_event_count_lag7_7d`, `acled_fatalities_lag7_30d`, `acled_battle_count_lag7_14d` |
| GDELT historical | `gdelt_num_articles_1d`, `gdelt_avg_tone_7d`, `gdelt_goldstein_mean_7d`, `gdelt_num_mentions_7d` |
| GDELT recent | `gdelt_doc_volume_1d`, `gdelt_doc_volume_zscore_30d`, `gdelt_doc_tone_7d` |
| Economic | `VIX`, `WTI`, `Gold`, `DXY`, `STLFSI4`, `VIX_7d_change`, `WTI_7d_change` |
| Time | `day_of_week`, `month`, `week_of_year` |
| Optional interaction | `gdelt_tone_x_vix`, `acled_event_x_news_volume` |

위 컬럼들은 최종 확정 feature가 아니라 feature schema 논의를 위한 예시 후보이다. 최종 feature list와 column order는 모델 담당자가 확정한 model artifact 및 feature_list.json에 맞춰 고정한다.

### 주의사항

- **최종 모델이 확정되면 feature list와 column order를 artifact로 고정해야 한다.** 학습 시 feature와 inference 시 feature가 반드시 일치해야 한다. `[설계 필요]`
- 결측치 처리(zero-fill / forward-fill / indicator column) 기준을 feature builder에서 명시적으로 처리해야 한다. `[설계 필요]`
- timezone과 date alignment 기준(UTC 기준일 cut-off 시각 등)을 통일해야 한다. `[설계 필요]`
- STLFSI4는 주별 데이터이므로 feature builder에서 일별 forward-fill 처리가 필요하다. `[확인됨]`
- ACLED 결측 여부를 별도 컬럼(예: `acled_missing_flag`)으로 명시하는 방식을 고려한다. `[설계 필요]`
- 이 문서는 특정 모델 실험의 피처 구성을 최종값으로 단정하지 않는다.

---

## 6. Model Inference Interface 초안

> 이 섹션은 특정 모델 설명이 아니라 인터페이스 계약 중심으로 작성한다. 최종 모델은 팀원이 추후 확정한다. 이 문서에서는 모델을 black-box로 보고, 입력/출력 인터페이스만 정의한다.

### 입력

```
input/processed/features_country_daily.parquet
  └── 컬럼: country, date, [feature columns...]
```

### Model Artifact 후보 `[설계 필요]`

```
artifacts/final_model.pkl        # 직렬화된 모델 객체
artifacts/feature_list.json      # 학습 시 사용한 feature 이름과 순서
artifacts/model_metadata.json    # 모델 이름, 버전, 학습 날짜, 타깃 변수 등
```

### 출력 후보 `[설계 필요]`

```
outputs/predictions/model_predictions.parquet
또는 DB table: model_predictions
```

### 출력 Schema

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `country` | VARCHAR(3) | ISO3 국가 코드 |
| `date` | DATE | 예측 대상일 |
| `model_name` | VARCHAR | 모델 식별자 |
| `model_version` | VARCHAR | 모델 버전 태그 |
| `predicted_probability` | FLOAT [0, 1] | 모델 출력 확률값 |
| `created_at` | TIMESTAMP | 예측 생성 시각 |

> **`model_predictions`와 `dashboard_country_risk`의 관계**
> - `model_predictions`는 모델 원본 출력 보관용이다. 후처리 없이 그대로 저장한다.
> - `dashboard_country_risk`는 risk scoring과 후처리(temperature_score, risk_level, delta 등)가 적용된 대시보드 서빙용 output이다.
> - 따라서 risk scoring 방식을 바꾸거나 대시보드 스키마를 변경하더라도, `model_predictions`의 원본 예측값은 보존되어 재가공이 가능하다.

### 설계 원칙

- LightGBM, stacking, LSTM, 앙상블 등 어떤 모델이 와도 외부 인터페이스는 동일하게 유지한다.
- 내부 모델 구조가 달라져도 `predicted_probability`만 반환하면 이후 pipeline은 수정 없이 유지된다.
- 모델 원본 출력과 대시보드용 점수(risk_level, temperature_score 등)는 분리해서 저장한다.
- 모델 상세 구조, 성능 지표, ablation 결과는 모델 담당자 문서에서 별도 관리한다.

---

## 7. Risk Scoring 후보

> 리스크 스코어링 방식은 아직 확정하지 않는다. 아래는 팀 회의에 가져갈 후보안이다. `predicted_probability`는 실제 절대 발생 확률로 단정하지 않고, 모델 기반 단기 위험 신호 점수로 해석한다.

---

### Candidate 1. Probability-only Score

```
temperature_score = predicted_probability × 100
```

**장점**: 구현 단순, 모델 출력 직접 반영, 해석 직관적
**단점**: 확률값 자체의 절대 수준에 의존 — calibration 품질에 민감, 국가 간 비교 어려움

---

### Candidate 2. Model-centered Composite Score

```
temperature_score = 100 × (0.75 × predicted_probability
                          + 0.15 × news_signal
                          + 0.10 × event_signal)
```

**장점**: 모델 확률 외 뉴스/이벤트 신호를 반영, 단일 지표 과의존 완화
**단점**: 가중치 결정 근거 필요, 신호 정의와 집계 방식을 별도로 확정해야 함

---

### Candidate 3. Rank-based Score

```
temperature_score = percentile_rank(predicted_probability) × 100
```

**장점**: 국가 간 상대 비교에 유리, 모델 calibration 품질과 독립적
**단점**: 절대 위험 수준 정보 소실, 전체 위험이 낮은 시기에도 상위 국가가 고위험으로 표시될 수 있음

---

**MVP 기본 후보**: Candidate 1로 출발할 수 있으나, **최종 선택은 팀 회의에서 결정한다.**

---

## 8. Dashboard Output Schema 초안

> 대시보드는 모델 내부를 알 필요 없이 이 schema만 읽으면 된다. 이 schema가 pipeline의 최종 계약이다.

### Table / File 이름 후보 `[설계 필요]`

```
파일: public/data/dashboard_country_risk.csv
      public/data/dashboard_country_risk.json
DB:   dashboard_country_risk
```

### Schema

| 컬럼 | 타입 | 설명 | 생성 단계 | 비고 |
|------|------|------|----------|------|
| `country` | VARCHAR(3) | ISO3 국가 코드 | Model Inference | 필수 |
| `date` | DATE | 예측 대상일 | Model Inference | 필수 |
| `model_name` | VARCHAR | 사용된 모델 식별자 | Model Inference | 필수 |
| `model_version` | VARCHAR | 모델 버전 태그 | Model Inference | 필수 |
| `predicted_probability` | FLOAT [0,1] | 모델 원본 출력 확률 | Model Inference | 필수 |
| `temperature_score` | FLOAT [0,100] | risk scoring 후보 적용 결과 | Risk Scoring | 방식 확정 필요 |
| `risk_level` | VARCHAR | 대시보드 표시용 위험 등급 레이블 | Risk Scoring | 단계 수 및 threshold 확정 필요 |
| `delta_1d` | FLOAT | 전일 대비 temperature_score 변화 | Risk Scoring | |
| `delta_7d` | FLOAT | 전주 대비 temperature_score 변화 | Risk Scoring | |
| `rank_today` | INT | 당일 전체 국가 중 위험 순위 | Risk Scoring | |
| `confidence_level` | VARCHAR | 데이터 품질/커버리지 힌트 — 모델 confidence가 아니라 수집 충분성/결측 여부를 요약한 운영상 플래그 (예: `data_sparse`) | Risk Scoring | Optional; `data_quality_flag`로 변경 후보 |
| `main_driver_1` | VARCHAR | 주요 위험 요인 1위 | Post-processing | MVP에서는 Optional |
| `main_driver_2` | VARCHAR | 주요 위험 요인 2위 | Post-processing | MVP에서는 Optional |
| `main_driver_3` | VARCHAR | 주요 위험 요인 3위 | Post-processing | MVP에서는 Optional |
| `updated_at` | TIMESTAMP | 해당 행의 생성 시각 | Export 단계 | 필수 |

### risk_level 단계 후보

| 후보 | 단계 |
|------|------|
| Option A | `Low` / `Medium` / `High` |
| Option B | `Low` / `Moderate` / `High` / `Critical` |

최종 label 수와 threshold는 risk scoring 방식 및 대시보드 표현 방식에 맞춰 팀 회의에서 결정한다.

### 설계 원칙

- 대시보드는 `model_name`과 `model_version`만 참조하면 되고, 모델 내부 구조를 알 필요 없다.
- `risk_level` 임계값 및 단계 수는 아직 팀에서 결정해야 한다. `[설계 필요]`
- `confidence_level`이라는 이름은 모델의 통계적 확신도로 오해될 수 있다. 이 컬럼은 데이터 희소성, 최근 수집 실패 여부, 주요 source 결측 여부 등을 요약하는 운영상 품질 플래그다. 팀 회의에서 `data_quality_flag`로 컬럼명을 바꿀지 결정한다. `[설계 필요]`
- `main_driver`는 MVP에서는 optional로 두고, 이후 feature importance 기반으로 생성할 수 있다. `[설계 필요]`
- CSV/JSON 기반 MVP와 DB 기반 확장안을 모두 열어둔다. `[설계 필요]`
- 파티션 키 후보: `date` (날짜 기준 조회가 많을 경우)

---

## 9. 파이프라인 실행 옵션 비교

### Option A. Batch-first Pipeline (현재 권장)

`run_historical.py`는 초기 historical backfill 및 대규모 과거 수집 전용이다. 실제 운영 또는 준실시간 갱신은 별도 `run_daily_pipeline.py`를 두는 구조가 적합하다. 두 역할을 분리하면 과거 데이터 재수집 로직과 일별 업데이트 로직이 섞이지 않는다.

**초기 구축 (historical backfill)**

- 실행 목적: 과거 기간 raw data를 처음 수집하거나 재수집할 때
- 실행: `python collect/run_historical.py --sources acled gdelt economic --start {start_date} --end {end_date}`
- 스크립트: `collect/run_historical.py` `[확인됨]`

**운영/갱신 (daily update)**

- 실행 목적: 최근 데이터 수집부터 대시보드 output 갱신까지 일괄 실행
- 실행: `python run_daily_pipeline.py` `[설계 필요]`
- daily pipeline은 각 collector의 `collect_recent` 계열 함수를 호출하는 방향이 적합하다.
- 포함 단계:
  1. `collect_recent` — 각 collector에서 최근 N일 증분 수집
  2. `build_features` — feature table 갱신 `[설계 필요]`
  3. `run_inference` — model artifact 로드 후 예측 생성 `[설계 필요]`
  4. `run_risk_scoring` — temperature_score, risk_level 등 파생 컬럼 생성 `[설계 필요]`
  5. `export_dashboard` — dashboard_country_risk 출력 갱신 `[설계 필요]`

**장점**
- 구현 단순, 추가 인프라 불필요
- 현재 repository 구조(스크립트 기반)와 잘 맞음
- 발표/MVP에 적합
- 수집 실패 시 체크포인트 기반 재시도와 복구가 쉬움

**단점**
- 완전한 실시간 API 구조는 아님
- batch 실행 주기에 따라 최신성이 결정됨
- 단계 실패 시 이후 단계를 수동으로 재시작해야 함

---

### Option B. Service-oriented / Docker-based Pipeline

```
[컨테이너 구성 예시]
  collector-container        수집 담당
  feature-builder-container  피처 빌딩 담당
  model-inference-container  모델 추론 담당
  api-server-container       또는 dashboard-exporter 담당
  frontend-container         대시보드 UI
  database                   공유 저장소 (PostgreSQL 등)
```

**장점**
- 팀원이 가져올 Docker/DB 구성과 연결하기 좋음
- 모듈별 독립 교체 가능
- API 기반 대시보드로 확장 가능
- 태스크별 재시도 및 모니터링 용이

**단점**
- MVP에는 구현 부담이 큼
- 컨테이너 간 dependency 관리 필요
- 발표 전 안정화 비용이 큼

---

**최종 제안**

- **1차 구현은 Option A로 진행**한다.
- Schema와 모듈 경계는 Option B로 확장 가능하게 설계해 둔다.
- Docker/DB 통합 시점에는 각 스크립트를 컨테이너 진입점으로 래핑하는 방식으로 전환한다.

---

## 10. 팀 회의에서 결정해야 할 사항

> 아래 항목은 이 문서가 결정하지 않으며, 팀 회의에서 합의 후 이 문서에 반영한다.

**모델 / 아티팩트**
- [ ] 최종 모델 종류와 artifact 저장 형식 (`.pkl`, `.pt`, 기타)
- [ ] 최종 feature list와 column order (학습-inference 동기화 기준)

**Feature Builder**
- [ ] ACLED lag 기준 (3일 / 7일 / 14일 등)
- [ ] GDELT recent feature를 모델 입력에 넣을지, briefing용으로만 쓸지
- [ ] economic indicator feature 변환 방식 (level 그대로 / 변화율 / rolling 등)
- [ ] 결측치 처리 기준 (zero-fill / ffill / indicator column)

**Risk Scoring / Dashboard**
- [ ] risk scoring 후보(Candidate 1/2/3) 중 MVP 적용 방식
- [ ] `risk_level`을 3단계(`Low/Medium/High`)로 할지 4단계(`Low/Moderate/High/Critical`)로 할지
- [ ] `risk_level` threshold (각 단계의 경계값)
- [ ] `confidence_level` 컬럼명을 유지할지 `data_quality_flag`로 바꿀지 (모델 통계적 신뢰도가 아닌 데이터 품질/커버리지 플래그임을 명확히)
- [ ] `main_driver` 생성 방식 (feature importance 기반 등)
- [ ] dashboard output을 CSV/JSON으로 둘지 DB로 둘지
- [ ] daily batch 주기 (매일 / 매주 / 수동)
- [ ] 수집 실패 시 fallback 정책 (이전 날짜 유지 / 에러 표시 등)

**인프라**
- [ ] Docker 통합 시 container boundary
- [ ] `run_historical.py`와 별도로 daily pipeline script(`run_daily_pipeline.py`)를 만들지, 단일 스크립트로 통합할지

**Feature 확정**
- [ ] 위 예시 컬럼들은 초안이며, 최종 feature list는 모델 artifact(`feature_list.json`)와 함께 확정해야 한다는 점 팀 공유

---

## 11. 이후 구현 단계

1. **collector output schema 최종 확인** — raw parquet 컬럼/타입/날짜 형식 검증
2. **feature table schema 확정** — key 컬럼, feature group, 결측 처리 기준 고정
3. **feature builder 초안 작성** — raw → `features_country_daily.parquet` 변환
4. **model loader / inference interface 작성** — artifact 로드 → 예측 생성 → `model_predictions` 저장
5. **risk scoring 함수 작성** — `model_predictions`를 입력으로 받아 `dashboard_country_risk` schema로 변환 (팀 회의에서 선택된 후보 구현)
6. **dashboard output exporter 작성** — `dashboard_country_risk`를 CSV/JSON/DB 중 선택한 방식으로 내보내기
7. **팀 회의에서 선택지 확정** — risk scoring, risk_level 임계값, 서빙 방식
8. **Docker/DB 확정 후 연결** — 각 스크립트를 컨테이너 진입점으로 래핑

---

## 12. 참고 파일

### 핵심 참고 파일 (이 문서 작성 기준)

| 파일 | 내용 |
|------|------|
| `collect/config.py` | 57개국 ISO2/ISO3/GDELT FIPS 매핑, 수집 기간, API 설정 |
| `collect/acled_collector.py` | ACLED 수집 전체 구현 |
| `collect/gdelt_collector.py` | GDELT BQ / DOC 2.0 수집 전체 구현 |
| `collect/economic_collector.py` | 경제지표 수집 전체 구현 |
| `collect/run_historical.py` | 수집 오케스트레이터 |
| `collect/utils.py` | 로거, retry 데코레이터, Checkpoint 클래스 |
| `docs/eda-summary.md` | 데이터셋 EDA 요약 (양성률, 기간, 국가 분포) |
| `docs/data-sources.md` | ACLED API 상세 스펙 |

### 모델 담당자 참고 문서 (이 문서 범위 외)

| 파일 | 내용 |
|------|------|
| `model/README_D_category_stacking.md` | D-category 모델 구조 및 실험 결과 |
| `model/run_stacking_d_with_mask_feature_ablation.py` | 모델 학습 스크립트 |
| `docs/model-study.md` | 4개 모델 카테고리 설계 |
| `outputs/reports/d_category_team_report.md` | D-category 절제 실험 전체 결과 |
