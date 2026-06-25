# 무력충돌 예측 조기경보 시스템

> KUBIG Conference — E2E with VIBE 4팀

뉴스·경제·분쟁 이벤트 데이터를 통합하여 단기간 내 무력충돌 발생 가능성을 0-100 위험도 점수로 산출하는 E2E 파이프라인.

- **타겟**: `y_escalation` — 다음 3일 내 분쟁 발발 또는 급격한 악화 (양성 ~4.3%)
- **데이터 범위**: 2014-01-01 ~ 2026-03-28 (58개국, 일단위, 259,260행)
- **현재 단계**: 데이터 수집·전처리 및 탐색적 데이터 분석(EDA) 완료. 모델링 진행 중.

---

## Quick Start

### 1. 환경 세팅

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # API 키 입력
```

필요한 키: `ACLED_USERNAME`, `ACLED_PASSWORD`, `GOOGLE_APPLICATION_CREDENTIALS`, `FRED_API_KEY`.

### 2. 데이터 수집 (과거)

```bash
python -m src.collect.run_historical --start 2014-01-01 --end 2026-03-31
python -m src.collect.run_historical --validate-only
```

소스별 산출: `input/raw/{acled,gdelt,economic}/`. 체크포인트 기반 재시작 가능.

### 3. 병합 · 전처리 · 피처 · 라벨 · 데이터셋

```bash
python -m src.process.merge_raw          # input/raw/ → input/raw_merged/ (다회차 통합본)
python -m src.process.preprocess         # 국가x일 집계
python -m src.process.feature_builder    # 54개 피처
python -m src.process.label_builder      # y / y_onset / y_escalation
python -m src.process.build_dataset      # train/val/test split
```

데이터 흐름: `input/raw/` (수집 원본) → `input/raw_merged/` (학습용 통합본) → `input/processed/{features,labels,dataset}/`.

### 4. 탐색적 데이터 분석

```bash
python scripts/eda_2014_2026.py
```

산출: `output/evaluation/eda-plots/`에 PNG 7장 + CSV 3장 생성.

---

## 데이터 소스

| 소스 | 역할 | 갱신 주기 | 비고 |
|---|---|---|---|
| **ACLED** | 분쟁 이벤트(라벨 + 피처) | 주 1회 | OAuth 2.0, 피처 사용 시 7일 lag 필수 |
| **GDELT** | 글로벌 뉴스 시그널 | 15분 | BigQuery `events_partitioned` |
| **경제지표** | VIX, WTI, Gold, DXY, STLFSI4 | 일 1회 | yfinance + FRED |
| (선택) Reddit, Telegram | OSINT 보강 | - | 미적용 |

---

## 폴더 구조

```
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── collect/        # ACLED / GDELT / 경제지표 수집
│   ├── process/        # 병합·전처리·피처·라벨·데이터셋
│   ├── model/          # (모델링 단계 진행 예정)
│   └── dashboard/      # (대시보드 단계 진행 예정)
├── scripts/
│   ├── eda_2014_2026.py        # EDA 실행 스크립트
│   └── merge_gdelt_2016_2017.py
├── input/              # 수집 원본 + 전처리 산출 (실행 시 생성)
├── output/             # 학습·평가 산출물 (실행 시 생성)
└── tests/
```

---

## 데이터 분할

| Split | 기간 | 행수 | y_escalation 양성률 |
|---|---|---|---|
| train | 2014-01-01 ~ 2023-12-31 | 211,816 | 4.29% |
| val | 2024-01-01 ~ 2024-06-30 | 10,556 | 4.07% |
| test | 2024-07-01 ~ 2025-03-28 | 15,718 | 4.06% |

`test_end = 2025-03-28`은 ACLED 라벨 컷오프(주 1회 갱신 + 3일 lookahead) 기준.

---

## 핵심 규칙

1. **타임존**: 모든 timestamp UTC 통일 후 국가x일 집계
2. **ACLED 피처 lag**: 주 1회 갱신 → 피처로 사용 시 최소 7일 lag (라벨로는 lag 없이)
3. **수집 idempotent**: 모든 수집 스크립트 체크포인트 기반 재시작 가능
4. **`input/raw/` 보호**: 명시적 지시 없이 수정 금지

---

## 증분 수집 파이프라인

과거 수집 완료 후 최신 데이터만 주기적으로 BigQuery에 적재하는 파이프라인.

- 파이프라인 문서: [`docs/incremental-data-collection.md`](docs/incremental-data-collection.md)
- 현황 감사: [`docs/data-pipeline-audit.md`](docs/data-pipeline-audit.md)
- GitHub Actions workflow: [`.github/workflows/incremental-data-collection.yml`](.github/workflows/incremental-data-collection.yml)
- **현재 schedule 상태: 비활성** (수동 실행 `workflow_dispatch`만 가능)

### 로컬 dry-run (BQ 쓰기 없음)

```bash
cp .env.example .env  # GCP_PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS 등 입력
python -m src.collect.incremental.run_incremental --sources gdelt economic --dry-run
```

### GitHub Actions 수동 실행

GitHub → Actions → Incremental Data Collection → Run workflow

---

## 데이터 수집 가이드

수집기별 옵션·체크포인트 위치·재시작 절차는 각 모듈 docstring 참조:

- `src/collect/acled_collector.py`
- `src/collect/gdelt_collector.py`
- `src/collect/economic_collector.py`
- `src/collect/run_historical.py` (통합 진입점 — 과거 수집)
- `src/collect/incremental/run_incremental.py` (증분 수집 진입점)
