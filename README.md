# 무력충돌 예측 조기경보 시스템

> KUBIG Conference — E2E with VIBE 4팀

뉴스·경제·분쟁 이벤트 데이터를 통합하여 국가별 단기 무력충돌 발생 가능성을 위험도 점수로 산출하고, 세계지도 대시보드로 서비스하는 E2E 파이프라인.

- **라이브 대시보드**: https://hong-junki.github.io/KUBIG_Conference-team-4/
- **대상**: 58개국, 2014-01-01 ~ 현재 (국가 x 일 단위)
- **최종 배포 모델**: 단일 LSTM — 분쟁 onset(신규 발생) 예측. 실시간 입력은 GDELT·경제지표만 사용 (ACLED 비의존)
- **현재 단계**: 프로젝트 완료 (컨퍼런스 발표 종료). 정기 갱신 스케줄은 중지 상태이며, 각 GitHub Actions 워크플로의 수동 실행(workflow_dispatch)으로 재가동 가능

---

## 시스템 구성

```
[수집]   GDELT events·뉴스 제목 / 경제지표  ──→  BigQuery raw          (GH Actions, 15분·일 단위)
[채점]   피처 빌드 → LSTM 채점             ──→  BigQuery model_scores  (GH Actions, 매시)
[동기화] BigQuery → Supabase               ──→  risk_scores_live.json  (GH Actions, 15분)
[서비스] Telegram 수집·LLM 요약 → 대시보드 재생성 → GitHub Pages       (GH Actions, 15분)
```

- 서빙 테이블·JSON 계약: [`SERVING_CONTRACTS.md`](SERVING_CONTRACTS.md)
- 채점·동기화 자동화 구성: [`SERVING_AUTOMATION.md`](SERVING_AUTOMATION.md)
- 대시보드 점수 표시 정책: [`docs/dashboard-scoring-policy.md`](docs/dashboard-scoring-policy.md)

## 최종 모델

- **단일 LSTM (3시드 평균)** — 국가 x 일 시계열로 분쟁 onset(신규 발생) 확률을 예측
- walk-forward 검증 PR-AUC **0.173** (calm 국가 대상, 무작위 기준 대비 약 2.5배), 상위 2% 경보 정밀도 27%
- **ACLED는 라벨·학습 전용**: ACLED 갱신 지연 시 성능이 급락하는 실험 결과에 따라, 실시간 추론은 GDELT(이벤트 + 뉴스 제목 임베딩)·경제지표만으로 동작
- 점수 2분리 정책: 지도 색상 = `current_risk`(현재 위험), 조기경보 = `onset_alert`(신규 발생 경보)
- 실험 과정(escalation 타겟 비교, 트리·시퀀스·스태킹 후보군)은 [`model/`](model/) 노트북·리포트 참조

## 대시보드

https://hong-junki.github.io/KUBIG_Conference-team-4/

- 세계지도(지구본/평면) 국가별 위험도 시각화 + 국가 상세 패널(모델 확률·신호)
- GDELT·Telegram 기반 LLM 한국어 요약 브리핑
- 한국어 검색(국가명·키워드)
- 생성기: `Telegram/scripts/build_live_osint_site.py` (15분 주기 재생성 구조)

---

## Quick Start

### 1. 환경 세팅

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # 용도별: requirements-collection.txt / requirements-serve.txt
cp .env.example .env              # API 키 입력
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
python -m src.process.feature_builder    # 피처 생성
python -m src.process.label_builder      # y / y_onset / y_escalation
python -m src.process.build_dataset      # train/val/test split
```

데이터 흐름: `input/raw/` (수집 원본) → `input/raw_merged/` (학습용 통합본) → `input/processed/{features,labels,dataset}/`.

### 4. 실시간 채점 (서빙)

```bash
python -m src.serve.build_model_input --start <D-95일> --end <D> --run-ts <ISO시각>
python -m src.serve.run_scoring --run-ts <같은 ISO시각>
```

Docker 이미지: [`Dockerfile.serve`](Dockerfile.serve). 운영 시에는 GitHub Actions `model-scoring.yml`이 같은 절차를 매시 실행.

---

## 데이터 소스

| 소스 | 역할 | 갱신 주기 | 비고 |
|---|---|---|---|
| **ACLED** | 라벨 + lag 피처 (학습 전용) | 주 1회 | OAuth 2.0. 실시간 추론에는 미사용 |
| **GDELT events** | 실시간 뉴스 이벤트 신호 | 15분 | BigQuery `events_partitioned` |
| **GDELT 뉴스 제목(GKG)** | 다국어 제목 임베딩·앵커 유사도 피처 | 15분 | BigQuery 적재 |
| **경제지표** | VIX, WTI, Gold, DXY, STLFSI4 | 일 1회 | yfinance + FRED |
| **Telegram** | OSINT 채널 → LLM 요약 브리핑 | 15분 | 대시보드 브리핑 전용 (모델 입력 아님) |

---

## 폴더 구조

```
├── README.md
├── SERVING_CONTRACTS.md            # 서빙 테이블·JSON 계약
├── SERVING_AUTOMATION.md           # 채점·동기화 자동화 문서
├── Dockerfile.serve / entrypoint_serve.sh
├── requirements.txt                # + requirements-collection.txt / requirements-serve.txt
├── sync_bq_model_scores_to_supabase.py        # BQ → Supabase 동기화 진입점
├── export_supabase_risk_scores_for_pages.py   # Supabase → 라이브 JSON 진입점
├── .github/workflows/  # 수집·채점·동기화·대시보드 재생성 워크플로 4종
├── docs/               # 수집 검증 리포트, 파이프라인 감사, 대시보드 점수 정책
├── src/
│   ├── collect/        # ACLED / GDELT / 경제지표 수집 (+ incremental/ 증분 수집)
│   ├── process/        # 병합·전처리·피처·라벨·데이터셋
│   ├── model/          # 학습·평가 파이프라인
│   ├── serve/          # 실시간 피처 빌드·채점
│   └── dashboard/      # 대시보드 API·프론트엔드
├── model/              # 모델 실험 노트북·리포트 (src/model 파이프라인과 별개)
├── scripts/            # 학습·EDA·임베딩 보조 스크립트
├── Telegram/           # 대시보드 사이트(GitHub Pages) + Telegram 수집·LLM 요약
├── migrations/         # Supabase 스키마
├── input/ · output/    # 데이터·산출물 (실행 시 생성)
└── tests/
```

---

## 데이터 분할 (지도학습 본선 기준)

| Split | 기간 | 행수 | y_escalation 양성률 |
|---|---|---|---|
| train | 2014-01-01 ~ 2023-12-31 | 211,816 | 4.29% |
| val | 2024-01-01 ~ 2024-06-30 | 10,556 | 4.07% |
| test | 2024-07-01 ~ 2025-03-28 | 15,718 | 4.06% |

`test_end = 2025-03-28`은 ACLED 라벨 컷오프(주 1회 갱신 + 3일 lookahead) 기준. 최종 onset 모델은 위 고정 분할 대신 walk-forward 다구간 검증으로 선정.

---

## 핵심 규칙

1. **타임존**: 모든 timestamp UTC 통일 후 국가x일 집계
2. **ACLED 피처 lag**: 주 1회 갱신 → 피처로 사용 시 최소 7일 lag (라벨로는 lag 없이)
3. **수집 idempotent**: 모든 수집 스크립트 체크포인트 기반 재시작 가능
4. **`input/raw/` 보호**: 명시적 지시 없이 수정 금지

---

## 자동화 (GitHub Actions)

| 워크플로 | 역할 | 주기 (운영 시) |
|---|---|---|
| `incremental-data-collection.yml` | GDELT·경제지표 증분 → BigQuery | 15분 / 일 / 주 |
| `model-scoring.yml` | 피처 빌드 + LSTM 채점 → BigQuery (+ 일 1회 임베딩 means 갱신) | 매시 |
| `sync_scores_to_supabase.yml` | BigQuery model_scores → Supabase | 15분 |
| `telegram-dashboard-refresh.yml` | Telegram 수집·LLM 요약·대시보드 재생성 | 15분 |

컨퍼런스 종료 후 정기 스케줄은 중지 상태. 각 워크플로는 GitHub → Actions → Run workflow로 수동 실행 가능.

- 증분 수집 파이프라인 문서: [`docs/incremental-data-collection.md`](docs/incremental-data-collection.md)
- 수집 현황 감사: [`docs/data-pipeline-audit.md`](docs/data-pipeline-audit.md)

### 로컬 dry-run (BQ 쓰기 없음)

```bash
python -m src.collect.incremental.run_incremental --sources gdelt economic --dry-run
```

---

## 데이터 수집 가이드

수집기별 옵션·체크포인트 위치·재시작 절차는 각 모듈 docstring 참조:

- `src/collect/acled_collector.py`
- `src/collect/gdelt_collector.py`
- `src/collect/economic_collector.py`
- `src/collect/run_historical.py` (통합 진입점 — 과거 수집)
- `src/collect/incremental/run_incremental.py` (증분 수집 진입점)
