# 무력충돌 예측 조기경보 시스템

KUBIG Conference · E2E with VIBE 4팀

**뉴스·경제 데이터 → 국가별 무력충돌 신규 발생(onset) 위험 예측 → 세계지도 대시보드**

라이브 대시보드: https://hong-junki.github.io/KUBIG_Conference-team-4/

GDELT 뉴스 신호와 경제지표만으로 58개국의 분쟁 신규 발생 확률을 매시 채점하고, 지도와 한국어 브리핑 대시보드로 서비스하는 E2E 파이프라인. 2014년~현재, 국가 x 일 단위.

## 프로젝트 개요: 무엇으로 예측하는가

| 데이터 | 역할 | 갱신 |
|---|---|---|
| ACLED (분쟁 이벤트 DB) | 정답 라벨 (학습 전용) | 주 1회 |
| GDELT 이벤트·뉴스 제목 | 실시간 위험 신호 (다국어 제목 임베딩·앵커 유사도) | 15분 |
| 경제지표 (VIX·WTI·Gold·DXY·금융스트레스) | 시장의 위험 선반영 신호 | 일 1회 |
| Telegram OSINT | LLM 한국어 브리핑 전용 (모델 입력 아님) | 15분 |

파이프라인은 GitHub Actions로 전 구간 자동:

```
수집(15분) → BigQuery → 피처 빌드·LSTM 채점(매시) → Supabase 동기화(15분) → 대시보드 재생성(15분)
```

컨퍼런스 종료 후 정기 스케줄은 중지 상태이며, 각 워크플로의 수동 실행(Run workflow)으로 재가동할 수 있다.

## 핵심 발견

1. **정답 데이터는 실시간에 못 쓴다.** ACLED는 주 1회 갱신이라 3일만 묵어도 PR-AUC가 70% 급락한다. 그래서 실시간 추론은 ACLED 없이 GDELT·경제지표만으로 동작하도록 재설계했다.
2. **"악화"는 뉴스에 전조가 안 남는다.** 악화(escalation) 예측은 어제 상태를 복사하는 기준선을 사실상 못 이겼다(성능 상한 0.086). 뉴스로 잡을 수 있는 **신규 발생(onset)**으로 문제를 다시 정의하자 성능이 2배로 뛰었다.
3. **결합보다 단일 모델.** 7개 모델을 스태킹으로 결합한 것(0.115)보다 단일 LSTM(0.173)이 높았고, walk-forward 6구간 전부에서 기준선을 상회했다. 복잡도가 성능이 아니다.

## 최종 모델 · 결과

단일 LSTM(3시드 평균)이 국가 x 일 시계열을 입력받아 onset 확률을 출력한다.

| 후보 | onset PR-AUC (walk-forward) |
|---|---|
| **LSTM (채택)** | **0.173** |
| BiLSTM | 0.157 |
| GRU | 0.150 |
| TCN / 7모델 스태킹 | 0.115 |
| 트리 (XGB·LGBM) | 0.091~0.098 |

무작위 기준(양성률 6.9%) 대비 약 2.5배, 상위 2% 경보의 정밀도 27%. 실험 과정(escalation 비교, 트리·시퀀스·스태킹 후보군)은 [`model/`](model/)의 노트북·리포트 참조.

대시보드 점수는 2종으로 분리했다: 지도 색상 = `current_risk`(현재 위험 수준), 조기경보 = `onset_alert`(신규 발생 임박 신호). 정책 상세: [`docs/dashboard-scoring-policy.md`](docs/dashboard-scoring-policy.md)

## 대시보드

https://hong-junki.github.io/KUBIG_Conference-team-4/

- 세계지도(지구본/평면) 국가별 위험도 + 국가 상세 패널(모델 확률·신호)
- GDELT·Telegram 기반 LLM 한국어 요약 브리핑, 한국어 검색(국가명·키워드)
- 매시 채점 결과가 15분 주기로 반영되는 구조 (생성기: `Telegram/scripts/build_live_osint_site.py`)

## 폴더 구조

```
├── src/
│   ├── collect/        # ACLED·GDELT·경제지표 수집 (+ incremental/ 증분 수집)
│   ├── process/        # 병합·전처리·피처·라벨·데이터셋
│   ├── model/          # 학습·평가 파이프라인
│   └── serve/          # 실시간 피처 빌드·채점        → SERVING_AUTOMATION.md
├── model/              # 모델 실험 노트북·리포트 (src/model 파이프라인과 별개)
├── scripts/            # 학습·EDA·임베딩 보조 스크립트
├── Telegram/           # 대시보드 사이트(GitHub Pages) + Telegram 수집·LLM 요약
├── docs/               # 수집 검증 리포트, 파이프라인 감사, 대시보드 점수 정책
├── .github/workflows/  # 수집·채점·동기화·대시보드 재생성 워크플로 4종
├── migrations/         # Supabase 스키마
├── Dockerfile.serve    # 채점 컨테이너 (계약 문서: SERVING_CONTRACTS.md)
├── input/ · output/    # 데이터·산출물 (실행 시 생성)
└── tests/
```

## 빠른 시작

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # 용도별: requirements-collection.txt / requirements-serve.txt
cp .env.example .env                 # ACLED·GCP·FRED 키 입력
```

과거 데이터 수집 → 학습 데이터셋:

```bash
python -m src.collect.run_historical --start 2014-01-01 --end 2026-03-31
python -m src.process.merge_raw
python -m src.process.preprocess
python -m src.process.feature_builder
python -m src.process.label_builder
python -m src.process.build_dataset
```

실시간 채점 (운영 시에는 GitHub Actions `model-scoring.yml`이 매시 같은 절차를 실행):

```bash
python -m src.serve.build_model_input --start <D-95일> --end <D> --run-ts <ISO시각>
python -m src.serve.run_scoring --run-ts <같은 ISO시각>
```

Docker 이미지: [`Dockerfile.serve`](Dockerfile.serve)

## 크레딧 · 데이터 출처

- 데이터: [ACLED](https://acleddata.com/) (학술 이용약관 준수) · [GDELT Project](https://www.gdeltproject.org/) · FRED · yfinance
- 재현·참고: Macis et al. (2024) LSTM Autoencoder (초기 재현 후 피처 추출기로 활용)
- 코드는 팀 자체 구현. ACLED 원본 데이터는 라이선스 정책상 레포에 포함하지 않는다.
