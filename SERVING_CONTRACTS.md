# 서빙 파이프라인 인터페이스 계약 (BQ 테이블 스키마)

> 서빙 파이프라인 단계 간 인터페이스를 BQ 테이블 스키마로 고정한다. 각 단계는 이 스키마에 맞춰 개발한다.

## 파이프라인 전체 흐름
```
[raw: gdelt_titles/events/econ (BQ)]  ← 수집 담당
   → [임베딩 적재: OpenAI→BQ gkg_embeddings]  ← 피처/모델 담당(src/serve/build_model_input)
   → [피처 빌더 → BQ model_input]              ← 피처/모델 담당
   → [스코어러 → BQ model_scores]              ← 피처/모델 담당(src/serve/run_scoring)
   → [risk_score 계산 → Supabase]              ← 스코어링 담당
   → [대시보드]                                ← 대시보드 담당
```
모델(torch+sklearn)은 **BQ 안에서 못 돈다** → 컨테이너(Cloud Run/GitHub Actions)가 BQ를 읽고 BQ에 쓴다.

---

## 1. `conflict_ew.model_input` (피처) — 피처/모델 담당 산출
| 컬럼 | 타입 | 설명 |
|---|---|---|
| country | STRING | ISO3 |
| date | DATE | 국가-일 (UTC) |
| run_ts | TIMESTAMP | 파이프라인 실행 시각 (최신 run 선택용) |
| `<419 피처>` | FLOAT64 | 모델 입력 (gkg_*, gdelt_*, gdev_*, econ_* 등). ACLED 피처 없음 |

- 최신값 선택: `QUALIFY ROW_NUMBER() OVER (PARTITION BY country,date ORDER BY run_ts DESC)=1`

## 2. `conflict_ew.model_scores` (모델 출력) — 피처/모델 담당 산출 → **스코어링 담당 입력**
| 컬럼 | 타입 | 설명 |
|---|---|---|
| country | STRING | ISO3 |
| date | DATE | 국가-일 |
| run_ts | TIMESTAMP | 실행 시각 |
| base_pred | FLOAT64 | LSTM escalation-head 예측 (0~1) |
| **onset_prob** | FLOAT64 | **onset 점수 (메타 출력, 0~1).** 순위/티어의 기준값 |
| calm_flag | INT64 | 1=평온국(past14d_event_count==0)=onset 경보 대상 / 0=분쟁중(현황 모니터링) |

> ⚠️ `onset_prob` 는 **캘리브레이션된 절대확률이 아니라 순위 점수**(class-weight 로지스틱). 대시보드는 **순위/퍼센타일 기반**으로 쓸 것.

## 3. 스코어링 담당 작업: model_scores → risk_score → Supabase
- model_scores 읽어 **국가별 risk_score(0~100)** 계산. 후보 식 4종 계산 후 채택(2024 onset top-K 기준).
  - 제안식 ④(순위블렌딩): `score=100*(0.2*B + 0.4*C_pct + 0.4*F_pct)`, F_pct=onset_prob의 calm 국가내 퍼센타일. 4단계 티어(상위 2/5/15%/나머지).
- **onset 패널**: calm_flag==1 국가만 onset 경보. calm_flag==0 은 "현재 분쟁강도" 모니터링 패널로 분리(대시보드 2분할).
- 절대기준 병기(조용한 날 과민반응 방지): onset_prob 원값도 같이 표시.
- 결과 → Supabase (대시보드 read 레이어).

## 4. 모델 사실 (스코어링 담당 참고)
- **★ 배포 onset 모델 = 단일 2층 LSTM (window45·전피처, 3시드 평균).** 트리(LGBM/XGB)+시퀀스(GRU/LSTM/TCN/양방향)+ 7개 base를 스태킹으로 *탐색*했으나, onset(calm) honest 선택에서 **`LSTM_W45_ALL` 단독이 최적**으로 선택됨. 2번째 base를 무엇을 넣어도 held-out 점수가 떨어짐(LSTM 단독 0.1732 vs +어떤 base든 0.13~0.173). 메타는 1입력 로지스틱(=단조변환)이라 **서빙 점수 = LSTM escalation-head 예측**과 사실상 동일. 나머지 6 base 가중치는 보관(향후 직교 신호 추가 시 메타 재구성용).
  - *왜 단일?* onset은 평가 표본이 작고(calm 양성 ~100/폴드) 시퀀스 base들이 서로 중복이라, 추가 base가 새 정보 없이 과적합만 유발 → honest 선택이 1개에서 멈춤. (escalation 전체 타겟이면 4 base 스태킹이 이득이었으나, **onset = calm 부분집합으로 평가창이 좁아지며 단일로 수렴**.)
- onset honest CV PR-AUC = **0.1732** (검증 재현됨). 표본 작아 폴드별 0.14~0.19 변동 → "범위/추가검증 필요"로 표기.
- 운영지표(2024): 상위 2% 정밀도 ~27%(무작위 5배). **확정예측 아닌 우선순위 워치리스트**로 포지셔닝.

## 5. 실행 방법 & 전달물 (How to run / handoff)
**코드 위치**: `src/serve/`(피처빌더·스코어러), `Dockerfile.serve`+`entrypoint_serve.sh`(컨테이너). 의존: `scripts/gkg_embed/`·`scripts/gdelt_*`·`scripts/seq_gru_aclfree.py`·`src/process/{feature_builder,gkg_feature_builder}.py`.

**필요 env**(.env): `GCP_PROJECT`, `BQ_DATASET`, `GOOGLE_APPLICATION_CREDENTIALS`(SA키 경로), `OPENAI_API_KEY`(임베딩 적재용). ⚠️ 코드에 키 하드코딩 금지 — 전부 env에서.

**⚠️ 모델 가중치 별도 전달**: `output/models/onset_prod/`(LSTM `.pt` + `meta.pkl` + `manifest.json`)는 **gitignore라 레포에 없음**. 드라이브로 받아 그 경로에 배치해야 스코어러가 동작. (임베딩 fit 아티팩트 `output/models/gkg_pca/`·`input/processed/features/anchor_embeddings.npz`도 동일)

**실행 (2단계, 수집 배치 이후 주기 실행)**:
1. 피처: `python -m src.serve.build_model_input --start <YYYY-MM-DD> --end <YYYY-MM-DD> --run-ts <ISO>` → BQ `model_input` (rolling 위해 start는 60일 전쯤)
2. 스코어: `python -m src.serve.run_scoring --run-ts <ISO> [--target-date <YYYY-MM-DD>]` → BQ `model_scores`
   - 도커: `docker build -f Dockerfile.serve -t conflict-ew-serve .` 후 `docker run ... conflict-ew-serve {features|score} [args]`
   - ⚠️ 비용: `gdelt_titles` 월파티션 스캔(~$0.12/run) + OpenAI 임베딩. 윈도우 좁게.

**검증 상태**: 로컬 419 parity·onset CV 0.1732 재현·라이브 BQ base 동작 확인. 전체 e2e BQ run은 첫 운영 실행 시 검증.
