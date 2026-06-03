# 에스컬레이션 예측 실험 요약 (팀 공유)

> 작성 기준: `New_analyze` 워크스페이스, `Merged_data/processed` 기본 데이터, 타깃 `y_escalation`  
> 참고 문서: `meeting_2026-05-07.md` (평가 원칙·베이스라인 논의)

---

## 요약

- **데이터**: ACLED·GDELT·경제지표 등 피처, 58개국 일별 패널, 학습/검증/테스트 split 고정.
- **SE 피처**: 현재 파이프라인에 반영되지 않아 **미포함** 상태로 실험.
- **평가 버그 수정**: 검증 CSV의 행 순서와 지표 계산 시 정렬 순서가 어긋나 **PR-AUC 등이 과소평가**되던 문제를 `evaluate_predictions`에서 수정함. **이전에 공유했던 ~0.04대 PR-AUC 수치는 재현 시 수정 필요.**
- **선정 모델**: 검증 **PR-AUC 최대** 기준 **XGBoost** (`train_models.py`, `base`, `date_mode=ordinal`).
- **한 줄 결론**: 무작위 대비로는 **순위( PR-AUC )가 개선**되었으나, 미팅에서 중요하게 본 **persistence 대비 이득(`persistence_gain`)은 여전히 음수**로, “어제 라벨 베이스라인”을 넘지는 못한 상태.

---

## 1. 수행한 실험

### 1.1 EDA (`scripts/eda.py`)

| 목적 | 내용 | 결과 요약 |
|------|------|-------------|
| 데이터 스케일·분포 파악 | train/val/test 및 full 조인 뷰 요약 | train 약 21만 행, 검증·테스트 양성 비율 약 **4%**대, SE용 `full_se.parquet` 없음 → SE 구간 스킵 |

### 1.2 다중 모델 학습·비교 (`scripts/train_models.py`)

| 목적 | 모델 | 조정·설정 |
|------|------|-----------|
| 동일 split·동일 피처에서 공정 비교 | LogisticRegression, LightGBM, XGBoost, RandomForest, ExtraTrees, SGD(log loss) | `random_state=42`, `class_weight` 등 스크립트 기본값, `date_mode=ordinal` |
| 베스트 선정 | 검증 **PR-AUC** 최대 모델을 `best_val_model__base.joblib`로 저장 | 아래 §2 참고 |

**정렬 수정 이후** 검증 지표(동일 조건) 대략 순위:

- XGBoost **PR-AUC ≈ 0.102** (최고)
- RandomForest ≈ 0.098  
- LightGBM ≈ 0.095  
- ExtraTrees ≈ 0.090  
- SGD ≈ 0.074  
- LogReg ≈ 0.071  

공통적으로 **`persistence_gain`은 음수** (베이스라인 대비 PR-AUC 열세).

### 1.3 변수 선택 + 보정 XGB (`scripts/escalation_model_analysis.py`)

| 목적 | 내용 |
|------|------|
| 피처 수 과다·해석 부담 완화 | 학습 구간 **상호정보량(MI)**으로 수치 피처 일부 선택, `country`·`date` 고정 |
| 확률 해석 | **Sigmoid 보정**이 붙은 XGBoost (`CalibratedClassifierCV`) |
| 검증 K 선택 | MI 상위 개수 K를 소수 후보 중 검증 PR-AUC로 선택 (탐색적) |

**요지**: 축소+보정 모델이 **검증 PR-AUC·ECE** 일부에서 전체 피처 XGB 대비 소폭 유리했던 구간이 있었으나, **`train_models`의 공식 베스트 선정 루틴과는 별도 실험**으로 두는 것이 좋음.

### 1.4 타깃·베이스라인 변형 (`scripts/run_target_experiments.py`)

| 목적 | 조정 |
|------|------|
| persistence 난제 구조 이해 | (1) `y_escalation` + **`lag1_y_escalation` 피처** (2) **`y_escalation_no_persist`** 타깃 (3) **`y_escalation_fwd7` / `fwd14`** (향후 H일 내 일별 플래그 OR) |
| 다중 일 persistence | `fwd7`/`fwd14` 실험 시 **일별 `y_escalation`의 최근 H일 max** 대비 `persistence_multilag_gain` 등 추가 출력 |

**요지**: 본 타깃(3d `y_escalation`)에서의 **미팅식 persistence 이기기**와는 정의가 다를 수 있음. `no_persist` 등은 **보조 진단**에 가깝다.

### 1.5 하이브리드(상위 소수국 전용 로컬 모델) (`scripts/train_hybrid_country_models.py`)

| 목적 | 조정 |
|------|------|
| 국가 이질성 | 학습 양성 상위 K개국만 **국가별 LightGBM** (`country` 제외), 나머지는 **글로벌** LightGBM |
| 임계값 | 예: 상위 5국, train 행 ≥800, 양성 ≥40 |

**결과(해당 설정)**: 집계 검증 **PR-AUC는 글로벌 단독보다 하이브리드가 낮음** → 이 설정에서는 채택 이점 없음.

### 1.6 평가 로직 수정 (`src/metrics.py`)

| 문제 | 조치 |
|------|------|
| `val` 등이 `date, country` 정렬과 다를 때 `y_true`와 `y_prob` 순서 불일치 | `evaluate_predictions` 내부에서 확률을 **정렬된 eval 순서에 맞게 재배열** |

**영향**: `train_models`, `run_target_experiments`, `escalation_model_analysis` 등 **동일 함수를 쓰는 모든 검증 PR-AUC**가 이전 대비 달라질 수 있음. **과거 보고서 수치는 재실행으로 갱신 권장.**

---

## 2. 결과적으로 선정한 모델 및 평가 지표

### 2.1 선정 규칙

- **스크립트**: `scripts/train_models.py`  
- **데이터셋**: `base` (`train.parquet` / `val.parquet` / `test.parquet`)  
- **날짜 인코딩**: `date_mode=ordinal`  
- **선정 기준**: 검증 **`pr_auc` (Average Precision / PR-AUC) 최대**

### 2.2 선정 모델: **XGBoost**

검증(`val`)에서의 주요 지표 (정렬 수정 반영 후, `train_models.py` 최신 실행 기준):

| 지표 | 값 | 비고 |
|------|-----|------|
| **PR-AUC** | **0.1020** | 양성 비율(~0.041) 대비 무작위 바닥 위 |
| persistence PR-AUC | 0.2339 | 전날 동일 타깃 기반 베이스라인 |
| **persistence_gain** | **−0.1320** | 모델 − persistence (음수 → 베이스라인 미추월) |
| P@top 1% | 0.1792 | 상위 1% 구간 정밀도 |
| P@top 5% | 0.1402 | |
| P@top 10% | 0.1231 | |
| R @ P≥0.10 | 0.4535 | 정밀도 10% 이상 구간에서 재현율 |
| R @ P≥0.20 | 0.0442 | |
| R @ P≥0.30 | 0.0349 | |
| ECE (10 bins) | 0.0067 | 구간 보정 오차 (낮을수록 유리) |
| positive_rate (val) | 0.0407 | 참고용 |

**산출물**

- 비교표: `artifacts/model_comparison__base.csv`  
- 베스트 가중치: `artifacts/best_val_model__base.joblib`  
- 설정 JSON: `artifacts/best_model_config__base.json`

---

## 3. 모델·실험의 한계 및 개선 제언

### 3.1 한계

1. **persistence 대비 열세**  
   미팅에서 제안된 **`persistence_gain > 0`** 같은 “베이스라인 대비 가치” 기준에는 **현 구성으로 미달**. 분쟁 패널에서 **전날 상태가 강한 신호**인 점이 크다.

2. **SE·외생 신호 부재**  
   문서상 강한 베이스라인(LightGBM+SE)과 **동일 정보로 비교 불가**. 피처 측면에서 불리할 수 있음.

3. **타깃·베이스라인 정의 일관성**  
   전방 OR(`fwd7` 등)나 `no_persist`는 **질문이 바뀌는 실험**이므로, 본 타깃(3d `y_escalation`)의 합격 여부와 **직접 연결해 말하기 어렵다**.

4. **검증 기반 선택의 누수 위험**  
   MI 개수 K, 축소 피처 등은 검증을 보며 고른 부분이 있어, **테스트는 최종 1회** 원칙을 지킬 것.

5. **하이브리드(소수국)**  
   시도한 설정에서는 **집계 PR-AUC 악화** — 국가별 데이터 양·양성 희소, 글로벌 모델의 `country` 표현과 중복 등으로 이득이 제한적일 수 있음.

### 3.2 개선 제언 (우선순위 제안)

| 우선순위 | 방향 |
|----------|------|
| 높음 | **SE(또는 동급 구조 피처) 재생성·병합** 후 `base`와 동일 split으로 재학습·재비교 |
| 높음 | **persistence를 입력으로 포함**하거나 **잔차/온셋 타깃**을 팀 합의 하에 정의해, “베이스라인 대비 이득”과 **예측 질문을 정렬** |
| 중간 | **타깃 창(3/7/14일)** 변경 시 **동일 스케일의 persistence** 정의를 같이 두고, 보고서에 **문제 정의 표**를 고정 |
| 중간 | 미팅 지표( **Macro PR-AUC**, **P/R@top-5%**, **Lead time** 등)를 `evaluate_predictions` 확장 또는 별도 스크립트로 **자동 집계** |
| 낮음 | 딥러닝(LSTM/Transformer)은 **현 신호 대비 비용·해석** 측면에서 후순위; **피처·타깃 정합**이 먼저 |

---