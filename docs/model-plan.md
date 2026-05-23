# 모델 플랜 — 무력충돌 예측 조기경보

> 작성: 2026-05-14
> 짝 문서: **model-study.md** (모델별 특징·이론) / 본 문서는 **구현 순서·산출물**
> 데이터: 2014-01-01 ~ 2026-03-28, 58개국, 259,260행, 양성률 4.29%
> Split: train 211,816 (10년) / val 10,556 / test 15,718

---

## 0. 본 단계의 목표

EDA에서 확정된 환경(불균형 4.29%, 단변량 lift 1.27 약함, 상관 >0.95 피처 23쌍, ACLED 2014-2017 결측)을 전제로:

1. **합격선**: `persistence_gain > 0` — Persistence baseline PR-AUC를 모든 모델이 이겨야 함
2. **운영 지표**: top-5% Precision/Recall (매일 위험 상위 5% 국가 알람)
3. **논문 비교**: Macis(2024) 재현 AUC + 우리 차별점(ACLED + 경제 피처 추가) 정량화
4. **최종 산출**: Stacking 앙상블 → 위험도 0-100 점수 → 대시보드 이관

---

## 1. 4축 구성

| 축 | 역할 | 담당 카테고리 |
|---|---|---|
| 지도학습 본선 | 라벨 직접 학습, 평가축의 주력 | 선형(A) + 트리 부스팅(B) |
| 비지도 피처 보강 | 시퀀스 이상치 신호를 피처화 | 딥러닝(C) → 트리(B) 투입 |
| 딥러닝 분류 | 시퀀스 정보 직접 활용 | 딥러닝(C) |
| 앙상블 + 캘리브레이션 | 단일 모델 약점 보완 + 확률 보정 | 앙상블(D) |

→ model-study.md의 4 카테고리(A 선형 / B 트리 부스팅 / C 딥러닝 시퀀스 / D 앙상블)와 1:1 매핑.

---

## 2. 모델 구현 순서 (7종)

### 2-1. Logistic Regression — sanity baseline [담당 A]
- **목적**: 모든 복잡 모델이 이겨야 할 최소 기준선
- **구현 포인트**:
  - `sklearn.linear_model.LogisticRegression(class_weight='balanced')`
  - StandardScaler 필수
  - L1/L2/ElasticNet 3종 변형 비교
  - log1p 전처리 (skew>5 피처 11개)
- **산출**: `output/models/logreg__y_escalation/`, regularization 비교 표

### 2-2. LightGBM — 본선 메인 [담당 B]
- **목적**: 평가축의 주력. 본 환경에 최적합 (EDA ⑤·⑥ 환경 대응)
- **구현 포인트**:
  - `scale_pos_weight ≈ 22` 또는 `is_unbalance=True`
  - Optuna 50~100 trial 튜닝 (val 기준)
  - Early stopping + SHAP 지원
  - feature_importance = gain 기준
- **산출**: `output/models/lightgbm__y_escalation/`, SHAP 피처 중요도

### 2-3. LightGBM + SE 피처 — 논문 확장 [담당 B + C 협력]
- **목적**: Macis SE(시퀀스 이상치)가 트리 모델 보강에 기여하는지 검증
- **구현 포인트**:
  - C 담당이 Macis Autoencoder로 국가x일별 SE 점수 추출 → parquet 저장
  - LightGBM 입력에 SE 컬럼 1개 추가 → 55피처로 학습
  - Ablation: SE 포함 vs 미포함 PR-AUC 비교
- **산출**: `output/models/lightgbm_se__y_escalation/`, SE 기여도 표

### 2-4. XGBoost — 비교 모델 [담당 B]
- **목적**: LightGBM과 head-to-head. 둘 중 우세한 쪽을 앙상블 구성요소로
- **구현 포인트**: LightGBM과 동일 튜닝 파이프라인, MODEL_REGISTRY 등록만
- **산출**: `output/models/xgboost__y_escalation/`

### 2-5. Macis LSTM Autoencoder — 논문 재현 [담당 C]
- **목적**: 비지도 이상치 탐지 재현 + 피처 추출기 역할
- **구현 포인트**:
  - Encoder(LSTM) → Latent → Decoder(LSTM), 평시 데이터로만 학습
  - 재구성 오차(SE) = 이상치 점수
  - 원 논문은 GDELT only — 우리는 ACLED+GDELT+경제 합본 → 다른 결과 가능
  - 학습 장시간 → Colab T4 GPU 이관
- **산출**: SE 점수 parquet(58국x일별) → 2-3에 전달, 재현 AUC 보고

### 2-6. LSTM Classifier [담당 C]
- **목적**: 시퀀스 정보를 직접 분류에 활용 (이상치 탐지와 다른 경로)
- **구현 포인트**:
  - 입력: (batch, seq_len=30, n_features=54) → LSTM → Dense → Sigmoid
  - BCE + `pos_weight` 또는 Focal Loss(γ=2)
  - Dropout + Early stopping
  - Focal Loss 사용 시 Platt Scaling 캘리브레이션 필수
  - Colab T4 GPU 이관
- **산출**: `output/models/lstm_classifier__y_escalation/`

### 2-7. Stacking Ensemble [담당 D]
- **목적**: 단일 모델 약점 보완, 최종 성능 극대화
- **구성**:
  - Level 0 (base): LightGBM(+SE), XGBoost, LSTM Classifier
  - Level 1 (meta): Logistic Regression (얕은 깊이로 overfit 방지)
  - **Out-of-fold (OOF) 예측 기반** — train에서 base 예측 만들 때 누수 방지
- **추가**: Platt Scaling / Isotonic Regression 캘리브레이션 (Brier/ECE 측정)
- **산출**: `output/models/stacking__y_escalation/`, calibration plot

---

## 3. Ablation 실험 매트릭스

> 각 담당이 본인 모델로 최소 A-1, A-2 수행. A-3~A-5는 우선순위 낮음 — 1차 평가 안정화 후 진행.

| ID | 변수 | 옵션 | 우선순위 | 가설 |
|---|---|---|---|---|
| A-1 | 예측 lookahead | 1d / 2d / **3d** / 7d | 최우선 | 3d 정합성 검증 + 2d 알람 가치 비교 |
| A-2 | Macis SE 피처 | 포함 / 제외 | 높음 | 시퀀스 이상치 신호의 트리 모델 기여도 |
| A-3 | 피처 그룹 | ACLED만 / GDELT만 / 경제만 / 전체 | 중 | 소스별 예측력 분리 (논문 대비 차별점) |
| A-4 | ACLED 결측 마스크 | 추가 / 미추가 | 중 | EDA ① 권장안 — "진짜 0 vs 데이터 없음" 구분 |
| A-5 | log1p 변환 | 적용 / 미적용 | 낮음 | 선형/시퀀스 모델 한정 (트리는 효과 없음) |

각 셀에 PR-AUC / persistence_gain / top-5% Precision 3개 지표 기록.

---

## 4. 백테스트 3케이스

시간순으로 재생된 예측으로 실전 시나리오 검증 (학습 기간에서 제외하고 사후 평가):

| 케이스 | 날짜 | Split |
|---|---|---|
| Ukraine 침공 | 2022-02-24 | train 기간 → holdout 평가 |
| Sudan 내전 | 2023-04-15 | train 기간 → holdout 평가 |
| Gaza 전쟁 | 2023-10-07 | train 기간 → holdout 평가 |

**기록 지표**:
- Lead Time: 위험도가 임계점(0.5 or top-5%)을 처음 넘은 시점 (D-day 대비 며칠 전)
- 최대 위험도: D-7~D+7 구간 peak 점수
- False alarm 일수: 같은 국가 평시 기간 대비 오탐 빈도

**팔레스타인 주의** (EDA ③): test 기간 양성률 감소 → escalation 정의가 만성 분쟁국에 안 맞을 가능성. 모델 한계로 명시.

---

## 5. 위험도 점수 0-100 통합 (대시보드 이관 준비)

WorldMonitor CII 구조 차용 — baseline anchor + 컴포넌트 분해로 0-100 range 활용 보장.

```
risk_score = 0.2·B + 0.4·C_state + 0.4·F + hotspot_bonus
```

- **B**: Baseline prior (국가별 ACLED 5y 평균 fatalities log → percentile rank)
- **C_state**: U/C/S/I 서브스코어 합성 (현재 상태)
- **F**: ML Forecast (Stacking 모델의 calibrated 확률)
- **hotspot_bonus**: 인접국 동시 격화 시 가산

**완료 기준**:
- `score_p95 ≥ 50`, `score_max ≥ 80`, `usable_range ≥ 40`, `positive_lift ≥ 15`
- Ukraine/Sudan/Gaza 각각 D-1~D+1 시점에 risk_score ≥ +30 급상승

---

## 6. 최종 산출물 형식

```
output/
├── models/{모델명}__y_escalation/
│   ├── model.pt              # 가중치
│   ├── config.json           # 하이퍼파라미터
│   └── metrics.json          # PR-AUC / persistence_gain / top-k precision
├── submissions/
│   └── submission_{모델명}.csv   # test set 예측
├── features_lists/
│   └── features_{모델명}.txt     # 사용 피처
└── evaluation/
    ├── comparison.md             # 전 모델 비교 + ablation 종합
    └── plots/                    # SHAP / PR curve / ROC / Lead Time
```

---

## 7. 진행 순서

각 단계는 이전 단계 산출물에 의존하므로 **순차 진행**. 같은 단계 내 항목들은 담당자별 병렬 가능.

| 단계 | 작업 | 선행 의존 |
|---|---|---|
| 1 | 회의 결정 5건 확정 (§9) | — |
| 2 | LogReg + Persistence baseline 구축 → `persistence_gain` 기준선 확보 | 1 |
| 3 | LightGBM 튜닝 → 본선 메인 | 1 |
| 4 | Macis AE 재학습 (Colab) + SE 피처 parquet 추출 | 1 |
| 5 | LightGBM + SE 학습 | 3, 4 |
| 6 | 위험도 점수 0-100 통합 + 백테스트 3케이스 1차 | 5 |
| 7 | XGBoost — LightGBM head-to-head | 3 |
| 8 | LSTM Classifier | 1 |
| 9 | Stacking Ensemble + 캘리브레이션 (Platt/Isotonic) | 5, 7, 8 |
| 10 | Ablation 매트릭스 완성 (A-1, A-2 필수) | 5, 7, 8 |
| 11 | 종합 리포트 + 대시보드 이관 | 9, 10 |

우선순위 조정, 모델 추가/제외 의견은 회의에서.

---

## 8. 분담 매핑

| 담당 | model-study 카테고리 | 본 문서 구현 항목 | 주요 산출물 |
|---|---|---|---|
| A | 선형 + 베이스라인 | 2-1 LogReg + Persistence | regularization 비교 표 |
| B | 트리 부스팅 | 2-2 LightGBM, 2-3 LightGBM+SE, 2-4 XGBoost | PR-AUC + persistence_gain 표 |
| C | 딥러닝 시퀀스 | 2-5 Macis AE, 2-6 LSTM Classifier | 재현 AUC + SE 피처 parquet (B에 전달) |
| D | 앙상블 + 캘리브레이션 | 2-7 Stacking + Calibration | Brier/ECE + top-k precision |

공통 의무: A-1(lookahead), A-2(SE) ablation은 본인 모델로 필수 수행.

---

## 9. 회의 결정 대기 항목

model-study.md §7 참조. 본 플랜은 아래 5건 결정 후 확정:

1. lookahead 3d 유지 vs 2d로 좁힘
2. ACLED 결측 마스크 컬럼 추가 여부
3. Train 시작일 (2014 / 2016 / 2018)
4. CV 전략 (hold-out / walk-forward 5-fold)
5. Ablation 우선순위 (5-1~5-5 중 무엇부터)
