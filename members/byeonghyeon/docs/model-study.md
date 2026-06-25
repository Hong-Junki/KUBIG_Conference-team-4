# 모델 스터디 — 무력충돌 예측 조기경보

> 작성: 2026-05-14
> 목적: 회의 전 모델별 특징 정리, 4명 파트 분담, 모델 확정 전 실험할 변수 목록화
> 데이터 환경 요약(중요): 양성률 4.29%, 211k행, 54피처, 단변량 lift 1.0~1.27(약함), 상관 >0.95 피처쌍 23개, skew>5 피처 11개, ACLED 2014-2017 결측

---

## 0. 큰 그림 — 왜 이 모델들인가

EDA에서 나온 환경적 제약:
- **불균형 4.29%** → 단순 정확도 무의미, PR-AUC + Precision@top-k 가 본 평가축
- **단일 신호 약함(lift 1.27이 1위)** → 피처 interaction 자동 학습이 가능한 트리 부스팅이 이론적으로 강함
- **피처 중복 + skew** → LogReg/LSTM은 전처리 부담, 트리는 그대로 OK
- **시계열 + 양성 희소** → 시퀀스 모델(LSTM)은 단독 분류보다 **이상치 탐지(Macis AE)** 가 더 자연스러움

→ 4개 카테고리 (선형 / 트리 부스팅 / 딥러닝 시퀀스 / 앙상블) 로 묶고, 각자 1카테고리씩 깊게 본다.

---

## 1. 카테고리 A — 선형 모델 & 베이스라인 [담당자 A]

### 1-1. Persistence Baseline (참고용, 학습 X)
- "오늘 분쟁 있으면 내일도 있다" — 시계열 trivial baseline
- 우리가 만든 모델은 반드시 이걸 이겨야 함 (`persistence_gain > 0` 합격선)
- 구현: 직전 N일 평균을 그대로 prob으로 사용

### 1-2. Logistic Regression
- **장점**: 학습 빠름, 계수 해석 가능, SHAP 없이도 영향력 파악
- **단점**: 선형 결정경계 → interaction 못 잡음 (EDA ⑤의 한계 직격)
- **본 환경 적합도**: 낮음 (단변량 lift 약함 = 선형으로는 한계)
- **필수 전처리**:
  - log1p 변환 (skew>5 피처 11개, EDA ④)
  - 윈도우 슬림화 또는 PCA (상관 >0.95 쌍 23개, EDA ⑥)
  - StandardScaler

### 1-3. Regularization 변형 (실험 항목)
- **L1 (Lasso)**: 피처 자동 선택, 54개 중 살아남는 거 보면 인사이트
- **L2 (Ridge)**: 다중공선성 완화 (EDA ⑥ 대응)
- **ElasticNet**: 둘 합친 거, α 튜닝 필요

### 1-4. 안 쓸 모델 — 이유
| 모델 | 안 쓰는 이유 |
|---|---|
| Naive Bayes | 피처 독립 가정 → 상관 >0.95 쌍 23개 환경에서 가정 위배 |
| SVM (RBF) | 211k행에 O(n²) 학습 → 너무 느림, 확률 추정도 별도 calibration 필요 |
| KNN | 고차원(54) + 불균형 환경에서 거리 측정 무의미 |
| Linear Discriminant Analysis | 정규분포 가정, skew>5 피처와 불일치 |

---

## 2. 카테고리 B — 트리 부스팅 [담당자 B]

> **본 프로젝트의 메인 베팅.** EDA ⑤에서 "트리 모델이 본 환경에 강함"을 명시.

### 2-1. LightGBM
- **메커니즘**: leaf-wise 트리 성장 + histogram 기반 split → 빠르고 메모리 효율
- **장점**:
  - skew/스케일 무관 (전처리 최소)
  - 자동 interaction 학습 (EDA ⑤ 약점 보완)
  - `is_unbalance=True` 또는 `scale_pos_weight=22` 로 불균형 처리
  - 카테고리 피처 native 지원
- **단점**: leaf-wise → overfitting 위험 (early stopping 필수), 작은 데이터셋엔 부적합 (우리는 OK)
- **본 환경 적합도**: 최상

### 2-2. XGBoost
- **메커니즘**: level-wise + 정규화 항 포함된 objective
- **vs LightGBM**:
  - level-wise → 더 보수적, overfitting에 강함
  - 학습 속도 LightGBM 대비 느림
  - 결과는 보통 LightGBM ± 0.5% 수준 (앙상블 다양성용으로 가치 있음)
- **본 환경 적합도**: 상

### 2-3. LightGBM + SE 피처
- Macis LSTM Autoencoder 의 재구성 오차(SE)를 추가 피처로 투입
- "이 국가의 현재 상태가 평소 패턴과 얼마나 다른가" 신호
- 트리 모델의 interaction 학습 + 시퀀스 이상치 신호 결합 — SE 포함/제외 ablation으로 기여도 확인

### 2-4. 안 쓸 모델 — 이유
| 모델 | 안 쓰는 이유 |
|---|---|
| Random Forest | 부스팅이 tabular 벤치마크에서 일관되게 우위, RF는 분산축소가 강점인데 우리 문제는 bias가 더 큼 |
| CatBoost | 카테고리 피처 강점인데 우리는 country one-hot 정도 외엔 numeric → 이점 약함 |
| ExtraTrees | RF 변형, 마찬가지 이유 |
| GBM (sklearn) | LightGBM/XGBoost가 모든 면에서 상위호환 |
| AdaBoost | 이상치/노이즈에 약함, 분쟁 데이터엔 부적합 |

---

## 3. 카테고리 C — 딥러닝 시퀀스 [담당자 C]

### 3-1. Macis et al.(2024) LSTM Autoencoder — 논문 재현
- **논문 핵심**: "지도학습으로는 양성 너무 적어 안 잡힌다 → **이상치 탐지로 접근**"
- **구조**: Encoder(LSTM) → Latent → Decoder(LSTM), 평시 데이터로만 학습
- **추론**: 입력 시퀀스의 재구성 오차(SE = Squared Error) = 이상치 점수
- **논문 결과**: 우크라이나 AUC 0.9377, 부르키나파소 0.8723, 스리랑카 0.8656
- **우리 활용**: 분류 모델로 직접 쓰지 않고 **SE를 피처로 추출**해 LightGBM+SE에 투입
- **재현 시 주의**: 논문은 GDELT만 사용, 우리는 ACLED+GDELT+경제 합본 → 다른 결과 가능

### 3-2. LSTM Classifier
- **메커니즘**: 일별 시퀀스(L=30일) → LSTM → sigmoid
- **장점**: 시간 의존성 명시적 모델링
- **단점**:
  - 양성 4% × 시퀀스 → 유효 양성 시퀀스 더 적어짐
  - skew/상관 전처리 부담 (LogReg와 동일)
  - 트리 대비 학습 오래 걸림 (Colab T4 이관)
- **본 환경 적합도**: 중 (재현성보다 다양성 가치)

### 3-3. 안 쓸 모델 — 이유
| 모델 | 안 쓰는 이유 |
|---|---|
| Transformer (vanilla) | data-hungry, 양성 4% × 211k에서 학습 어려움. 시계열 길이도 30일로 짧음 |
| Temporal Fusion Transformer (TFT) | 강력하지만 forecasting 전용 설계, 분류로 쓰려면 overkill |
| TCN (Temporal Conv) | LSTM과 유사 성능, 추가 가치 적음 |
| GRU | LSTM과 동급 성능 — ablation 한 번만 비교하고 메인은 LSTM |
| N-BEATS / DeepAR | 시계열 예측 모델, 분류 문제에 부적합 |

---

## 4. 카테고리 D — 앙상블 & 메타학습 [담당자 D]

### 4-1. Stacking
- **구조**:
  - Level 0 (base): LightGBM, XGBoost, LSTM Classifier
  - Level 1 (meta): LogReg 또는 LightGBM (작은 깊이)
- **주의**: base 모델 예측을 train에서 만들 때 **out-of-fold(OOF)** 로 만들어야 누수 방지
- **본 환경 적합도**: 상 (단일 모델 약점 보완)

### 4-2. Calibration (대시보드 위험도 0-100 산출에 필수)
- **Platt Scaling**: LogReg를 sigmoid output에 fit, 단순
- **Isotonic Regression**: 비단조 보정 가능, val 데이터 충분할 때 선호
- 본 평가에서 Brier Score / ECE 로 측정

### 4-3. 안 쓸 방법 — 이유
| 방법 | 안 쓰는 이유 |
|---|---|
| Hard Voting | 확률 정보 손실 → top-k precision 평가에 부적합 |
| Soft Voting | Stacking이 일반적으로 우위 (학습된 meta가 더 잘 가중) |
| Blending (holdout) | Stacking(CV) 대비 데이터 효율 낮음 |
| Bayesian Model Averaging | 구현 비용 대비 효과 작음, 회의 일정과 안 맞음 |
| Bagging (단독) | RF/부스팅이 이미 bagging 변형 — 추가 이득 없음 |

---

## 5. 모델 확정 전에 돌려볼 변수 (Ablation Grid)

> 회의에서 우선순위 결정. 시간 한정상 전부는 못 돌림.

### 5-1. 데이터/라벨 변수
| 변수 | 옵션 | 설명 / 가설 |
|---|---|---|
| 예측 lookahead (= 며칠 안에 발생할지 예측하는 시간 윈도우) | 1d / 2d / **3d**(현재) / 7d | 아래 "lookahead 길이 선택 근거" 표 참조 |
| ACLED 결측 마스크 컬럼 (= "이 행은 ACLED 데이터 없음"을 알려주는 0/1 컬럼) | 추가 / 미추가 | EDA ① — 추가 시 모델이 "진짜 0(평화)"인지 "데이터 없어서 0"인지 구분 가능 |
| Train 시작일 | 2014-01 / 2016-01 / 2018-01 | EDA ② — 2014-01 양성률 spike(24%) 같은 초기 노이즈를 학습에서 제외 |
| 윈도우 슬림화 (= 7일/14일/30일 rolling 평균 피처 중 일부만 남기기) | 7d/14d/30d 모두 / 14d만 | EDA ⑥ — 상관 >0.95 피처를 여러 개 쓰면 선형/시퀀스 모델 학습 불안정 |

#### lookahead 길이 선택 근거 — 왜 3일이 디폴트인가

| 옵션 | 양성률(예상) | 장점 | 단점 |
|---|---|---|---|
| **1d** | ~1.5% | 가장 즉각적, 알람 가치 최고 | 양성 너무 희소(현재 4.29%의 1/3 수준) → 학습 신호 부족, GDELT 뉴스→ACLED 이벤트 lag(보통 1~2일) 못 잡음 |
| **2d** | ~3% | 1d보다 양성 보존, 운영가치 유지 | GDELT 뉴스→ACLED 이벤트 인과 윈도우(평균 2~3일)와 빠듯하게 겹침 |
| **3d (현재)** | **4.29%** | (a) 양성 보존 — 학습 가능 수준 (b) GDELT 신호가 ACLED 이벤트로 이어지는 인과 윈도우와 자연스럽게 매칭 (c) 일반적 조기경보 권장 윈도우(24-72h)에 부합 (d) ACLED 주1회 갱신·7일 lag 운영 조건과 정합 | 1d/2d 대비 즉각성 떨어짐 |
| 7d | ~9% | 양성 풍부, 학습 쉬움 | escalation 라벨 인과성 약화(7일 안에 별개 사건 섞임), 알람 가치 하락 |

**결론**: 3일이 "양성 보존 × 인과 명확성 × 운영가치"의 절충점. 단 **2d ablation은 반드시 돌려본다** — 2일에서도 학습 가능하면 알람 가치가 더 큼. 1d는 양성 1.5%로 사실상 학습 불가, 7d는 라벨 의미 약함이라 우선순위 낮음.

### 5-2. 피처 변수
| 변수 | 옵션 | 설명 / 가설 |
|---|---|---|
| SE 피처 (= Squared Error, Macis Autoencoder의 재구성 오차값 = "이 시점이 평소와 얼마나 다른가" 신호) | 포함 / 제외 | 시퀀스 이상치 신호가 트리 모델 PR-AUC에 기여하는지 측정 |
| 피처 그룹 ablation (= 데이터 소스별로 끊어 학습해 보기) | ACLED만 / GDELT만 / 경제만 / 전체 | 어느 소스가 예측력의 핵심 기여인지 분리 |
| log1p 변환 (= log(1+x) — 한쪽으로 치우친 분포를 좌우대칭에 가깝게 만드는 변환) | 적용 / 미적용 | EDA ④ — skew>5 피처에 적용 시 LogReg/LSTM 학습 안정화 (트리 모델엔 효과 없음) |
| 비율 피처만 (ratio_*) (= "전체 이벤트 중 비중", 절대 횟수 아님) | only / + 절대량 | EDA ⑤ — 비율 피처가 단변량 lift 상위 점령 |

### 5-3. 모델 하이퍼파라미터
| 변수 | 옵션 | 설명 / 가설 |
|---|---|---|
| scale_pos_weight (= 학습 시 양성 샘플 손실에 곱하는 가중치, 불균형 보정용) | 22(현재, ≈음성/양성 비) / √22≈4.7 / 10 | 가중치 22는 비율 그대로 — 너무 강하면 false positive 폭증, 너무 약하면 양성 놓침 |
| 손실함수 | BCE (= Binary Cross Entropy, 표준 이진분류 손실) / Focal Loss γ=2 (= 쉬운 샘플 손실↓ + 어려운 샘플↑, 불균형 특화) | Focal이 불균형에 강하다는 통설 검증 |
| Threshold tuning (= 모델이 뱉은 확률값을 "알람 발생"으로 변환하는 기준점) | top-5% / top-10% / Youden's J (= TPR−FPR 최대화 지점) | 운영 알람 정책과 직결 — top-5%는 매일 상위 5% 국가만 알람 |
| LightGBM num_leaves (= 트리 한 그루의 잎 수 = 모델 복잡도) | 31 / 63 / 127 | 잎 많을수록 표현력↑ 하지만 overfit 위험↑ |
| LSTM 시퀀스 길이 L (= 한 번에 입력하는 과거 며칠치 데이터) | 14 / 30 / 60 | 짧으면 학습 쉽고 양성 시퀀스 수 증가, 길면 장기 패턴 포착 가능 |

### 5-4. 검증 변수
| 변수 | 옵션 | 설명 / 가설 |
|---|---|---|
| CV 전략 (= 학습/검증 데이터 분할 방식, Cross Validation) | hold-out (= train/val/test로 한 번 고정 분할) / walk-forward 5-fold (= 시간순으로 학습 윈도우를 굴리며 5번 평가) | 시계열은 walk-forward가 미래 누수 방지에 정확 |
| 평가 시점 | val / test / backtest 3케이스 (= 우크라이나/수단/가자 실제 사건일 사후 평가) | test 결과만으로 결론짓지 말 것 — 백테스트 케이스에서도 신호 잡혀야 의미 |

---

## 6. 4명 분담 제안

#### 미리 알아둘 평가 지표 용어
- **PR-AUC** (= Precision-Recall AUC): 불균형 데이터의 핵심 지표. 양성 4%인 우리 환경에서 ROC-AUC보다 정직함
- **persistence_gain**: "우리 모델 PR-AUC − Persistence baseline PR-AUC". `>0` 이어야 합격선
- **top-k precision**: 모델이 위험 상위 k% 국가로 뽑은 것 중 실제 양성 비율 (운영 알람 정확도)
- **Brier Score / ECE**: 모델이 뱉은 확률값이 실제 발생 빈도와 얼마나 일치하는가 (낮을수록 좋음, calibration 측정)
- **AUC** (ROC-AUC): 양성/음성 무작위 쌍을 올바르게 순위 매길 확률 (논문 비교용)

| 담당 | 카테고리 | 담당 모델 | 주요 산출물 |
|---|---|---|---|
| A | 선형 + 베이스라인 | LogReg 변형 3종 + Persistence | regularization (= L1/L2/ElasticNet) 비교 표 |
| B | 트리 부스팅 | LightGBM, XGBoost, +SE | PR-AUC + persistence_gain 표 |
| C | 딥러닝 시퀀스 | Macis AE 재학습, LSTM 분류기 | 재현 AUC 보고 + SE 피처값 parquet으로 저장(B 카테고리에 전달) |
| D | 앙상블 + 캘리브레이션 | Stacking, Platt/Isotonic | Brier/ECE + top-k precision |

공통: 각자 본인 모델로 **5-1, 5-2 ablation 표** 제출.

---

## 7. 회의에서 결정할 것

1. **lookahead 3d 유지** vs 2d로 좁힘 (5-1 근거 표 참조 — 3d가 인과·양성·운영가치 절충, 2d ablation으로 검증 권장)
2. **ACLED 결측 마스크 컬럼** 추가 여부 (EDA ① 권장안)
3. **Train 시작일** — 2014 유지 vs 2016/2018 후행화 (EDA ②)
4. **CV 전략** — hold-out 유지 vs walk-forward 도입
5. **시간 한정상 ablation 우선순위** (5-1~5-4 중 무엇부터)

---

## 8. 참고 자료 (별도 공유)

- **eda-summary.md** — 12년 EDA 발견 7가지 (공유 완료)
- **macis2024_breaking_the_trend.md** — Macis(2024) 논문 전문 정리 (별도 공유 예정)
- **model-plan.md** — 본 스터디와 짝, 구현 순서/일정/산출물 정리 (함께 공유)
