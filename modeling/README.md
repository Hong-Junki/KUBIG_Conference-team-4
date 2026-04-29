# Conflict Early Warning — LightGBM Baseline

`y_escalation` (onset + 급격 악화)을 예측하는 LightGBM binary classifier baseline.

---

## 실행 환경

Python 3.10+, 프로젝트 루트(`conflict-early-warning/`)에서 실행.

```bash
pip install -r modeling/requirements.txt
```

---

## 실행 순서

### Step 1 — 학습

```bash
python modeling/train_lightgbm.py
```

- `train.parquet`으로 LightGBM을 학습한다.
- `val.parquet`으로 early stopping과 성능 평가를 수행한다.
- test set은 이 단계에서 **절대 사용하지 않는다**.

출력 파일:

```
outputs/models/lightgbm_baseline.pkl
outputs/reports/lightgbm_val_metrics.json
```

### Step 2 — 예측

```bash
python modeling/predict_lightgbm.py
```

- 저장된 모델을 로드하고 `test.parquet` 전체에 대해 `y_prob`를 생성한다.
- 제출 CSV 형식을 자동으로 검증한다.

출력 파일:

```
outputs/predictions/predictions__lightgbm__byeonghyeon.csv
```

---

## 출력 파일 구조

```
outputs/
├── models/
│   └── lightgbm_baseline.pkl          # model + feature_cols + country_categories
├── predictions/
│   └── predictions__lightgbm__byeonghyeon.csv   # date, country, y_prob
└── reports/
    └── lightgbm_val_metrics.json      # val PR-AUC, P@top5%, R@P>=0.10, ECE
```

### 제출 CSV 형식

```
date,country,y_prob
2024-07-01,AFG,0.1234
2024-07-01,UKR,0.7823
...
```

- 행 수: 15,718 (test set 전체)
- `date`: YYYY-MM-DD (UTC)
- `country`: ISO3 코드
- `y_prob`: 0~1 실수

---

## 모듈 구조

| 파일 | 역할 |
|------|------|
| `utils.py` | 경로 상수, 데이터 로드, feature 컬럼 계산, 제출 검증 |
| `evaluate.py` | PR-AUC, P@top5%, R@P≥0.10, ECE 계산 |
| `train_lightgbm.py` | 학습 + val 평가 + 모델 저장 |
| `predict_lightgbm.py` | 저장된 모델로 test 예측 + CSV 저장 |

---

## 설계 원칙

- **test set 격리**: `train_lightgbm.py`는 test set을 로드하지 않는다.
- **feature 자동 계산**: 제외 컬럼(LABEL_META_COLS + date)을 실제 데이터에서 빼서 feature를 자동으로 계산한다. 하드코딩 없음.
- **모델 artifact**: 학습한 feature 컬럼 목록과 country 카테고리 매핑을 모델과 함께 저장하여 predict 단계에서 재현성을 보장한다.
- **클래스 불균형**: `scale_pos_weight = neg_count / pos_count`로 대응 (train 기준 약 20배).
- **early stopping**: val PR-AUC 기준, 50 rounds patience.

---

## 다음 단계 (이후 실험)

- [ ] Phase 4: SE score (`full_se.parquet`) 포함 실험
- [ ] Optuna 하이퍼파라미터 튜닝
- [ ] SHAP feature importance 시각화
- [ ] 웹 대시보드에서 predictions CSV 연결
