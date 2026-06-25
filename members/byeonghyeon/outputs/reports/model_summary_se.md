# 최종 후보 모델 요약 — LightGBM + SE Score

작성일: 2026-04-29

---

## 1. 사용 모델

**LightGBM binary classifier + macis_se_score (Macis SE score)**

- `boosting_type`: gbdt
- `learning_rate`: 0.05
- `num_leaves`: 63
- `scale_pos_weight`: 20.04 (train neg/pos 비율 자동 계산)
- `early_stopping`: val average_precision 기준, 50 rounds patience
- `best_iteration`: 112
- `random_seed`: 42

---

## 2. 사용 데이터

| 파일 | 용도 | 행수 |
|------|------|------|
| `input/processed/dataset/train.parquet` | 모델 학습 | 42,340 |
| `input/processed/dataset/val.parquet` | early stopping + 성능 평가 | 10,556 |
| `input/processed/dataset/test.parquet` | 최종 예측 (학습/튜닝 사용 금지) | 15,718 |
| `input/processed/dataset/full_se.parquet` | macis_se_score 추출용 | 68,614 |

`full_se.parquet`에서는 `date, country, macis_se_score` 3개 컬럼만 추출하여 train/val/test에 `date + country` 기준으로 left merge.  
병합 후 row 수 변화 없음, 결측치 0건.

---

## 3. 학습 타겟

```
y_escalation  (onset + 급격 악화 여부)
```

- 양성 비율: train 4.75% / val 4.07% / test 4.06%
- `y`(양성 69.99%)는 persistence baseline에 지배되어 사용 금지

---

## 4. 제외 컬럼

```python
LABEL_META_COLS = [
    "y",                    # 라벨 (사용 금지)
    "y_onset",              # 보조 라벨
    "y_escalation",         # 주 타겟 (X에서 제외)
    "fatalities_next3d",    # 미래 정보 누수
    "event_count_next3d",   # 미래 정보 누수
    "past14d_event_count",  # 라벨 보조 컬럼
    "past14d_fatalities_mean",  # 라벨 보조 컬럼
]
DATE_COL = "date"           # 시간 인덱스 (직접 feature 사용 안 함)
```

---

## 5. Feature 구성

| 구분 | 개수 | 내용 |
|------|------|------|
| Numeric | 55 | ACLED 20개 + GDELT 20개 + 경제지표 15개 + macis_se_score 1개 |
| Categorical | 1 | `country` (ISO3, LightGBM categorical feature) |
| **합계** | **56** | baseline 55개 + macis_se_score 1개 추가 |

feature 수는 실제 데이터 컬럼에서 제외 컬럼을 뺀 뒤 자동 계산 (하드코딩 없음).

---

## 6. Validation 성능

val set 기준 (2024-01-01 ~ 2024-06-30, 10,556행):

| 지표 | 값 |
|------|-----|
| **PR-AUC** | **0.1628** |
| **P@top5%** | **0.2254** |
| **R@P≥0.10** | **0.6209** |
| ECE | 0.1829 |
| Positive rate | 0.0407 |
| Best iteration | 112 |

---

## 7. Baseline 대비 개선폭

| 지표 | Baseline LightGBM | LightGBM + SE | Delta |
|------|-------------------|---------------|-------|
| PR-AUC | 0.1173 | **0.1628** | **+0.0455** |
| P@top5% | 0.1477 | **0.2254** | **+0.0777** |
| R@P≥0.10 | 0.4419 | **0.6209** | **+0.1791** |
| ECE | 0.2314 | **0.1829** | **−0.0484** (개선) |
| Best iteration | 38 | 112 | — |

팀 가이드 참고 기준치 (test set):

| 모델 | PR-AUC | persistence_gain | P@top5% |
|------|--------|-----------------|---------|
| Persistence baseline | 0.0354 | 0 | — |
| 팀 LightGBM (기존) | 0.0779 | +0.0424 | 0.120 |
| 팀 LightGBM + SE (기존 1위) | 0.1307 | +0.0952 | 0.190 |
| **이번 LightGBM + SE (val)** | **0.1628** | **+0.127↑** | **0.225** |

> ※ 이번 성능은 val 기준. test 기준 비교는 팀 평가 스크립트 실행 후 확인.

---

## 8. 생성된 Prediction 파일

```
outputs/predictions/predictions__lightgbm_se__byeonghyeon.csv
```

형식:
```
date,country,y_prob
2024-07-01,UKR,0.1219
2024-07-02,UKR,0.4234
...
2025-03-28,GTM,0.xxxx
```

- 행 수: **15,718** (test set 전체, 누락 없음)
- `date`: YYYY-MM-DD (UTC)
- `country`: ISO3 코드
- `y_prob`: 0~1 실수 (범위: [0.002, 0.916])
- 인코딩: UTF-8

---

## 9. 결론

**이 파일을 팀 최종 제출 후보로 사용한다.**

- PR-AUC, P@top5%, R@P≥0.10, ECE **4개 지표 모두 baseline 대비 개선**
- 팀 기존 LightGBM + SE 모델(PR-AUC 0.1307) 대비 val 기준 추가 개선 확인
- persistence_gain 양수 확인 (trivial baseline 초과)
- 팀원 모델과의 공식 비교는 동일 test set 기준 팀 평가 스크립트로 수행

---

## 10. 한계 및 해석 주의사항

1. **ECE 0.1829**: calibration이 적용되지 않은 상태. `y_prob = 0.4`가 실제 40% 발생 확률을 의미하지 않음. 대시보드에서는 **"모델 기반 escalation 위험 신호"** 또는 **"risk score"**로 표현해야 하며, 실제 확률로 과장하지 말 것.
2. **val 기준 성능**: test 기준 최종 수치는 팀 평가 스크립트 실행 후 확인 필요. test로 모델 선택 금지 원칙에 따라 val 성능 기준으로 이 모델을 선정함.
3. **macis_se_score 의존성**: SE score가 없는 국가·날짜에 대한 예측은 불안정할 수 있음. 현재 데이터에서는 결측 0건이나, 향후 real-time 운영 시 SE score 가용성 확인 필요.
4. **하이퍼파라미터 미튜닝**: 현재 기본 파라미터 사용. Optuna 튜닝 시 추가 성능 향상 가능성 있음.
5. **SHAP 미적용**: feature importance는 확인 가능하나, 국가별 local explanation은 구현되지 않음.
