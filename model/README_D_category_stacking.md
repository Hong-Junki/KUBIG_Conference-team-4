# D-Category 스태킹 앙상블 — 참고 코드 패키지

> **작성자**: byeonghyeon (D-category 담당)
> **작성일**: 2026-05-24
> **대상**: 최종 모델 통합 담당 팀원
> **위치**: KUBIG Conference Team 4 GitHub `model/` 폴더

---

## 이 파일들을 올린 이유

D-category는 **스태킹 앙상블 + 확률 보정**을 담당합니다.
8종의 절제 실험(ablation study)을 완료했고, 최종 모델이 확정되었습니다.

팀원이 아래 사항을 직접 확인할 수 있도록 핵심 스크립트 3개를 공유합니다:
- 현재 최선 D 모델의 전체 파이프라인 구조
- Train / Val / Test 분할 처리 방식
- 예측 대상(y_escalation)이 3d 기반임을 확인
- 7d/2d 타깃 미존재 이유
- y_escalation vs y_onset 구분
- LSTM 통합 시도 결과와 현재 제외 이유

CSV 예측 파일, 모델 아티팩트(.pkl/.pt), parquet 데이터 파일은 **의도적으로 업로드하지 않습니다**.

---

## 현재 최선 D-category 모델 구조

```
실험명 : stacking_tree_only_12y_with_mask_feature
스크립트: run_stacking_d_with_mask_feature_ablation.py

입력 데이터 (train/val/test.parquet, 2014-01-01 ~ 2025-03-28)
  │
  ├─ ACLED features         (이벤트 수, 행위자 유형 등 20개)
  ├─ GDELT/news features    (Goldstein 지수, 뉴스 톤 등 17개)
  ├─ Economic features      (DXY, Gold 등 17개)
  ├─ macis_se_score         (LSTM AE 재구성 오차 — 분쟁 이상 신호)
  ├─ acled_missing_mask     (ACLED 결측 여부 이진값)
  └─ country                (ISO3 국가 코드, 범주형)
         │  총 57개 피처
         ▼
  Expanding-window OOF (6-fold)
    F1: train 2014-2017 → predict 2018
    F2: train 2014-2018 → predict 2019
    F3: train 2014-2019 → predict 2020
    F4: train 2014-2020 → predict 2021
    F5: train 2014-2021 → predict 2022
    F6: train 2014-2022 → predict 2023
         │
  ┌──────┴──────┐
  │             │
LightGBM      XGBoost
(num_leaves=63 (max_depth=6
  spw=22)       spw=22)
  │             │
  └──────┬──────┘
         │  OOF 예측값 [lgbm_prob, xgb_prob]
         ▼
  Level 1: Logistic Regression meta model
    C=0.01, class_weight='balanced'
    학습 입력: OOF [lgbm_prob, xgb_prob]
    coef: [LGBM=1.9562, XGB=2.3605]
         │
         ▼
  Platt Calibration (LogReg, C=1.0)
         │
         ▼
  최종 출력: y_prob (country-date별 escalation 위험 확률)
```

**검증 지표** (val 2024-01~06 기준):

| 지표 | 값 |
|------|-----|
| Stacking Platt PR-AUC | **0.2714** |
| P@top5% | **0.2689** |
| ECE | 0.0083 |
| Brier Score | 0.0359 |

---

## 입력 / 출력 구조

### 입력 파일 경로 (원본 프로젝트 기준)

```
input/processed/dataset/train.parquet   (2014-01-01 ~ 2023-12-31, 211,816행)
input/processed/dataset/val.parquet     (2024-01-01 ~ 2024-06-30, 10,556행)
input/processed/dataset/test.parquet    (2024-07-01 ~ 2025-03-28, 15,718행)
output/macis_12y/se_scores.parquet      (LSTM AE 재구성 오차, date+country 병합)
```

### 최종 출력 파일 (원본 프로젝트 기준)

```
outputs/predictions/predictions__stacking_tree_only_12y_with_mask_feature_platt__D_byeonghyeon.csv
```

컬럼: `date`, `country`, `y_prob` (3개만, y_true 없음)
기간: 2024-07-01 ~ 2025-03-28 (test 전체)

이 파일이 대시보드 연결 및 최종 제출에 사용하는 파일입니다.

---

## 예측 대상: y_escalation (3일 기반)

- **TARGET_COL = "y_escalation"** — 다음 3일 내 급격한 충돌 악화 이진 지표
- Val 양성률: 4.07%, Train 양성률: 4.29%
- `y` (이벤트 발생 여부, 양성률 57.5%)나 `y_onset` (새 분쟁 발발, 양성률 1.03%)과 다름

### y_escalation vs y_onset 차이

| 타깃 | 의미 | 양성률 | D-category 사용 여부 |
|------|------|--------|----------------------|
| `y_escalation` | 기존 분쟁 지역에서 충돌이 급격히 악화됨 | 4.29% | **사용** (현재 타깃) |
| `y_onset` | 평화 지역에서 분쟁이 처음 시작됨 | 1.03% | 미사용 (피처에서 제외) |
| `y` | 다음 3일 내 분쟁 이벤트 발생 여부 | 57.5% | 미사용 (피처에서 제외) |

`y_onset`은 `y_escalation`의 서브셋 (y_onset=1이면 항상 y_escalation=1).
`y_onset`은 양성 수가 너무 적어 단독 학습이 불안정하며, 현재 D 실험에서는 레이블 누출 방지를 위해 피처에서 제외만 했습니다.

### 피처에서 항상 제외하는 컬럼 (ALWAYS_EXCLUDE / LABEL_META_COLS)

```python
LABEL_META_COLS = [
    "y",
    "y_onset",
    "y_escalation",       # TARGET_COL 자체
    "fatalities_next3d",  # 미래 누출
    "event_count_next3d", # 미래 누출
    "past14d_event_count",
    "past14d_fatalities_mean",
]
```

---

## 7d/2d 타깃 현재 없음

현재 parquet 파일(train/val/test)에 아래 컬럼들이 **존재하지 않습니다**:

```
event_count_next7d     ← 없음
fatalities_next7d      ← 없음
y_escalation_7d        ← 없음
event_count_next2d     ← 없음
y_escalation_2d        ← 없음
```

따라서 **7d/2d lookahead ablation을 현재 실행할 수 없습니다.**
실행하려면 전처리 파이프라인에서 위 컬럼을 생성하고 parquet을 재생성해야 합니다.

---

## Val / Test 파일 구분

이 점이 팀원 간 혼동을 일으켰기 때문에 명확히 정리합니다.

### val_predictions__ 파일 (검증용, 제출 불가)

```
컬럼: date, country, y_true, y_prob_lgbm, y_prob_xgb, y_prob_stack_raw, y_prob_stack_platt, y_prob_stack_isotonic
기간: 2024-01-01 ~ 2024-06-30 (val 기간)
```

- `y_true` 포함 — val.parquet의 레이블 (test 레이블 아님)
- 모델 비교, C 탐색, calibration 학습, PR-AUC/P@5%/ECE 계산에 사용
- **대시보드에 연결하면 안 됨** — val 기간만 예측, y_true 포함

### predictions__ 파일 (대시보드/제출용)

```
컬럼: date, country, y_prob (3개만)
기간: 2024-07-01 ~ 2025-03-28 (test 기간)
```

- `y_true` 없음 — test 레이블은 어떤 모델 선택에도 사용하지 않음
- **대시보드 연결 및 최종 제출에 사용하는 파일**

**보고된 모든 PR-AUC, P@5%, ECE 수치는 validation 기준입니다. test 레이블은 모델 선택에 일절 사용하지 않았습니다.**

---

## LSTM 결과 요약

### 1. standalone LSTM baseline (`run_lstm_classifier_baseline.py`)

- 모델: LSTM Classifier, sequence length = 30일, target = y_escalation
- OOF 구조: Expanding-window 6-fold (동일 구조)
- **val PR-AUC: ~0.103** (tree-only 0.26~0.27 대비 크게 낮음)
- 결론: 단독으로는 tree-only 대비 성능 열위

### 2. LGBM + XGB + LSTM stacking (`run_stacking_d_lgbm_xgb_lstm_ablation.py`)

- Level 0에 LSTM 예측을 추가한 3-model stacking
- LGBM-LSTM 예측 상관관계: 0.487, XGB-LSTM: 0.519 (다양성은 존재)
- 그러나 stacking 후 성능:

| 실험 | val Platt PR-AUC | P@5% | ECE |
|------|-----------------|------|-----|
| Tree-only (최선) | **0.2714** | 0.2689 | 0.0083 |
| LGBM+XGB+LSTM | 0.2656 | 0.2670 | 0.0067 |
| delta | −0.0058 | −0.0019 | 개선 |

- **결론: 현재 LSTM 품질(PR-AUC ~0.10)로는 stacking에 추가 시 잡음 효과 > 다양성 효과**
- LSTM 성능이 PR-AUC 0.20 이상으로 개선되면 `run_stacking_d_lgbm_xgb_lstm_ablation.py`에 개선된 LSTM OOF/val/test 파일 경로를 교체해서 재실험 가능

---

## 8종 절제 실험 결과 요약

| 실험 | val PR-AUC | P@5% | ECE | 결론 |
|------|-----------|------|-----|------|
| `stacking_tree_only_12y` (기준) | 0.2656 | 0.2614 | 0.0074 | 기준선 |
| no-SE ablation | 0.1057 | 0.1591 | 0.0035 | SE 필수 |
| **with_mask_feature** | **0.2714** | **0.2689** | **0.0083** | **★ 최선** |
| mask0-only ablation | 0.2512 | 0.2614 | 0.0066 | mask=1 유지 |
| 2016-start (8y) ablation | 0.2496 | 0.2576 | 0.0081 | 2014-start 유지 |
| LGBM+XGB+LSTM stacking | 0.2656 | 0.2670 | 0.0067 | LSTM 현재 제외 |
| feature group ablation v2 (8종) | 0.1670~0.2697 | — | — | ACLED 핵심 확인 |
| hyperparameter sensitivity (9종) | ≤0.2551 | — | — | 현재 설정 유지 |

---

## 최종 결론 — 현재 유지 결정 사항

- **SE(`macis_se_score`) 반드시 포함** — 제거 시 PR-AUC −0.16 급락
- **`acled_missing_mask` 피처로 포함** — PR-AUC +0.006 개선
- **mask=1 행 제거하지 않음** — 제거 시 PR-AUC −0.020
- **Train 시작일 2014 유지** — 2016-start 시도했으나 PR-AUC −0.022
- **현재 LSTM Level 0에서 제외** — stacking 추가 시 PR-AUC −0.006
- **최종 D 모델 후보**: `stacking_tree_only_12y_with_mask_feature` + Platt (val PR-AUC = **0.2714**)

---

## 스크립트 파일 목록

| 파일 | 역할 | 비고 |
|------|------|------|
| `run_stacking_d_with_mask_feature_ablation.py` | **현재 최선 D 모델** 전체 파이프라인 | 최종 제출 스크립트 |
| `run_lstm_classifier_baseline.py` | standalone LSTM classifier baseline | OOF/val/test 예측 생성용 |
| `run_stacking_d_lgbm_xgb_lstm_ablation.py` | LGBM+XGB+LSTM stacking 실험 | LSTM 통합 로직 참고용 |

> **참고**: 원본 프로젝트 repo에서 `from evaluate import ...` 형태로 `modeling/evaluate.py`를 사용합니다. 스크립트를 다른 환경에서 실행하려면 `evaluate.py`도 함께 필요합니다.
