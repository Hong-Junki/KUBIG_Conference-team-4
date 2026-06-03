# PROJECT HANDOFF — Conflict Early Warning

> 작성일: 2026-04-29  
> 목적: 세션 종료 후 다음 세션에서 맥락 없이 바로 이어받을 수 있도록 현재 작업 상태를 완전히 기록

---

## 1. 프로젝트 목적

ACLED(정치적 폭력 이벤트), GDELT(글로벌 뉴스), 경제지표(VIX/WTI/Gold/DXY/STLFSI4)를 사용해 국가별 무력충돌 escalation 위험을 예측하는 조기경보 시스템.

**이번 세션 목표 (Phase 1~3):**
1. `y_escalation`을 타겟으로 LightGBM baseline 모델을 학습
2. test set 전체에 대한 예측 파일 `predictions__lightgbm__byeonghyeon.csv`를 생성
3. 이후 웹 대시보드에서 해당 CSV를 읽어 시각화하는 구조를 만들기 위한 기반 확보

**전체 PRD 및 팀 가이드:** `PRD.md`, `team-collaboration-guide.md` 참조.

---

## 2. 프로젝트 루트 및 폴더 구조

**프로젝트 루트:**
```
/Users/byeonghyeonkim/Desktop/공부/활동/KUBIG/26-1 Vibe Coding/conflict-early-warning/
```

**현재 폴더 구조:**
```
conflict-early-warning/
├── PRD.md                          # 전체 제품 요구사항 정의서
├── team-collaboration-guide.md     # 팀원 공유 데이터·모델 제출 가이드
├── PROJECT_HANDOFF.md              # 이 문서
│
├── input/
│   └── processed/
│       ├── dataset/
│       │   ├── train.parquet       # 학습 셋
│       │   ├── val.parquet         # 검증 셋
│       │   ├── test.parquet        # 테스트 셋 (예측 전용)
│       │   ├── full.parquet        # 전체 데이터
│       │   └── full_se.parquet     # SE score 포함 전체 데이터
│       ├── features/
│       │   ├── baseline_scores.parquet  # 국가별 5년 사상자 prior B(0-100)
│       │   ├── se_scores.parquet        # Macis SE score 단독
│       │   └── features.parquet         # 중간 feature 산출물
│       └── acled/, gdelt/, economic/, labels/   # 중간 산출물
│
├── modeling/                       # ← 이번 세션에서 생성
│   ├── utils.py
│   ├── evaluate.py
│   ├── train_lightgbm.py
│   ├── predict_lightgbm.py
│   ├── requirements.txt
│   └── README.md
│
└── outputs/                        # ← 이번 세션에서 생성
    ├── models/
    │   └── lightgbm_baseline.pkl   # 학습된 모델 artifact
    ├── predictions/
    │   └── predictions__lightgbm__byeonghyeon.csv
    └── reports/
        └── lightgbm_val_metrics.json
```

**아직 없는 폴더:** `dashboard/` (Phase 5에서 생성 예정)

---

## 3. 사용 데이터 파일

| 파일 | 행수 | 컬럼 | 기간 | 이번 세션 사용 여부 |
|------|------|------|------|---------------------|
| `train.parquet` | 42,340 | 63 | 2022-01-01 ~ 2023-12-31 | ✅ 학습 |
| `val.parquet` | 10,556 | 63 | 2024-01-01 ~ 2024-06-30 | ✅ 검증 |
| `test.parquet` | 15,718 | 63 | 2024-07-01 ~ 2025-03-28 | ✅ 예측 |
| `full.parquet` | 68,614 | 63 | 전체 | 미사용 |
| `full_se.parquet` | 68,614 | 64 | 전체 + SE | 미사용 (Phase 4 예정) |
| `baseline_scores.parquet` | — | — | — | 미사용 (대시보드 연동 시 사용 예정) |
| `se_scores.parquet` | 66,890 | — | — | 미사용 (Phase 4 예정) |

**컬럼 구성 (63개):**
- ACLED 피처: 20개 (`acled_event_count_7d`, `acled_fatalities_7d`, ... 등)
- GDELT 피처: 20개 (`gdelt_goldstein_mean_7d`, `gdelt_tone_mean_7d`, ... 등)
- 경제지표 피처: 15개 (`econ_vix`, `econ_wti`, `econ_gold`, ... 등)
- categorical: 1개 (`country`)
- 라벨/메타: 7개 (아래 제외 컬럼 참조)
- 날짜: 1개 (`date`)

---

## 4. 모델링 규칙

### 타겟

```
y_escalation  (onset + 급격 악화 여부)
```

- 양성 비율: train 4.75%, val 4.07%, test 4.06%
- `y`(양성 69.99%)는 persistence baseline에 지배되어 사용 금지

### 제외 컬럼 (LABEL_META_COLS + date)

```python
LABEL_META_COLS = [
    "y", "y_onset", "y_escalation",
    "fatalities_next3d", "event_count_next3d",
    "past14d_event_count", "past14d_fatalities_mean",
]
DATE_COL = "date"
```

제외 후 사용 feature: **55개** (numeric 54 + categorical 1 `[country]`)

### Split 사용 원칙

| Split | 용도 | 금지 사항 |
|-------|------|-----------|
| train | 모델 학습 | — |
| val | early stopping, 성능 평가, 모델 선택 | test 대체 사용 금지 |
| test | 최종 예측 CSV 생성만 | 학습·튜닝·모델 선택 절대 금지 |

### 기타 학습 설정

- `scale_pos_weight = neg_count / pos_count` (train 기준 약 20.04)
- `random seed = 42`
- early stopping: val `average_precision` 기준, 50 rounds patience
- `country`: LightGBM categorical feature로 사용 (pandas Categorical 변환)

---

## 5. 생성된 modeling 파일 목록

| 파일 | 역할 |
|------|------|
| `modeling/utils.py` | 경로 상수, 데이터 로드, feature 컬럼 자동 계산, 제출 CSV 검증 |
| `modeling/evaluate.py` | PR-AUC, P@top5%, R@P≥0.10, ECE 계산 함수 |
| `modeling/train_lightgbm.py` | 학습 + val 평가 + 모델 artifact 저장 |
| `modeling/predict_lightgbm.py` | 저장된 모델로 test 예측 + CSV 저장 및 검증 |
| `modeling/requirements.txt` | 의존성 (pandas, pyarrow, numpy, scikit-learn, lightgbm, joblib) |
| `modeling/README.md` | 실행 방법, 설계 원칙 설명 |

**설계 포인트:**
- feature 컬럼 수는 하드코딩 없이 실제 데이터에서 자동 계산
- `lightgbm_baseline.pkl`에 model + feature_cols + country_categories를 함께 저장하여 predict 단계에서 train 데이터 없이도 동일 feature 구성 재현 가능
- `train_lightgbm.py`는 test set을 로드하지 않음 (설계 수준에서 격리)

---

## 6. 실행한 명령어

```bash
# 의존성 설치 (최초 1회)
pip install -r modeling/requirements.txt

# macOS libomp 설치 (LightGBM 실행 필수, 최초 1회)
brew install libomp

# Step 1: 학습
python modeling/train_lightgbm.py

# Step 2: 예측
python modeling/predict_lightgbm.py
```

**주의:** 모든 명령어는 프로젝트 루트(`conflict-early-warning/`)에서 실행해야 함.

---

## 7. Baseline LightGBM Validation 성능

| 지표 | 값 | 비고 |
|------|----|------|
| **PR-AUC** | **0.1173** | 팀 LightGBM baseline(0.0779)보다 높음 |
| **P@top5%** | **0.1477** | 합격선 0.10 초과 |
| **R@P≥0.10** | **0.4419** | — |
| ECE | 0.2314 | calibration 미적용, 개선 여지 있음 |
| positive rate (val) | 0.0407 | — |
| best iteration | 38 | (early stopping) |
| scale_pos_weight | 20.04 | train neg/pos 비율 |

**팀 가이드 기준 비교 (test set 기준 참고값):**

| 모델 | PR-AUC | persistence_gain | P@top5% |
|------|--------|-----------------|---------|
| Persistence baseline | 0.0354 | 0 | — |
| 팀 LightGBM (기존) | 0.0779 | +0.0424 | 0.120 |
| **이번 baseline (val)** | **0.1173** | **+0.082** | **0.148** |
| 팀 LightGBM + SE (1위) | 0.1307 | +0.0952 | 0.190 |

> 이번 val 성능이 팀 기존 LightGBM test 성능보다 높게 나온 이유: early stopping 라운드(38), 하이퍼파라미터, 피처셋 구성의 차이일 수 있음. test 기준 비교는 predict 후 평가 스크립트를 따로 돌려야 함.

---

## 8. 생성된 Output 파일 경로

| 파일 | 경로 | 크기 |
|------|------|------|
| 모델 artifact | `outputs/models/lightgbm_baseline.pkl` | 264 KB |
| val 성능 지표 | `outputs/reports/lightgbm_val_metrics.json` | 239 B |
| 제출 예측 CSV | `outputs/predictions/predictions__lightgbm__byeonghyeon.csv` | 531 KB |

**예측 CSV 형식:**
```
date,country,y_prob
2024-07-01,UKR,0.3482107544455627
2024-07-02,UKR,0.28078967255532306
...
2025-03-28,GTM,0.24681580548605153
```
- 행 수: 15,718 (header 제외)
- y_prob 범위: [0.013997, 0.713725]
- y_prob 평균: 0.259684

---

## 9. 현재 한계

1. **ECE 0.23**: 확률이 calibration되어 있지 않음. 실제 위험 확률로 직접 해석하면 오해 소지 있음. (Platt scaling 또는 isotonic regression으로 보정 가능)
2. **Early stopping at round 38**: num_leaves=63, learning_rate=0.05 설정에서 매우 이르게 멈춤. 하이퍼파라미터 튜닝으로 개선 여지 있음.
3. **SE score 미포함**: 팀 1위 모델(PR-AUC 0.1307)은 SE score 포함. `full_se.parquet` 병합 실험 미수행.
4. **test set 성능 미평가**: val 기준 성능만 확인. test y_escalation 라벨이 있어 평가 가능하나 아직 미수행 (PRD 원칙상 test로 모델 선택 금지이므로 참고용으로만 확인 가능).
5. **SHAP 미구현**: feature importance는 LightGBM 내장 기능으로 확인 가능하나 local SHAP explanation은 미구현.
6. **대시보드 미연결**: `predictions__lightgbm__byeonghyeon.csv`는 생성됐으나 Next.js 대시보드는 아직 없음.

---

## 10. 다음 작업 후보

### Phase 3.5 — Baseline summary 저장 (선택, 빠름)
- val 성능 + feature importance top 20을 `outputs/reports/`에 저장
- 팀원 모델과 비교할 때 참조 기록으로 사용

### Phase 4 — LightGBM + SE score 실험
- `input/processed/dataset/full_se.parquet`에서 `macis_se_score` 컬럼 추출
- train/val/test에 `date + country` key로 병합
- 기존 baseline과 val PR-AUC 비교
- 성능 향상 시 새로운 제출 CSV 생성: `predictions__lightgbm_se__byeonghyeon.csv`

### Phase 5 — 웹 대시보드 MVP
- `dashboard/` 폴더에 Next.js + TypeScript + Tailwind CSS 프로젝트 생성
- `predictions__lightgbm__byeonghyeon.csv`를 정적 파일로 로드
- 위험도 상위 국가 카드, 국가별 시계열 그래프, 모델 선택 드롭다운 구현
- `risk_score = y_prob × 100`, risk_level 분류 (Low/Moderate/High/Critical)

---

## 11. Next Session Prompt

다음 세션 시작 시 아래 프롬프트를 그대로 붙여넣으면 맥락 없이 바로 이어받을 수 있음.

---

```
PROJECT_HANDOFF.md와 PRD.md, team-collaboration-guide.md를 읽고 현재 프로젝트 상태를 파악해줘.

[현재 상태 요약]
- 프로젝트 루트: conflict-early-warning/
- Phase 1~3 완료: LightGBM baseline 학습 및 test 예측 CSV 생성 완료
- 생성된 파일:
  - modeling/ (utils.py, evaluate.py, train_lightgbm.py, predict_lightgbm.py)
  - outputs/models/lightgbm_baseline.pkl
  - outputs/reports/lightgbm_val_metrics.json (PR-AUC: 0.1173)
  - outputs/predictions/predictions__lightgbm__byeonghyeon.csv (15,718행)

[다음 작업 목표 — 아래 중 하나를 선택해서 지시할 것]

옵션 A: LightGBM + SE score 실험 (Phase 4)
  - input/processed/dataset/full_se.parquet에서 macis_se_score를 train/val/test에 병합
  - 기존 baseline과 val PR-AUC 비교
  - 새 예측 파일 predictions__lightgbm_se__byeonghyeon.csv 생성

옵션 B: 웹 대시보드 MVP (Phase 5)
  - dashboard/ 폴더에 Next.js + TypeScript + Tailwind CSS 프로젝트 생성
  - predictions__lightgbm__byeonghyeon.csv를 정적 파일로 읽어 시각화
  - 위험도 카드, 시계열 그래프, 모델 선택 드롭다운 구현

옵션 C: Baseline summary 저장 + Optuna 튜닝 (Phase 3.5)
  - val 성능 + feature importance top 20 저장
  - Optuna로 하이퍼파라미터 튜닝 후 성능 비교

어떤 옵션을 진행할지 말해주면 코드 작성 전에 계획부터 제시할게.
```
