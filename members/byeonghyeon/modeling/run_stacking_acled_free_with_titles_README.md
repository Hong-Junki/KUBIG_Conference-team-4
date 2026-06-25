# 실험 C: ACLED-free + GDELT Title Features 실행 가이드

## 목적

B baseline (GDELT events + 경제 + country = 35개 피처) 위에  
GDELT title/tone/count/domain/language aggregate feature 22개를 추가해  
성능 개선 여부를 검증한다.

**채택 기준**: Stacking Platt PR-AUC (val) ≥ **0.0594** (B baseline 0.0564 + 0.003)

---

## 피처 구성 (총 57개)

| 카테고리 | 피처 수 | 설명 |
|----------|---------|------|
| B — GDELT events | 19 | goldstein/tone/mentions/event_count/quadclass |
| B — Economic | 15 | VIX/WTI/Gold/DXY/STLFSI4 각×3 |
| B — Country | 1 | ISO3, categorical |
| **신규** GDELT title 1d | 10 | count/tone/domain/lang 당일 집계 |
| **신규** GDELT title 7d | 11 | 7일 rolling + tone_trend |
| **신규** coverage_mask | 1 | 2015-02-17 이전 BQ gap 표시 |
| **합계** | **57** | |

---

## 전제 조건

### 1. GDELT title feature parquet 생성

실험 C 실행 전 반드시 아래가 존재해야 한다:

```
members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_features.parquet
```

없으면:
```bash
python members/byeonghyeon/modeling/build_gdelt_title_features.py
```

### 2. 패키지

```bash
pip install lightgbm xgboost scikit-learn pandas numpy pyarrow
```

---

## 실행 방법

### Smoke test 먼저 (~5분)

```bash
cd /path/to/KUBIG_Conference-team-4

SMOKE_TEST=1 python members/byeonghyeon/modeling/run_stacking_acled_free_with_titles.py
```

Smoke test 설정:
- OOF: F6 1개 fold만 (train ≤2022 → predict 2023)
- 라운드: OOF 50, Final 100
- 실험명: `stacking_acled_free_with_titles_smoke`

> Smoke test 결과는 성능 참고용 아님. 파이프라인 동작 확인용.

### 전체 실행 (~45~90분)

```bash
cd /path/to/KUBIG_Conference-team-4

python members/byeonghyeon/modeling/run_stacking_acled_free_with_titles.py
```

기본 DATA_ROOT 자동 탐지: `../conflict-early-warning` (스크립트 위치 기준)

### DATA_ROOT 명시

```bash
DATA_ROOT=/path/to/conflict-early-warning \
  python members/byeonghyeon/modeling/run_stacking_acled_free_with_titles.py
```

---

## 예상 실행 시간

| 단계 | 예상 시간 |
|------|----------|
| 데이터 로드 + GDELT merge | < 1분 |
| OOF 6-fold (LGBM + XGB) | 30~60분 |
| Final LGBM + XGB | 10~20분 |
| Meta + Calibration + 지표 | < 1분 |
| **총계** | **약 45~90분** |

---

## 출력

```
members/byeonghyeon/outputs/reports/acled_free_with_titles_results.md
```

포함 내용:
- 채택 여부 (PR-AUC ≥ 0.0594 여부)
- B vs C 지표 비교표 (PR-AUC, P@5%, R@P≥.10, Brier, ECE)
- OOF fold별 요약
- Meta LogReg 선택된 C 값

예측 CSV는 저장하지 않음.

---

## 결과 해석 기준

| 결과 | 의미 | 다음 단계 |
|------|------|----------|
| PR-AUC ≥ 0.0594 | ✅ C 채택 — GDELT titles 피처 효과 있음 | 실험 D (themes/persons) 진행 |
| PR-AUC < 0.0594 | ❌ C 미채택 — GDELT titles 피처 효과 없음 | 다른 피처 탐색 또는 실험 종료 |

> B baseline(0.0564)과의 비교가 핵심.  
> Reference(0.2714)는 ACLED+SE 포함 모델로 직접 비교 대상이 아님.  
> test는 아직 평가하지 않는다 — B/C/D 비교 후 최종 모델 결정 시 1회만 평가.

---

## 파일 구조

```
members/byeonghyeon/
├── modeling/
│   ├── run_stacking_acled_free_with_titles.py     ← 실험 C 스크립트
│   ├── run_stacking_acled_free_with_titles_README.md  ← 이 파일
│   ├── run_stacking_acled_free_baseline.py        ← 실험 B (참고)
│   ├── build_gdelt_title_features.py              ← GDELT feature 생성
│   └── evaluate.py                                ← 지표 계산 모듈
├── input/processed/gdelt_titles/
│   └── gdelt_title_features.parquet               ← C 필수 입력 (gitignore)
└── outputs/reports/
    └── acled_free_with_titles_results.md          ← C 결과 저장
```
