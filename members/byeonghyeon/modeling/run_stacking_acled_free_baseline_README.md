# 실험 B: ACLED-free Stacking Baseline 실행 가이드

## 목적

ACLED 기반 피처와 macis_se_score를 제거하고,  
GDELT events(19) + 경제지표(15) + country(1) = **35개 피처**만으로  
기존 stacking 구조를 동일하게 학습해 성능 baseline을 측정한다.

이 결과가 실험 C (ACLED-free + GDELT titles)의 비교 기준점이 된다.

---

## 전제 조건

### 필요 파일

```
conflict-early-warning/
└── input/processed/dataset/
    ├── train.parquet   (211,816행 × 64컬럼)
    ├── val.parquet     (10,556행  × 64컬럼)
    └── test.parquet    (15,718행  × 64컬럼)
```

> 팀 레포(`KUBIG_Conference-team-4/`)에는 parquet 파일이 없다.  
> 실험은 반드시 `conflict-early-warning/` 개인 레포 루트에서 실행해야 한다.

### 필요 패키지

```bash
pip install lightgbm xgboost scikit-learn pandas numpy pyarrow
```

`evaluate.py`는 같은 디렉토리(`members/byeonghyeon/modeling/`)에 있으며 자동으로 임포트된다.

---

## 실행 방법

> **중요**: 스크립트 파일은 **팀 레포**(`KUBIG_Conference-team-4/`)에 있다.  
> 학습 데이터는 **개인 레포**(`conflict-early-warning/`)에 있다.  
> 스크립트는 자신의 위치에서 `DATA_ROOT`를 자동으로 계산하므로, 보통 방법 1만으로 충분하다.

### 방법 1: 팀 레포 루트에서 실행 (권장, DATA_ROOT 자동 탐지)

```bash
cd /path/to/KUBIG_Conference-team-4

python members/byeonghyeon/modeling/run_stacking_acled_free_baseline.py
```

스크립트가 자신의 위치 기준으로 `../../../conflict-early-warning`을 `DATA_ROOT`로 자동 계산한다.

### 방법 2: DATA_ROOT 환경변수로 명시 (경로가 다를 경우)

```bash
DATA_ROOT=/path/to/conflict-early-warning \
  python /path/to/KUBIG_Conference-team-4/members/byeonghyeon/modeling/run_stacking_acled_free_baseline.py
```

### 방법 3: 개인 레포 루트에서 상대경로로 실행

```bash
cd /path/to/conflict-early-warning

python ../KUBIG_Conference-team-4/members/byeonghyeon/modeling/run_stacking_acled_free_baseline.py
```

`DATA_ROOT`가 자동 계산되어 `conflict-early-warning/input/...`을 올바르게 탐지한다.

### Smoke test (기능 검증, ~5분)

전체 실행 전에 경로·feature 설정·파이프라인 흐름을 빠르게 검증한다.

```bash
cd /path/to/KUBIG_Conference-team-4

SMOKE_TEST=1 python members/byeonghyeon/modeling/run_stacking_acled_free_baseline.py
```

Smoke test 설정:
- OOF: F6 1개 fold만 (train ≤2022 → predict 2023)
- 라운드: OOF 50, Final 100 (전체의 10%)
- 출력 파일명에 `_smoke` 접미사 추가

> Smoke test 결과는 성능 참고용이 아님. 파이프라인 동작 확인용.

---

## 예상 실행 시간

| 단계 | 예상 시간 |
|------|----------|
| 데이터 로드 | < 1분 |
| OOF 6-fold (LGBM + XGB) | 30~60분 |
| Final LGBM + XGB 학습 | 10~20분 |
| Meta LogReg + Calibration | < 1분 |
| **총계** | **약 45~90분** |

> OOF 학습이 CPU bound. 멀티코어 머신에서 더 빠름.

---

## 출력

학습 완료 후 아래 파일 1개만 생성된다. **예측 CSV는 저장하지 않는다.**

```
members/byeonghyeon/outputs/reports/acled_free_baseline_results.md
```

리포트 포함 내용:
- 피처 구성 (35개)
- val 지표 전체 표 (LGBM, XGB, Stacking raw/Platt/Isotonic)
- Reference 모델(0.2714) 대비 delta
- OOF fold별 PR-AUC 요약
- Meta LogReg 선택된 C 값

---

## 결과 해석 기준

| 항목 | 기준 |
|------|------|
| 실험 C 진행 여부 | 실험 B 결과를 먼저 확인 후 결정 |
| 실험 C 개선 임계값 | B val Stacking Platt PR-AUC + **0.003** 이상 |
| Reference 비교 | Reference(0.2714)는 ACLED+SE 포함이므로 **직접 비교 대상 아님** |

---

## 검증 단계 (Step 0 출력 확인)

실행 시 스크립트가 자동으로 아래를 검증한다:

```
[Step 0] Feature 검증
  feature_cols (35개):
    GDELT events : 19개  — [gdelt_goldstein_mean_7d, ...]
    economic     : 15개  — [econ_vix, ...]
    other/country:  1개  — [country]
  ACLED/SE 컬럼 완전 제거 확인 ✅
```

이 출력 없이 학습이 진행되면 중단하고 확인해야 한다.

---

## 파일 구조

```
members/byeonghyeon/modeling/
├── run_stacking_acled_free_baseline.py   ← 실험 B 스크립트
├── run_stacking_acled_free_baseline_README.md  ← 이 파일
├── evaluate.py                            ← 지표 계산 모듈 (의존성)
└── ...

members/byeonghyeon/outputs/reports/
└── acled_free_baseline_results.md        ← 실행 결과 저장 위치
```

---

## 주의사항

1. **기존 스크립트를 덮어쓰지 않는다.** 이 스크립트는 신규 파일이다.
2. **macis_se_score를 merge하지 않는다.** SE_PATH 로드 코드가 없다.
3. **예측 CSV를 저장하지 않는다.** 리포트 markdown만 생성된다.
4. **val 지표가 낙관적 추정임을 항상 인지한다.**  
   val이 early stopping / C 탐색 / Platt calibration에 모두 사용된다.
5. **실험 C 진행 전에 반드시 이 파일의 결과를 먼저 확인한다.**
