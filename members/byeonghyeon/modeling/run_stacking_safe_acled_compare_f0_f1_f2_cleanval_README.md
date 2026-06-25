# Clean Validation 비교: F0/F1/F2 실행 가이드

## 목적

기존 F0/F1 스크립트에서 val set을 early stopping, meta C 선택, calibration fit, 평가에
중복 사용하던 문제를 해결한 공정 비교 스크립트.

**핵심 변경**: val_eval(2024-H1)을 어떤 학습/튜닝/calibration에도 사용하지 않고
순수 평가 전용으로만 사용한다.

---

## Split 설계

| Split | 기간 | 행수 (약) | 역할 |
|-------|------|----------|------|
| `train_fit` | 2014-01-01 ~ 2022-12-31 | 190,646 | base model 학습, OOF 생성, meta learner 학습 |
| `tune_cal`  | 2023-01-01 ~ 2023-12-31 | 21,170  | early stopping, meta C 선택, calibration fit |
| `val_eval`  | 2024-01-01 ~ 2024-06-30 | 10,556  | **순수 평가 전용** |
| `test`      | 2024-07-01 ~             | 15,718  | 미사용 (최종 확정 후 1회) |

### 기존 스크립트 대비 차이

| 단계 | 기존 F0/F1 | 이 스크립트 |
|------|-----------|------------|
| Final base model early stopping | val (2024-H1) ⚠️ | tune_cal (2023) ✅ |
| Meta LogReg C 선택 | val (2024-H1) ⚠️ | tune_cal (2023) ✅ |
| Platt/Isotonic calibration fit | val (2024-H1) ⚠️ | tune_cal (2023) ✅ |
| PR-AUC/ECE/Brier 평가 | val (fit=eval → 낙관적) ⚠️ | val_eval (fit≠eval → 해석 가능) ✅ |

---

## OOF fold 설계 (train_fit 내부)

| fold | 학습 기간 | 예측 연도 |
|------|----------|---------|
| OOF_F1 | ≤2017 | 2018 |
| OOF_F2 | ≤2018 | 2019 |
| OOF_F3 | ≤2019 | 2020 |
| OOF_F4 | ≤2020 | 2021 |
| OOF_F5 | ≤2021 | 2022 |

2023은 tune_cal로, 2024는 val_eval로 분리됨.

---

## 비교 실험

| 실험 | feature 수 | 구성 |
|------|-----------|------|
| F0_clean | 50 | B 35 + safe ACLED 15 |
| F1_clean | 72 | F0 + GDELT title 21 + coverage_mask 1 |
| F2_clean | 94 | F1 + GDELT theme/person 22 |

한 번 실행으로 F0→F1→F2 순서로 학습·평가.

---

## 사전 준비

| 파일 | 생성 스크립트 | 비고 |
|------|--------------|------|
| `conflict-early-warning/.../train.parquet` | 팀 pipeline | |
| `conflict-early-warning/.../val.parquet` | 팀 pipeline | |
| `conflict-early-warning/.../test.parquet` | 팀 pipeline | |
| `.../acled_safe/safe_acled_lag_features.parquet` | `build_safe_acled_lag_features.py` | gitignore 대상 |
| `.../gdelt_titles/gdelt_title_features.parquet` | `build_gdelt_title_features.py` | gitignore 대상 |
| `.../gdelt_titles/gdelt_theme_person_features.parquet` | `build_gdelt_theme_person_features.py` | F2용, 없으면 F2 skip |

---

## 실행 방법

### Smoke test (~10분, 파이프라인 검증)

```bash
cd <KUBIG_Conference-team-4 루트>
SMOKE_TEST=1 python members/byeonghyeon/modeling/run_stacking_safe_acled_compare_f0_f1_f2_cleanval.py
```

smoke test: OOF 1 fold (OOF_F5: train≤2021 → predict 2022), 라운드 축소.

### 전체 실행 (full run, ~60–120분)

```bash
python members/byeonghyeon/modeling/run_stacking_safe_acled_compare_f0_f1_f2_cleanval.py
```

---

## 출력

| 항목 | 경로 |
|------|------|
| 비교 리포트 | `members/byeonghyeon/outputs/reports/safe_acled_cleanval_f0_f1_f2_comparison.md` |
| 예측 CSV | 저장 없음 |

---

## ECE/Brier 해석

- **기존 F0/F1**: calibration fit과 eval이 동일 val → ECE≈0은 수학적으로 당연한 값 (무의미)
- **이 스크립트**: tune_cal(2023)에서 fit, val_eval(2024)에서 eval → fit≠eval → ECE/Brier 해석 가능

단, Platt calibration은 단조 변환이므로 PR-AUC는 raw/Platt 간 거의 동일함. **모델 선택 기준은 PR-AUC**.

---

## 채택 기준

| 실험 | 채택 기준 |
|------|---------|
| F0_clean | ≥ 0.0594 (B + 0.003) |
| F1_clean | ≥ F0_clean + 0.003 |
| F2_clean | ≥ F1_clean + 0.003 |

---

## Full Run 결과 요약 (2026-06-04 기준)

| 실험 | feature 수 | Stacking Platt PR-AUC (val_eval) |
|------|-----------|----------------------------------|
| F0_clean | 50 | 0.0781 |
| F1_clean | 72 | 0.0836 (+0.0055) |
| **F2_clean** | **94** | **0.1027 (+0.0191)** |

**F2_clean이 clean validation 기준 최고 모델 후보.**

각 feature group 기여:
- safe ACLED lag → +0.0217 (B 대비)
- GDELT title → +0.0055 (F0 대비)
- GDELT theme/person → +0.0191 (F1 대비)

---

## test set 평가 정책

> test set은 최종 feature/model 구조 확정 후 딱 한 번만 평가한다.  
> 현재 F2_clean이 validation 기준 최종 후보이며, feature importance 분석 후 test를 진행한다.
