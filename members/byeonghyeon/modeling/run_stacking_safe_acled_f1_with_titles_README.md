# 실험 F1: Safe ACLED Lag + B + GDELT Title 실행 가이드

## 목적

F0 (B 35 + safe ACLED 15 = 50개, PR-AUC=0.0996) 위에  
**GDELT title C feature 21개 + coverage_mask 1개**를 추가해  
title 정보의 추가 기여도를 측정한다.

**채택 기준**: Stacking Platt PR-AUC ≥ **0.1026** (F0 baseline 0.0996 + 0.003)

---

## 사전 준비

| 파일 | 생성 스크립트 | 비고 |
|------|--------------|------|
| `conflict-early-warning/input/processed/dataset/train.parquet` | 팀 pipeline | |
| `conflict-early-warning/input/processed/dataset/val.parquet` | 팀 pipeline | |
| `conflict-early-warning/input/processed/dataset/test.parquet` | 팀 pipeline | |
| `members/byeonghyeon/input/processed/acled_safe/safe_acled_lag_features.parquet` | `build_safe_acled_lag_features.py` | gitignore 대상 |
| `members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_features.parquet` | `build_gdelt_title_features.py` | gitignore 대상 |

---

## 실행 방법

### 전체 실행 (full run, ~30–60분)

```bash
cd <KUBIG_Conference-team-4 루트>
python members/byeonghyeon/modeling/run_stacking_safe_acled_f1_with_titles.py
```

### Smoke test (~5분)

```bash
SMOKE_TEST=1 python members/byeonghyeon/modeling/run_stacking_safe_acled_f1_with_titles.py
```

---

## Feature set (총 72개)

| 카테고리 | 피처 수 |
|----------|---------|
| B: GDELT events | 19 |
| B: Economic | 15 |
| B: Country | 1 |
| safe ACLED lag (F0) | 15 |
| GDELT title 1d/7d (C) | 21 |
| coverage_mask | 1 |
| **합계** | **72** |

---

## 출력

| 항목 | 경로 |
|------|------|
| 결과 리포트 | `members/byeonghyeon/outputs/reports/safe_acled_f1_with_titles_results.md` |
| 예측 CSV | 저장 없음 |

---

## 채택 기준 및 다음 단계

| 기준 | 값 |
|------|----|
| B baseline | 0.0564 |
| C baseline (ACLED-free+title) | 0.0653 |
| F0 PR-AUC (B+safe ACLED) | 0.0996 |
| **F1 채택 기준** | **≥ 0.1026 (F0 + 0.003)** |

**F1 채택 시** → F2 (F1 + D theme/person 22개 = 94개) 진행 또는 F1을 최종 모델로 확정  
**F1 미채택 시** → F0 유지, title 기여도 없음으로 판단
