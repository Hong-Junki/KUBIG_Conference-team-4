# 실험 F0: Safe ACLED Lag + B feature 실행 가이드

## 목적

B ACLED-free baseline (35개) 위에 **leakage-safe ACLED lag feature 15개**를 추가해  
ACLED 과거 정보의 기여도를 측정한다.

**채택 기준**: Stacking Platt PR-AUC ≥ **0.0594** (B baseline 0.0564 + 0.003)

---

## 사전 준비

아래 parquet가 반드시 존재해야 한다.

| 파일 | 생성 스크립트 | 비고 |
|------|--------------|------|
| `conflict-early-warning/input/processed/dataset/train.parquet` | 팀 pipeline | |
| `conflict-early-warning/input/processed/dataset/val.parquet` | 팀 pipeline | |
| `conflict-early-warning/input/processed/dataset/test.parquet` | 팀 pipeline | |
| `members/byeonghyeon/input/processed/acled_safe/safe_acled_lag_features.parquet` | `build_safe_acled_lag_features.py` | gitignore 대상 |

safe ACLED parquet가 없으면 먼저 생성:
```bash
python members/byeonghyeon/modeling/build_safe_acled_lag_features.py
```

---

## 실행 방법

### 전체 실행 (full run, ~30–60분)

```bash
cd <KUBIG_Conference-team-4 루트>
python members/byeonghyeon/modeling/run_stacking_safe_acled_f0.py
```

### Smoke test (~5분, 파이프라인 검증용)

```bash
SMOKE_TEST=1 python members/byeonghyeon/modeling/run_stacking_safe_acled_f0.py
```

F6 (2022 학습 → 2023 예측) 1 fold만, 라운드 축소.  
리포트 제목에 `⚠️ SMOKE TEST` 태그 붙어 full run 결과와 구분.

### DATA_ROOT 명시 실행

```bash
DATA_ROOT=/path/to/conflict-early-warning \
  python members/byeonghyeon/modeling/run_stacking_safe_acled_f0.py
```

---

## Feature set (총 50개)

| 카테고리 | 피처 수 | 설명 |
|----------|---------|------|
| B: GDELT events | 19 | goldstein/tone/mentions/event_count × {7d,14d,30d}, quadclass ratio |
| B: Economic | 15 | econ_{vix,wti,gold,dxy,stlfsi4} × {level,pct_1d,pct_7d} |
| B: Country | 1 | country |
| safe ACLED count/fatalities | 9 | {event_count,fatalities,fatalities_max} × {7d,14d,30d}_lag7 |
| safe ACLED event ratio | 3 | ratio_{battles,explosions,vac}_lag7 |
| safe ACLED actor ratio | 2 | ratio_{state_forces,external_forces}_lag7 |
| safe ACLED missing mask | 1 | safe_acled_missing_mask |
| **합계** | **50** | |

### 제외 항목

| 항목 | 이유 |
|------|------|
| `macis_se_score` | leakage 가능성 (train-only 생성 의심) |
| `acled_event_count_*`, `acled_fatalities_*` 등 기존 acled_* | lag 적용 여부 불확실 → 새 safe feature로 대체 |
| `acled_missing_mask` (구버전) | `safe_acled_missing_mask`로 대체 |
| `y`, `y_onset`, `y_escalation` | label columns |
| `event_count_next3d`, `fatalities_next3d` | future label |

### leakage-free 설계

```
t일 feature → shift(7) 후 rolling → 최대 t-7일 ACLED만 사용
label window → t+1 ~ t+3
gap = 8일 이상 ✅
```

---

## 출력

| 항목 | 경로 |
|------|------|
| 결과 리포트 | `members/byeonghyeon/outputs/reports/safe_acled_f0_results.md` |
| 예측 CSV | 저장 없음 |

---

## 리포트 내용

1. **채택 여부 판정** — F0 vs B(0.0564) vs C(0.0653) 3-way 비교
2. **val 지표 비교** — LightGBM / XGBoost / Stacking (raw/Platt/Isotonic)
3. **OOF fold 요약** — fold별 PR-AUC
4. **Meta LogReg 선택 C**
5. **test set 평가 정책** 명시

---

## 채택 기준 및 다음 단계

| 기준 | 값 |
|------|----|
| B baseline PR-AUC | 0.0564 |
| C baseline PR-AUC | 0.0653 |
| **F0 채택 기준** | **≥ 0.0594 (B + 0.003)** |

**F0 채택 시** → F1 (F0 + C title features) 진행 (72개 feature)  
**F0 미채택 시** → safe ACLED feature 재검토 또는 C 유지

자세한 실험 계획: `se_free_acled_lag_experiment_plan.md`
