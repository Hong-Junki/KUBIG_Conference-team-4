# 실험 C2: ACLED-free + GDELT Title + C2 Derived Features 실행 가이드

## 목적

C feature set (B 35개 + GDELT title 21개 + coverage_mask 1개 = 57개) 위에  
C2 파생 피처 **19개**를 추가해 분류 성능 개선을 검증한다.

C2 피처는 기존 C parquet를 재사용해 로컬에서 생성 (**BigQuery 추가 비용 없음**).

**채택 기준**: Stacking Platt PR-AUC ≥ **0.0683** (C baseline 0.0653 + 0.003)

---

## 사전 준비

아래 두 parquet가 반드시 존재해야 한다. 없으면 각 스크립트를 먼저 실행한다.

| 파일 | 생성 스크립트 |
|------|--------------|
| `members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_features.parquet` | `build_gdelt_title_features.py` |
| `members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_c2_features.parquet` | `build_gdelt_title_c2_features.py` |

두 파일 모두 `.gitignore`로 추적되지 않음 (로컬 전용).

---

## 실행 방법

### 전체 실행 (full run, ~30–60분)

```bash
cd <KUBIG_Conference-team-4 루트>
python members/byeonghyeon/modeling/run_stacking_acled_free_with_titles_c2.py
```

### Smoke test (~5분, 결과 검증용)

```bash
SMOKE_TEST=1 python members/byeonghyeon/modeling/run_stacking_acled_free_with_titles_c2.py
```

Smoke test는 OOF를 F6 (2022 학습 → 2023 예측) **1 fold**만 실행하고,  
LGB/XGB rounds를 대폭 축소해 빠르게 파이프라인 전체를 검증한다.  
Smoke test 결과는 리포트 제목에 `⚠️ SMOKE TEST` 태그가 붙어 full run과 구분된다.

### DATA_ROOT 명시 실행

```bash
DATA_ROOT=/path/to/conflict-early-warning \
  python members/byeonghyeon/modeling/run_stacking_acled_free_with_titles_c2.py
```

---

## Feature set (총 76개)

| 카테고리 | 피처 수 | 피처 목록 (요약) |
|----------|---------|------------------|
| B: GDELT events | 19 | goldstein/tone/mentions/event_count × {7d,14d,30d}, quadclass ratio |
| B: Economic | 15 | econ_{vix,wti,gold,dxy,stlfsi4} × {level,pct_1d,pct_7d} |
| B: Country | 1 | country (LightGBM: category, XGBoost: int) |
| C: GDELT title 1d | 10 | count/nonnull_count/tone_mean/tone_std/tone_min/negative/positive/eng/domain/lang |
| C: GDELT title 7d | 11 | 위 10개 + tone_trend_7d |
| C: coverage_mask | 1 | 2015-02-17 이전 = 1 |
| C2: 3일 rolling | 9 | count/nonnull/negative/positive/eng/tone_mean/tone_min/domain/lang _3d |
| C2: Spike | 5 | count/negative 3d_vs_7d, 3d_vs_14d, tone_drop_3d_vs_7d |
| C2: Country-norm | 5 | count/negative/tone z-score/ratio (90d rolling) |
| **합계** | **76** | |

---

## 출력

| 항목 | 경로 |
|------|------|
| 결과 리포트 | `members/byeonghyeon/outputs/reports/acled_free_with_titles_c2_results.md` |
| 예측 CSV | 저장 없음 |

---

## 리포트 내용

리포트(`acled_free_with_titles_c2_results.md`)에는 다음이 포함된다:

1. **채택 여부 판정표** — B / C / C2 PR-AUC 3-way 비교, C2-C delta, C2-B delta
2. **val 지표 비교** — LightGBM, XGBoost, Stacking (raw/Platt/Isotonic) × B/C/C2
3. **OOF fold 요약** — fold별 n_train, n_pred, 양성률, LGBM/XGB PR-AUC
4. **Meta LogReg** — 선택된 C 값
5. **해석 주의사항** — val 낙관적 추정 경고, test 미평가 안내

Smoke test 결과는 제목과 OOF 섹션에 `⚠️ SMOKE TEST` 태그가 붙어  
full run 결과와 시각적으로 구분된다.

---

## 채택 기준 요약

| 기준 | 값 |
|------|----|
| B baseline PR-AUC | 0.0564 |
| C baseline PR-AUC | 0.0653 |
| **C2 채택 기준** | **≥ 0.0683 (C + 0.003)** |

C2 채택 시 → 실험 D (themes/persons 피처 추가)로 진행  
C2 미채택 시 → C feature set 유지, 다른 파생 방향 검토
