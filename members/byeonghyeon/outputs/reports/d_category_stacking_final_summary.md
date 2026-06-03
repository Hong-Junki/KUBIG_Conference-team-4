# D-Category 스태킹 앙상블 최종 요약 리포트

> **작성자**: byeonghyeon (Category D 담당)
> **최초 작성**: 2026-05-23 | **최종 업데이트**: 2026-05-24
> **대상 실험군**: `stacking_tree_only` 계열 (tree-only LightGBM + XGBoost) + LSTM 추가 절제 실험 + 피처 그룹 기여도 절제 실험 v2 + 하이퍼파라미터 민감도 절제 실험
> **수치 출처**: 각 실험별 `val_metrics.json` 자동 추출 (수동 기입 없음)

---

## 1. Category D 역할 정의

`docs/model-study.md` Section 4 기준:

- **Level 0 (기본 예측기)**: LightGBM, XGBoost (LSTM 추가 실험 완료 → 현재 버전에서는 제외, 아래 Section 6 참고)
- **Level 1 (메타 러너)**: Logistic Regression (OOF 예측값으로 학습)
- **OOF 전략**: Expanding-window 6-fold (F1: 2014–2017→2018, …, F6: 2014–2022→2023)
- **캘리브레이션**: Platt Scaling (LogReg on raw stack probs) + Isotonic Regression
- **평가 지표**: PR-AUC (주지표), P@top5%, Brier Score, ECE

---

## 2. 구현 내용

### 공통 파이프라인

```
원시 데이터 로드
  → acled_missing_mask 계산
  → SE 피처 병합 (macis_se_score, null → 0)
  → Expanding-window OOF 생성 (Level 0)
  → OOF 스택 → Level 1 LogReg 학습 (val C 탐색: [0.01, 0.1, 1.0, 10.0])
  → Platt 캘리브레이션 → Isotonic 캘리브레이션
  → 검증/테스트 예측 저장
```

### SE 피처

- 출처: `output/macis_12y/se_scores.parquet` (LSTM 오토인코더 재구성 오차)
- 병합 키: `(date, country)`, null → 0 채움
- SE 포함 실험: 56개 피처 / SE 제외: 55개 피처 / mask feature 추가: 57개 피처

### ACLED 결측 마스크

- `acled_missing_mask`: ACLED 데이터 누락 여부 이진 컬럼
- 전체 학습 데이터(12y) 중 mask=1 비율: 약 13.8% (29,262행 / 211,816행)
- 2014~2015년 mask=1 비율: 40~49% (ACLED 커버리지 낮음)
- 2018년 이후 mask=1 비율: 0.84% (ACLED 거의 완전)
- 피처로 사용 시: 57개 피처 (`ALWAYS_EXCLUDE`에서 제외)
- mask0-only 학습 시: 학습셋 → 182,989행 (SE null rate 15.0% → 1.4%)

---

## 3. 완료된 절제 실험 (8개)

| 실험명 | 변경 내용 | 스크립트 |
|--------|-----------|----------|
| `stacking_tree_only_12y` | 기준 (SE 포함, full train, mask 피처 미포함, 2014-start) | `run_stacking_d_prototype.py` |
| `stacking_tree_only_12y_no_se` | SE 피처 제거 | `run_stacking_d_no_se_ablation.py` |
| `stacking_tree_only_12y_with_mask_feature` | acled_missing_mask 피처 추가 | `run_stacking_d_with_mask_feature_ablation.py` |
| `stacking_tree_only_12y_mask0` | mask=1 행 학습에서 제외 | `run_stacking_d_mask0_ablation.py` |
| `stacking_tree_only_8y_with_mask_feature` | Train 시작일 2016으로 후행화 (2014-2015 제외) | `run_stacking_d_train2016_ablation.py` |
| `stacking_lgbm_xgb_lstm_12y_with_mask_feature` | Level 0에 LSTM Classifier 30d 추가 (기존 예측 파일 재사용) | `run_stacking_d_lgbm_xgb_lstm_ablation.py` |
| `stacking_tree_only_12y_feature_group_ablation_v2` | 피처 그룹 기여도 분석 8종 — ACLED/GDELT/ECON/SE+mask 단독 및 제거 (country 포함 수정 버전; v1은 country 누락 버그로 폐기) | `run_stacking_d_feature_group_ablation.py` |
| `stacking_tree_only_12y_hyperparam_sensitivity_ablation` | 하이퍼파라미터 민감도 분석 9종 — num_leaves/scale_pos_weight/min_child_samples/max_depth를 한 번에 하나씩 변경 (one-factor-at-a-time) | `run_stacking_d_hyperparam_sensitivity_ablation.py` |

---

## 4. 절제 실험 전체 요약표 (Stacking Platt 기준)

> 수치는 각 실험의 `val_metrics.json`에서 자동 추출. delta 기준: 해당 실험 − 기준 실험(12y_wmf 또는 이전 단계).

| 구분 | 실험명 | SE | mask=1 행 | mask feature | Level 0 | Train 시작 | Platt PR-AUC | P@5% | ECE | 결론 |
|------|--------|----|-----------|--------------|---------|-----------|-------------|------|-----|------|
| 기준선 | `stacking_tree_only_12y` | ✓ | 유지 | ✗ | LGBM+XGB | 2014 | 0.2656 | 0.2614 | 0.0074 | 기준 |
| SE 절제 | `stacking_tree_only_12y_no_se` | ✗ | 유지 | ✗ | LGBM+XGB | 2014 | 0.1057 | 0.1591 | 0.0035 | SE 필수 (**−0.1599**) |
| mask feature 추가 | `stacking_tree_only_12y_with_mask_feature` | ✓ | 유지 | ✓ | LGBM+XGB | 2014 | **0.2714** | **0.2689** | 0.0083 | ★ **현재 최선** (+0.0059 vs 기준) |
| mask=0 only | `stacking_tree_only_12y_mask0` | ✓ | 제외 | ✗ | LGBM+XGB | 2014 | 0.2512 | 0.2614 | 0.0066 | mask=1 유지 (−0.0202 vs wmf) |
| 2016-start | `stacking_tree_only_8y_with_mask_feature` | ✓ | 유지 | ✓ | LGBM+XGB | **2016** | 0.2496 | 0.2576 | 0.0081 | 2014-start 유지 (−0.0218 vs wmf) |
| LSTM 추가 | `stacking_lgbm_xgb_lstm_12y_with_mask_feature` | ✓ | 유지 | ✓ | LGBM+XGB+**LSTM** | 2014 | 0.2656 | 0.2670 | 0.0067 | LSTM 제외 유지 (**−0.0058** vs wmf) |

**범례**: ✓ 포함 / ✗ 미포함 / wmf = with_mask_feature

### 피처 그룹 기여도 절제 v2 (57개 피처, country 포함)

> v1(country 누락 버그, full_best PR-AUC=0.2453)은 폐기. v2에서 country를 포함하여 재실행 (full_best PR-AUC=0.2697, 현재 최선 대비 −0.0017). delta 기준: 현재 최선 0.2714.

| 그룹 | 피처 수 | LGBM PR-AUC | XGB PR-AUC | Platt PR-AUC | P@5% | ECE | delta | 결론 |
|------|---------|-------------|------------|-------------|------|-----|-------|------|
| `full_best` | 57 | 0.2654 | 0.2598 | **0.2697** | 0.2614 | 0.0074 | −0.0017 | 현재 최선 재현 ✓ |
| `no_acled_features` | 37 | 0.1724 | 0.1219 | 0.1670 | 0.2008 | 0.0046 | **−0.1044** | ACLED 필수 |
| `no_gdelt_features` | 38 | 0.2545 | 0.2278 | 0.2486 | 0.2557 | 0.0060 | −0.0228 | GDELT 유용 |
| `no_economic_features` | 42 | 0.2687 | 0.2364 | 0.2552 | 0.2652 | 0.0076 | −0.0162 | ECON 기여 낮음 |
| `acled_only_plus_se_mask` | 22 | 0.2319 | 0.2270 | 0.2302 | 0.2273 | 0.0057 | −0.0412 | ACLED 단독은 약함 |
| `gdelt_only_plus_se_mask` | 21 | 0.1051 | 0.1073 | 0.1079 | 0.1364 | 0.0030 | −0.1635 | GDELT 단독 매우 약함 |
| `economic_only_plus_se_mask` | 17 | 0.0623 | 0.0592 | 0.0625 | 0.0814 | 0.0000 | −0.2089 | ECON 단독 거의 무의미 |
| `se_mask_only` | 2 | 0.0608 | 0.0628 | 0.0620 | 0.0814 | 0.0000 | −0.2094 | SE+mask 단독 불충분 |

어떤 그룹도 현재 최선(0.2714)을 초과하지 않음 → 현재 57개 피처 구성 유지.

### 하이퍼파라미터 민감도 절제 (9개 변형, one-factor-at-a-time)

> 베이스라인: LightGBM num_leaves=63 / scale_pos_weight=22 / min_child_samples=20, XGBoost max_depth=4 / scale_pos_weight=22. delta 기준: 현재 최선 0.2714.

| 변형 | 변경 내용 | Platt PR-AUC | P@5% | ECE | delta |
|------|----------|-------------|------|-----|-------|
| `lgbm_spw_sqrt` | LGBM scale_pos_weight=4.73 (√(neg/pos)) | **0.2551** | 0.2595 | 0.0063 | −0.0163 |
| `xgb_depth_5` | XGB max_depth=5 (더 깊게) | 0.2493 | 0.2500 | 0.0072 | −0.0221 |
| `xgb_spw_sqrt` | XGB scale_pos_weight=4.73 (√(neg/pos)) | 0.2466 | 0.2481 | 0.0040 | −0.0248 |
| `lgbm_spw_10` | LGBM scale_pos_weight=10 | 0.2457 | 0.2519 | 0.0064 | −0.0257 |
| `lgbm_min_child_50` | LGBM min_child_samples=50 | 0.2442 | 0.2462 | 0.0062 | −0.0272 |
| `lgbm_num_leaves_31` | LGBM num_leaves=31 (더 얕게) | 0.2413 | 0.2481 | 0.0062 | −0.0301 |
| `xgb_depth_3` | XGB max_depth=3 (더 얕게) | 0.2386 | 0.2443 | 0.0067 | −0.0328 |
| `lgbm_num_leaves_127` | LGBM num_leaves=127 (더 깊게) | 0.2374 | 0.2367 | 0.0056 | −0.0340 |
| `xgb_spw_10` | XGB scale_pos_weight=10 | 0.2335 | 0.2424 | 0.0049 | −0.0379 |

어떤 변형도 현재 최선(0.2714)을 초과하지 않음 → 현재 하이퍼파라미터 유지. 테스트 예측 파일 저장 없음.

---

## 5. 최종 선정 D-Category 출력

**선정 모델**: `stacking_tree_only_12y_with_mask_feature` + Platt 캘리브레이션

| 항목 | 값 |
|------|-----|
| PR-AUC (val) | **0.2714** |
| P@top5% (val) | 0.2689 |
| Brier Score (val) | 0.0359 |
| ECE (val) | 0.0083 |
| 학습 전략 | full train (mask=1 포함), mask 피처 포함, SE 피처 포함 |
| Train 윈도우 | 2014~2023 (12년) |
| 캘리브레이션 | Platt (LogReg) |
| 메타 LogReg C | 0.01 |
| 피처 수 | 57개 |

**테스트 예측 파일** (팀 제출용):
```
outputs/predictions/predictions__stacking_tree_only_12y_with_mask_feature_platt__D_byeonghyeon.csv
```

**검증 예측 파일**:
```
outputs/predictions/val_predictions__stacking_tree_only_12y_with_mask_feature__D_byeonghyeon.csv
```

---

## 6. 절제 실험 해석

### SE 피처: 절대적으로 중요

- SE 제거 시 Stacking Platt PR-AUC: 0.2656 → 0.1057 (**−0.1599 ↓**)
- P@top5%도 0.2614 → 0.1591 (**−0.1023 ↓**)
- `macis_se_score`는 LSTM 오토인코더 재구성 오차로, 분쟁 이상 신호를 직접 포착
- **결론: SE 피처 반드시 유지**

### acled_missing_mask 피처 추가: 유의미한 개선

- Stacking Platt PR-AUC: 0.2656 → 0.2714 (**+0.0059 ↑**, 임계값 0.005 초과)
- XGBoost PR-AUC도 +0.0097 개선
- mask 피처가 메타 러너에게 데이터 신뢰도 정보를 제공하여 예측 품질 향상
- **결론: acled_missing_mask 피처 포함 권장**

### mask=1 행 제거: 오히려 성능 저하

- mask0-only 학습 시 Stacking Platt PR-AUC: 0.2714 → 0.2512 (**−0.0202 ↓**)
- 학습 데이터 29,262행 손실이 앙상블 다양성에 악영향
- mask0에서 메타 LogReg가 C=10.0을 선택 (다른 실험은 모두 C=0.01) — OOF 분포 변화 반영
- **결론: mask=1 행 유지 (제거하지 말 것)**

### Train 시작일 2016-start: 성능 저하

- 2016-start(8y) Stacking Platt PR-AUC: 0.2714 → 0.2496 (**−0.0218 ↓**)
- XGBoost 하락 폭이 가장 큼: 0.2631 → 0.2342 (−0.0289)
- LightGBM은 거의 동일(−0.0002)하나, 앙상블 전체 성능이 저하
- 메타 LogReg C 탐색 시 모든 C 후보(0.01~10.0)에서 val PR-AUC = 0.2496으로 동일 → OOF 분포가 메타 가중치 학습에 불충분
- 2014~2015년 ACLED mask_rate(40~49%)가 높지만, `acled_missing_mask` 피처가 이미 데이터 부재를 모델에게 알려주므로 굳이 제거하지 않아도 됨
- **결론: 2014-start(12y) 유지. 2016-start로의 전환 불필요**

### LSTM Level 0 추가: 성능 저하로 제외

- LSTM 단독 val PR-AUC = 0.1030 — 트리 기반 모델(0.26~0.27) 대비 현저히 낮음
- LGBM-LSTM 확률 상관관계: **0.487**, XGB-LSTM: **0.519** → 다양성은 존재
- 그러나 LSTM 추가 시 Stacking Platt PR-AUC: 0.2714 → 0.2656 (**−0.0058 ↓**)
- P@5%도 0.2689 → 0.2670 (**−0.0019 ↓**) — ECE만 0.0083 → 0.0067로 소폭 개선
- Top-5% 알람 집합 Jaccard: tree-only vs +LSTM stack = **0.899** → 최종 출력 거의 동일
- 메타 러너 LSTM 계수: 0.69 (LGBM: 2.09, XGB: 2.21) — LSTM 낮은 신뢰도 반영
- **결론: 현재 LSTM 품질로는 스태킹 편입 시 잡음 효과가 다양성 효과를 압도. LSTM 제외 유지.**

### 피처 그룹 기여도 절제 v2: ACLED이 핵심 소스

- **v1 버그**: `ALWAYS_EXCLUDE`에 `country`가 잘못 포함되어 full_best PR-AUC = 0.2453 (실제보다 −0.0244 저평가). 56개 피처로 실행되었으며 결과 폐기.
- **v2 수정**: country를 피처로 복원(57개), LightGBM categorical + XGBoost label-encoded 처리. full_best PR-AUC = **0.2697** (현재 최선 0.2714 대비 −0.0017, 정상 변동 범위).
- **ACLED 피처 그룹 절대적 중요**: ACLED 20개 피처 제거 시 PR-AUC 0.2697 → 0.1670 (**−0.103 ↓**). 단독으로도 0.2302 수준 가능.
- **GDELT 피처 이차적 기여**: GDELT 제거 시 0.2697 → 0.2486 (−0.021 ↓). GDELT 단독은 0.1079로 매우 낮음.
- **ECON 피처 기여 낮음**: ECON 제거 시 0.2697 → 0.2552 (−0.015 ↓). 단독으로는 0.0625 수준 — 사실상 분쟁 예측력 없음.
- **SE+mask 단독 불충분**: 2개 피처만으로는 0.0620 — 베이스라인 수준.
- **현재 최선 구성 유지**: 어떤 그룹도 0.2714 + 0.003 임계값(0.2744)을 초과하지 않아 테스트 예측 저장 없음.
- **결론: ACLED 피처 그룹이 D-category 예측의 핵심 소스. 현재 57개 피처 구성(country 포함) 유지.**

### 하이퍼파라미터 민감도 절제: 현재 설정이 로컬 최적점

- **9개 변형** 테스트 (one-factor-at-a-time): num_leaves 31/127, scale_pos_weight sqrt/10 (LGBM·XGB 각각), min_child_samples 50, max_depth 3/5.
- **최고 변형**: `lgbm_spw_sqrt` (LightGBM scale_pos_weight=4.73) — Platt PR-AUC=0.2551 (delta −0.0163). 기준 22 대비 ECE는 소폭 개선(0.0083→0.0063)하나 PR-AUC는 유의미하게 하락.
- **num_leaves 민감도**: num_leaves=63(현재) > 31 (−0.030) > 127 (−0.034). 더 얕거나 더 깊은 트리 모두 성능 저하.
- **max_depth 민감도**: depth=4(현재) > 5 (−0.022) > 3 (−0.033). depth=5가 3보다 낫지만 현재 설정이 최적.
- **scale_pos_weight 민감도**: spw=22(현재)가 spw=sqrt(4.73)이나 spw=10보다 PR-AUC에서 우월. spw 감소 시 ECE 개선(0.003–0.006)이 관찰되나 PR-AUC 희생이 동반.
- **min_child_samples**: 50으로 증가 시 PR-AUC −0.027. 정규화 강화가 이 데이터셋에서는 불리.
- **어떤 변형도 0.2714+0.003=0.2744를 초과하지 않음** → 테스트 예측 저장 없음, 현재 하이퍼파라미터 유지.
- **결론: 현재 LightGBM(num_leaves=63, spw=22, min_child=20) + XGBoost(depth=4, spw=22) 설정이 탐색 공간 내 로컬 최적점. 추가 하이퍼파라미터 튜닝 불필요.**

### Platt vs Isotonic 캘리브레이션

- Isotonic ECE≈0 : 검증셋에서 과적합 (검증셋 31개 고유값 → 완벽 보정처럼 보임)
- Platt ECE=0.0083 : 일반화 가능한 부드러운 보정
- Brier Score는 Platt(0.0359)와 Isotonic(0.0337/0.0339)이 유사하나, ECE 관점에서 Platt이 안전
- **결론: Platt 캘리브레이션 채택 (Isotonic은 val 과적합 위험)**

---

## 7. 팀 공유용 요약 결론

1. **SE는 반드시 유지해야 함** — 제거 시 PR-AUC −0.16 수준의 치명적 하락. `output/macis_12y/se_scores.parquet` 경로 고정 필수.

2. **acled_missing_mask는 피처로 포함하는 것이 가장 좋음** — PR-AUC +0.006 개선. 데이터 신뢰도 정보를 메타 러너에 전달.

3. **mask=1 행은 제거하지 않는 것이 좋음** — 제거 시 학습 데이터 29K행 손실로 PR-AUC −0.020 하락. acled_missing_mask 피처가 이미 부재 신호를 처리.

4. **2014-2015를 제거한 2016-start는 성능이 하락했으므로 2014-start 유지** — 8y 실험 PR-AUC −0.022. 2014-2015 데이터가 모델에 유용한 패턴 정보를 제공함.

5. **현재 D-category 최종 확정 모델: `stacking_tree_only_12y_with_mask_feature` + Platt** — val Stacking Platt PR-AUC = **0.2714**, P@5% = 0.2689, ECE = 0.0083. 모든 절제 실험(8종) 완료 후 어떤 변형도 이를 초과하지 못함.

6. **LSTM은 현재 버전에서 Level 0에서 제외** — 자체 제작 LSTM(PR-AUC 0.1030)을 추가한 결과 PR-AUC −0.0058 하락. 다양성은 있으나 품질이 낮아 잡음 효과가 우세. LSTM PR-AUC가 0.20 이상으로 개선되면 재편입 검토.

7. **ACLED 피처 그룹이 D-category 예측의 핵심 입력 소스** — 피처 그룹 절제 v2 결과: ACLED 20개 피처 제거 시 PR-AUC 0.2697 → 0.1670 (−0.103 ↓). GDELT는 이차적 기여(−0.022), ECON은 기여 낮음(−0.016). 현재 57개 피처(country 포함) 구성 유지 권장.

8. **하이퍼파라미터는 현재 설정 유지** — 9개 one-factor-at-a-time 변형 테스트 결과: LightGBM num_leaves=63, scale_pos_weight=22, min_child_samples=20 / XGBoost max_depth=4, scale_pos_weight=22가 탐색 공간 내 로컬 최적점임을 확인. 어떤 단일 변형도 현재 최선(0.2714)을 초과하지 못함. D-category 하이퍼파라미터 튜닝 추가 불필요.

---

## 8. 한계 및 주의사항

1. **LSTM 실험 완료 → 현재 버전 제외**: 자체 제작 LSTM Classifier 30d를 Level 0에 추가한 결과 PR-AUC −0.0058 하락. 현재 구현(hidden_size=64, 10 epochs)으로는 메타 러너 편입 효과 없음. LSTM 성능(PR-AUC 0.20+) 개선 후 재편입 검토.

2. **검증 성능 낙관 편향**: val 데이터를 메타 C 탐색 + Platt 학습 + 성능 평가에 모두 사용하므로 val 지표가 실제 일반화 성능보다 높을 수 있음.

3. **Isotonic 과적합**: val 셋 크기가 작아 Isotonic이 완벽 보정처럼 보이나 실제 일반화 미보장. 리포트에는 참고용으로만 기재.

4. **OOF 불균형**: mask=0 행이 많은 후반 fold는 mask=1 행이 포함된 현실적 분포와 다를 수 있음 (mask 피처로 보정 중).

5. **SE 경로 의존성**: `output/macis_12y/se_scores.parquet` 부재 시 no-SE 수준으로 성능 급락. 파일 경로 고정 필요.

6. **2018-start 미실행**: 2016-start가 이미 열위를 보였으므로 2018-start는 더 낮은 성능이 예상됨. 우선순위 낮음.

---

## 9. 권고 다음 단계

1. **[우선순위 1] 대시보드 업데이트**: 최종 모델(`stacking_tree_only_12y_with_mask_feature` + Platt)이 확정되었으므로 `dashboard/` 업데이트 검토 가능.

2. **[우선순위 2] LSTM 성능 개선 후 재편입 검토**: hidden_size 128~256, seq_len 60~90일, LR 스케줄, 더 많은 에포크 등으로 LSTM PR-AUC 0.20+ 달성 시 Level 0 재추가 가능.

3. **[우선순위 3] Lookahead 2d 절제**: 팀 협의 후 결정. `conflict_indicator_2d_ahead` 타깃 사용 시 실용성 향상 여부 검토.

4. **[참고] 최종 제출 파일 경로**:
   - Test: `outputs/predictions/predictions__stacking_tree_only_12y_with_mask_feature_platt__D_byeonghyeon.csv`
   - Val: `outputs/predictions/val_predictions__stacking_tree_only_12y_with_mask_feature__D_byeonghyeon.csv`

---

## 절제 실험 관련 상세 리포트

| 리포트 파일 | 내용 |
|------------|------|
| `outputs/reports/stacking_tree_only_12y_se_ablation_comparison.md` | SE vs no-SE 비교 |
| `outputs/reports/stacking_tree_only_12y_mask_feature_ablation_comparison.md` | mask 피처 추가 vs 미추가 비교 |
| `outputs/reports/stacking_tree_only_12y_mask0_ablation_comparison.md` | mask0-only vs full-train 3-way 비교 |
| `outputs/reports/stacking_tree_only_12y_train_start_ablation_comparison.md` | Train 시작일 12y vs 8y 비교 |
| `outputs/reports/stacking_lgbm_xgb_lstm_12y_with_mask_feature_val_metrics.md` | LGBM+XGB+LSTM 스태킹 검증 지표 |
| `outputs/reports/stacking_lgbm_xgb_lstm_vs_tree_only_comparison.md` | tree-only vs +LSTM 스태킹 비교 |
| `outputs/reports/stacking_tree_only_12y_feature_group_ablation_v2.csv` | 피처 그룹 기여도 절제 v2 — 8그룹 전체 지표 CSV |
| `outputs/reports/stacking_tree_only_12y_feature_group_ablation_v2.md` | 피처 그룹 기여도 절제 v2 — 상세 리포트 (v1 버그 수정본) |
| `outputs/reports/stacking_tree_only_12y_hyperparam_sensitivity_ablation.csv` | 하이퍼파라미터 민감도 절제 — 9개 변형 전체 지표 CSV |
| `outputs/reports/stacking_tree_only_12y_hyperparam_sensitivity_ablation.md` | 하이퍼파라미터 민감도 절제 — 상세 리포트 (한국어 해석 포함) |
