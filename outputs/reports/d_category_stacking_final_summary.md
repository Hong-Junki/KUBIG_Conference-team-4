# D-Category 스태킹 앙상블 최종 요약 리포트

> **작성자**: byeonghyeon (Category D 담당)
> **최초 작성**: 2026-05-23 | **최종 업데이트**: 2026-05-24
> **대상 실험군**: `stacking_tree_only` 계열 (tree-only LightGBM + XGBoost) + LSTM 추가 절제 실험
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

## 3. 완료된 절제 실험 (6개)

| 실험명 | 변경 내용 | 스크립트 |
|--------|-----------|----------|
| `stacking_tree_only_12y` | 기준 (SE 포함, full train, mask 피처 미포함, 2014-start) | `run_stacking_d_prototype.py` |
| `stacking_tree_only_12y_no_se` | SE 피처 제거 | `run_stacking_d_no_se_ablation.py` |
| `stacking_tree_only_12y_with_mask_feature` | acled_missing_mask 피처 추가 | `run_stacking_d_with_mask_feature_ablation.py` |
| `stacking_tree_only_12y_mask0` | mask=1 행 학습에서 제외 | `run_stacking_d_mask0_ablation.py` |
| `stacking_tree_only_8y_with_mask_feature` | Train 시작일 2016으로 후행화 (2014-2015 제외) | `run_stacking_d_train2016_ablation.py` |
| `stacking_lgbm_xgb_lstm_12y_with_mask_feature` | Level 0에 LSTM Classifier 30d 추가 (기존 예측 파일 재사용) | `run_stacking_d_lgbm_xgb_lstm_ablation.py` |

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

5. **현재 D-category 최선 모델은 `stacking_tree_only_12y_with_mask_feature` + Platt** — val Stacking Platt PR-AUC = **0.2714**, P@5% = 0.2689, ECE = 0.0083.

6. **LSTM은 현재 버전에서 Level 0에서 제외** — 자체 제작 LSTM(PR-AUC 0.1030)을 추가한 결과 PR-AUC −0.0058 하락. 다양성은 있으나 품질이 낮아 잡음 효과가 우세. LSTM PR-AUC가 0.20 이상으로 개선되면 재편입 검토.

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
