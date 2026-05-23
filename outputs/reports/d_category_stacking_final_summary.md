# D-Category 스태킹 앙상블 최종 요약 리포트

> **작성자**: byeonghyeon (Category D 담당)
> **작성일**: 2026-05-23
> **대상 실험군**: `stacking_tree_only_12y` 계열 (tree-only, 12년 데이터)

---

## 1. Category D 역할 정의

`docs/model-study.md` Section 4 기준:

- **Level 0 (기본 예측기)**: LightGBM, XGBoost, (추후 LSTM 추가 예정)
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
- SE 포함 실험: 56개 피처 / SE 제외: 55개 피처

### ACLED 결측 마스크

- `acled_missing_mask`: ACLED 데이터 누락 여부 이진 컬럼
- 전체 학습 데이터 중 mask=1 비율: 약 13.8% (29,262행 / 212,282행)
- 피처로 사용 시: 57개 피처 (`ALWAYS_EXCLUDE`에서 제외)
- mask0-only 학습 시: 학습셋 → 182,989행 (SE null rate 15.0% → 1.4%)

---

## 3. 완료된 절제 실험

| 실험명 | 변경 내용 | 스크립트 |
|--------|-----------|----------|
| `stacking_tree_only_12y` | 기준 (SE 포함, full train, mask 피처 미포함) | `run_stacking_d_prototype.py` |
| `stacking_tree_only_12y_no_se` | SE 피처 제거 | `run_stacking_d_no_se_ablation.py` |
| `stacking_tree_only_12y_with_mask_feature` | acled_missing_mask 피처 추가 | `run_stacking_d_with_mask_feature_ablation.py` |
| `stacking_tree_only_12y_mask0` | mask=1 행 학습에서 제외 | `run_stacking_d_mask0_ablation.py` |

---

## 4. 주요 결과 비교표

### PR-AUC (높을수록 좋음)

| 모델 | A: 기준 | no-SE | B: +mask feat | C: mask0 |
|------|---------|-------|---------------|---------|
| LightGBM | 0.2596 | 0.1067 | 0.2606 | 0.2542 |
| XGBoost | 0.2534 | 0.1041 | 0.2631 | 0.2341 |
| Stacking (raw) | 0.2656 | 0.1057 | 0.2714 | 0.2512 |
| **Stacking (Platt)** | **0.2656** | 0.1057 | **0.2714** | 0.2512 |
| Stacking (Isotonic) | 0.2535 | 0.1035 | 0.2617 | 0.2382 |

### P@top5% (높을수록 좋음)

| 모델 | A: 기준 | no-SE | B: +mask feat | C: mask0 |
|------|---------|-------|---------------|---------|
| LightGBM | 0.2708 | 0.1629 | 0.2614 | 0.2746 |
| XGBoost | 0.2500 | 0.1458 | 0.2576 | 0.2424 |
| Stacking (raw) | 0.2614 | 0.1591 | 0.2689 | 0.2614 |
| **Stacking (Platt)** | **0.2614** | 0.1591 | **0.2689** | 0.2614 |
| Stacking (Isotonic) | 0.2746 | 0.1648 | 0.2727 | 0.2708 |

### Brier Score (낮을수록 좋음)

| 모델 | A: 기준 | no-SE | B: +mask feat | C: mask0 |
|------|---------|-------|---------------|---------|
| LightGBM | 0.1443 | 0.1274 | 0.1462 | 0.1422 |
| XGBoost | 0.1314 | 0.2037 | 0.1255 | 0.1489 |
| Stacking (raw) | 0.3135 | 0.3553 | 0.3049 | 0.3341 |
| **Stacking (Platt)** | **0.0359** | 0.0381 | **0.0359** | 0.0362 |
| Stacking (Isotonic) | 0.0339 | 0.0377 | 0.0337 | 0.0342 |

### ECE (낮을수록 좋음)

| 모델 | A: 기준 | no-SE | B: +mask feat | C: mask0 |
|------|---------|-------|---------------|---------|
| LightGBM | 0.2796 | 0.2779 | 0.2839 | 0.2773 |
| XGBoost | 0.2663 | 0.3911 | 0.2594 | 0.2986 |
| Stacking (raw) | 0.5071 | 0.5599 | 0.4989 | 0.5245 |
| **Stacking (Platt)** | **0.0074** | 0.0035 | **0.0083** | 0.0066 |
| Stacking (Isotonic) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

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
| 캘리브레이션 | Platt (LogReg) |
| 메타 LogReg C | 0.01 |
| 피처 수 | 57개 |

**검증 예측 파일**:
```
outputs/predictions/predictions__stacking_tree_only_12y_with_mask_feature_platt__D_byeonghyeon.csv
```

**테스트 예측 파일**:
```
outputs/predictions/test_predictions__stacking_tree_only_12y_with_mask_feature_platt__D_byeonghyeon.csv
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

- mask0-only 학습 시 Stacking Platt PR-AUC: 0.2714 → 0.2512 (**−0.0203 ↓** vs 실험 B)
- 학습 데이터 29,262행 손실이 앙상블 다양성에 악영향
- mask0에서 메타 LogReg가 C=10.0을 선택 (다른 실험은 모두 C=0.01) — OOF 분포 변화 반영
- **결론: mask=1 행 유지 (제거하지 말 것)**

### Platt vs Isotonic 캘리브레이션

- Isotonic ECE≈0 : 검증셋에서 과적합 (검증셋 31개 고유값 → 완벽 보정처럼 보임)
- Platt ECE=0.0083 : 일반화 가능한 부드러운 보정
- Brier Score는 Platt(0.0359)와 Isotonic(0.0337/0.0339)이 유사하나, ECE 관점에서 Platt이 안전
- **결론: Platt 캘리브레이션 채택 (Isotonic은 val 과적합 위험)**

---

## 7. 한계 및 주의사항

1. **LSTM 미포함**: Level 0에 LSTM이 없어 앙상블 다양성 미달. C-category 팀원의 OOF/val/test 예측 파일 수령 후 `BASE_MODELS`에 추가 예정.

2. **검증 성능 낙관 편향**: val 데이터를 메타 C 탐색 + Platt 학습 + 성능 평가에 모두 사용하므로 val 지표가 실제 일반화 성능보다 높을 수 있음.

3. **Isotonic 과적합**: val 셋 크기가 작아 Isotonic이 완벽 보정처럼 보이나 실제 일반화 미보장. 리포트에는 참고용으로만 기재.

4. **OOF 불균형**: mask=0 행이 많은 후반 fold는 mask=1 행이 포함된 현실적 분포와 다를 수 있음 (mask 피처로 보정 중).

5. **SE 경로 의존성**: `output/macis_12y/se_scores.parquet` 부재 시 no-SE 수준으로 성능 급락. 파일 경로 고정 필요.

---

## 8. 권고 다음 단계

1. **[우선순위 1] C-category LSTM OOF 수령**: `outputs/oof/oof__lstm_*__C_*.csv` 형식으로 수령 시 `BASE_MODELS`에 추가 → 앙상블 다양성 확대 후 재실험.

2. **[우선순위 2] Lookahead 2d 절제**: 팀 협의 후 결정. `conflict_indicator_2d_ahead` 타깃 사용 시 실용성 향상 여부 검토.

3. **[우선순위 3] 대시보드 업데이트**: 최종 모델 결정 후 `dashboard/` 업데이트. 현재 PR-AUC 0.2714가 D 최선이나, LSTM 추가 후 재평가 필요.

4. **[참고] 제출 파일 경로**:
   - Val: `outputs/predictions/predictions__stacking_tree_only_12y_with_mask_feature_platt__D_byeonghyeon.csv`
   - Test: `outputs/predictions/test_predictions__stacking_tree_only_12y_with_mask_feature_platt__D_byeonghyeon.csv`

---

## 절제 실험 관련 상세 리포트

| 리포트 파일 | 내용 |
|------------|------|
| `outputs/reports/stacking_tree_only_12y_se_ablation_comparison.md` | SE vs no-SE 비교 |
| `outputs/reports/stacking_tree_only_12y_mask_feature_ablation_comparison.md` | mask 피처 추가 vs 미추가 비교 |
| `outputs/reports/stacking_tree_only_12y_mask0_ablation_comparison.md` | mask0-only vs full-train 3-way 비교 |
