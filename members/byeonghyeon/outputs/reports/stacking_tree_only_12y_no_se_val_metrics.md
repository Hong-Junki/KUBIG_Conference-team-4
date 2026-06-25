# 검증 지표 — stacking_tree_only_12y_no_se

**실험**: stacking_tree_only_12y_no_se
**정책**: full  (acled_missing_mask=1 포함, 피처에서 제외)
**SE 피처**: 미포함 (SE ablation)
**Base 모델**: LightGBM (no SE), XGBoost (no SE)
**Meta 모델**: Logistic Regression (선택된 C = 0.01)
**비교 기준**: 단일 모델 LightGBM+SE mask0_only val PR-AUC = 0.1741

> ⚠ Final base 모델은 val을 early stopping에 사용 → val 지표가 소폭 낙관적.
> SE 기여도 측정을 위한 절제 실험. 전체 비교: se_ablation_comparison 리포트 참조.

## 지표 비교

| 모델                           | PR-AUC | P@5%   | R@P≥.10 | R@P≥.20 | R@P≥.30 | Brier  | ECE    |
|--------------------------------|--------|--------|---------|---------|---------|--------|--------|
| LightGBM (no SE)              | 0.1067 | 0.1629 | 0.4000 | 0.0674 | 0.0163 | 0.1274 | 0.2779 |
| XGBoost (no SE)               | 0.1041 | 0.1458 | 0.4419 | 0.0442 | 0.0070 | 0.2037 | 0.3911 |
| Stacking (raw)                | 0.1057 | 0.1591 | 0.4581 | 0.0419 | 0.0070 | 0.3553 | 0.5599 |
| Stacking (Platt)              | 0.1057 | 0.1591 | 0.4581 | 0.0419 | 0.0070 | 0.0381 | 0.0035 |
| Stacking (Isotonic)           | 0.1035 | 0.1648 | 0.4558 | 0.0256 | 0.0070 | 0.0377 | 0.0000 |

## 기준 대비 평가

- 단일 모델 기준 PR-AUC: **0.1741**
- Stacking (Isotonic) PR-AUC: **0.1035**
- 결과: **✗ 미개선**
