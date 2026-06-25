# 검증 지표 — stacking_tree_only_12y_with_mask_feature

**실험**: stacking_tree_only_12y_with_mask_feature
**정책**: full  (acled_missing_mask=1 포함, 피처로도 포함)
**acled_missing_mask 피처**: 포함 (ablation 변수)
**SE 피처**: 포함 (macis_se_score)
**Base 모델**: LightGBM + SE + mask, XGBoost + SE + mask
**Meta 모델**: Logistic Regression (선택된 C = 0.01)
**비교 기준**: 단일 모델 LightGBM+SE mask0_only val PR-AUC = 0.1741

> ⚠ Final base 모델은 val을 early stopping에 사용 → val 지표가 소폭 낙관적.
> mask 피처 기여도 측정을 위한 절제 실험. 전체 비교: mask_feature_ablation_comparison 리포트 참조.

## 지표 비교

| 모델                               | PR-AUC | P@5%   | R@P≥.10 | R@P≥.20 | R@P≥.30 | Brier  | ECE    |
|------------------------------------|--------|--------|---------|---------|---------|--------|--------|
| LightGBM + SE + mask              | 0.2606 | 0.2614 | 0.7163 | 0.4279 | 0.2860 | 0.1462 | 0.2839 |
| XGBoost + SE + mask               | 0.2631 | 0.2576 | 0.7349 | 0.3814 | 0.2884 | 0.1255 | 0.2594 |
| Stacking (raw)                    | 0.2714 | 0.2689 | 0.7512 | 0.3791 | 0.3000 | 0.3049 | 0.4989 |
| Stacking (Platt)                  | 0.2714 | 0.2689 | 0.7512 | 0.3791 | 0.3000 | 0.0359 | 0.0083 |
| Stacking (Isotonic)               | 0.2617 | 0.2727 | 0.6977 | 0.3442 | 0.3000 | 0.0337 | 0.0000 |

## 기준 대비 평가

- 단일 모델 기준 PR-AUC: **0.1741**
- Stacking (Isotonic) PR-AUC: **0.2617**
- 결과: **✓ 개선됨**
