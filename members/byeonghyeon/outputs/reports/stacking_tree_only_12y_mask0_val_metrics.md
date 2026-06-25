# 검증 지표 — stacking_tree_only_12y_mask0

**실험**: stacking_tree_only_12y_mask0
**정책**: mask0_only  (acled_missing_mask=1 행 제거, 피처에서도 제외)
**acled_missing_mask 피처**: 미포함
**SE 피처**: 포함 (macis_se_score)
**Base 모델**: LightGBM + SE (mask0), XGBoost + SE (mask0)
**Meta 모델**: Logistic Regression (선택된 C = 10.0)
**비교 기준**: 단일 모델 LightGBM+SE mask0_only val PR-AUC = 0.1741

> ⚠ Final base 모델은 val을 early stopping에 사용 → val 지표가 소폭 낙관적.
> mask=1 행 제거 효과 측정. 3-way 비교: mask0_ablation_comparison 리포트 참조.

## 지표 비교

| 모델                               | PR-AUC | P@5%   | P@10%  | R@P≥.10 | R@P≥.20 | R@P≥.30 | Brier  | ECE    |
|------------------------------------|--------|--------|--------|---------|---------|---------|--------|--------|
| LightGBM + SE (mask0)             | 0.2542 | 0.2746 | 0.1856 | 0.7116 | 0.4256 | 0.2907 | 0.1422 | 0.2773 |
| XGBoost + SE (mask0)              | 0.2341 | 0.2424 | 0.1705 | 0.6907 | 0.3395 | 0.2465 | 0.1489 | 0.2986 |
| Stacking (raw)                    | 0.2512 | 0.2614 | 0.1828 | 0.7140 | 0.4093 | 0.2837 | 0.3341 | 0.5245 |
| Stacking (Platt)                  | 0.2512 | 0.2614 | 0.1828 | 0.7140 | 0.4093 | 0.2837 | 0.0362 | 0.0066 |
| Stacking (Isotonic)               | 0.2382 | 0.2708 | 0.1856 | 0.6372 | 0.3465 | 0.2395 | 0.0342 | 0.0000 |

## 기준 대비 평가

- 단일 모델 기준 PR-AUC: **0.1741**
- Stacking (Isotonic) PR-AUC: **0.2382**
- 결과: **✓ 개선됨**
