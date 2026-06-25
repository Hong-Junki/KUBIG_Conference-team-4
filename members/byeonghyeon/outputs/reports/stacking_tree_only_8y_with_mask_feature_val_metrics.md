# 검증 지표 — stacking_tree_only_8y_with_mask_feature

**실험**: stacking_tree_only_8y_with_mask_feature
**정책**: full  (acled_missing_mask=1 포함, 피처로도 포함)
**Train 윈도우**: 8y (2016-2023, 12y 대비 2014-2015 제외)
**acled_missing_mask 피처**: 포함
**SE 피처**: 포함 (macis_se_score)
**Base 모델**: LightGBM + SE + mask (2016+), XGBoost + SE + mask (2016+)
**Meta 모델**: Logistic Regression (선택된 C = 0.01)
**비교 기준**: 12y_with_mask_feature Stacking Platt PR-AUC = 0.2714

> ⚠ Final base 모델은 val을 early stopping에 사용 → val 지표가 소폭 낙관적.
> Train 시작일 절제 실험 (2016-start). 전체 비교: train_start_ablation_comparison 리포트 참조.

## 지표 비교

| 모델                                   | PR-AUC | P@5%   | R@P≥.10 | R@P≥.20 | R@P≥.30 | Brier  | ECE    |
|----------------------------------------|--------|--------|---------|---------|---------|--------|--------|
| LightGBM + SE + mask (8y/2016+)       | 0.2604 | 0.2614 | 0.7442 | 0.4605 | 0.3000 | 0.1387 | 0.2737 |
| XGBoost  + SE + mask (8y/2016+)       | 0.2342 | 0.2481 | 0.7442 | 0.3698 | 0.2674 | 0.1572 | 0.3127 |
| Stacking (raw)                        | 0.2496 | 0.2576 | 0.7674 | 0.4419 | 0.2721 | 0.3863 | 0.5773 |
| Stacking (Platt)                      | 0.2496 | 0.2576 | 0.7674 | 0.4419 | 0.2721 | 0.0361 | 0.0081 |
| Stacking (Isotonic)                   | 0.2431 | 0.2595 | 0.7488 | 0.4326 | 0.2605 | 0.0342 | 0.0000 |

## 기준 대비 평가

- 12y_with_mask_feature Stacking Platt PR-AUC: **0.2714**
- 8y_with_mask_feature  Stacking Platt PR-AUC: **0.2496**
- 결과: **✗ 12y 대비 미개선**
