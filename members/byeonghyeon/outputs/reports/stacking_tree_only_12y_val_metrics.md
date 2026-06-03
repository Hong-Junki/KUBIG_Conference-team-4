# 검증 지표 — stacking_tree_only_12y

**실험**: stacking_tree_only_12y
**정책**: full  (acled_missing_mask=1 포함, 피처에서 제외)
**Base 모델**: LightGBM + 12y SE, XGBoost + 12y SE (tree-only 1차 프로토타입)
**Meta 모델**: Logistic Regression (선택된 C = 0.01)
**비교 기준**: 단일 모델 LightGBM+SE mask0_only val PR-AUC = 0.1741

> ⚠ Final base 모델은 val을 early stopping에 사용 → val 지표가 소폭 낙관적일 수 있음.
> LSTM은 C담당 파일 수령 후 추가 예정 (BASE_MODELS 리스트에 항목 추가).

## 지표 비교

| 모델                         | PR-AUC | P@5%   | R@P≥.10 | R@P≥.20 | R@P≥.30 | Brier  | ECE    |
|------------------------------|--------|--------|---------|---------|---------|--------|--------|
| LightGBM + 12y SE           | 0.2596 | 0.2708 | 0.7070 | 0.4326 | 0.3093 | 0.1443 | 0.2796 |
| XGBoost + 12y SE            | 0.2534 | 0.2500 | 0.7488 | 0.3535 | 0.2628 | 0.1314 | 0.2663 |
| Stacking (raw)              | 0.2656 | 0.2614 | 0.7419 | 0.3977 | 0.2837 | 0.3135 | 0.5071 |
| Stacking (Platt)            | 0.2656 | 0.2614 | 0.7419 | 0.3977 | 0.2837 | 0.0359 | 0.0074 |
| Stacking (Isotonic)         | 0.2535 | 0.2746 | 0.7349 | 0.3535 | 0.2558 | 0.0339 | 0.0000 |

## 기준 대비 평가

- 단일 모델 기준 PR-AUC: **0.1741**
- Stacking (Isotonic) PR-AUC: **0.2535**
- 결과: **✓ 개선됨**

## 다음 단계

1. `stacking_tree_only_12y_mask0` ablation 실행
   (mask=0 행만으로 재학습 → full-train vs mask0-only 비교)
2. C담당 LSTM 예측 파일 수령 후 BASE_MODELS에 추가 후 재실행
3. Isotonic calibrated test 파일을 대시보드 F 항목 후보로 사용
