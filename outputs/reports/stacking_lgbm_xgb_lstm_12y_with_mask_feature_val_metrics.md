# Stacking LGBM+XGB+LSTM — Validation Metrics

> Experiment: `stacking_lgbm_xgb_lstm_12y_with_mask_feature` | Owner: `D_byeonghyeon`
> Meta-learner best C: 1.0

## Base Model Metrics

| Model | PR-AUC | P@5% | Brier | ECE |
|-------|--------|------|-------|-----|
| LGBM | 0.2606 | 0.2614 | 0.1462 | 0.2839 |
| XGB | 0.2631 | 0.2576 | 0.1255 | 0.2594 |
| LSTM | 0.1030 | 0.1231 | 0.2225 | 0.3943 |

## Stacking Metrics

| Variant | PR-AUC | P@5% | Brier | ECE |
|---------|--------|------|-------|-----|
| stack_raw | 0.2656 | 0.2670 | 0.3058 | 0.4953 |
| stack_platt | 0.2656 | 0.2670 | 0.0359 | 0.0067 |
| stack_isotonic | 0.2557 | 0.2652 | 0.0339 | 0.0000 |

## Tree-Only vs LGBM+XGB+LSTM (Platt PR-AUC delta)

| Item | Tree-only | +LSTM | delta |
|------|-----------|-------|-------|
| PR-AUC | 0.2714 | 0.2656 | -0.0058 |
| P@5% | 0.2689 | 0.2670 | -0.0019 |
| ECE | 0.0083 | 0.0067 | -0.0016 |

## Meta-Model Coefficients

| Base Model | Coefficient |
|------------|-------------|
| lgbm | 2.0881 |
| xgb | 2.2084 |
| lstm | 0.6921 |

## Base Model Probability Correlations (Val)

| | lgbm | xgb | lstm |
|---|---|---|---|
| lgbm | 1.0000 | 0.8842 | 0.4874 |
| xgb | 0.8842 | 1.0000 | 0.5187 |
| lstm | 0.4874 | 0.5187 | 1.0000 |

## Top-5% Alert Overlap (Jaccard / Recall)

| Pair | Jaccard | A-in-B | B-in-A |
|------|---------|--------|--------|
| LGBM vs XGB | 0.620 | 0.765 | 0.765 |
| LGBM vs LSTM | 0.110 | 0.199 | 0.199 |
| XGB vs LSTM | 0.105 | 0.189 | 0.189 |
| tree-only vs +LSTM stack | 0.899 | 0.947 | 0.947 |
