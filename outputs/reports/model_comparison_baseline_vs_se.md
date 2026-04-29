# Baseline vs SE Score — Model Comparison

Generated: 2026-04-29

## Feature Set

| Model | Total features | macis_se_score |
|-------|---------------|----------------|
| Baseline LightGBM | 55 | No |
| LightGBM + SE     | 56  | Yes |

## Validation Set Metrics

| Metric | Baseline | + SE | Delta |
|--------|----------|------|-------|
| PR-AUC            | 0.1173 | 0.1628 | +0.0455 |
| P@top5%           | 0.1477 | 0.2254 | +0.0777 |
| R@P≥0.10          | 0.4419 | 0.6209 | +0.1791 |
| ECE               | 0.2314 | 0.1829 | -0.0484 |
| Best iteration    | 38 | 112 | — |

## Interpretation

macis_se_score가 **PR-AUC와 P@top5% 모두를 개선**했습니다. SE 모델(`predictions__lightgbm_se__byeonghyeon.csv`)을 팀 제출 후보로 우선 고려하세요. ECE도 개선되어 확률 calibration 품질이 향상됐습니다.

## Submission Files

| Model | Prediction file |
|-------|----------------|
| Baseline | `outputs/predictions/predictions__lightgbm__byeonghyeon.csv` |
| + SE     | `outputs/predictions/predictions__lightgbm_se__byeonghyeon.csv` |