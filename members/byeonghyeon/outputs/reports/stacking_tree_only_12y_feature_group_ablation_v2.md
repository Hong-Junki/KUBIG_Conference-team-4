# Feature Group Contribution Ablation

> Base experiment: `stacking_tree_only_12y_with_mask_feature`
> Current best Stacking Platt PR-AUC: **0.2714**
> Owner: `D_byeonghyeon`

## Feature Group Summary

| Group | n_feat | LGBM PR-AUC | XGB PR-AUC | Stack Platt PR-AUC | P@5% | ECE | Best C | delta vs best |
|-------|--------|-------------|------------|-------------------|------|-----|--------|---------------|
| full_best | 57 | 0.2654 | 0.2598 | 0.2697 | 0.2614 | 0.0074 | 0.01 | -0.0017 |
| acled_only_plus_se_mask | 22 | 0.2319 | 0.2270 | 0.2302 | 0.2273 | 0.0057 | 0.01 | -0.0412 |
| gdelt_only_plus_se_mask | 21 | 0.1051 | 0.1073 | 0.1079 | 0.1364 | 0.0030 | 0.1 | -0.1635 |
| economic_only_plus_se_mask | 17 | 0.0623 | 0.0592 | 0.0625 | 0.0814 | 0.0000 | 0.01 | -0.2089 |
| se_mask_only | 2 | 0.0608 | 0.0628 | 0.0620 | 0.0814 | 0.0000 | 1.0 | -0.2094 |
| no_acled_features | 37 | 0.1724 | 0.1219 | 0.1670 | 0.2008 | 0.0046 | 10.0 | -0.1044 |
| no_gdelt_features | 38 | 0.2545 | 0.2278 | 0.2486 | 0.2557 | 0.0060 | 0.01 | -0.0228 |
| no_economic_features | 42 | 0.2687 | 0.2363 | 0.2552 | 0.2652 | 0.0076 | 0.01 | -0.0162 |

## Detailed Platt Metrics per Group

| Group | Platt PR-AUC | Platt P@5% | Platt P@10% | R@P≥0.10 | R@P≥0.20 | R@P≥0.30 | Brier | ECE |
|-------|-------------|-----------|------------|---------|---------|---------|-------|-----|
| full_best | 0.2697 | 0.2614 | 0.1818 | 0.7558 | 0.4070 | 0.3070 | 0.0358 | 0.0074 |
| acled_only_plus_se_mask | 0.2302 | 0.2273 | 0.1638 | 0.7349 | 0.3163 | 0.2302 | 0.0363 | 0.0057 |
| gdelt_only_plus_se_mask | 0.1079 | 0.1364 | 0.1042 | 0.2860 | 0.0791 | 0.0395 | 0.0384 | 0.0030 |
| economic_only_plus_se_mask | 0.0625 | 0.0814 | 0.0814 | 0.0349 | 0.0000 | 0.0000 | 0.0390 | 0.0000 |
| se_mask_only | 0.0620 | 0.0814 | 0.0739 | 0.0302 | 0.0000 | 0.0000 | 0.0388 | 0.0000 |
| no_acled_features | 0.1670 | 0.2008 | 0.1392 | 0.5302 | 0.2465 | 0.1395 | 0.0374 | 0.0046 |
| no_gdelt_features | 0.2486 | 0.2557 | 0.1790 | 0.7465 | 0.3767 | 0.2628 | 0.0361 | 0.0060 |
| no_economic_features | 0.2552 | 0.2652 | 0.1771 | 0.7302 | 0.4000 | 0.3047 | 0.0359 | 0.0076 |

## Feature Group Composition

| Group | Features included |
|-------|------------------|
| full_best | 57 cols: acled_actor_type_1_ratio, acled_actor_type_2_ratio, acled_actor_type_3_ratio, acled_actor_type_4_ratio, acled_actor_type_5_ratio... |
| acled_only_plus_se_mask | 22 cols: acled_actor_type_1_ratio, acled_actor_type_2_ratio, acled_actor_type_3_ratio, acled_actor_type_4_ratio, acled_actor_type_5_ratio... |
| gdelt_only_plus_se_mask | 21 cols: gdelt_event_count_14d, gdelt_event_count_30d, gdelt_event_count_7d, gdelt_goldstein_mean_14d, gdelt_goldstein_mean_30d... |
| economic_only_plus_se_mask | 17 cols: econ_dxy, econ_dxy_pct_1d, econ_dxy_pct_7d, econ_gold, econ_gold_pct_1d... |
| se_mask_only | 2 cols: macis_se_score, acled_missing_mask |
| no_acled_features | 37 cols: acled_missing_mask, country, econ_dxy, econ_dxy_pct_1d, econ_dxy_pct_7d... |
| no_gdelt_features | 38 cols: acled_actor_type_1_ratio, acled_actor_type_2_ratio, acled_actor_type_3_ratio, acled_actor_type_4_ratio, acled_actor_type_5_ratio... |
| no_economic_features | 42 cols: acled_actor_type_1_ratio, acled_actor_type_2_ratio, acled_actor_type_3_ratio, acled_actor_type_4_ratio, acled_actor_type_5_ratio... |
