# O0_clean ACLED-free baseline results

실행일: 2026-06-04
실험명: `o0_acled_free_clean_baseline`

## Split

| split | rows | period | positive rate |
|---|---:|---|---:|
| train_fit | 190,646 | 2014-01-01 ~ 2022-12-31 | 0.0427 |
| tune_cal | 21,170 | 2023-01-01 ~ 2023-12-31 | 0.0441 |
| val_eval | 10,556 | 2024-01-01 ~ 2024-06-30 | 0.0407 |
| test | not evaluated | 2024-07-01 onward | - |

## Feature Set

- O0_clean feature 수: 35
- 사용 feature group: GDELT events 19, economic 15, country 1
- ACLED/safe_ACLED/macis_se_score/GDELT title/theme/person/embedding/cosine/vector 미사용
- label/future/next/past14d 컬럼 미사용

## val_eval Metrics

| model | PR-AUC | P@top1% | P@top5% | P@top10% | Lift@top1% | Lift@top5% | Lift@top10% | R@P>=0.10 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.0590 | 0.0755 | 0.0852 | 0.0710 | 1.8527 | 2.0922 | 1.7435 | 0.0163 | 0.1528 | 0.3165 |
| XGBoost | 0.0454 | 0.0377 | 0.0379 | 0.0388 | 0.9264 | 0.9299 | 0.9531 | 0.0000 | 0.2102 | 0.3996 |
| Stacking raw | 0.0583 | 0.0755 | 0.0852 | 0.0710 | 1.8527 | 2.0922 | 1.7435 | 0.0163 | 0.3358 | 0.5435 |
| Stacking Platt | 0.0583 | 0.0755 | 0.0852 | 0.0710 | 1.8527 | 2.0922 | 1.7435 | 0.0163 | 0.0389 | 0.0004 |
| Stacking Isotonic | 0.0538 | 0.0849 | 0.0530 | 0.0720 | 2.0843 | 1.3018 | 1.7668 | 0.0047 | 0.0389 | 0.0011 |

## OOF Folds

| fold | pred_year | n_train | n_pred | pos_rate | LGB PR-AUC | XGB PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| OOF_F1 | 2018 | 84,738 | 21,170 | 0.0514 | 0.0842 | 0.0694 |
| OOF_F2 | 2019 | 105,908 | 21,170 | 0.0461 | 0.0618 | 0.0606 |
| OOF_F3 | 2020 | 127,078 | 21,228 | 0.0468 | 0.0523 | 0.0475 |
| OOF_F4 | 2021 | 148,306 | 21,170 | 0.0456 | 0.0620 | 0.0587 |
| OOF_F5 | 2022 | 169,476 | 21,170 | 0.0480 | 0.0600 | 0.0549 |

## Baseline Comparison

- old B PR-AUC: 0.0564
- O0_clean Stacking Platt PR-AUC: 0.0583
- delta: +0.0019
- Meta LogReg C: 1.0
- 판단: O1_clean으로 확장 필요

## Test Policy

test set은 로드/평가하지 않았다. 예측 CSV와 모델 파일도 저장하지 않았다.