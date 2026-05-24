# Final Tree Candidate Runner

최종 모델 선정 담당자에게 공유할 담당 B 후보 실행 코드입니다.

## 실행 명령

```powershell
python scripts\run_final_tree_candidates.py
```

## 사용하는 데이터

- Dataset: `se`
- Split:
  - train: `Merged_data/processed/dataset/train.parquet`
  - val: `Merged_data/processed/dataset/val.parquet`
  - test: `Merged_data/processed/dataset/test.parquet`
- SE feature:
  - `Merged_data/processed/features/se_scores.parquet`
- Feature set:
  - `ACLED + se_score`
  - includes `acled_missing_mask`

## 후보 모델 3개

### 1. `lightgbm_acled_se_rank1`

Best LightGBM by validation PR-AUC.

```text
scale_pos_weight = 4.7
num_leaves = 31
min_child_samples = 20
n_estimators = 500
learning_rate = 0.05
```

### 2. `lightgbm_acled_se_rank2`

Second-best LightGBM by validation PR-AUC. Test에서는 rank1보다 좋았다.

```text
scale_pos_weight = 4.7
num_leaves = 31
min_child_samples = 50
n_estimators = 500
learning_rate = 0.05
```

### 3. `xgboost_acled_se_rank1`

Best XGBoost by validation PR-AUC and best calibration.

```text
scale_pos_weight = 1.0
max_depth = 5
min_child_weight = 5
n_estimators = 500
learning_rate = 0.04
```

## 출력 파일

Metrics:

```text
artifacts/final_tree_candidate_metrics.csv
```

Predictions:

```text
artifacts/submissions/val_predictions__{candidate}.csv
artifacts/submissions/test_predictions__{candidate}.csv
```

## 최신 실행 결과

| Candidate | Split | PR-AUC | Persistence gain | P@top5% | R@P>=0.10 | ECE |
|---|---|---:|---:|---:|---:|---:|
| lightgbm_acled_se_rank1 | val | 0.2534 | +0.0195 | 0.2595 | 0.7395 | 0.1001 |
| lightgbm_acled_se_rank1 | test | 0.2276 | -0.0123 | 0.2532 | 0.6708 | 0.1020 |
| lightgbm_acled_se_rank2 | val | 0.2505 | +0.0166 | 0.2689 | 0.7326 | 0.1018 |
| lightgbm_acled_se_rank2 | test | 0.2335 | -0.0063 | 0.2634 | 0.7132 | 0.1041 |
| xgboost_acled_se_rank1 | val | 0.2492 | +0.0153 | 0.2595 | 0.7047 | 0.0074 |
| xgboost_acled_se_rank1 | test | 0.2342 | -0.0057 | 0.2545 | 0.7100 | 0.0038 |

## 공유 시 주의점

- val에서는 세 후보 모두 `persistence_gain > 0`.
- test에서는 세 후보 모두 persistence baseline보다 아주 약간 낮다.
- test 기준 PR-AUC 최고는 `xgboost_acled_se_rank1`.
- test 기준 P@top5% 최고는 `lightgbm_acled_se_rank2`.
- SE 생성 과정이 train-only인지 확인해야 최종 성능으로 확정할 수 있다.
