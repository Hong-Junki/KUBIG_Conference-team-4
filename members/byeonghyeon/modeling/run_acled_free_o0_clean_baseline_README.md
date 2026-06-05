# O0_clean ACLED-free Baseline

## 목적

ACLED-free 운영 모델의 clean validation baseline을 재측정한다.

## Split

| split | period | role |
|---|---|---|
| train_fit | 2014-01-01 ~ 2022-12-31 | base model 학습, OOF 생성 |
| tune_cal | 2023-01-01 ~ 2023-12-31 | early stopping, meta C 선택, Platt calibration |
| val_eval | 2024-01-01 ~ 2024-06-30 | 순수 평가 |
| test | 2024-07-01 onward | 미평가 |

## Feature Set

O0_clean은 기존 로컬 dataset parquet에 이미 있는 35개 feature만 사용한다.

- GDELT events: 19
- Economic: 15
- Country: 1

제외:

- `acled_*`
- `safe_acled_*`
- `macis_se_score`
- `gdelt_title_*`
- `gdelt_theme_*`
- `gdelt_person_*`
- embedding/cosine/vector 계열
- label/future/next/past14d 컬럼

## 실행

```bash
python -m py_compile members/byeonghyeon/modeling/run_acled_free_o0_clean_baseline.py
SMOKE_TEST=1 python members/byeonghyeon/modeling/run_acled_free_o0_clean_baseline.py
python members/byeonghyeon/modeling/run_acled_free_o0_clean_baseline.py
```

## 출력

```text
members/byeonghyeon/outputs/reports/o0_acled_free_clean_baseline_results.md
```

예측 CSV와 모델 파일은 저장하지 않는다.
