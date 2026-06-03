# LightGBM + SE + Time Features — Validation Experiment Report

Generated: 2026-04-30
Model artifact: `outputs/models/lightgbm_se_time_features.pkl`
Evaluated on: **val set only** (2024-01 ~ 2024-06)

> **⚠️ 제출 CSV 형식 주의사항**
> 최종 제출 파일은 반드시 `date,country,y_prob` **세 컬럼만** 유지해야 합니다.
> 이 스크립트는 제출 CSV를 생성하지 않습니다.
> 제출 파일명 형식: `predictions__{model_name}__byeonghyeon.csv`

## 1. Feature Engineering Overview

### train+val concat 후 feature 생성 이유

val 초반부(2024-01)의 rolling z-score와 momentum feature는 train 말미(2023-12 등)의
과거 관측값을 baseline으로 사용해야 정상적으로 계산됩니다.
train과 val을 따로 처리하면 val 초반부에 NaN이 다수 발생하거나
rolling baseline이 truncated되어 z-score가 왜곡됩니다.
concat 후 country+date 정렬로 feature를 생성하고 split으로 재분리하면
이 문제를 해결할 수 있으며, 시간상 과거 데이터만 사용하므로 leakage가 아닙니다.
(test set은 concat에 포함하지 않았습니다.)

### Leakage 방지 방식

| Feature 유형 | Leakage 방지 방식 |
|-------------|-----------------|
| Difference / Ratio | 동일 row의 7d/14d/30d window는 각각 과거 데이터만 포함 — 추가 조치 불필요 |
| Rolling z-score | `shift(1)` 적용: rolling window에서 현재 row 제외 (t 시점 z-score = [t-90, t-1] 기반) |
| Momentum lag3 | `shift(3)` 적용: 3일 전 값과의 차이 — 미래 정보 없음 |
| Interaction | z-score·diff 파생값 × macis_se_score — 모두 과거 정보 기반 |

### Feature 생성 결과

| 항목 | 값 |
|------|-----|
| Base feature 수 (7d/14d/30d 모두 존재) | 8 |
| 생성된 derived feature 수 | 54 |
| 품질 필터로 제거된 feature 수 | 0 |
| 최종 사용 derived feature 수 | 54 |
| 기존 SE feature 수 (macis_se_score 포함) | 56 |
| 최종 전체 feature 수 | 110 |

**대상 base features:**

- `acled_event_count` (→ `acled_event_count_7d/14d/30d`)
- `acled_fatalities` (→ `acled_fatalities_7d/14d/30d`)
- `acled_fatalities_max` (→ `acled_fatalities_max_7d/14d/30d`)
- `gdelt_event_count` (→ `gdelt_event_count_7d/14d/30d`)
- `gdelt_goldstein_mean` (→ `gdelt_goldstein_mean_7d/14d/30d`)
- `gdelt_goldstein_std` (→ `gdelt_goldstein_std_7d/14d/30d`)
- `gdelt_mentions_sum` (→ `gdelt_mentions_sum_7d/14d/30d`)
- `gdelt_tone_mean` (→ `gdelt_tone_mean_7d/14d/30d`)

## 2. Model Comparison — Val Set Metrics

| Metric | SE (기존) | SE+TF (이번) | Delta |
|--------|----------|-------------|-------|
| PR-AUC | 0.1628 | 0.1640 | +0.0012 ▲ |
| P@top1% | 0.3491 | 0.3679 | +0.0189 ▲ |
| P@top3% | 0.2524 | 0.2461 | -0.0063 ▼ |
| P@top5% | 0.2254 | 0.2216 | -0.0038 ▼ |
| P@top10% | 0.1600 | 0.1544 | -0.0057 ▼ |
| R@top1% | 0.0860 | 0.0907 | +0.0047 ▲ |
| R@top3% | 0.1860 | 0.1814 | -0.0047 ▼ |
| R@top5% | 0.2767 | 0.2721 | -0.0047 ▼ |
| R@top10% | 0.3930 | 0.3791 | -0.0140 ▼ |
| R@P≥0.10 | 0.6209 | 0.5977 | -0.0233 ▼ |
| R@P≥0.20 | 0.3256 | 0.3047 | -0.0209 ▼ |
| R@P≥0.30 | 0.1116 | 0.1140 | +0.0023 ▲ |
| ECE (↓ good) | 0.1829 | 0.1889 | +0.0060 ▲ (악화) |

## 3. Monthly PR-AUC

| Month | SE (기존) | SE+TF (이번) | Delta |
|-------|----------|-------------|-------|
| 2024-01 | 0.1774 | 0.1647 | -0.0127 ▼ |
| 2024-02 | 0.1310 | 0.1584 | +0.0274 ▲ |
| 2024-03 | 0.1750 | 0.1794 | +0.0043 ▲ |
| 2024-04 | 0.1576 | 0.1259 | -0.0317 ▼ |
| 2024-05 | 0.1944 | 0.1995 | +0.0051 ▲ |
| 2024-06 | 0.1844 | 0.1927 | +0.0083 ▲ |

## 4. Feature Importance Top 30 (Gain 기준)

| Rank | Feature | Importance (Gain) | Derived? |
|------|---------|-------------------|---------|
| 1 | country | 71100.97 |  |
| 2 | tf_ratio30_acled_fatalities | 43672.88 | ✓ |
| 3 | macis_se_score | 40322.61 |  |
| 4 | tf_ix_se_z_acled_event_count | 16945.08 | ✓ |
| 5 | tf_diff30_acled_fatalities | 11751.04 | ✓ |
| 6 | tf_ix_se_z_acled_fatalities | 11597.45 | ✓ |
| 7 | tf_z90_acled_fatalities_max | 10153.35 | ✓ |
| 8 | tf_z90_acled_event_count | 9789.92 | ✓ |
| 9 | tf_ratio14_acled_fatalities | 9436.08 | ✓ |
| 10 | econ_stlfsi4_pct_7d | 8351.37 |  |
| 11 | tf_ratio30_gdelt_tone_mean | 8098.80 | ✓ |
| 12 | tf_mom3_gdelt_goldstein_mean | 8034.84 | ✓ |
| 13 | econ_wti_pct_7d | 7755.56 |  |
| 14 | gdelt_quadclass_2_ratio | 7499.47 |  |
| 15 | tf_z90_gdelt_tone_mean | 7255.62 | ✓ |
| 16 | acled_fatalities_14d | 6765.56 |  |
| 17 | tf_z90_acled_fatalities | 6610.75 | ✓ |
| 18 | tf_mom3_acled_fatalities_max | 6586.86 | ✓ |
| 19 | tf_ix_se_d30_gdelt_tone_mean | 6575.84 | ✓ |
| 20 | tf_ratio30_gdelt_goldstein_std | 6555.75 | ✓ |
| 21 | tf_ratio14_gdelt_goldstein_std | 6517.52 | ✓ |
| 22 | tf_mom3_gdelt_goldstein_std | 6269.31 | ✓ |
| 23 | tf_ratio30_acled_fatalities_max | 6268.47 | ✓ |
| 24 | econ_dxy_pct_7d | 6213.88 |  |
| 25 | tf_ratio30_gdelt_goldstein_mean | 6210.45 | ✓ |
| 26 | tf_mom3_gdelt_tone_mean | 6018.66 | ✓ |
| 27 | tf_ratio14_gdelt_tone_mean | 5948.82 | ✓ |
| 28 | tf_diff14_gdelt_goldstein_mean | 5944.35 | ✓ |
| 29 | tf_z90_gdelt_goldstein_mean | 5927.75 | ✓ |
| 30 | tf_ratio30_acled_event_count | 5857.63 | ✓ |

## 5. Verdict

**조건부 채택**: PR-AUC는 개선됐으나 P@top5%는 저하됐습니다. 전반적 순위 품질을 중시하면 채택, 상위 알림 정밀도를 중시하면 기존 SE 유지.

**개선된 지표 (4개):** PR-AUC, P@top1%, R@top1%, R@P≥0.30

**악화된 지표 (9개):** P@top3%, P@top5%, P@top10%, R@top3%, R@top5%, R@top10%, R@P≥0.10, R@P≥0.20, ECE (↓ good)

**Top-10 feature 중 derived feature 포함 수:** 7개

---

*이 리포트는 val set 전용 진단입니다. test set은 사용되지 않았습니다.*