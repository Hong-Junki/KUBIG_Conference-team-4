# F2_clean Feature Importance 분석

**분석일**: 2026-06-04  
**모델**: LightGBM + XGBoost (cleanval split, train_fit=2014~2022, tune_cal=2023)  
**feature 수**: 94개 (B 35 + safe ACLED 15 + GDELT title 21 + coverage_mask 1 + theme/person 22)  

---

## Feature Group별 Importance

| Group | feature 수 | LGB gain % | LGB split % | XGB gain % |
|-------|-----------|-----------|------------|-----------|
| **country** | 1 |  24.60% |  10.95% |   2.36% |
| **economic** | 15 |  23.18% |  27.47% |  12.64% |
| **gdelt_events** | 19 |  21.19% |  25.14% |  23.32% |
| **gdelt_title** | 21 |  15.17% |  17.84% |  24.71% |
| **gdelt_theme_person** | 22 |  11.52% |  14.40% |  21.87% |
| **safe_acled** | 15 |   4.34% |   4.20% |  15.11% |
| **coverage_mask** | 1 |   0.00% |   0.00% |   0.00% |

---

## Individual Feature Top 30 (LightGBM gain 기준)

| rank | feature | group | LGB gain % | LGB split % | XGB gain % |
|------|---------|-------|-----------|------------|-----------|
|  1 | `country` | country |  24.60% |  10.95% |   2.36% |
|  2 | `econ_gold` | economic |   4.49% |   2.68% |   1.37% |
|  3 | `econ_wti` | economic |   3.62% |   3.17% |   1.22% |
|  4 | `gdelt_title_tone_std_7d` | gdelt_title |   2.30% |   2.35% |   1.08% |
|  5 | `gdelt_tone_mean_14d` | gdelt_events |   2.23% |   2.01% |   1.05% |
|  6 | `econ_stlfsi4_pct_7d` | economic |   1.96% |   2.81% |   0.94% |
|  7 | `econ_vix` | economic |   1.91% |   2.56% |   0.99% |
|  8 | `gdelt_person_density_7d` | gdelt_theme_person |   1.87% |   1.90% |   1.18% |
|  9 | `econ_gold_pct_7d` | economic |   1.73% |   2.57% |   0.80% |
| 10 | `gdelt_quadclass_2_ratio` | gdelt_events |   1.71% |   1.77% |   0.86% |
| 11 | `econ_dxy` | economic |   1.65% |   2.01% |   0.94% |
| 12 | `gdelt_goldstein_mean_14d` | gdelt_events |   1.58% |   1.75% |   1.22% |
| 13 | `gdelt_title_tone_trend_7d` | gdelt_title |   1.52% |   1.72% |   1.15% |
| 14 | `econ_stlfsi4` | economic |   1.51% |   2.01% |   0.92% |
| 15 | `gdelt_tone_mean_30d` | gdelt_events |   1.42% |   1.75% |   0.97% |
| 16 | `gdelt_title_positive_count_7d` | gdelt_title |   1.42% |   1.00% |   1.14% |
| 17 | `econ_wti_pct_7d` | economic |   1.41% |   1.96% |   0.83% |
| 18 | `gdelt_goldstein_mean_7d` | gdelt_events |   1.40% |   1.82% |   0.98% |
| 19 | `econ_dxy_pct_7d` | economic |   1.40% |   1.99% |   0.74% |
| 20 | `gdelt_tone_mean_7d` | gdelt_events |   1.34% |   1.87% |   0.88% |
| 21 | `gdelt_title_tone_min_7d` | gdelt_title |   1.33% |   1.66% |   0.86% |
| 22 | `gdelt_goldstein_mean_30d` | gdelt_events |   1.28% |   1.34% |   0.95% |
| 23 | `econ_vix_pct_7d` | economic |   1.28% |   2.01% |   0.80% |
| 24 | `gdelt_quadclass_3_ratio` | gdelt_events |   1.27% |   1.77% |   0.82% |
| 25 | `gdelt_quadclass_1_ratio` | gdelt_events |   1.25% |   1.53% |   1.09% |
| 26 | `safe_acled_fatalities_max_7d_lag7` | safe_acled |   1.20% |   0.58% |   1.44% |
| 27 | `gdelt_goldstein_std_14d` | gdelt_events |   1.11% |   1.42% |   1.78% |
| 28 | `gdelt_goldstein_std_7d` | gdelt_events |   1.08% |   1.44% |   1.32% |
| 29 | `gdelt_title_tone_mean_1d` | gdelt_title |   1.06% |   1.30% |   1.27% |
| 30 | `safe_acled_fatalities_7d_lag7` | safe_acled |   1.04% |   0.84% |   1.38% |

---

## Feature Group별 주요 Feature (LGB gain 상위 5개)

### country

| feature | LGB gain % | LGB split % |
|---------|-----------|------------|
| `country` |  24.60% |  10.95% |

### economic

| feature | LGB gain % | LGB split % |
|---------|-----------|------------|
| `econ_gold` |   4.49% |   2.68% |
| `econ_wti` |   3.62% |   3.17% |
| `econ_stlfsi4_pct_7d` |   1.96% |   2.81% |
| `econ_vix` |   1.91% |   2.56% |
| `econ_gold_pct_7d` |   1.73% |   2.57% |

### gdelt_events

| feature | LGB gain % | LGB split % |
|---------|-----------|------------|
| `gdelt_tone_mean_14d` |   2.23% |   2.01% |
| `gdelt_quadclass_2_ratio` |   1.71% |   1.77% |
| `gdelt_goldstein_mean_14d` |   1.58% |   1.75% |
| `gdelt_tone_mean_30d` |   1.42% |   1.75% |
| `gdelt_goldstein_mean_7d` |   1.40% |   1.82% |

### gdelt_title

| feature | LGB gain % | LGB split % |
|---------|-----------|------------|
| `gdelt_title_tone_std_7d` |   2.30% |   2.35% |
| `gdelt_title_tone_trend_7d` |   1.52% |   1.72% |
| `gdelt_title_positive_count_7d` |   1.42% |   1.00% |
| `gdelt_title_tone_min_7d` |   1.33% |   1.66% |
| `gdelt_title_tone_mean_1d` |   1.06% |   1.30% |

### gdelt_theme_person

| feature | LGB gain % | LGB split % |
|---------|-----------|------------|
| `gdelt_person_density_7d` |   1.87% |   1.90% |
| `gdelt_theme_refugee_count_7d` |   0.84% |   1.01% |
| `gdelt_theme_protest_count_7d` |   0.82% |   0.91% |
| `gdelt_theme_military_count_7d` |   0.71% |   0.91% |
| `gdelt_theme_sanction_count_7d` |   0.69% |   1.03% |

### safe_acled

| feature | LGB gain % | LGB split % |
|---------|-----------|------------|
| `safe_acled_fatalities_max_7d_lag7` |   1.20% |   0.58% |
| `safe_acled_fatalities_7d_lag7` |   1.04% |   0.84% |
| `safe_acled_fatalities_30d_lag7` |   0.30% |   0.33% |
| `safe_acled_event_count_30d_lag7` |   0.29% |   0.14% |
| `safe_acled_ratio_battles_lag7` |   0.24% |   0.39% |

### coverage_mask

| feature | LGB gain % | LGB split % |
|---------|-----------|------------|
| `gdelt_title_coverage_mask` |   0.00% |   0.00% |

---

## 해석

### 가장 중요한 feature group

LGB gain 기준 1위: **country**, 2위: **economic**

**safe_acled** (4.3%): 과거 ACLED 분쟁 기록 (shift 7일 lag). 예상보다 낮을 경우 GDELT 신호가 더 강함을 의미.

**gdelt_title** (15.2%): GDELT 기사 수/tone/다양성 피처. safe ACLED 위에서도 독립적 기여 확인.

**gdelt_theme_person** (11.5%): 테마/인물 집계 피처. F2-F1 delta +0.0191의 주요 원인으로 확인됨.

**economic** (23.2%): 글로벌 경제 지표. 실질적 기여 확인.

**country** (24.6%): ISO3 국가 코드. 높으면 국가 고정효과 absorb (overfitting 주의).

---

## 주의사항

1. **feature importance는 근사적 해석**일 뿐 causal effect가 아니다.  
   높은 importance = '모델이 이 feature를 많이 사용함'이지, '실제 원인'이 아님.

2. **gain importance의 correlated feature 문제**: 서로 상관된 feature(예: count_7d vs count_14d)는  
   중요도가 분산되거나 한쪽에 몰릴 수 있다. group 단위 집계가 개별 feature보다 더 안정적이다.

3. **country feature**: 국가별 고정효과를 흡수하는 경향이 있어 importance가 높게 나올 수 있다.  
   이는 과적합 징조일 수 있으므로 주의.

4. **최종 성능 평가는 아직 test set에서 하지 않았다.**  
   현재 val_eval 기준으로 F2_clean이 최고이며, test 1회 평가 후 최종 확정한다.

*생성: 2026-06-04*