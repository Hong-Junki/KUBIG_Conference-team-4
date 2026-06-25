# Feature Group Ablation — Validation Report

Generated: 2026-04-30
Evaluated on: **val set only** (2024-01 ~ 2024-06)
Test set: **NOT loaded**  |  Submission CSV: **NOT produced**

> **⚠️ 제출 CSV 형식 주의사항**
> 최종 제출 파일은 반드시 `date,country,y_prob` **세 컬럼만** 유지해야 합니다.

## 1. Experiment Overview

| Experiment | Feature groups added | n_derived |
|-----------|---------------------|-----------|
| A — SE baseline | (없음) | 0 |
| B — SE + diff | tf_diff14_*, tf_diff30_* | 16 |
| C — SE + ratio | tf_ratio14_*, tf_ratio30_* | 16 |
| D — SE + z-score | tf_z90_* | 8 |
| E — SE + momentum | tf_mom3_* | 8 |
| F — SE + interaction | tf_ix_se_* | 6 |
| G — best groups | Group F | 6 |

**G 선택 기준 (보수적):**

- PR-AUC ≥ SE baseline **AND** P@top5% ≥ SE baseline 인 group만 채택
- P@top5% < SE baseline 인 group은 PR-AUC 개선 여부에 관계없이 제외
- P@top1% 개선이 있어도 P@top5%, R@P≥0.10, ECE 중 하나라도 악화되면 대시보드 기본 모델로 채택하지 않음

## 2. Key Delta Summary vs SE Baseline

| Metric | SE + diff | SE + ratio | SE + z-score | SE + momentum | SE + interaction | SE + best groups |
|--------|-------|-------|-------|-------|-------|-------|
| PR-AUC Δ | -0.0019 ▼ | -0.0079 ▼ | -0.0053 ▼ | -0.0002 ▼ | +0.0105 ▲ | +0.0105 ▲ |
| P@top1% Δ | -0.0094 ▼ | -0.1038 ▼ | -0.0189 ▼ | +0.0377 ▲ | +0.0283 ▲ | +0.0283 ▲ |
| P@top5% Δ | -0.0303 ▼ | -0.0170 ▼ | -0.0227 ▼ | -0.0133 ▼ | +0.0095 ▲ | +0.0095 ▲ |
| R@P≥0.10 Δ | +0.0209 ▲ | -0.0209 ▼ | -0.0256 ▼ | +0.0279 ▲ | -0.0116 ▼ | -0.0116 ▼ |
| ECE Δ | -0.0327 ▼(개선) | +0.0068 ▲(악화) | +0.0314 ▲(악화) | +0.0184 ▲(악화) | +0.0083 ▲(악화) | +0.0083 ▲(악화) |

## 3. Full Metrics — All Experiments

| Metric | SE baseline | SE + diff | SE + ratio | SE + z-score | SE + momentum | SE + interaction | SE + best groups |
|--------|-------|-------|-------|-------|-------|-------|-------|
| PR-AUC | 0.1673 | 0.1608 (-0.0019 ▼) | 0.1549 (-0.0079 ▼) | 0.1575 (-0.0053 ▼) | 0.1626 (-0.0002 ▼) | 0.1733 (+0.0105 ▲) | 0.1733 (+0.0105 ▲) |
| P@top1% | 0.3774 | 0.3396 (-0.0094 ▼) | 0.2453 (-0.1038 ▼) | 0.3302 (-0.0189 ▼) | 0.3868 (+0.0377 ▲) | 0.3774 (+0.0283 ▲) | 0.3774 (+0.0283 ▲) |
| P@top3% | 0.2650 | 0.2461 (-0.0063 ▼) | 0.2524 (+0.0000) | 0.2271 (-0.0252 ▼) | 0.2681 (+0.0158 ▲) | 0.2776 (+0.0252 ▲) | 0.2776 (+0.0252 ▲) |
| P@top5% | 0.2121 | 0.1951 (-0.0303 ▼) | 0.2083 (-0.0170 ▼) | 0.2027 (-0.0227 ▼) | 0.2121 (-0.0133 ▼) | 0.2348 (+0.0095 ▲) | 0.2348 (+0.0095 ▲) |
| P@top10% | 0.1562 | 0.1534 (-0.0066 ▼) | 0.1562 (-0.0038 ▼) | 0.1496 (-0.0104 ▼) | 0.1506 (-0.0095 ▼) | 0.1610 (+0.0009 ▲) | 0.1610 (+0.0009 ▲) |
| R@top1% | 0.0930 | 0.0837 (-0.0023 ▼) | 0.0605 (-0.0256 ▼) | 0.0814 (-0.0047 ▼) | 0.0953 (+0.0093 ▲) | 0.0930 (+0.0070 ▲) | 0.0930 (+0.0070 ▲) |
| R@top3% | 0.1953 | 0.1814 (-0.0047 ▼) | 0.1860 (+0.0000) | 0.1674 (-0.0186 ▼) | 0.1977 (+0.0116 ▲) | 0.2047 (+0.0186 ▲) | 0.2047 (+0.0186 ▲) |
| R@top5% | 0.2605 | 0.2395 (-0.0372 ▼) | 0.2558 (-0.0209 ▼) | 0.2488 (-0.0279 ▼) | 0.2605 (-0.0163 ▼) | 0.2884 (+0.0116 ▲) | 0.2884 (+0.0116 ▲) |
| R@top10% | 0.3837 | 0.3767 (-0.0163 ▼) | 0.3837 (-0.0093 ▼) | 0.3674 (-0.0256 ▼) | 0.3698 (-0.0233 ▼) | 0.3953 (+0.0023 ▲) | 0.3953 (+0.0023 ▲) |
| R@P≥0.10 | 0.6395 | 0.6419 (+0.0209 ▲) | 0.6000 (-0.0209 ▼) | 0.5953 (-0.0256 ▼) | 0.6488 (+0.0279 ▲) | 0.6093 (-0.0116 ▼) | 0.6093 (-0.0116 ▼) |
| R@P≥0.20 | 0.2674 | 0.2395 (-0.0860 ▼) | 0.2814 (-0.0442 ▼) | 0.2674 (-0.0581 ▼) | 0.2674 (-0.0581 ▼) | 0.3372 (+0.0116 ▲) | 0.3372 (+0.0116 ▲) |
| R@P≥0.30 | 0.1860 | 0.1209 (+0.0093 ▲) | 0.0581 (-0.0535 ▼) | 0.1047 (-0.0070 ▼) | 0.1907 (+0.0791 ▲) | 0.1465 (+0.0349 ▲) | 0.1465 (+0.0349 ▲) |
| ECE (↓ good) | 0.2248 | 0.1503 (-0.0327 ▼(개선)) | 0.1898 (+0.0068 ▲(악화)) | 0.2144 (+0.0314 ▲(악화)) | 0.2013 (+0.0184 ▲(악화)) | 0.1912 (+0.0083 ▲(악화)) | 0.1912 (+0.0083 ▲(악화)) |

## 4. Monthly PR-AUC

| Month | SE baseline | SE + diff | SE + ratio | SE + z-score | SE + momentum | SE + interaction | SE + best groups |
|-------|-------|-------|-------|-------|-------|-------|-------|
| 2024-01 | 0.1712 | 0.1705 | 0.1795 | 0.1782 | 0.1731 | 0.1807 | 0.1807 |
| 2024-02 | 0.1401 | 0.1161 | 0.1253 | 0.1426 | 0.1604 | 0.1579 | 0.1579 |
| 2024-03 | 0.2096 | 0.1847 | 0.1742 | 0.1664 | 0.1763 | 0.2082 | 0.2082 |
| 2024-04  ⚠️ | 0.1570 | 0.1875 | 0.1427 | 0.1474 | 0.1368 | 0.1508 | 0.1508 |
| 2024-05 | 0.2060 | 0.1843 | 0.1935 | 0.2053 | 0.1972 | 0.2046 | 0.2046 |
| 2024-06 | 0.1639 | 0.1777 | 0.1526 | 0.1611 | 0.2000 | 0.1633 | 0.1633 |

### Monthly PR-AUC Statistics

| Experiment | Mean | Min | 2024-04 | 2024-04 악화? |
|-----------|------|-----|---------|-------------|
| SE baseline | 0.1746 | 0.1401 | 0.1570 | — |
| SE + diff | 0.1701 | 0.1161 | 0.1875 | — |
| SE + ratio | 0.1613 | 0.1253 | 0.1427 | ✓ 악화 |
| SE + z-score | 0.1668 | 0.1426 | 0.1474 | ✓ 악화 |
| SE + momentum | 0.1740 | 0.1368 | 0.1368 | ✓ 악화 |
| SE + interaction | 0.1776 | 0.1508 | 0.1508 | ✓ 악화 |
| SE + best groups | 0.1776 | 0.1508 | 0.1508 | ✓ 악화 |

## 5. Feature Importance Top-5 per Experiment (Gain)

**SE baseline**

| Rank | Feature | Gain | Derived? |
|------|---------|------|---------|
| 1 | macis_se_score | 69181.5 |  |
| 2 | country | 57888.6 |  |
| 3 | acled_fatalities_7d | 53873.7 |  |
| 4 | acled_fatalities_max_7d | 18399.5 |  |
| 5 | econ_vix | 14095.1 |  |

**SE + diff**

| Rank | Feature | Gain | Derived? |
|------|---------|------|---------|
| 1 | country | 74011.3 |  |
| 2 | macis_se_score | 64976.5 |  |
| 3 | tf_diff30_acled_fatalities | 48454.8 | ✓ |
| 4 | acled_fatalities_7d | 29236.2 |  |
| 5 | econ_gold_pct_7d | 17197.1 |  |

**SE + ratio**

| Rank | Feature | Gain | Derived? |
|------|---------|------|---------|
| 1 | country | 67602.5 |  |
| 2 | macis_se_score | 61037.0 |  |
| 3 | tf_ratio30_acled_fatalities | 60597.2 | ✓ |
| 4 | tf_ratio30_gdelt_tone_mean | 12461.7 | ✓ |
| 5 | tf_ratio30_acled_event_count | 11913.2 | ✓ |

**SE + z-score**

| Rank | Feature | Gain | Derived? |
|------|---------|------|---------|
| 1 | country | 50578.2 |  |
| 2 | macis_se_score | 48640.8 |  |
| 3 | tf_z90_acled_fatalities | 48555.9 | ✓ |
| 4 | acled_fatalities_7d | 27574.2 |  |
| 5 | tf_z90_acled_fatalities_max | 15919.2 | ✓ |

**SE + momentum**

| Rank | Feature | Gain | Derived? |
|------|---------|------|---------|
| 1 | macis_se_score | 69662.1 |  |
| 2 | country | 65898.5 |  |
| 3 | acled_fatalities_7d | 46702.6 |  |
| 4 | acled_fatalities_max_7d | 14650.4 |  |
| 5 | tf_mom3_gdelt_goldstein_std | 14476.8 | ✓ |

**SE + interaction**

| Rank | Feature | Gain | Derived? |
|------|---------|------|---------|
| 1 | country | 64340.9 |  |
| 2 | tf_ix_se_z_acled_fatalities | 58576.5 | ✓ |
| 3 | macis_se_score | 37905.8 |  |
| 4 | acled_fatalities_7d | 25103.3 |  |
| 5 | tf_ix_se_d30_acled_fatalities | 22609.4 | ✓ |

**SE + best groups**

| Rank | Feature | Gain | Derived? |
|------|---------|------|---------|
| 1 | country | 64340.9 |  |
| 2 | tf_ix_se_z_acled_fatalities | 58576.5 | ✓ |
| 3 | macis_se_score | 37905.8 |  |
| 4 | acled_fatalities_7d | 25103.3 |  |
| 5 | tf_ix_se_d30_acled_fatalities | 22609.4 | ✓ |

## 6. Group-level Analysis

### 어떤 group이 P@top1%를 올렸는가?

개선 그룹: SE + momentum, SE + interaction

### 어떤 group이 P@top5%를 악화시켰는가?

악화 그룹: SE + diff, SE + ratio, SE + z-score, SE + momentum

### 어떤 group이 R@P≥0.10을 악화시켰는가?

악화 그룹: SE + ratio, SE + z-score, SE + interaction

### 2024-04 PR-AUC가 악화된 group

2024-04 악화 그룹: SE + ratio, SE + z-score, SE + momentum, SE + interaction

### 월별 PR-AUC 안정성 (분산 기준)

| Experiment | Monthly std | SE 대비 안정성 |
|-----------|-------------|--------------|
| SE baseline | 0.0253 | — |
| SE + diff | 0.0248 | △ 불안정 |
| SE + ratio | 0.0233 | △ 불안정 |
| SE + z-score | 0.0208 | △ 불안정 |
| SE + momentum | 0.0216 | △ 불안정 |
| SE + interaction | 0.0223 | △ 불안정 |
| SE + best groups | 0.0223 | △ 불안정 |

## 7. Final Verdict

### Group별 평가

- **SE + diff**: ❌ 채택 불가 (PR-AUC·P@top5% 모두 악화)
- **SE + ratio**: ❌ 채택 불가 (PR-AUC·P@top5% 모두 악화)
- **SE + z-score**: ❌ 채택 불가 (PR-AUC·P@top5% 모두 악화)
- **SE + momentum**: ❌ 채택 불가 (PR-AUC·P@top5% 모두 악화)
- **SE + interaction**: ✅ G 조합 채택 (PR-AUC·P@top5% 모두 개선, 일부 지표 소폭 악화)

### G 최종 판단

G 조합: **Group F**

**채택 권장**: G 모델이 SE 대비 PR-AUC(0.1733 vs 0.1628)와 P@top5%(0.2348 vs 0.2254) 모두를 유지하거나 개선했습니다.
다음 단계로 `lightgbm_se_time_features_g.pkl`을 후보 모델로 사용하세요.

### 최종 추천

→ **특정 feature group 추가** (Group F). G 모델을 채택하세요.
→ SE+TF 전체 모델(54개 동시 투입)은 ECE·P@top5% 악화로 **폐기**.

---

*이 리포트는 val set 전용 진단입니다. test set은 사용되지 않았습니다.*