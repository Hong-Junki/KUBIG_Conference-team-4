# 트리 하이퍼파라미터 민감도 절제 리포트

> **기준 모델**: `stacking_tree_only_12y_with_mask_feature` + Platt
> **기준 Platt PR-AUC**: **0.2714**  |  P@5%: 0.2689  |  ECE: 0.0083
> **Owner**: `D_byeonghyeon`

---

## 베이스라인 하이퍼파라미터 (이 민감도 실험의 기준)

| 파라미터 | 베이스라인 값 |
|---------|------------|
| LightGBM num_leaves | 63 |
| LightGBM scale_pos_weight | 22 |
| LightGBM min_child_samples | 20 |
| XGBoost max_depth | 4 |
| XGBoost scale_pos_weight | 22 |
| sqrt(neg/pos) | 4.7258 |

---

## 전체 변형 결과 (Platt PR-AUC 내림차순)

| 변형 | 설명 | LGBM | XGB | Platt PR-AUC | P@5% | ECE | best C | delta |
|------|------|------|-----|-------------|------|-----|--------|-------|
| `lgbm_spw_sqrt` | LightGBM scale_pos_weight=sqrt(neg/pos)=4.7258 | 0.2700 | 0.2195 | 0.2551 | 0.2595 | 0.0063 | 1.0 | -0.0163 |
| `xgb_depth_5` | XGBoost max_depth=5 (deeper) | 0.2606 | 0.2314 | 0.2493 | 0.2500 | 0.0072 | 0.01 | -0.0221 |
| `xgb_spw_sqrt` | XGBoost scale_pos_weight=sqrt(neg/pos)=4.7258 | 0.2606 | 0.2362 | 0.2466 | 0.2481 | 0.0040 | 0.01 | -0.0248 |
| `lgbm_spw_10` | LightGBM scale_pos_weight=10 | 0.2617 | 0.2195 | 0.2457 | 0.2519 | 0.0064 | 0.01 | -0.0257 |
| `lgbm_min_child_50` | LightGBM min_child_samples=50 | 0.2620 | 0.2195 | 0.2442 | 0.2462 | 0.0062 | 0.01 | -0.0272 |
| `lgbm_num_leaves_31` | LightGBM num_leaves=31 (shallower) | 0.2636 | 0.2195 | 0.2413 | 0.2481 | 0.0062 | 0.01 | -0.0301 |
| `xgb_depth_3` | XGBoost max_depth=3 (shallower) | 0.2606 | 0.2135 | 0.2386 | 0.2443 | 0.0067 | 0.01 | -0.0328 |
| `lgbm_num_leaves_127` | LightGBM num_leaves=127 (deeper) | 0.2519 | 0.2195 | 0.2374 | 0.2367 | 0.0056 | 0.01 | -0.0340 |
| `xgb_spw_10` | XGBoost scale_pos_weight=10 | 0.2606 | 0.2190 | 0.2335 | 0.2424 | 0.0049 | 0.01 | -0.0379 |

---

## 상세 Platt 지표

| 변형 | PR-AUC | P@5% | P@10% | R@P≥.10 | R@P≥.20 | R@P≥.30 | Brier | ECE |
|------|--------|------|-------|---------|---------|---------|-------|-----|
| `lgbm_spw_sqrt` | 0.2551 | 0.2595 | 0.1828 | 0.6884 | 0.3953 | 0.2814 | 0.0357 | 0.0063 |
| `xgb_depth_5` | 0.2493 | 0.2500 | 0.1799 | 0.7558 | 0.3907 | 0.2581 | 0.0360 | 0.0072 |
| `xgb_spw_sqrt` | 0.2466 | 0.2481 | 0.1818 | 0.7535 | 0.4047 | 0.2558 | 0.0356 | 0.0040 |
| `lgbm_spw_10` | 0.2457 | 0.2519 | 0.1818 | 0.7163 | 0.3884 | 0.2558 | 0.0358 | 0.0064 |
| `lgbm_min_child_50` | 0.2442 | 0.2462 | 0.1761 | 0.7163 | 0.3977 | 0.2535 | 0.0359 | 0.0062 |
| `lgbm_num_leaves_31` | 0.2413 | 0.2481 | 0.1809 | 0.7140 | 0.4070 | 0.2302 | 0.0360 | 0.0062 |
| `xgb_depth_3` | 0.2386 | 0.2443 | 0.1790 | 0.7233 | 0.3907 | 0.2465 | 0.0361 | 0.0067 |
| `lgbm_num_leaves_127` | 0.2374 | 0.2367 | 0.1723 | 0.7140 | 0.3814 | 0.2395 | 0.0361 | 0.0056 |
| `xgb_spw_10` | 0.2335 | 0.2424 | 0.1790 | 0.7302 | 0.3651 | 0.2349 | 0.0359 | 0.0049 |

---

## 해석 및 결론

### 1. 가장 강한 변형
- 최고 Platt PR-AUC: `lgbm_spw_sqrt` — LightGBM scale_pos_weight=sqrt(neg/pos)=4.7258
  PR-AUC=0.2551  P@5%=0.2595  ECE=0.0063  (delta -0.0163)
- 어떤 변형도 현재 최선(0.2714) + 임계값(0.003) = 0.2744 초과하지 않음 → 테스트 예측 저장 없음.

### 2. num_leaves 민감도 (LightGBM)
- num_leaves=31:  PR-AUC=0.2413  (delta -0.0301)
- num_leaves=63:  PR-AUC=0.2714  (기준)
- num_leaves=127: PR-AUC=0.2374  (delta -0.0340)
- 현재 설정은 num_leaves 변화에 일정 수준 민감함.

### 3. scale_pos_weight 민감도
- LGBM spw=sqrt(4.73): PR-AUC=0.2551  ECE=0.0063
- LGBM spw=10:          PR-AUC=0.2457  ECE=0.0064
- LGBM spw=22(기준):    PR-AUC=0.2714  ECE=0.0083
- XGB  spw=sqrt(4.73): PR-AUC=0.2466  ECE=0.0040
- XGB  spw=10:          PR-AUC=0.2335  ECE=0.0049
- XGB  spw=22(기준):    PR-AUC=0.2714  ECE=0.0083
- scale_pos_weight 감소(22→sqrt)는 PR-AUC 하락 가능성이 있으나 ECE 개선 효과 관찰 가능. 보정 개선이 목표라면 spw 감소를 추가 검토할 수 있음.

### 4. XGBoost max_depth 민감도
- max_depth=3: PR-AUC=0.2386  (delta -0.0328)
- max_depth=4: PR-AUC=0.2714  (기준, 이 실험 베이스라인)
- max_depth=5: PR-AUC=0.2493  (delta -0.0221)

### 5. LightGBM min_child_samples 민감도
- min_child_samples=20(기준): PR-AUC=0.2714
- min_child_samples=50:       PR-AUC=0.2442  (delta -0.0272)

### 6. 현재 최선 모델 변경 여부
- **변경 불필요.** 어떤 단일 하이퍼파라미터 변형도 현재 최선(0.2714)보다 유의미하게 높지 않음.
- `stacking_tree_only_12y_with_mask_feature` + Platt 유지 권장.

### 7. 권고 다음 과제
- D-category 하이퍼파라미터 민감도 분석 완료 — 현재 설정이 로컬 최적점임을 확인.
- 대시보드 업데이트 또는 팀 결과 공유를 다음 단계로 권장.
- 필요시 LightGBM/XGBoost 학습률(0.05→0.03)이나 subsample 비율 민감도 추가 실험 가능.
