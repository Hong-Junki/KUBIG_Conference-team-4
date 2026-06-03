# Train 시작일 절제 비교 리포트 (12y vs 8y)

> **기준 (12y)**: `stacking_tree_only_12y_with_mask_feature` (train 2014-2023)
> **비교 (8y)**: `stacking_tree_only_8y_with_mask_feature` (train 2016-2023)
> **작성일**: 2026-05-23
> **delta = 8y − 12y**: 양수이면 2016-start가 도움, 음수이면 역효과

---

## PR-AUC 비교

| 모델 | 12y (2014-start) | 8y (2016-start) | delta (8y − 12y) |
|------|------------------|-----------------|------------------|
| LightGBM | 0.2606 | 0.2604 | -0.0002 ≈ |
| XGBoost | 0.2631 | 0.2342 | -0.0289 ↓ |
| Stacking (raw) | 0.2714 | 0.2496 | -0.0218 ↓ |
| Stacking (Platt) | 0.2714 | 0.2496 | -0.0218 ↓ |
| Stacking (Isotonic) | 0.2617 | 0.2431 | -0.0187 ↓ |

## P@top5% 비교

| 모델 | 12y | 8y | delta |
|------|-----|-----|-------|
| LightGBM | 0.2614 | 0.2614 | 0.0000 ≈ |
| XGBoost | 0.2576 | 0.2481 | -0.0095 ↓ |
| Stacking (raw) | 0.2689 | 0.2576 | -0.0114 ↓ |
| Stacking (Platt) | 0.2689 | 0.2576 | -0.0114 ↓ |
| Stacking (Isotonic) | 0.2727 | 0.2595 | -0.0133 ↓ |

## Brier Score 비교 (낮을수록 좋음)

| 모델 | 12y | 8y | delta (8y − 12y) |
|------|-----|-----|------------------|
| LightGBM | 0.1462 | 0.1387 | -0.0076 ↓ (개선) |
| XGBoost | 0.1255 | 0.1572 | +0.0318 ↑ (악화) |
| Stacking (raw) | 0.3049 | 0.3863 | +0.0814 ↑ (악화) |
| Stacking (Platt) | 0.0359 | 0.0361 | 0.0002 ≈ |
| Stacking (Isotonic) | 0.0337 | 0.0342 | 0.0005 ≈ |

## ECE 비교 (낮을수록 좋음)

| 모델 | 12y | 8y | delta (8y − 12y) |
|------|-----|-----|------------------|
| LightGBM | 0.2839 | 0.2737 | -0.0103 ↓ (개선) |
| XGBoost | 0.2594 | 0.3127 | +0.0533 ↑ (악화) |
| Stacking (raw) | 0.4989 | 0.5773 | +0.0785 ↑ (악화) |
| Stacking (Platt) | 0.0083 | 0.0081 | -0.0002 ≈ |
| Stacking (Isotonic) | 0.0000 | 0.0000 | -0.0000 ≈ |

## 해석 및 결론

- **LightGBM PR-AUC delta (8y − 12y)**: -0.0002 ≈
- **XGBoost PR-AUC delta (8y − 12y)**: -0.0289 ↓
- **Stacking Platt PR-AUC (12y)**: 0.2714
- **Stacking Platt PR-AUC (8y)**: 0.2496
- **Stacking Platt PR-AUC delta**: -0.0218 ↓

> **결론**: 2016-start(8y)가 오히려 PR-AUC 하락 (-0.0218). **2014-start(12y) 유지 권장.** 2014-2015 데이터가 유용한 신호를 포함하고 있음.

## 다음 단계 권고

1. `stacking_tree_only_6y_with_mask_feature` — 2018-start 절제 실험 (5-fold OOF 필요)
2. C담당 LSTM OOF 파일 수령 시 BASE_MODELS에 추가 → 최종 앙상블 재실험
