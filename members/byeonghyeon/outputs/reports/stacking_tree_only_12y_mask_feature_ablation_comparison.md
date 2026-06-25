# mask_feature vs no-mask_feature 절제 비교 리포트

> **기준**: `stacking_tree_only_12y` (mask 피처 미포함)
> **비교**: `stacking_tree_only_12y_with_mask_feature` (mask 피처 포함)
> **작성일**: 2026-05-23
> **delta = with_mask − baseline**: 양수이면 mask 피처가 도움, 음수이면 역효과

---

## PR-AUC 비교

| 모델 | baseline (mask 미포함) | with_mask | delta (with − base) |
|------|------------------------|-----------|---------------------|
| LightGBM | 0.2596 | 0.2606 | +0.0010 ↑ |
| XGBoost | 0.2534 | 0.2631 | +0.0097 ↑ |
| Stacking (raw) | 0.2656 | 0.2714 | +0.0059 ↑ |
| Stacking (Platt) | 0.2656 | 0.2714 | +0.0059 ↑ |
| Stacking (Isotonic) | 0.2535 | 0.2617 | +0.0082 ↑ |

## P@top5% 비교

| 모델 | baseline | with_mask | delta |
|------|----------|-----------|-------|
| LightGBM | 0.2708 | 0.2614 | -0.0095 ↓ |
| XGBoost | 0.2500 | 0.2576 | +0.0076 ↑ |
| Stacking (raw) | 0.2614 | 0.2689 | +0.0076 ↑ |
| Stacking (Platt) | 0.2614 | 0.2689 | +0.0076 ↑ |
| Stacking (Isotonic) | 0.2746 | 0.2727 | -0.0019 ↓ |

## Brier Score 비교 (낮을수록 좋음)

| 모델 | baseline | with_mask | delta (with − base) |
|------|----------|-----------|---------------------|
| LightGBM | 0.1443 | 0.1462 | +0.0019 ↑ (Brier 변화) |
| XGBoost | 0.1314 | 0.1255 | -0.0059 ↓ (Brier 변화) |
| Stacking (raw) | 0.3135 | 0.3049 | -0.0085 ↓ (Brier 변화) |
| Stacking (Platt) | 0.0359 | 0.0359 | -0.0000 ≈ (Brier 변화) |
| Stacking (Isotonic) | 0.0339 | 0.0337 | -0.0002 ≈ (Brier 변화) |

## ECE 비교 (낮을수록 좋음)

| 모델 | baseline | with_mask | delta (with − base) |
|------|----------|-----------|---------------------|
| LightGBM | 0.2796 | 0.2839 | +0.0044 ↑ |
| XGBoost | 0.2663 | 0.2594 | -0.0069 ↓ |
| Stacking (raw) | 0.5071 | 0.4989 | -0.0082 ↓ |
| Stacking (Platt) | 0.0074 | 0.0083 | +0.0009 ↑ |
| Stacking (Isotonic) | 0.0000 | 0.0000 | -0.0000 ≈ |

## 해석 및 결론

- **LightGBM PR-AUC delta**: +0.0010 ↑
- **XGBoost PR-AUC delta**: +0.0097 ↑
- **Stacking Platt PR-AUC delta**: +0.0059 ↑

> **결론**: acled_missing_mask 피처 추가 시 스태킹 PR-AUC 유의미하게 개선됨. **mask 피처 포함 권장.**

## 다음 단계 권고

1. `stacking_tree_only_12y_mask0` — mask=1 행 제외 후 재학습 (레퍼런스 조건 정합)
2. C담당 LSTM 파일 수령 시 BASE_MODELS에 추가
