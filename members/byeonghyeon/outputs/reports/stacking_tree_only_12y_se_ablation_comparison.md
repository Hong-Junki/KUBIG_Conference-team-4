# SE vs no-SE 절제 비교 리포트

> **기준**: `stacking_tree_only_12y` (SE 포함) vs `stacking_tree_only_12y_no_se` (SE 없음)
> **작성일**: 2026-05-23
> **delta = SE − no-SE**: 양수이면 SE가 도움, 음수이면 SE가 역효과

---

## PR-AUC 비교

| 모델 | SE 포함 | SE 없음 | delta (SE − no-SE) |
|------|---------|---------|---------------------|
| LightGBM | 0.2596 | 0.1067 | +0.1528 ↑ |
| XGBoost | 0.2534 | 0.1041 | +0.1493 ↑ |
| Stacking (raw) | 0.2656 | 0.1057 | +0.1599 ↑ |
| Stacking (Platt) | 0.2656 | 0.1057 | +0.1599 ↑ |
| Stacking (Isotonic) | 0.2535 | 0.1035 | +0.1500 ↑ |

## P@top5% 비교

| 모델 | SE 포함 | SE 없음 | delta |
|------|---------|---------|-------|
| LightGBM | 0.2708 | 0.1629 | +0.1080 ↑ |
| XGBoost | 0.2500 | 0.1458 | +0.1042 ↑ |
| Stacking (raw) | 0.2614 | 0.1591 | +0.1023 ↑ |
| Stacking (Platt) | 0.2614 | 0.1591 | +0.1023 ↑ |
| Stacking (Isotonic) | 0.2746 | 0.1648 | +0.1098 ↑ |

## Brier Score 비교 (낮을수록 좋음)

| 모델 | SE 포함 | SE 없음 | delta (SE − no-SE) |
|------|---------|---------|---------------------|
| LightGBM | 0.1443 | 0.1274 | +0.0169 ↑ (SE brier 변화) |
| XGBoost | 0.1314 | 0.2037 | -0.0723 ↓ (SE brier 변화) |
| Stacking (raw) | 0.3135 | 0.3553 | -0.0418 ↓ (SE brier 변화) |
| Stacking (Platt) | 0.0359 | 0.0381 | -0.0022 ↓ (SE brier 변화) |
| Stacking (Isotonic) | 0.0339 | 0.0377 | -0.0038 ↓ (SE brier 변화) |

## ECE 비교 (낮을수록 좋음)

| 모델 | SE 포함 | SE 없음 | delta (SE − no-SE) |
|------|---------|---------|---------------------|
| LightGBM | 0.2796 | 0.2779 | +0.0017 ↑ |
| XGBoost | 0.2663 | 0.3911 | -0.1247 ↓ |
| Stacking (raw) | 0.5071 | 0.5599 | -0.0528 ↓ |
| Stacking (Platt) | 0.0074 | 0.0035 | +0.0039 ↑ |
| Stacking (Isotonic) | 0.0000 | 0.0000 | -0.0000 ≈ |

## 해석 및 결론

- **LightGBM PR-AUC delta**: +0.1528 ↑
- **XGBoost PR-AUC delta**: +0.1493 ↑
- **Stacking Platt PR-AUC delta**: +0.1599 ↑

> **결론**: SE 피처가 스태킹 PR-AUC를 유의미하게 개선함. **SE 유지 권장.**

## 다음 단계 권고

1. `stacking_tree_only_12y_mask0` — mask=1 행 제외 후 재학습 (레퍼런스 조건 정합)
2. C담당 LSTM 파일 수령 시 BASE_MODELS에 추가
