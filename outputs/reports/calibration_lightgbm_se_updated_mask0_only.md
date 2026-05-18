# updated_mask0_only 모델 — 확률 보정(Calibration) 리포트

생성일: 2026-05-18
기반 모델: `outputs/models/lightgbm_se_updated_mask0_only.pkl` (재학습 없음)
평가 데이터: **val 세트 전용** (2024-01-01 ~ 2024-06-30)
test 세트: 레이블 미사용 — 최종 예측 파일 생성에만 사용

> **⚠️ 평가 한계 주의**
> 보정기(calibrator)를 val 세트 예측으로 **피팅한 뒤 동일 val 세트로 평가**합니다.
> Platt scaling(파라미터 2개)은 과적합 위험이 낮지만,
> Isotonic regression(가변 구간)은 val ECE/Brier 개선이 **과대 추정**될 수 있습니다.
> 또한 기반 LightGBM 모델이 val PR-AUC 기준 early stopping으로 학습됐으므로
> val은 완전한 홀드아웃 세트가 아닙니다. 모든 수치를 낙관적 추정으로 해석하세요.

---

## 1. val 세트 메트릭 비교 (보정 전/후)

| 방법 | PR-AUC | P@top5% | R@P≥0.10 | R@P≥0.20 | R@P≥0.30 | ECE | Brier |
|------|--------|---------|---------|---------|---------|-----|-------|
| raw LightGBM (기준) | 0.1741 | 0.2159 | 0.6209 | 0.2953 | 0.1698 | 0.2771 | 0.1390 |
| Platt scaling | 0.1741 | 0.2159 | 0.6209 | 0.2953 | 0.1698 | 0.0029 | 0.0365 |
| Isotonic regression | 0.1686 | 0.2178 | 0.5744 | 0.2953 | 0.1674 | 0.0000 | 0.0361 |

**보정 후 − raw 차이 (−이 개선, +이 악화)**

| 방법 | ΔPR-AUC | ΔP@top5% | ΔECE | ΔBrier |
|------|---------|---------|------|--------|
| Platt scaling      | +0.0000 | +0.0000 | -0.2742 | -0.1025 |
| Isotonic regression | -0.0055 | +0.0019 | -0.2771 | -0.1029 |

## 2. Platt scaling의 순위 메트릭 보존 여부

Platt scaling은 단조 증가 함수(sigmoid)이므로 이론적으로 순위를 변경하지 않습니다. PR-AUC, P@top5%, Recall@Precision 등 모든 순위 기반 지표는 raw와 동일해야 합니다.

✅ 확인: PR-AUC 변화 = +0.0000 (부동소수점 오차 수준, 순위 완전 보존)
✅ 확인: P@top5% 변화 = +0.0000 (순위 완전 보존)

Isotonic regression은 구간별 단조이므로 동점 구간 발생 시 순위가 미세하게 달라질 수 있습니다.
Isotonic PR-AUC 변화 = -0.0055 (⚠ 동점 구간 확인 권장)

## 3. ECE 및 Brier score 개선 여부

**Platt scaling**: ECE 0.2771 → 0.0029 (-0.2742, **개선**). sigmoid 보정이 모델의 체계적 과신(overconfidence)을 교정합니다.

**Isotonic regression**: ECE 0.2771 → 0.0000 (-0.2771, **개선**). 단, 동일 val 세트에서 피팅·평가했으므로 **과대 추정 가능성**이 있습니다.

Brier score: raw=0.1390  Platt=0.0365 (-0.1025)  Isotonic=0.0361 (-0.1029)

## 4. 최종 제출 및 대시보드 권장 파일

| 용도 | 권장 파일 | 이유 |
|------|----------|------|
| **팀 최종 제출** | `predictions__lightgbm_se_updated_mask0_only_platt__byeonghyeon.csv` | 순위 보존 + ECE/Brier 개선, 과적합 위험 낮음 |
| **대시보드 위험 점수** | `predictions__lightgbm_se_updated_mask0_only_platt__byeonghyeon.csv` | 보정된 확률이 절댓값 해석에 더 안전 |
| **국가 순위 / alert** | raw 또는 platt 동일 | 순위 불변이므로 어느 것이든 무방 |

> **참고**: Isotonic 파일은 ECE가 더 낮아 보이지만 val 과적합 위험이 있으므로 대외 제출보다는 내부 실험 용도로 활용하세요.

## 5. 보정 결과 해석 시 주의사항

1. **동일 세트 과적합**: 보정기를 val 세트로 피팅·평가했으므로 실제 미래(test 이후) 데이터에서의 개선폭은 더 작을 수 있습니다.

2. **LightGBM early stopping**: 기반 모델이 val PR-AUC 기준 early stopping으로 학습됐으므로 val은 완전한 홀드아웃 세트가 아닙니다. 보정 수치는 낙관적 추정으로 해석하세요.

3. **드문 사건 (양성 비율 ~4%)**: 소수의 양성 사례에서 보정 곡선이 불안정합니다. 특히 고확률 구간(0.3 이상)의 보정 품질을 과신하지 마세요.

4. **시간 이동 (temporal shift)**: val(2024-01~06) 분포가 test(2024-07~2025) 분포와 다를 경우 보정 파라미터의 일반화 성능이 저하될 수 있습니다.

5. **절대 확률 ≠ 실제 위험 확률**: 모델 출력 30%는 '이 국가에서 3일 내 분쟁 상승 확률이 정확히 30%'가 아닙니다. 상위 X% 위험 국가 형태의 **순위 기반 표현**이 가장 안전한 소통 방식입니다.

## 6. 구버전 LightGBM+SE 대비 성능 비교

| 지표 | 구버전 raw | updated_mask0_only raw | Δ |
|------|-----------|----------------------|---|
| PR-AUC      | 0.1628  | 0.1741  | +0.0113 ▲ |
| P@top5%     | 0.2254  | 0.2159  | -0.0095 ▼ |
| R@P≥0.10    | 0.6209 | 0.6209 | +0.0000 ─ |
| ECE         | 0.1829 | 0.2771  | +0.0942 ▼ |

updated_mask0_only는 PR-AUC에서 구버전 대비 0.0113 향상됐지만, P@top5%는 -0.0095 소폭 하락했습니다. 전반적인 순위 품질은 개선됐으나 최상위 정밀도는 구버전이 우수합니다. **대회 평가 기준이 PR-AUC 중심이라면 updated_mask0_only 채택을 권장합니다.** ECE는 구버전(0.1829) 대비 updated_mask0_only raw(0.2771)가 악화됐으나, Platt 보정 후 ECE가 개선되는지 확인하세요.

---

## 요약

| 항목 | 수치 |
|------|------|
| 기반 모델 train 행 수 | 182,554 (mask=0 only) |
| val PR-AUC (raw) | 0.1741 |
| val ECE (raw → Platt) | 0.2771 → 0.0029 (-0.2742) |
| val Brier (raw → Platt) | 0.1390 → 0.0365 (-0.1025) |
| 제출 권장 파일 | `predictions__lightgbm_se_updated_mask0_only_platt__byeonghyeon.csv` |
| 대시보드 권장 파일 | `predictions__lightgbm_se_updated_mask0_only_platt__byeonghyeon.csv` |

*보정기는 val 세트로 피팅·평가했으므로 모든 개선 수치는 낙관적 추정입니다.*
*test 세트 레이블은 평가에 사용되지 않았습니다.*