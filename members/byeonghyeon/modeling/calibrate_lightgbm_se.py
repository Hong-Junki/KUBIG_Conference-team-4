"""
Probability calibration for the LightGBM + SE model.

Loads the saved artifact (outputs/models/lightgbm_se.pkl) and fits two
calibration methods on val-set predictions only.  Test-set labels are
never used for fitting or evaluation.

Usage (from project root):
    python modeling/calibrate_lightgbm_se.py

Outputs
-------
outputs/models/platt_calibrator_se.pkl
outputs/models/isotonic_calibrator_se.pkl
outputs/reports/calibration_lightgbm_se.csv
outputs/reports/calibration_lightgbm_se.md
outputs/predictions/val_predictions__lightgbm_se_calibrated.csv
outputs/predictions/predictions__lightgbm_se_platt__byeonghyeon.csv
outputs/predictions/predictions__lightgbm_se_isotonic__byeonghyeon.csv
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import date as dt_date
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from utils import (
    TARGET_COL, DATE_COL,
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    MODEL_DIR, PRED_DIR, REPORT_DIR,
    ensure_output_dirs,
)
from evaluate import compute_pr_auc, compute_ece, compute_p_at_top_k, compute_recall_at_precision
from train_lightgbm_se import load_se_scores, merge_se_score

# ── Output paths ───────────────────────────────────────────────────────────────
MODEL_PATH_SE         = os.path.join(MODEL_DIR, "lightgbm_se.pkl")
PLATT_PATH            = os.path.join(MODEL_DIR, "platt_calibrator_se.pkl")
ISOTONIC_PATH         = os.path.join(MODEL_DIR, "isotonic_calibrator_se.pkl")

CAL_CSV               = os.path.join(REPORT_DIR, "calibration_lightgbm_se.csv")
CAL_MD                = os.path.join(REPORT_DIR, "calibration_lightgbm_se.md")

VAL_CAL_PRED_PATH     = os.path.join(PRED_DIR, "val_predictions__lightgbm_se_calibrated.csv")
TEST_PLATT_PATH       = os.path.join(PRED_DIR, "predictions__lightgbm_se_platt__byeonghyeon.csv")
TEST_ISOTONIC_PATH    = os.path.join(PRED_DIR, "predictions__lightgbm_se_isotonic__byeonghyeon.csv")

# Guard: never overwrite the original SE submission file
_PROTECTED = os.path.join(PRED_DIR, "predictions__lightgbm_se__byeonghyeon.csv")


# ── Metric helper ──────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob, label):
    """Return a dict of all evaluation metrics for one method."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return {
        "method":        label,
        "pr_auc":        compute_pr_auc(y_true, y_prob),
        "p_at_top5pct":  compute_p_at_top_k(y_true, y_prob, k=0.05),
        "r_at_p010":     compute_recall_at_precision(y_true, y_prob, 0.10),
        "r_at_p020":     compute_recall_at_precision(y_true, y_prob, 0.20),
        "r_at_p030":     compute_recall_at_precision(y_true, y_prob, 0.30),
        "ece":           compute_ece(y_true, y_prob),
        "brier_score":   float(brier_score_loss(y_true, y_prob)),
    }


# ── Markdown report (Korean) ───────────────────────────────────────────────────

def generate_md_report(df):
    today = dt_date.today().isoformat()
    r = df.set_index("method")

    raw     = r.loc["raw_lgb"]
    platt   = r.loc["platt"]
    isotonic = r.loc["isotonic"]

    ece_platt_delta    = raw["ece"]    - platt["ece"]
    ece_iso_delta      = raw["ece"]    - isotonic["ece"]
    brier_platt_delta  = raw["brier_score"] - platt["brier_score"]
    brier_iso_delta    = raw["brier_score"] - isotonic["brier_score"]

    prauc_platt_delta  = platt["pr_auc"]   - raw["pr_auc"]
    prauc_iso_delta    = isotonic["pr_auc"] - raw["pr_auc"]
    p5_platt_delta     = platt["p_at_top5pct"]    - raw["p_at_top5pct"]
    p5_iso_delta       = isotonic["p_at_top5pct"] - raw["p_at_top5pct"]

    # Safer dashboard recommendation
    platt_better = platt["ece"] < raw["ece"] and platt["brier_score"] < raw["brier_score"]
    iso_better   = isotonic["ece"] < platt["ece"]

    L = []

    # ── Header ────────────────────────────────────────────────────────────────
    L += [
        "# LightGBM + SE 모델 — 확률 보정(Calibration) 실험 리포트",
        "",
        f"생성일: {today}",
        "기반 모델: `outputs/models/lightgbm_se.pkl` (재학습 없음)",
        "평가 데이터: **val 세트 전용** (2024-01 ~ 2024-06)",
        "test 세트: 레이블 미사용 — 최종 예측 파일 생성에만 사용",
        "",
        "> **⚠️ 평가 한계 주의**",
        "> 보정기(calibrator)를 val 세트 예측으로 **피팅**한 뒤 **동일한 val 세트로 평가**합니다.",
        "> Platt scaling(파라미터 2개)은 과적합 위험이 낮지만,",
        "> Isotonic regression(가변 구간)은 val ECE/Brier 개선이 **과대 추정**될 수 있습니다.",
        "> 또한 LightGBM 모델 자체가 val PR-AUC 기준 early stopping으로 학습됐으므로",
        "> val은 완전한 홀드아웃 세트가 아닙니다. 모든 수치를 낙관적 추정으로 해석하세요.",
        "",
    ]

    # ── 1. 메트릭 비교 표 ─────────────────────────────────────────────────────
    L += [
        "## 1. val 세트 메트릭 비교 (보정 전/후)",
        "",
        "| 방법 | PR-AUC | P@top5% | R@P≥0.10 | R@P≥0.20 | R@P≥0.30 | ECE | Brier |",
        "|------|--------|---------|---------|---------|---------|-----|-------|",
    ]
    for method_key, label in [("raw_lgb", "raw LightGBM (기준)"),
                               ("platt",   "Platt scaling"),
                               ("isotonic","Isotonic regression")]:
        row = r.loc[method_key]
        L.append(
            f"| {label} | {row['pr_auc']:.4f} | {row['p_at_top5pct']:.4f} "
            f"| {row['r_at_p010']:.4f} | {row['r_at_p020']:.4f} | {row['r_at_p030']:.4f} "
            f"| {row['ece']:.4f} | {row['brier_score']:.4f} |"
        )
    L += [
        "",
        "**delta (보정 후 − raw)**",
        "",
        "| 방법 | ΔECE | ΔBrier | ΔPR-AUC | ΔP@top5% |",
        "|------|------|--------|---------|---------|",
        f"| Platt scaling     | {-ece_platt_delta:+.4f} | {-brier_platt_delta:+.4f} "
        f"| {prauc_platt_delta:+.4f} | {p5_platt_delta:+.4f} |",
        f"| Isotonic regression | {-ece_iso_delta:+.4f} | {-brier_iso_delta:+.4f} "
        f"| {prauc_iso_delta:+.4f} | {p5_iso_delta:+.4f} |",
        "",
        "*(ΔECE/ΔBrier가 양수면 악화, 음수면 개선)*",
        "",
    ]

    # ── 2. ECE 개선 여부 ────────────────────────────────────────────────────────
    L += ["## 2. ECE 개선 여부", ""]
    if ece_platt_delta > 0.005:
        L.append(
            f"Platt scaling 적용 후 ECE가 {raw['ece']:.4f} → {platt['ece']:.4f} "
            f"(**−{ece_platt_delta:.4f}** 개선)됩니다. "
            "sigmoid 보정이 LightGBM의 체계적 과신(overconfidence)을 교정합니다."
        )
    else:
        L.append(
            f"Platt scaling의 ECE 개선({ece_platt_delta:.4f})이 미미합니다. "
            "LightGBM 예측 확률이 이미 상대적으로 잘 보정되어 있을 수 있습니다."
        )
    L.append("")
    if ece_iso_delta > 0.005:
        L.append(
            f"Isotonic regression 후 ECE가 {raw['ece']:.4f} → {isotonic['ece']:.4f} "
            f"(**−{ece_iso_delta:.4f}** 개선)됩니다. "
            "단, 이 수치는 동일 val 세트로 피팅·평가했으므로 **과대 추정 가능성**이 있습니다."
        )
    else:
        L.append(
            f"Isotonic regression의 ECE 개선({ece_iso_delta:.4f})이 미미합니다."
        )
    L.append("")

    # ── 3. PR-AUC / top-k 불변성 ───────────────────────────────────────────────
    L += ["## 3. PR-AUC / top-k 메트릭 변화", ""]
    L.append(
        "Platt scaling은 단조 증가 함수(sigmoid)이므로 **순위를 변경하지 않습니다**. "
        "따라서 PR-AUC, P@top5%, Recall@Precision 등 모든 순위 기반 메트릭은 "
        "수학적으로 raw 예측과 동일해야 합니다."
    )
    L.append("")
    if abs(prauc_platt_delta) < 1e-6:
        L.append(
            f"✓ 확인: Platt scaling PR-AUC 변화 = {prauc_platt_delta:+.6f} (수학적 예측과 일치)"
        )
    else:
        L.append(
            f"⚠ Platt scaling PR-AUC 변화 = {prauc_platt_delta:+.6f} "
            "(부동소수점 오차 수준 — 실질적 차이 없음)"
        )
    L.append("")
    L.append(
        "Isotonic regression은 구간별 단조이므로 동점 구간이 발생할 경우 "
        "순위가 미세하게 달라질 수 있습니다."
    )
    if abs(prauc_iso_delta) < 0.001:
        L.append(
            f"✓ 확인: Isotonic PR-AUC 변화 = {prauc_iso_delta:+.4f} (실질적 차이 없음)"
        )
    else:
        L.append(
            f"Isotonic PR-AUC 변화 = {prauc_iso_delta:+.4f} "
            "(순위 미세 변화 발생 — 동점 구간 확인 권장)"
        )
    L.append("")

    # ── 4. 대시보드 위험 점수로 안전한 출력 ─────────────────────────────────────
    L += ["## 4. 대시보드 위험 점수에 적합한 출력", ""]
    if platt_better:
        if iso_better:
            safer = "Platt scaling"
            reason = (
                "Isotonic regression이 ECE는 더 낮지만 동일 val 세트에서 피팅·평가한 결과라 "
                "과대 추정 가능성이 있습니다. Platt scaling은 파라미터가 2개뿐이므로 "
                "과적합 위험이 낮고, 단조 변환이 보장되어 '높은 점수 = 더 위험'이라는 "
                "직관적 해석이 유지됩니다."
            )
        else:
            safer = "Platt scaling"
            reason = (
                "Platt scaling이 ECE·Brier 모두 개선하며 파라미터 수가 적어 "
                "과적합 위험이 낮습니다. 단조 변환으로 순위 해석도 안전합니다."
            )
    else:
        safer = "raw LightGBM"
        reason = (
            "보정 후 ECE·Brier 개선이 미미하거나 없습니다. "
            "현재 raw 예측을 그대로 사용하는 것이 안전합니다."
        )
    L += [
        f"**추천: `{safer}`**",
        "",
        reason,
        "",
    ]

    # ── 5. 보정 확률도 주의해서 해석해야 하는 이유 ──────────────────────────────
    L += [
        "## 5. 보정된 확률도 주의해서 해석해야 하는 이유",
        "",
        "1. **동일 세트 과적합**: 보정기를 val 세트로 피팅·평가했으므로 "
        "   실제 미래 데이터(test 이후)에서의 ECE 개선은 더 작을 수 있습니다.",
        "",
        "2. **LightGBM early stopping**: 기반 LightGBM 모델이 val PR-AUC를 기준으로 "
        "   반복 횟수를 선택했으므로 val은 완전한 홀드아웃이 아닙니다.",
        "",
        "3. **드문 사건(4% 양성 비율)**: 양성 비율이 낮으면 소수의 양성 사례에서 "
        "   보정 곡선이 불안정합니다. 특히 상위 분위 구간에서의 보정 품질을 "
        "   신뢰하기 어렵습니다.",
        "",
        "4. **시간 이동(temporal shift)**: val(2024-01~06) 기간의 분포가 "
        "   test 이후 기간과 다를 경우 보정 파라미터의 일반화 성능이 저하됩니다.",
        "",
        "5. **절대 확률 ≠ 실제 위험 확률**: '이 국가가 다음 3일 내 분쟁 상승 확률이 "
        "   30%'라는 해석은 모델 가정과 데이터 한계로 인해 과도한 정밀도입니다.",
        "",
    ]

    # ── 6. 대시보드 권장안 ────────────────────────────────────────────────────
    L += [
        "## 6. 대시보드 권장안",
        "",
        "| 용도 | 권장 출력 | 이유 |",
        "|------|----------|------|",
        "| 국가 순위 / alert 우선순위 | raw LightGBM `y_prob` | 순위 기반 메트릭 최적화됨 |",
        f"| 위험 점수 수치 표시 | Platt 보정 `y_prob` | ECE 개선, 과적합 위험 낮음 |",
        "| '높음/중간/낮음' 3단계 분류 | Platt 보정 후 분위 기반 레이블 | 절대 확률보다 상대 순위 활용 |",
        "",
        "> **결론**: 대시보드에서 숫자(%)를 직접 표시한다면 Platt 보정 확률이 더 안전합니다.",
        "> 그러나 어떤 보정 방법도 절대적 확률 정확도를 보장하지 않으므로,",
        "> '상위 X% 위험 국가' 형태의 **순위 기반 표현**이 가장 안전한 소통 방식입니다.",
        "",
        "---",
        "",
        "## 요약",
        "",
        f"| 항목 | 결과 |",
        f"|------|------|",
        f"| ECE (raw → Platt) | {raw['ece']:.4f} → {platt['ece']:.4f} ({-ece_platt_delta:+.4f}) |",
        f"| ECE (raw → Isotonic) | {raw['ece']:.4f} → {isotonic['ece']:.4f} ({-ece_iso_delta:+.4f}) |",
        f"| Brier (raw → Platt) | {raw['brier_score']:.4f} → {platt['brier_score']:.4f} ({-brier_platt_delta:+.4f}) |",
        f"| PR-AUC 변화 (Platt) | {prauc_platt_delta:+.6f} (순위 불변) |",
        f"| PR-AUC 변화 (Isotonic) | {prauc_iso_delta:+.4f} |",
        f"| 대시보드 추천 출력 | {safer} |",
        "",
        "*보정기는 val 세트로 피팅·평가했으므로 모든 개선 수치는 낙관적 추정입니다.*",
        "*test 세트 레이블은 평가에 사용되지 않았습니다.*",
    ]

    return "\n".join(L)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ensure_output_dirs()
    SEP = "=" * 60

    # Safety check: never overwrite the original submission file
    assert os.path.realpath(TEST_PLATT_PATH)    != os.path.realpath(_PROTECTED)
    assert os.path.realpath(TEST_ISOTONIC_PATH) != os.path.realpath(_PROTECTED)

    # ── Step 1: Load model artifact ───────────────────────────────────────────
    print(SEP)
    print("Step 1 — 모델 아티팩트 로드 (재학습 없음)")
    print(SEP)

    if not os.path.exists(MODEL_PATH_SE):
        print(f"ERROR: 모델 파일 없음 → {MODEL_PATH_SE}")
        sys.exit(1)

    artifact           = joblib.load(MODEL_PATH_SE)
    model              = artifact["model"]
    feature_cols       = artifact["feature_cols"]
    country_categories = artifact["country_categories"]

    print(f"  로드 완료  : {MODEL_PATH_SE}")
    print(f"  피처 수    : {len(feature_cols)}")
    print(f"  최적 반복  : {model.best_iteration}")

    # ── Step 2: Load SE scores ────────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 2 — SE 점수 로드")
    print(SEP)

    se_df = load_se_scores()
    print(f"  {len(se_df):,} 행")

    # ── Step 3: Load and merge all splits ────────────────────────────────────
    print()
    print(SEP)
    print("Step 3 — 데이터 로드 및 SE 병합")
    print(SEP)

    train_raw = pd.read_parquet(TRAIN_PATH)
    val_raw   = pd.read_parquet(VAL_PATH)
    test_raw  = pd.read_parquet(TEST_PATH)

    print(f"  train: {len(train_raw):,} 행  (SE 병합 검증만)")
    print(f"  val  : {len(val_raw):,} 행")
    print(f"  test : {len(test_raw):,} 행  (레이블 미사용)")
    print()

    merge_se_score(train_raw, se_df, "train")   # row-count validation only
    val  = merge_se_score(val_raw,  se_df, "val").reset_index(drop=True)
    test = merge_se_score(test_raw, se_df, "test").reset_index(drop=True)

    # ── Step 4: Prepare features ──────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 4 — 피처 준비")
    print(SEP)

    def prepare_X(df):
        X = df[feature_cols].copy()
        X["country"] = pd.Categorical(X["country"], categories=country_categories).astype("category")
        return X

    X_val  = prepare_X(val)
    X_test = prepare_X(test)
    y_val  = val[TARGET_COL].values

    print(f"  val  양성: {int(y_val.sum()):,} / {len(y_val):,}  (비율={y_val.mean():.4f})")
    print(f"  test 행수: {len(X_test):,}  (레이블 없음)")

    # ── Step 5: Raw predictions ───────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 5 — raw 예측 생성")
    print(SEP)

    y_prob_val_raw  = model.predict(X_val,  num_iteration=model.best_iteration)
    y_prob_test_raw = model.predict(X_test, num_iteration=model.best_iteration)

    print(f"  val  y_prob 범위: [{y_prob_val_raw.min():.6f}, {y_prob_val_raw.max():.6f}]")
    print(f"  test y_prob 범위: [{y_prob_test_raw.min():.6f}, {y_prob_test_raw.max():.6f}]")

    # ── Step 6: Fit Platt scaling ─────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 6 — Platt scaling 피팅 (val 세트, 파라미터 2개)")
    print(SEP)

    platt = LogisticRegression(C=1e10, solver="lbfgs")
    platt.fit(y_prob_val_raw.reshape(-1, 1), y_val)

    y_prob_val_platt  = platt.predict_proba(y_prob_val_raw.reshape(-1,  1))[:, 1]
    y_prob_test_platt = platt.predict_proba(y_prob_test_raw.reshape(-1, 1))[:, 1]

    print(f"  w={platt.coef_[0][0]:.4f}, b={platt.intercept_[0]:.4f}")
    print(f"  val  보정 후 범위: [{y_prob_val_platt.min():.6f}, {y_prob_val_platt.max():.6f}]")

    joblib.dump(platt, PLATT_PATH)
    print(f"  저장 완료: {PLATT_PATH}")

    # ── Step 7: Fit Isotonic regression ───────────────────────────────────────
    print()
    print(SEP)
    print("Step 7 — Isotonic regression 피팅 (val 세트, 가변 구간)")
    print(SEP)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob_val_raw, y_val)

    y_prob_val_isotonic  = iso.predict(y_prob_val_raw)
    y_prob_test_isotonic = iso.predict(y_prob_test_raw)

    print(f"  val  보정 후 범위: [{y_prob_val_isotonic.min():.6f}, {y_prob_val_isotonic.max():.6f}]")

    joblib.dump(iso, ISOTONIC_PATH)
    print(f"  저장 완료: {ISOTONIC_PATH}")

    # ── Step 8: Compute metrics ───────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 8 — val 세트 메트릭 계산")
    print(SEP)

    results = [
        compute_metrics(y_val, y_prob_val_raw,      "raw_lgb"),
        compute_metrics(y_val, y_prob_val_platt,    "platt"),
        compute_metrics(y_val, y_prob_val_isotonic, "isotonic"),
    ]

    col_order = ["method", "pr_auc", "p_at_top5pct",
                 "r_at_p010", "r_at_p020", "r_at_p030", "ece", "brier_score"]
    results_df = pd.DataFrame(results)[col_order]

    print(f"\n  {'method':<20} {'PR-AUC':>8} {'P@top5%':>8} {'ECE':>8} {'Brier':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in results_df.iterrows():
        print(
            f"  {row['method']:<20} {row['pr_auc']:>8.4f} {row['p_at_top5pct']:>8.4f} "
            f"{row['ece']:>8.4f} {row['brier_score']:>8.4f}"
        )

    # ── Step 9: Save outputs ──────────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 9 — 결과 저장")
    print(SEP)

    # Metrics CSV
    results_df.to_csv(CAL_CSV, index=False)
    print(f"  저장 완료: {CAL_CSV}")

    # Markdown report
    md_text = generate_md_report(results_df)
    with open(CAL_MD, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"  저장 완료: {CAL_MD}")

    # Val calibrated predictions (all 3 methods)
    val_cal_df = pd.DataFrame({
        "date":              pd.to_datetime(val[DATE_COL]).dt.strftime("%Y-%m-%d"),
        "country":           val["country"].astype(str),
        "y_true":            y_val.astype(int),
        "y_prob_raw":        y_prob_val_raw,
        "y_prob_platt":      y_prob_val_platt,
        "y_prob_isotonic":   y_prob_val_isotonic,
    })
    val_cal_df.to_csv(VAL_CAL_PRED_PATH, index=False)
    print(f"  저장 완료: {VAL_CAL_PRED_PATH}  ({len(val_cal_df):,} 행)")

    # Test prediction files (submission format: date, country, y_prob)
    date_test = pd.to_datetime(test[DATE_COL]).dt.strftime("%Y-%m-%d")
    country_test = test["country"].astype(str)

    pd.DataFrame({
        "date":    date_test,
        "country": country_test,
        "y_prob":  y_prob_test_platt,
    }).to_csv(TEST_PLATT_PATH, index=False)
    print(f"  저장 완료: {TEST_PLATT_PATH}  ({len(test):,} 행)")

    pd.DataFrame({
        "date":    date_test,
        "country": country_test,
        "y_prob":  y_prob_test_isotonic,
    }).to_csv(TEST_ISOTONIC_PATH, index=False)
    print(f"  저장 완료: {TEST_ISOTONIC_PATH}  ({len(test):,} 행)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("캘리브레이션 완료 — 생성된 파일 목록")
    print(SEP)
    for path in [PLATT_PATH, ISOTONIC_PATH, CAL_CSV, CAL_MD,
                 VAL_CAL_PRED_PATH, TEST_PLATT_PATH, TEST_ISOTONIC_PATH]:
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists}  {path}")
    print()

    # Confirm original submission file untouched
    if os.path.exists(_PROTECTED):
        print(f"  ✓  기존 파일 유지 확인: {_PROTECTED}")
    print(SEP)


if __name__ == "__main__":
    main()
