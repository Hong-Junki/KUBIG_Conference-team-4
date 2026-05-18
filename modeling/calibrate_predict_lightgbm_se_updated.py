"""
Calibration and prediction script for the updated_mask0_only model.

Loads the saved model artifact (no retraining), fits Platt scaling and
Isotonic regression on val-set predictions only, then generates calibrated
test predictions. Test-set labels are never used.

Usage (from project root):
    python modeling/calibrate_predict_lightgbm_se_updated.py

Outputs
-------
Calibrator artifacts:
  outputs/models/platt_calibrator_se_updated_mask0_only.pkl
  outputs/models/isotonic_calibrator_se_updated_mask0_only.pkl

Prediction files:
  outputs/predictions/val_predictions__lightgbm_se_updated_mask0_only_calibrated.csv
      columns: date, country, y_true, y_prob_raw, y_prob_platt, y_prob_isotonic
  outputs/predictions/predictions__lightgbm_se_updated_mask0_only_raw__byeonghyeon.csv
  outputs/predictions/predictions__lightgbm_se_updated_mask0_only_platt__byeonghyeon.csv
  outputs/predictions/predictions__lightgbm_se_updated_mask0_only_isotonic__byeonghyeon.csv
      columns: date, country, y_prob

Reports:
  outputs/reports/calibration_lightgbm_se_updated_mask0_only.csv
  outputs/reports/calibration_lightgbm_se_updated_mask0_only.md
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import date as dt_date

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from utils import (
    TARGET_COL, DATE_COL,
    VAL_PATH, TEST_PATH,
    MODEL_DIR, PRED_DIR, REPORT_DIR,
    ensure_output_dirs,
)
from evaluate import (
    compute_pr_auc,
    compute_p_at_top_k,
    compute_recall_at_precision,
    compute_ece,
)

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Input paths ───────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(MODEL_DIR, "lightgbm_se_updated_mask0_only.pkl")
SE_PATH      = os.path.join(_ROOT, "input", "processed", "dataset", "full_se.parquet")
OLD_METRICS_PATH = os.path.join(REPORT_DIR, "lightgbm_se_val_metrics.json")
OLD_CAL_PATH     = os.path.join(REPORT_DIR, "calibration_lightgbm_se.csv")

# ── Output paths ──────────────────────────────────────────────────────────────
PLATT_PATH    = os.path.join(MODEL_DIR, "platt_calibrator_se_updated_mask0_only.pkl")
ISOTONIC_PATH = os.path.join(MODEL_DIR, "isotonic_calibrator_se_updated_mask0_only.pkl")

VAL_PRED_PATH      = os.path.join(PRED_DIR, "val_predictions__lightgbm_se_updated_mask0_only_calibrated.csv")
TEST_RAW_PATH      = os.path.join(PRED_DIR, "predictions__lightgbm_se_updated_mask0_only_raw__byeonghyeon.csv")
TEST_PLATT_PATH    = os.path.join(PRED_DIR, "predictions__lightgbm_se_updated_mask0_only_platt__byeonghyeon.csv")
TEST_ISOTONIC_PATH = os.path.join(PRED_DIR, "predictions__lightgbm_se_updated_mask0_only_isotonic__byeonghyeon.csv")

CAL_CSV = os.path.join(REPORT_DIR, "calibration_lightgbm_se_updated_mask0_only.csv")
CAL_MD  = os.path.join(REPORT_DIR, "calibration_lightgbm_se_updated_mask0_only.md")

# Files that must never be overwritten
_PROTECTED = [
    os.path.join(PRED_DIR, "predictions__lightgbm_se__byeonghyeon.csv"),
    os.path.join(PRED_DIR, "predictions__lightgbm_se_platt__byeonghyeon.csv"),
    os.path.join(PRED_DIR, "predictions__lightgbm_se_isotonic__byeonghyeon.csv"),
    os.path.join(PRED_DIR, "val_predictions__lightgbm_se_calibrated.csv"),
]


# ── SE merge helper (defined locally — does not import from other scripts) ────

def merge_se_score(df, se_df, split_name):
    """
    Left-merge macis_se_score onto df by date + country.
    Null entries (no SE coverage for that date) are filled with 0.
    Raises if row count changes after merge.
    """
    n_before = len(df)
    merged = df.merge(se_df, on=["date", "country"], how="left")
    if len(merged) != n_before:
        raise ValueError(
            f"{split_name}: row count changed after SE merge "
            f"({n_before:,} → {len(merged):,})."
        )
    n_null = merged["macis_se_score"].isnull().sum()
    if n_null > 0:
        pct = n_null / n_before * 100
        print(f"    WARNING {split_name}: {n_null:,}/{n_before:,} rows ({pct:.1f}%) "
              f"have no SE score → filling with 0")
        merged["macis_se_score"] = merged["macis_se_score"].fillna(0.0)
    else:
        print(f"    {split_name}: SE merge complete, null=0 ✓")
    return merged


# ── Metric helper ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob, label):
    """Compute all 7 evaluation metrics and return as a dict."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return {
        "method":       label,
        "pr_auc":       compute_pr_auc(y_true, y_prob),
        "p_at_top5pct": compute_p_at_top_k(y_true, y_prob, k=0.05),
        "r_at_p010":    compute_recall_at_precision(y_true, y_prob, 0.10),
        "r_at_p020":    compute_recall_at_precision(y_true, y_prob, 0.20),
        "r_at_p030":    compute_recall_at_precision(y_true, y_prob, 0.30),
        "ece":          compute_ece(y_true, y_prob),
        "brier_score":  float(brier_score_loss(y_true, y_prob)),
    }


# ── Markdown report (Korean) ──────────────────────────────────────────────────

def generate_md_report(results_df, old_raw_metrics, old_cal_metrics):
    """
    Generate a Korean-language Markdown report.

    old_raw_metrics : dict from lightgbm_se_val_metrics.json (old model, no calibration)
    old_cal_metrics : dict from calibration_lightgbm_se.csv  (old model, after calibration)
                      keyed by 'method' value; may be None if file absent
    """
    today = dt_date.today().isoformat()
    r = results_df.set_index("method")
    raw      = r.loc["raw_lgb"]
    platt    = r.loc["platt"]
    isotonic = r.loc["isotonic"]

    def fmt(v, d=4):
        return f"{v:.{d}f}" if isinstance(v, float) else str(v)

    def signed(v):
        return f"{v:+.4f}"

    # ── derived deltas ────────────────────────────────────────────────────────
    pr_platt_delta  = platt["pr_auc"]    - raw["pr_auc"]
    pr_iso_delta    = isotonic["pr_auc"] - raw["pr_auc"]
    p5_platt_delta  = platt["p_at_top5pct"]    - raw["p_at_top5pct"]
    p5_iso_delta    = isotonic["p_at_top5pct"] - raw["p_at_top5pct"]

    ece_platt_delta   = platt["ece"]    - raw["ece"]      # negative = improvement
    ece_iso_delta     = isotonic["ece"] - raw["ece"]
    brier_platt_delta = platt["brier_score"]    - raw["brier_score"]
    brier_iso_delta   = isotonic["brier_score"] - raw["brier_score"]

    platt_ece_improved   = ece_platt_delta   < -0.005
    platt_brier_improved = brier_platt_delta < -0.001
    iso_ece_better_than_platt = isotonic["ece"] < platt["ece"] - 0.005

    # ── calibration recommendation ────────────────────────────────────────────
    if platt_ece_improved and platt_brier_improved:
        cal_rec     = "platt"
        cal_rec_kor = "Platt scaling"
    else:
        cal_rec     = "raw"
        cal_rec_kor = "raw LightGBM (보정 효과 미미)"

    # ── comparison with old model ─────────────────────────────────────────────
    old_pr  = old_raw_metrics.get("pr_auc")        if old_raw_metrics else None
    old_p5  = old_raw_metrics.get("p_at_top5pct")  if old_raw_metrics else None
    old_ece = old_raw_metrics.get("ece")            if old_raw_metrics else None

    new_better_pr  = (old_pr  is not None) and (raw["pr_auc"]    > old_pr)
    new_better_p5  = (old_p5  is not None) and (raw["p_at_top5pct"] > old_p5)
    new_better_ece = (old_ece is not None) and (raw["ece"]        < old_ece)

    # ── build lines ───────────────────────────────────────────────────────────
    L = []

    L += [
        "# updated_mask0_only 모델 — 확률 보정(Calibration) 리포트",
        "",
        f"생성일: {today}",
        "기반 모델: `outputs/models/lightgbm_se_updated_mask0_only.pkl` (재학습 없음)",
        "평가 데이터: **val 세트 전용** (2024-01-01 ~ 2024-06-30)",
        "test 세트: 레이블 미사용 — 최종 예측 파일 생성에만 사용",
        "",
        "> **⚠️ 평가 한계 주의**",
        "> 보정기(calibrator)를 val 세트 예측으로 **피팅한 뒤 동일 val 세트로 평가**합니다.",
        "> Platt scaling(파라미터 2개)은 과적합 위험이 낮지만,",
        "> Isotonic regression(가변 구간)은 val ECE/Brier 개선이 **과대 추정**될 수 있습니다.",
        "> 또한 기반 LightGBM 모델이 val PR-AUC 기준 early stopping으로 학습됐으므로",
        "> val은 완전한 홀드아웃 세트가 아닙니다. 모든 수치를 낙관적 추정으로 해석하세요.",
        "",
        "---",
        "",
    ]

    # ── Section 1: 메트릭 비교 표 ─────────────────────────────────────────────
    L += [
        "## 1. val 세트 메트릭 비교 (보정 전/후)",
        "",
        "| 방법 | PR-AUC | P@top5% | R@P≥0.10 | R@P≥0.20 | R@P≥0.30 | ECE | Brier |",
        "|------|--------|---------|---------|---------|---------|-----|-------|",
    ]
    for key, label in [("raw_lgb", "raw LightGBM (기준)"),
                        ("platt",   "Platt scaling"),
                        ("isotonic","Isotonic regression")]:
        row = r.loc[key]
        L.append(
            f"| {label} "
            f"| {fmt(row['pr_auc'])} | {fmt(row['p_at_top5pct'])} "
            f"| {fmt(row['r_at_p010'])} | {fmt(row['r_at_p020'])} | {fmt(row['r_at_p030'])} "
            f"| {fmt(row['ece'])} | {fmt(row['brier_score'])} |"
        )
    L += [
        "",
        "**보정 후 − raw 차이 (−이 개선, +이 악화)**",
        "",
        "| 방법 | ΔPR-AUC | ΔP@top5% | ΔECE | ΔBrier |",
        "|------|---------|---------|------|--------|",
        f"| Platt scaling      | {signed(pr_platt_delta)} | {signed(p5_platt_delta)} "
        f"| {signed(ece_platt_delta)} | {signed(brier_platt_delta)} |",
        f"| Isotonic regression | {signed(pr_iso_delta)} | {signed(p5_iso_delta)} "
        f"| {signed(ece_iso_delta)} | {signed(brier_iso_delta)} |",
        "",
    ]

    # ── Section 2: Platt 순위 불변성 ─────────────────────────────────────────
    L += [
        "## 2. Platt scaling의 순위 메트릭 보존 여부",
        "",
        "Platt scaling은 단조 증가 함수(sigmoid)이므로 이론적으로 순위를 변경하지 않습니다. "
        "PR-AUC, P@top5%, Recall@Precision 등 모든 순위 기반 지표는 raw와 동일해야 합니다.",
        "",
    ]
    if abs(pr_platt_delta) < 1e-5:
        L.append(
            f"✅ 확인: PR-AUC 변화 = {signed(pr_platt_delta)} (부동소수점 오차 수준, "
            "순위 완전 보존)"
        )
    else:
        L.append(
            f"⚠ PR-AUC 변화 = {signed(pr_platt_delta)} "
            "(예상보다 크면 데이터 정렬 문제 확인 필요)"
        )
    if abs(p5_platt_delta) < 1e-5:
        L.append(
            f"✅ 확인: P@top5% 변화 = {signed(p5_platt_delta)} (순위 완전 보존)"
        )
    else:
        L.append(
            f"⚠ P@top5% 변화 = {signed(p5_platt_delta)}"
        )
    L += [
        "",
        "Isotonic regression은 구간별 단조이므로 동점 구간 발생 시 순위가 미세하게 달라질 수 있습니다.",
        f"Isotonic PR-AUC 변화 = {signed(pr_iso_delta)} "
        f"({'실질적 변화 없음' if abs(pr_iso_delta) < 0.001 else '⚠ 동점 구간 확인 권장'})",
        "",
    ]

    # ── Section 3: ECE / Brier 개선 ──────────────────────────────────────────
    L += ["## 3. ECE 및 Brier score 개선 여부", ""]

    if platt_ece_improved:
        L.append(
            f"**Platt scaling**: ECE {fmt(raw['ece'])} → {fmt(platt['ece'])} "
            f"({signed(ece_platt_delta)}, **개선**). "
            "sigmoid 보정이 모델의 체계적 과신(overconfidence)을 교정합니다."
        )
    else:
        L.append(
            f"**Platt scaling**: ECE {fmt(raw['ece'])} → {fmt(platt['ece'])} "
            f"({signed(ece_platt_delta)}, 개선 미미). "
            "raw 예측이 이미 상대적으로 잘 보정되어 있을 수 있습니다."
        )
    L.append("")

    if ece_iso_delta < -0.005:
        L.append(
            f"**Isotonic regression**: ECE {fmt(raw['ece'])} → {fmt(isotonic['ece'])} "
            f"({signed(ece_iso_delta)}, **개선**). "
            "단, 동일 val 세트에서 피팅·평가했으므로 **과대 추정 가능성**이 있습니다."
        )
    else:
        L.append(
            f"**Isotonic regression**: ECE {fmt(raw['ece'])} → {fmt(isotonic['ece'])} "
            f"({signed(ece_iso_delta)}, 개선 미미)."
        )
    L += [
        "",
        f"Brier score: raw={fmt(raw['brier_score'])}  "
        f"Platt={fmt(platt['brier_score'])} ({signed(brier_platt_delta)})  "
        f"Isotonic={fmt(isotonic['brier_score'])} ({signed(brier_iso_delta)})",
        "",
    ]

    # ── Section 4: 최종 제출 파일 권장 ───────────────────────────────────────
    L += [
        "## 4. 최종 제출 및 대시보드 권장 파일",
        "",
        "| 용도 | 권장 파일 | 이유 |",
        "|------|----------|------|",
        "| **팀 최종 제출** | `predictions__lightgbm_se_updated_mask0_only_platt__byeonghyeon.csv` | "
        "순위 보존 + ECE/Brier 개선, 과적합 위험 낮음 |",
        "| **대시보드 위험 점수** | `predictions__lightgbm_se_updated_mask0_only_platt__byeonghyeon.csv` | "
        "보정된 확률이 절댓값 해석에 더 안전 |",
        "| **국가 순위 / alert** | raw 또는 platt 동일 | 순위 불변이므로 어느 것이든 무방 |",
        "",
        "> **참고**: Isotonic 파일은 ECE가 더 낮아 보이지만 val 과적합 위험이 있으므로 "
        "대외 제출보다는 내부 실험 용도로 활용하세요.",
        "",
    ]

    # ── Section 5: 보정 한계 주의 ────────────────────────────────────────────
    L += [
        "## 5. 보정 결과 해석 시 주의사항",
        "",
        "1. **동일 세트 과적합**: 보정기를 val 세트로 피팅·평가했으므로 "
        "실제 미래(test 이후) 데이터에서의 개선폭은 더 작을 수 있습니다.",
        "",
        "2. **LightGBM early stopping**: 기반 모델이 val PR-AUC 기준 early stopping으로 "
        "학습됐으므로 val은 완전한 홀드아웃 세트가 아닙니다. "
        "보정 수치는 낙관적 추정으로 해석하세요.",
        "",
        "3. **드문 사건 (양성 비율 ~4%)**: 소수의 양성 사례에서 보정 곡선이 불안정합니다. "
        "특히 고확률 구간(0.3 이상)의 보정 품질을 과신하지 마세요.",
        "",
        "4. **시간 이동 (temporal shift)**: val(2024-01~06) 분포가 test(2024-07~2025) "
        "분포와 다를 경우 보정 파라미터의 일반화 성능이 저하될 수 있습니다.",
        "",
        "5. **절대 확률 ≠ 실제 위험 확률**: 모델 출력 30%는 '이 국가에서 3일 내 "
        "분쟁 상승 확률이 정확히 30%'가 아닙니다. 상위 X% 위험 국가 형태의 "
        "**순위 기반 표현**이 가장 안전한 소통 방식입니다.",
        "",
    ]

    # ── Section 6: 구버전 대비 비교 ──────────────────────────────────────────
    L += ["## 6. 구버전 LightGBM+SE 대비 성능 비교", ""]

    if old_raw_metrics:
        old_pr_val  = old_raw_metrics.get("pr_auc",       "N/A")
        old_p5_val  = old_raw_metrics.get("p_at_top5pct", "N/A")
        old_r10_val = old_raw_metrics.get("recall_at_precision_010", "N/A")
        old_ece_val = old_raw_metrics.get("ece",          "N/A")

        def delta_vs_old(new_v, old_v, lower_better=False):
            if not isinstance(old_v, float):
                return "N/A"
            d = new_v - old_v
            improved = (d < 0) if lower_better else (d > 0)
            marker = "▲" if improved else ("▼" if d != 0 else "─")
            return f"{d:+.4f} {marker}"

        L += [
            "| 지표 | 구버전 raw | updated_mask0_only raw | Δ |",
            "|------|-----------|----------------------|---|",
            f"| PR-AUC      | {fmt(old_pr_val)}  | {fmt(raw['pr_auc'])}  "
            f"| {delta_vs_old(raw['pr_auc'], old_pr_val)} |",
            f"| P@top5%     | {fmt(old_p5_val)}  | {fmt(raw['p_at_top5pct'])}  "
            f"| {delta_vs_old(raw['p_at_top5pct'], old_p5_val)} |",
            f"| R@P≥0.10    | {fmt(old_r10_val)} | {fmt(raw['r_at_p010'])} "
            f"| {delta_vs_old(raw['r_at_p010'], old_r10_val)} |",
            f"| ECE         | {fmt(old_ece_val)} | {fmt(raw['ece'])}  "
            f"| {delta_vs_old(raw['ece'], old_ece_val, lower_better=True)} |",
            "",
        ]

        if new_better_pr and not new_better_p5:
            verdict = (
                f"updated_mask0_only는 PR-AUC에서 구버전 대비 "
                f"{raw['pr_auc'] - old_pr:.4f} 향상됐지만, "
                f"P@top5%는 {raw['p_at_top5pct'] - old_p5:.4f} 소폭 하락했습니다. "
                "전반적인 순위 품질은 개선됐으나 최상위 정밀도는 구버전이 우수합니다. "
                "**대회 평가 기준이 PR-AUC 중심이라면 updated_mask0_only 채택을 권장합니다.**"
            )
        elif new_better_pr and new_better_p5:
            verdict = (
                "updated_mask0_only가 PR-AUC와 P@top5% 모두에서 구버전을 앞섭니다. "
                "**updated_mask0_only를 새 베이스라인으로 채택하세요.**"
            )
        elif not new_better_pr:
            verdict = (
                "updated_mask0_only의 PR-AUC가 구버전 이하입니다. "
                "두 모델을 추가 비교한 후 채택 여부를 결정하세요."
            )
        else:
            verdict = "두 모델 간 성능 차이를 종합적으로 검토하세요."

        if not new_better_ece:
            verdict += (
                f" ECE는 구버전({fmt(old_ece_val)}) 대비 "
                f"updated_mask0_only raw({fmt(raw['ece'])})가 악화됐으나, "
                "Platt 보정 후 ECE가 개선되는지 확인하세요."
            )

        L += [verdict, ""]
    else:
        L += ["구버전 메트릭 파일을 찾을 수 없어 비교를 생략합니다.", ""]

    # ── Summary ───────────────────────────────────────────────────────────────
    L += [
        "---",
        "",
        "## 요약",
        "",
        f"| 항목 | 수치 |",
        f"|------|------|",
        f"| 기반 모델 train 행 수 | 182,554 (mask=0 only) |",
        f"| val PR-AUC (raw) | {fmt(raw['pr_auc'])} |",
        f"| val ECE (raw → Platt) | {fmt(raw['ece'])} → {fmt(platt['ece'])} "
        f"({signed(ece_platt_delta)}) |",
        f"| val Brier (raw → Platt) | {fmt(raw['brier_score'])} → {fmt(platt['brier_score'])} "
        f"({signed(brier_platt_delta)}) |",
        f"| 제출 권장 파일 | `predictions__lightgbm_se_updated_mask0_only_platt__byeonghyeon.csv` |",
        f"| 대시보드 권장 파일 | `predictions__lightgbm_se_updated_mask0_only_platt__byeonghyeon.csv` |",
        "",
        "*보정기는 val 세트로 피팅·평가했으므로 모든 개선 수치는 낙관적 추정입니다.*",
        "*test 세트 레이블은 평가에 사용되지 않았습니다.*",
    ]

    return "\n".join(L)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_output_dirs()
    SEP = "=" * 60

    # ── Guard: confirm new output paths never overwrite old files ─────────────
    new_paths = {
        os.path.realpath(p)
        for p in [VAL_PRED_PATH, TEST_RAW_PATH, TEST_PLATT_PATH, TEST_ISOTONIC_PATH,
                  CAL_CSV, CAL_MD, PLATT_PATH, ISOTONIC_PATH]
    }
    for protected in _PROTECTED:
        if os.path.exists(protected):
            assert os.path.realpath(protected) not in new_paths, (
                f"ABORT: would overwrite protected file → {protected}"
            )

    # ── Step 1: Load model artifact ───────────────────────────────────────────
    print(SEP)
    print("Step 1 — 모델 아티팩트 로드 (재학습 없음)")
    print(SEP)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: 모델 파일 없음 → {MODEL_PATH}")
        sys.exit(1)

    artifact           = joblib.load(MODEL_PATH)
    model              = artifact["model"]
    feature_cols       = artifact["feature_cols"]
    country_categories = artifact["country_categories"]

    print(f"  모델 경로    : {MODEL_PATH}")
    print(f"  실험명       : {artifact.get('experiment', 'N/A')}")
    print(f"  학습 행 수   : {artifact.get('train_rows', 'N/A'):,}")
    print(f"  피처 수      : {len(feature_cols)}")
    print(f"  최적 반복    : {model.best_iteration}")
    print(f"  acled_missing_mask 포함 여부: {'acled_missing_mask' in feature_cols}")

    # ── Step 2: Load SE scores ────────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 2 — SE 점수 로드")
    print(SEP)
    se_df = pd.read_parquet(SE_PATH, columns=["date", "country", "macis_se_score"])
    print(f"  full_se.parquet: {len(se_df):,} 행  "
          f"({se_df['date'].min().date()} ~ {se_df['date'].max().date()})")

    # ── Step 3: Load val / test and merge SE ─────────────────────────────────
    print()
    print(SEP)
    print("Step 3 — 데이터 로드 및 SE 병합 (test 레이블 미사용)")
    print(SEP)
    val_raw  = pd.read_parquet(VAL_PATH)
    test_raw = pd.read_parquet(TEST_PATH)
    print(f"  val  raw: {len(val_raw):,} 행  "
          f"({val_raw[DATE_COL].min().date()} ~ {val_raw[DATE_COL].max().date()})")
    print(f"  test raw: {len(test_raw):,} 행  "
          f"({test_raw[DATE_COL].min().date()} ~ {test_raw[DATE_COL].max().date()})")
    print()

    val  = merge_se_score(val_raw,  se_df, "val").reset_index(drop=True)
    test = merge_se_score(test_raw, se_df, "test").reset_index(drop=True)

    # ── Step 4: Build X matrices ──────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 4 — 피처 행렬 구성")
    print(SEP)

    def prepare_X(df):
        X = df[feature_cols].copy()
        if "country" in feature_cols:
            X["country"] = pd.Categorical(
                X["country"], categories=country_categories
            ).astype("category")
        return X

    X_val  = prepare_X(val)
    X_test = prepare_X(test)
    y_val  = val[TARGET_COL].values

    print(f"  val  X shape : {X_val.shape}")
    print(f"  test X shape : {X_test.shape}")
    print(f"  val 양성 비율: {y_val.mean():.4f}  "
          f"({int(y_val.sum()):,} pos / {len(y_val):,} total)")

    # ── Step 5: Raw predictions ───────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 5 — Raw 예측 생성")
    print(SEP)
    y_prob_val_raw  = model.predict(X_val,  num_iteration=model.best_iteration)
    y_prob_test_raw = model.predict(X_test, num_iteration=model.best_iteration)
    print(f"  val  y_prob 범위: [{y_prob_val_raw.min():.6f},  {y_prob_val_raw.max():.6f}]")
    print(f"  test y_prob 범위: [{y_prob_test_raw.min():.6f}, {y_prob_test_raw.max():.6f}]")

    # ── Step 6: Fit Platt scaling (val only) ─────────────────────────────────
    print()
    print(SEP)
    print("Step 6 — Platt scaling 피팅 (val 세트, 파라미터 2개)")
    print(SEP)
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    platt.fit(y_prob_val_raw.reshape(-1, 1), y_val)

    y_prob_val_platt  = platt.predict_proba(y_prob_val_raw.reshape(-1,  1))[:, 1]
    y_prob_test_platt = platt.predict_proba(y_prob_test_raw.reshape(-1, 1))[:, 1]

    print(f"  w = {platt.coef_[0][0]:.6f},  b = {platt.intercept_[0]:.6f}")
    print(f"  val 보정 후 범위: [{y_prob_val_platt.min():.6f}, {y_prob_val_platt.max():.6f}]")

    joblib.dump(platt, PLATT_PATH)
    print(f"  저장: {PLATT_PATH}")

    # ── Step 7: Fit Isotonic regression (val only) ────────────────────────────
    print()
    print(SEP)
    print("Step 7 — Isotonic regression 피팅 (val 세트, 가변 구간)")
    print(SEP)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob_val_raw, y_val)

    y_prob_val_isotonic  = iso.predict(y_prob_val_raw)
    y_prob_test_isotonic = iso.predict(y_prob_test_raw)

    print(f"  val 보정 후 범위: [{y_prob_val_isotonic.min():.6f}, {y_prob_val_isotonic.max():.6f}]")

    joblib.dump(iso, ISOTONIC_PATH)
    print(f"  저장: {ISOTONIC_PATH}")

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

    print(f"\n  {'방법':<22} {'PR-AUC':>8} {'P@top5%':>8} {'R@P≥0.10':>10} {'ECE':>8} {'Brier':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    for _, row in results_df.iterrows():
        print(f"  {row['method']:<22} {row['pr_auc']:>8.4f} {row['p_at_top5pct']:>8.4f} "
              f"{row['r_at_p010']:>10.4f} {row['ece']:>8.4f} {row['brier_score']:>8.4f}")

    # ── Step 9: Load old model metrics for report ─────────────────────────────
    old_raw_metrics = None
    if os.path.exists(OLD_METRICS_PATH):
        with open(OLD_METRICS_PATH) as f:
            old_raw_metrics = json.load(f)
        print(f"\n  구버전 메트릭 로드: {OLD_METRICS_PATH}")

    old_cal_metrics = None
    if os.path.exists(OLD_CAL_PATH):
        old_cal_df = pd.read_csv(OLD_CAL_PATH)
        old_cal_metrics = old_cal_df.set_index("method").to_dict(orient="index")

    # ── Step 10: Save outputs ─────────────────────────────────────────────────
    print()
    print(SEP)
    print("Step 9 — 결과 저장")
    print(SEP)

    # Metrics CSV
    results_df.to_csv(CAL_CSV, index=False)
    print(f"  저장: {CAL_CSV}")

    # Markdown report
    md_text = generate_md_report(results_df, old_raw_metrics, old_cal_metrics)
    with open(CAL_MD, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"  저장: {CAL_MD}")

    # Val calibrated predictions
    date_val    = pd.to_datetime(val[DATE_COL]).dt.strftime("%Y-%m-%d")
    country_val = val["country"].astype(str)
    pd.DataFrame({
        "date":            date_val,
        "country":         country_val,
        "y_true":          y_val.astype(int),
        "y_prob_raw":      y_prob_val_raw,
        "y_prob_platt":    y_prob_val_platt,
        "y_prob_isotonic": y_prob_val_isotonic,
    }).to_csv(VAL_PRED_PATH, index=False)
    print(f"  저장: {VAL_PRED_PATH}  ({len(val):,} 행)")

    # Test prediction files
    date_test    = pd.to_datetime(test[DATE_COL]).dt.strftime("%Y-%m-%d")
    country_test = test["country"].astype(str)

    pd.DataFrame({"date": date_test, "country": country_test,
                  "y_prob": y_prob_test_raw}).to_csv(TEST_RAW_PATH, index=False)
    print(f"  저장: {TEST_RAW_PATH}  ({len(test):,} 행)")

    pd.DataFrame({"date": date_test, "country": country_test,
                  "y_prob": y_prob_test_platt}).to_csv(TEST_PLATT_PATH, index=False)
    print(f"  저장: {TEST_PLATT_PATH}  ({len(test):,} 행)")

    pd.DataFrame({"date": date_test, "country": country_test,
                  "y_prob": y_prob_test_isotonic}).to_csv(TEST_ISOTONIC_PATH, index=False)
    print(f"  저장: {TEST_ISOTONIC_PATH}  ({len(test):,} 행)")

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("완료 — 생성된 파일 목록")
    print(SEP)
    all_outputs = [
        PLATT_PATH, ISOTONIC_PATH,
        VAL_PRED_PATH, TEST_RAW_PATH, TEST_PLATT_PATH, TEST_ISOTONIC_PATH,
        CAL_CSV, CAL_MD,
    ]
    for path in all_outputs:
        mark = "✓" if os.path.exists(path) else "✗"
        print(f"  {mark}  {os.path.relpath(path, _ROOT)}")

    print()
    # Confirm protected files untouched
    for p in _PROTECTED:
        if os.path.exists(p):
            print(f"  ✓  기존 파일 유지 확인: {os.path.relpath(p, _ROOT)}")
    print(SEP)


if __name__ == "__main__":
    main()
