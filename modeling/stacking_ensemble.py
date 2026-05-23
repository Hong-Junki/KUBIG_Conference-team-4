"""
Category D — Stacking Ensemble Scaffold
========================================
담당: D (앙상블 + 캘리브레이션)
참고: docs/model-study.md §4, docs/model-plan.md §2-7

이 스크립트의 역할
──────────────────
  1. 각 base 모델(A/B/C 담당)이 생성한 예측 파일의 존재 여부와 포맷을 검증한다.
  2. OOF train 예측 파일이 없거나 포맷이 틀리면 한국어 체크리스트를 출력하고 안전하게 종료한다.
  3. 모든 파일이 준비되면 Level-1 meta-learner(Logistic Regression) 학습 → val 평가 →
     test 앙상블 예측 생성 → Platt/Isotonic 캘리브레이션으로 이어진다.

현재 상태 (스캐폴드)
─────────────────────
  파일 검증 + 안전 종료 로직만 구현되어 있다.
  실제 학습/예측/저장은 base 파일이 모두 준비된 후 활성화한다.

누수 방지 원칙 (docs/model-study.md §4-1)
──────────────────────────────────────────
  - base 모델의 train 예측은 반드시 OOF(Out-of-Fold) 방식이어야 한다.
  - 일반 train 예측(학습 데이터 전체를 학습 후 같은 데이터로 예측)은 사용 금지.
  - test 예측은 meta-learner 학습에 절대 사용하지 않는다.

Usage (from project root):
    python modeling/stacking_ensemble.py

Outputs (파일이 모두 준비된 후 활성화)
──────────────────────────────────────
  outputs/predictions/val_predictions__stacking__byeonghyeon.csv
      columns: date, country, y_true, y_prob
  outputs/predictions/predictions__stacking__byeonghyeon.csv
      columns: date, country, y_prob
  outputs/reports/stacking_val_metrics.json
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Expected row counts (docs/model-plan.md §0) ───────────────────────────────
EXPECTED_OOF_ROWS  = 211_816   # train 전체
EXPECTED_VAL_ROWS  = 10_556    # val 셋
EXPECTED_TEST_ROWS = 15_718    # test 셋

# ── Required columns by file type ─────────────────────────────────────────────
OOF_COLS  = {"date", "country", "y_true", "y_prob_oof"}
VAL_COLS  = {"date", "country", "y_true", "y_prob"}
TEST_COLS = {"date", "country", "y_prob"}

# ── Base model registry ───────────────────────────────────────────────────────
# 각 항목: (model_name, owner, priority)
# 파일이 준비되면 이 목록에 추가한다.
# 파일 경로 규칙:
#   OOF  → outputs/predictions/oof_predictions__{model_name}__{owner}.csv
#   Val  → outputs/predictions/val_predictions__{model_name}__{owner}.csv
#   Test → outputs/predictions/predictions__{model_name}__{owner}.csv
BASE_MODELS = [
    # (model_name,              owner,          설명)
    ("lightgbm_se",             "byeonghyeon",  "B: LightGBM + SE 피처"),
    ("xgboost",                 "byeonghyeon",  "B: XGBoost"),
    ("lstm_classifier",         "담당자C이름",  "C: LSTM Classifier"),
    # ("logreg",                "담당자A이름",  "A: Logistic Regression"),  # 선택사항
]

# ── Output paths ──────────────────────────────────────────────────────────────
PRED_DIR   = os.path.join(_ROOT, "outputs", "predictions")
REPORT_DIR = os.path.join(_ROOT, "outputs", "reports")


# ── Helper: build file path ───────────────────────────────────────────────────

def _oof_path(model_name: str, owner: str) -> str:
    return os.path.join(PRED_DIR, f"oof_predictions__{model_name}__{owner}.csv")


def _val_path(model_name: str, owner: str) -> str:
    return os.path.join(PRED_DIR, f"val_predictions__{model_name}__{owner}.csv")


def _test_path(model_name: str, owner: str) -> str:
    return os.path.join(PRED_DIR, f"predictions__{model_name}__{owner}.csv")


# ── Validation functions ───────────────────────────────────────────────────────

def _check_file(path: str, required_cols: set, expected_rows: int, label: str) -> list[str]:
    """
    파일 하나를 검사한다. 문제가 있으면 오류 문자열 리스트를 반환한다.
    """
    errors = []

    if not os.path.exists(path):
        errors.append(f"파일 없음: {os.path.relpath(path, _ROOT)}")
        return errors  # 파일이 없으면 나머지 검사 불필요

    try:
        df = pd.read_csv(path, nrows=0)  # 컬럼만 읽어 빠르게 확인
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            errors.append(
                f"[{label}] 필수 컬럼 누락: {sorted(missing_cols)}"
                f"  (파일: {os.path.relpath(path, _ROOT)})"
            )
    except Exception as e:
        errors.append(f"[{label}] CSV 읽기 오류: {e}  (파일: {os.path.relpath(path, _ROOT)})")
        return errors

    # 행 수 확인 (전체 로드)
    try:
        df_full = pd.read_csv(path, usecols=["date", "country"])
        actual_rows = len(df_full)
        if actual_rows != expected_rows:
            errors.append(
                f"[{label}] 행 수 불일치: 예상 {expected_rows:,}행, 실제 {actual_rows:,}행"
                f"  (파일: {os.path.relpath(path, _ROOT)})"
            )
    except Exception as e:
        errors.append(f"[{label}] 행 수 확인 오류: {e}")

    return errors


def _check_merge_keys(paths_and_labels: list[tuple[str, str]]) -> list[str]:
    """
    여러 파일 간에 (date, country) 키 집합이 동일한지 확인한다.
    paths_and_labels: [(path, label), ...]
    """
    errors = []
    key_sets = {}

    for path, label in paths_and_labels:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, usecols=["date", "country"])
            key_sets[label] = set(zip(df["date"].astype(str), df["country"].astype(str)))
        except Exception as e:
            errors.append(f"[{label}] 키 확인 오류: {e}")

    if len(key_sets) < 2:
        return errors  # 비교할 파일이 충분하지 않음

    labels = list(key_sets.keys())
    ref_label, ref_keys = labels[0], key_sets[labels[0]]

    for label in labels[1:]:
        keys = key_sets[label]
        only_in_ref = ref_keys - keys
        only_in_other = keys - ref_keys
        if only_in_ref or only_in_other:
            errors.append(
                f"merge 키 불일치 [{ref_label}] vs [{label}]: "
                f"ref에만 있는 키 {len(only_in_ref)}개, "
                f"other에만 있는 키 {len(only_in_other)}개"
            )

    return errors


def _leakage_check(model_name: str, owner: str) -> list[str]:
    """
    OOF 대신 일반 train 예측 파일이 잘못 제출되지 않았는지 경고한다.
    일반 train 예측(oof_predictions__ 아닌 predictions__로 된 train 크기 파일)을 감지하면 경고.
    """
    warnings = []
    # 혹시 predictions__{model}__{owner}.csv 파일이 EXPECTED_OOF_ROWS 크기이면 의심
    test_path = _test_path(model_name, owner)
    if os.path.exists(test_path):
        try:
            n = len(pd.read_csv(test_path, usecols=["date"]))
            if n == EXPECTED_OOF_ROWS:
                warnings.append(
                    f"[누수 경고] predictions__{model_name}__{owner}.csv 가 "
                    f"{EXPECTED_OOF_ROWS:,}행 (train 크기)입니다. "
                    f"OOF 파일이 아닌 일반 train 예측이 잘못 제출된 것은 아닌지 확인하세요."
                )
        except Exception:
            pass
    return warnings


# ── Main validation runner ─────────────────────────────────────────────────────

def validate_all_inputs() -> tuple[bool, list[str], list[str]]:
    """
    모든 base 모델 파일을 검사한다.
    Returns (all_ok, errors, warnings).
    """
    all_errors: list[str] = []
    all_warnings: list[str] = []

    oof_paths_for_merge  = []
    val_paths_for_merge  = []
    test_paths_for_merge = []

    for model_name, owner, description in BASE_MODELS:
        oof_p  = _oof_path(model_name, owner)
        val_p  = _val_path(model_name, owner)
        test_p = _test_path(model_name, owner)

        label_prefix = f"{description} ({model_name}__{owner})"

        # OOF 검사
        errs = _check_file(oof_p, OOF_COLS, EXPECTED_OOF_ROWS, f"OOF [{label_prefix}]")
        all_errors.extend(errs)

        # Val 검사
        errs = _check_file(val_p, VAL_COLS, EXPECTED_VAL_ROWS, f"Val [{label_prefix}]")
        all_errors.extend(errs)

        # Test 검사
        errs = _check_file(test_p, TEST_COLS, EXPECTED_TEST_ROWS, f"Test [{label_prefix}]")
        all_errors.extend(errs)

        # 누수 경고
        warns = _leakage_check(model_name, owner)
        all_warnings.extend(warns)

        oof_paths_for_merge.append((oof_p,  f"OOF [{model_name}]"))
        val_paths_for_merge.append((val_p,  f"Val [{model_name}]"))
        test_paths_for_merge.append((test_p, f"Test [{model_name}]"))

    # Merge 키 일관성 검사 (각 split별로)
    all_errors.extend(_check_merge_keys(oof_paths_for_merge))
    all_errors.extend(_check_merge_keys(val_paths_for_merge))
    all_errors.extend(_check_merge_keys(test_paths_for_merge))

    return (len(all_errors) == 0), all_errors, all_warnings


# ── Korean checklist printer ───────────────────────────────────────────────────

def _print_missing_checklist(errors: list[str], warnings: list[str]) -> None:
    separator = "=" * 70

    print()
    print(separator)
    print("  [D 담당] Stacking Ensemble — 입력 파일 준비 체크리스트")
    print(separator)
    print()
    print("  ▶ 현재 상태: 필수 파일이 준비되지 않아 학습을 시작할 수 없습니다.")
    print("  ▶ 아래 항목을 팀원에게 요청하거나 직접 준비하세요.")
    print()

    print("  [ 오류 목록 ]")
    for i, err in enumerate(errors, 1):
        print(f"    {i:2d}. {err}")

    if warnings:
        print()
        print("  [ 경고 목록 ]")
        for w in warnings:
            print(f"    ⚠  {w}")

    print()
    print(separator)
    print("  [ 파일 포맷 규칙 ]")
    print()
    print("  ① OOF 예측 파일  (base 모델별 필수 — 누수 방지 핵심)")
    print("     경로: outputs/predictions/oof_predictions__{model_name}__{owner}.csv")
    print("     컬럼: date, country, y_true, y_prob_oof")
    print("     행수: 211,816 (train 전체)")
    print("     생성법: TimeSeriesSplit 또는 walk-forward k-fold로")
    print("             각 fold의 val 구간 예측을 순서대로 쌓아서 만들 것.")
    print("     ⚠ 주의: 학습 데이터 전체를 다시 예측한 일반 train 예측은 사용 불가.")
    print()
    print("  ② Val 예측 파일  (meta 평가 + calibration용)")
    print("     경로: outputs/predictions/val_predictions__{model_name}__{owner}.csv")
    print("     컬럼: date, country, y_true, y_prob")
    print("     행수: 10,556")
    print()
    print("  ③ Test 예측 파일  (최종 앙상블 제출용)")
    print("     경로: outputs/predictions/predictions__{model_name}__{owner}.csv")
    print("     컬럼: date, country, y_prob")
    print("     행수: 15,718")
    print("     ⚠ 주의: test 예측은 meta-learner 학습에 절대 사용하지 않음.")
    print()
    print(separator)
    print("  [ 현재 등록된 base 모델 — 파일을 요청해야 할 담당자 ]")
    print()
    for model_name, owner, description in BASE_MODELS:
        print(f"    • {description}")
        print(f"      OOF : outputs/predictions/oof_predictions__{model_name}__{owner}.csv")
        print(f"      Val : outputs/predictions/val_predictions__{model_name}__{owner}.csv")
        print(f"      Test: outputs/predictions/predictions__{model_name}__{owner}.csv")
        print()
    print(separator)
    print()


# ── Stacking training (활성화 대기 중) ────────────────────────────────────────

def run_stacking():
    """
    모든 파일이 준비된 후 활성화할 학습 로직.
    현재는 파일 검증 통과 후 안내 메시지만 출력한다.

    TODO (파일 준비 완료 후 구현):
      1. OOF 파일 로드 → date, country 기준 merge → 211,816 × n_models 행렬 구성
      2. train.parquet에서 y_true 로드 및 merge 키 정합성 확인
      3. Level-1 LogisticRegression(class_weight='balanced') 학습
      4. Val 파일 로드 → meta-learner predict_proba → val PR-AUC, ECE, Brier 계산
      5. Test 파일 로드 → 앙상블 예측 생성
      6. validate_prediction_file로 포맷 검증 → 저장
         (outputs/predictions/predictions__stacking__byeonghyeon.csv)
      7. calibrate_stacking.py 에서 Platt/Isotonic 적용
    """
    print()
    print("  ✓ 모든 입력 파일이 준비되었습니다.")
    print("  ✓ 다음 단계: run_stacking() 본문을 구현하여 학습을 시작하세요.")
    print()
    print("  구현 순서 (docs/model-plan.md §2-7 참고):")
    print("    1. OOF 파일 merge  →  Level-1 LogReg 학습")
    print("    2. Val 파일로 PR-AUC / ECE / Brier 평가")
    print("    3. Test 파일로 앙상블 예측 생성 + validate_prediction_file 검증")
    print("    4. calibrate_stacking.py  →  Platt / Isotonic 캘리브레이션")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 70)
    print("  [카테고리 D] Stacking Ensemble — 입력 파일 검증 시작")
    print("=" * 70)
    print()
    print("  등록된 base 모델:")
    for model_name, owner, description in BASE_MODELS:
        print(f"    • {description}  ({model_name}__{owner})")
    print()
    print("  검증 항목:")
    print("    - OOF / Val / Test 파일 존재 여부")
    print("    - 필수 컬럼 포함 여부")
    print("    - 행 수 일치 여부")
    print("    - date × country 병합 키 일관성")
    print("    - 누수 위험 파일 감지")
    print()

    all_ok, errors, warnings = validate_all_inputs()

    if not all_ok:
        _print_missing_checklist(errors, warnings)
        print("  → 학습을 시작하지 않고 종료합니다.")
        print("  → 위 체크리스트를 해결한 후 다시 실행하세요.")
        print()
        sys.exit(0)   # 오류가 아닌 정상 종료 — 파일 미준비는 예상된 상황

    if warnings:
        print("  [ 경고 ]")
        for w in warnings:
            print(f"    ⚠  {w}")
        print()

    run_stacking()


if __name__ == "__main__":
    main()
