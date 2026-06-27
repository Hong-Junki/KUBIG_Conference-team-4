#!/usr/bin/env bash
# onset 서빙 엔트리포인트. 첫 인자로 모드 선택.
#   features                  → raw(BQ) → model_input(BQ) 피처 빌더
#   score --run-ts <ISO>      → model_input(BQ) → model_scores(BQ) 스코어러
set -euo pipefail
MODE="${1:-score}"; shift || true
case "$MODE" in
  features)
    exec python -m src.serve.build_model_input "$@" ;;
  score)
    exec python -m src.serve.run_scoring "$@" ;;
  *)
    echo "usage: {features|score} [args]"; exit 1 ;;
esac
