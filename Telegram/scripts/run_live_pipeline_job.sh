#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/kubig-telegram-osint}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LIMIT_PER_CHANNEL="${LIMIT_PER_CHANNEL:-100}"
SINCE_DAYS="${SINCE_DAYS:-14}"

cd "$APP_DIR"
mkdir -p logs

"$PYTHON_BIN" scripts/run_live_pipeline.py \
  --limit-per-channel "$LIMIT_PER_CHANNEL" \
  --since-days "$SINCE_DAYS"
