#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

MODEL="${MODEL:-deepseek-v4-flash}"
MODE="${MODE:-all}"
BENCH="${BENCH:-core}"
GENERICAGENT_ROOT="${GENERICAGENT_ROOT:-../GenericAgent}"
PYTHON_BIN="${PYTHON_BIN:-$GENERICAGENT_ROOT/.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-results/sop_lifelong_${MODEL//-/_}}"

"$PYTHON_BIN" \
  experiments/run_sop_lifelong.py \
  --genericagent-root "$GENERICAGENT_ROOT" \
  --model "$MODEL" \
  --mode "$MODE" \
  --bench "$BENCH" \
  --out-root "$OUT_ROOT" \
  "$@"
