#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_DIR/exp/assign_enhanced_loss_v2t/checkpoints/checkpoint_best.pt}"
RESULT_PATH="${RESULT_PATH:-$PROJECT_DIR/exp/eval_assign_enhanced_loss_v2t}"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
python -B inference.py --config-dir "$PROJECT_DIR/configs/" --config-name inference.yaml \
  dataset.gen_subset=test \
  common_eval.path="$CHECKPOINT_PATH" \
  common_eval.results_path="$RESULT_PATH" \
  override.modalities="['video']" \
  common.user_dir="$PROJECT_DIR" \
  "$@"
