#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Chapter 6.1: seq2seq CE + auxiliary CTC loss for AV2T.
DATA_DIR="${DATA_DIR:-/data2/final_project_data/30h_data}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/data2/final_project_data/spm1000/spm_unigram1000.model}"
PRETRAIN_PATH="${PRETRAIN_PATH:-/data2/final_project_ckpt/pretrained_model.pth}"
CTC_WEIGHT="${CTC_WEIGHT:-0.3}"
CTC_TAG="${CTC_WEIGHT//./p}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/exp/ctc_aux_av2t_w$CTC_TAG}"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
if compgen -G "/opt/conda/lib/python*/site-packages/nvidia/*/lib" >/dev/null; then
  export LD_LIBRARY_PATH="$(printf "%s:" /opt/conda/lib/python*/site-packages/nvidia/*/lib)${LD_LIBRARY_PATH:-}"
fi
python -u main.py --config-dir "$PROJECT_DIR/configs/" \
  --config-name audiovideo2text.yaml \
  task.data="$DATA_DIR" \
  task.label_dir="$DATA_DIR" \
  task.tokenizer_bpe_model="$TOKENIZER_PATH" \
  model.pretrained_path="$PRETRAIN_PATH" \
  model.ctc_weight="$CTC_WEIGHT" \
  criterion.ctc_weight="$CTC_WEIGHT" \
  hydra.run.dir="$RUN_DIR" \
  common.user_dir="$PROJECT_DIR" \
  "$@"
