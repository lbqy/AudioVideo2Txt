#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Default feature-mask setting selected as the simpler positive V2T variant.
DATA_DIR="${DATA_DIR:-/data2/final_project_data/30h_data}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/data2/final_project_data/spm1000/spm_unigram1000.model}"
PRETRAIN_PATH="${PRETRAIN_PATH:-/data2/final_project_ckpt/pretrained_model.pth}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/exp/assign_feature_mask_v2t}"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
python -u main.py --config-dir "$PROJECT_DIR/configs/" \
  --config-name video2text.yaml \
  task.data="$DATA_DIR" \
  task.label_dir="$DATA_DIR" \
  task.tokenizer_bpe_model="$TOKENIZER_PATH" \
  model.pretrained_path="$PRETRAIN_PATH" \
  model.apply_mask=true \
  model.mask_prob=0.05 \
  model.mask_length=10 \
  model.mask_channel_prob=0.0 \
  hydra.run.dir="$RUN_DIR" \
  common.user_dir="$PROJECT_DIR" \
  "$@"
