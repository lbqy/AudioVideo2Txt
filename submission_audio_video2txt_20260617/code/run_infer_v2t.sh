#!/usr/bin/env bash
set -euo pipefail
cd /home/limusheng/my_projects/RZD-HW
export PYTHONPATH="/home/limusheng/my_projects/RZD-HW:${PYTHONPATH:-}"
export PATH="/home/limusheng/miniconda3/envs/RZD/bin:$PATH"
CKPT="/home/limusheng/my_projects/RZD-HW/exp/v2t_run/checkpoints/checkpoint_best.pt"
RES="/home/limusheng/my_projects/RZD-HW/results/inference/v2t"
mkdir -p "$RES"
CUDA_VISIBLE_DEVICES=1 python -B inference.py --config-dir "$PWD/configs/" --config-name inference.yaml \
  dataset.gen_subset=test \
  common_eval.path="$CKPT" \
  common_eval.results_path="$RES" \
  override.modalities="['video']" \
  common.user_dir="$PWD"
