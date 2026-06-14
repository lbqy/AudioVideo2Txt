#!/usr/bin/env bash
set -euo pipefail
cd /home/limusheng/my_projects/RZD-HW
export PYTHONPATH="/home/limusheng/my_projects/RZD-HW:${PYTHONPATH:-}"
export PATH="/home/limusheng/miniconda3/envs/RZD/bin:$PATH"
DATA_DIR="/home/limusheng/my_projects/RZD-HW/final_project_data/30h_data"
TOKENIZER_PATH="/home/limusheng/my_projects/RZD-HW/final_project_data/spm1000/spm_unigram1000.model"
PRETRAIN_PATH="/home/limusheng/my_projects/RZD-HW/final_project_ckpt/pretrained_model.pth"
CUDA_VISIBLE_DEVICES=1 python -u main.py --config-dir "$PWD/configs/" \
  --config-name audiovideo2text.yaml \
  task.data="$DATA_DIR" task.label_dir="$DATA_DIR" \
  task.tokenizer_bpe_model="$TOKENIZER_PATH" \
  model.pretrained_path="$PRETRAIN_PATH" \
  hydra.run.dir="$PWD/exp/av2t_run" \
  common.user_dir="$PWD" \
  distributed_training.distributed_port=29683
