#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # Background launches do not inherit an interactive conda activation.
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-rzd}"
fi

DRY_RUN="${DRY_RUN:-1}"
GPU_IDS="${GPU_IDS:-1,2,3,4}"
OFFSET="${OFFSET:-0}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
COUNT="${COUNT:-${#GPUS[@]}}"

EXPERIMENTS=(
  "md_p002_a03_span10_30|0.02|0.3|span|10|30"
  "md_p010_a03_span10_30|0.10|0.3|span|10|30"
  "md_p005_a05_span10_30|0.05|0.5|span|10|30"
  "md_p005_a03_sample|0.05|0.3|sample|10|30"
  "md_p005_a03_span5_15|0.05|0.3|span|5|15"
  "md_p005_a03_span20_50|0.05|0.3|span|20|50"
  "md_p001_a03_span10_30|0.01|0.3|span|10|30"
  "md_p003_a03_span10_30|0.03|0.3|span|10|30"
  "md_p002_a02_span10_30|0.02|0.2|span|10|30"
  "md_p002_a04_span10_30|0.02|0.4|span|10|30"
)

mkdir -p results/repro_logs

export DATA_DIR="${DATA_DIR:-/data2/final_project_data/30h_data}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/data2/final_project_data/spm1000/spm_unigram1000.model}"
export PRETRAIN_PATH="${PRETRAIN_PATH:-/data2/final_project_ckpt/pretrained_model.pth}"
export LD_LIBRARY_PATH="$(printf "%s:" /opt/conda/lib/python3.11/site-packages/nvidia/*/lib)${LD_LIBRARY_PATH:-}"

end=$((OFFSET + COUNT))
if (( end > ${#EXPERIMENTS[@]} )); then
  end=${#EXPERIMENTS[@]}
fi

for ((idx = OFFSET; idx < end; idx++)); do
  gpu="${GPUS[$((idx - OFFSET))]}"
  IFS='|' read -r tag drop audio mode span_min span_max <<< "${EXPERIMENTS[$idx]}"

  run_dir="$PROJECT_DIR/exp/sweep_5_2_${tag}_20260613"
  log_file="$PROJECT_DIR/results/repro_logs/sweep_5_2_${tag}_20260613.log"
  cmd=(
    env
    CUDA_VISIBLE_DEVICES="$gpu"
    RUN_DIR="$run_dir"
    bash scripts/train_modality_dropout_av2t.sh
    "model.modality_dropout=$drop"
    "model.audio_dropout=$audio"
    "model.modality_dropout_mode=$mode"
    "model.modality_dropout_span_min=$span_min"
    "model.modality_dropout_span_max=$span_max"
  )

  echo "[$tag] gpu=$gpu run_dir=$run_dir log=$log_file"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  '
    printf '%q ' "${cmd[@]}"
    printf '> %q 2>&1 &\n' "$log_file"
  else
    nohup "${cmd[@]}" > "$log_file" 2>&1 &
    echo "  pid=$!"
  fi
done
