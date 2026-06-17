#!/usr/bin/env bash
# Reproducible setup for the RZD conda environment (verified working).
#
# Key notes / gotchas (already reflected below and in the source code):
#   * numpy is pinned to 1.23.5, NOT the 1.24.4 in requirements.txt: numpy 1.24
#     removed np.float / np.int, which the pinned fairseq still uses -> ImportError.
#   * Installing the pinned fairseq downgrades omegaconf/hydra to its own caps
#     (2.0.6 / 1.0.7); we restore omegaconf==2.3.0 / hydra-core==1.3.2 afterwards
#     (the configs and CLI override syntax need the newer versions).
#   * Two small runtime shims live in the SOURCE (no env action needed):
#       - dataset.py re-exposes cupy.core.core.ndarray for python_speech_features_cuda
#       - inference.py aliases omegaconf._utils.is_primitive_type for fairseq ckpt load
#
# Adjust CONDA_SH below for your machine if conda lives elsewhere.
set -euo pipefail
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SH"

echo "===== [1/5] create env RZD (python 3.10) ====="
conda create -n RZD python=3.10 -y
conda activate RZD
PIP="python -m pip"
$PIP install --upgrade "pip<24.1" setuptools wheel

echo "===== [2/5] build deps (numpy 1.23.5 for fairseq np.float compat) ====="
$PIP install "numpy==1.23.5" "Cython==3.0.6"

echo "===== [3/5] torch 2.1.1 + cu121 ====="
$PIP install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121

echo "===== [4/5] fairseq (pinned commit) then restore hydra/omegaconf ====="
$PIP install git+https://github.com/facebookresearch/fairseq.git@afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb
# fairseq pins omegaconf<2.1 / hydra<1.1; restore the versions the configs need:
$PIP install --no-deps "omegaconf==2.3.0" "hydra-core==1.3.2" "antlr4-python3-runtime==4.9.3"
# fairseq may have re-pulled numpy; force it back to 1.23.5:
$PIP install "numpy==1.23.5"

echo "===== [5/5] remaining runtime deps ====="
$PIP install \
  "cupy-cuda12x==12.3.0" \
  "python-speech-features==0.6" \
  "python-speech-features-cuda==0.0.10" \
  "opencv-python==4.5.4.60" \
  "scipy==1.10.1" \
  "sentencepiece==0.1.96" \
  "editdistance==0.6.0" \
  "tensorboard==2.14.0" \
  "sacrebleu==2.3.2" \
  "tabulate==0.9.0" \
  "matplotlib==3.7.4" \
  "tqdm==4.66.1" \
  "regex==2023.10.3" \
  "PyYAML==6.0.1" \
  "portalocker==2.8.2" \
  "fastrlock==0.8.2" \
  "bitarray"

echo "===== DONE: verifying imports ====="
python - <<'PY'
import numpy, torch, fairseq, cupy, cv2, scipy, sentencepiece, editdistance, hydra, omegaconf
print("numpy", numpy.__version__, "(expect 1.23.5)")
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
print("fairseq", fairseq.__version__)
print("hydra", hydra.__version__, "| omegaconf", omegaconf.__version__)
print("cupy", cupy.__version__, "| cv2", cv2.__version__)
import python_speech_features_cuda
from python_speech_features_cuda import logfbank
print("psf_cuda + logfbank OK")
PY
echo "ALL_SETUP_DONE"
