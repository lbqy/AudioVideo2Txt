# AudioVideo2Txt Reproduction Guide

This package contains the project code, `answer.md`, and the key result files used in the report. The best checkpoint is hosted separately on Hugging Face.

## Package Contents

- `code/`: source code, configs, and experiment scripts.
- `checkpoints/README.md`: link and file name for the best Chapter 6.1 checkpoint.
- `results_summary/`: WER files for key decoded experiments.
- `answer.md`: final report.
- `docs/`: original requirement and handoff documents.

The full training dataset and original pretrained checkpoint are not included in this zip. They are expected to be available in the same server layout used by this project:

```bash
/data2/final_project_data/30h_data
/data2/final_project_data/spm1000/spm_unigram1000.model
/data2/final_project_ckpt/pretrained_model.pth
```

If the unzipped dataset and checkpoint are under the project directory, create the same symlinks:

```bash
mkdir -p /data2
ln -sfn /base/project/AudioVideo2Txt/final_project_data /data2/final_project_data
ln -sfn /base/project/AudioVideo2Txt/final_project_ckpt /data2/final_project_ckpt
```

## Environment

The verified environment is documented in `code/ENV.md`. The short version is:

```bash
docker exec -it lbqy0 bash
cd /base/project/AudioVideo2Txt
source /opt/conda/etc/profile.d/conda.sh
conda activate rzd
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$(printf "%s:" /opt/conda/lib/python*/site-packages/nvidia/*/lib)${LD_LIBRARY_PATH:-}
```

Important package versions:

```text
Python 3.10.20
PyTorch 2.1.1 + CUDA 12.1
fairseq 1.0.0a0+afc77bd
hydra-core 1.3.2
omegaconf 2.3.0
numpy 1.23.5
```

## Reproduce Chapter 6.1 Default Training

The default Chapter 6.1 setting is AV2T with `ctc_weight=0.3` and `max_update=30000`.

```bash
cd /base/project/AudioVideo2Txt
CUDA_VISIBLE_DEVICES=0 \
CTC_WEIGHT=0.3 \
RUN_DIR=$PWD/exp/ctc_aux_av2t_w03_reproduce \
bash scripts/train_ctc_aux_av2t.sh
```

Expected final validation result is close to:

```text
valid_loss: 43.316
valid_accuracy: 91.583%
```

## Download and Evaluate the Best Checkpoint

The best checkpoint is hosted at:

```text
https://huggingface.co/lbqy/AudioVideo2Txt
```

Download it with `huggingface-cli`, or download it manually from the Hugging Face page:

```bash
huggingface-cli download lbqy/AudioVideo2Txt \
  ctc_aux_av2t_w03_checkpoint_best.pt \
  --local-dir /base/project/AudioVideo2Txt/submission_audio_video2txt_20260617/checkpoints
```

Then run:

```bash
cd /base/project/AudioVideo2Txt
CUDA_VISIBLE_DEVICES=0 \
CHECKPOINT_PATH=/base/project/AudioVideo2Txt/submission_audio_video2txt_20260617/checkpoints/ctc_aux_av2t_w03_checkpoint_best.pt \
RESULT_PATH=/base/project/AudioVideo2Txt/exp/eval_ctc_aux_av2t_w03_packaged \
bash scripts/test_av2t.sh
```

Expected test result:

```text
WER: 13.63062996215775
err / num_ref_words = 1837 / 13477
```

## Other Useful Commands

Run the baseline AV2T evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 \
CHECKPOINT_PATH=$PWD/exp/exp_av2t/checkpoints/checkpoint_best.pt \
RESULT_PATH=$PWD/exp/eval_av2t \
bash scripts/test_av2t.sh
```

Run CTC auxiliary training with another weight:

```bash
CUDA_VISIBLE_DEVICES=0 CTC_WEIGHT=0.1 bash scripts/train_ctc_aux_av2t.sh
CUDA_VISIBLE_DEVICES=0 CTC_WEIGHT=0.5 bash scripts/train_ctc_aux_av2t.sh
```

Generate training curves from available logs:

```bash
python scripts/plot_all_training_curves.py
```
