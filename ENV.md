# ENV

本文记录 2026-06-12 在容器 `lbqy0` 中为本项目配置并验证的开发/运行环境。

## 2026-06-13 新服务器补充记录

更换服务器后，容器名和项目路径保持不变，但 `rzd` 环境需要重新补齐：

- 新容器初始缺少 `git`，安装 fairseq 固定提交前需先安装。
- 新容器初始缺少 C++ 编译器，`editdistance==0.6.0` 需要编译 wheel，因此安装 `gxx_linux-64`。
- `cupy-cuda12x==12.3.0` 在 `rzd` 环境中导入时需要 CUDA 动态库；当前容器的 CUDA runtime 库位于 base 环境的 `nvidia/*/lib` 目录，训练前需要设置 `LD_LIBRARY_PATH`。

新服务器上使用的补充命令：

```bash
conda install -n rzd -c conda-forge git gxx_linux-64 -y
export LD_LIBRARY_PATH=$(printf "%s:" /opt/conda/lib/python3.11/site-packages/nvidia/*/lib)${LD_LIBRARY_PATH:-}
```

验证结果：

```text
torch 2.1.1+cu121
fairseq 1.0.0a0+afc77bd
hydra 1.3.2
omegaconf 2.3.0
cv2 4.5.4
cupy 12.3.0, cuda runtime 12060
```

数据包内 manifest 仍写死 `/data2/final_project_data/...`，新服务器也已在容器内恢复软链接：

```bash
mkdir -p /data2
ln -sfn /base/project/AudioVideo2Txt/final_project_data /data2/final_project_data
ln -sfn /base/project/AudioVideo2Txt/final_project_ckpt /data2/final_project_ckpt
```

5.2 默认配置烟测已通过：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/train_modality_dropout_av2t.sh \
  optimization.max_update=1 \
  checkpoint.no_save=true \
  dataset.num_workers=0 \
  common.log_interval=1 \
  hydra.run.dir=/base/project/AudioVideo2Txt/exp/smoke_modality_dropout_fixed
```

摘要：训练 1 update 后完整 valid 验证完成，`EncoderBackbone Config` 中确认保留
`modality_dropout=0.05`、`audio_dropout=0.3`、`modality_dropout_mode=span`、
`modality_dropout_span_min=10`、`modality_dropout_span_max=30`。

## 基本信息

- 宿主项目路径：`/public/home/lvbqy/project/AudioVideo2Txt`
- 容器项目路径：`/base/project/AudioVideo2Txt`
- 容器：`lbqy0`
- Conda 环境：`rzd`
- Python：`3.10.20`
- 代理：
  ```bash
  export http_proxy=http://59.66.143.200:7897
  export https_proxy=http://59.66.143.200:7897
  export HTTP_PROXY=$http_proxy
  export HTTPS_PROXY=$https_proxy
  export PIP_INDEX_URL=https://pypi.org/simple
  ```

代理可用性已验证：容器内访问 PyPI 和 conda-forge repodata 返回正常 HTTP 响应。

## 环境配置步骤

进入容器和项目目录：

```bash
docker exec -it lbqy0 bash
cd /base/project/AudioVideo2Txt
source /opt/conda/etc/profile.d/conda.sh
conda activate rzd
```

`rzd` 原本是 Python 3.11，不适合本项目若干旧依赖，已降到 Python 3.10：

```bash
conda install -n rzd python=3.10 -y
```

基础工具和兼容版本：

```bash
python -m pip install "pip<24.1"
python -m pip install "numpy==1.23.5" "Cython==3.0.6"
```

PyTorch 使用 conda 安装。原因是 pip 的 CUDA wheel 体积很大，通过当前代理下载曾出现超时/校验失败；conda 安装成功。

```bash
conda install -n rzd pytorch==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

安装 fairseq 固定提交：

```bash
python -m pip install --timeout 300 --retries 5 \
  git+https://github.com/facebookresearch/fairseq.git@afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb
```

fairseq 会把 Hydra/OmegaConf 降到旧版本；本项目配置需要恢复到下面版本：

```bash
python -m pip install --no-deps \
  "omegaconf==2.3.0" \
  "hydra-core==1.3.2" \
  "antlr4-python3-runtime==4.9.3"
python -m pip install "numpy==1.23.5"
```

安装运行依赖：

```bash
python -m pip install --timeout 300 --retries 5 \
  "cupy-cuda12x==12.3.0" \
  "python-speech-features==0.6" \
  "python-speech-features-cuda==0.0.10" \
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
```

容器缺少 `libGL.so.1`，直接使用 `opencv-python==4.5.4.60` 会导入失败。已改用同版本 headless 包，满足服务器读取视频帧需求：

```bash
python -m pip uninstall -y opencv-python
python -m pip install --timeout 300 --retries 5 opencv-python-headless==4.5.4.60
```

## 与 requirements.txt 的差异

- `numpy` 使用 `1.23.5`，不是 `requirements.txt` 中的 `1.24.4`。原因：固定提交的 fairseq 仍依赖 `np.float` / `np.int` 等旧别名。
- `opencv-python` 换为 `opencv-python-headless==4.5.4.60`。原因：当前容器缺少 `libGL.so.1`，headless 包更适合服务器环境。
- `scipy` 使用 `1.10.1`，不是 `requirements.txt` 中的 `1.5.4`。原因：Python 3.10 下 `scipy==1.5.4` 没有合适 wheel，且项目使用的接口在 1.10.1 可用。
- `torch==2.1.1` 通过 conda 的 `pytorch-cuda=12.1` 安装，不使用 pip CUDA wheel。
- 未安装 `dlib`、`scikit-image==0.17.2`、`scikit-video` 等非基线启动必需包；当前第 4 章训练入口和导入验证未依赖它们。

## 验证结果

导入验证命令：

```bash
python - <<'PY'
import sys, importlib.metadata as md
import numpy, torch, fairseq, cupy, cv2, scipy, hydra, omegaconf
from python_speech_features_cuda import logfbank
print("python", sys.version.split()[0])
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("fairseq", fairseq.__version__)
print("hydra", hydra.__version__, "omegaconf", omegaconf.__version__)
print("cupy", cupy.__version__, "cv2", cv2.__version__, "scipy", scipy.__version__)
print("sentencepiece", md.version("sentencepiece"), "editdistance", md.version("editdistance"))
print("python-speech-features-cuda", md.version("python-speech-features-cuda"), "logfbank", logfbank is not None)
PY
```

实际输出摘要：

```text
python 3.10.20
numpy 1.23.5
torch 2.1.1 cuda True devices 8
fairseq 1.0.0a0+afc77bd
hydra 1.3.2 omegaconf 2.3.0
cupy 12.3.0 cv2 4.5.4 scipy 1.10.1
sentencepiece 0.1.96 editdistance 0.6.0
python-speech-features-cuda 0.0.10 logfbank True
```

## 数据和 ckpt

已将项目根目录下的压缩包解压：

```text
final_project_data.zip  -> final_project_data/
final_project_ckpt.zip  -> final_project_ckpt/
```

解压后的关键文件：

```text
final_project_data/30h_data/{train,valid,test}.tsv
final_project_data/30h_data/{train,valid,test}.wrd
final_project_data/spm1000/spm_unigram1000.model
final_project_ckpt/pretrained_model.pth
final_project_ckpt/base_vox_iter5_converted.pt
```

数据包内的 tsv 仍使用官方绝对路径 `/data2/final_project_data/...`。为了不改官方 tsv，已在容器中创建软链接：

```bash
mkdir -p /data2
ln -sfn /base/project/AudioVideo2Txt/final_project_data /data2/final_project_data
ln -sfn /base/project/AudioVideo2Txt/final_project_ckpt /data2/final_project_ckpt
```

这样 `scripts/*.sh` 的默认路径可以直接使用。

## 第 4 章基线运行

第 4 章基线入口：

```bash
bash scripts/train_v2t.sh
bash scripts/train_av2t.sh
```

脚本默认路径：

- 数据：`/data2/final_project_data/30h_data`
- tokenizer：`/data2/final_project_data/spm1000/spm_unigram1000.model`
- 预训练模型：`/data2/final_project_ckpt/pretrained_model.pth`

已完成 v2t 的 1 update 烟测：

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/train_v2t.sh \
  optimization.max_update=1 \
  dataset.num_workers=0 \
  common.fp16=false \
  common.log_interval=1 \
  hydra.run.dir=/tmp/rzd_v2t_smoke2
```

结果：训练、验证均完成，说明视频数据加载、tokenizer、预训练 ckpt 加载、前后向和 checkpoint 保存均可用。摘要：

```text
train_loss=259.316, train_accuracy=0.571, valid_loss=194.454, valid_accuracy=0.747
done training in 66.1 seconds
```

已完成 av2t 的 1 update 烟测：

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/train_av2t.sh \
  optimization.max_update=1 \
  dataset.num_workers=0 \
  common.fp16=false \
  common.log_interval=1 \
  hydra.run.dir=/tmp/rzd_av2t_smoke
```

结果：训练、验证均完成，说明视频+音频数据加载、CuPy 音频特征、tokenizer、预训练 ckpt 加载、前后向和 checkpoint 保存均可用。摘要：

```text
train_loss=259.981, train_accuracy=0.571, valid_loss=194.300, valid_accuracy=0.923
done training in 87.4 seconds
```

完整第 4 章基线训练可直接运行：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> bash scripts/train_v2t.sh
CUDA_VISIBLE_DEVICES=<gpu_id> bash scripts/train_av2t.sh
```

若不使用 `/data2` 软链接，也可以用环境变量覆盖：

```bash
export DATA_DIR=/path/to/final_project_data/30h_data
export TOKENIZER_PATH=/path/to/final_project_data/spm1000/spm_unigram1000.model
export PRETRAIN_PATH=/path/to/final_project_ckpt/pretrained_model.pth
CUDA_VISIBLE_DEVICES=<gpu_id> bash scripts/train_v2t.sh
CUDA_VISIBLE_DEVICES=<gpu_id> bash scripts/train_av2t.sh
```
