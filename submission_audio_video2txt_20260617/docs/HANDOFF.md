# 交接说明（第 4 章已完成 → 第 5、6 章接手指南）

> 本文档给接手第 5、6 章的队友。第 4 章「基本框架实现」（Task 1–11）已全部完成并验证，
> 详见 `answer.md`。第 5、6 章在此基础上继续即可。

---

## 1. 第 5、6 章到底需要哪些「现有的东西」？

### 1.1 需要 ✅
- **本仓库的全部源代码**（已包含第 4 章实现）。这是地基：第 5 章的三个方法都建立在它之上
  （例如 `compute_enhanced_loss` 会调用第 4 章实现的 `label_smoothed_nll_loss`；任何训练/推理
  都依赖第 4 章的 `dataset.py / modality_encoder.py / multihead_attention.py`）。
  **不能用原始空壳代码，否则什么都跑不通。**
- **预训练模型 `pretrained_model.pth`**（在 `final_project_ckpt.zip` 里，单独提供）。
  第 5 章所有训练脚本都是从它开始微调的。
- **数据集 `final_project_data.zip`**（单独提供）。和第 4 章同一份数据，无需新数据。
- **RZD conda 环境**（用 `setup_env.sh` 复现，注意下面第 3 节的坑）。
- **第 4 章的 baseline 结果**（`results/` + `answer.md` §12）：第 6 章做对比分析时，
  进阶方法要和它对照。

### 1.2 不需要 ❌
- **不需要我训练好的微调 checkpoint**（`exp/v2t_run`、`exp/av2t_run` 里的 `checkpoint_best.pt`）。
  第 5 章是从 `pretrained_model.pth` **重新微调**，不是从我的基线 checkpoint 继续。
  这些基线 checkpoint **唯一的用处**是：第 6 章若想用「完全一致的基线」重新跑一次推理对比，
  可以省去重训基线的时间。否则直接引用 `results/` 里记录的数字即可。
  （因为体积大约 3.8GB，默认不打包；需要再单独拷。）

---

## 2. 第 5、6 章具体要做什么

第 5 章「至少选两种」，对应三个进阶训练方法：

| Task | 方法 | 文件 · 函数 | 配套脚本 |
| ---- | ---- | ---- | ---- |
| 12 | 损失改进（focal + confidence penalty） | `losses/label_smoothed_cross_entropy.py · LabelSmoothedCrossEntropyCriterion.compute_enhanced_loss` | `scripts/train_enhanced_loss_v2t.sh` / `scripts/test_enhanced_loss_v2t.sh` |
| 13 | Modality Dropout | `models/encoder_backbone.py · EncoderBackbone.apply_modality_dropout` | `scripts/train_modality_dropout_av2t.sh` / `scripts/test_modality_dropout_av2t.sh` |
| 14 | Feature Mask | `models/encoder_backbone.py · EncoderBackbone.apply_feature_mask` | `scripts/train_feature_mask_v2t.sh` / `scripts/test_feature_mask_v2t.sh` |

> 这三个函数均已实现，并通过对应配置开关控制：
> `criterion.focal_gamma/confidence_penalty` 控制 Task 12，
> `model.modality_dropout/audio_dropout/modality_dropout_mode` 控制 Task 13，
> `model.apply_mask/mask_prob/mask_channel_prob` 控制 Task 14。
> Task 13 和 Task 14 可独立启用，也可在同一次 AV2T 训练中叠加。

第 6 章「二选一」：路线一是对第 5 章方法做对照实验分析（Task 15），路线二是提出新方法（Task 16）。
对照的「基线」就是第 4 章我已经跑出来的 v2t / av2t 结果（见 `results/` 与 `answer.md` §12.1）。

---

## 3. 环境复现的关键坑（务必照做，否则跑不起来）

直接运行 `bash setup_env.sh` 即可（已把下面的坑都处理好了）。要点：

1. **numpy 必须是 1.23.5**，不能用 `requirements.txt` 里写的 1.24.4——1.24 删除了 `np.float/np.int`，
   而本项目固定版本的 fairseq 仍在用它们，会直接 import 失败。
2. **装 fairseq 会把 omegaconf/hydra 降级**到它自己的上限（2.0.6 / 1.0.7），装完要再恢复成
   `omegaconf==2.3.0 / hydra-core==1.3.2 / antlr4-python3-runtime==4.9.3`（`setup_env.sh` 已做）。
3. 源码里有两处**运行时兼容垫片**（已写进代码，无需额外操作）：
   - `dataset.py` 顶部：为 `python_speech_features_cuda` 重新暴露被新版 cupy 移除的
     `cupy.core.core.ndarray`。
   - `inference.py` 顶部：为 fairseq 加载 checkpoint 补回被 omegaconf 改名的
     `is_primitive_type`。

---

## 4. 数据路径坑（重要）

数据 tsv（`<data>/30h_data/{train,valid,test}.tsv`）里第 2、3 列是**视频/音频的绝对路径**。
官方数据包里写的是服务器路径 `/data2/final_project_data/...`：
- **在公共服务器上跑**：路径正好对，无需改。
- **在自己机器上跑**：要把这些前缀替换成你本机的数据目录，例如：
  ```bash
  cd <你的数据目录>/30h_data
  NEW="<你的数据目录>"     # final_project_data 的父目录里那个 final_project_data 的绝对路径
  sed -i "s|/data2/final_project_data|$NEW/final_project_data|g" train.tsv valid.tsv test.tsv
  ```
> 注意：**不要用我本机改过的 tsv**（里面是我的本地路径）。请用官方数据包里的原始 tsv，按上面方法改成你的环境路径。

---

## 5. 一个一定要知道的「血泪经验」

第 4 章 `dataset.py · get_label` 里 **`append_eos` 必须为 True**（已修好）。
如果漏了，目标序列没有 `<eos>`，模型学不会停止生成：teacher-forcing 准确率看起来正常（~65%），
但 beam search 自回归解码会一直生成到最大长度，WER 直接飙到 ~400%。
第 5 章新训练的模型同样依赖这一点，千万别在合并代码时把它改回去。

---

## 6. 怎么跑（命令速查）

```bash
conda activate RZD
# 指定一张空闲 GPU（公共服务器请用分配给你的那张）
export CUDA_VISIBLE_DEVICES=<gpu_id>
export DATA_DIR=<...>/30h_data
export TOKENIZER_PATH=<...>/spm1000/spm_unigram1000.model
export PRETRAIN_PATH=<...>/pretrained_model.pth

# 第 5 章：实现对应函数后训练 + 推理
bash scripts/train_enhanced_loss_v2t.sh        # Task 12
bash scripts/test_enhanced_loss_v2t.sh
bash scripts/train_modality_dropout_av2t.sh    # Task 13
bash scripts/test_modality_dropout_av2t.sh
bash scripts/train_feature_mask_v2t.sh         # Task 14
bash scripts/test_feature_mask_v2t.sh
bash scripts/train_feature_mask_av2t.sh        # Task 14 on AV2T
bash scripts/test_feature_mask_av2t.sh
bash scripts/train_modality_dropout_feature_mask_av2t.sh  # Task 13 + 14
bash scripts/test_modality_dropout_feature_mask_av2t.sh

# 画训练曲线（解析日志 -> results/plots + results/metrics）
python plot_curves.py <训练stdout日志> <tag> results
```

> `scripts/*.sh` 默认用服务器路径 `/data2/...`；在自己机器上跑时用上面的环境变量覆盖
> （`DATA_DIR/TOKENIZER_PATH/PRETRAIN_PATH`），或参考我写的本地包装脚本
> `run_train_v2t.sh / run_train_av2t.sh / run_infer_*.sh`（里面是绝对路径示例）。

---

## 7. 第 4 章基线结果（第 6 章对照用）

| 模型 | train loss | train acc | valid loss | valid acc | inference WER |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 视频 v2t | 59.96 | 67.95% | 63.94 | 66.65% | 46.29 |
| 视频+音频 av2t | 33.21 | 94.69% | 37.36 | 91.49% | 13.85 |

明细见 `results/`（曲线 `plots/`、指标 `metrics/`、逐句推理 `inference/`、日志 `logs/`）与 `answer.md`。

---

## 8. 2026-06-13 当前状态：第 5 章默认实验重跑

新服务器环境已恢复，补充记录见 `ENV.md`。容器内 `/data2/final_project_data` 和
`/data2/final_project_ckpt` 已重新指向项目目录，`cupy` 需要训练前设置：

```bash
export LD_LIBRARY_PATH=$(printf "%s:" /opt/conda/lib/python3.11/site-packages/nvidia/*/lib)${LD_LIBRARY_PATH:-}
```

### 8.1 Task 13 / 5.2 代码复查结论

旧 5.2 默认实验结果暂缓采信。复查发现两个实现/配置风险：

1. `model.modality_dropout_mode=span` 曾经可能被预训练 checkpoint 的旧 cfg 覆盖回默认 `sample`。
   已在 `models/audio_video_model.py` 中对 merge 后配置再次应用 overrides，烟测确认
   `EncoderBackbone Config` 保留 `span / 0.05 / 0.3 / 10-30`。
2. 旧实现先在 audio/video 特征上置零，再 concat 并做 `LayerNorm(1536)`。
   对 concat 融合而言，LayerNorm 会把被置零模态变成非零归一化常数，导致“drop 掉的模态”没有真正保持缺失。
   已改为默认 concat 路径下在 `LayerNorm` 之后、`post_extract_proj` 之前做 fused modality dropout；
   add 融合路径仍保留 pre-fusion fallback。

已完成 1 update 烟测：训练和完整 valid 均可跑通。

### 8.2 正在运行的默认实验

四个实验已在 GPU1-4 启动，每个实验单卡：

| 任务 | 模态 | GPU | run dir | log |
| ---- | ---- | --- | ------- | --- |
| 5.1 enhanced loss | V2T | 1 | `exp/default_5_1_enhanced_loss_v2t_20260613` | `results/repro_logs/default_5_1_enhanced_loss_v2t_20260613.log` |
| 5.2 modality dropout fixed | AV2T | 2 | `exp/default_5_2_modality_dropout_av2t_fixed_20260613` | `results/repro_logs/default_5_2_modality_dropout_av2t_fixed_20260613.log` |
| 5.3 feature mask | V2T | 3 | `exp/default_5_3_feature_mask_v2t_20260613` | `results/repro_logs/default_5_3_feature_mask_v2t_20260613.log` |
| 5.3 feature mask | AV2T | 4 | `exp/default_5_3_feature_mask_av2t_20260613` | `results/repro_logs/default_5_3_feature_mask_av2t_20260613.log` |

启动后均已进入训练并至少到达 `num_updates=200`。GPU5 有约 28GB 显存占用，未使用以免影响别人任务。

新增了 `scripts/train_enhanced_loss_av2t.sh`，用于后续补跑 5.1 的 AV2T 对照。

### 8.3 6.1 对比与 5.2 调参建议

第 6.1 先做“默认配置横向对比”：

- V2T：baseline vs 5.1 enhanced loss vs 5.3 feature mask。
- AV2T：baseline vs 5.2 modality dropout fixed vs 5.3 feature mask；补跑 5.1 AV2T 后再加入。
- 每组记录 train/valid loss、accuracy、best checkpoint 的 test WER；不要只看 teacher-forcing valid accuracy。

5.2 深入调参建议先做小网格，避免一次铺太大：

| 组别 | `modality_dropout` | `audio_dropout` | mode | span |
| ---- | ------------------ | --------------- | ---- | ---- |
| default fixed | 0.05 | 0.3 | span | 10-30 |
| lower p | 0.02 | 0.3 | span | 10-30 |
| higher p | 0.10 | 0.3 | span | 10-30 |
| sample mode | 0.05 | 0.3 | sample | - |
| more video-only pressure | 0.05 | 0.5 | span | 10-30 |
| shorter span | 0.05 | 0.3 | span | 5-15 |

若默认 fixed 仍没有 clean WER 收益，建议额外加 robustness eval：测试时分别遮挡/扰动 audio 或 video，
因为 modality dropout 的主要收益可能体现为鲁棒性，而不是干净验证集上的最高 accuracy。
