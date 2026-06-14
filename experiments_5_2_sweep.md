# 5.2 Modality Dropout Sweep Plan

本文记录 Task 13 / 5.2 的细扫参方案。当前默认配置为：

```text
modality_dropout=0.05
audio_dropout=0.3
modality_dropout_mode=span
modality_dropout_span_min=10
modality_dropout_span_max=30
```

默认含义：每个样本以 5% 概率做模态 dropout；被选中后 30% 概率遮 audio，70% 概率遮 video；
span 模式只遮一段连续时间片。

## 第一阶段：主效应小网格

目标是先回答四个问题：

- dropout 概率是不是太弱或太强？
- 遮 audio / 遮 video 的比例是否合理？
- span dropout 是否优于 sample-level 整段模态 dropout？
- span 长度是否过短或过长？

| ID | 目的 | `modality_dropout` | `audio_dropout` | mode | span | 备注 |
| -- | ---- | ------------------ | --------------- | ---- | ---- | ---- |
| md_p002_a03_span10_30 | 降低扰动强度 | 0.02 | 0.3 | span | 10-30 | 如果默认伤 clean acc，先看更轻正则 |
| md_p005_a03_span10_30 | fixed default | 0.05 | 0.3 | span | 10-30 | 已在跑，可作为中心点 |
| md_p010_a03_span10_30 | 提高扰动强度 | 0.10 | 0.3 | span | 10-30 | 看是否需要更强鲁棒正则 |
| md_p005_a05_span10_30 | audio/video 更均衡 | 0.05 | 0.5 | span | 10-30 | 默认更常遮 video；这里改成均衡 |
| md_p005_a03_sample | sample vs span | 0.05 | 0.3 | sample | - | 整个样本遮单模态，强扰动 |
| md_p005_a03_span5_15 | 更短局部遮挡 | 0.05 | 0.3 | span | 5-15 | 更像局部噪声/短缺失 |
| md_p005_a03_span20_50 | 更长局部遮挡 | 0.05 | 0.3 | span | 20-50 | 更像长时间模态缺失 |

第一阶段优先顺序：

1. `md_p002_a03_span10_30`
2. `md_p010_a03_span10_30`
3. `md_p005_a05_span10_30`
4. `md_p005_a03_sample`
5. `md_p005_a03_span5_15`
6. `md_p005_a03_span20_50`

当前正在跑的 fixed default 不重复启动，等它完成后直接填入表格。

## 第二阶段：围绕赢家细化

根据第一阶段结果选择一个方向细化：

如果低概率更好：

| ID | `modality_dropout` | `audio_dropout` | mode | span |
| -- | ------------------ | --------------- | ---- | ---- |
| md_p001_a03_span10_30 | 0.01 | 0.3 | span | 10-30 |
| md_p003_a03_span10_30 | 0.03 | 0.3 | span | 10-30 |

如果高概率更好：

| ID | `modality_dropout` | `audio_dropout` | mode | span |
| -- | ------------------ | --------------- | ---- | ---- |
| md_p015_a03_span10_30 | 0.15 | 0.3 | span | 10-30 |
| md_p020_a03_span10_30 | 0.20 | 0.3 | span | 10-30 |

如果遮挡方向敏感：

| ID | `modality_dropout` | `audio_dropout` | mode | span |
| -- | ------------------ | --------------- | ---- | ---- |
| md_p005_a01_span10_30 | 0.05 | 0.1 | span | 10-30 |
| md_p005_a07_span10_30 | 0.05 | 0.7 | span | 10-30 |
| md_p005_a09_span10_30 | 0.05 | 0.9 | span | 10-30 |

## 评估口径

每个实验至少记录：

- 训练最终 `train_loss / train_accuracy`
- 最优验证 `valid_loss / valid_accuracy`
- `checkpoint_best.pt` 的 test WER
- 与第 4 章 AV2T baseline 对比：`valid_accuracy=91.49%`，`WER=13.85`

注意：modality dropout 的收益可能不是 clean set 上的最高 accuracy，而是鲁棒性。因此建议对排名靠前的
2-3 个 checkpoint 增加 robustness eval：

- clean test WER
- audio-missing / audio-noise WER
- video-missing / video-occlusion WER

## 启动方式

使用 `scripts/launch_modality_dropout_sweep.sh`。默认是 dry run，不会真的启动：

```bash
bash scripts/launch_modality_dropout_sweep.sh
```

确认命令没问题后再启动，例如用 GPU 1-4 跑前四个待跑实验：

```bash
DRY_RUN=0 GPU_IDS=1,2,3,4 OFFSET=0 COUNT=4 bash scripts/launch_modality_dropout_sweep.sh
```

等这一批结束后，继续跑后两个：

```bash
DRY_RUN=0 GPU_IDS=1,2 OFFSET=4 COUNT=2 bash scripts/launch_modality_dropout_sweep.sh
```
