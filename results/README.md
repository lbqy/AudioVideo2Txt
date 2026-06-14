# results/ —— 实验结果总览

本文件夹整理了第 4 章基本框架实现的全部训练 / 推理结果，供报告引用。

## 关键数值

| 模型 | train loss | train acc | valid loss | valid acc | inference WER | Baseline WER |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 视频 v2t | 59.96 | 67.95% | 63.94 | 66.65% | **46.29** | 46.46 |
| 视频+音频 av2t | 33.21 | 94.69% | 37.36 | 91.49% | **13.85** | 15.17 |

两个模型均达到 / 略优于文档表 1 的 Baseline。

## 目录结构

- `plots/`
  - `v2t_loss.png` / `v2t_accuracy.png` / `v2t_curves.png`：视频模型训练-验证的损失 / 准确率曲线。
  - `av2t_loss.png` / `av2t_accuracy.png` / `av2t_curves.png`：音视频模型对应曲线。
- `metrics/`
  - `{v2t,av2t}_metrics.json`：从训练日志解析出的逐点曲线数据（train/valid 的 loss、accuracy、nll）。
  - `{v2t,av2t}_summary.json`：最终 / 最佳指标汇总。
  - `{v2t,av2t}_final.json`：最终五项指标 + 与 Baseline 的对照。
- `inference/`
  - `v2t/`、`av2t/`：`hypo-*.json`（逐句 REF/HYP）、`wer.*`（WER 数值与错误统计）、`decode.log`。
- `logs/`
  - `train_{v2t,av2t}.log`：完整训练日志（json 行，含每 200 步的 train_inner 与每个 epoch 的 valid）。
  - `infer_{v2t,av2t}.log`：推理日志（逐句解码 + 最终 WER）。

## 复现方式

环境：conda 环境 `RZD`（见仓库根目录 `setup_env.sh` 与 `requirements.txt`）。

```bash
# 训练（指定空闲 GPU）
bash run_train_v2t.sh        # 视频
bash run_train_av2t.sh       # 视频+音频
# 推理
bash run_infer_v2t.sh
bash run_infer_av2t.sh
# 绘图（解析日志 -> plots/ + metrics/）
python plot_curves.py results/logs/train_v2t.log v2t results
python plot_curves.py results/logs/train_av2t.log av2t results
```

> 训练好的模型 checkpoint 位于 `exp/v2t_run/checkpoints/checkpoint_best.pt` 与
> `exp/av2t_run/checkpoints/checkpoint_best.pt`。
