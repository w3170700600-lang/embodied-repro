---
library_name: lerobot
tags:
  - robotics
  - act
  - imitation-learning
  - so101
  - manipulation
license: mit
---

# ACT · SO-101 抓取策略

基于 [LeRobot](https://github.com/huggingface/lerobot) 0.6.1 训练的 ACT（Action Chunking with Transformers）策略，
控制一条 SO-101 从动臂完成「抓取绿色方块并抬起」的单臂操作任务。

> 本仓库为 **100K 步训练完成的最终权重**（4 小时 12 分，单张 RTX A4000）。

## 模型信息

| 项 | 值 |
|---|---|
| 算法 | ACT (Action Chunking with Transformers) |
| 参数量 | 51,597,190 (≈52M) |
| 视觉骨干 | ResNet18（ImageNet 预训练） |
| 动作块长度 `chunk_size` | 100 |
| 每次执行步数 `n_action_steps` | 100 |
| 隐变量 | CVAE，`latent_dim=32` |
| 输入 | 1 路单目图像 480×640 + 6 维关节状态 |
| 输出 | 6 维关节目标位置（度） |

## 训练信息

| 项 | 值 |
|---|---|
| 数据集 | [`w3170700600/so101_test`](https://huggingface.co/datasets/w3170700600/so101_test) |
| 示范条数 | 56 条 |
| 总帧数 | 18,794 帧 @ 30 fps |
| 训练硬件 | NVIDIA RTX A4000 (16 GB) |
| batch size | 8 |
| 实际步数 | 100,000 |
| 训练耗时 | 4 小时 12 分（6.7 步/秒） |
| 优化器 | AdamW，lr 1e-5 |
| 最终 loss | 0.060（l1 = 0.060） |

## 离线验证结果

开环评测：输入真实画面，预测未来 100 步动作，与人类示范对比 L1 误差（35 个采样点）。
基线为「始终输出训练集平均动作」（10.15°），用于确认模型确实在利用视觉输入，而非依赖动作先验。

| checkpoint | 平均 L1 | 中位 | 最差样本 | 相对基线 |
|---|---|---|---|---|
| 未训练 | 35.74° | — | — | 差 227% |
| 10K | 3.48° | 3.27° | 9.61° | 好 66% |
| 20K | 2.67° | 2.31° | 9.44° | 好 74% |
| 30K | 2.64° | 2.13° | 11.02° | 好 74% |
| 50K | 2.06° | 1.90° | 8.28° | 好 80% |
| 80K | 1.41° | 1.31° | 2.66° | 好 86% |
| **100K（本版本）** | **1.32°** | **1.15°** | **4.14°** | **好 87%** |

30K 附近出现的误差平台是暂时的，50K 之后继续下降，说明 100K 步不是过度训练。

## 使用方法

```python
from lerobot.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained("w3170700600/lerobot_act_so101_pickplace")
policy.eval()
```

实机推理（需连接 SO-101 主从臂）：

```bash
lerobot-rollout \
  --robot.type=so101_follower --robot.port=COM4 --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --policy.path=w3170700600/lerobot_act_so101_pickplace \
  --task="Grab the green cube" --duration=60
```

## 局限

- 单目相机，无腕部相机，精细对准能力受限
- 单任务模型，不接受语言指令（ACT 无语言分支）
- 训练物体为绿色方块，换物体或换背景未见泛化验证
- 相机位置必须与训练时一致，否则性能显著下降

## 引用

```bibtex
@article{zhao2023learning,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Zhao, Tony Z. and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={Robotics: Science and Systems},
  year={2023}
}
```
