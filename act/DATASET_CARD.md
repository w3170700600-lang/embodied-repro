---
task_categories:
  - robotics
tags:
  - lerobot
  - so101
  - imitation-learning
  - manipulation
license: mit
---

# SO-101 抓取绿色方块 · 56 条示范

用 SO-101 主从臂遥操作采集的模仿学习数据集，遵循 [LeRobotDataset](https://github.com/huggingface/lerobot) v3.0 格式。
任务统一为：**Grab the green cube**（从桌面抓取绿色方块并抬离桌面）。

## 数据集概览

| 项 | 值 |
|---|---|
| 示范条数 | 56 |
| 总帧数 | 18,794 |
| 帧率 | 30 fps |
| 单条时长 | 8.8 ~ 21.7 秒（中位 10.4 秒） |
| 机器人 | SO-101 follower（6 自由度） |
| 采集方式 | leader 臂遥操作（leader-follower joint mapping） |
| codebase version | v3.0 |

## 目录结构

```
meta/          info.json, stats.json, tasks.parquet, episodes/
data/          chunk-000/file-000..006.parquet    # 状态与动作
videos/
  observation.images.front/
    chunk-000/file-000..006.mp4                   # 前视相机画面
```

## 字段说明

| 字段 | 形状 | 含义 |
|---|---|---|
| `observation.images.front` | (3, 480, 640) | 前视 USB 相机，固定在桌面 |
| `observation.state` | (6,) | 从动臂当前关节位置（度） |
| `action` | (6,) | 主臂给出的目标关节位置（度） |
| `task_index` | () | 任务索引，全部指向 `Grab the green cube` |

关节顺序：`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`。

> 注意：`action` 是**主臂**位置（意图），`observation.state` 是**从臂**实际位置，两者存在跟随延迟，这是遥操作采集的正常特征。

## 数据质量

逐条视觉核验（检测绿色方块在画面中的位移）：

- 56 条全部成功抬起物体，抬升幅度占画面高度 **37% ~ 70%**
- 单帧最大动作跳变 4.5 ~ 9.7 度
- 静止帧占比 29% ~ 61%（早期示范偏高，后期稳定在 35~45%）

## 采集配置

| 项 | 值 |
|---|---|
| 相机 | 单个 USB 相机，640×480 @ 30fps，桌面固定俯视 |
| 主臂 / 从臂 | 各自独立 DC 供电，CH343 USB 转串口 |
| 校准 | 逐臂关节行程标定，`homing_offset` + `range_min/max` |
| 录制工具 | `lerobot-record`（0.6.1） |

## 使用方法

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("w3170700600/so101_test")
print(len(ds), ds.fps)
```

## 局限

- 单相机视角，无腕部相机
- 物体位置变化范围约 ±3 cm，超出该范围未验证
- 背景包含少量杂物（纸箱、排插），未做严格清理
- 仅单一物体（绿色方块）、单一任务

## 引用

```bibtex
@article{zhao2023learning,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Zhao, Tony Z. and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={Robotics: Science and Systems},
  year={2023}
}
```
