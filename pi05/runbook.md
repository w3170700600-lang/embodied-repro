# π0.5 × SO-101 × 3090(24G) LeRobot 路线执行清单

> 目标：用 `huggingface/lerobot`（而不是 `Physical-Intelligence/openpi`）在 SO-101 上把 π0.5 微调跑通。
> 前提：训练在 **3090 服务器（Linux）** 上做，本机 Windows 只当终端用。
> 版本锚点：lerobot ≥ 0.6.x（0.6.0 起基础包变轻，pi 微调必须同时装 training extras）。

---

## 0. 先认清代价（决定要不要走这条）

| 项 | 官方参考值 | 3090 24G 的现实 |
|---|---|---|
| 推理 | > 8 GB | 可以 |
| LoRA 微调 | > 22.5 GB | 边缘（openpi 的 JAX 路径才有 LoRA，LeRobot 侧要用 PEFT） |
| 全参微调 | > 70 GB | 不行 |

LeRobot 官方 pi05 示例是 **单张 80G 卡、batch_size=64** 写的。24G 上必须改：
`batch_size ≤ 8` + `gradient_checkpointing=true` + `dtype=bfloat16`，大概率还要
`freeze_vision_encoder=true` + `train_expert_only=true`（冻结 VLM 只训 action expert，成功率会掉一点）。

**先跑通再加码**：第一轮只求 200~500 step 不 OOM，再谈收敛。

---

## 1. 环境（服务器上，别在 Windows 装）

```bash
conda create -n pi05 python=3.10 -y && conda activate pi05
# 必须带 training（0.6.x 起 base 包不含训练代码）；要驱动机器人再加 core_scripts,feetech
pip install 'lerobot[pi,training]'
```

权重是 **gated** 的（PaliGemma tokenizer）：

1. 浏览器里接受 `google/paligemma-3b-pt-224` 许可
2. `hf auth login`（老版本是 `huggingface-cli login`）

国内网络先 `export HF_ENDPOINT=https://hf-mirror.com`。

---

## 2. 数据：LeRobot 格式是硬门槛

有真机就直接录（这一层是 openpi 没有的，省掉写映射）：

```bash
lerobot-find-port
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=f1
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=l1
# 录制命令见 docs 的 il_robots 页（so101 页未给），核心参数是 robot.type/teleop.type/dataset.repo_id/fps
```

没有真机就把已有 LeRobot 数据集（Aloha / 自采 / LIBERO）直接用；相机 key 和策略对不上时用 `--rename_map` 规范化，**别去改数据集**。

SO-101 = 6 自由度（pan/lift/elbow/wrist_flex/wrist_roll/gripper），state/action 都是 6 维 + 相机 1~3 路，config 里 `action_dim` 要对齐成模型期望的 32 维填充（LeRobot 内部处理，openpi 要你自己写）。

### 已有 56 条真机数据时的前置体检

训练前**必须先体检**，否则 56 条的问题（静止帧、丢维度、缺 stats）会被
"loss 降了但机器人不动"这个现象掩盖掉：

```bash
python check_lerobot_dataset.py --root /path/to/your_so101_dataset
```

重点看四项：静止帧比例（>40% 要裁）、每任务条数（单任务 <40 条只够做玩具 demo）、
动作维度是否有整条不变（漏接）、`stats.json` 是否有 q01/q99。

---

## 2.5 体检结论：so101_test（2026-11-07 实测 56 条）

**结构合规**：v3.0 布局正确，56 条 / 18794 帧 / 30fps / 单任务（`Grab the green cube`）/ 6 维关节位置 + 1 路 front 640×480。

**P0 必须处理**

1. **`meta/stats.json` 是旧的，不是当前这份数据算的**（实测证据）：
   | 项 | stats.json | 实际数据 |
   |---|---|---|
   | wrist_roll q99 | -2.20 | +19.13（差 21.3°）|
   | elbow_flex q01 | -43.10 | -68.53（差 25.4°）|
   | episode_index max | 57 | 55 |
   | index max | 19334 | 18793 |

   实际分位数范围**比 stats 更宽** → 归一化后动作会超出 [0,1] 被截断，
   典型症状是"训出来的动作幅度偏小、够不到远端"。**训练前必须重算。**
2. **ep3 开头静止 184 帧（6.1s）**；ep0/1/3 长度 651/519/561 帧，远超中位 313（10.4s）。
   这 3 条拖沓样本会被正常样本稀释，建议裁首尾或直接删 ep3。
3. **34/56 条起点几乎相同**（初始位姿两两最近邻中位 2.6°），方块固定放在白色纸垫上
   → 模型会"背轨迹"。接受这个上限，或补数据时刻意移动起点。
4. **视频是 AV1 (crf30)**，训练机需要 av1 解码（torchcodec/pyav + ffmpeg libdav1d）。
   报 decode 错误就批量转 h264。

**P1 决定上限**

5. 只有 1 路第三人称，**夹爪在抓取相位会遮挡方块**（8s 帧可见），无腕部近景。
6. **方块只占画面 4.4%（28×26 px）**，缩到 224 后约 10 px，而 SigLIP patch = 14 px
   → 物体不足 1 个 patch，位置分辨率约 4 cm，低于抓取所需精度。
   对策：裁下半区再送模型 / 加腕部相机 / 重采时把相机拉近。

**数据本身没问题的点**：静止帧 18.6%（健康）；action−state 均值偏差 ≤3.2°（leader/follower
标定对齐良好），偏差主要来自跟踪滞后（与速度相关 0.46~0.78），说明采集链路干净。

## 2.6 单相机（桌面固定）约束下的策略

现状：**1 路 front 相机，固定在桌面**，视角稳定 → 现有 56 条内部一致，可直接用。

推论：**任何改变相机位置/角度的动作都会让这 56 条作废**（视觉分布变了）。
所以分三步走，成本递增：

| 阶段 | 做什么 | 代价 | 收益 |
|---|---|---|---|
| 1（本周） | 直接用现有 56 条训 baseline | 0 采集成本 | 判断单相机够不够 |
| 2 | **相机不动**，只移动方块位置再录 40~60 条 | 约 1 天 | 直接解决"起点集中"，新旧数据可直接续录合并 |
| 3 | 加腕部相机 | 需全量重采 | 解决抓取相位遮挡，上限提升最大 |

阶段 2 是性价比最高的一步：**相机不换 = 数据可续录到同一个 repo_id**（`lerobot-record --resume`），
不用重新适配。阶段 3 才需要重采，放在 baseline 结论出来之后再决定。

单相机训练配置要点：π0.5 base 期望 3 路图像输入，只有 1 路时必须补空槽
`--policy.empty_cameras=2`，否则会因为缺 wrist 相机报错或学到错位的输入绑定。

若重采，建议把相机升高到接近俯视 30~45°，能明显减少夹爪对物体的遮挡。

---

## 3. 归一化统计量（第一个必炸点）

π0.5 默认对 STATE/ACTION 用 **分位数归一化**，`meta/stats.json` 必须有 `q01/q99`。旧数据集只有 min/max/mean/std，会在第一个 batch 抛：

```
ValueError: QUANTILES normalization mode requires q01 and q99 stats
```

三种解法，按场景选：

```bash
# A. 重算（注意：结果写到 $HF_LEROBOT_HOME，不是原来的缓存目录）
lerobot-edit-dataset --repo_id your_ds --new_repo_id your_ds \
  --operation.type recompute_stats --operation.overwrite true
# 训练时补 --dataset.root=$HF_LEROBOT_HOME/your_ds

# B. 不碰数据，显式改成 mean/std 归一化（官方示例就是这么干的）
# 命令行加：--policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'

# C. 直方图估计（只重算 STATE/ACTION，不动图像，省下载）
python src/lerobot/scripts/augment_dataset_quantile_stats.py --repo-id=your_ds --overwrite --skip-images
```

---

## 4. 训练（3090 保守版）

```bash
lerobot-train \
  --dataset.repo_id=你的hf用户名/so101_task \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
  --policy.n_action_steps=20 \
  --policy.empty_cameras=2 \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=./outputs/pi05_so101 \
  --job_name=pi05_so101_smoke \
  --batch_size=4 \
  --num_workers=4 \
  --steps=300 \
  --save_freq=100
```

跑通后逐步放开：`batch_size 4→8` → `train_expert_only=false` → `freeze_vision_encoder=false`，每次只动一个。

**`--policy.path` vs `--policy.pretrained_path`（关键陷阱）**

| | `--policy.path` | `--policy.pretrained_path` |
|---|---|---|
| 加载 | 权重 + checkpoint 的 config.json | 只加载权重 |
| 特征名 | 来自 checkpoint | 来自你的数据集 |
| `n_action_steps` 等 | 继承 | **回退默认值（50）** |
| `--policy.type` | 必须省略 | 必须写 |

用 `pretrained_path` 时**一定要显式写** `n_action_steps` 和 `empty_cameras`，否则会在真机上出现"动作块长度 50"这种莫名其妙的延迟。

---

## 5. 推理 / 上机

```bash
# 策略服务器 + 客户端两段式（和 openpi 一个思路）
lerobot-... serve  # 或按 docs 的 rollout 页（0.6.x 起部署迁到 lerobot-rollout）
```

先在仿真或空跑里验证 `policy.select_action` 输出的 shape 和频率，再接到 SO-101 上。真机第一次务必：
人守着 + 急停可达 + `max_relative_target` 限速。

---

## 6. 坑清单

- [ ] 0.6.x 之后 `pip install lerobot` 不带训练代码 → 必须 `[pi,training]`
- [ ] 教程里出现 `python -m lerobot.scripts.train` 的都是老版本，别照抄
- [ ] `transformers` 版本敏感；openpi PyTorch 路径会 patch transformers 且污染 uv cache（LeRobot 路线不需要这步）
- [ ] `recompute_stats` 写库位置 ≠ 训练读取位置（最容易白跑一轮）
- [ ] batch_size 是**全局**还是**每卡**，在 openpi 和 LeRobot 里语义不同，多卡前先确认
- [ ] 24G 卡别开 `compile_model=true` 首发，编译期显存峰值会骗你
