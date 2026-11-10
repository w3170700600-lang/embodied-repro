# π0.5 微调执行文档（对齐 LeRobot π0.5 官方文档）

> 路线：`huggingface/lerobot` 官方 π0.5 文档 + SO-101 文档。
> 来源标记约定：
> - `[DOC]` = LeRobot 官方文档原文
> - `[实测]` = 已在 `D:\lerobot-data\so101_test` 上验证过的结论
> - `[推算]` = 文档没有、按显存/经验推的，**必须实测确认**

---

## 1. 环境

```bash
conda create -n pi05 python=3.10 -y && conda activate pi05
pip install 'lerobot[pi,training]'          # [DOC] 0.6.x 起 base 包不含训练代码
hf auth login                                # [DOC] 需先接受 google/paligemma-3b-pt-224 许可
```

国内网络：`export HF_ENDPOINT=https://hf-mirror.com`

---

## 2. 数据现状 [实测]

| 项 | 值 |
|---|---|
| 版本 | v3.0，布局合规 |
| 规模 | 56 条 / 18794 帧 / 30fps / 10.4 分钟 |
| 任务 | 单任务 `Grab the green cube` |
| 动作/状态 | 6 维关节位置（pan/lift/elbow/wflex/wroll/gripper） |
| 视觉 | 1 路 `observation.images.front` 640×480 |

已知问题：
- ep3 开头静止 184 帧（6.1s）；ep0/1/3 长 651/519/561 帧（中位 313）
- 34/56 条初始位姿两两距离 < 3°（起点物理固定）
- 方块占画面 4.4%（28×26px），缩到 224 后约 10px

---

## 3. 归一化：用 MEAN_STD，不要重算 [实测 + DOC]

### 实测事实（这一条更正了先前的判断）

| 项 | stats.json | 实际数据 | 差 |
|---|---|---|---|
| action mean/std（6 维） | — | — | **全部 0.00** |
| state mean 偏差 | — | — | **0.0** |
| wrist_roll q99 | −2.20 | +19.13 | 21.3° |
| elbow_flex q01 | −43.10 | −68.53 | 25.4° |
| shoulder_pan q99 | 48.77 | 59.82 | 11.1° |

**结论修正**：`stats.json` 与这份数据**同源**（mean/std 分毫不差），不是"从别的数据集继承来的"。
问题只出在 **q01/q99**：它是按逐 episode 聚合出来的，比真实的全局分位数**窄**。
如果 `episode_index max=57 / index max=19334` 这两项也和当前数据对不上，那是旧索引编号的残留。

### 影响

π0.5 默认对 STATE/ACTION 用分位数归一化 `[DOC]`。用这份偏窄的 q01/q99 归一化，
超出部分会被压到边界 → **典型症状：模型动作幅度偏小、够不到远端**。

### 决策：用 MEAN_STD 归一化（一步绕开，不碰数据）

```bash
--policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'
```

理由：`mean/std` 已实测与数据完全一致，可直接用；而 `q01/q99` 需要重算才有意义。
这个写法也是官方示例命令采用的 `[DOC]`。

> 备选：`lerobot-edit-dataset --operation.type recompute_stats --operation.overwrite true`
> 但注意结果写在 `$HF_LEROBOT_HOME` 下，训练要补 `--dataset.root=`；且若聚合逻辑不变，
> 重算后可能得到同样偏窄的分位数——不如直接用 MEAN_STD 干净。

---

## 4. 训练（3090 24G 保守配置）

```bash
lerobot-train \
  --dataset.repo_id=你的名/so101_test \
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
  --job_name=pi05_so101 \
  --batch_size=4 \
  --num_workers=4 \
  --steps=5000 \
  --save_freq=1000
```

参数归属：

| 参数 | 来源 | 说明 |
|---|---|---|
| `normalization_mapping` | `[DOC]` | 官方示例写法 |
| `n_action_steps` | `[DOC]` | 用 `pretrained_path` 时**必须显式给**，否则回退默认 50 |
| `empty_cameras` | `[推算]` | π0.5 期望 3 路图像，你只有 1 路 → 补 2 个空槽，**跑起来后查日志确认绑定** |
| `dtype` / `gradient_checkpointing` | `[DOC]` | 官方标为省显存手段 |
| `freeze_vision_encoder` / `train_expert_only` | `[DOC]` | 官方给的"冻结 VLM 省显存"组合，成功率略降 |
| `batch_size=4` / `num_workers=4` | `[推算]` | 官方示例是 80G 卡 batch 64，24G 无参考值 |
| `steps=5000` | `[推算]` | 先跑通再谈收敛 |

`--policy.path` vs `--policy.pretrained_path` `[DOC]`：

| | `--policy.path` | `--policy.pretrained_path` |
|---|---|---|
| 加载 | 权重 + config | **只加载权重** |
| 特征名 | 来自 checkpoint | 来自数据集 |
| `n_action_steps` | 继承 | **回退默认 50** |
| `--policy.type` | 必须省略 | 必须写 |

跑通后再逐项放开：`batch_size 4→8` → `train_expert_only=false` → `freeze_vision_encoder=false`。

---

## 5. 推理

```bash
# 策略服务 [DOC]
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_so101 --policy.dir=checkpoints/.../5000
```
LeRobot 侧用 rollout 流程；上真机前先在仿真/空跑确认 `select_action` 输出 shape 与频率。

---

## 6. 阶段 2：补数据（相机不动，只移动方块）

录制命令 `[DOC]`（SO-101 版）：

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_leader \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/so101_test_YYYYMMDD_HHMMSS \
  --dataset.root=/path/to/so101_test \
  --dataset.num_episodes=40 \
  --dataset.single_task="Grab the green cube" \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2
```

续录关键点 `[DOC]`：
- 加 `--resume=true`，且 `--dataset.repo_id` 必须是**带时间戳的完整名**
- `--dataset.num_episodes` 在这里表示**本次新增数量**，不是总集数
- `--resume` 时 `--dataset.root` 是**必填**

录制键位 `[DOC]`：`→`/`n` 进入下一条，`←`/`r` 丢弃重录，`ESC`/`q` 结束并编码。

采集建议 `[DOC]`：至少 50 条、**每个位置 10 条**、相机固定、物体始终在视野内。

⚠️ 文档明确的坑 `[DOC]`：**帧时间戳由索引推导**，实际只跑到 25Hz 数据集仍会声称 30Hz
（表现为动作"变快"）。录完务必看 cadence 报告确认实际帧率。

---

## 7. 待实测清单

- [ ] `empty_cameras=2` 是否真的正确绑定唯一那路相机（看训练日志的输入 key）
- [ ] `batch_size=4` 在 24G 上是否 OOM；能开多大
- [ ] AV1 视频在训练机的解码（torchcodec/pyav + ffmpeg libdav1d）
- [ ] `freeze_vision_encoder` + `train_expert_only` 的成功率代价有多大
- [ ] 5~10 条 held-out 上的离线动作误差，以及真机闭环成功率

## 8. 出处

- `huggingface.co/docs/lerobot/pi05` — 安装 extras、pretrained_path、normalization_mapping、
  n_action_steps 默认值、q01/q99 报错、freeze/train_expert_only、PEFT 说明、参数默认值表
- `huggingface.co/docs/lerobot/so101` — `so101_follower`/`so101_leader`、6 自由度、校准命令
- `huggingface.co/docs/lerobot/il_robots` — record 命令、续录规则、键位、采集建议、fps 陷阱
- `Physical-Intelligence/openpi` — 仅作方法论骨架（转数据→算统计→训练→serve），命令未采用
