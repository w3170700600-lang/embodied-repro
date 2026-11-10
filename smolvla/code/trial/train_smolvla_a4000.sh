#!/usr/bin/env bash
# SmolVLA 微调 so101_test（56 ep / 18794 帧 / 单路 front 相机）
# 对齐官方教程 lerobot 0.6.1，参数已按你的真实数据填好，只需改 DATASET_ROOT。
# 先冒烟：STEPS=200 BATCH=8 bash train_smolvla_a4000.sh

set -euo pipefail

# ============ 改这里 ============
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/so101_test}"   # A4000 上数据集的绝对路径
OUT_DIR="${OUT_DIR:-$HOME/outputs/train/smolvla_so101_56ep}"
JOB_NAME="${JOB_NAME:-smolvla_so101_56ep}"
STEPS="${STEPS:-15000}"        # 正式 15000；冒烟 200
BATCH="${BATCH:-16}"           # 16GB 起点；单路相机可先试 32，OOM 就 16 / 8
# ================================

# 你的数据集只有一路相机 front，映射到 smolvla_base 期望的 camera1。
# camera2 / camera3 缺失会被源码直接跳过（modeling_smolvla.py:391-432），不用补。
RENAME_MAP='{"observation.images.front": "observation.images.camera1"}'

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/so101_test \
  --dataset.root="${DATASET_ROOT}" \
  --policy.device=cuda \
  --rename_map="${RENAME_MAP}" \
  --batch_size="${BATCH}" \
  --steps="${STEPS}" \
  --num_workers=6 \
  --save_freq=2000 \
  --log_freq=20 \
  --seed=1000 \
  --trackio.enable=false \
  --output_dir="${OUT_DIR}" \
  --job_name="${JOB_NAME}"

# ---------------------------------------------------------------------------
# 参数都按你这批数据定过，别随手改：
#
# · --rename_map 必需。官方："Without it, training stops with a feature mismatch error."
#   只有一路相机时只映射 camera1，剩下两个槽位自动跳过。
#
# · --num_workers=6：你这批视频是 AV1 编码（info.json 里 video.codec=av1），
#   CPU 解码比 h264 重，worker 太少会被数据加载卡住 GPU。
#   如果看到 torchcodec 相关报错，加一行 --dataset.video_backend=pyav
#   （数据集本来就是这个后端录的，pyav 一定解得开）。
#
# · 不要动 freeze_vision_encoder / train_expert_only：config 默认 True/True，
#   只训动作专家。16GB 靠这个活着，解冻必 OOM。
#
# · steps=15000：18794 帧 / batch 16 ≈ 1175 步/epoch → 约 12.8 个 epoch。
#   batch 降到 8 就把 steps 翻到 30000。
#
# · 续训用官方参数 --policy.pretrained_path=<上次的模型目录>。
#
# 冒烟正常的样子：20 步内 loss 从 1.x 往下走，nvidia-smi 显存稳定不涨。
