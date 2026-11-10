#!/usr/bin/env bash
# 在 A4000 机器上一次性把 SmolVLA 环境装好（约 10~20 分钟，取决于网速）
# 前提：这台机器上有 conda，且能 sudo 装系统包（ffmpeg 那步要手动跑一次）

set -euo pipefail

ENV_NAME="${ENV_NAME:-smolvla}"
LEROBOT_VER="${LEROBOT_VER:-0.6.1}"

echo "==> 0. 系统依赖（需要 sudo，若已装过可跳过）"
sudo apt update
sudo apt install -y ffmpeg libavformat-dev libavcodec-dev libavdevice-dev \
                    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev

echo "==> 1. 建独立 conda 环境（不碰队友的 robotwin 环境）"
conda create -n "${ENV_NAME}" python=3.12 -y
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "==> 2. 装 lerobot + smolvla 额外依赖"
python -m pip install --upgrade pip
pip install "lerobot[smolvla]==${LEROBOT_VER}"

echo "==> 3. 国内镜像 + 拉预训练权重（约 900MB）"
export HF_ENDPOINT=https://hf-mirror.com
hf download lerobot/smolvla_base

echo "==> 4. 自检"
python - <<'PY'
import torch, lerobot
print("lerobot:", lerobot.__version__)
print("torch  :", torch.__version__)
print("cuda   :", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu    : {p.name}  {p.total_memory/1024**3:.1f} GB")
PY

echo
echo "完成。下一步："
echo "  conda activate ${ENV_NAME}"
echo "  lerobot-train --help | head -5     # 确认命令在"
echo "  python check_dataset.py ~/datasets/so101_test   # 再确认一次数据（可选）"
echo "  STEPS=200 BATCH=8 bash train_smolvla_a4000.sh   # 冒烟"
