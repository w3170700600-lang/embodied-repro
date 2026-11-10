#!/usr/bin/env bash
# A4000 上跑 SmolVLA 微调（so101_test）
# 环境变量：STEPS / BATCH / OUT / JOB / GPU / SAVE_FREQ
set -uo pipefail

ROOT=/data/workspace/ei/challenge-532514/smolvla
V=$ROOT/venv
export HF_HOME=$ROOT/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${GPU:-0}

STEPS=${STEPS:-200}
BATCH=${BATCH:-8}
JOB=${JOB:-smoke}
OUT=${OUT:-$ROOT/outputs/$JOB}
SAVE_FREQ=${SAVE_FREQ:-2000}
EPS=${EPS:-}                       # 可选：--dataset.episodes 的子集，如 [0,1,2]

EXTRA=()
if [ -n "$EPS" ]; then
  EXTRA+=(--dataset.episodes="$EPS")
  echo "使用 episode 子集，共 $(echo "$EPS" | tr -cd ',' | wc -c | awk '{print $1+1}') 条"
fi

echo "===== $(date '+%F %T') 开始  steps=$STEPS batch=$BATCH gpu=$CUDA_VISIBLE_DEVICES out=$OUT ====="
$V/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/so101_test \
  --dataset.root="$ROOT/datasets/so101_test" \
  --dataset.video_backend=pyav \
  "${EXTRA[@]}" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.scheduler_decay_steps="$STEPS" \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --num_workers=4 \
  --save_freq="$SAVE_FREQ" \
  --log_freq=10 \
  --seed=1000 \
  --wandb.enable=false \
  --output_dir="$OUT" \
  --job_name="$JOB"
echo "===== $(date '+%F %T') 退出码 $? ====="
