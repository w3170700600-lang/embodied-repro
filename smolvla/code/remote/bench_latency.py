# -*- coding: utf-8 -*-
"""测单次推理延迟（决定能不能跟上 30fps 闭环）。"""
import sys
import time

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ROOT = "/data/workspace/ei/challenge-532514/smolvla"
CKPT = f"{ROOT}/outputs/full32/checkpoints/015000/pretrained_model"

policy = SmolVLAPolicy.from_pretrained(CKPT).to("cuda").eval()
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT)
ds = LeRobotDataset("local/so101_test", root=f"{ROOT}/datasets/so101_test")
item = ds[0]
obs = {"observation.images.front": item["observation.images.front"],
       "observation.state": item["observation.state"],
       "task": "Grab the green cube"}
batch = pre(obs)

for _ in range(3):
    policy.reset()
    policy.predict_action_chunk(batch)

ts = []
for _ in range(10):
    policy.reset()
    t0 = time.perf_counter()
    with torch.no_grad():
        a = policy.predict_action_chunk(batch)
        post(a)
    torch.cuda.synchronize()
    ts.append((time.perf_counter() - t0) * 1000)

ts.sort()
print(f"单次推理（含 10 步去噪 + 后处理）: 中位 {ts[len(ts)//2]:.0f} ms | 最快 {ts[0]:.0f} | 最慢 {ts[-1]:.0f}")
print(f"一次 chunk 覆盖 {policy.config.n_action_steps} 步 = {policy.config.n_action_steps/30*1000:.0f} ms 的动作")
print(f"结论: {'跟得上 30fps' if ts[len(ts)//2] < policy.config.n_action_steps/30*1000 else '跟不上，需要降频或减小 chunk'}")
