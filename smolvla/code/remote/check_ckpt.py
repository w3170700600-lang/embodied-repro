# -*- coding: utf-8 -*-
"""检查导出的 checkpoint 能否被 lerobot 正常加载。参数：checkpoint 目录"""
import sys

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/full32/checkpoints/015000/pretrained_model"
p = SmolVLAPolicy.from_pretrained(path)
n = sum(x.numel() for x in p.parameters())
cams = [k for k in p.config.input_features if "image" in k]
print("加载成功 | 参数量 %.1fM" % (n / 1e6))
print("chunk_size=%d n_action_steps=%d num_steps=%d" % (
    p.config.chunk_size, p.config.n_action_steps, p.config.num_steps))
print("相机键:", cams)
print("state 维:", p.config.input_features["observation.state"]["shape"] if isinstance(
    p.config.input_features["observation.state"], dict) else p.config.input_features["observation.state"].shape)
