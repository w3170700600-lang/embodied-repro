# -*- coding: utf-8 -*-
"""检查数值范围：关节是否弧度、夹爪量纲、动作与状态是否一致。只读。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / ".pydeps"))

import pandas as pd

ROOT = Path(r"D:/lerobot-data/so101_test")
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

d = pd.read_parquet(ROOT / "data" / "chunk-000" / "file-000.parquet")
print("数据列:", list(d.columns))
print("行数:", len(d))
print("\n前 3 行 state / action：")
for i in range(3):
    st = d["observation.state"].iloc[i]
    ac = d["action"].iloc[i]
    print(f"  ep{d['episode_index'].iloc[i]} f{d['frame_index'].iloc[i]}")
    print("    state :", " ".join(f"{v:+.3f}" for v in st))
    print("    action:", " ".join(f"{v:+.3f}" for v in ac))

print("\n各关节取值范围（全文件）：")
import numpy as np

st = np.stack(d["observation.state"].values)
ac = np.stack(d["action"].values)
for j, n in enumerate(NAMES):
    print(f"  {n:14s} state[{st[:, j].min():+.3f}, {st[:, j].max():+.3f}]   "
          f"action[{ac[:, j].min():+.3f}, {ac[:, j].max():+.3f}]")

# 动作与状态的关系：act[t] 是否约等于 state[t+1]
delta = np.abs(ac[:-1] - st[1:]).mean(axis=0)
print("\n|action[t] - state[t+1]| 逐关节均值（越小说明动作就是下一位姿）：")
print("  " + "  ".join(f"{n}={v:.4f}" for n, v in zip(NAMES, delta)))
print(f"  总体均值 = {delta.mean():.4f}")
