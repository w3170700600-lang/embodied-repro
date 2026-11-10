# -*- coding: utf-8 -*-
"""验证"末端相对最低点抬高 >60mm 且夹爪闭合"作为成功判据：触发时机 / 误触发 / 余量。"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fk_lib import fk  # noqa: E402

files = sorted(glob.glob(r"D:/lerobot-data/so101_test/data/**/*.parquet", recursive=True))
df = pd.concat([pd.read_parquet(f) for f in files])

THR = 60.0
tails, early = [], 0
print("ep   n   z0   z_min  min_i  fire_i  fire_z  余帧  夹爪@fire")
for ep, g in df.groupby("episode_index"):
    g = g.sort_values("frame_index")
    st = np.stack(g["observation.state"].values)
    z = np.array([fk(q)[2, 3] for q in st]) * 1000.0
    gr = st[:, 5]
    zmin, fire = z[0], None
    for i in range(len(z)):
        zmin = min(zmin, z[i])
        if z[i] - zmin > THR and gr[i] < 35:
            fire = i
            break
    min_i = int(np.argmin(z))
    if fire is None:
        print(f"{ep:3d} {len(z):4d} {z[0]:5.0f} {z.min():6.0f} {min_i:5d}    未触发")
        continue
    if fire < min_i:      # 在到达最低点之前就触发 = 误判（还没抓）
        early += 1
    tail = len(z) - fire
    tails.append(tail)
    print(f"{ep:3d} {len(z):4d} {z[0]:5.0f} {z.min():6.0f} {min_i:5d} {fire:7d} {z[fire]:6.0f}"
          f" {tail:5d} {gr[fire]:6.1f}")

print(f"\n触发 {len(tails)}/56，其中早于最低点（疑似误判）{early} 个")
print("触发后剩余帧: 中位 %.0f 最多 %d（30fps ≈ %.1fs）"
      % (np.median(tails), max(tails), np.median(tails) / 30))
