# -*- coding: utf-8 -*-
"""用 FK 看"夹爪闭合后末端抬升多少"，为成功判据定阈值。"""
import glob
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fk_lib import fk  # noqa: E402

files = sorted(glob.glob(r"D:/lerobot-data/so101_test/data/**/*.parquet", recursive=True))
df = pd.concat([pd.read_parquet(f) for f in files])

rows = []
for ep, g in df.groupby("episode_index"):
    g = g.sort_values("frame_index")
    st = np.stack(g["observation.state"].values)
    z = np.array([fk(q)[2, 3] for q in st]) * 1000.0
    gr = st[:, 5]
    open_i = int(np.argmax(gr))
    # 夹爪从张开回到闭合的时刻
    close_i = None
    for i in range(open_i, len(gr)):
        if gr[i] < 30:
            close_i = i
            break
    if close_i is None:
        rows.append((int(ep), len(st), round(gr.max(), 0), round(z[0], 0), None, None, None))
        continue
    z_g = z[close_i]
    rise = z[close_i:] - z_g
    rows.append((int(ep), len(st), round(gr.max(), 0), round(z[0], 0),
                 round(z_g, 0), round(rise.max(), 0), int(np.argmax(rise))))

print("ep frames gr_max z0 z_grasp max_rise rise_idx")
for r in rows:
    print(r)

rises = [r[5] for r in rows if r[5] is not None]
print("\n闭合后末端抬升 mm: min %.0f  p10 %.0f  中位 %.0f  p90 %.0f  max %.0f"
      % (min(rises), np.percentile(rises, 10), np.median(rises),
         np.percentile(rises, 90), max(rises)))

# 闭合前（接近阶段）末端最高能到多少，用于确认"抬升只发生在闭合之后"
print("\n起始末端高度 mm: 中位 %.0f" % np.median([r[3] for r in rows]))
print("闭合时刻末端高度 mm: 中位 %.0f" % np.median([r[4] for r in rows if r[4] is not None]))
