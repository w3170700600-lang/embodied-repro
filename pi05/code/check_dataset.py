# -*- coding: utf-8 -*-
"""采集自查：条数 / 时长 / 静止帧 / 起点离散度 / 夹爪闭合。

    python check_dataset.py
"""
import glob
import json

import numpy as np
import pandas as pd

ROOT = "D:/WorkBuddyData/Workspace/2026-11-06-14-41-14/so101_more"
NAMES = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]


def main():
    files = sorted(glob.glob(ROOT + "/data/chunk-*/file-*.parquet"))
    if not files:
        print("找不到数据：", ROOT)
        return
    info = json.load(open(ROOT + "/meta/info.json"))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "action", "observation.state"])
                    for f in files], ignore_index=True)
    A = np.stack(df["action"].to_numpy())
    S = np.stack(df["observation.state"].to_numpy())
    L = df.groupby("episode_index").size()

    print("=" * 58)
    print(f"条数 {len(L)}   总帧 {len(df)}   声明 {info['total_episodes']} 条 / {info['total_frames']} 帧")
    print(f"单条帧数 平均 {int(L.mean())}  最短 {L.min()}  最长 {L.max()}")
    print(f"单条时长 平均 {L.mean()/30:.1f}s  最长 {L.max()/30:.1f}s")

    dd = np.abs(np.diff(S, axis=0)).max(axis=1)
    idle = 100 * (dd < 1e-3).mean()
    flag = "OK" if idle < 25 else ("偏高" if idle < 35 else "过高，建议裁")
    print(f"静止帧占比 {idle:.1f}%   [{flag}]")

    init = np.stack([g["observation.state"].to_numpy()[0] for _, g in df.groupby("episode_index")])
    print("起点离散度（换位置后 pan/lift 应该明显变大）:")
    for i, n in enumerate(NAMES):
        span = init[:, i].max() - init[:, i].min()
        print(f"  {n:6s} 跨度 {span:6.1f}°")

    D = np.abs(init[:, None, :] - init[None, :, :]).max(axis=2)
    np.fill_diagonal(D, np.inf)
    print(f"起点两两最近邻 中位 {np.median(D.min(axis=1)):.1f}°")

    close_ok = 100 * np.mean([g["action"].apply(lambda x: x[5]).max() > 25
                              for _, g in df.groupby("episode_index")])
    print(f"夹爪闭合过(>25)的条数占比 {close_ok:.0f}%   [应接近 100%]")

    print("\n逐条明细（按帧数降序，>450 帧的重点看）:")
    rows = []
    for ep, sub in df.groupby("episode_index"):
        s = np.stack(sub["observation.state"].to_numpy())
        a = np.stack(sub["action"].to_numpy())
        d = np.abs(np.diff(s, axis=0)).max(axis=1)
        rows.append((int(ep), len(s), len(s) / 30, 100 * (d < 1e-3).mean(), a[:, 5].max()))
    rows.sort(key=lambda r: -r[1])
    print(f"{'ep':>4} {'帧':>5} {'秒':>6} {'静止%':>7} {'夹爪max':>8}")
    for r in rows:
        warn = "  <-- 过长" if r[1] > 450 else ("  <-- 静止偏高" if r[3] > 45 else "")
        print(f"{r[0]:>4} {r[1]:>5} {r[2]:>6.1f} {r[3]:>7.1f} {r[4]:>8.1f}{warn}")
    print("=" * 58)


if __name__ == "__main__":
    main()
