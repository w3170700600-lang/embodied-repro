# -*- coding: utf-8 -*-
"""用数据集 API 精确取每条 episode 的首帧，量方块位置（cv2 按时间戳跳转不可靠，已弃用）。

    python check_cube_positions.py --last 18     # 只看最近 18 条（推荐，每批录完跑一次）
    python check_cube_positions.py --all         # 全部（慢，几分钟）
"""
import argparse
import glob

import numpy as np
import pandas as pd

ROOT = "D:/WorkBuddyData/Workspace/2026-11-06-14-41-14/so101_more"


def cube_center(img_t):
    """HSV 绿色检测。RGB 阈值对暗绿不可靠（桌面阴影下会整个漏掉），已弃用。"""
    from PIL import Image

    a = (img_t.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    hsv = np.array(Image.fromarray(a).convert("HSV")).astype(int)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = (h > 50) & (h < 115) & (s > 70) & (v > 40)
    ys, xs = np.nonzero(m)
    if len(xs) < 30:
        return None
    cx, cy = int(np.median(xs)), int(np.median(ys))
    keep = (abs(xs - cx) < 60) & (abs(ys - cy) < 60)
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 30:
        return None
    return cx, cy, int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    eps = pd.concat([pd.read_parquet(f) for f in
                     sorted(glob.glob(ROOT + "/meta/episodes/chunk-*/file-*.parquet"))],
                    ignore_index=True)
    ds = LeRobotDataset(repo_id="local/so101_more", root=ROOT)
    eps = eps.sort_values("episode_index")
    if a.all or not a.last:
        sel = eps
    else:
        sel = eps.tail(a.last)
    print(f"数据集共 {ds.meta.total_episodes} 条，本次检查 {len(sel)} 条\n")

    pts = []
    print(f"{'ep':>4} {'方块中心':>12} {'尺寸':>9}  备注")
    small = 0
    for _, r in sel.iterrows():
        ep = int(r["episode_index"])
        idx = int(r["dataset_from_index"])
        try:
            c = cube_center(ds[idx]["observation.images.front"])
        except Exception as e:  # noqa: BLE001
            print(f"{ep:>4}  取帧失败 {type(e).__name__}")
            continue
        if c is None:
            print(f"{ep:>4} {'未检测到':>12}")
            small += 1
            continue
        w, h = c[2], c[3]
        note = ""
        if w < 25:
            note = "<- 偏小，疑似遮挡"
            small += 1
        print(f"{ep:>4} ({c[0]:>3},{c[1]:>3}) {w:>4}x{h:<4} {note}")
        pts.append((ep, c[0], c[1]))

    if small:
        print(f"\n注意: {small} 条方块偏小或未检测到")

    if len(pts) < 2:
        return
    P = np.array([[p[1], p[2]] for p in pts], float)
    print("\n=== 汇总 ===")
    print(f"x 范围 {P[:,0].min():.0f}~{P[:,0].max():.0f}  跨度 {P[:,0].max()-P[:,0].min():.0f}px")
    print(f"y 范围 {P[:,1].min():.0f}~{P[:,1].max():.0f}  跨度 {P[:,1].max()-P[:,1].min():.0f}px")
    D = np.abs(P[:, None, :] - P[None, :, :]).max(axis=2)
    np.fill_diagonal(D, np.inf)
    print(f"两两最近邻 中位 {np.median(D.min(axis=1)):.0f}px")
    print("判读: 每批 6 条内部应 <10px（同一位置），批与批之间应 >40px（不同位置）")


if __name__ == "__main__":
    main()
