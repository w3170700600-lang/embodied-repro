"""
check_episodes_visual.py -- 用视觉证据逐条复核：每条示范是否真的把方块抬起来了。

关键实现细节（踩过的坑）：
  * 不能用 cv2.CAP_PROP_POS_FRAMES 随机 seek —— 对 mp4 定位不准，
    会读到别的 episode 的帧，导致基线算错。改为顺序读帧 + 帧计数。
  * 基线取区间内方块 y 的最大值（画面最下方 = 还在桌面），
    终点取 y 的最小值（画面最上方 = 抬起来了）。差值即抬升量。
"""
import sys
import glob
import json
import os

import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_cube import find_green

CAM_KEY = "observation.images.front"
LIFT_THRESHOLD = 0.15      # 抬升超过画面高度的 15% 判为成功
MIN_AREA, MAX_AREA = 120, 4000


def probe_range(cap, f_from, f_to, step, fps):
    """顺序读到 f_to，只在 [f_from, f_to) 区间内检测。返回 y 列表。"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap.read()
    ys = []
    idx = 0
    while idx < f_to:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= f_from and (idx - f_from) % step == 0:
            r = find_green(frame)
            if r is not None and MIN_AREA <= r[2] <= MAX_AREA:
                ys.append(r[1])
        idx += 1
    return ys


def main(root):
    info = json.load(open(os.path.join(root, "meta", "info.json"), encoding="utf-8"))
    print(f"数据集: {info.get('total_episodes')} 条 / {info.get('total_frames')} 帧 / {info.get('fps')} fps")

    parts = [pd.read_parquet(f) for f in
             sorted(glob.glob(os.path.join(root, "meta", "episodes", "chunk-*", "file-*.parquet")))]
    eps = pd.concat(parts, ignore_index=True)

    col_file = f"videos/{CAM_KEY}/file_index"
    col_from = f"videos/{CAM_KEY}/from_timestamp"
    col_to = f"videos/{CAM_KEY}/to_timestamp"

    rows = []
    cap, cur_path, cur_fps = None, None, 30
    for _, r in eps.sort_values("episode_index").iterrows():
        ep = int(r["episode_index"])
        fi = int(r[col_file])
        vpath = os.path.join(root, "videos", CAM_KEY, "chunk-000", f"file-{fi:03d}.mp4")
        if vpath != cur_path:
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(vpath)
            cur_path = vpath
            cur_fps = cap.get(cv2.CAP_PROP_FPS) or info.get("fps", 30)
            if not cap.isOpened():
                print(f"  ep{ep}: 打不开 {vpath}")
                continue

        f0 = int(float(r[col_from]) * cur_fps)
        f1 = int(float(r[col_to]) * cur_fps)
        step = max(1, (f1 - f0) // 50)
        ys = probe_range(cap, f0, f1, step, cur_fps)

        if len(ys) < 5:
            rows.append(dict(ep=ep, base=0, top=0, lift=0, ratio=0.0, ok=False, note=f"仅检测到{len(ys)}帧"))
            continue
        base = max(ys)     # 最低位置（桌面）
        top = min(ys)      # 最高位置（抬起）
        lift = base - top
        ratio = lift / base if base else 0
        rows.append(dict(ep=ep, base=base, top=top, lift=lift, ratio=ratio,
                         ok=ratio > LIFT_THRESHOLD, note=f"{len(ys)}帧"))

    if cap is not None:
        cap.release()

    print()
    print(f"{'ep':>4} {'桌面y':>6} {'最高y':>6} {'抬升px':>7} {'抬升%':>7}  判定")
    bad = []
    for x in rows:
        if not x["ok"]:
            bad.append(x["ep"])
        tag = "OK" if x["ok"] else "★可疑"
        print(f"{x['ep']:>4} {x['base']:>6} {x['top']:>6} {x['lift']:>7} {x['ratio']*100:>6.0f}%  {tag}  {x['note']}")

    ok_n = sum(1 for x in rows if x["ok"])
    print()
    print(f"合计 {len(rows)} 条，成功抬升 {ok_n} 条，可疑 {len(bad)} 条")
    if bad:
        print(f"可疑编号: {bad}")
        print("确认是坏示范再执行删除：")
        print(f'  D:\\lerobot-venv\\Scripts\\lerobot-edit-dataset.exe --operation.type=delete_episodes '
              f'--operation.episode_indices="{bad}" --repo_id=local/so101_test '
              f'--root="{root}" --new_repo_id=local/so101_test --new_root="{root}" --push_to_hub=false')
    return rows


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "D:/lerobot-data/so101_test"
    main(root)
