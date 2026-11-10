"""
detect_cube.py -- 通过颜色检测绿方块的位置，用于自动判定抓取是否成功。

思路：相机固定、方块是纯色，用 HSV 阈值找绿色连通域，取最大块的质心。
   - 起点 y（画面下方）   -> 方块在桌面上
   - 抬起后 y 明显变小    -> 方块被抬离桌面

用法：
  python detect_cube.py <图片或视频路径> [--all]
  python detect_cube.py D:/lerobot-data/so101_test/videos/.../file-000.mp4 --all
"""
import sys
import argparse

import cv2
import numpy as np

# 绿色方块的 HSV 范围（OpenCV: H 0-180, S 0-255, V 0-255）
# 这个范围覆盖偏黄绿到偏青绿，能容忍室内白平衡漂移
GREEN_LO = np.array([35, 80, 60])
GREEN_HI = np.array([90, 255, 255])


def find_green(img_bgr, min_area=150):
    """返回 (cx, cy, area) 或 None。只保留周长最接近方形的大连通域。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    # 开运算去噪，闭运算补洞
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue
        ar = w / float(h)
        # 方块的长宽比接近 1（透视下放宽到 0.4~2.5）
        if not (0.4 <= ar <= 2.5):
            continue
        if best is None or a > best[0]:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            best = (a, int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), mask)
    if best is None:
        return None
    area, cx, cy, _ = best
    return cx, cy, area


def analyze_video(path, step=15, out_dir=None):
    """逐帧检测，返回 (起始位置, 最高位置, 位移)。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"打不开: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    samples = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            r = find_green(frame)
            if r:
                samples.append((i, r[0], r[1], r[2]))
        i += 1
    cap.release()

    if not samples:
        print(f"  未检测到绿色方块（{path}）")
        return None

    ys = [s[2] for s in samples]
    start_y = samples[0][2]
    min_y = min(ys)          # 画面最上方 = 抬得最高
    lift_px = start_y - min_y
    lift_ratio = lift_px / start_y if start_y else 0

    print(f"  帧数 {n}  fps {fps:.0f}  采样 {len(samples)} 次")
    print(f"  起点 y={start_y}   最高 y={min_y}   抬升 {lift_px}px ({lift_ratio*100:.0f}%)")
    print(f"  概率判定: {'【成功】' if lift_ratio > 0.15 else '【失败/未抬起】'}  (阈值 15%)")
    return dict(start_y=start_y, min_y=min_y, lift_px=lift_px, lift_ratio=lift_ratio)


def analyze_image(path):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"读不到图片: {path}")
    r = find_green(img)
    if r is None:
        print(f"  未检测到绿色方块 ({path})")
        return None
    print(f"  {path}: 中心=({r[0]},{r[1]})  面积={r[2]:.0f}px")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--all", action="store_true", help="按视频处理，逐段分析")
    ap.add_argument("--step", type=int, default=15, help="采样间隔帧数")
    a = ap.parse_args()

    if a.all or a.path.lower().endswith((".mp4", ".avi", ".mov")):
        analyze_video(a.path, step=a.step)
    else:
        analyze_image(a.path)
