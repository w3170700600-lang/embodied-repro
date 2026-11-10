# -*- coding: utf-8 -*-
"""相机自检：不开窗口、不需要人看，直接把结果打成文字 + 存一张标注图。

用法（必须先把 persistent_trial.py 停掉，否则相机被占用）：
    D:/smolvla-env/Scripts/python.exe check_cam.py

输出：
    _frames/trials/cam_check.png   —— 原始帧 + 检测框（没有框就是没认出来）
    _frames/trials/cam_check_raw.png —— 纯原始帧
终端会打印：分辨率、亮度、帧率、严格/宽松绿色像素数、候选色块 HSV。
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import auto_trial as AT  # noqa: E402
import cam_util  # noqa: E402

OUT_DIR = AT.FRAME_DIR


def main():
    print("识别相机...")
    CAM_INDEX, _ = cam_util.pick()
    print(f"打开相机 index={CAM_INDEX} ...")
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("!! 相机打不开。如果试验脚本还在跑，先 Ctrl+C 停掉它。")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    t0 = time.time()
    frames = []
    for _ in range(30):
        ok, fr = cap.read()
        if ok and fr is not None:
            frames.append(fr)
    dt = time.time() - t0
    cap.release()

    if not frames:
        print("!! 一帧都没读出来 —— 相机在，但不出图（换 USB 口 / 重新插一下）")
        return 1

    means = [float(f.mean()) for f in frames]
    img = frames[-1]
    print(f"读到 {len(frames)}/30 帧，{len(frames)/max(dt,1e-6):.1f} fps，"
          f"分辨率 {img.shape[1]}x{img.shape[0]}")
    print(f"亮度 mean: 最小 {min(means):.1f} 最大 {max(means):.1f} 末帧 {means[-1]:.1f}")
    if max(means) < 5:
        print("!! 所有帧几乎全黑 —— 相机没出图（不是方块的问题）")
        cv2.imwrite(os.path.join(OUT_DIR, "cam_check_raw.png"), img)
        return 1

    cube = AT.find_cube(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    strict = cv2.inRange(hsv, (35, 60, 40), (85, 255, 255))
    print(f"严格绿 {int(strict.sum()/255)}px   find_cube -> {cube}")

    vis = img.copy()
    if cube:
        cx, cy, w = cube
        cv2.drawMarker(vis, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 26, 2)
        cv2.putText(vis, f"cube {cx:.0f},{cy:.0f} w{w:.0f}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        dx, dy = int(cx) - 283, int(cy) - 393
        ok = abs(dx) <= 45 and abs(dy) <= 45
        cv2.putText(vis, f"offset {dx:+d},{dy:+d}  {'OK' if ok else 'MOVE'}", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if ok else (0, 0, 255), 2)
        cv2.rectangle(vis, (283 - 45, 393 - 45), (283 + 45, 393 + 45), (255, 255, 255), 1)
    else:
        loose = cv2.inRange(hsv, (20, 30, 20), (100, 255, 255))
        loose = cv2.morphologyEx(loose, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(loose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
        print(f"宽松绿 {int(loose.sum()/255)}px，前 {len(cnts)} 个色块：")
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            m = np.zeros(hsv.shape[:2], np.uint8)
            cv2.drawContours(m, [c], -1, 255, -1)
            hh, ss, vv = cv2.mean(hsv, mask=m)[:3]
            print(f"   面积{int(cv2.contourArea(c)):5d}px 中心({x+w//2},{y+h//2}) "
                  f"{w}x{h} HSV≈({hh:.0f},{ss:.0f},{vv:.0f})")
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 2)
        cv2.putText(vis, "NO CUBE (strict)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    raw_p = os.path.join(OUT_DIR, "cam_check_raw.png")
    vis_p = os.path.join(OUT_DIR, "cam_check.png")
    cv2.imwrite(raw_p, img)
    cv2.imwrite(vis_p, vis)
    print(f"已保存：{raw_p}\n        {vis_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
