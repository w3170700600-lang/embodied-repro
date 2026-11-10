# -*- coding: utf-8 -*-
"""相机机位自检：与训练数据首帧做几何对齐，直接告诉你偏差多大。

用法（每次挪完相机跑一次）：
    D:/smolvla-env/Scripts/python.exe verify_cam.py

判定标准（训练机位对齐时的实测值）：四角偏差 < 15px、缩放 0.97~1.03。
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "_frames", "f2s.png")  # 训练数据画面
SHOT = os.path.join(HERE, "_frames", "cam", "verify_latest.png")

orb = cv2.ORB_create(3000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def grab():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    frame = None
    for _ in range(20):
        ok, frame = cap.read()
    cap.release()
    return frame


def main():
    ref = cv2.imread(REF)
    cur = grab()
    if cur is None:
        print("相机打不开")
        return
    cv2.imwrite(SHOT, cur)
    g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    m = sorted(bf.match(d1, d2), key=lambda x: x.distance)[:300]
    pa = np.float32([k1[x.queryIdx].pt for x in m])
    pb = np.float32([k2[x.trainIdx].pt for x in m])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
    inl = int(mask.sum())
    scale = np.sqrt(abs(np.linalg.det(H[:2, :2])))
    corners = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    err = float(np.abs(proj - corners.reshape(-1, 2)).max())

    print(f"匹配 {len(m)} | 内点 {inl} | 缩放 {scale:.3f} | 四角最大偏差 {err:.0f}px")
    print("(注: 几何判定在纯色桌面上不可靠，仅作参考)")
    print("已保存当前画面到 _frames/cam/verify_latest.png")
    print("方块位置自检: 跑一次 auto_trial 前先喊我看图，或观察画面中方块是否在白纸中央偏右")


if __name__ == "__main__":
    main()
