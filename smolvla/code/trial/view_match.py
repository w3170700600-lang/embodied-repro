# -*- coding: utf-8 -*-
"""相机视角漂移检测：把"现在"和"训练时"对齐，量化相机被挪了多少。

用法（先停掉试验脚本）：
    D:/smolvla-env/Scripts/python.exe view_match.py
    D:/smolvla-env/Scripts/python.exe view_match.py --ref _frames/ref_dataset.png --now _frames/trials/cam_check_raw.png

原理：用背景（桌子边缘、插排、台灯这些不动的物体）做特征匹配 + RANSAC 求单应，
再分解出"平移 / 旋转 / 缩放"。机械臂和椅子是不同的，会被 RANSAC 当外点丢掉。

判读：
    |dx| < 8px 且 |dy| < 8px 且 |旋转| < 1.5° 且 |缩放| < 3%  → 相机没动，可以跑
    超过这个范围 → 视角已经不是训练时的视角，策略会失配，必须把相机调回去
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def match(ref, now):
    g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(now, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=4000, fastThreshold=10)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(d1, d2, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return None
    inl = int(mask.sum())
    # 分解：把训练图的四条角映射过去，看中心位移/旋转/缩放
    h, w = ref.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    c_ref = np.array([w / 2, h / 2], np.float32).reshape(-1, 1, 2)
    c_now = cv2.perspectiveTransform(c_ref, H).reshape(-1, 2)[0]
    dx, dy = c_now[0] - w / 2, c_now[1] - h / 2
    v = proj[1] - proj[0]
    ang = np.degrees(np.arctan2(v[1], v[0]))
    scale = np.linalg.norm(v) / w
    return dict(H=H, inl=inl, total=len(good), dx=dx, dy=dy, ang=ang, scale=scale,
                proj=proj, k1=k1, k2=k2, good=good, mask=mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.join(HERE, "_frames", "ref_dataset.png"))
    ap.add_argument("--now", default=os.path.join(HERE, "_frames", "trials", "cam_check_raw.png"))
    ap.add_argument("--out", default=os.path.join(HERE, "_frames", "view_match.png"))
    a = ap.parse_args()

    ref = cv2.imread(a.ref)
    now = cv2.imread(a.now)
    if ref is None or now is None:
        print(f"读不到图：ref={a.ref} now={a.now}")
        return 1
    if not os.path.exists(a.ref):
        print(f"缺参考帧 {a.ref}")

    r = match(ref, now)
    if r is None:
        print("!! 特征太少，对不上 —— 可能画面差异过大（相机转了很多）或画面太糊/太暗")
        return 1

    print(f"匹配点 {r['total']}，RANSAC 内点 {r['inl']}（{r['inl']/r['total']*100:.0f}%）")
    print(f"视野中心位移 dx {r['dx']:+.1f}px  dy {r['dy']:+.1f}px")
    print(f"旋转 {r['ang']:+.2f}°   缩放 {r['scale']*100-100:+.1f}%")

    bad = (abs(r["dx"]) > 8 or abs(r["dy"]) > 8 or abs(r["ang"]) > 1.5
           or abs(r["scale"] - 1) > 0.03 or r["inl"] < 30)
    if bad:
        print("\n>> 视角已经偏离训练时的取景，策略会失配：")
        if abs(r["dx"]) > 8 or abs(r["dy"]) > 8:
            print(f"   画面整体平移了 ({r['dx']:+.0f},{r['dy']:+.0f})px → 相机左右/俯仰需回正")
        if abs(r["ang"]) > 1.5:
            print(f"   画面转了 {r['ang']:+.1f}° → 相机绕光轴歪了，需要转回来")
        if abs(r["scale"] - 1) > 0.03:
            print(f"   远近差 {r['scale']*100-100:+.0f}% → 相机前后位置变了")
    else:
        print("\n>> 与训练取景一致，可以跑")

    # 可视化：上图=训练帧叠加当前帧的投影框；下图=当前帧叠加匹配点
    vis_ref = ref.copy()
    cv2.polylines(vis_ref, [np.int32(r["proj"])], True, (0, 255, 0), 2)
    vis_now = now.copy()
    ok = [(m, k) for m, k in zip(r["good"], r["mask"].ravel()) if k]
    for m, _ in ok[:60]:
        p1 = tuple(np.int32(r["k1"][m.queryIdx].pt))
        p2 = tuple(np.int32(r["k2"][m.trainIdx].pt))
        cv2.circle(vis_now, p2, 3, (0, 255, 0), -1)
        cv2.circle(vis_ref, p1, 3, (0, 255, 0), -1)
    cv2.putText(vis_ref, "TRAINING + projected frame of NOW", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(vis_now, f"NOW  d=({r['dx']:+.0f},{r['dy']:+.0f}) rot{r['ang']:+.1f}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if bad else (0, 255, 0), 2)
    cv2.imwrite(a.out, np.vstack([vis_ref, vis_now]))
    print(f"对比图已存 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
