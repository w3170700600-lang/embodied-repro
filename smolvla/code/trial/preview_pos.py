# -*- coding: utf-8 -*-
"""实时取景调位窗口：把方块对进白框。

用法（先关掉试验脚本，否则相机被占用）：
    D:/smolvla-env/Scripts/python.exe preview_pos.py

显示内容：
    白框      = 目标位置（经验校正位 283,393 —— 实测夹爪真实落点）
    十字标    = 方块当前中心（实时跟随）
    X / Y     = 数值与 OK 标记，两项都 OK 时十字变绿、出现 READY
按 ESC 退出。
"""
import cv2
import numpy as np

import cam_util

TARGET_X, TARGET_Y = 283, 393   # 经验校正位：实测夹爪落点反推（左约3cm、靠近相机约2cm）
TOL = 45                        # 与 persistent_trial.GATE_TOL 保持一致


def find_cube(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 60, 40), (85, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 80:
        return None
    x, y, w, h = cv2.boundingRect(c)
    return x + w / 2, y + h / 2, w


def loose_blobs(frame, k=3):
    """严格阈值找不到时，用放宽阈值列出可疑色块，帮助判断是"没摆对"还是"颜色不认"。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 30, 20), (100, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:k]
    return [c for c in cnts if cv2.contourArea(c) > 80]


def main():
    cam_index, _ = cam_util.pick()
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("相机打不开（试验脚本可能还占着相机，先关掉它）")
        return
    print(f"用的是 index={cam_index}  目标位置 ({TARGET_X}, {TARGET_Y})  容差 ±{TOL}px")
    print("把方块对进白框，两项都 OK 后按 ESC 退出")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        c = find_cube(frame)

        cv2.rectangle(frame, (TARGET_X - TOL, TARGET_Y - TOL),
                      (TARGET_X + TOL, TARGET_Y + TOL), (255, 255, 255), 2)
        cv2.drawMarker(frame, (TARGET_X, TARGET_Y), (200, 200, 200),
                       cv2.MARKER_TILTED_CROSS, 14, 1)

        if c:
            cx, cy, w = c
            dx, dy = TARGET_X - cx, TARGET_Y - cy
            inx, iny = abs(dx) <= TOL, abs(dy) <= TOL
            color = (0, 255, 0) if (inx and iny) else (0, 0, 255)
            cv2.drawMarker(frame, (int(cx), int(cy)), color, cv2.MARKER_CROSS, 26, 2)
            cv2.putText(frame, f"X {cx:.0f}  {'OK' if inx else f'move {dx:+.0f}px'}",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Y {cy:.0f}  {'OK' if iny else f'move {dy:+.0f}px'}",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"W {w:.0f}px", (10, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            if inx and iny:
                cv2.putText(frame, "READY - press ESC", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO CUBE (strict hsv)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            for c in loose_blobs(frame):
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 1)
                cv2.putText(frame, f"?{int(cv2.contourArea(c))}", (x, max(y - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        cv2.imshow("place cube", frame)
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
