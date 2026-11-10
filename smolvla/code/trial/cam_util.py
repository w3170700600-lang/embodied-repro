# -*- coding: utf-8 -*-
"""相机索引自动识别。

Windows 上 USB 摄像头的 index 是按枚举顺序给的，插拔 / 重启 / 多接一个设备都会整体错位。
之前固定写死 index=1，某次之后就变成了对着天花板的另一个摄像头 —— 症状是"方块明明摆好了
却检测不到"，实际是根本没看桌子。

这里改成用画面内容认：谁能看见绿方块 / 谁看的是桌面，就用谁。
识别结果写到 .camera_index，下次直接复用（避免每次开机都去开一遍所有相机）。
"""
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".camera_index")
CANDIDATES = (0, 1, 2, 3)
GREEN_LO = (35, 60, 40)
GREEN_HI = (85, 255, 255)
MIN_GREEN = 120          # 认作"看得见方块"的最小绿色像素数


def probe(idx, frames=10):
    """开一次相机读若干帧，返回统计。失败返回 None。"""
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    img = None
    for _ in range(frames):
        ok, fr = cap.read()
        if ok and fr is not None:
            img = fr
    cap.release()
    time.sleep(0.15)          # 给 DSHOW 一点释放时间，避免下一个 index 抢到黑帧
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = int(cv2.inRange(hsv, GREEN_LO, GREEN_HI).sum() / 255)
    return {"idx": idx, "mean": float(img.mean()), "green": green,
            "shape": img.shape[:2], "img": img}


def _cached():
    try:
        with open(CACHE, encoding="utf-8") as fh:
            return int(fh.read().strip())
    except Exception:
        return None


def _save(idx):
    try:
        with open(CACHE, "w", encoding="utf-8") as fh:
            fh.write(str(idx))
    except Exception:
        pass


def pick(preferred=None, verbose=True):
    """挑一个能看见方块（且不是全黑）的相机索引。返回 (idx, info)，info 可能为 None。"""
    tried = []
    if preferred is not None:
        info = probe(preferred)
        tried.append((preferred, info))
        if info is not None and info["mean"] > 20:
            _save(preferred)
            if verbose:
                print(f"相机 index={preferred}（手动指定） mean={info['mean']:.0f} "
                      f"绿{info['green']}px")
            return preferred, info
        if verbose:
            print(f"  手动指定的 index={preferred} 不可用，改为自动识别")

    cached = _cached()
    order = []
    if cached is not None:
        order.append(cached)
    order += [i for i in CANDIDATES if i not in order]

    best, best_info = None, None
    for idx in order:
        if idx == preferred and tried:
            info = tried[0][1]
        else:
            info = probe(idx)
        tried.append((idx, info))
        if info is None or info["mean"] < 20:
            continue
        if best_info is None or info["green"] > best_info["green"]:
            best, best_info = idx, info
        if info["green"] >= MIN_GREEN:
            break                 # 已经找到看得见方块的相机，不用再试

    if verbose:
        for idx, info in tried:
            if info is None:
                print(f"  相机 index={idx}: 打不开")
            else:
                print(f"  相机 index={idx}: {info['shape'][1]}x{info['shape'][0]} "
                      f"亮度{info['mean']:.0f} 绿色{info['green']}px")

    if best is None:
        print("!! 没有可用的相机（全黑或打不开）")
        return 0, None

    _save(best)
    if verbose:
        if best_info["green"] >= MIN_GREEN:
            print(f"-> 选用 index={best}（看得见绿方块，{best_info['green']}px）")
        else:
            print(f"-> 选用 index={best}（没有相机看见绿方块，先按亮度兜底；"
                  f"请确认方块在画面里）")
    return best, best_info


if __name__ == "__main__":
    idx, info = pick()
    if info is not None:
        out = os.path.join(HERE, "_frames", "trials", "cam_pick.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, info["img"])
        print("已保存", out)
