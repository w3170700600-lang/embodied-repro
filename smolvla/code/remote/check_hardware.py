# -*- coding: utf-8 -*-
"""检查本机接了什么：串口（机械臂）、相机、以及 lerobot 的机器人配置是否可用。"""
import sys


def ports():
    print("== 串口设备 ==")
    try:
        from serial.tools import list_ports
    except ImportError:
        print("  pyserial 未安装")
        return []
    ps = list(list_ports.comports())
    if not ps:
        print("  没有检测到任何串口")
    for p in ps:
        print(f"  {p.device:8s} | {p.description} | {p.hwid}")
    return [p.device for p in ps]


def cameras():
    print("\n== 相机 ==")
    try:
        import cv2
    except ImportError:
        print("  cv2 未安装（lerobot 的 opencv 依赖没装上）")
        return
    print(f"  OpenCV {cv2.__version__}")
    found = []
    for i in range(4):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"  索引 {i}: {frame.shape[1]}x{frame.shape[0]} 可用")
            found.append(i)
        else:
            print(f"  索引 {i}: 能打开但读不到画面")
        cap.release()
    if not found:
        print("  0~3 都没有可用相机")


def arms():
    print("\n== lerobot 机器人类型 ==")
    try:
        from lerobot.robots import RobotConfig
        import lerobot.robots as R
        names = [n for n in dir(R) if n.endswith("Config")]
        print("  可用配置:", ", ".join(sorted(names)[:12]))
    except Exception as e:
        print(f"  {type(e).__name__}: {e}")


if __name__ == "__main__":
    ports()
    cameras()
    arms()
