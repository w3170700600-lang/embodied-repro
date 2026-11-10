# -*- coding: utf-8 -*-
"""总线健康度体检：把"感觉电流不稳"变成数字。

用法（先停掉试验脚本，串口/相机不能被占用）：
    D:/smolvla-env/Scripts/python.exe bus_probe.py            # 默认 COM4、300 次
    D:/smolvla-env/Scripts/python.exe bus_probe.py --port COM4 --n 500

做三件事，全程不发送任何运动指令（只读）：
    1. 以 num_read_retries=0 读 300 次关节 —— 暴露链路的原始错误率
    2. 以 num_read_retries=2（当前配置）再读 300 次 —— 看重试后的实际可用性
    3. 相机连读 150 帧 —— 看帧率与掉帧

判读：
    p99 < 30ms、失败 0 次  → 链路健康，抖是策略/机械问题
    p99 忽高忽低、失败 > 2% → 接触不良或供电不足（换线/换口/换有源 hub）
"""
import argparse
import statistics
import time

import cv2
import numpy as np

from lerobot.robots import so_follower  # noqa: F401  注册 draccus 类型
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig


def read_pass(port, robot_id, n, retries, label):
    cfg = SOFollowerRobotConfig(port=port, id=robot_id, num_read_retries=retries,
                                max_relative_target=None, cameras={})
    robot = SOFollower(cfg)
    print(f"\n[{label}] 连接 {port} (num_read_retries={retries}) ...")
    t0 = time.time()
    try:
        robot.connect()
    except Exception as exc:
        print(f"  连接失败：{type(exc).__name__}: {exc}")
        print("  → 检查：12V 电源是否打开、USB 线是否插紧、COM 口是否被别的程序占用")
        return None
    print(f"  连接耗时 {(time.time()-t0)*1000:.0f}ms")

    lat, errs = [], {}
    for _ in range(n):
        t = time.perf_counter()
        try:
            robot.get_observation()
            lat.append((time.perf_counter() - t) * 1000)
        except Exception as exc:
            key = type(exc).__name__
            errs[key] = errs.get(key, 0) + 1
        time.sleep(0.005)

    try:
        robot.disconnect()
    except Exception:
        pass

    if not lat:
        print(f"  {n} 次全部失败：{errs}")
        return None
    lat.sort()
    p = lambda q: lat[min(len(lat) - 1, int(len(lat) * q))]  # noqa: E731
    fail = sum(errs.values())
    print(f"  成功 {len(lat)}/{n}  失败 {fail} ({fail/n*100:.1f}%)"
          + (f"  {errs}" if errs else ""))
    print(f"  单次读取 p50 {p(.5):.1f}ms  p95 {p(.95):.1f}ms  p99 {p(.99):.1f}ms  "
          f"max {lat[-1]:.1f}ms  抖动(std) {statistics.pstdev(lat):.1f}ms")
    return {"fail": fail, "n": n, "p99": p(.99), "max": lat[-1]}


def cam_pass(seconds=5.0):
    """相机连读，看帧率和掉帧。"""
    import cam_util
    idx, info = cam_util.pick(verbose=False)
    print(f"\n[相机] index={idx}，连读 {seconds:.0f}s ...")
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("  打不开（被占用？）")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    n, bad, means = 0, 0, []
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, fr = cap.read()
        n += 1
        if not ok or fr is None:
            bad += 1
        else:
            means.append(float(fr.mean()))
    dt = time.time() - t0
    cap.release()
    print(f"  读 {n} 次，失败 {bad}，实测 {n/dt:.1f} fps")
    if means:
        print(f"  亮度 mean {min(means):.0f}~{max(means):.0f}"
              f"（若出现 0 说明中间掉过图/黑帧）")


def wiggle_pass(port, robot_id, seconds, retries=0):
    """边抖线边读：滚动显示失败数与延迟，用来验证"是不是接触不良"。

    跑起来之后用手轻轻推/扭 USB 插头、扯一下线、拍一下 hub —— 只要失败数跳起来，
    就锁定是接触问题，而不是"电流"。
    """
    cfg = SOFollowerRobotConfig(port=port, id=robot_id, num_read_retries=retries,
                                max_relative_target=None, cameras={})
    robot = SOFollower(cfg)
    print(f"\n[抖线测试] 连接 {port}，持续 {seconds:.0f}s（现在开始推/扭 USB 插头）...")
    try:
        robot.connect()
    except Exception as exc:
        print(f"  连接失败：{type(exc).__name__}: {exc}")
        return
    n, fails, lat, errs = 0, 0, [], {}
    t0 = time.time()
    last_print = 0
    while time.time() - t0 < seconds:
        n += 1
        t = time.perf_counter()
        try:
            robot.get_observation()
            lat.append((time.perf_counter() - t) * 1000)
        except Exception as exc:
            fails += 1
            key = type(exc).__name__ + ": " + str(exc)[:40]
            errs[key] = errs.get(key, 0) + 1
        if n % 20 == 0 and time.time() - last_print > 0.4:
            last_print = time.time()
            p95 = np.percentile(lat[-200:], 95) if lat else 0
            print(f"  {time.time()-t0:5.1f}s  读 {n:5d}  失败 {fails:4d} "
                  f"({fails/n*100:5.1f}%)  近200次 p95 {p95:5.1f}ms", end="\r")
        time.sleep(0.005)
    try:
        robot.disconnect()
    except Exception:
        pass
    print()
    print(f"  合计 {n} 次，失败 {fails}（{fails/max(n,1)*100:.1f}%）")
    for k, v in errs.items():
        print(f"    {v:4d} × {k}")
    if fails:
        print("  → 确认是接触不良/线材问题：换线 + 直插机身口 + 应力释放")
    else:
        print("  → 这一段没抖出问题，故障点可能在别处（换一根线再试同一动作）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--id", default="my_follower")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--no-cam", action="store_true")
    ap.add_argument("--wiggle", type=float, default=0,
                    help="抖线测试时长（秒），给了就只跑这个")
    a = ap.parse_args()

    if a.wiggle > 0:
        wiggle_pass(a.port, a.id, a.wiggle)
        return

    r0 = read_pass(a.port, a.id, a.n, 0, "裸读（不带重试）")
    r2 = read_pass(a.port, a.id, a.n, 2, "当前配置（重试 2 次）")
    if not a.no_cam:
        cam_pass()

    print("\n==== 结论 ====")
    if r0 is None:
        print("总线连不上 —— 先解决硬件连接，别急着跑实验")
    elif r0["fail"] == 0 and r0["p99"] < 30:
        print("链路健康（裸读零失败、p99 < 30ms）→ 抖动不是 USB 的锅")
    else:
        rate = r0["fail"] / r0["n"] * 100
        print(f"链路有问题：裸读失败率 {rate:.1f}%、p99 {r0['p99']:.0f}ms、max {r0['max']:.0f}ms")
        print("  → 换一根短的带屏蔽 USB 数据线，直插机身口（不要经 hub / 延长线）")
        print("  → 线材做应力释放（扎带固定在桌边），别让插头受力")
        print("  → 若插头在座子里晃：是设备端座子焊点疲劳，需要补焊或换座")


if __name__ == "__main__":
    main()
