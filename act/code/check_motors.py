# -*- coding: utf-8 -*-
"""扫描 SO-101 舵机总线，检查 ID 是否与 LeRobot 期望一致。

用法:
    python check_motors.py COM5
    python check_motors.py COM5 --expect 1,2,3,4,5,6

它做三件事:
  1. 在所有支持波特率上广播 ping，列出真实存在的 ID
  2. 与期望 ID 集合对比，报出缺失 / 重复 / 多余
  3. 如果多个波特率都有响应，说明总线里混了不同波特率的电机（需要单独修）
"""
import argparse
import sys

EXPECTED_NAMES = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", help="串口号，例如 COM5")
    ap.add_argument("--expect", default="1,2,3,4,5,6", help="期望存在的 ID，逗号分隔")
    args = ap.parse_args()

    expected = sorted({int(x) for x in args.expect.split(",") if x.strip()})

    try:
        from lerobot.motors.feetech import FeetechMotorsBus
    except Exception as e:  # noqa: BLE001
        print("[FAIL] 无法导入 lerobot，请先激活 D:\\lerobot-venv: %s" % e)
        return 2

    print("扫描 %s ..." % args.port)
    try:
        result = FeetechMotorsBus.scan_port(args.port)
    except Exception as e:  # noqa: BLE001
        print("[FAIL] 打开串口失败: %s" % e)
        print("       常见原因: 串口号写错 / 被别的程序占用 / 没装 CH340 驱动")
        return 2

    if not result:
        print("[FAIL] 所有波特率都没 ping 到任何电机")
        print("       检查: 电机是否上电 / 总线线序(A-B)是否接反 / 拓展坞是否被识别")
        return 1

    print("\n--- 扫描结果 ---")
    for baudrate in sorted(result, reverse=True):
        ids = sorted(result[baudrate])
        print("  %8d bps : IDs %s" % (baudrate, ids))
        for i in ids:
            print("              %-2d -> %s" % (i, EXPECTED_NAMES.get(i, "未知/多余")))

    if len(result) > 1:
        print("\n[WARN] 有多个波特率都响应了，说明总线里混了不同波特率的电机。")
        print("       setup-motors 只会在找到的第一个波特率上操作，这种情况必须逐个单独接上电机重设。")

    found = sorted({i for ids in result.values() for i in ids})

    missing = [i for i in expected if i not in found]
    extra = [i for i in found if i not in expected]
    # 同一 ID 出现两次在扫描里看不出来，但「ID 数 != 期望数」是强信号
    ok = not missing and not extra and len(found) == len(expected)

    print("\n--- 判定 ---")
    print("  期望: %s" % expected)
    print("  实际: %s" % found)
    if missing:
        print("  [X] 缺失 ID: %s -> 对应关节 %s 还没写 ID 或没接上"
              % (missing, [EXPECTED_NAMES.get(i) for i in missing]))
    if extra:
        print("  [X] 多余 ID: %s -> 有电机 ID 不在 1-6 范围内，或写错了" % extra)
    if len(found) != len(expected):
        print("  [X] 数量不对: 期望 %d 个，实际 %d 个。少=没接/没写ID；多=有ID重复的电机在同一总线上"
              % (len(expected), len(found)))

    if ok:
        print("  [OK] 6 个电机 ID 齐全且无多余，可以进入 lerobot-calibrate")
        return 0
    print("\n  修法: 把出问题的那一个电机单独接到控制板，重跑 lerobot-setup-motors 对应的那一轮。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
