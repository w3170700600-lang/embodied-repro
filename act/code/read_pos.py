# -*- coding: utf-8 -*-
"""读取指定端口上 6 个舵机的原始位置，用于判断主臂/从臂。
用法: python read_pos.py COM3
"""
import sys

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    if len(sys.argv) < 2:
        print("用法: python read_pos.py COM3")
        return 2
    port = sys.argv[1]
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    motors = {n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(NAMES)}
    bus = FeetechMotorsBus(port=port, motors=motors)
    try:
        bus._connect(handshake=False)
        bus.set_baudrate(1_000_000)
        pos = bus.sync_read("Present_Position", normalize=False)
        print("%s: %s" % (port, "  ".join("%s=%d" % (n, pos[n]) for n in NAMES)))
    except Exception as e:
        print("%s 读取失败: %s: %s" % (port, type(e).__name__, e))
        return 1
    finally:
        try:
            bus.port_handler.closePort()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
