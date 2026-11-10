# -*- coding: utf-8 -*-
"""把从动臂精确驱动到训练起始姿态。

用法：
    D:/smolvla-env/Scripts/python.exe pose_reset.py

训练起始姿态（56 条 episode 的均值）：pan 5 / lift -101 / elbow 90 /
wrist_flex 73 / wrist_roll -16 / gripper 3（夹爪基本闭合）。
自动试验脚本会在每轮开始前调用本模块，避免"回位"带来的几度漂移——
实测几度偏差会让策略进入"原地冻结"模式而非伸手抓取。
"""
import time

TARGET = {
    "shoulder_pan": 5.0,
    "shoulder_lift": -101.0,
    "elbow_flex": 90.0,
    "wrist_flex": 73.0,
    "wrist_roll": -16.0,
    "gripper": 3.0,
}


def reset(port="COM4", robot_id="my_follower", verbose=True):
    """驱动到起始姿态，返回执行后的实际关节角（度）。"""
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SO101Follower

    cfg = SO101FollowerConfig(
        port=port, id=robot_id,
        disable_torque_on_disconnect=True,
        max_relative_target=30.0,
    )
    robot = SO101Follower(cfg)
    robot.connect()
    action = {f"{k}.pos": v for k, v in TARGET.items()}
    for _ in range(60):
        robot.send_action(dict(action))
        time.sleep(0.05)
    obs = robot.get_observation()
    robot.disconnect()
    result = {k.split(".")[0]: round(v, 1) for k, v in obs.items() if k.endswith(".pos")}
    if verbose:
        print(f"  姿态校准完成: {result}")
    return result


if __name__ == "__main__":
    reset()
