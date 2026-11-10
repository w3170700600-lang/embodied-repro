# -*- coding: utf-8 -*-
"""只读测速：机械臂总线读一次状态要多久（不发送任何动作）。"""
import statistics
import time

from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower

cfg = SO101FollowerConfig(port="COM4", id="my_follower", disable_torque_on_disconnect=True)
robot = SO101Follower(cfg)
print("连接中...")
robot.connect()
print("已连接")

obs = robot.get_observation()
print("一次完整状态（6 关节 + 6 电机）:")
for k, v in obs.items():
    if k.endswith(".pos"):
        print(f"  {k:24s} {v:+.1f}°")

ts = []
for _ in range(20):
    t = time.perf_counter()
    robot.get_observation()
    ts.append((time.perf_counter() - t) * 1000)
ts.sort()
print(f"\nget_observation 耗时: 中位 {statistics.median(ts):.1f} ms | 最小 {ts[0]:.1f} | 最大 {ts[-1]:.1f}")
print(f"  若按 30Hz 跑，光是读状态就占 {statistics.median(ts)/33.3*100:.0f}% 的时间预算")

robot.disconnect()
print("已断开")
