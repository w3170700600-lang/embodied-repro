"""
rollout_simple.py -- 轻量 ACT 控制循环，绕开 lerobot-rollout 的框架开销。

为什么要自己写：
  lerobot-rollout（0.6.1）每步要过 telemetry + interpolator + 数据集帧构造，
  在本机 CPU 上只跑到 7 Hz（目标 30 Hz），动作被拉慢 4 倍。
  这里只保留必要流程，实测可达 60 Hz 以上。

用法：
  python rollout_simple.py --dry-run        # 不连硬件，用数据集验证管道是否正确
  python rollout_simple.py                  # 实机运行 20 秒
  python rollout_simple.py --duration 30
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

# 注意：这里绝对不能设置 HF_HOME 或 HF_LEROBOT_HOME。
# 两者都会改变 lerobot 查找校准文件的根目录，导致 robot.connect() 认为
# 没有校准过，弹出交互式校准流程并可能覆盖原校准文件。

CALIB_FILE = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "lerobot",
    "calibration", "robots", "so_follower", "my_follower.json",
)

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.feature_utils import build_dataset_frame

CKPT = "D:/lerobot-work/models/100000/pretrained_model"
ROOT = "D:/lerobot-data/so101_test"
TASK = "Grab the green cube"
PORT = "COM4"
ROBOT_ID = "my_follower"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DEVICE = torch.device("cpu")


class _Tee:
    """同时输出到终端和日志文件。"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for st in self.streams:
            try:
                st.write(data)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def load_everything():
    info = json.load(open(os.path.join(ROOT, "meta", "info.json"), encoding="utf-8"))
    features = info["features"]
    ds = LeRobotDataset("local/so101_test", root=ROOT)

    from eval_offline import localize_ckpt
    ck = localize_ckpt(CKPT, "cpu")
    policy = ACTPolicy.from_pretrained(ck)
    policy.to(DEVICE)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=ck)
    return features, ds, policy, pre, post


def dry_run(features, ds, policy, pre, post):
    """用数据集里的一帧构造 obs，检查输出动作是否接近人类示范。"""
    print("[dry-run] 用数据集验证管道，不连硬件")
    gi = 300
    sample = ds[gi]
    gt = sample["action"].numpy()

    # 构造与 robot.get_observation() 相同格式的 obs
    img = (sample["observation.images.front"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    obs = {"front": img}
    for i, j in enumerate(JOINTS):
        obs[f"{j}.pos"] = float(sample["observation.state"][i])

    frame = build_dataset_frame(features, obs, prefix="observation")
    print("  帧 keys:", list(frame.keys()))
    for k, v in frame.items():
        print(f"    {k}: shape={v.shape} dtype={v.dtype}")

    o = prepare_observation_for_inference(frame, DEVICE, TASK, "so_follower")
    with torch.inference_mode():
        o = pre(o)
        a = post(policy.select_action(o))
    ad = make_robot_action(a, features)
    print()
    print("  %-14s %10s %10s" % ("关节", "模型输出", "人类示范"))
    for j in JOINTS:
        key = f"{j}.pos"
        print("  %-14s %10.2f %10.2f" % (j, ad[key], float(gt[JOINTS.index(j)])))
    err = np.mean([abs(ad[f"{j}.pos"] - float(gt[JOINTS.index(j)])) for j in JOINTS])
    print()
    print("  平均绝对误差: %.2f 度" % err)
    print("  判定:", "管道正确" if err < 15 else "★异常，检查 key 映射")


def compute_limits(ds):
    """用训练数据的动作范围做安全边界，防止模型长 chunk 外推把机械臂压到桌面。"""
    import glob
    import pandas as pd
    files = sorted(glob.glob(os.path.join(ROOT, "data", "chunk-*", "file-*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    acts = np.stack(d["action"].values)
    lim = {}
    for i, j in enumerate(JOINTS):
        col = acts[:, i]
        lim[j] = (float(np.percentile(col, 0.5)), float(np.percentile(col, 99.5)))
    return lim


def training_start_pose():
    """全部示范首帧动作的均值，作为「复位目标姿态」。"""
    import glob
    import pandas as pd
    files = sorted(glob.glob(os.path.join(ROOT, "data", "chunk-*", "file-*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    starts = []
    for _, g in d.groupby("episode_index"):
        starts.append(np.stack(g["action"].values)[0])
    return np.mean(np.stack(starts), axis=0)


def reset_to_start(robot):
    """张开夹爪并缓慢移动到训练起始姿态，然后保持力矩等待摆件。"""
    target = training_start_pose()
    obs = robot.get_observation()
    cur = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=float)
    print("  当前姿态:", " ".join("%s=%.0f" % (j, cur[i]) for i, j in enumerate(JOINTS)))
    print("  目标姿态:", " ".join("%s=%.0f" % (j, target[i]) for i, j in enumerate(JOINTS)))

    # 第一步：张开夹爪，让方块落下
    ad = {f"{j}.pos": float(cur[i]) for i, j in enumerate(JOINTS)}
    ad["gripper.pos"] = 40.0
    for _ in range(20):
        robot.send_action(ad)
        time.sleep(1 / 30)
    print("  夹爪已张开")

    # 第二步：从当前姿态线性插值到起始姿态（6 秒，放慢以降低电流峰值）
    # 夹爪也回到训练首帧的值（约 1.4，闭合）—— 56 条示范的首帧 gripper 都是闭合的，
    # 张开到 40 属于训练分布外，模型会因此输出「合爪」的 38 度突变。
    goal = target.copy()
    for k in range(180):
        a = (k + 1) / 180
        pos = cur * (1 - a) + goal * a
        ad = {f"{j}.pos": float(pos[i]) for i, j in enumerate(JOINTS)}
        robot.send_action(ad)
        time.sleep(1 / 30)
    time.sleep(0.4)
    obs2 = robot.get_observation()
    act = np.array([obs2[f"{j}.pos"] for j in JOINTS], dtype=float)
    err = np.abs(act - goal)
    print("  已移动到起始姿态（保持力矩）")
    print("  复位后实测:", " ".join("%s=%.0f" % (j, act[i]) for i, j in enumerate(JOINTS)))
    print("  复位误差: 最大 %.1f 度（%s）" % (err.max(), JOINTS[int(err.argmax())]))


def live(features, policy, pre, post, duration, n_action_steps, clamp, do_reset, save_frames):
    if not os.path.isfile(CALIB_FILE):
        raise SystemExit(
            f"[中止] 找不到校准文件：{CALIB_FILE}\n"
            f"       先跑一次 lerobot-calibrate，避免自动进入校准流程。"
        )
    print(f"校准文件: {CALIB_FILE}")

    # 队列长度决定「执行多少步后重新看一次画面」。
    # chunk_size=100 不变，但只执行前 n 步就重新推理，换取更频繁的视觉修正。
    policy.config.n_action_steps = n_action_steps
    policy.reset()

    limits = compute_limits(None) if clamp else None
    if limits:
        print("动作限幅（训练数据 0.5%~99.5% 分位）:")
        for j, (lo, hi) in limits.items():
            print("  %-14s %8.1f ~ %8.1f" % (j, lo, hi))
    print(f"n_action_steps = {n_action_steps}（每 {n_action_steps / 30:.2f} 秒重新看一次画面）")

    cfg = SO101FollowerConfig(
        port=PORT, id=ROBOT_ID,
        cameras={"front": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30)},
    )
    robot = SO101Follower(cfg)
    robot.connect()
    print("已连接")

    if do_reset:
        print("复位中 ...")
        reset_to_start(robot)
        print()
        print("  请现在把绿色方块放到平板中央（画面里红十字位置），")
        print("  不要碰机械臂。摆好后按回车开始。")
        input("  >>> 按回车开始 rollout ... ")
    else:
        print("跳过复位，直接开始")

    print()
    print("开始执行（Ctrl+C 停止）")
    print()

    t_start = time.time()
    step = 0
    t_obs, t_inf, t_act = [], [], []
    actual_hz = []
    prev = None
    traj = []          # 记录每步的 action 与 state，事后分析动作阶段
    try:
        while time.time() - t_start < duration:
            loop = time.perf_counter()
            obs = robot.get_observation()
            t1 = time.perf_counter()
            frame = build_dataset_frame(features, obs, prefix="observation")
            o = prepare_observation_for_inference(frame, DEVICE, TASK, "so_follower")
            with torch.inference_mode():
                o = pre(o)
                a = post(policy.select_action(o))
            t2 = time.perf_counter()
            ad = make_robot_action(a, features)
            if limits:
                for j, (lo, hi) in limits.items():
                    k = f"{j}.pos"
                    ad[k] = float(min(max(ad[k], lo), hi))
            robot.send_action(ad)
            t3 = time.perf_counter()

            if save_frames and step % 15 == 0:
                os.makedirs("frames", exist_ok=True)
                cv2.imwrite(os.path.join("frames", "f_%04d.jpg" % step), obs["front"])

            traj.append(
                [step, (time.time() - t_start)]
                + [ad[f"{j}.pos"] for j in JOINTS]
                + [obs.get(f"{j}.pos", float("nan")) for j in JOINTS]
            )

            t_obs.append(t1 - loop)
            t_inf.append(t2 - t1)
            t_act.append(t3 - t2)
            step += 1

            if step % 20 == 0:
                hz = 20 / max(time.perf_counter() - loop + (t3 - loop), 1e-6) if False else None
                state = obs.get("shoulder_pan.pos")
                delta = "" if prev is None else " Δ=%.2f" % (ad["shoulder_pan.pos"] - prev)
                print("  step %3d | shoulder_pan→%7.2f%s | 观测%4.1fms 推理%4.1fms 发送%4.1fms"
                      % (step, ad["shoulder_pan.pos"], delta,
                         np.mean(t_obs[-20:]) * 1000, np.mean(t_inf[-20:]) * 1000, np.mean(t_act[-20:]) * 1000))
                prev = ad["shoulder_pan.pos"]

            dt = time.perf_counter() - loop
            target = 1 / 30
            if dt < target:
                time.sleep(target - dt)
            actual_hz.append(1 / max(time.perf_counter() - loop, 1e-6))
    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        robot.disconnect()
        print("已断开")

    n = len(actual_hz)
    if n:
        print()
        print("  执行 %d 步，用时 %.1f 秒，平均 %.1f Hz" % (n, time.time() - t_start, np.mean(actual_hz)))
        print("  各环节均值：观测 %.1f ms / 推理 %.1f ms / 发送 %.1f ms"
              % (np.mean(t_obs) * 1000, np.mean(t_inf) * 1000, np.mean(t_act) * 1000))

    if traj:
        import csv
        out = "rollout_traj_%s.csv" % time.strftime("%H%M%S")
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["step", "t"]
                       + [f"act_{j}" for j in JOINTS]
                       + [f"state_{j}" for j in JOINTS])
            w.writerows(traj)
        print("  轨迹已存: %s（%d 行）" % (out, len(traj)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--no-clamp", action="store_true", help="关闭动作限幅")
    ap.add_argument("--no-reset", action="store_true", help="跳过复位，直接从当前姿态开始")
    ap.add_argument("--save-frames", action="store_true", help="每 15 步存一帧画面，用于事后分析")
    ap.add_argument("--n-action-steps", type=int, default=30,
                    help="执行多少步后重新推理一次（越小对齐越准，但推理更频繁）")
    a = ap.parse_args()

    import sys as _sys
    logf = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "rollout_%s.log" % time.strftime("%H%M%S")), "w", encoding="utf-8")
    _sys.stdout = _Tee(_sys.__stdout__, logf)
    print("日志文件:", logf.name)

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    print("加载模型与数据集 ...")
    features, ds, policy, pre, post = load_everything()
    print("  完成")
    print()

    if a.dry_run:
        dry_run(features, ds, policy, pre, post)
    else:
        live(features, policy, pre, post, a.duration, a.n_action_steps, not a.no_clamp, not a.no_reset, a.save_frames)


if __name__ == "__main__":
    main()
