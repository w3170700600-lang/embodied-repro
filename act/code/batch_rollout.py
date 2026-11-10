"""
batch_rollout.py -- 连续跑 N 次抓取，自动判定成功/失败，统计成功率。

判定方式：运行过程中每 30 步检测一次绿色方块在画面中的 y 坐标。
方块被夹起来会让 y 明显变小（画面中上移）。取全程最小的 y 与起始 y 比较，
上移超过 40 像素即判为成功。

用法（由 run_batch.bat 启动，或直接）：
  python batch_rollout.py --n 10 --duration 11
"""
import argparse
import csv
import datetime
import math
import os
import sys
import time

import cv2
import numpy as np
import torch

import rollout_simple as R
from detect_cube import find_green

LIFT_THRESHOLD_PX = 40


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


VIDEO_DIR = "videos_rollout"
GRASP_GAP_THRESHOLD = 6.0   # 夹爪闭合后停在该角度以上 = 指缝里有物体


def run_episode(robot, features, policy, pre, post, duration, run_id=0, fps=30.0, video_path=None, n_action_steps=100):
    """跑一轮，返回 (起始 y, 最小 y, 采样数)。"""
    t0 = time.time()
    step = 0
    base_y = None
    min_y = 10 ** 9
    n_probe = 0
    max_delta_normal = 15.0   # 统一 15 度/步（30fps 下即 450 度/秒）
    max_delta_first = 15.0    # 不再对起始步特殊收紧
    clipped = 0
    # 关键：用当前实际位置初始化参考值，否则第一帧不受限速约束，
    # 模型输出与当前位姿差多少就会全速甩过去（导致电流爆表、断线）
    # ★ 关键修复：清空动作队列。
    # ACTPolicy.select_action 内部有 maxlen=n_action_steps 的 deque，
    # 队列非空时不会重新推理，只会一直 popleft。
    # 不清空的话，新一轮开头执行的是【上一轮末尾剩余的约 40 个动作】，
    # 表现为机械臂一启动就朝桌面砸下去，然后才重新看画面。
    policy.config.n_action_steps = n_action_steps
    policy.reset()
    print("    动作队列已清空（n_action_steps=%d，约每 %.1f 秒重新看一次画面）"
          % (n_action_steps, n_action_steps / 30.0))

    obs_init = robot.get_observation()
    prev_ad = {f"{j}.pos": float(obs_init[f"{j}.pos"]) for j in R.JOINTS}
    print("    起始位姿已读取，首帧将受限速约束")

    vw = None
    if video_path:
        w0, h0 = obs_init["front"].shape[1], obs_init["front"].shape[0]
        vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w0, h0))
        print("    录像中 -> %s" % video_path)

    clip_joint = {j: 0 for j in R.JOINTS}
    max_jump = {j: 0.0 for j in R.JOINTS}
    frames = []
    traj = []

    while time.time() - t0 < duration:
        loop = time.perf_counter()
        obs = robot.get_observation()
        frame = R.build_dataset_frame(features, obs, prefix="observation")
        o = R.prepare_observation_for_inference(frame, R.DEVICE, R.TASK, "so_follower")
        with torch.inference_mode():
            o = pre(o)
            a = post(policy.select_action(o))
        ad = R.make_robot_action(a, features)
        raw_ad = dict(ad)          # 限速前的模型原始输出，用于诊断
        max_delta = max_delta_first if step < 10 else max_delta_normal
        if prev_ad is not None:
            for k in list(ad.keys()):
                d = ad[k] - prev_ad[k]
                jn = k.replace(".pos", "")
                max_jump[jn] = max(max_jump[jn], abs(d))
                if abs(d) > max_delta:
                    if clipped == 0:
                        print("    ★ 检测到 %.1f 度的突变（%s），已限速到 %.1f 度/步" % (abs(d), jn, max_delta))
                    ad[k] = prev_ad[k] + max_delta * (1 if d > 0 else -1)
                    clip_joint[jn] += 1
                    clipped += 1
        prev_ad = dict(ad)
        row = [round(time.time() - t0, 3), step]
        for j in R.JOINTS:
            row.append(round(float(obs[f"{j}.pos"]), 2))      # 执行前实际状态
        for j in R.JOINTS:
            row.append(round(float(raw_ad[f"{j}.pos"]), 2))   # 模型原始输出
        for j in R.JOINTS:
            row.append(round(float(ad[f"{j}.pos"]), 2))       # 限速后实际发送
        traj.append(row)
        robot.send_action(ad)

        if vw is not None:
            # 注意：LeRobot 相机返回的是 RGB，cv2.VideoWriter 按 BGR 编码，
            # 直接写入会导致红蓝对调（蓝色机械臂变橙色）。
            vw.write(cv2.cvtColor(obs["front"], cv2.COLOR_RGB2BGR))
        if step % 5 == 0:
            frames.append((step, cv2.cvtColor(obs["front"], cv2.COLOR_RGB2BGR)))

        if step % 30 == 0:
            g = find_green(obs["front"])
            if g:
                n_probe += 1
                if base_y is None:
                    base_y = g[1]
                min_y = min(min_y, g[1])
        step += 1
        dt = time.perf_counter() - loop
        if dt < 1 / fps:
            time.sleep(1 / fps - dt)
    if clipped:
        worst = max(clip_joint, key=clip_joint.get)
        print("    （限速生效 %d 次，最多的是 %s：%d 次）" % (clipped, worst, clip_joint[worst]))
        big = [(j, round(v, 1)) for j, v in max_jump.items() if v > 3]
        if big:
            print("    各关节单帧最大变化: %s" % ", ".join("%s=%.1f" % (j, v) for j, v in big))
    if vw is not None:
        vw.release()
        print("    录像已保存")
    if frames:
        os.makedirs("frames_batch", exist_ok=True)
        for st, fr in frames:
            cv2.imwrite(os.path.join("frames_batch", "r%02d_s%04d.jpg" % (run_id, st)), fr)
    # 判据二（物理）：夹爪第一次张开后，再闭合时能合到多少度。
    # 合不底（>6 度）说明指缝之间有东西 -> 真的夹住了；
    # 一路合到 ~1.5 度说明指缝是空的 -> 夹空。
    grasp_gap = None
    if traj:
        gi = 2 + R.JOINTS.index("gripper")
        ser = np.array([row[gi] for row in traj], dtype=float)
        if len(ser) > 30:
            oi = int(np.argmax(ser))
            if oi < len(ser) - 5:
                grasp_gap = float(ser[oi:].min())

    if traj:
        os.makedirs("traj_batch", exist_ok=True)
        hdr = ["t", "step"] + ["st_" + j for j in R.JOINTS] + ["raw_" + j for j in R.JOINTS] + ["cmd_" + j for j in R.JOINTS]
        with open(os.path.join("traj_batch", "r%02d.csv" % run_id), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            w.writerows(traj)
    return base_y, min_y, n_probe, grasp_gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--duration", type=float, default=11.0)
    ap.add_argument("--manual", action="store_true", help="改为按回车开始（默认是10秒倒计时）")
    ap.add_argument("--fps", type=float, default=30.0, help="控制频率，降低可缓解通信中断")
    ap.add_argument("--video", action="store_true", help="每轮录制 mp4 视频")
    ap.add_argument("--n-action-steps", type=int, default=100,
                    help="执行多少步后重新推理一次（100=3.3s，30=1s，越小视觉修正越频繁）")
    a = ap.parse_args()

    logf = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "batch_%s.log" % time.strftime("%H%M%S")), "w", encoding="utf-8")
    sys.stdout = R._Tee(sys.__stdout__, logf)
    sys.stderr = sys.stdout   # 异常也写进日志
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    print("加载模型 ...")
    features, ds, policy, pre, post = R.load_everything()
    print("模型就绪")

    def make_robot():
        return R.SO101Follower(R.SO101FollowerConfig(
            port=R.PORT, id=R.ROBOT_ID,
            cameras={"front": R.OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30)},
        ))

    def connect_retry():
        for attempt in range(1, 4):
            try:
                rb = make_robot()
                rb.connect()
                if attempt > 1:
                    print("  机械臂已连接（第 %d 次尝试）" % attempt)
                return rb
            except Exception as e:
                print("  连接失败 %d/3: %s" % (attempt, str(e)[:70]))
                try:
                    rb.disconnect()
                except Exception:
                    pass
                time.sleep(2.0)
        return None

    manual = a.manual
    if a.video:
        os.makedirs(VIDEO_DIR, exist_ok=True)
    results = []
    rows = []
    t_start = time.time()
    try:
        for i in range(1, a.n + 1):
            print("=" * 58)
            print("第 %d / %d 次" % (i, a.n))
            robot = connect_retry()
            if robot is None:
                print("  ★ 连接失败，停止")
                break
            try:
                R.reset_to_start(robot)
                if manual:
                    input("  把方块放到平板中央，按回车开始 ... ")
                else:
                    for sec in range(6, 0, -1):
                        print("  请把方块放到平板中央 ... %2d 秒后开始" % sec, end="\r", flush=True)
                        time.sleep(1)
                    print(" " * 46, end="\r")
            except (EOFError, KeyboardInterrupt):
                print("\n  [中断] 停止后续轮次")
                break

            try:
                base_y, min_y, n_probe, grasp_gap = run_episode(
                    robot, features, policy, pre, post, a.duration, i, a.fps,
                    os.path.join(VIDEO_DIR, "round%02d.mp4" % i) if a.video else None,
                    a.n_action_steps)
            except (EOFError, KeyboardInterrupt):
                print("\n  [中断] 停止后续轮次")
                break
            time.sleep(0.3)
            obs = robot.get_observation()
            cv2.imwrite("final_%02d.jpg" % i, obs["front"])

            lift = (base_y - min_y) if (base_y is not None and n_probe) else None
            if grasp_gap is None:
                ok = None
                print("  ⚠ 无法判读夹爪，需人工看 final_%02d.jpg" % i)
            else:
                ok = grasp_gap >= GRASP_GAP_THRESHOLD
                print("  夹爪闭合停止在 %.1f 度（%s）   方块上移 %s px"
                      % (grasp_gap,
                         "夹住物体" if ok else "夹空",
                         lift if lift is not None else "?"))
                print("  → %s" % ("成功" if ok else "失败"))

            results.append(ok)
            rows.append([i, base_y, min_y, (base_y - min_y) if base_y else "", round(grasp_gap,1) if grasp_gap is not None else "", ok])
            try:
                robot.disconnect()
                robot = None
            except Exception:
                robot = None
            valid = [r for r in results if r is not None]
            if valid:
                print("  累计: %d/%d = %.0f%%" % (sum(valid), len(valid), sum(valid) / len(valid) * 100))
            print()
    except KeyboardInterrupt:
        print("\n[中断] 已完成的结果仍会统计")
    finally:
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass
        print()

    valid = [r for r in results if r is not None]
    unknown = len(results) - len(valid)
    if not valid:
        print("没有有效结果")
        return
    k, n = sum(valid), len(valid)
    lo, hi = wilson(k, n)
    print("=" * 58)
    print("  成功  %d / %d = %.1f%%" % (k, n, k / n * 100))
    print("  95%% 置信区间  %.0f%% ~ %.0f%%" % (lo * 100, hi * 100))
    if unknown:
        print("  待人工确认 %d 次（final_XX.jpg）" % unknown)
    print("  总耗时 %.1f 分钟" % ((time.time() - t_start) / 60))
    print("=" * 58)

    out = "batch_result_%s.csv" % time.strftime("%H%M%S")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["round", "base_y", "min_y", "lift_px", "grasp_gap_deg", "success"])
        w.writerows(rows)
    print("  明细: " + out)


if __name__ == "__main__":
    main()
