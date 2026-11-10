# -*- coding: utf-8 -*-
"""常驻版自动试验 v3：在控制回路内部判定"抓起来了"，命中即冻结保持。

用法：
    D:/smolvla-env/Scripts/python.exe persistent_trial.py --pos fixed --times 5

v2 → v3 改了什么（对应"抬升后原地抖动、又松爪、判失败"）：

1. 不再跨线程抢总线。v2 在主线程每 0.25s 调一次 get_observation()，和 30fps 的
   控制回路抢同一条串口，控制节奏被打断 → 臂在目标点附近来回抖。v3 在
   robot_wrapper 的 get_observation / send_action 上挂观察钩子，只"旁听"控制回路
   本来就要读的那一帧，零额外总线访问。
2. 判据从"画面里绿方块上移 40px"换成关节角 + 正运动学：末端（FK）相对本回合最低点
   抬高 >55mm，且夹爪处于"合上但没合死"（5.5°~24°，夹着东西）、且正在合上
   （排除张开过程中路过该区间），持续 0.25s。
   依据：56 条训练回合里空合钳 1.4~2.1°，夹住 3cm 方块 12.8~13.7°，抓取瞬间末端
   离桌 0~20mm，抬起后到 130~200mm。方块被夹爪挡住时图像判据失效，关节角不会。
3. 命中即冻结：不再让策略继续输出动作，关节锁在触发瞬间的目标，夹爪锁在当时的
   实际值（不会自己松爪），臂就停在空中夹着方块，等你去取。
"""
import argparse
import collections
import os
import queue
import statistics
import sys
import threading
import time

# 这台机器访问 huggingface.co 被直接拒绝（Errno 10061），而模型与处理器都已完整缓存在本地。
# 不设离线模式时，transformers 会为 chat_template.jinja 发一次 HEAD 请求、失败后重试 5 次
# （1+2+4+8+16 秒 ≈ 30 秒白等）。设为离线后直接从本地缓存取，约 7 秒完成。
# 必须在 transformers / huggingface_hub 被导入之前设置。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "fk"))
import auto_trial as AT  # 复用 find_cube / judge / beep
import cam_util  # noqa: E402
import trial_log  # noqa: E402
from fk_lib import tip_z  # noqa: E402

# 与 run_rollout.py 完全一致的参数（单一来源已拆开，避免模块级打印/输入阻塞）
# 相机 index 不写死：Windows 下 USB 相机的枚举编号会因插拔/重启而错位，
# 写死 1 曾经变成"对着天花板的另一个摄像头"，症状就是"方块摆好了却检测不到"。
CAM_ARGS = ("--robot.cameras={ front: {type: opencv, index_or_path: %d, "
            "width: 640, height: 480, fps: 30}}")

CLI_ARGS = [
    "--strategy.type=base",
    "--robot.type=so101_follower",
    "--robot.port=COM4",
    "--robot.id=my_follower",
    "--robot.max_relative_target=10",
    # 读失败时立即重试（只在失败时发生，稳态零开销）。这台机器的 USB-C 插头接触不良，
    # 偶尔会坏一个包，重试 3 次能吃掉大部分瞬时丢包；真正的断连重试救不了。
    "--robot.num_read_retries=3",
    '--rename_map={"observation.images.front": "observation.images.camera1"}',
    "--task=Grab the green cube",
    "--policy.path=D:/smolvla-ckpt/pretrained_model",
    "--policy.n_action_steps=20",
    "--policy.num_steps=5",  # 与既有基线（--fast）保持一致
    "--device=cuda",
    "--fps=30",
    "--duration=100",
]

TARGET_CUBE = (283, 393)  # 参考位（实测成功落点区中心）
GATE_TOL = 45               # 位置容差 px（约 3.6cm）：实测成功区间比这更宽

START_POSE = {
    "shoulder_pan": 5.0,
    "shoulder_lift": -101.0,
    "elbow_flex": 90.0,
    "wrist_flex": 73.0,
    "wrist_roll": -16.0,
    "gripper": 3.0,
}

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _frame_from_obs(obs):
    """从观测里取相机帧并转成 BGR uint8。"""
    for k, v in obs.items():
        if k.endswith(".pos") or not hasattr(v, "shape") or getattr(v, "ndim", 0) != 3:
            continue
        arr = np.asarray(v)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.5 else arr.astype(np.uint8)
        return np.ascontiguousarray(arr[..., ::-1])
    return None


class VideoRecorder:
    """把"策略自己看到的画面"录成 mp4，带末端高度/夹爪角/判定状态叠加。

    控制线程只做一次拷贝 + put_nowait，编码和画字全在后台线程，
    避免写盘把 30fps 的控制回路拖慢（队列满了就丢帧，绝不让控制回路等）。
    """

    def __init__(self, path, fps=30, size=(640, 480)):
        self.path = path
        self.fps = fps
        self.q = queue.Queue(maxsize=40)
        self.dropped = 0
        self.written = 0
        self.stop = False
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.ok = self.writer.isOpened()
        if not self.ok:
            return
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def push(self, rgb, meta):
        if not self.ok:
            return
        try:
            self.q.put_nowait((rgb, meta))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while not self.stop:
            try:
                item = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            rgb, (t, z, gr, tag) = item
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(frame, f"t={t:5.1f}s   EE {z:5.0f}mm   grip {gr:4.1f}deg",
                        (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            if tag:
                color = (0, 220, 0) if tag.startswith("GRASP") else (0, 165, 255)
                cv2.putText(frame, tag, (8, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            self.writer.write(frame)
            self.written += 1
            # 控制回路偶尔被推理/总线拖慢，塞进来的帧会少于 30fps。若直接按 30fps 编号，
            # 视频就会"快进"。这里按真实时间轴补重复帧，保证回放速度 = 实际速度。
            want = int(round(t * self.fps))
            if want > self.written:
                for _ in range(min(want - self.written, int(self.fps))):
                    self.writer.write(frame)
                    self.written += 1

    def close(self):
        self.stop = True
        if self.ok:
            try:
                self.th.join(timeout=5)
            except Exception:
                pass
            self.writer.release()
        return self.written, self.dropped


def _raw_frame(obs):
    """取原始 RGB 帧（不做颜色转换/不拷贝判断，由调用方决定）。"""
    for k, v in obs.items():
        if k.endswith(".pos") or not hasattr(v, "shape") or getattr(v, "ndim", 0) != 3:
            continue
        arr = np.asarray(v)
        if arr.dtype != np.uint8:
            return None
        return arr
    return None


class GraspGuard:
    """旁听控制回路的观测，判定"是否已经把方块夹起来"，命中后冻结动作。

    只读观测里已有的数据，不额外访问总线；FK 是纯 numpy，单次约 50µs。
    """

    def __init__(self, rise_mm=55.0, gr_lo=5.5, gr_hi=24.0, hold_ticks=8,
                 img_every=6, min_time_s=1.5):
        self.rise_mm = rise_mm
        self.gr_lo = gr_lo
        self.gr_hi = gr_hi
        self.hold_ticks = hold_ticks
        self.img_every = img_every
        self.min_time_s = min_time_s

        self.hold = False
        self.hold_action = None
        self.success = None
        self.on_trigger = None
        self.fault = None          # 回合被中止的原因（如相机掉线）
        self.on_fault = None
        self.enabled = True
        self.trace = []
        self.dt_hist = collections.deque(maxlen=30)   # 控制回路观测间隔，用来看总线卡顿
        self.last_obs_t = None
        self.bus_slow = 0
        self.video = None          # 设了 VideoRecorder 就录这份视角
        self.reset()

    def median_dt_ms(self):
        if not self.dt_hist:
            return 0.0
        return statistics.median(self.dt_hist) * 1000

    def reset(self, base_cube=None):
        self.enabled = True
        self.z_low = None
        self.gr_ema = None
        self.cond_run = 0
        self.success = None
        self.fault = None
        self.dt_hist.clear()
        self.last_obs_t = None
        self.bus_slow = 0
        self.dark_run = 0
        self.hold = False
        self.hold_action = None
        self.t0 = time.time()
        self.base_cube = base_cube
        self.tick = 0
        self.last_z = None
        self.last_gr = None
        self.trace = []

    # -- 观测侧 -------------------------------------------------------------
    def observe(self, obs):
        if not self.enabled:
            return
        try:
            q = [float(obs[f"{n}.pos"]) for n in ARM_JOINTS]
        except (KeyError, TypeError):
            return
        self.tick += 1
        now = time.time()
        if self.last_obs_t is not None:
            self.dt_hist.append(now - self.last_obs_t)
        self.last_obs_t = now
        z = tip_z(q) * 1000.0
        self.z_low = z if self.z_low is None else min(self.z_low, z)
        self.last_z, self.last_gr = z, q[5]
        self.trace.append((round(time.time() - self.t0, 2), round(z, 1), round(q[5], 1)))

        if self.video is not None:
            raw = _raw_frame(obs)
            if raw is not None:
                tag = "GRASP OK - held" if self.success is not None else ""
                self.video.push(raw.copy(), (time.time() - self.t0, z, q[5], tag))

        if self.hold or self.success is not None:
            return

        rise = z - self.z_low
        gr = q[5]
        # 排除"夹爪正在快速张开、路过这个区间"的假象（训练集 ep11 就是 1.4°→31° 的路上
        # 被误判）。不能用"必须下降"：真机上夹爪可能本来就贴着（2°），方块顶进去把爪
        # 撑开到 13°，读数是**上升**的。所以判据是"在合上 或 已经稳住"。
        prev = self.gr_ema
        opening_fast = prev is not None and (gr - prev) > 0.8 and abs(gr - prev) > 1.5
        self.gr_ema = gr if prev is None else 0.7 * prev + 0.3 * gr
        cond = rise > self.rise_mm and self.gr_lo < gr < self.gr_hi and not opening_fast
        self.cond_run = self.cond_run + 1 if cond else 0

        img = None
        if cond or self.tick % self.img_every == 0:
            img = _frame_from_obs(obs)

        # 相机掉线保护：USB 接触不良时会连续出黑帧，此时策略是在对着冻结画面盲动，
        # 必须立刻中止，而不是等它把桌子撞一遍。
        if img is not None:
            if float(img.mean()) < 5.0:
                self.dark_run += 1
                if self.dark_run >= 5:      # 约 1 秒
                    self.fault = "相机连续黑帧（USB 掉了/接触不良）"
                    if self.on_fault is not None:
                        self.on_fault()
                    return
            else:
                self.dark_run = 0

        if self.cond_run >= self.hold_ticks and (time.time() - self.t0) > self.min_time_s:
            self._trigger(q, z, rise, "joint", img)
            return

        # 兜底：关节判据没命中（URDF 偏差）时，仍用画面里方块上移兜一下
        if img is not None and self.base_cube is not None:
            cube = AT.find_cube(img)
            if cube is not None and (self.base_cube[1] - cube[1]) > 45:
                self._trigger(q, z, rise, "vision", img)

    def _trigger(self, q, z, rise, why, img):
        self.success = {
            "z": z, "rise": rise, "gr": q[5], "why": why,
            "img": img, "q": list(q), "t": time.time() - self.t0,
        }
        # 冻结目标 = 最近一次实际下发的关节目标；夹爪改成"此刻的实际值"，
        # 位置模式下舵机就会原地保持这个开度 —— 既不会被策略下一步开爪指令
        # 松掉，也不会继续往死里夹（避免 id=6 的 Overload）
        base = dict(self.hold_action) if self.hold_action else {
            f"{n}.pos": v for n, v in zip(ARM_JOINTS, q)
        }
        base["gripper.pos"] = float(q[5])
        self.hold_action = base
        self.hold = True
        if self.on_trigger is not None:
            self.on_trigger()

    # -- 动作侧 -------------------------------------------------------------
    def clamp(self, action):
        if not self.hold or self.hold_action is None:
            return action
        out = dict(self.hold_action)
        for k, v in action.items():
            out.setdefault(k, v)
        return out

    def install(self, wrapper):
        """在 ThreadSafeRobot 实例上挂钩子（实例属性会覆盖类方法）。

        RTC 引擎、strategy、teardown 拿到的都是同一个 wrapper 对象，
        所以无论谁调用都会被观察到 / 被钳制。
        """
        orig_get = wrapper.get_observation
        orig_send = wrapper.send_action

        def get_observation():
            obs = orig_get()
            try:
                self.observe(obs)
            except Exception as exc:  # 判定出错绝不能拖垮控制回路
                print(f"  [guard] observe 异常 {type(exc).__name__}: {exc}")
            return obs

        def send_action(action):
            try:
                if isinstance(action, dict):
                    action = self.clamp(action)
            except Exception:
                pass
            sent = orig_send(action)
            if isinstance(action, dict):
                self.hold_action = dict(action)
            return sent

        wrapper.get_observation = get_observation
        wrapper.send_action = send_action


def reset_pose(ctx, guard=None, verbose=True):
    """通过已连接的机器人直接驱动到训练起始姿态（不新开串口）。"""
    if guard is not None:
        guard.hold = False  # 复位时要允许动作，先解除冻结
        guard.enabled = False  # 复位途中的姿态不参与成功判定
    wrapper = ctx.hardware.robot_wrapper
    act = {f"{k}.pos": v for k, v in START_POSE.items()}
    for _ in range(60):
        wrapper.send_action(dict(act))
        time.sleep(0.05)
    if verbose:
        obs = wrapper.get_observation()
        cur = {k.split(".")[0]: round(v, 1) for k, v in obs.items() if k.endswith(".pos")}
        print(f"  姿态校准: {cur}")


def grab_from_ctx(ctx, timeout_s=20):
    """从已连接的相机取一帧（BGR），等待画面稳定。"""
    wrapper = ctx.hardware.robot_wrapper
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        img = _frame_from_obs(wrapper.get_observation())
        if img is not None:
            return img
        time.sleep(0.1)
    raise RuntimeError("相机取帧超时")


def diagnose_no_cube(img, path):
    """没检测到方块时：存图 + 用放宽的阈值找可疑色块，给出可读的诊断。

    返回 (cx, cy, w) 或 None（宽松判据找到的疑似方块）。
    """
    cv2.imwrite(path, img)
    mean = float(img.mean())
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    strict = cv2.inRange(hsv, (35, 60, 40), (85, 255, 255))
    loose = cv2.inRange(hsv, (20, 30, 20), (100, 255, 255))
    loose = cv2.morphologyEx(loose, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(loose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]

    print(f"  诊断：画面均值 {mean:.0f}，严格绿 {int(strict.sum()/255)}px，"
          f"宽松绿 {int(loose.sum()/255)}px")
    if mean < 8:
        print("  画面几乎全黑 —— 相机没出图/被遮挡，不是方块的问题")
    print(f"  诊断图已存：{path}")
    if not cnts:
        print("  画面里没有任何绿色色块 —— 方块不在视野内，或整个偏蓝/偏灰")
        return None

    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        m = np.zeros(hsv.shape[:2], np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        hh, ss, vv = cv2.mean(hsv, mask=m)[:3]
        print(f"    色块 面积{int(area):5d}px 中心({x + w // 2},{y + h // 2}) 尺寸{w}x{h} "
              f"HSV≈({hh:.0f},{ss:.0f},{vv:.0f})")
        if best is None and area > 80 and 0.5 < w / max(h, 1) < 2.0:
            best = (x + w / 2, y + h / 2, w)
    if best:
        print(f"  宽松判据把 {best[0]:.0f},{best[1]:.0f} 当成方块（严格判据不认，多半是光照/色偏）")
    return best


def main():
    import draccus
    # 注册 draccus 可选类型（与 lerobot-rollout 入口脚本同样的导入顺序）
    from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
    from lerobot.robots import so_follower  # noqa: F401
    from lerobot.rollout.configs import RolloutConfig
    from lerobot.rollout.context import build_rollout_context
    from lerobot.rollout.strategies import create_strategy

    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", choices=["fixed", "random"], required=True)
    ap.add_argument("--times", type=int, default=10)
    ap.add_argument("--rtc", action="store_true", help="启用 RTC（实时 chunk 接续）推理")
    ap.add_argument("--rise-mm", type=float, default=55.0, help="判定成功所需的末端抬升（mm）")
    ap.add_argument("--cam", type=int, default=None,
                    help="相机 index；不给就按画面内容自动识别（推荐）")
    ap.add_argument("--video", action="store_true",
                    help="录制策略视角视频到 _frames/videos/trialNN.mp4")
    ap.add_argument("--no-gate", action="store_true",
                    help="不做方块位置闸门（测位置包线时用：方块可以摆得比标准位更偏）")
    args = ap.parse_args()

    print("正在识别相机...")
    cam_idx, cam_info = cam_util.pick(preferred=args.cam)
    if cam_info is not None:
        os.makedirs(AT.FRAME_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(AT.FRAME_DIR, "session_start.png"), cam_info["img"])
        cube0 = AT.find_cube(cam_info["img"])
        print(f"开机画面已存 _frames/trials/session_start.png，"
              f"画面里的方块: {cube0 if cube0 else '没找到'}")

    print("正在初始化（模型加载 + 硬件连接，约 20 秒）...")
    # 手动装配配置：--policy.path 需走 from_pretrained（lerobot CLI 的 path 字段处理）
    from lerobot.configs.policies import PreTrainedConfig

    base_args = [
        a for a in CLI_ARGS
        if not a.startswith(("--policy.path", "--policy.n_action_steps", "--policy.num_steps"))
    ] + ["--policy.type=smolvla", CAM_ARGS % cam_idx]
    if args.rtc:
        base_args += ["--inference.type=rtc", "--inference.rtc.execution_horizon=10"]
        print("已启用 RTC 推理（execution_horizon=10）")
    cfg = draccus.parse(config_class=RolloutConfig, args=base_args)
    cfg.policy = PreTrainedConfig.from_pretrained(
        "D:/smolvla-ckpt/pretrained_model", local_files_only=True
    )
    cfg.policy.pretrained_path = "D:/smolvla-ckpt/pretrained_model"
    cfg.policy.n_action_steps = 20
    cfg.policy.num_steps = 5
    shutdown_event = threading.Event()
    ctx = build_rollout_context(cfg, shutdown_event)
    strategy = create_strategy(cfg.strategy)

    guard = GraspGuard(rise_mm=args.rise_mm)
    guard.install(ctx.hardware.robot_wrapper)
    guard.on_trigger = shutdown_event.set
    guard.on_fault = shutdown_event.set

    strategy.setup(ctx)
    print("初始化完成。之后每轮直接开跑，无需再等模型加载。\n")
    print(f"成功判据：末端抬高 >{args.rise_mm:.0f}mm 且夹爪夹住东西（5.5°~24°，空合钳约 1.5°）"
          f"持续 0.25s；命中即冻结保持\n")

    trial_no = AT._trial_no
    done = 0
    try:
        for i in range(1, args.times + 1):
            ans = input(
                f"===== 第 {i}/{args.times} 轮（已判定 {done}）"
                f"——方块放回、手离开后按回车，q 退出 ===== "
            ).strip().lower()
            if ans == "q":
                break

            guard.hold = False
            reset_pose(ctx, guard)
            # 方块位置闸门：偏离训练位置会直接导致抓空（不是模型问题）
            while True:
                before_img = grab_from_ctx(ctx)
                before = AT.find_cube(before_img)
                if before is None:
                    stamp = time.strftime("%H%M%S")
                    path = os.path.join(AT.FRAME_DIR, f"reject_{stamp}.png")
                    cand = diagnose_no_cube(before_img, path)
                    hint = "（y 用疑似方块继续 / w 强行用标准位 / q 退出）" if cand else "（w 强行用标准位 / q 退出）"
                    ans0 = input(f"  怎么处理？回车重拍{hint}: ").strip().lower()
                    if ans0 == "q":
                        raise SystemExit("用户退出")
                    if ans0 == "y" and cand:
                        before = cand
                        print(f"  采用宽松判据的方块位置 {before[0]:.0f},{before[1]:.0f}")
                        break
                    if ans0 == "w":
                        before = (float(TARGET_CUBE[0]), float(TARGET_CUBE[1]), 50.0)
                        print("  已强制使用标准位继续（本轮的方块位置只写进记录，不参与抓取）")
                        break
                    continue
                dx, dy = TARGET_CUBE[0] - before[0], TARGET_CUBE[1] - before[1]
                if abs(dx) <= GATE_TOL and abs(dy) <= GATE_TOL:
                    break
                if args.no_gate:
                    print(f"  [--no-gate] 方块偏离标准位 (dx={dx:+.0f} dy={dy:+.0f}px)，照跑")
                    break
                tips = []
                if abs(dx) > GATE_TOL:
                    tips.append(f"{'往右' if dx > 0 else '往左'} {abs(dx)*0.08:.1f}cm")
                if abs(dy) > GATE_TOL:
                    tips.append(f"{'往桌里' if dy > 0 else '往桌沿'} {abs(dy)*0.08:.1f}cm")
                print(f"  方块偏离标准位 (dx={dx:+.0f} dy={dy:+.0f}px)，请 {'、'.join(tips)}")
                ans2 = input("  摆好后按回车重拍，输入 w 强行继续: ").strip().lower()
                if ans2 == "w":
                    break
            cv2.imwrite(os.path.join(AT.FRAME_DIR, f"trial{trial_no:02d}_before.png"), before_img)

            # 记录方块实际摆放位置（相对标准位），用于事后统计"位置-成功率"包线。
            # 12.5 px/cm 是这套机位的实测标定（早期按 0.08 cm/px 校正过）。
            off_dx = before[0] - TARGET_CUBE[0]
            off_dy = before[1] - TARGET_CUBE[1]
            pos_tag = (f"偏移({off_dx:+.0f},{off_dy:+.0f})px"
                       f"≈({off_dx/12.5:+.1f},{off_dy/12.5:+.1f})cm")
            print(f"  方块位置 {pos_tag}")

            # 清空策略队列与插值状态，保证每轮从干净状态开始
            try:
                strategy._engine.reset()
                strategy._interpolator.reset()
            except Exception:
                pass

            guard.reset(base_cube=before)
            shutdown_event.clear()
            err_box = {}

            rec = None
            if args.video:
                vdir = os.path.join(HERE, "_frames", "videos")
                os.makedirs(vdir, exist_ok=True)
                vpath = os.path.join(vdir, f"trial{trial_no:02d}.mp4")
                rec = VideoRecorder(vpath, fps=int(cfg.fps))
                if rec.ok:
                    guard.video = rec
                    print(f"  录制中 -> {vpath}")
                else:
                    print("  视频编码器打不开，跳过录制")
                    rec = None

            def _run_episode():
                try:
                    strategy.run(ctx)
                except Exception as exc:  # 线程内异常带回主线程
                    err_box["err"] = exc

            print(f"---- 第 {i} 轮开始（最长 {int(cfg.duration)} 秒，抬起即停）----")
            runner = threading.Thread(target=_run_episode, daemon=True)
            runner.start()

            # 主线程只读 guard 缓存的数值做展示，不碰总线
            warned = False
            slow_ticks = 0
            while runner.is_alive():
                time.sleep(0.5)
                dt_ms = guard.median_dt_ms()
                # 每 0.5s 刷一次；观测间隔中位数正常约 33ms（30fps）。
                # 明显的周期性长间隔是策略推理造成的（约 300ms 一次），中位数不受影响；
                # 中位数持续 >80ms 才是总线在打嗝（USB 接触不良）。
                if guard.last_z is not None:
                    print(f"    t={time.time()-guard.t0:5.1f}s  末端 {guard.last_z:6.0f}mm"
                          f"  夹爪 {guard.last_gr:5.1f}°  回路 {dt_ms:4.0f}ms", end="\r")
                if dt_ms > 80:
                    slow_ticks += 1
                    guard.bus_slow += 1
                    if guard.bus_slow >= 4 and not warned:
                        warned = True
                        print("\n  !! 控制回路明显变慢（观测间隔中位 %.0fms）—— "
                              "USB 接触不良，去动一下那块板上的白色插头" % dt_ms)
                else:
                    guard.bus_slow = 0
            runner.join(timeout=8)
            print()

            # 记录本轮控制链路是否干净：手动扶 USB / 接触不良会在多个 0.5s 采样里表现为
            # 回路中位间隔 >80ms。这条信息写进 trials.csv，用于事后区分
            # "策略自己做到的" 和 "靠人维持链路才做到的"。
            link = "链路干净" if slow_ticks == 0 else f"链路卡顿{slow_ticks}次"
            pos_link = f"{pos_tag}，{link}"

            if rec is not None:
                guard.video = None
                if guard.success is not None:
                    # 成功后多录 1.5 秒，让视频里看得到"臂夹着方块悬停"
                    t_end = time.time() + 1.5
                    while time.time() < t_end:
                        try:
                            im = grab_from_ctx(ctx, timeout_s=1)
                        except Exception:
                            break
                        rec.push(np.ascontiguousarray(im[..., ::-1]),
                                 (time.time() - guard.t0, guard.last_z or 0.0,
                                  guard.last_gr or 0.0, "GRASP OK - held"))
                        time.sleep(1 / 20)
                written, dropped = rec.close()
                print(f"  视频已保存：{rec.path}（{written} 帧，丢 {dropped} 帧）")

            if err_box.get("err") is not None:
                print(f">>> 回合内异常: {type(err_box['err']).__name__}: {err_box['err']}")

            trace_path = os.path.join(AT.FRAME_DIR, f"trial{trial_no:02d}_trace.csv")
            try:
                with open(trace_path, "w", encoding="utf-8") as fh:
                    fh.write("t_s,z_mm,gripper_deg\n")
                    for row in guard.trace:
                        fh.write("%.2f,%.1f,%.1f\n" % row)
            except Exception:
                pass

            succ = None if guard.fault else guard.success
            if guard.fault:
                trial_log.add(argparse.Namespace(pos=args.pos, ok="0", fail="?",
                                                 note=f"回合中止：{guard.fault}，{pos_link}"))
                print(f">>> 第 {trial_no} 轮中止：{guard.fault}")
                print("    这不是策略问题 —— 先检查 USB 线/接口，重插后再跑")
            elif succ is not None:
                img = succ.get("img")
                if img is None:
                    try:
                        img = grab_from_ctx(ctx, timeout_s=5)
                    except Exception:
                        img = before_img
                cv2.imwrite(os.path.join(AT.FRAME_DIR, f"trial{trial_no:02d}_after.png"), img)

                # 交叉核对：方块若还清楚地在原位没动，说明抬起的是空的
                after = AT.find_cube(img) if img is not None else None
                contradicted = after is not None and (before[1] - after[1]) < 15
                if contradicted:
                    trial_log.add(argparse.Namespace(
                        pos=args.pos, ok="0", fail="?",
                        note=f"关节判据命中(抬升{succ['rise']:.0f}mm 夹爪{succ['gr']:.1f}°)但画面里方块没动，需复核，{pos_link}"))
                    print(f">>> 第 {trial_no} 轮：判据命中（抬升 {succ['rise']:.0f}mm）"
                          f" 但画面里方块没动 → 记 '?' 待复核")
                else:
                    note = (f"末端抬升 {succ['rise']:.0f}mm、夹爪 {succ['gr']:.1f}° 夹住"
                            f"（{succ['why']}，t={succ['t']:.1f}s 已冻结保持），{pos_link}")
                    trial_log.add(argparse.Namespace(pos=args.pos, ok="1", fail="", note=note))
                    print(f">>> 第 {trial_no} 轮: 成功  抬升 {succ['rise']:.0f}mm、"
                          f"夹爪 {succ['gr']:.1f}°（已冻结，不会自己松爪）")
                    done += 1
            else:
                time.sleep(1.0)
                after_img = grab_from_ctx(ctx)
                after = AT.find_cube(after_img)
                cv2.imwrite(os.path.join(AT.FRAME_DIR, f"trial{trial_no:02d}_after.png"), after_img)

                ok, fail, note = AT.judge(before, after)
                note = f"{note}，{pos_link}"
                if ok is None:
                    print(f">>> 第 {trial_no} 轮无法自动判定（{note}）")
                else:
                    trial_log.add(argparse.Namespace(pos=args.pos, ok=str(ok), fail=fail, note=note))
                    print(f">>> 第 {trial_no} 轮: {'成功' if ok else '失败'}  [{fail or '-'}]  {note}")
                    done += 1

            AT.beep(1)
            if succ is not None:
                print(">>> 臂已停在空中夹着方块（一声短响）——请先把方块取下，再按回车开始下一轮")
            elif guard.fault:
                print(">>> 回合已中止，机械臂停在原地（一声短响）——修好 USB 再继续")
            else:
                print(">>> 可以复位方块了（一声短响）")
            trial_no += 1
    finally:
        print("收尾中...")
        guard.hold = False
        guard.enabled = False
        try:
            strategy.teardown(ctx)
        except Exception as e:
            print(f"teardown 异常: {type(e).__name__}: {e}")

    print(f"\n全部完成，本轮判定 {done} 次，记录见 trials.csv。")


if __name__ == "__main__":
    main()
