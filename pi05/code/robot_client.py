#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 远程推理客户端（时间对齐执行版）。

与旧版的差别（解决"前进一步又退一步"）：
  旧版把新动作块【追加】到队列尾，机械臂先走完旧块尾巴再接新块第 0 步；
  而新块第 0 步对应"请求那一刻"的状态，早已过时 → 于是反向运动。
  新版按"请求发出到现在过了多久"跳读到块内正确位置，接续当前状态。

    dry-run:  python robot_client.py --server http://127.0.0.1:8000 --dry-run
    真机:     python robot_client.py --server http://127.0.0.1:8000 --port COM4 \
                --robot-id my_follower --camera 0 --task "Grab the green cube" \
                --seconds 20 --max-step-deg 6 --speed 0.8
"""
import argparse
import base64
import json
import threading
import time
import urllib.request

import cv2
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def post_act(server, jpeg_bytes, state, task, timeout=20):
    payload = json.dumps({
        "image_jpeg_b64": base64.b64encode(jpeg_bytes).decode(),
        "state": [float(x) for x in state],
        "task": task,
    }).encode()
    req = urllib.request.Request(server.rstrip("/") + "/act", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Fetcher(threading.Thread):
    """后台请求；返回结果时带上请求发出时刻，供主循环做时间对齐。"""

    def __init__(self, server, task):
        super().__init__(daemon=True)
        self.server, self.task = server, task
        self.lock = threading.Lock()
        self.req = None
        self.result = None
        self.inflight = False
        self.stop = False
        self.stats = {"calls": 0, "ms": [], "fail": 0}

    def request(self, jpeg, state):
        with self.lock:
            self.req = (jpeg, np.asarray(state, np.float32))

    def take_result(self):
        with self.lock:
            r, self.result = self.result, None
            return r

    def run(self):
        while not self.stop:
            with self.lock:
                req, self.req = self.req, None
            if req is None or self.inflight:
                time.sleep(0.004)
                continue
            jpeg, state = req
            self.inflight = True
            t0 = time.time()
            try:
                r = post_act(self.server, jpeg, state, self.task)
                with self.lock:
                    self.result = (np.asarray(r["actions"], dtype=np.float32), t0)
                self.stats["calls"] += 1
                self.stats["ms"].append((time.time() - t0) * 1000)
            except Exception as e:  # noqa: BLE001
                self.stats["fail"] += 1
                print(f"[warn] 请求失败: {type(e).__name__}: {e}", flush=True)
                time.sleep(0.2)
            finally:
                self.inflight = False


def state_from_obs(obs):
    vec, names = [], []
    for j in JOINTS:
        key = f"{j}.pos"
        if key not in obs:
            cand = [k for k in obs if k.startswith(j) and k.endswith(".pos")]
            if not cand:
                raise KeyError(f"观测里找不到 {key}，实际 keys={list(obs)}")
            key = cand[0]
        vec.append(float(obs[key]))
        names.append(key)
    return np.array(vec, dtype=np.float32), names


def image_from_obs(obs):
    cands = []
    for k, v in obs.items():
        arr = v if isinstance(v, np.ndarray) else np.asarray(v)
        if isinstance(arr, np.ndarray) and arr.ndim == 3:
            cands.append((k, arr))
    if not cands:
        raise KeyError(f"观测里没有图像，实际 keys={list(obs)}")
    cands.sort(key=lambda kv: ("image" not in kv[0].lower(), kv[0]))
    key, img = cands[0]
    if img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[-1]:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img, key


def encode_jpeg(img, quality=85):
    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1] if img.shape[2] == 3 else img,
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--task", default="Grab the green cube")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--robot-id", default="my_follower")
    ap.add_argument("--camera", default="0")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--max-step-deg", type=float, default=6.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--min-remaining", type=int, default=15, help="剩余少于多少步就预取下一块")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="动作低通 0~0.9。注意：会按比例削弱幅度（0.5 = 只走一半），默认关闭")
    ap.add_argument("--skip-lead", type=int, default=4,
                    help="每块开头跳过几步（避开相机滞后导致的回撤段）")
    ap.add_argument("--mode", choices=["overlap", "serial"], default="serial",
                    help="overlap=预取+时间对齐；serial=执行完一块再请求（不跳步，但有停顿）")
    ap.add_argument("--n-use", type=int, default=20, help="serial 模式下每块只执行前几步")
    ap.add_argument("--log", default="", help="把每步实际发出的动作记到 CSV，事后分析抖动")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    fetcher = Fetcher(a.server, a.task)
    fetcher.start()

    if a.dry_run:
        print("[dry-run] 不接机械臂")
        print("  health:", urllib.request.urlopen(a.server.rstrip("/") + "/health", timeout=10).read().decode())
        img = (np.random.rand(a.cam_height, a.cam_width, 3) * 255).astype(np.uint8)
        for i in range(3):
            t0 = time.time()
            r = post_act(a.server, encode_jpeg(img), np.zeros(6, np.float32), a.task)
            print(f"  第{i+1}次: {r['n']} 步, 服务端 {r['ms']}ms, 端到端 {(time.time()-t0)*1000:.0f}ms")
        fetcher.stop = True
        return

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    cam = int(a.camera) if str(a.camera).isdigit() else a.camera
    try:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        cameras = {"front": OpenCVCameraConfig(index_or_path=cam, width=a.cam_width,
                                               height=a.cam_height, fps=int(a.fps))}
    except Exception:
        cameras = {"front": {"type": "opencv", "index_or_path": cam, "width": a.cam_width,
                             "height": a.cam_height, "fps": int(a.fps)}}

    robot = SO101Follower(SO101FollowerConfig(
        port=a.port, id=a.robot_id, cameras=cameras, max_relative_target=a.max_step_deg))
    robot.connect()
    print(f"[robot] 已连接 {a.port} id={a.robot_id}")

    period = 1.0 / (a.fps * a.speed)
    t_end = time.time() + a.seconds
    active, idx, names, last, sent = None, 0, None, None, 0
    skipped = []
    log_rows = []
    wait_ms = 0.0
    t_start = time.time()
    print(f"[mode] {a.mode}  每块 {a.n_use} 步  跳过开头 {a.skip_lead} 步  "
          f"平滑 {a.smooth}  限幅 ±{a.max_step_deg}°  速度 {a.speed}x")
    try:
        while time.time() < t_end:
            t0 = time.time()
            obs = robot.get_observation()
            state, names = state_from_obs(obs)
            img, _ = image_from_obs(obs)

            remain = (len(active) - idx) if active is not None else 0

            if a.mode == "serial":
                # 执行完就请求，新块第 0 步对应请求那一刻 → skip 恒为 0
                if remain <= 0 and not fetcher.inflight:
                    fetcher.request(encode_jpeg(img), state)
                res = fetcher.take_result()
                if res is not None:
                    acts, _t = res
                    lead = a.skip_lead if sent > 0 else 0   # 第一块不跳
                    active = acts[: a.n_use + lead]
                    idx = lead
                    skipped.append(lead)
            else:
                if remain < a.min_remaining and not fetcher.inflight:
                    fetcher.request(encode_jpeg(img), state)
                res = fetcher.take_result()
                if res is not None:
                    acts, t_req = res
                    skip = int(round((time.time() - t_req) * a.fps * a.speed))
                    skip = max(0, min(skip, len(acts) - 1))
                    skipped.append(skip)
                    active, idx = acts, skip          # 接续，而不是重放

            if active is not None and idx < len(active):
                act = np.array(active[idx], dtype=np.float32)
                idx += 1
                if last is not None:
                    act = np.clip(act, last - a.max_step_deg, last + a.max_step_deg)
                    act = a.smooth * last + (1 - a.smooth) * act
                last = act
                robot.send_action({names[i]: float(act[i]) for i in range(len(names))})
                if a.log:
                    log_rows.append([round(time.time() - t_start, 4), sent]
                                    + [round(float(act[i]), 3) for i in range(len(names))])
                sent += 1
                if sent % 60 == 0:
                    ms = fetcher.stats["ms"][-5:]
                    print(f"  执行 {sent} 步 | 块内剩余 {len(active)-idx} | 推理 {sum(ms)/len(ms):.0f}ms "
                          f"| 对齐跳过 {skipped[-1] if skipped else 0} 步 | 失败 {fetcher.stats['fail']}", flush=True)
            else:
                time.sleep(0.004)

            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[stop] Ctrl-C")
    finally:
        fetcher.stop = True
        try:
            robot.disconnect()
        except Exception:
            pass
        ms = fetcher.stats["ms"]
        print(f"[done] 执行 {sent} 步，请求 {fetcher.stats['calls']} 次，失败 {fetcher.stats['fail']} 次"
              + (f"，推理均值 {sum(ms)/len(ms):.0f}ms" if ms else "")
              + (f"，平均对齐跳过 {np.mean(skipped):.1f} 步" if skipped else ""))
        if a.log and log_rows:
            import csv as _csv
            import os as _os
            path = a.log if _os.path.isabs(a.log) else _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)), a.log)
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = _csv.writer(f)
                w.writerow(["t", "step"] + list(names or []))
                w.writerows(log_rows)
            print(f"[log] 动作序列已存 {path}（{len(log_rows)} 步）")


if __name__ == "__main__":
    main()
