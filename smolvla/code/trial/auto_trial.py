# -*- coding: utf-8 -*-
"""自动循环真机试验（v2）：判定照改在下一轮开始时拍，避开手与相机的竞争。

用法：
    D:/smolvla-env/Scripts/python.exe auto_trial.py --pos fixed --times 10

启动前必须做两件事：
    1. 把机械臂摆到训练起始姿态（伏在桌上、夹爪朝下接近白纸）——
       rollout 连接那一刻的位姿会被记住并作为每轮的回位点
    2. 方块放回白纸上

流程：每轮按回车 → 拍"轮开始照"（同时充当上一轮的判定照）→ 跑 rollout →
下一轮按回车时自动判定上一轮。最后一轮结束后再按一次回车拍最终判定照。
"""
import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import trial_log  # noqa: E402

CAM_INDEX = 1
PY = sys.executable
FRAME_DIR = os.path.join(HERE, "_frames", "trials")
os.makedirs(FRAME_DIR, exist_ok=True)

START_POSE = "pan≈+5 / lift≈-101 / elbow≈+90 / wrist_flex≈+73 / wrist_roll≈-16 / gripper≈+3（伏桌）"

_trial_no = sum(1 for f in os.listdir(FRAME_DIR) if f.endswith("_before.png")) + 1


def looks_like_our_scene(frame, require_cube_on_paper=True):
    """确认画面来自训练场景。

    通用判据：画面下方必须有白纸亮斑（>1500px）。
    require_cube_on_paper=True 时还要求绿方块紧邻白纸——
    只适用于「跑前」画面；「跑后」画面里方块可能已被夹起离开桌面，
    此时方块不在纸旁是正常的。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    roi = hsv[h // 3:, :]
    bright = ((roi[:, :, 1] < 45) & (roi[:, :, 2] > 195)).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in cnts if cv2.contourArea(c) > 1500]
    if not big:
        return False
    if not require_cube_on_paper:
        return True
    cube = find_cube(frame)
    if cube is None:
        return False
    cx, cy, _ = cube
    for c in big:
        px, py = c.mean(axis=0)[0]
        py += h // 3
        if ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 < 60:
            return True
    return False


def grab_frame(timeout_s=60, require_cube_on_paper=True):
    """打开相机并取通过场景校验的帧。会自动等到手离开画面。"""
    deadline = time.time() + timeout_s
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    attempt = 0
    while time.time() < deadline:
        backend = backends[attempt % len(backends)]
        cap = cv2.VideoCapture(CAM_INDEX, backend)
        frame = None
        for _ in range(30):
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
        cap.release()
        if frame is not None and looks_like_our_scene(frame, require_cube_on_paper):
            return frame
        attempt += 1
        if frame is not None:
            cv2.imwrite(os.path.join(FRAME_DIR, "..", "cam", "rejected.png"), frame)
            hint = "画面不对（手还在画面里？等手离开自动继续）"
        else:
            hint = "打不开"
        print(f"    等待有效画面（第 {attempt} 次，{hint}）...")
        time.sleep(1.5)
    raise RuntimeError(f"相机 {timeout_s} 秒内没等到有效画面（索引 {CAM_INDEX}）")


def find_cube(frame):
    """检测绿色方块，返回 (cx, cy, w) 或 None。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 60, 40), (85, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 80:
        return None
    x, y, w, h = cv2.boundingRect(c)
    return (x + w / 2, y + h / 2, w)


def judge(before, after):
    """抬升是成功主判据；横向位移而无抬升=疑似撞飞。"""
    if before is None:
        return None, "?", "跑前没检测到方块"
    if after is None:
        return 0, "?", "跑后方块不可见（可能被夹进爪里遮挡或撞出画面）"
    bx, by, _ = before
    ax, ay, _ = after
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    lift = by - ay
    if lift > 40:
        return 1, "", f"方块抬升 {lift:.0f}px（位移 {dist:.0f}px）"
    if dist > 60:
        return 0, "?", f"位移 {dist:.0f}px 但无抬升，疑似撞飞，需人工复核"
    return 0, "C", f"方块基本没动（位移 {dist:.0f}px）"


def log_trial(pos, ok, fail, note, n):
    trial_log.add(argparse.Namespace(pos=pos, ok=str(ok), fail=fail, note=note))
    tag = "成功" if ok else "失败"
    print(f">>> 第 {n} 轮判定: {tag}  [{fail or '-'}]  {note}")
    subprocess.call([sys.executable, os.path.join(HERE, "trial_log.py"), "summary"])


def beep(times=2):
    """提示音：rollout 结束后连响=请勿碰桌面；捕获完成短响=可以动手了。"""
    try:
        import winsound
        for _ in range(times):
            winsound.Beep(1200, 350)
            time.sleep(0.1)
    except Exception:
        pass


def main():
    global _trial_no
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", choices=["fixed", "random"], required=True)
    ap.add_argument("--times", type=int, default=10)
    args = ap.parse_args()

    print(f"起始姿态目标: {START_POSE}")
    print("（rollout 连接那一刻的臂位姿会成为每轮回位点，务必先摆好再回车）\n")

    done = 0
    for i in range(1, args.times + 1):
        ans = input(
            f"===== 第 {i}/{args.times} 轮（已判定 {done}）"
            f"——方块放回、手离开画面后按回车，q 退出 ===== "
        ).strip().lower()
        if ans == "q":
            break

        # 每轮先程序化校准到训练起始姿态（几度漂移会触发策略冻结模式）
        try:
            import pose_reset
            pose_reset.reset()
        except Exception as e:
            print(f"  姿态校准失败（{type(e).__name__}: {e}），用当前姿态继续")

        before_img = grab_frame()
        before = find_cube(before_img)
        n = _trial_no
        cv2.imwrite(os.path.join(FRAME_DIR, f"trial{n:02d}_before.png"), before_img)
        _trial_no += 1

        print(f"---- 第 {i} 轮启动 rollout（60 秒，机械臂将开始动作）----")
        r = subprocess.call([sys.executable, os.path.join(HERE, "run_rollout.py"), "--yes", "--fast"], cwd=HERE)
        if r != 0:
            print(f"\n!! rollout 异常退出（exit {r}），本轮不计入统计，重跑本轮即可。")
            _trial_no -= 1
            os.remove(os.path.join(FRAME_DIR, f"trial{n:02d}_before.png"))
            continue

        # 判定照必须在方块被复位之前拍：等机械臂回位，然后连续尝试
        print("\a>>> 【重要】请勿碰桌面！正在拍判定照（约 10 秒），听到两声短响后再动手 <<<")
        beep(2)
        time.sleep(5)  # 等机械臂回初始位姿
        try:
            after_img = grab_frame(timeout_s=90, require_cube_on_paper=False)  # 抓后方块可离开纸
            beep(1)
            after = find_cube(after_img)
            cv2.imwrite(os.path.join(FRAME_DIR, f"trial{n:02d}_after.png"), after_img)
            ok, fail, note = judge(before, after)
            if ok is not None:
                log_trial(args.pos, ok, fail, note, n)
                done += 1
            else:
                print(f">>> 第 {n} 轮无法自动判定，记 '?' 稍后口头补录")
                trial_log.add(argparse.Namespace(pos=args.pos, ok="0", fail="?", note=note))
        except RuntimeError as e:
            print(f">>> 判定照失败（{e}）。本轮先记 '?'，跑完口头告诉我结果，我来补录。")
            trial_log.add(argparse.Namespace(pos=args.pos, ok="0", fail="?", note=f"判定照失败: {e}"))

    print(f"\n全部完成，记录见 trials.csv。")


if __name__ == "__main__":
    main()
