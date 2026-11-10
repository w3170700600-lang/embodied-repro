# -*- coding: utf-8 -*-
"""真机 rollout 启动器（带安全检查）。

用法：
    python run_rollout.py --dry-run      # 只打印命令，不驱动机械臂
    python run_rollout.py                # 打印清单并要求输入 RUN 后才启动
    python run_rollout.py --yes          # 跳过确认，直接跑

改参数直接改下面 CONFIG。
"""
import subprocess
import sys

CONFIG = {
    "port": "COM4",                       # COM3=主动臂(leader)  COM4=从动臂(follower)
    "robot_id": "my_follower",
    "camera_index": 1,
    "policy": "D:/smolvla-ckpt/pretrained_model",
    "task": "Grab the green cube",
    "n_action_steps": 20,
    "max_relative_target": 10,            # 单步最大关节变化（度），防止突跳
    "duration": 60,                       # 秒；0=无限
    "fps": 30,
}

CMD = [
    r"D:/smolvla-env/Scripts/lerobot-rollout.exe",
    "--strategy.type=base",
    "--robot.type=so101_follower",
    f"--robot.port={CONFIG['port']}",
    f"--robot.id={CONFIG['robot_id']}",
    "--robot.cameras={ front: {type: opencv, index_or_path: %d, width: 640, height: 480, fps: 30}}"
    % CONFIG["camera_index"],
    f"--robot.max_relative_target={CONFIG['max_relative_target']}",
    '--rename_map={"observation.images.front": "observation.images.camera1"}',
    f"--task={CONFIG['task']}",
    f"--policy.path={CONFIG['policy']}",
    f"--policy.n_action_steps={CONFIG['n_action_steps']}",
    "--device=cuda",
    f"--fps={CONFIG['fps']}",
    f"--duration={CONFIG['duration']}",
]

# --fast：把去噪步数从 10 降到 5，单次推理约省一半时间
if "--fast" in sys.argv:
    CMD.append("--policy.num_steps=5")

CHECKLIST = [
    ("机械臂悬空", "末端离桌面 20cm 以上，行程内没有易碎/贵重物品"),
    ("起始姿态", "必须接近训练起始位姿，见下方角度表"),
    ("电源", "笔记本已插电源（电池模式 GPU 会降频）"),
    ("急停", "手放在电源开关旁，随时能断电"),
    ("相机", "相机 1 视角没被动过，方块在画面中间偏右"),
    ("限幅", f"单步最大 {CONFIG['max_relative_target']}°（已内置）"),
]

# 训练数据 56 条 episode 的起始位姿（度）
START_POSE = {
    "shoulder_pan": 5.2, "shoulder_lift": -101.3, "elbow_flex": 89.5,
    "wrist_flex": 72.7, "wrist_roll": -16.1, "gripper": 3.1,
}

print("=" * 60)
print("启动前检查")
print("=" * 60)
for i, (k, v) in enumerate(CHECKLIST, 1):
    print(f"  {i}. {k}: {v}")
print("-" * 60)
print("训练起始位姿（度）:")
print("   " + "  ".join(f"{k}={v:+.0f}" for k, v in START_POSE.items()))
print("-" * 60)
print("命令:")
for a in CMD[1:]:
    print("   " + a)
print("=" * 60)

if "--dry-run" in sys.argv:
    print("dry-run：未启动。去掉 --dry-run 才会真的动机械臂。")
    sys.exit(0)

if "--yes" not in sys.argv:
    ans = input("确认以上都做到了，输入 RUN 启动（其他任意键取消）：").strip().upper()
    if ans != "RUN":
        print("已取消")
        sys.exit(0)

print("启动中... Ctrl+C 可中断\n")
import os
import time as _t

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rollout.log")
with open(LOG, "w", encoding="utf-8", errors="replace") as fh:
    fh.write(f"# {_t.strftime('%Y-%m-%d %H:%M:%S')}\n# {' '.join(CMD)}\n")
    fh.flush()
    p = subprocess.Popen(CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in p.stdout:
            print(line, end="")
            fh.write(line)
            fh.flush()
    except KeyboardInterrupt:
        p.terminate()
    p.wait()
print(f"\n===== 输出已保存到 {LOG} =====")
sys.exit(p.returncode)
