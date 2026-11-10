# -*- coding: utf-8 -*-
"""批量跑真机试验，自动记录成功率到 CSV。

    python run_trials.py --tag P5-v2 --server http://127.0.0.1:8000 --trials 5

每次跑完会问「抓到了吗」，输入 y / n / s（跳过）。
结果追加到 trials.csv，跑完打印本组成功率。
"""
import argparse
import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CLIENT = os.path.join(HERE, "robot_client.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="本组标签，如 P5-v2")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--robot-id", default="my_follower")
    ap.add_argument("--camera", default="0")
    ap.add_argument("--task", default="Grab the green cube")
    ap.add_argument("--max-step-deg", type=float, default=6.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default="trials.csv")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    exists = os.path.exists(out)
    if not exists:
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["time", "tag", "trial", "result"])

    print("=" * 62)
    print(f" 组别 {a.tag}   服务 {a.server}   共 {a.trials} 次   每次 {a.seconds:.0f}s")
    print("=" * 62)

    ok = fail = skip = 0
    for i in range(1, a.trials + 1):
        print(f"\n---------- 第 {i}/{a.trials} 次 ----------")
        cmd = [
            a.python, CLIENT,
            "--server", a.server,
            "--port", a.port,
            "--robot-id", a.robot_id,
            "--camera", str(a.camera),
            "--task", a.task,
            "--seconds", str(a.seconds),
            "--max-step-deg", str(a.max_step_deg),
            "--speed", str(a.speed),
        ]
        subprocess.run(cmd, cwd=HERE)

        while True:
            r = input("\n抓到了吗？(y=成功 / n=失败 / s=跳过): ").strip().lower()
            if r in ("y", "n", "s"):
                break
            print("  输入 y / n / s")

        res = {"y": "success", "n": "fail", "s": "skip"}[r]
        ok += res == "success"
        fail += res == "fail"
        skip += res == "skip"
        with open(out, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), a.tag, i, res])

        if i < a.trials:
            input("  >> 复位方块和机械臂，回车继续...")

    total = ok + fail
    print("\n" + "=" * 62)
    print(f" {a.tag}: 成功 {ok} / 有效 {total}" + (f" = {100*ok/total:.0f}%" if total else "") + f"  (跳过 {skip})")
    print("=" * 62)
    print(f" 明细追加到 {out}")


if __name__ == "__main__":
    main()
