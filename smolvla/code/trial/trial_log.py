# -*- coding: utf-8 -*-
"""真机试验记录与统计。

用法：
    python trial_log.py add --pos fixed --ok 1
    python trial_log.py add --pos random --ok 0 --fail C --note "停在方块右侧 5mm"
    python trial_log.py summary

失败类型（按之前的分类）：
    A 不动/乱动      B 走了但偏       C 到位了抓空
    D 抓到又掉       E 抬起时失败     ? 其他
"""
import argparse
import csv
import datetime as dt
import math
import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trials.csv")
FIELDS = ["time", "pos", "ok", "fail", "note"]


def load():
    if not os.path.exists(PATH):
        return []
    with open(PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add(args):
    rows = load()
    with open(PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow({
            "time": dt.datetime.now().strftime("%m-%d %H:%M"),
            "pos": args.pos, "ok": args.ok,
            "fail": args.fail or "", "note": args.note or "",
        })
    print(f"已记录（共 {len(rows) + 1} 条）")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def summary(_):
    rows = load()
    if not rows:
        print("还没有记录。用 add 子命令加一条。")
        return
    print(f"共 {len(rows)} 次试验\n")
    for tag, label in (("fixed", "固定位置"), ("fixed_rtc", "固定+RTC"), ("random", "随机位置")):
        sub = [r for r in rows if r["pos"] == tag]
        if not sub:
            continue
        k = sum(1 for r in sub if r["ok"] == "1")
        lo, hi = wilson(k, len(sub))
        print(f"{label}: {k}/{len(sub)} = {k/len(sub)*100:.0f}%  "
              f"(95% 置信区间 {lo*100:.0f}%~{hi*100:.0f}%)")
    k = sum(1 for r in rows if r["ok"] == "1")
    lo, hi = wilson(k, len(rows))
    print(f"合计    : {k}/{len(rows)} = {k/len(rows)*100:.0f}%  "
          f"(95% 置信区间 {lo*100:.0f}%~{hi*100:.0f}%)")

    fails = [r["fail"] for r in rows if r["ok"] == "0" and r["fail"]]
    if fails:
        print("\n失败类型分布:")
        for t in "ABCDE?":
            c = fails.count(t)
            if c:
                name = {"A": "不动/乱动", "B": "走了但偏", "C": "到位抓空",
                        "D": "抓到又掉", "E": "抬起失败", "?": "其他"}[t]
                print(f"  {t} {name:8s} {c} 次  {'█' * c}")
    print("\n最近 5 条:")
    for r in rows[-5:]:
        print(f"  {r['time']}  {r['pos']:6s}  {'成功' if r['ok']=='1' else '失败'}  "
              f"{r['fail']}  {r['note']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--pos", choices=["fixed", "fixed_rtc", "random"], required=True)
    a.add_argument("--ok", choices=["0", "1"], required=True)
    a.add_argument("--fail", choices=["A", "B", "C", "D", "E", "?"], default="")
    a.add_argument("--note", default="")
    sub.add_parser("summary")
    args = ap.parse_args()
    add(args) if args.cmd == "add" else summary(args)
