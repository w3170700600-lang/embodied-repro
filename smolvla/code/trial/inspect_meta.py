# -*- coding: utf-8 -*-
"""读取 LeRobotDataset v3.0 的 meta（任务指令、episode 长度）。只读。"""
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / ".pydeps"))

ROOT = Path(r"D:/lerobot-data/so101_test")

try:
    import pandas as pd
except Exception as e:  # noqa
    sys.exit(f"no pandas: {e}")

# ---- tasks ----
tasks_path = ROOT / "meta" / "tasks.parquet"
if tasks_path.exists():
    t = pd.read_parquet(tasks_path)
    print("tasks.parquet 列:", list(t.columns), "行数:", len(t))
    for i, row in t.iterrows():
        print(f"  [{i}] {dict(row)}")
else:
    print("没有 meta/tasks.parquet")

# ---- episodes ----
files = glob.glob(str(ROOT / "meta" / "episodes" / "**" / "*.parquet"), recursive=True)
print("\nepisodes 文件数:", len(files))
if files:
    e = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print("列:", list(e.columns))
    if "length" in e.columns:
        s = sorted(int(x) for x in e["length"])
        n = len(s)
        print(f"episodes={n}  min={s[0]}  p10={s[n // 10]}  median={s[n // 2]}  max={s[-1]}  sum={sum(s)}")
        print(f"最短 {s[0]} 帧 ≈ {s[0] / 30:.1f}s，中位 {s[n // 2]} 帧 ≈ {s[n // 2] / 30:.1f}s")
        print("最短的 5 条:", s[:5])
        print("最长的 5 条:", s[-5:])
    for col in ("tasks", "task_index"):
        if col in e.columns:
            v = e[col].iloc[0]
            print(f"{col} 首条: {list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else v}")
