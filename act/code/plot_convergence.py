"""plot_convergence.py -- 自动扫描各 checkpoint 的离线评测结果，画收敛曲线"""
import os
import glob
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = "D:/lerobot-work"
BASELINE = 10.15

# 未训练（20 步随机权重）单独登记，目录名不同
# 从未训练模型（20 步随机权重）开始
points = [(0, 35.74)]
smoke = os.path.join(ROOT, "offline_eval_smoke", "summary.json")
if os.path.isfile(smoke):
    points[0] = (0, json.load(open(smoke, encoding="utf-8"))["l1_mean"])

# 步数从 summary.json 里的模型路径提取（目录名不可靠，05k 其实是 50K）
seen = {0}
for d in sorted(glob.glob(os.path.join(ROOT, "offline_eval_*"))):
    s = os.path.join(d, "summary.json")
    if not os.path.isfile(s):
        continue
    data = json.load(open(s, encoding="utf-8"))
    ck = data.get("ckpt", "")
    if "act_smoke" in ck:
        continue
    m = re.search(r"models[\\/](\d+)[\\/]", ck)
    if not m:
        print(f"  [skip] 无法解析步数: {ck}")
        continue
    step = int(m.group(1)) // 1000
    if step in seen:
        continue
    seen.add(step)
    points.append((step, data["l1_mean"]))
    print(f"  {step}K: L1={data['l1_mean']:.2f}  最差={data['l1_max']:.2f}  ({os.path.basename(d)})")

points.sort()
xs = [p[0] for p in points]
ys = [p[1] for p in points]

fig, ax = plt.subplots(figsize=(9.5, 5.4))
ax.plot(xs[1:], ys[1:], "o-", lw=2.6, ms=9, color="#2c7fb8", label="ACT 离线 L1 误差",
        zorder=4)
ax.scatter([xs[0]], [ys[0]], color="#de2d26", s=110, marker="X", zorder=5,
           label=f"未训练 {ys[0]:.1f} 度")
ax.axhline(BASELINE, color="#888", ls=":", lw=2, label=f"瞎猜基线 {BASELINE} 度")

for x, y in zip(xs[1:], ys[1:]):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 13),
                ha="center", fontsize=10, fontweight="bold", color="#1f4e79")

ax.set_xlabel("训练步数（千步）")
ax.set_ylabel("平均 L1 误差（度）")
ax.set_title("离线开环验证：误差随训练步数的收敛\n"
             "56 条示范 / 35 个采样点 / 动作块 100 步 / CPU 推理", fontsize=12)
ax.set_xticks(xs)
ax.set_xticklabels([("0" if x == 0 else f"{x}K") for x in xs])
ax.grid(alpha=0.3)
ax.legend(loc="upper right")
fig.tight_layout()
out = os.path.join(ROOT, "offline_convergence.png")
fig.savefig(out, dpi=140)
print("已保存:", out)
