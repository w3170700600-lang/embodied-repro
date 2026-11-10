# -*- coding: utf-8 -*-
"""SmolVLA 离线开环评估：把训练集里的帧喂给模型，比较预测动作与真值动作。

用法（在 A4000 上，venv 里跑）：
    python eval_offline.py <checkpoint_dir> [episodes逗号分隔] [步长]

注意：这是**训练集**上的误差，衡量的是拟合度，不是泛化能力。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ROOT = Path("/data/workspace/ei/challenge-532514/smolvla")
CKPT = Path(sys.argv[1]).resolve()
EPISODES = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,20").split(",")]
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 8
TASK = "Grab the green cube"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
OUTDIR = ROOT / "outputs" / "eval"
OUTDIR.mkdir(parents=True, exist_ok=True)
TAG = CKPT.parent.name if CKPT.name == "pretrained_model" else CKPT.name

torch.manual_seed(0)
np.random.seed(0)

print(f"== 加载模型 {TAG} ==")
policy = SmolVLAPolicy.from_pretrained(str(CKPT)).to("cuda").eval()
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(CKPT))
print("== 加载数据集 ==")
ds = LeRobotDataset(repo_id="local/so101_test", root=str(ROOT / "datasets" / "so101_test"))

ep_meta = pq.read_table(ROOT / "datasets" / "so101_test" / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
                        columns=["episode_index", "dataset_from_index", "dataset_to_index"]).to_pydict()
ranges = dict(zip(ep_meta["episode_index"], zip(ep_meta["dataset_from_index"], ep_meta["dataset_to_index"])))

records = []
per_ep = {}
for ep in EPISODES:
    s, e = ranges[ep]
    idxs = list(range(s, e, STRIDE))
    errs = []
    for i in idxs:
        item = ds[i]
        obs = {
            "observation.images.front": item["observation.images.front"],
            "observation.state": item["observation.state"],
            "task": TASK,
        }
        batch = pre(obs)
        policy.reset()
        with torch.no_grad():
            chunk = post(policy.predict_action_chunk(batch))
        pred = chunk[0, 0].float().cpu().numpy()
        gt = item["action"].float().cpu().numpy()
        err = np.abs(pred - gt)
        errs.append(err)
        per_ep.setdefault(ep, []).append({"t": i - s, "pred": pred, "gt": gt})
    errs = np.stack(errs)
    mae = errs.mean(axis=0)
    print(f"ep{ep}: {len(idxs)} 帧 | 总体 MAE {errs.mean():.3f}° | "
          + " ".join(f"{n}={v:.2f}" for n, v in zip(JOINTS, mae)))
    records.append(mae)

all_mae = np.stack(records).mean(axis=0)
print("\n== 各关节平均绝对误差（度）==")
for n, v in zip(JOINTS, all_mae):
    print(f"  {n:14s} {v:6.2f}")
print(f"  {'总体':14s} {all_mae.mean():6.2f}")

json.dump({"ckpt": str(CKPT), "tag": TAG, "episodes": EPISODES, "stride": STRIDE,
           "mae_per_joint": dict(zip(JOINTS, all_mae.tolist())),
           "mae_overall": float(all_mae.mean())},
          open(OUTDIR / f"metrics_{TAG}.json", "w"), indent=2)

# 存逐帧预测，供后续正运动学换算使用
fp, fg, fe, fi = [], [], [], []
for ep, rec in per_ep.items():
    for r in rec:
        fp.append(r["pred"])
        fg.append(r["gt"])
        fe.append(ep)
        fi.append(r["t"])
np.savez(OUTDIR / f"pred_{TAG}.npz", pred=np.array(fp), gt=np.array(fg),
         ep=np.array(fe), t=np.array(fi))
print(f"逐帧预测已存: pred_{TAG}.npz ({len(fp)} 帧)")

# ---- 画轨迹图 ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = EPISODES[0]
    rec = per_ep[ep]
    t = [r["t"] / 30.0 for r in rec]
    fig, axes = plt.subplots(6, 1, figsize=(11, 14), sharex=True)
    for j in range(6):
        ax = axes[j]
        ax.plot(t, [r["gt"][j] for r in rec], color="#1f77b4", lw=2, label="ground truth")
        ax.plot(t, [r["pred"][j] for r in rec], color="#d62728", lw=1.6, ls="--", label="SmolVLA")
        ax.set_ylabel(JOINTS[j], fontsize=9)
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=9, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"ep{ep} open-loop: predicted vs recorded joint targets", fontsize=12)
    fig.tight_layout()
    out = OUTDIR / f"traj_ep{ep}_{TAG}.png"
    fig.savefig(out, dpi=110)
    print(f"\n轨迹图: {out}")
except Exception as e:  # noqa
    print(f"[警告] 画图失败: {e}")
