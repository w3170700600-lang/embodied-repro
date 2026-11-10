"""
eval_offline.py -- 不接真机的模型验证（离线开环评测）。

做什么：
  1. 加载训练好的 ACT checkpoint
  2. 在数据集里抽样若干时间点，用「当前画面 + 当前状态」预测未来 100 步动作
  3. 与人类示范的真实动作逐年比，算 L1 误差 / 相关系数
  4. 和「瞎猜基线」对比（始终输出训练集平均动作），证明模型确实学到了东西
  5. 出图：预测 vs 真实 的 6 个关节曲线

为什么有用：
  这一步不需要机械臂、不需要 GPU，能在训练刚结束就先判断模型是否可用。
  如果离线 L1 都很大、曲线完全对不上，接真机也白跑。

用法：
  python eval_offline.py --ckpt <pretrained_model 目录> [--episodes 8] [--samples 6]
"""
import os
os.environ.setdefault("HF_HOME", "D:/lerobot-data/hf-home")
os.environ.setdefault("HF_LEROBOT_HOME", "D:/lerobot-data/hf-lerobot")

import argparse
import glob
import hashlib
import json

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows 自带中文字体，避免图上全是方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def localize_ckpt(ckpt, device="cpu"):
    """checkpoint 里的 processor json 写死了训练时的 device（cuda）。
    本地无 GPU 时，生成一份把 device 改成 cpu 的副本（硬链接省空间，不动原文件）。"""
    import shutil
    import tempfile

    need = False
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        p = os.path.join(ckpt, name)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                if '"cuda"' in f.read():
                    need = True
    if not need or device == "cuda":
        return ckpt

    tmp = os.path.join(tempfile.gettempdir(),
                       "lerobot_ckpt_%s_%s" % (device, hashlib.md5(os.path.abspath(ckpt).encode()).hexdigest()[:8]))
    os.makedirs(tmp, exist_ok=True)
    for name in os.listdir(ckpt):
        src, dst = os.path.join(ckpt, name), os.path.join(tmp, name)
        if os.path.exists(dst):
            continue
        try:
            os.link(src, dst)          # 同盘硬链接，不占额外空间
        except OSError:
            shutil.copy2(src, dst)
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        p = os.path.join(tmp, name)
        if os.path.isfile(p):
            txt = open(p, encoding="utf-8").read().replace('"cuda"', f'"{device}"')
            open(p, "w", encoding="utf-8").write(txt)
    print(f"      (已生成 {device} 版配置副本: {tmp})")
    return tmp


def load_episodes_meta(root):
    parts = [pd.read_parquet(f) for f in
             sorted(glob.glob(os.path.join(root, "meta", "episodes", "chunk-*", "file-*.parquet")))]
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="D:/lerobot-data/so101_test")
    ap.add_argument("--episodes", type=int, default=8, help="抽多少条 episode")
    ap.add_argument("--samples", type=int, default=6, help="每条抽多少个时间点")
    ap.add_argument("--out", default="D:/lerobot-work/offline_eval")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    print(f"[1/4] 加载数据集 {a.root}")
    ds = LeRobotDataset(repo_id="local/so101_test", root=a.root)
    print(f"      {len(ds)} 帧 / {ds.fps} fps")

    print(f"[2/4] 加载模型 {a.ckpt}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = localize_ckpt(a.ckpt, device=dev)
    policy = ACTPolicy.from_pretrained(ckpt_dir)
    policy.to(dev)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt_dir)
    chunk = policy.config.chunk_size
    print(f"      chunk_size={chunk}  n_action_steps={policy.config.n_action_steps}  "
          f"参数量={sum(p.numel() for p in policy.parameters()):,}")

    print("[3/4] 逐点预测")
    eps = load_episodes_meta(a.root)
    ep_ids = list(eps["episode_index"])[:: max(1, len(eps) // a.episodes)][: a.episodes]

    results = []
    curves = []      # 用于画图
    for ep in ep_ids:
        row = eps[eps["episode_index"] == ep].iloc[0]
        f0, f1 = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        length = f1 - f0
        picks = np.linspace(0.1, 0.85, a.samples)
        for frac in picks:
            gi = f0 + int(frac * length)
            if gi + chunk >= f1:
                continue
            sample = ds[gi]
            gt = torch.stack([ds[gi + k]["action"] for k in range(chunk)])   # (chunk, 6) 真实动作（原始尺度）

            batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in sample.items()}
            with torch.no_grad():
                inp = pre(batch)
                pred = post(policy.predict_action_chunk(inp))[0]             # (chunk, 6)
            err = (pred - gt).abs().mean().item()
            results.append(dict(ep=ep, frac=frac, l1=err))
            curves.append((ep, frac, pred.numpy(), gt.numpy()))

    if not results:
        raise SystemExit("没有采到有效样本")

    # 瞎猜基线：始终输出该条样本的动作均值
    rng = np.random.default_rng(0)
    base_errs = []
    for r in results[:20]:
        ep = r["ep"]
        row = eps[eps["episode_index"] == ep].iloc[0]
        f0, f1 = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        gi = f0 + int(r["frac"] * (f1 - f0))
        gt = torch.stack([ds[gi + k]["action"] for k in range(chunk)])
        base_errs.append((gt.mean(0).expand_as(gt) - gt).abs().mean().item())

    l1s = np.array([r["l1"] for r in results])
    print()
    print(f"      样本数 {len(results)}   平均 L1 = {l1s.mean():.2f} 度   中位 {np.median(l1s):.2f}   最差 {l1s.max():.2f}")
    print(f"      瞎猜基线 L1 = {np.mean(base_errs):.2f} 度   → 提升 {100*(1-l1s.mean()/np.mean(base_errs)):.0f}%")

    print("[4/4] 出图")
    # 取 L1 最接近中位数的样本作为代表，6 个关节各一格
    order = np.argsort([r["l1"] for r in results])
    rep = order[len(order) // 2]
    ep, frac, pred, gt = curves[rep]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for j, ax in enumerate(axes.ravel()):
        ax.plot(gt[:, j], lw=2.2, color="#2c7fb8", label="人类示范")
        ax.plot(pred[:, j], lw=1.8, ls="--", color="#de2d26", label="ACT 预测")
        ax.set_title(f"{JOINTS[j]}    L1 = {np.abs(pred[:, j] - gt[:, j]).mean():.2f} 度")
        ax.set_xlabel("动作块内步数")
        ax.set_ylabel("角度 (度)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"离线开环验证 · 代表样本 ep{ep} @ {frac*100:.0f}% · 全样本平均 L1 = {l1s.mean():.2f} 度",
                 fontsize=13)
    fig.tight_layout()
    png = os.path.join(a.out, "pred_vs_gt.png")
    fig.savefig(png, dpi=130)
    print(f"      图: {png}")

    summary = dict(ckpt=a.ckpt, samples=len(results),
                   l1_mean=float(l1s.mean()), l1_median=float(np.median(l1s)),
                   l1_max=float(l1s.max()), baseline_l1=float(np.mean(base_errs)),
                   improvement_pct=float(100 * (1 - l1s.mean() / np.mean(base_errs))))
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("      汇总:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
