#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmolVLA 兼容性体检（只读，不修改任何文件）

用法：
    python check_dataset.py /path/to/your_act_dataset

它只读 meta/info.json（以及 meta/episodes*），不加载视频、不依赖 lerobot，
所以在那台 A4000 机器上先跑这个最省事。跑完把输出整段贴回对话即可。
"""

import json
import sys
from pathlib import Path

# SmolVLA 官方 base 模型的配置（来自 lerobot SmolVLAConfig 默认值）
SMOLVLA_CAMERAS = 3        # camera1 / camera2 / camera3
SMOLVLA_CHUNK = 50         # chunk_size，动作块长度
SMOLVLA_MAX_DIM = 32       # max_state_dim / max_action_dim，短向量会补零到 32
SMOLVLA_MIN_EPISODES = 50  # 官方推荐下限


def resolve_root(root: Path) -> Path:
    """允许传：数据集目录 / 数据集目录的上级 / meta/info.json 本身"""
    if root.is_file() and root.name == "info.json":
        return root.parent.parent
    if (root / "meta" / "info.json").exists():
        return root
    # 往上/往下一层找：常见情况是点到了 datasets 目录或数据目录的父级
    for cand in list(root.glob("*/meta/info.json")) + list(root.glob("*/*/meta/info.json")):
        print(f"[i] 你给的不是数据集根目录，自动定位到：{cand.parent.parent}")
        return cand.parent.parent
    return root


def load_info(root: Path) -> dict:
    info = root / "meta" / "info.json"
    if not info.exists():
        sys.exit(f"[×] 没找到 {info}\n    这个目录不像 LeRobotDataset。"
                 f"如果你是 ACT 原生 hdf5（一堆 .hdf5 文件），需要先转成 LeRobot 格式。")
    return json.loads(info.read_text(encoding="utf-8"))


def episode_lengths(root: Path):
    """v2.0 读 meta/episodes.jsonl；v3.0 读 meta/episodes/**/*.parquet"""
    jsonl = root / "meta" / "episodes.jsonl"
    if jsonl.exists():
        out = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(int(json.loads(line)["length"]))
        return out, "meta/episodes.jsonl"

    parquet_dir = root / "meta" / "episodes"
    if parquet_dir.exists():
        try:
            import pandas as pd  # 可选依赖
        except ImportError:
            return [], "meta/episodes/*.parquet（缺 pandas，跳过）"
        files = sorted(parquet_dir.rglob("*.parquet"))
        if not files:
            return [], "meta/episodes/ 为空"
        df = pd.concat([pd.read_parquet(f, columns=["length"]) for f in files])
        return [int(x) for x in df["length"].tolist()], f"meta/episodes/*.parquet ({len(files)} 个)"

    return [], "未找到 episode 长度信息"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = resolve_root(Path(sys.argv[1]).expanduser().resolve())
    info = load_info(root)

    feats = info.get("features", {})
    cams = {k: v for k, v in feats.items() if k.startswith("observation.images")}
    state = feats.get("observation.state")
    action = feats.get("action")

    print("=" * 72)
    print(f"数据集：{root}")
    print(f"格式版本：{info.get('codebase_version', '未知')}     robot_type：{info.get('robot_type')}")
    print(f"episodes：{info.get('total_episodes')}    frames：{info.get('total_frames')}    "
          f"tasks：{info.get('total_tasks')}    fps：{info.get('fps')}")
    print(f"数据量：{round(info.get('total_frames', 0) / max(info.get('total_episodes', 1), 1), 1)} 帧/episode")
    print("-" * 72)
    print("特征：")
    for k, v in feats.items():
        print(f"  {k:38s} dtype={v.get('dtype'):8s} shape={v.get('shape')}")
    print("-" * 72)

    lens, src = episode_lengths(root)
    if lens:
        lens_sorted = sorted(lens)
        print(f"episode 长度（{src}）：min={lens_sorted[0]}  median={lens_sorted[len(lens)//2]}  "
              f"max={lens_sorted[-1]}  p10={lens_sorted[max(0, len(lens)//10)]}")
        print(f"  最短 episode ≈ {lens_sorted[0] / max(info.get('fps') or 30, 1):.1f} 秒")
    else:
        print(f"episode 长度：{src}")
    print("=" * 72)

    # ---------------- 判定 ----------------
    problems, notes, flags = [], [], []

    ver = str(info.get("codebase_version", ""))
    if ver.startswith("v2"):
        problems.append(
            "数据集是 v2.0 格式。新版 lerobot(>=0.4) 默认按 v3.0 读，可能直接报错或读到空。\n"
            "      两条路：① 用 lerobot 自带的 v2->v3 转换脚本先转一份副本；\n"
            "              ② 装与数据同期的老版 lerobot（但 SmolVLA 要新版本，基本只能选①）。"
        )
    elif ver.startswith("v3"):
        notes.append("数据集是 v3.0 格式，新版 lerobot 可直接读。")

    n_cam = len(cams)
    keys = sorted(cams.keys())
    if n_cam == SMOLVLA_CAMERAS:
        notes.append(f"相机 {n_cam} 路，正好对上预训练的 camera1/2/3。")
    elif n_cam < SMOLVLA_CAMERAS:
        notes.append(f"相机 {n_cam} 路，比预训练少 {SMOLVLA_CAMERAS - n_cam} 路。"
                     f"**不需要 empty_cameras** —— 源码里缺的相机键会被直接跳过"
                     f"（modeling_smolvla.py:391-432），空槽位只在 empty_cameras>0 时才补。")
    else:
        notes.append(f"相机 {n_cam} 路，多于预训练的 3 路。多出来的会被忽略，"
                     f"但注意 rename_map 里别把多余的键映射进去。")
    if n_cam >= 1:
        mapped = ", ".join(f'"{k}": "observation.images.camera{i + 1}"' for i, k in enumerate(keys[:3]))
        flags.append(f"--rename_map='{{{mapped}}}'")
        notes.append("  相机键名必须显式映射（见文末【建议追加的训练参数】），官方原文："
                     "“Without it, training stops with a feature mismatch error.” "
                     "映射顺序决定语义：camera1 给主视角。")

    for name, feat, tag in (("state", state, "observation.state"), ("action", action, "action")):
        if feat is None:
            problems.append(f"缺 {tag} 字段，这不是一个可训练的动作数据集。")
            continue
        dim = feat.get("shape", [0])[0] if feat.get("shape") else 0
        if dim == 6:
            notes.append(f"{tag} 是 {dim} 维，与 SO-100/101 的 6DOF 完全对齐，最理想。")
        elif dim < SMOLVLA_MAX_DIM:
            problems.append(f"{tag} 是 {dim} 维，不是 6。pre-训练权重的 config 里固定是 6 维，"
                            f"modeling_smolvla.py:382 会做严格的 shape 校验并直接报错"
                            f"（max_state_dim={SMOLVLA_MAX_DIM} 的补零发生在模型内部，入口不收非 6 维）。"
                            f"要么改 config，要么放弃这条路线。")
        else:
            problems.append(f"{tag} 是 {dim} 维，超过 SmolVLA 的 max_dim={SMOLVLA_MAX_DIM}，需要改模型配置。")

    n_ep = info.get("total_episodes") or 0
    if n_ep < SMOLVLA_MIN_EPISODES:
        problems.append(f"只有 {n_ep} 条 episode，低于官方推荐的 {SMOLVLA_MIN_EPISODES} 条"
                        f"（官方明确说 25 条效果明显不行）。可以练通流程，但别期待成功率。")
    else:
        notes.append(f"{n_ep} 条 episode ≥ 官方推荐 {SMOLVLA_MIN_EPISODES} 条，量是够的；"
                     f"关键看这 {n_ep} 条是否覆盖了足够多的初始位姿变化。")

    if lens and min(lens) < SMOLVLA_CHUNK + 10:
        problems.append(f"最短 episode 只有 {min(lens)} 帧，接近 chunk_size={SMOLVLA_CHUNK}，"
                        f"动作块会跨越 episode 边界，建议把 --policy.chunk_size 调小到 {max(10, min(lens)//2)}。")

    fps = info.get("fps")
    if fps and (fps < 15 or fps > 60):
        notes.append(f"fps={fps}，偏离常见的 30；确认推理时用同样的 fps 采集，否则动作节奏会错。")

    if not cams:
        problems.append("没有 observation.images.* 字段，SmolVLA 需要视觉输入。")

    print("\n【结论】")
    if problems:
        print("  ✗ 有阻塞问题：")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  ✓ 没有阻塞问题，可以直接微调。")
    for n in notes:
        print(f"  · {n}")

    print("\n【建议追加的训练参数】")
    print("  " + (" ".join(flags) if flags else "（无，默认即可）"))
    print("\n【参考】smolvla_base 的硬约束（读官方 config.json 得到）："
          "camera1/2/3 三路可选、state 6 维、action 6 维、chunk_size=50；"
          "默认 freeze_vision_encoder=True + train_expert_only=True。")


if __name__ == "__main__":
    main()
