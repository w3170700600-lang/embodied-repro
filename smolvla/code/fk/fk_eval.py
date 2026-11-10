# -*- coding: utf-8 -*-
"""把关节角误差换算成末端执行器偏移（毫米）。

输入：predictions npz（pred/gt，单位度）+ SO-101 URDF
输出：末端位置误差分布、逐关节贡献、位姿合理性检查
"""
import sys
import xml.etree.ElementTree as ET

import numpy as np

URDF = "fk/so101_new_calib.urdf"
NPZ = sys.argv[1] if len(sys.argv) > 1 else "fk/pred_020000.npz"
CHAIN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TIP = "gripper_frame_link"


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_angle_R(axis, th):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def T(R, p):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p
    return M


# ---- 解析 URDF ----
root = ET.parse(URDF).getroot()
joints = {}
for j in root.findall("joint"):
    o = j.find("origin")
    xyz = np.array([float(x) for x in (o.get("xyz", "0 0 0")).split()])
    rpy = [float(x) for x in (o.get("rpy", "0 0 0")).split()]
    ax = j.find("axis")
    axis = np.array([float(x) for x in ax.get("xyz", "0 0 1").split()]) if ax is not None else None
    joints[j.get("name")] = {
        "type": j.get("type"), "parent": j.find("parent").get("link"),
        "child": j.find("child").get("link"), "xyz": xyz, "R": rpy_to_R(*rpy), "axis": axis,
    }

by_child = {v["child"]: (k, v) for k, v in joints.items()}
chain = []
link = TIP
while link in by_child:
    name, j = by_child[link]
    chain.append((name, j))
    link = j["parent"]
    if link == "base_link":
        break
chain.reverse()
print("关节链:", " -> ".join(n for n, _ in chain))

RAD = np.pi / 180.0


def fk(q_deg):
    M = np.eye(4)
    for name, j in chain:
        M = M @ T(j["R"], j["xyz"])
        if j["type"] != "fixed":
            i = CHAIN.index(name)
            M = M @ T(axis_angle_R(j["axis"], q_deg[i] * RAD), np.zeros(3))
    return M


d = np.load(NPZ)
pred, gt = d["pred"], d["gt"]
print(f"帧数: {len(pred)}")

p_gt = np.array([fk(q)[:3, 3] for q in gt])
p_pr = np.array([fk(q)[:3, 3] for q in pred])

print("\n[合理性检查] GT 末端位置（米）")
for i, ax in enumerate("xyz"):
    print(f"  {ax}: [{p_gt[:, i].min():+.3f}, {p_gt[:, i].max():+.3f}]")
print(f"  离基座距离: {np.linalg.norm(p_gt, axis=1).mean():.3f} m (范围 "
      f"{np.linalg.norm(p_gt, axis=1).min():.3f}~{np.linalg.norm(p_gt, axis=1).max():.3f})")

err_mm = np.linalg.norm(p_pr - p_gt, axis=1) * 1000
print("\n[末端偏移] 预测 vs 真值")
print(f"  平均 {err_mm.mean():.1f} mm | 中位 {np.median(err_mm):.1f} mm | "
      f"p90 {np.percentile(err_mm, 90):.1f} mm | 最大 {err_mm.max():.1f} mm")

print("\n[逐关节贡献] 单独扰动该关节（保持其余不变）造成的末端偏移")
for i, name in enumerate(CHAIN):
    dq = np.abs(pred[:, i] - gt[:, i])
    contrib = []
    for k, q in enumerate(gt):
        q2 = q.copy()
        q2[i] = pred[k, i]
        contrib.append(np.linalg.norm(fk(q2)[:3, 3] - p_gt[k]) * 1000)
    contrib = np.array(contrib)
    print(f"  {name:14s} 角度误差 {dq.mean():5.2f}°  →  末端 {contrib.mean():5.1f} mm "
          f"(p90 {np.percentile(contrib, 90):5.1f})")

# 与角度误差总和对比（验证非线性程度）
naive = np.abs(pred - gt).sum(axis=1).mean()
print(f"\n角度误差绝对值和均值: {naive:.2f}°")
