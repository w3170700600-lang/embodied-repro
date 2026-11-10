# -*- coding: utf-8 -*-
"""SO-101 正运动学（纯 numpy，无第三方依赖）。

从 URDF 解析关节链，输入 6 个关节角（度），输出末端（gripper_frame_link）位姿。
用途：把"臂到底抬起来了没有"变成物理量（末端高度 mm），而不是靠图像里方块可见性——
方块被夹爪挡住时图像判据会失效，关节角永远不会。
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "so101_new_calib.urdf")
CHAIN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TIP = "gripper_frame_link"
RAD = np.pi / 180.0


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_angle_R(axis, th):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _T(R, p):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p
    return M


def _load_chain(urdf=URDF):
    root = ET.parse(urdf).getroot()
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(x) for x in (o.get("xyz", "0 0 0")).split()])
        rpy = [float(x) for x in (o.get("rpy", "0 0 0")).split()]
        ax = j.find("axis")
        axis = np.array([float(x) for x in ax.get("xyz", "0 0 1").split()]) if ax is not None else None
        joints[j.get("name")] = {
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": xyz,
            "R": rpy_to_R(*rpy),
            "axis": axis,
        }
    by_child = {v["child"]: (k, v) for k, v in joints.items()}
    chain, link = [], TIP
    while link in by_child:
        name, j = by_child[link]
        chain.append((name, j))
        link = j["parent"]
        if link == "base_link":
            break
    chain.reverse()
    return chain


_CHAIN = _load_chain()


def fk(q_deg):
    """6 关节角（度）→ 4x4 末端位姿。"""
    M = np.eye(4)
    for name, j in _CHAIN:
        M = M @ _T(j["R"], j["xyz"])
        if j["type"] != "fixed":
            i = CHAIN.index(name)
            M = M @ _T(axis_angle_R(j["axis"], float(q_deg[i]) * RAD), np.zeros(3))
    return M


def tip_z(q_deg):
    """末端高度（米，基座坐标系 z 向上）。"""
    return float(fk(q_deg)[2, 3])


if __name__ == "__main__":
    q = [5, -101, 90, 73, -16, 3]
    p = fk(q)[:3, 3]
    print("起始姿态末端 (m):", np.round(p, 3), " 高度 mm:", round(p[2] * 1000, 1))
