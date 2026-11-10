# -*- coding: utf-8 -*-
"""常驻版冒烟测试：8 秒单轮，验证初始化/姿态校准/取帧/判定全链路。"""
import threading
import time

import cv2
import numpy as np


def main():
    import draccus
    from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
    from lerobot.robots import so_follower  # noqa: F401
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.rollout.configs import RolloutConfig
    from lerobot.rollout.context import build_rollout_context
    from lerobot.rollout.strategies import create_strategy

    import persistent_trial as P
    import auto_trial as AT

    base_args = [
        a for a in P.CLI_ARGS
        if not a.startswith(("--policy.path", "--policy.n_action_steps", "--policy.num_steps", "--duration"))
    ] + ["--policy.type=smolvla", "--duration=8"]
    cfg = draccus.parse(config_class=RolloutConfig, args=base_args)
    cfg.policy = PreTrainedConfig.from_pretrained(
        "D:/smolvla-ckpt/pretrained_model", local_files_only=True
    )
    cfg.policy.pretrained_path = "D:/smolvla-ckpt/pretrained_model"
    cfg.policy.n_action_steps = 20
    cfg.policy.num_steps = 5

    print("构建上下文...")
    ctx = build_rollout_context(cfg, threading.Event())
    strategy = create_strategy(cfg.strategy)
    strategy.setup(ctx)
    print("初始化完成")

    P.reset_pose(ctx)
    before_img = P.grab_from_ctx(ctx)
    cv2.imwrite("_frames/trials/smoke_before.png", before_img)
    print("跑前方块:", AT.find_cube(before_img))

    print("运行 8 秒回合...")
    t0 = time.time()
    strategy.run(ctx)
    print(f"回合结束，用时 {time.time()-t0:.1f}s")

    time.sleep(1)
    after_img = P.grab_from_ctx(ctx)
    cv2.imwrite("_frames/trials/smoke_after.png", after_img)
    print("跑后方块:", AT.find_cube(after_img))
    print("判定:", AT.judge(AT.find_cube(before_img), AT.find_cube(after_img)))

    strategy.teardown(ctx)
    print("冒烟测试通过")


if __name__ == "__main__":
    main()
