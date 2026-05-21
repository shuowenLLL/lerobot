#!/usr/bin/env python
from __future__ import annotations

import argparse
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--follower-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM1")

    parser.add_argument("--follower-id", type=str, default="my_awesome_follower_arm")
    parser.add_argument("--leader-id", type=str, default="my_awesome_leader_arm")

    parser.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()

    follower_cfg = SO101FollowerConfig(
        port=args.follower_port,
        id=args.follower_id,
        use_degrees=True,
    )

    leader_cfg = SO101LeaderConfig(
        port=args.leader_port,
        id=args.leader_id,
    )

    follower = SO101Follower(follower_cfg)
    leader = SO101Leader(leader_cfg)

    print("[INFO] Connecting follower...")
    follower.connect()

    print("[INFO] Connecting leader...")
    leader.connect()

    print()
    print("[INFO] Calibration ids used:")
    print(f"  follower id = {args.follower_id}")
    print(f"  leader id   = {args.leader_id}")
    print()
    print("[INFO] Expected calibration files:")
    print(
        f"  ~/.cache/huggingface/lerobot/calibration/robots/so_follower/{args.follower_id}.json"
    )
    print(
        f"  ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/{args.leader_id}.json"
    )
    print()
    print("[INFO] Starting calibrated joint-space teleoperation.")
    print("[INFO] Press Ctrl+C to stop.")
    print()

    try:
        while True:
            t0 = time.perf_counter()

            # 读取 leader 当前关节位置
            leader_action = leader.get_action()

            # 直接发送给 follower
          
            sent_action = follower.send_action(leader_action)

            # 简单打印，避免刷屏太多
            print(f"\r[TELEOP] action = {sent_action}", end="", flush=True)

            dt = time.perf_counter() - t0
            precise_sleep(max(1.0 / args.fps - dt, 0.0))

    except KeyboardInterrupt:
        print("\n[INFO] Stopping teleoperation...")

    finally:
        try:
            follower.disconnect()
        except Exception:
            pass

        try:
            leader.disconnect()
        except Exception:
            pass

        print("[INFO] Disconnected.")


if __name__ == "__main__":
    main()