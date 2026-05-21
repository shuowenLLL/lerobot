#!/usr/bin/env python
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import rerun as rr

try:
    from rerun import urdf as rr_urdf
except ImportError:
    rr_urdf = None

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


# =========================
# 配置区
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/svla_so101_pickplace"
SAMPLE_INDEX = 0

FALLBACK_TASK = "put the pink lego brick into the transparent box"

SO101_URDF_PATH = "/home/shuowen/Repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

# ===== 真机控制开关 =====
SEND_TO_REAL_ROBOT = True          # True: 同时发给真实机械臂；False: 只做离线+rerrun
REPLAY_ON_RERUN = True             # 是否在 rerun 中回放
LIMIT_STEPS = None                 # None 表示整段都发；例如 10 表示只发前 10 步，第一次上真机建议先设成 5 或 10
SLEEP_PER_STEP_SEC = 0.08          # 发给真机的步间隔，和你当前 rerun 回放一致

# ===== 真机端口与ID =====
FOLLOWER_PORT = "/dev/ttyACM0"     # 改成你的真实端口
FOLLOWER_ID = "my_follower_arm"       # 改成你的真实校准ID
ROBOT_TYPE = "so101_follower"      # 仅做记录/打印用

# ===== 角度解释 =====
# 你的当前离线脚本把数据按“度”来理解，并在 rerun 时转成弧度。
# observation.state / action 在该数据集里你当前实验流程按 degrees 工作是成立的。
DATA_IS_DEGREES = True

OUT_DIR = Path("outputs/offline_smolvla_realrobot")
OUT_DIR.mkdir(parents=True, exist_ok=True)

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

JOINT_MAP = {
    "shoulder_pan": "shoulder_pan",
    "shoulder_lift": "shoulder_lift",
    "elbow_flex": "elbow_flex",
    "wrist_flex": "wrist_flex",
    "wrist_roll": "wrist_roll",
}


# =========================
# 工具函数
# =========================
def maybe_deg_to_rad(x: np.ndarray) -> np.ndarray:
    if DATA_IS_DEGREES:
        return np.deg2rad(x)
    return x


def get_urdf_joint(robot_urdf, joint_name: str):
    joints_obj = robot_urdf.joints

    if isinstance(joints_obj, dict):
        return joints_obj[joint_name]

    if callable(joints_obj):
        joints_obj = joints_obj()

        if isinstance(joints_obj, dict):
            return joints_obj[joint_name]

        for joint in joints_obj:
            if getattr(joint, "name", None) == joint_name:
                return joint

    try:
        for joint in joints_obj:
            if getattr(joint, "name", None) == joint_name:
                return joint
    except TypeError:
        pass

    raise KeyError(f"URDF joint '{joint_name}' not found")


def build_offline_obs_frame(sample: dict, task: str) -> dict:
    """
    把 SO101 数据集的 up/side 两路图，映射成 smolvla_base 期望的
    camera1/camera2/camera3 三路输入。
    """
    img_up = sample["observation.images.up"].detach().cpu()
    img_side = sample["observation.images.side"].detach().cpu()
    empty = torch.zeros_like(img_up)

    return {
        "observation.images.camera1": img_up,
        "observation.images.camera2": img_side,
        "observation.images.camera3": empty,
        "observation.state": sample["observation.state"].detach().cpu(),
        "task": task,
    }


def save_debug_artifacts(sample: dict, task: str, action_chunk_deg: np.ndarray) -> None:
    state_deg = sample["observation.state"].detach().cpu().numpy()

    np.save(OUT_DIR / "input_state_deg.npy", state_deg)
    np.save(OUT_DIR / "predicted_action_chunk_deg.npy", action_chunk_deg)
    np.save(OUT_DIR / "predicted_action_chunk_rad.npy", maybe_deg_to_rad(action_chunk_deg))

    with (OUT_DIR / "predicted_action_chunk_deg.csv").open("w", encoding="utf-8") as f:
        f.write("step,shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper\n")
        for i, row in enumerate(action_chunk_deg):
            f.write(str(i) + "," + ",".join(f"{v:.6f}" for v in row) + "\n")

    meta = {
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "sample_index": SAMPLE_INDEX,
        "task": task,
        "device": DEVICE,
        "data_is_degrees": DATA_IS_DEGREES,
        "urdf_path": SO101_URDF_PATH,
        "input_state_deg": state_deg.tolist(),
        "chunk_shape": list(action_chunk_deg.shape),
        "joint_order_used_for_urdf": JOINT_ORDER,
        "send_to_real_robot": SEND_TO_REAL_ROBOT,
        "follower_port": FOLLOWER_PORT if SEND_TO_REAL_ROBOT else None,
        "follower_id": FOLLOWER_ID if SEND_TO_REAL_ROBOT else None,
    }
    (OUT_DIR / "debug_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def init_rerun_and_load_urdf(urdf_path: str):
    rr.init("smolvla_offline_chunk_rerun", spawn=True)

    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF 不存在: {urdf_path}")

    rr.log_file_from_path(urdf_path, static=True)

    if rr_urdf is None:
        raise ImportError("无法 import rerun.urdf。请确认你的 rerun 版本支持 URDF。")

    if hasattr(rr_urdf, "UrdfTree"):
        return rr_urdf.UrdfTree.from_file_path(urdf_path)

    raise RuntimeError("你的 rerun.urdf 模块里没有 UrdfTree，当前脚本无法继续。")


def log_joint_positions_to_rerun(robot_urdf, joint_positions_rad: dict[str, float]) -> None:
    for logical_name, angle_rad in joint_positions_rad.items():
        urdf_joint_name = JOINT_MAP[logical_name]
        joint = get_urdf_joint(robot_urdf, urdf_joint_name)

        if not hasattr(joint, "compute_transform"):
            raise AttributeError(
                f"joint '{urdf_joint_name}' 没有 compute_transform(angle) 接口，当前 rerun API 不匹配。"
            )

        transform = joint.compute_transform(float(angle_rad))
        rr.log("transforms", transform)


def replay_action_chunk_with_urdf(robot_urdf, action_chunk_deg: np.ndarray) -> None:
    print("[INFO] 开始在 rerun 中回放 action chunk ...")
    print(f"[INFO] chunk shape = {action_chunk_deg.shape}")

    for step_idx, action_deg in enumerate(action_chunk_deg):
        rr.set_time("step", sequence=step_idx)

        action_rad = maybe_deg_to_rad(action_deg)

        joint_positions_rad = {
            "shoulder_pan": float(action_rad[0]),
            "shoulder_lift": float(action_rad[1]),
            "elbow_flex": float(action_rad[2]),
            "wrist_flex": float(action_rad[3]),
            "wrist_roll": float(action_rad[4]),
        }

        log_joint_positions_to_rerun(robot_urdf, joint_positions_rad)

        rr.log("debug/action_deg/shoulder_pan", rr.Scalars(float(action_deg[0])))
        rr.log("debug/action_deg/shoulder_lift", rr.Scalars(float(action_deg[1])))
        rr.log("debug/action_deg/elbow_flex", rr.Scalars(float(action_deg[2])))
        rr.log("debug/action_deg/wrist_flex", rr.Scalars(float(action_deg[3])))
        rr.log("debug/action_deg/wrist_roll", rr.Scalars(float(action_deg[4])))
        rr.log("debug/action_deg/gripper", rr.Scalars(float(action_deg[5])))

        time.sleep(SLEEP_PER_STEP_SEC)

    print("[INFO] rerun 回放完成。")


def connect_real_robot():
    """
    连接真实 SO101 follower。
    这里沿用你项目里 using_smolvla_example.py 的写法：
    用 SO100Follower / SO100FollowerConfig 这套通用 so_follower 接口。
    """
    robot_cfg = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
        use_degrees=False,  # 一般底层接口按弧度/标准空间发送；保留默认机械臂接口习惯
    )
    robot = SO101Follower(robot_cfg)
    robot.connect()
    return robot


def build_robot_dataset_features(robot):
    """
    构造 make_robot_action 所需的 feature 映射。
    """
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    return dataset_features


def send_action_chunk_to_real_robot(robot, dataset_features, action_chunk_tensor: torch.Tensor) -> None:
    """
    将 [T, D] 的 action chunk 逐步发给真实机械臂。
    注意：
    - 这里输入应当是 postprocess 之后、CPU 上的 tensor
    - make_robot_action 会把 policy 输出转成 robot.send_action 可接受的字典/结构
    """
    if action_chunk_tensor.ndim != 2:
        raise ValueError(f"Expected [T, D], got shape {tuple(action_chunk_tensor.shape)}")

    total_steps = action_chunk_tensor.shape[0]
    if LIMIT_STEPS is not None:
        total_steps = min(total_steps, LIMIT_STEPS)

    print(f"[INFO] 准备向真实机械臂发送 {total_steps} 步 action")

    for step_idx in range(total_steps):
        action_t = action_chunk_tensor[step_idx].unsqueeze(0)  # [1, D]
        robot_action = make_robot_action(action_t, dataset_features)

        print(f"[REAL] step={step_idx:03d} action={action_t.squeeze(0).numpy()}")
        #robot.send_action(robot_action)
        time.sleep(SLEEP_PER_STEP_SEC)

    print("[INFO] 真实机械臂 action chunk 发送完成")


def main():
    print(f"[INFO] DEVICE           = {DEVICE}")
    print(f"[INFO] MODEL_ID         = {MODEL_ID}")
    print(f"[INFO] DATASET_ID       = {DATASET_ID}")
    print(f"[INFO] SAMPLE_INDEX     = {SAMPLE_INDEX}")
    print(f"[INFO] SEND_TO_REAL     = {SEND_TO_REAL_ROBOT}")
    print(f"[INFO] FOLLOWER_PORT    = {FOLLOWER_PORT}")
    print(f"[INFO] FOLLOWER_ID      = {FOLLOWER_ID}")
    print(f"[INFO] ROBOT_TYPE       = {ROBOT_TYPE}")

    robot = None

    try:
        # 1) 加载模型
        model = SmolVLAPolicy.from_pretrained(MODEL_ID)
        model.eval()
        print("[INFO] model expected image keys:", list(model.config.image_features.keys()))

        # 2) 加载数据集与样本
        dataset = LeRobotDataset(DATASET_ID)
        sample = dataset[SAMPLE_INDEX]

        task = sample.get("task", None)
        if task is None or (isinstance(task, str) and len(task.strip()) == 0):
            task = FALLBACK_TASK

        print("[INFO] task:", task)
        print("[INFO] sample keys:", list(sample.keys()))
        print("[INFO] observation.state:", sample["observation.state"])

        # 3) 构建 pre/post processor
        preprocess, postprocess = make_pre_post_processors(
            model.config,
            dataset_stats=dataset.meta.stats,
        )

        # 4) 构造离线输入
        obs_frame = build_offline_obs_frame(sample, task)

        # 5) preprocess
        obs = preprocess(obs_frame)

        # 6) 一次性预测整个 action chunk
        with torch.no_grad():
            action_chunk = model.predict_action_chunk(obs)
            action_chunk = postprocess(action_chunk)

        if hasattr(action_chunk, "actions"):
            action_chunk = action_chunk.actions

        if not isinstance(action_chunk, torch.Tensor):
            raise TypeError(f"Unexpected action_chunk type: {type(action_chunk)}")

        if action_chunk.ndim != 3:
            raise ValueError(f"Expected action_chunk shape [B, T, D], got {tuple(action_chunk.shape)}")

        action_chunk = action_chunk.detach().cpu()
        action_chunk_2d = action_chunk[0]          # [T, D]
        action_chunk_deg = action_chunk_2d.numpy() # 用于保存和 rerun

        print("[INFO] predicted action_chunk shape:", action_chunk_deg.shape)
        print("[INFO] first action:", action_chunk_deg[0])
        print("[INFO] last  action:", action_chunk_deg[-1])

        # 7) 保存调试结果
        save_debug_artifacts(sample, task, action_chunk_deg)

        # 8) rerun 回放
        if REPLAY_ON_RERUN:
            robot_urdf = init_rerun_and_load_urdf(SO101_URDF_PATH)

            init_state_deg = sample["observation.state"].detach().cpu().numpy()
            rr.set_time("step", sequence=-1)
            rr.log("debug/input_state_deg/shoulder_pan", rr.Scalars(float(init_state_deg[0])))
            rr.log("debug/input_state_deg/shoulder_lift", rr.Scalars(float(init_state_deg[1])))
            rr.log("debug/input_state_deg/elbow_flex", rr.Scalars(float(init_state_deg[2])))
            rr.log("debug/input_state_deg/wrist_flex", rr.Scalars(float(init_state_deg[3])))
            rr.log("debug/input_state_deg/wrist_roll", rr.Scalars(float(init_state_deg[4])))
            rr.log("debug/input_state_deg/gripper", rr.Scalars(float(init_state_deg[5])))

            replay_action_chunk_with_urdf(robot_urdf, action_chunk_deg)

        # 9) 发给真实机械臂
        if SEND_TO_REAL_ROBOT:
            robot = connect_real_robot()
            dataset_features_for_robot = build_robot_dataset_features(robot)
            send_action_chunk_to_real_robot(
                robot=robot,
                dataset_features=dataset_features_for_robot,
                action_chunk_tensor=action_chunk_2d,
            )

        print(f"[DONE] 调试文件已保存到: {OUT_DIR.resolve()}")

    finally:
        if robot is not None:
            try:
                robot.disconnect()
                print("[INFO] 机械臂已断开连接")
            except Exception as e:
                print(f"[WARN] 机械臂断开连接时出错: {e}")


if __name__ == "__main__":
    main()