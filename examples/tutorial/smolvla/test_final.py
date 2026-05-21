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

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


# =========================
# 配置区
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/svla_so101_pickplace"
SAMPLE_INDEX = 0

FALLBACK_TASK = "pick up the pink lego brick"

SO101_URDF_PATH = "/home/shuowen/Repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

# ===== 真机控制开关 =====
SEND_TO_REAL_ROBOT = False
REPLAY_ON_RERUN = True
LIMIT_STEPS = None
SLEEP_PER_STEP_SEC = 0.08

# ===== 真机端口与ID =====
FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "my_follower_arm"
ROBOT_TYPE = "so101_follower"

# ===== RealSense 相机配置 =====
CAMERA_CONFIG = {
    "camera1": RealSenseCameraConfig(
        serial_number_or_name="815412070997",
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        use_depth=False,
        rotation=Cv2Rotation.NO_ROTATION,
    ),
}

# ===== 单位解释 =====
# 这里只控制 rerun/保存文件时是否把 action 当 degree 再转 rad。
# 如果你后面确认 action 本身更像弧度，这里改成 False。
DATA_IS_DEGREES = True

OUT_DIR = Path("outputs/live_smolvla_realrobot_camera1_only")
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


def flatten_state_tensor(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().flatten().numpy()
    return np.asarray(x).flatten()


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


def build_robot_dataset_features(robot):
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    return dataset_features


def build_live_obs_frame_camera1_only(
    robot,
    dataset_features: dict,
    task: str,
    device: torch.device,
) -> dict:
    """
    真机版本：
    - state: 来自 robot.get_observation() -> build_inference_frame(...)
    - image: 来自真实 RealSense camera1
    - camera2 / camera3: 全零
    """
    raw_obs = robot.get_observation()
    print("[INFO] robot observation keys:", list(raw_obs.keys()))

    for k, v in raw_obs.items():
        if isinstance(v, torch.Tensor):
            cpu_v = v.detach().cpu()
            if cpu_v.ndim <= 1:
                print(f"[OBS] {k}: shape={tuple(cpu_v.shape)}, dtype={cpu_v.dtype}, value={cpu_v}")
            else:
                print(
                    f"[OBS] {k}: shape={tuple(cpu_v.shape)}, dtype={cpu_v.dtype}, "
                    f"min={cpu_v.min().item():.3f}, max={cpu_v.max().item():.3f}"
                )
        else:
            print(f"[OBS] {k}: type={type(v)}, value={v}")

    robot_obs_frame = build_inference_frame(
        observation=raw_obs,
        ds_features=dataset_features,
        device=device,
        task=task,
        robot_type=ROBOT_TYPE,
    )

    if "observation.state" not in robot_obs_frame:
        raise KeyError(
            f"build_inference_frame(...) 没生成 'observation.state'。当前 keys={list(robot_obs_frame.keys())}"
        )

    if "observation.images.camera1" not in robot_obs_frame:
        raise KeyError(
            f"build_inference_frame(...) 没生成 'observation.images.camera1'。当前 keys={list(robot_obs_frame.keys())}"
        )

    real_state = robot_obs_frame["observation.state"].detach().cpu().float()
    real_img_camera1 = robot_obs_frame["observation.images.camera1"].detach().cpu()

    print("[INFO] state from build_inference_frame:", real_state)
    print("[INFO] camera1 from build_inference_frame shape:", tuple(real_img_camera1.shape))

    empty = torch.zeros_like(real_img_camera1)

    live_obs_frame = {
        "observation.images.camera1": real_img_camera1,
        "observation.images.camera2": empty.clone(),
        "observation.images.camera3": empty.clone(),
        "observation.state": real_state,
        "task": task,
    }

    return live_obs_frame


def save_debug_artifacts(task: str, input_state_used, action_chunk_raw: np.ndarray) -> None:
    input_state_used = np.asarray(input_state_used, dtype=np.float32).flatten()

    np.save(OUT_DIR / "model_input_state_used.npy", input_state_used)
    np.save(OUT_DIR / "predicted_action_chunk_raw.npy", action_chunk_raw)
    np.save(OUT_DIR / "predicted_action_chunk_rad_for_rerun.npy", maybe_deg_to_rad(action_chunk_raw))

    with (OUT_DIR / "predicted_action_chunk_raw.csv").open("w", encoding="utf-8") as f:
        f.write("step,shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper\n")
        for i, row in enumerate(action_chunk_raw):
            f.write(str(i) + "," + ",".join(f"{v:.6f}" for v in row) + "\n")

    meta = {
        "model_id": MODEL_ID,
        "dataset_id_for_stats_only": DATASET_ID,
        "sample_index_for_task_only": SAMPLE_INDEX,
        "task": task,
        "device": DEVICE,
        "data_is_degrees": DATA_IS_DEGREES,
        "urdf_path": SO101_URDF_PATH,
        "model_input_state_used": input_state_used.tolist(),
        "chunk_shape": list(action_chunk_raw.shape),
        "joint_order_used_for_urdf": JOINT_ORDER,
        "send_to_real_robot": SEND_TO_REAL_ROBOT,
        "follower_port": FOLLOWER_PORT if SEND_TO_REAL_ROBOT else None,
        "follower_id": FOLLOWER_ID if SEND_TO_REAL_ROBOT else None,
        "state_source": "real robot via robot.get_observation() -> build_inference_frame(...)",
        "image_source": "real RealSense camera1 via robot.get_observation() -> build_inference_frame(...)",
        "camera2_3_source": "zeros",
    }
    (OUT_DIR / "debug_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def init_rerun_and_load_urdf(urdf_path: str):
    rr.init("smolvla_live_chunk_rerun_camera1_only", spawn=True)

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


def replay_action_chunk_with_urdf(robot_urdf, init_state_raw: np.ndarray, action_chunk_raw: np.ndarray) -> None:
    print("[INFO] 开始在 rerun 中回放 action chunk ...")
    print(f"[INFO] chunk shape = {action_chunk_raw.shape}")

    # 先把输入 state 作为 step=-1 的初始姿态打到 URDF
    init_state_rad = maybe_deg_to_rad(init_state_raw[:5])

    rr.set_time("step", sequence=-1)
    init_joint_positions_rad = {
        "shoulder_pan": float(init_state_rad[0]),
        "shoulder_lift": float(init_state_rad[1]),
        "elbow_flex": float(init_state_rad[2]),
        "wrist_flex": float(init_state_rad[3]),
        "wrist_roll": float(init_state_rad[4]),
    }
    log_joint_positions_to_rerun(robot_urdf, init_joint_positions_rad)

    rr.log("debug/input_state_raw/shoulder_pan", rr.Scalars(float(init_state_raw[0])))
    rr.log("debug/input_state_raw/shoulder_lift", rr.Scalars(float(init_state_raw[1])))
    rr.log("debug/input_state_raw/elbow_flex", rr.Scalars(float(init_state_raw[2])))
    rr.log("debug/input_state_raw/wrist_flex", rr.Scalars(float(init_state_raw[3])))
    rr.log("debug/input_state_raw/wrist_roll", rr.Scalars(float(init_state_raw[4])))
    rr.log("debug/input_state_raw/gripper", rr.Scalars(float(init_state_raw[5])))

    for step_idx, action_raw in enumerate(action_chunk_raw):
        rr.set_time("step", sequence=step_idx)

        action_rad = maybe_deg_to_rad(action_raw)

        joint_positions_rad = {
            "shoulder_pan": float(action_rad[0]),
            "shoulder_lift": float(action_rad[1]),
            "elbow_flex": float(action_rad[2]),
            "wrist_flex": float(action_rad[3]),
            "wrist_roll": float(action_rad[4]),
        }

        log_joint_positions_to_rerun(robot_urdf, joint_positions_rad)

        rr.log("debug/action_raw/shoulder_pan", rr.Scalars(float(action_raw[0])))
        rr.log("debug/action_raw/shoulder_lift", rr.Scalars(float(action_raw[1])))
        rr.log("debug/action_raw/elbow_flex", rr.Scalars(float(action_raw[2])))
        rr.log("debug/action_raw/wrist_flex", rr.Scalars(float(action_raw[3])))
        rr.log("debug/action_raw/wrist_roll", rr.Scalars(float(action_raw[4])))
        if len(action_raw) > 5:
            rr.log("debug/action_raw/gripper", rr.Scalars(float(action_raw[5])))

        time.sleep(SLEEP_PER_STEP_SEC)

    print("[INFO] rerun 回放完成。")


def connect_real_robot():
    robot_cfg = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
        cameras=CAMERA_CONFIG,
        use_degrees=False,
    )
    robot = SO101Follower(robot_cfg)
    robot.connect()
    return robot


def send_action_chunk_to_real_robot(robot, dataset_features, action_chunk_tensor: torch.Tensor) -> None:
    if action_chunk_tensor.ndim != 2:
        raise ValueError(f"Expected [T, D], got shape {tuple(action_chunk_tensor.shape)}")

    total_steps = action_chunk_tensor.shape[0]
    if LIMIT_STEPS is not None:
        total_steps = min(total_steps, LIMIT_STEPS)

    print(f"[INFO] 准备向真实机械臂发送 {total_steps} 步 action")

    for step_idx in range(total_steps):
        action_t = action_chunk_tensor[step_idx].unsqueeze(0)
        robot_action = make_robot_action(action_t, dataset_features)

        print(f"[REAL] step={step_idx:03d} action={action_t.squeeze(0).numpy()}")
        #robot.send_action(robot_action)
        time.sleep(SLEEP_PER_STEP_SEC)

    print("[INFO] 真实机械臂 action chunk 发送完成")


def main():
    print(f"[INFO] DEVICE                = {DEVICE}")
    print(f"[INFO] MODEL_ID              = {MODEL_ID}")
    print(f"[INFO] DATASET_ID            = {DATASET_ID}")
    print(f"[INFO] SAMPLE_INDEX          = {SAMPLE_INDEX}")
    print(f"[INFO] SEND_TO_REAL          = {SEND_TO_REAL_ROBOT}")
    print(f"[INFO] REPLAY_ON_RERUN       = {REPLAY_ON_RERUN}")
    print(f"[INFO] FOLLOWER_PORT         = {FOLLOWER_PORT}")
    print(f"[INFO] FOLLOWER_ID           = {FOLLOWER_ID}")
    print(f"[INFO] ROBOT_TYPE            = {ROBOT_TYPE}")
    print(f"[INFO] DATA_IS_DEGREES       = {DATA_IS_DEGREES}")
    print("[INFO] image mode            = real camera1 only, camera2/3 zeros")

    robot = None
    device = torch.device(DEVICE)

    try:
        model = SmolVLAPolicy.from_pretrained(MODEL_ID)
        model.eval()
        print("[INFO] model expected image keys:", list(model.config.image_features.keys()))

        # 数据集现在只用来提供 task / stats
        dataset = LeRobotDataset(DATASET_ID)
        sample = dataset[SAMPLE_INDEX]

        task = sample.get("task", None)
        if task is None or (isinstance(task, str) and len(task.strip()) == 0):
            task = FALLBACK_TASK
        task = FALLBACK_TASK
        print("[INFO] task:", task)
        print("[INFO] sample keys:", list(sample.keys()))
        print("[INFO] dataset only used for stats/task, not images")

        preprocess, postprocess = make_pre_post_processors(
            model.config,
            dataset_stats=dataset.meta.stats,
        )

        # 必须连接真机，因为 state 和 image 都来自真机
        robot = connect_real_robot()
        dataset_features_for_robot = build_robot_dataset_features(robot)

        print("[INFO] 已连接真实机械臂，开始读取当前 state + RealSense 图像")
        obs_frame = build_live_obs_frame_camera1_only(
            robot=robot,
            dataset_features=dataset_features_for_robot,
            task=task,
            device=device,
        )

        print("[INFO] model input state:", obs_frame["observation.state"])

        with torch.no_grad():
            obs = preprocess(obs_frame)
            action_chunk = model.predict_action_chunk(obs)
            action_chunk = postprocess(action_chunk)

        if hasattr(action_chunk, "actions"):
            action_chunk = action_chunk.actions

        if not isinstance(action_chunk, torch.Tensor):
            raise TypeError(f"Unexpected action_chunk type: {type(action_chunk)}")

        if action_chunk.ndim != 3:
            raise ValueError(f"Expected action_chunk shape [B, T, D], got {tuple(action_chunk.shape)}")

        action_chunk = action_chunk.detach().cpu()
        action_chunk_2d = action_chunk[0]
        action_chunk_raw = action_chunk_2d.numpy()

        print("[INFO] predicted action_chunk shape:", action_chunk_raw.shape)
        print("[INFO] first action:", action_chunk_raw[0])
        print("[INFO] last  action:", action_chunk_raw[-1])
        print("[INFO] full predicted action chunk:")
        for step_idx, action in enumerate(action_chunk_raw):
            print(f"[CHUNK] step={step_idx:02d} action={np.round(action, 3)}")

        save_debug_artifacts(
            task=task,
            input_state_used=flatten_state_tensor(obs_frame["observation.state"]),
            action_chunk_raw=action_chunk_raw,
        )

        if REPLAY_ON_RERUN:
            robot_urdf = init_rerun_and_load_urdf(SO101_URDF_PATH)
            replay_action_chunk_with_urdf(
                robot_urdf=robot_urdf,
                init_state_raw=flatten_state_tensor(obs_frame["observation.state"]),
                action_chunk_raw=action_chunk_raw,
            )

        if SEND_TO_REAL_ROBOT:
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