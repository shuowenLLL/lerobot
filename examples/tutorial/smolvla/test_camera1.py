from __future__ import annotations

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
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


MAX_EPISODES = 1
MAX_STEPS_PER_EPISODE = 50
SLEEP_PER_STEP_SEC = 0.08

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "lerobot/smolvla_base"

FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "my_follower_arm"
ROBOT_TYPE = "so101_follower"

TASK = "pick up the pink lego brick"

SO101_URDF_PATH = "/home/shuowen/Repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

# 你当前日志里 state / action 看起来是“度”风格，所以 rerun 前转弧度
DATA_IS_DEGREES = True

# 真机相机：只用 camera1
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

OUT_DIR = Path("outputs/using_style_with_rerun")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def maybe_deg_to_rad(x: np.ndarray) -> np.ndarray:
    if DATA_IS_DEGREES:
        return np.deg2rad(x)
    return x


def flatten_tensor(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().flatten().numpy()
    return np.asarray(x).flatten()


def get_urdf_joint(robot_urdf, joint_name: str):
    joints_obj = robot_urdf.joints

    if isinstance(joints_obj, dict):
        return joints_obj[joint_name]

    if callable(joints_obj):
        joints_obj = joints_obj()

    try:
        for joint in joints_obj:
            if getattr(joint, "name", None) == joint_name:
                return joint
    except TypeError:
        pass

    raise KeyError(f"URDF joint '{joint_name}' not found")


def init_rerun_and_load_urdf(urdf_path: str):
    rr.init("smolvla_using_style_rerun", spawn=True)

    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF 不存在: {urdf_path}")

    rr.log_file_from_path(urdf_path, static=True)

    if rr_urdf is None:
        raise ImportError("无法 import rerun.urdf，请确认 rerun 版本支持 URDF。")

    if hasattr(rr_urdf, "UrdfTree"):
        return rr_urdf.UrdfTree.from_file_path(urdf_path)

    raise RuntimeError("当前 rerun.urdf 模块里没有 UrdfTree。")


def log_joint_positions_to_rerun(robot_urdf, joint_positions_rad: dict[str, float]) -> None:
    """
    注意：每个关节必须写到不同 path，不能都 rr.log('transforms', transform)
    否则会互相覆盖，表现成 URDF 卡住/不正常。
    """
    for logical_name, angle_rad in joint_positions_rad.items():
        urdf_joint_name = JOINT_MAP[logical_name]
        joint = get_urdf_joint(robot_urdf, urdf_joint_name)

        if not hasattr(joint, "compute_transform"):
            raise AttributeError(f"joint '{urdf_joint_name}' 没有 compute_transform(angle) 接口。")

        transform = joint.compute_transform(float(angle_rad))
        rr.log(f"transforms/{urdf_joint_name}", transform)


def main():
    print(f"[INFO] DEVICE       = {DEVICE}")
    print(f"[INFO] MODEL_ID     = {MODEL_ID}")
    print(f"[INFO] FOLLOWER_PORT= {FOLLOWER_PORT}")
    print(f"[INFO] FOLLOWER_ID  = {FOLLOWER_ID}")
    print(f"[INFO] ROBOT_TYPE   = {ROBOT_TYPE}")
    print(f"[INFO] TASK         = {TASK}")
    print(f"[INFO] DATA_IS_DEGREES = {DATA_IS_DEGREES}")

    device = torch.device(DEVICE)

    # 1) 模型
    model = SmolVLAPolicy.from_pretrained(MODEL_ID)

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # 2) 机器人
    robot_cfg = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
        cameras=CAMERA_CONFIG,
        use_degrees=False,   # 保持和你现在真机读取一致；不影响 rerun 里我们自己转弧度
    )
    robot = SO101Follower(robot_cfg)
    robot.connect()

    # 3) 用来匹配 raw observation / action 到 policy 格式
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # 4) rerun
    robot_urdf = init_rerun_and_load_urdf(SO101_URDF_PATH)

    step_counter = 0

    try:
        for episode_idx in range(MAX_EPISODES):
            print(f"[INFO] Episode {episode_idx + 1}/{MAX_EPISODES}")

            for local_step in range(MAX_STEPS_PER_EPISODE):
                raw_obs = robot.get_observation()

                obs_frame = build_inference_frame(
                    observation=raw_obs,
                    ds_features=dataset_features,
                    device=device,
                    task=TASK,
                    robot_type=ROBOT_TYPE,
                )

                # 记录当前输入 state
                state_raw = flatten_tensor(obs_frame["observation.state"])

                rr.set_time("global_step", sequence=step_counter)

                rr.log("debug/input_state/shoulder_pan", rr.Scalars(float(state_raw[0])))
                rr.log("debug/input_state/shoulder_lift", rr.Scalars(float(state_raw[1])))
                rr.log("debug/input_state/elbow_flex", rr.Scalars(float(state_raw[2])))
                rr.log("debug/input_state/wrist_flex", rr.Scalars(float(state_raw[3])))
                rr.log("debug/input_state/wrist_roll", rr.Scalars(float(state_raw[4])))
                if len(state_raw) > 5:
                    rr.log("debug/input_state/gripper", rr.Scalars(float(state_raw[5])))

                # 初始姿态也同步到 URDF，便于对照当前观测
                state_rad = maybe_deg_to_rad(state_raw[:5])
                state_joint_positions_rad = {
                    "shoulder_pan": float(state_rad[0]),
                    "shoulder_lift": float(state_rad[1]),
                    "elbow_flex": float(state_rad[2]),
                    "wrist_flex": float(state_rad[3]),
                    "wrist_roll": float(state_rad[4]),
                }
                log_joint_positions_to_rerun(robot_urdf, state_joint_positions_rad)

                obs = preprocess(obs_frame)

                # 官方结构：单步 select_action()
                raw_action = model.select_action(obs)

                # ---------- 先看 postprocess 前 ----------
                if hasattr(raw_action, "actions"):
                    raw_action_tensor = raw_action.actions
                else:
                    raw_action_tensor = raw_action

                if not isinstance(raw_action_tensor, torch.Tensor):
                    raise TypeError(f"Unexpected raw_action type: {type(raw_action_tensor)}")

                raw_action_np = flatten_tensor(raw_action_tensor)

                print(f"[STEP {step_counter:03d}] RAW  action = {raw_action_np}")
                print(
                    f"[STEP {step_counter:03d}] RAW  range  = "
                    f"min={raw_action_tensor.min().item():.3f}, max={raw_action_tensor.max().item():.3f}"
                )

                # 可选：把 RAW 也打到 rerun
                rr.log("debug/raw_action/shoulder_pan", rr.Scalars(float(raw_action_np[0])))
                rr.log("debug/raw_action/shoulder_lift", rr.Scalars(float(raw_action_np[1])))
                rr.log("debug/raw_action/elbow_flex", rr.Scalars(float(raw_action_np[2])))
                rr.log("debug/raw_action/wrist_flex", rr.Scalars(float(raw_action_np[3])))
                rr.log("debug/raw_action/wrist_roll", rr.Scalars(float(raw_action_np[4])))
                if len(raw_action_np) > 5:
                    rr.log("debug/raw_action/gripper", rr.Scalars(float(raw_action_np[5])))

                # ---------- 再做 postprocess ----------
                post_action = postprocess(raw_action)

                if hasattr(post_action, "actions"):
                    post_action_tensor = post_action.actions
                else:
                    post_action_tensor = post_action

                if not isinstance(post_action_tensor, torch.Tensor):
                    raise TypeError(f"Unexpected post_action type: {type(post_action_tensor)}")

                post_action_np = flatten_tensor(post_action_tensor)

                print(f"[STEP {step_counter:03d}] POST action = {post_action_np}")
                print(
                    f"[STEP {step_counter:03d}] POST range  = "
                    f"min={post_action_tensor.min().item():.3f}, max={post_action_tensor.max().item():.3f}"
                )

                # 记录当前输出 action（postprocess 后）
                rr.log("debug/output_action/shoulder_pan", rr.Scalars(float(post_action_np[0])))
                rr.log("debug/output_action/shoulder_lift", rr.Scalars(float(post_action_np[1])))
                rr.log("debug/output_action/elbow_flex", rr.Scalars(float(post_action_np[2])))
                rr.log("debug/output_action/wrist_flex", rr.Scalars(float(post_action_np[3])))
                rr.log("debug/output_action/wrist_roll", rr.Scalars(float(post_action_np[4])))
                if len(post_action_np) > 5:
                    rr.log("debug/output_action/gripper", rr.Scalars(float(post_action_np[5])))

                # 用“postprocess 后的目标动作”驱动 URDF
                action_rad = maybe_deg_to_rad(post_action_np[:5])
                action_joint_positions_rad = {
                    "shoulder_pan": float(action_rad[0]),
                    "shoulder_lift": float(action_rad[1]),
                    "elbow_flex": float(action_rad[2]),
                    "wrist_flex": float(action_rad[3]),
                    "wrist_roll": float(action_rad[4]),
                }
                log_joint_positions_to_rerun(robot_urdf, action_joint_positions_rad)

                robot_action = make_robot_action(post_action, dataset_features)
                print(f"[STEP {step_counter:03d}] ROBOT action = {robot_action}")

                print(
                    f"[STEP {step_counter:03d}] "
                    f"state={np.round(state_raw, 3)} "
                    f"raw_action={np.round(raw_action_np, 3)} "
                    f"post_action={np.round(post_action_np, 3)}"
                )

                # robot.send_action(robot_action)

                step_counter += 1
                time.sleep(SLEEP_PER_STEP_SEC)

            print("Episode finished! Starting new episode...")

    finally:
        robot.disconnect()
        print("[INFO] 机械臂已断开连接")


if __name__ == "__main__":
    main()