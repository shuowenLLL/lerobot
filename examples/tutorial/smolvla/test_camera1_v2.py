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
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


MAX_EPISODES = 3
MAX_STEPS_PER_EPISODE = 30
SLEEP_PER_STEP_SEC = 0.08

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 使用 jhou 的 model card
# 如果要用本地备份，可以改成：
# MODEL_ID = "/home/shuowen/Repos/lerobot/models/smolvla_pickplace"
MODEL_ID = "jhou/smolvla_pickplace"

DATASET_ID = "lerobot/svla_so101_pickplace"

FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "my_follower_arm"
ROBOT_TYPE = "so101_follower"

# TASK = "pick up the pink lego brick"
TASK = "put the pink lego brick into the transparent box"

SO101_URDF_PATH = "/home/shuowen/Repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

# 你当前日志里 state / action 看起来是“度”风格，所以 rerun 前转弧度
DATA_IS_DEGREES = True

# ===== 真机单相机映射到 jhou/smolvla_pickplace 的哪个 view =====
# 可选：
#   "up"   : camera1 -> observation.images.up, side 由 policy.empty_cameras mask 掉
#   "side" : camera1 -> observation.images.side, up 由 policy.empty_cameras mask 掉
REAL_CAMERA_AS_VIEW = "up"
EMPTY_CAMERAS = 1

# Rerun 折线图显示开关
LOG_INPUT_STATE = False
LOG_RAW_ACTION = False
LOG_POST_ACTION = True

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

OUT_DIR = Path("outputs/jhou_pickplace_with_rerun")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RRD_PATH = OUT_DIR / "test_camera1_v2_9test_camera_v2_2.rrd"
RERUN_PORT = 9876


def maybe_deg_to_rad(x: np.ndarray) -> np.ndarray:
    if DATA_IS_DEGREES:
        return np.deg2rad(x)
    return x


def flatten_tensor(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().flatten().numpy()
    return np.asarray(x).flatten()


def image_to_uint8_hwc(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    else:
        image = np.asarray(image)

    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Expected a single image batch, got shape={image.shape}")
        image = image[0]

    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape={image.shape}")

    # Convert CHW to HWC if needed.
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)

    if image.shape[-1] == 1:
        image = image[..., 0]

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.size > 0 and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


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
    rr.init("jhou_smolvla_post_action_rerun")
    rr.spawn(port=RERUN_PORT, connect=False)
    rr.set_sinks(
        rr.GrpcSink(f"rerun+http://127.0.0.1:{RERUN_PORT}/proxy"),
        rr.FileSink(RRD_PATH),
    )
    print(f"[INFO] Rerun Viewer spawned on port {RERUN_PORT}")
    print(f"[INFO] Rerun recording will be saved to: {RRD_PATH}")

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


def make_camera1_rename_map(real_camera_as_view: str) -> dict[str, str]:
    """
    build_inference_frame(...) 根据 CAMERA_CONFIG 生成：
        observation.images.camera1

    按 LeRobot rename_map 约定，把 source key 映射到 policy 期望的 key。
    缺失的另一个 view 不再手动补零图，而是交给 policy.empty_cameras。
    """
    if real_camera_as_view not in {"up", "side"}:
        raise ValueError(
            f"Unknown REAL_CAMERA_AS_VIEW={real_camera_as_view}. "
            "Use 'up' or 'side'."
        )

    return {"observation.images.camera1": f"observation.images.{real_camera_as_view}"}


def set_preprocessor_rename_map(preprocessor, rename_map: dict[str, str]) -> None:
    for step in preprocessor.steps:
        if step.__class__.__name__ == "RenameObservationsProcessorStep":
            step.rename_map = rename_map
            return

    raise RuntimeError("preprocessor 里没有 RenameObservationsProcessorStep，无法应用 rename_map。")


def main():
    print(f"[INFO] DEVICE       = {DEVICE}")
    print(f"[INFO] MODEL_ID     = {MODEL_ID}")
    print(f"[INFO] DATASET_ID   = {DATASET_ID}")
    print(f"[INFO] FOLLOWER_PORT= {FOLLOWER_PORT}")
    print(f"[INFO] FOLLOWER_ID  = {FOLLOWER_ID}")
    print(f"[INFO] ROBOT_TYPE   = {ROBOT_TYPE}")
    print(f"[INFO] TASK         = {TASK}")
    print(f"[INFO] REAL_CAMERA_AS_VIEW = {REAL_CAMERA_AS_VIEW}")
    print(f"[INFO] EMPTY_CAMERAS = {EMPTY_CAMERAS}")
    print(f"[INFO] DATA_IS_DEGREES = {DATA_IS_DEGREES}")
    print(f"[INFO] LOG_INPUT_STATE = {LOG_INPUT_STATE}")
    print(f"[INFO] LOG_RAW_ACTION  = {LOG_RAW_ACTION}")
    print(f"[INFO] LOG_POST_ACTION = {LOG_POST_ACTION}")

    device = torch.device(DEVICE)
    rename_map = make_camera1_rename_map(REAL_CAMERA_AS_VIEW)

    # 1) 模型
    config = PreTrainedConfig.from_pretrained(MODEL_ID)
    config.empty_cameras = EMPTY_CAMERAS
    config.device = str(device)

    model = SmolVLAPolicy.from_pretrained(MODEL_ID, config=config).to(device)
    model.eval()

    print("[INFO] rename_map:", rename_map)
    print("[INFO] model expected image keys:", list(model.config.image_features.keys()))
    print("[INFO] model input features:", model.config.input_features)
    print("[INFO] model output features:", model.config.output_features)

    # 2) 数据集统计量：用于 action/state 的归一化和反归一化
    # jhou/smolvla_pickplace 没有 policy_preprocessor.json，
    # 所以这里直接用 dataset.meta.stats，避免 warning。
    dataset = LeRobotDataset(DATASET_ID)

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    set_preprocessor_rename_map(preprocess, rename_map)

    # 3) 机器人
    robot_cfg = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
        cameras=CAMERA_CONFIG,
        use_degrees=False,
    )
    robot = SO101Follower(robot_cfg)
    robot.connect()

    # 4) 用来匹配 raw observation / action 到 policy 格式
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # 5) rerun
    robot_urdf = init_rerun_and_load_urdf(SO101_URDF_PATH)

    step_counter = 0

    try:
        for episode_idx in range(MAX_EPISODES):
            print(f"[INFO] Episode {episode_idx + 1}/{MAX_EPISODES}")

            for local_step in range(MAX_STEPS_PER_EPISODE):
                raw_obs = robot.get_observation()

                raw_obs_frame = build_inference_frame(
                    observation=raw_obs,
                    ds_features=dataset_features,
                    device=device,
                    task=TASK,
                    robot_type=ROBOT_TYPE,
                )

                if "observation.images.camera1" not in raw_obs_frame:
                    raise KeyError(
                        "build_inference_frame(...) 没有生成 'observation.images.camera1'。"
                        f"当前 raw_obs_frame keys={list(raw_obs_frame.keys())}"
                    )
                if "observation.state" not in raw_obs_frame:
                    raise KeyError(
                        "build_inference_frame(...) 没有生成 'observation.state'。"
                        f"当前 raw_obs_frame keys={list(raw_obs_frame.keys())}"
                    )

                obs_frame = raw_obs_frame

                if step_counter == 0:
                    print("[INFO] raw_obs_frame keys:", list(raw_obs_frame.keys()))
                    print("[INFO] preprocessor rename_map:", rename_map)
                    print(
                        "[INFO] image mapping:",
                        f"camera1 -> observation.images.{REAL_CAMERA_AS_VIEW}, "
                        "other view -> policy.empty_cameras",
                    )

                # 记录当前输入 state，但默认不在 Rerun 折线图显示
                state_raw = flatten_tensor(obs_frame["observation.state"])

                rr.set_time("global_step", sequence=step_counter)
                rr.log(
                    "camera/camera1",
                    rr.Image(image_to_uint8_hwc(raw_obs_frame["observation.images.camera1"])),
                )

                if LOG_INPUT_STATE:
                    rr.log("debug/input_state/shoulder_pan", rr.Scalars(float(state_raw[0])))
                    rr.log("debug/input_state/shoulder_lift", rr.Scalars(float(state_raw[1])))
                    rr.log("debug/input_state/elbow_flex", rr.Scalars(float(state_raw[2])))
                    rr.log("debug/input_state/wrist_flex", rr.Scalars(float(state_raw[3])))
                    rr.log("debug/input_state/wrist_roll", rr.Scalars(float(state_raw[4])))
                    if len(state_raw) > 5:
                        rr.log("debug/input_state/gripper", rr.Scalars(float(state_raw[5])))

                # 初始姿态同步到 URDF，便于对照当前观测
                state_rad = maybe_deg_to_rad(state_raw[:5])
                state_joint_positions_rad = {
                    "shoulder_pan": float(state_rad[0]),
                    "shoulder_lift": float(state_rad[1]),
                    "elbow_flex": float(state_rad[2]),
                    "wrist_flex": float(state_rad[3]),
                    "wrist_roll": float(state_rad[4]),
                }
                log_joint_positions_to_rerun(robot_urdf, state_joint_positions_rad)

                # preprocess: 进入模型前的处理
                obs = preprocess(obs_frame)

                # 官方结构：单步 select_action()
                raw_action = model.select_action(obs)

                # ---------- postprocess 前 raw_action ----------
                if hasattr(raw_action, "actions"):
                    raw_action_tensor = raw_action.actions
                else:
                    raw_action_tensor = raw_action

                if not isinstance(raw_action_tensor, torch.Tensor):
                    raise TypeError(f"Unexpected raw_action type: {type(raw_action_tensor)}")

                raw_action_np = flatten_tensor(raw_action_tensor)


                if LOG_RAW_ACTION:
                    rr.log("debug/raw_action/shoulder_pan", rr.Scalars(float(raw_action_np[0])))
                    rr.log("debug/raw_action/shoulder_lift", rr.Scalars(float(raw_action_np[1])))
                    rr.log("debug/raw_action/elbow_flex", rr.Scalars(float(raw_action_np[2])))
                    rr.log("debug/raw_action/wrist_flex", rr.Scalars(float(raw_action_np[3])))
                    rr.log("debug/raw_action/wrist_roll", rr.Scalars(float(raw_action_np[4])))
                    if len(raw_action_np) > 5:
                        rr.log("debug/raw_action/gripper", rr.Scalars(float(raw_action_np[5])))

                # ---------- postprocess 后 action ----------
                post_action = postprocess(raw_action)

                if hasattr(post_action, "actions"):
                    post_action_tensor = post_action.actions
                else:
                    post_action_tensor = post_action

                if not isinstance(post_action_tensor, torch.Tensor):
                    raise TypeError(f"Unexpected post_action type: {type(post_action_tensor)}")

                post_action_np = flatten_tensor(post_action_tensor)

                # 只显示 postprocess 后的折线图
                if LOG_POST_ACTION:
                    rr.log("debug/output_action/shoulder_pan", rr.Scalars(float(post_action_np[0])))
                    rr.log("debug/output_action/shoulder_lift", rr.Scalars(float(post_action_np[1])))
                    rr.log("debug/output_action/elbow_flex", rr.Scalars(float(post_action_np[2])))
                    rr.log("debug/output_action/wrist_flex", rr.Scalars(float(post_action_np[3])))
                    rr.log("debug/output_action/wrist_roll", rr.Scalars(float(post_action_np[4])))
                    if len(post_action_np) > 5:
                        rr.log("debug/output_action/gripper", rr.Scalars(float(post_action_np[5])))

                # 用 postprocess 后的目标动作驱动 URDF
                action_rad = maybe_deg_to_rad(post_action_np[:5])
                action_joint_positions_rad = {
                    "shoulder_pan": float(action_rad[0]),
                    "shoulder_lift": float(action_rad[1]),
                    "elbow_flex": float(action_rad[2]),
                    "wrist_flex": float(action_rad[3]),
                    "wrist_roll": float(action_rad[4]),
                }
                log_joint_positions_to_rerun(robot_urdf, action_joint_positions_rad)

                # 转成真机 action dict
                robot_action = make_robot_action(post_action, dataset_features)
                print(f"[STEP {step_counter:03d}] ROBOT action = {robot_action}")

                # 真正发送给机械臂
                #robot.send_action(robot_action)

                step_counter += 1
                time.sleep(SLEEP_PER_STEP_SEC)

            print("Episode finished! Starting new episode...")

    finally:
        try:
            rr.disconnect()
            print(f"[INFO] saved rerun recording to: {RRD_PATH}")
        except Exception as e:
            print(f"[WARN] failed to disconnect rerun cleanly: {e}")
            print(f"[INFO] rerun recording path: {RRD_PATH}")

        robot.disconnect()
        print("[INFO] 机械臂已断开连接")


if __name__ == "__main__":
    main()
