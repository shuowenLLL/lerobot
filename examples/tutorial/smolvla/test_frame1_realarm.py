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
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


# =========================
# 配置区
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 使用 jhou 的 model card
# 如果你想用本地备份，可以改成：
# MODEL_ID = "/home/shuowen/Repos/lerobot/models/smolvla_pickplace"
MODEL_ID = "jhou/smolvla_pickplace"

DATASET_ID = "lerobot/svla_so101_pickplace"
SAMPLE_INDEX = 150

FALLBACK_TASK = "put the pink lego brick into the transparent box"

SO101_URDF_PATH = "/home/shuowen/Repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

# ===== 选择只使用哪一个 frame/view =====
# 可选：
#   "up"   : 只用 observation.images.up，side 补零
#   "side" : 只用 observation.images.side，up 补零
SELECTED_FRAME_VIEW = "up"

# ===== 真机控制开关 =====
SEND_TO_REAL_ROBOT = False
REPLAY_ON_RERUN = True
LIMIT_STEPS = None
SLEEP_PER_STEP_SEC = 0.08

# ===== 真机端口与ID =====
FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "my_follower_arm"
ROBOT_TYPE = "so101_follower"

# ===== 单位解释 =====
DATA_IS_DEGREES = True

N_VIEW_STEPS = 50
DATASET_FPS = 30
ACTION_OFFSET_STEPS = 0

OUT_DIR = Path("outputs/test_frame1_realarm")
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


def format_values(values: np.ndarray | torch.Tensor, ndigits: int = 3) -> str:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()

    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return ", ".join(f"{v:.{ndigits}f}" for v in values)


def build_action_delta_timestamps(
    n_steps: int,
    fps: int | float,
    offset_steps: int = 0,
) -> dict[str, list[float]]:
    return {
        "action": [(i + offset_steps) / float(fps) for i in range(n_steps)]
    }


def get_action_pad_mask(sample: dict, n_steps: int) -> np.ndarray:
    candidate_keys = [
        "action_is_pad",
        "action_is_padding",
        "action_pad_mask",
        "action_is_pad_mask",
        "actions_id_pad",
    ]

    for key in candidate_keys:
        if key in sample:
            mask = sample[key]
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            mask = np.asarray(mask).astype(bool).reshape(-1)

            if mask.shape[0] >= n_steps:
                mask = mask[:n_steps]
            else:
                padded = np.ones(n_steps, dtype=bool)
                padded[: mask.shape[0]] = mask
                mask = padded

            return ~mask

    return np.ones(n_steps, dtype=bool)


def extract_gt_action_chunk_from_sample(sample: dict, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    gt = sample["action"]

    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    gt = np.asarray(gt, dtype=np.float32)

    if gt.ndim == 1:
        raise ValueError(
            f'sample["action"] is a single action, shape={gt.shape}. ' \
            "Use delta_timestamps to load an action chunk."
        )

    if gt.ndim != 2:
        raise ValueError(f'Expected sample["action"] shape [T, D], got shape={gt.shape}')

    gt = gt[:n_steps]
    valid_mask = get_action_pad_mask(sample, gt.shape[0])

    return gt, valid_mask


def compute_action_chunk_metrics(
    pred_action_chunk_deg: np.ndarray,
    gt_action_chunk_deg: np.ndarray,
    valid_mask: np.ndarray,
) -> dict:
    pred = np.asarray(pred_action_chunk_deg, dtype=np.float32)
    gt = np.asarray(gt_action_chunk_deg, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)

    t = min(pred.shape[0], gt.shape[0], valid_mask.shape[0])
    pred = pred[:t]
    gt = gt[:t]
    valid_mask = valid_mask[:t]

    if not np.any(valid_mask):
        return {
            "num_valid_steps": 0,
            "mae_overall": None,
            "rmse_overall": None,
            "mae_per_dim": None,
            "rmse_per_dim": None,
        }

    diff = pred[valid_mask] - gt[valid_mask]
    mae_per_dim = np.mean(np.abs(diff), axis=0)
    rmse_per_dim = np.sqrt(np.mean(diff ** 2, axis=0))

    return {
        "num_valid_steps": int(np.sum(valid_mask)),
        "mae_overall": float(np.mean(np.abs(diff))),
        "rmse_overall": float(np.sqrt(np.mean(diff ** 2))),
        "mae_per_dim": mae_per_dim.tolist(),
        "rmse_per_dim": rmse_per_dim.tolist(),
    }


def print_pred_vs_gt(
    pred_action_chunk_deg: np.ndarray,
    gt_action_chunk_deg: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    pred = np.asarray(pred_action_chunk_deg, dtype=np.float32)
    gt = np.asarray(gt_action_chunk_deg, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)

    t = min(pred.shape[0], gt.shape[0], valid_mask.shape[0])

    print("\n[PRED_VS_GT]")
    for i in range(t):
        pred_row = pred[i]
        gt_row = gt[i]
        diff_row = pred_row - gt_row
        print(
            "[PRED_VS_GT] "
            f"chunk_step={i:03d}, "
            f"valid_gt={bool(valid_mask[i])}, "
            f"pred=[{format_values(pred_row)}], "
            f"gt=[{format_values(gt_row)}], "
            f"diff=[{format_values(diff_row)}]"
        )


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


def make_single_view_images_from_sample(sample: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    jhou/smolvla_pickplace 期望两个 image keys：
      - observation.images.up
      - observation.images.side

    这里通过 SELECTED_FRAME_VIEW 控制只保留一个 view，另一个 view 补零。
    """
    img_up = sample["observation.images.up"].detach().cpu()
    img_side = sample["observation.images.side"].detach().cpu()

    if SELECTED_FRAME_VIEW == "up":
        selected_up = img_up
        selected_side = torch.zeros_like(img_side)
        print("[INFO] selected single frame/view: up")
        print("[INFO] image input: observation.images.up=real, observation.images.side=zeros")

    elif SELECTED_FRAME_VIEW == "side":
        selected_up = torch.zeros_like(img_up)
        selected_side = img_side
        print("[INFO] selected single frame/view: side")
        print("[INFO] image input: observation.images.up=zeros, observation.images.side=real")

    else:
        raise ValueError(
            f"Unknown SELECTED_FRAME_VIEW={SELECTED_FRAME_VIEW}. "
            "Use 'up' or 'side'."
        )

    return selected_up, selected_side


def build_offline_obs_frame_camera1_only(sample: dict, task: str) -> dict:
    """
    完全离线版本：
    - state: dataset sample
    - image: 从 up / side 中只选择一个 view
    - 另一个 view 补零

    注意：
    这里虽然函数名还叫 camera1_only，但为了适配 jhou/smolvla_pickplace，
    实际返回的是 observation.images.up / observation.images.side。
    """
    selected_up, selected_side = make_single_view_images_from_sample(sample)

    return {
        "observation.images.up": selected_up,
        "observation.images.side": selected_side,
        "observation.state": sample["observation.state"].detach().cpu().float(),
        "task": task,
    }


def tensor_image_to_numpy(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu()

    if img.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got shape={tuple(img.shape)}")

    if img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)

    img_np = img.numpy()

    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = img_np * 255.0
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    return img_np


def save_input_frame_images(sample: dict, sample_index: int) -> None:
    from PIL import Image

    img_up = tensor_image_to_numpy(sample["observation.images.up"])
    img_side = tensor_image_to_numpy(sample["observation.images.side"])

    up_path = OUT_DIR / f"sample_{sample_index}_OBS_IMAGE_up.png"
    side_path = OUT_DIR / f"sample_{sample_index}_OBS_IMAGE_side.png"

    Image.fromarray(img_up).save(up_path)
    Image.fromarray(img_side).save(side_path)

    for key in (
        "observation.images.wrist",
        "observation.images.wrist_image",
        "observation.images.wrist_left",
        "observation.images.wrist_right",
    ):
        if key in sample:
            img_wrist = tensor_image_to_numpy(sample[key])
            wrist_path = OUT_DIR / f"sample_{sample_index}_OBS_IMAGE_wrist.png"
            Image.fromarray(img_wrist).save(wrist_path)
            break


def build_robot_dataset_features(robot):
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    return dataset_features


def build_mixed_obs_frame_camera1_only(
    sample: dict,
    robot,
    dataset_features: dict,
    task: str,
    device: torch.device,
) -> dict:
    """
    混合输入版本：
    - state: 真机 raw_obs -> build_inference_frame(...)
    - image: 从 dataset sample 的 up / side 中只选择一个 view
    - 另一个 view 补零

    注意：
    jhou/smolvla_pickplace 期望 observation.images.up / observation.images.side，
    所以这里不再返回 camera1/camera2/camera3。
    """
    raw_obs = robot.get_observation()
    print("[INFO] robot observation keys:", list(raw_obs.keys()))

    for k, v in raw_obs.items():
        if isinstance(v, torch.Tensor):
            print(f"[OBS] {k}: shape={tuple(v.shape)}, dtype={v.dtype}, value={v.detach().cpu()}")
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

    real_state = robot_obs_frame["observation.state"].detach().cpu().float()
    print("[INFO] state from build_inference_frame:", real_state)

    selected_up, selected_side = make_single_view_images_from_sample(sample)

    mixed_obs_frame = {
        "observation.images.up": selected_up,
        "observation.images.side": selected_side,
        "observation.state": real_state,
        "task": task,
    }

    return mixed_obs_frame


def make_processors_for_model_or_dataset(
    model,
    dataset,
    device: torch.device,
):
    """
    优先尝试从 MODEL_ID 加载 pre/post processor。
    jhou/smolvla_pickplace 当前没有 policy_preprocessor.json 时，会 fallback 到 dataset.meta.stats。
    """
    try:
        print(f"[INFO] trying to load pre/post processors from model repo/path: {MODEL_ID}")

        preprocess, postprocess = make_pre_post_processors(
            model.config,
            MODEL_ID,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

        print("[INFO] loaded processors from model repo/path.")
        return preprocess, postprocess

    except Exception as e:
        print("[WARN] failed to load processors from model repo/path.")
        print(f"[WARN] reason: {repr(e)}")
        print("[WARN] falling back to model.config + dataset.meta.stats")

        preprocess, postprocess = make_pre_post_processors(
            model.config,
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

        return preprocess, postprocess


def save_debug_artifacts(sample: dict, task: str, input_state_used, action_chunk_deg: np.ndarray) -> None:
    dataset_state_deg = flatten_state_tensor(sample["observation.state"])
    input_state_used = np.asarray(input_state_used, dtype=np.float32).flatten()

    np.save(OUT_DIR / "dataset_input_state_deg.npy", dataset_state_deg)
    np.save(OUT_DIR / "model_input_state_used.npy", input_state_used)
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
        "dataset_input_state_deg": dataset_state_deg.tolist(),
        "model_input_state_used": input_state_used.tolist(),
        "chunk_shape": list(action_chunk_deg.shape),
        "joint_order_used_for_urdf": JOINT_ORDER,
        "send_to_real_robot": SEND_TO_REAL_ROBOT,
        "follower_port": FOLLOWER_PORT if SEND_TO_REAL_ROBOT else None,
        "follower_id": FOLLOWER_ID if SEND_TO_REAL_ROBOT else None,
        "state_source": (
            "robot.get_observation() -> build_inference_frame(...)"
            if SEND_TO_REAL_ROBOT
            else "dataset sample observation.state"
        ),
        "selected_frame_view": SELECTED_FRAME_VIEW,
        "image_keys_used": {
            "observation.images.up": "real" if SELECTED_FRAME_VIEW == "up" else "zeros",
            "observation.images.side": "real" if SELECTED_FRAME_VIEW == "side" else "zeros",
        },
    }

    (OUT_DIR / "debug_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def init_rerun_and_load_urdf(urdf_path: str):
    rr.init("jhou_smolvla_single_frame_rerun", spawn=True)

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

        # 每个 joint 写到不同 path，避免互相覆盖
        rr.log(f"transforms/{urdf_joint_name}", transform)


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
        if len(action_deg) > 5:
            rr.log("debug/action_deg/gripper", rr.Scalars(float(action_deg[5])))

        time.sleep(SLEEP_PER_STEP_SEC)

    print("[INFO] rerun 回放完成。")


def connect_real_robot():
    robot_cfg = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
        use_degrees=True,
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

        # 真机安全起见，默认先不发送。确认后再取消注释。
        # robot.send_action(robot_action)

        time.sleep(SLEEP_PER_STEP_SEC)

    print("[INFO] 真实机械臂 action chunk 发送完成")


def main():
    print(f"[INFO] DEVICE                = {DEVICE}")
    print(f"[INFO] MODEL_ID              = {MODEL_ID}")
    print(f"[INFO] DATASET_ID            = {DATASET_ID}")
    print(f"[INFO] SAMPLE_INDEX          = {SAMPLE_INDEX}")
    print(f"[INFO] SELECTED_FRAME_VIEW   = {SELECTED_FRAME_VIEW}")
    print(f"[INFO] SEND_TO_REAL          = {SEND_TO_REAL_ROBOT}")
    print(f"[INFO] REPLAY_ON_RERUN       = {REPLAY_ON_RERUN}")
    print(f"[INFO] FOLLOWER_PORT         = {FOLLOWER_PORT}")
    print(f"[INFO] FOLLOWER_ID           = {FOLLOWER_ID}")
    print(f"[INFO] ROBOT_TYPE            = {ROBOT_TYPE}")
    print(f"[INFO] DATA_IS_DEGREES       = {DATA_IS_DEGREES}")
    print("[INFO] image mode            = single view only; the other expected view is zeros")

    robot = None
    device = torch.device(DEVICE)

    try:
        # 1) 加载模型
        model = SmolVLAPolicy.from_pretrained(MODEL_ID).to(device)
        model.eval()

        print("[INFO] model expected image keys:", list(model.config.image_features.keys()))
        print("[INFO] model input features:", model.config.input_features)
        print("[INFO] model output features:", model.config.output_features)

        # 2) 加载 dataset sample，并读取 action chunk GT
        action_delta_timestamps = build_action_delta_timestamps(
            n_steps=N_VIEW_STEPS,
            fps=DATASET_FPS,
            offset_steps=ACTION_OFFSET_STEPS,
        )
        dataset = LeRobotDataset(
            DATASET_ID,
            delta_timestamps=action_delta_timestamps,
        )
        sample = dataset[SAMPLE_INDEX]

        save_input_frame_images(sample, SAMPLE_INDEX)

        # 3) task
        # 如果你想用 dataset 自带 task，可以改回下面这段：
        # task = sample.get("task", None)
        # if task is None or (isinstance(task, str) and len(task.strip()) == 0):
        #     task = FALLBACK_TASK
        task = FALLBACK_TASK

        print("[INFO] task:", task)
        print("[INFO] sample keys:", list(sample.keys()))
        print("[INFO] dataset observation.state:", sample["observation.state"])

        # 4) pre/post processor
        preprocess, postprocess = make_processors_for_model_or_dataset(
            model=model,
            dataset=dataset,
            device=device,
        )

        # 5) 构造输入
        if SEND_TO_REAL_ROBOT:
            robot = connect_real_robot()
            dataset_features_for_robot = build_robot_dataset_features(robot)

            print("[INFO] 已连接真实机械臂，开始读取当前 state（通过 build_inference_frame 处理）")
            obs_frame = build_mixed_obs_frame_camera1_only(
                sample=sample,
                robot=robot,
                dataset_features=dataset_features_for_robot,
                task=task,
                device=device,
            )
        else:
            print("[INFO] SEND_TO_REAL_ROBOT=False，使用纯离线 sample state")
            obs_frame = build_offline_obs_frame_camera1_only(sample, task)
            dataset_features_for_robot = None

        print("[INFO] model input state:", obs_frame["observation.state"])
        print("[INFO] obs_frame keys:", list(obs_frame.keys()))

        # 6) 推理
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
        action_chunk_deg = action_chunk_2d.numpy()

        # 7) GT 对比
        gt_action_chunk_deg, valid_gt_mask = extract_gt_action_chunk_from_sample(
            sample=sample,
            n_steps=action_chunk_deg.shape[0],
        )

        t = min(action_chunk_deg.shape[0], gt_action_chunk_deg.shape[0], valid_gt_mask.shape[0])
        action_chunk_deg = action_chunk_deg[:t]
        gt_action_chunk_deg = gt_action_chunk_deg[:t]
        valid_gt_mask = valid_gt_mask[:t]

        metrics = compute_action_chunk_metrics(
            pred_action_chunk_deg=action_chunk_deg,
            gt_action_chunk_deg=gt_action_chunk_deg,
            valid_mask=valid_gt_mask,
        )

        print("[INFO] predicted action_chunk shape:", action_chunk_deg.shape)
        print("[INFO] first action:", action_chunk_deg[0])
        print("[INFO] last  action:", action_chunk_deg[-1])
        print("[INFO] full predicted action chunk:")

        for step_idx, action in enumerate(action_chunk_deg):
            print(f"[CHUNK] step={step_idx:02d} action={np.round(action, 3)}")

        print("\n[METRICS]")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

        print_pred_vs_gt(
            pred_action_chunk_deg=action_chunk_deg,
            gt_action_chunk_deg=gt_action_chunk_deg,
            valid_mask=valid_gt_mask,
        )

        # 8) 保存调试文件
        save_debug_artifacts(
            sample=sample,
            task=task,
            input_state_used=flatten_state_tensor(obs_frame["observation.state"]),
            action_chunk_deg=action_chunk_deg,
        )

        # 8) Rerun 回放
        if REPLAY_ON_RERUN:
            robot_urdf = init_rerun_and_load_urdf(SO101_URDF_PATH)

            init_state_deg = flatten_state_tensor(obs_frame["observation.state"])
            rr.set_time("step", sequence=-1)
            rr.log("debug/input_state_deg/shoulder_pan", rr.Scalars(float(init_state_deg[0])))
            rr.log("debug/input_state_deg/shoulder_lift", rr.Scalars(float(init_state_deg[1])))
            rr.log("debug/input_state_deg/elbow_flex", rr.Scalars(float(init_state_deg[2])))
            rr.log("debug/input_state_deg/wrist_flex", rr.Scalars(float(init_state_deg[3])))
            rr.log("debug/input_state_deg/wrist_roll", rr.Scalars(float(init_state_deg[4])))
            rr.log("debug/input_state_deg/gripper", rr.Scalars(float(init_state_deg[5])))

            replay_action_chunk_with_urdf(robot_urdf, action_chunk_deg)

        # 9) 可选真机发送
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