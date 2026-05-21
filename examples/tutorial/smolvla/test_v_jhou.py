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
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


# =========================
# 配置区
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 关键修改：用你要测试的 model card
MODEL_ID = "jhou/smolvla_pickplace"

# 用同一个 SO101 pickplace 数据集做离线测试
DATASET_ID = "lerobot/svla_so101_pickplace"
SAMPLE_INDEX = 100

FALLBACK_TASK = "put the pink lego brick into the transparent box"

SO101_URDF_PATH = "/home/shuowen/Repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

ACTION_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

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

# 你当前 Rerun/URDF 按 degree -> rad 处理
DATA_IS_DEGREES = True

SLEEP_PER_STEP_SEC = 0.08
N_VIEW_STEPS = 50
DATASET_FPS = 30

# GT action chunk 起点偏移
ACTION_OFFSET_STEPS = 0

# 推荐默认：SO101 数据集一般是 up/top + side 两路
# 可选：
# "top_side_empty"
# "side_top_empty"
# "top_empty_side"
# "side_empty_top"
# "top_wrist_side"
CAMERA_MAPPING_MODE = "top_side_empty"

SEED = 0

OUT_DIR = Path("outputs/offline_jhou_smolvla_pickplace")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 基础工具函数
# =========================
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_deg_to_rad(x: np.ndarray) -> np.ndarray:
    if DATA_IS_DEGREES:
        return np.deg2rad(x)
    return x


def format_values(values: np.ndarray | torch.Tensor, ndigits: int = 3) -> str:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()

    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return ", ".join(f"{v:.{ndigits}f}" for v in values)


def unwrap_policy_action(x):
    """
    兼容不同 LeRobot 版本：
    - 有些版本 postprocess 返回 PolicyAction，里面有 .actions
    - 有些版本直接返回 Tensor
    """
    if hasattr(x, "actions"):
        return x.actions
    return x


def build_action_delta_timestamps(
    n_steps: int,
    fps: int | float,
    offset_steps: int = 0,
) -> dict[str, list[float]]:
    return {
        "action": [(i + offset_steps) / float(fps) for i in range(n_steps)]
    }


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


# =========================
# Camera / obs 构造
# =========================
def get_wrist_image_or_empty(sample: dict, empty: torch.Tensor) -> torch.Tensor:
    wrist_keys = [
        "observation.images.wrist",
        "observation.images.wrist_image",
        "observation.images.wrist_left",
        "observation.images.wrist_right",
    ]

    for key in wrist_keys:
        if key in sample:
            return sample[key].detach().cpu()

    return empty


def build_offline_obs_frame(sample: dict, task: str) -> dict:
    """
    jhou/smolvla_pickplace 这个 checkpoint 期望的 image keys 是：
      - observation.images.up
      - observation.images.side

    所以这里不要再映射成 camera1/camera2/camera3。
    """
    img_top = sample["observation.images.up"].detach().cpu()
    img_side = sample["observation.images.side"].detach().cpu()

    print("[INFO] effective image keys for this checkpoint: observation.images.up, observation.images.side")

    return {
        "observation.images.up": img_top,
        "observation.images.side": img_side,
        "observation.state": sample["observation.state"].detach().cpu(),
        "task": task,
    }


# =========================
# GT action chunk 提取
# =========================
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

            # dataset 里通常 True 表示 pad，所以这里取反得到 valid
            return ~mask

    return np.ones(n_steps, dtype=bool)


def extract_gt_action_chunk_from_sample(sample: dict, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    gt = sample["action"]

    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    gt = np.asarray(gt, dtype=np.float32)

    if gt.ndim == 1:
        raise ValueError(
            f'sample["action"] 还是单步 action，shape={gt.shape}。\n'
            "说明 LeRobotDataset 没有使用 delta_timestamps 加载 action chunk。"
        )

    if gt.ndim != 2:
        raise ValueError(f'Expected sample["action"] shape [T, D], got shape={gt.shape}')

    gt = gt[:n_steps]
    valid_mask = get_action_pad_mask(sample, gt.shape[0])

    return gt, valid_mask


# =========================
# 指标与诊断
# =========================
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


def compute_delta_arrays(
    state_deg: np.ndarray,
    pred_action_chunk_deg: np.ndarray,
    gt_action_chunk_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = np.asarray(state_deg, dtype=np.float32).reshape(-1)
    pred = np.asarray(pred_action_chunk_deg, dtype=np.float32)
    gt = np.asarray(gt_action_chunk_deg, dtype=np.float32)

    if state.shape[0] < pred.shape[1]:
        raise ValueError(f"state dim {state.shape[0]} < action dim {pred.shape[1]}")

    state = state[: pred.shape[1]]

    pred_delta = pred - state[None, :]
    gt_delta = gt - state[None, :]
    delta_diff = pred_delta - gt_delta

    return pred_delta, gt_delta, delta_diff


def print_processor_check(
    obs_frame: dict,
    obs_processed: dict,
    raw_pred: torch.Tensor,
    post_pred: torch.Tensor,
) -> None:
    print("\n[PROCESSOR_CHECK]")

    raw_state = obs_frame["observation.state"]
    print(f"[PROCESSOR_CHECK] raw observation.state=[{format_values(raw_state)}]")

    for key, value in obs_processed.items():
        if isinstance(value, torch.Tensor) and "state" in key:
            flat = value.detach().cpu().flatten()
            print(
                f"[PROCESSOR_CHECK] processed {key}: "
                f"shape={tuple(value.shape)}, "
                f"min={flat.min().item():.4f}, max={flat.max().item():.4f}, "
                f"first=[{format_values(flat[: min(10, flat.numel())])}]"
            )

    raw_tensor = raw_pred.detach().cpu()
    post_tensor = post_pred.detach().cpu()

    print(
        f"[PROCESSOR_CHECK] raw_pred normalized?: "
        f"shape={tuple(raw_tensor.shape)}, "
        f"min={raw_tensor.min().item():.4f}, max={raw_tensor.max().item():.4f}"
    )
    print(
        f"[PROCESSOR_CHECK] raw_pred first action=[{format_values(raw_tensor[0, 0, :6])}]"
    )

    print(
        f"[PROCESSOR_CHECK] post_pred unnormalized?: "
        f"shape={tuple(post_tensor.shape)}, "
        f"min={post_tensor.min().item():.4f}, max={post_tensor.max().item():.4f}"
    )
    print(
        f"[PROCESSOR_CHECK] post_pred first action=[{format_values(post_tensor[0, 0, :6])}]"
    )


# =========================
# 保存结果
# =========================
def save_debug_artifacts(
    sample: dict,
    task: str,
    pred_action_chunk_deg: np.ndarray,
    gt_action_chunk_deg: np.ndarray,
    valid_mask: np.ndarray,
    metrics: dict,
    diagnostics: dict | None = None,
) -> None:
    state_deg = sample["observation.state"].detach().cpu().numpy()

    pred_action_chunk_deg = np.asarray(pred_action_chunk_deg, dtype=np.float32)
    gt_action_chunk_deg = np.asarray(gt_action_chunk_deg, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)

    t = min(pred_action_chunk_deg.shape[0], gt_action_chunk_deg.shape[0], valid_mask.shape[0])
    pred_action_chunk_deg = pred_action_chunk_deg[:t]
    gt_action_chunk_deg = gt_action_chunk_deg[:t]
    valid_mask = valid_mask[:t]

    diff_action_chunk_deg = pred_action_chunk_deg - gt_action_chunk_deg

    pred_delta_deg, gt_delta_deg, delta_diff_deg = compute_delta_arrays(
        state_deg=state_deg,
        pred_action_chunk_deg=pred_action_chunk_deg,
        gt_action_chunk_deg=gt_action_chunk_deg,
    )

    np.save(OUT_DIR / "input_state.npy", state_deg)

    np.save(OUT_DIR / "predicted_action_chunk.npy", pred_action_chunk_deg)
    np.save(OUT_DIR / "predicted_action_chunk_rad_for_rerun.npy", maybe_deg_to_rad(pred_action_chunk_deg))

    np.save(OUT_DIR / "dataset_gt_action_chunk.npy", gt_action_chunk_deg)
    np.save(OUT_DIR / "dataset_gt_action_chunk_rad_for_rerun.npy", maybe_deg_to_rad(gt_action_chunk_deg))

    np.save(OUT_DIR / "diff_pred_minus_gt_action_chunk.npy", diff_action_chunk_deg)
    np.save(OUT_DIR / "valid_gt_action_mask.npy", valid_mask)

    np.save(OUT_DIR / "pred_delta_from_obs_state.npy", pred_delta_deg)
    np.save(OUT_DIR / "gt_delta_from_obs_state.npy", gt_delta_deg)
    np.save(OUT_DIR / "delta_diff_pred_minus_gt.npy", delta_diff_deg)

    with (OUT_DIR / "pred_vs_dataset_gt_action_chunk.csv").open("w", encoding="utf-8") as f:
        header = ["step", "valid_gt"]
        for name in ACTION_NAMES:
            header.append(f"pred_{name}")
        for name in ACTION_NAMES:
            header.append(f"gt_{name}")
        for name in ACTION_NAMES:
            header.append(f"diff_pred_minus_gt_{name}")
        for name in ACTION_NAMES:
            header.append(f"pred_delta_{name}")
        for name in ACTION_NAMES:
            header.append(f"gt_delta_{name}")
        for name in ACTION_NAMES:
            header.append(f"delta_diff_pred_minus_gt_{name}")

        f.write(",".join(header) + "\n")

        for i in range(t):
            row_values = [
                str(i),
                str(bool(valid_mask[i])),
            ]

            row_values += [f"{v:.6f}" for v in pred_action_chunk_deg[i]]
            row_values += [f"{v:.6f}" for v in gt_action_chunk_deg[i]]
            row_values += [f"{v:.6f}" for v in diff_action_chunk_deg[i]]
            row_values += [f"{v:.6f}" for v in pred_delta_deg[i]]
            row_values += [f"{v:.6f}" for v in gt_delta_deg[i]]
            row_values += [f"{v:.6f}" for v in delta_diff_deg[i]]

            f.write(",".join(row_values) + "\n")

    meta = {
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "sample_index": SAMPLE_INDEX,
        "task": task,
        "device": DEVICE,
        "seed": SEED,
        "data_is_degrees": DATA_IS_DEGREES,
        "dataset_fps": DATASET_FPS,
        "action_offset_steps": ACTION_OFFSET_STEPS,
        "action_offset_seconds": ACTION_OFFSET_STEPS / float(DATASET_FPS),
        "camera_mapping_mode": CAMERA_MAPPING_MODE,
        "urdf_path": SO101_URDF_PATH,
        "input_state": state_deg.tolist(),
        "pred_chunk_shape_saved_and_replayed": list(pred_action_chunk_deg.shape),
        "gt_chunk_shape_saved": list(gt_action_chunk_deg.shape),
        "joint_order_used_for_urdf": JOINT_ORDER,
        "n_view_steps": N_VIEW_STEPS,
        "metrics_pred_vs_gt": metrics,
        "diagnostics": diagnostics or {},
    }

    (OUT_DIR / "debug_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


# =========================
# Rerun / URDF
# =========================
def init_rerun_and_load_urdf(urdf_path: str):
    rr.init("jhou_smolvla_pickplace_offline_rerun", spawn=True)

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
        rr.log(f"transforms/{urdf_joint_name}", transform)


def log_observation_state_urdf_to_rerun(
    robot_urdf,
    state_deg: np.ndarray,
    step_idx: int = 0,
) -> None:
    state_deg = np.asarray(state_deg, dtype=np.float32).flatten()

    if state_deg.shape[0] < 6:
        raise ValueError(f"Expected observation.state dim >= 6, got shape={state_deg.shape}")

    rr.set_time("step", sequence=step_idx)

    state_rad = maybe_deg_to_rad(state_deg)

    joint_positions_rad = {
        "shoulder_pan": float(state_rad[0]),
        "shoulder_lift": float(state_rad[1]),
        "elbow_flex": float(state_rad[2]),
        "wrist_flex": float(state_rad[3]),
        "wrist_roll": float(state_rad[4]),
    }

    log_joint_positions_to_rerun(robot_urdf, joint_positions_rad)

    for dim, name in enumerate(ACTION_NAMES):
        rr.log(f"plots/obs_state/{name}", rr.Scalars(float(state_deg[dim])))


def log_pred_gt_scalar_group_to_rerun(
    pred_action_deg: np.ndarray,
    gt_action_deg: np.ndarray,
    diff_deg: np.ndarray,
    valid_gt: bool,
) -> None:
    for dim, name in enumerate(ACTION_NAMES):
        rr.log(f"plots/action/{name}/pred", rr.Scalars(float(pred_action_deg[dim])))
        rr.log(f"plots/action/{name}/gt", rr.Scalars(float(gt_action_deg[dim])))
        rr.log(f"plots/action/{name}/diff_pred_minus_gt", rr.Scalars(float(diff_deg[dim])))

    rr.log("plots/action/valid_gt", rr.Scalars(float(valid_gt)))


def replay_action_chunk_with_urdf(
    robot_urdf,
    pred_action_chunk_deg: np.ndarray,
    gt_action_chunk_deg: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    start_step_idx: int = 1,
) -> None:
    pred_action_chunk_deg = np.asarray(pred_action_chunk_deg, dtype=np.float32)

    if gt_action_chunk_deg is not None:
        gt_action_chunk_deg = np.asarray(gt_action_chunk_deg, dtype=np.float32)
        t = min(pred_action_chunk_deg.shape[0], gt_action_chunk_deg.shape[0])
        pred_action_chunk_deg = pred_action_chunk_deg[:t]
        gt_action_chunk_deg = gt_action_chunk_deg[:t]

        if valid_mask is None:
            valid_mask = np.ones(t, dtype=bool)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)[:t]
    else:
        t = pred_action_chunk_deg.shape[0]
        valid_mask = np.ones(t, dtype=bool)

    for local_step_idx, pred_action_deg in enumerate(pred_action_chunk_deg):
        rerun_step_idx = start_step_idx + local_step_idx

        rr.set_time("step", sequence=rerun_step_idx)

        pred_action_rad = maybe_deg_to_rad(pred_action_deg)

        joint_positions_rad = {
            "shoulder_pan": float(pred_action_rad[0]),
            "shoulder_lift": float(pred_action_rad[1]),
            "elbow_flex": float(pred_action_rad[2]),
            "wrist_flex": float(pred_action_rad[3]),
            "wrist_roll": float(pred_action_rad[4]),
        }

        log_joint_positions_to_rerun(robot_urdf, joint_positions_rad)

        if gt_action_chunk_deg is not None:
            gt_action_deg = gt_action_chunk_deg[local_step_idx]
            diff_deg = pred_action_deg - gt_action_deg
            valid_gt = bool(valid_mask[local_step_idx])

            log_pred_gt_scalar_group_to_rerun(
                pred_action_deg=pred_action_deg,
                gt_action_deg=gt_action_deg,
                diff_deg=diff_deg,
                valid_gt=valid_gt,
            )
        else:
            for dim, name in enumerate(ACTION_NAMES):
                rr.log(f"plots/action/{name}/pred", rr.Scalars(float(pred_action_deg[dim])))

        time.sleep(SLEEP_PER_STEP_SEC)


# =========================
# 图片保存
# =========================
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

    img_top = tensor_image_to_numpy(sample["observation.images.up"])
    img_side = tensor_image_to_numpy(sample["observation.images.side"])

    top_path = OUT_DIR / f"sample_{sample_index}_OBS_IMAGE_top_up.png"
    side_path = OUT_DIR / f"sample_{sample_index}_OBS_IMAGE_side.png"

    Image.fromarray(img_top).save(top_path)
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


# =========================
# Processor 构建：关键修改
# =========================
def make_processors_for_model_or_dataset(model, dataset, device: torch.device):
    """
    优先从 MODEL_ID 加载 processor。
    如果 jhou/smolvla_pickplace 仓库没有 processor json，就 fallback 到 dataset.meta.stats。
    """
    try:
        print(f"[INFO] trying to load pre/post processors from model repo: {MODEL_ID}")
        preprocess, postprocess = make_pre_post_processors(
            model.config,
            MODEL_ID,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        print("[INFO] loaded processors from model repo.")
        return preprocess, postprocess

    except Exception as e:
        print("[WARN] failed to load processors from model repo.")
        print(f"[WARN] reason: {repr(e)}")
        print("[WARN] falling back to processors built from model.config + dataset.meta.stats")

        preprocess, postprocess = make_pre_post_processors(
            model.config,
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        return preprocess, postprocess


# =========================
# 主流程
# =========================
def main():
    set_seed(SEED)

    device = torch.device(DEVICE)

    print(f"[INFO] DEVICE              = {DEVICE}")
    print(f"[INFO] MODEL_ID            = {MODEL_ID}")
    print(f"[INFO] DATASET_ID          = {DATASET_ID}")
    print(f"[INFO] SAMPLE_INDEX        = {SAMPLE_INDEX}")
    print(f"[INFO] N_VIEW_STEPS        = {N_VIEW_STEPS}")
    print(f"[INFO] DATASET_FPS         = {DATASET_FPS}")
    print(f"[INFO] ACTION_OFFSET_STEPS = {ACTION_OFFSET_STEPS}")
    print(f"[INFO] ACTION_OFFSET_SEC   = {ACTION_OFFSET_STEPS / float(DATASET_FPS):.6f}")
    print(f"[INFO] CAMERA_MAPPING_MODE = {CAMERA_MAPPING_MODE}")
    print(f"[INFO] SEED                = {SEED}")
    print(f"[INFO] OUT_DIR             = {OUT_DIR}")

    # 0) optionally skip action/state semantics inspection
    # 1) 加载模型
    print(f"\n[INFO] loading policy: {MODEL_ID}")
    model = SmolVLAPolicy.from_pretrained(MODEL_ID).to(device)
    model.eval()
    print("[INFO] model loaded.")

    # 2) 用 delta_timestamps 加载 dataset，拿 action chunk GT
    action_delta_timestamps = build_action_delta_timestamps(
        n_steps=N_VIEW_STEPS,
        fps=DATASET_FPS,
        offset_steps=ACTION_OFFSET_STEPS,
    )

    print("\n[INFO] loading dataset with action delta_timestamps:")
    print(action_delta_timestamps)

    dataset = LeRobotDataset(
        DATASET_ID,
        delta_timestamps=action_delta_timestamps,
    )

    sample = dataset[SAMPLE_INDEX]
    save_input_frame_images(sample, SAMPLE_INDEX)

    # 3) task
    task = sample.get("task", None)
    if task is None or (isinstance(task, str) and len(task.strip()) == 0):
        task = FALLBACK_TASK

    print("\n[INFO] task used for inference:", task)
    print("[INFO] observation.state:", sample["observation.state"])
    print("[INFO] sample keys:", list(sample.keys()))

    # 4) 取 GT action chunk
    gt_action_chunk_deg, valid_gt_mask = extract_gt_action_chunk_from_sample(
        sample=sample,
        n_steps=N_VIEW_STEPS,
    )

    # 5) 构建 pre/post processor
    preprocess, postprocess = make_processors_for_model_or_dataset(
        model=model,
        dataset=dataset,
        device=device,
    )

    # 6) 构造离线输入
    obs_frame = build_offline_obs_frame(sample, task)

    # 7) preprocess
    obs = preprocess(obs_frame)

    # 8) 一次性预测完整 action chunk
    with torch.no_grad():
        raw_pred_action_chunk = model.predict_action_chunk(obs)
        post_pred_action_chunk = postprocess(raw_pred_action_chunk)

    raw_pred_tensor = unwrap_policy_action(raw_pred_action_chunk)
    post_pred_tensor = unwrap_policy_action(post_pred_action_chunk)

    if not isinstance(raw_pred_tensor, torch.Tensor):
        raise TypeError(f"Unexpected raw_pred_tensor type: {type(raw_pred_tensor)}")

    if not isinstance(post_pred_tensor, torch.Tensor):
        raise TypeError(f"Unexpected post_pred_tensor type: {type(post_pred_tensor)}")

    if post_pred_tensor.ndim != 3:
        raise ValueError(f"Expected post_pred_tensor shape [B, T, D], got {tuple(post_pred_tensor.shape)}")

    print_processor_check(
        obs_frame=obs_frame,
        obs_processed=obs,
        raw_pred=raw_pred_tensor,
        post_pred=post_pred_tensor,
    )

    # 9) 取 postprocess 后 action，与 dataset GT 对齐
    full_pred_action_chunk = post_pred_tensor.detach().cpu()[0].numpy()
    pred_action_chunk_deg = full_pred_action_chunk[:N_VIEW_STEPS]

    t = min(pred_action_chunk_deg.shape[0], gt_action_chunk_deg.shape[0], valid_gt_mask.shape[0])
    pred_action_chunk_deg = pred_action_chunk_deg[:t]
    gt_action_chunk_deg = gt_action_chunk_deg[:t]
    valid_gt_mask = valid_gt_mask[:t]

    metrics = compute_action_chunk_metrics(
        pred_action_chunk_deg=pred_action_chunk_deg,
        gt_action_chunk_deg=gt_action_chunk_deg,
        valid_mask=valid_gt_mask,
    )

    print("\n[METRICS]")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    for i in range(t):
        pred_row = pred_action_chunk_deg[i]
        gt_row = gt_action_chunk_deg[i]
        diff_row = pred_row - gt_row

        print(
            "[PRED_VS_GT] "
            f"chunk_step={i:03d}, "
            f"valid_gt={bool(valid_gt_mask[i])}, "
            f"pred=[{format_values(pred_row)}], "
            f"gt=[{format_values(gt_row)}], "
            f"diff=[{format_values(diff_row)}]"
        )

    init_state_deg = sample["observation.state"].detach().cpu().numpy()

    # 10) 保存
    save_debug_artifacts(
        sample=sample,
        task=task,
        pred_action_chunk_deg=pred_action_chunk_deg,
        gt_action_chunk_deg=gt_action_chunk_deg,
        valid_mask=valid_gt_mask,
        metrics=metrics,
    )

    print(f"\n[INFO] saved debug artifacts to: {OUT_DIR}")

    # 12) Rerun 可视化
    robot_urdf = init_rerun_and_load_urdf(SO101_URDF_PATH)

    log_observation_state_urdf_to_rerun(
        robot_urdf=robot_urdf,
        state_deg=init_state_deg,
        step_idx=0,
    )

    time.sleep(SLEEP_PER_STEP_SEC)

    replay_action_chunk_with_urdf(
        robot_urdf=robot_urdf,
        pred_action_chunk_deg=pred_action_chunk_deg,
        gt_action_chunk_deg=gt_action_chunk_deg,
        valid_mask=valid_gt_mask,
        start_step_idx=1,
    )


if __name__ == "__main__":
    main()