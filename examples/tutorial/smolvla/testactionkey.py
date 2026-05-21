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
from lerobot.processor.converters import to_tensor


# =========================
# 配置区
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/svla_so101_pickplace"
SAMPLE_INDEX = 70

# FALLBACK_TASK = "pick up the pink lego brick"
FALLBACK_TASK = "put the pink lego brick into the transparent box"

SO101_URDF_PATH = "/home/shuowen/Repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

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

ACTION_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

DATA_IS_DEGREES = True
SLEEP_PER_STEP_SEC = 0.08

# 模型仍然预测完整 chunk，但这里只保存 / 打印 / Rerun 前 N_VIEW_STEPS step
N_VIEW_STEPS = 50

# svla_so101_pickplace 通常是 30fps
DATASET_FPS = 30

# =========================
# GT action 时间 offset
# =========================
ACTION_OFFSET_STEPS = 0

# =========================
# Camera mapping 消融
# =========================
# 可选：
# "top_wrist_side"
# "top_side_empty"
# "side_top_empty"
# "top_empty_side"
# "side_empty_top"
CAMERA_MAPPING_MODE = "side_empty_top"

# 固定随机种子，方便比较不同 offset / camera mapping 的结果
SEED = 0

# legacy action stats remap key
# 可选值：
#  - "so100-blue.buffer.action"
#  - "so100-red.buffer.action"
#  - "so100.buffer.action"
# 如果为 None，则按当前 helper 的 fallback 顺序选择第一个可用 key。
LEGACY_ACTION_STATS_KEY: str | None = "so100.buffer.action"

OUT_DIR = Path("outputs/offline_smolvla_rerun")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 工具函数
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


def build_action_delta_timestamps(
    n_steps: int,
    fps: int | float,
    offset_steps: int = 0,
) -> dict[str, list[float]]:
    return {
        "action": [(i + offset_steps) / float(fps) for i in range(n_steps)]
    }


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
    根据 CAMERA_MAPPING_MODE 构造 SmolVLA 期望的三路图像输入。
    """
    img_top = sample["observation.images.up"].detach().cpu()
    img_side = sample["observation.images.side"].detach().cpu()
    empty = torch.zeros_like(img_top)
    img_wrist = get_wrist_image_or_empty(sample, empty)

    if CAMERA_MAPPING_MODE == "top_wrist_side":
        cam1 = img_top
        cam2 = img_wrist
        cam3 = img_side
        mapping_description = "camera1=top/up, camera2=wrist_or_empty, camera3=side"

    elif CAMERA_MAPPING_MODE == "top_side_empty":
        cam1 = img_top
        cam2 = img_side
        cam3 = empty
        mapping_description = "camera1=top/up, camera2=side, camera3=empty"

    elif CAMERA_MAPPING_MODE == "side_top_empty":
        cam1 = img_side
        cam2 = img_top
        cam3 = empty
        mapping_description = "camera1=side, camera2=top/up, camera3=empty"

    elif CAMERA_MAPPING_MODE == "top_empty_side":
        cam1 = img_top
        cam2 = empty
        cam3 = img_side
        mapping_description = "camera1=top/up, camera2=empty, camera3=side"

    elif CAMERA_MAPPING_MODE == "side_empty_top":
        cam1 = img_side
        cam2 = empty
        cam3 = img_top
        mapping_description = "camera1=side, camera2=empty, camera3=top/up"

    else:
        raise ValueError(
            f"Unknown CAMERA_MAPPING_MODE={CAMERA_MAPPING_MODE}. "
            "Use one of: top_wrist_side, top_side_empty, side_top_empty, top_empty_side, side_empty_top"
        )

    print(f"[INFO] effective camera mapping: {mapping_description}")

    return {
        "observation.images.camera1": cam1,
        "observation.images.camera2": cam2,
        "observation.images.camera3": cam3,
        "observation.state": sample["observation.state"].detach().cpu(),
        "task": task,
    }


def get_action_pad_mask(sample: dict, n_steps: int) -> np.ndarray:
    candidate_keys = [
        "action_is_pad",
        "action_is_padding",
        "action_pad_mask",
        "action_is_pad_mask",
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


def remap_legacy_action_stats_to_action(processor, preferred_key: str | None = None) -> None:
    """Remap legacy action stat keys like so100.buffer.action to the generic action key."""
    for step in processor.steps:
        if not hasattr(step, "stats") or not isinstance(step.stats, dict):
            continue

        if "action" in step.stats:
            continue

        legacy_keys = [key for key in step.stats if key.endswith(".action")]
        if not legacy_keys:
            continue

        if preferred_key is not None and preferred_key in legacy_keys:
            chosen_key = preferred_key
        else:
            chosen_key = legacy_keys[0]

        step.stats["action"] = step.stats[chosen_key]
        step._tensor_stats = to_tensor(step.stats, device=getattr(step, "device", None), dtype=getattr(step, "dtype", None))

        print(
            f"[LEGACY_ACTION_REMAP] processor={type(processor).__name__}, "
            f"step={type(step).__name__}, "
            f"chosen_key={chosen_key}, "
            f"available_keys={legacy_keys}"
        )


def extract_gt_action_chunk_from_sample(sample: dict, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    gt = sample["action"]

    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    gt = np.asarray(gt, dtype=np.float32)

    if gt.ndim == 1:
        raise ValueError(
            f'sample["action"] 还是单步 action，shape={gt.shape}。\n'
            f"说明 LeRobotDataset 没有使用 delta_timestamps 加载。"
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


def print_delta_analysis(
    state_deg: np.ndarray,
    pred_action_chunk_deg: np.ndarray,
    gt_action_chunk_deg: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    pred_delta, gt_delta, delta_diff = compute_delta_arrays(
        state_deg=state_deg,
        pred_action_chunk_deg=pred_action_chunk_deg,
        gt_action_chunk_deg=gt_action_chunk_deg,
    )

    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    t = min(pred_delta.shape[0], gt_delta.shape[0], valid_mask.shape[0])

    print("\n[DELTA_ANALYSIS] relative to observation.state")
    print(f"[DELTA_ANALYSIS] observation.state=[{format_values(state_deg)}]")

    candidate_steps = [0, 1, 2, 5, 10, 20, 30, 40, t - 1]
    printed = set()

    for i in candidate_steps:
        if i in printed:
            continue
        printed.add(i)

        if 0 <= i < t:
            print(
                "[DELTA] "
                f"chunk_step={i:03d}, "
                f"valid_gt={bool(valid_mask[i])}, "
                f"pred_delta=[{format_values(pred_delta[i])}], "
                f"gt_delta=[{format_values(gt_delta[i])}], "
                f"delta_diff=[{format_values(delta_diff[i])}]"
            )


# =========================
# 新增：误差定位诊断函数
# =========================
def compute_segment_metrics(
    pred_action_chunk_deg: np.ndarray,
    gt_action_chunk_deg: np.ndarray,
    valid_mask: np.ndarray,
    segments: tuple[int, ...] = (5, 10, 20, 30, 40, 50),
) -> dict:
    """
    分段 MAE/RMSE：
    用来判断是前几步就错，还是后面 open-loop chunk 误差累积。
    """
    pred = np.asarray(pred_action_chunk_deg, dtype=np.float32)
    gt = np.asarray(gt_action_chunk_deg, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)

    results = {}

    print("\n[SEGMENT_METRICS]")
    for end in segments:
        t = min(end, pred.shape[0], gt.shape[0], valid_mask.shape[0])
        if t <= 0:
            continue

        mask = valid_mask[:t]
        if not np.any(mask):
            continue

        diff = pred[:t][mask] - gt[:t][mask]
        mae_overall = float(np.mean(np.abs(diff)))
        rmse_overall = float(np.sqrt(np.mean(diff ** 2)))
        mae_per_dim = np.mean(np.abs(diff), axis=0)
        rmse_per_dim = np.sqrt(np.mean(diff ** 2, axis=0))

        results[f"first_{t}_steps"] = {
            "mae_overall": mae_overall,
            "rmse_overall": rmse_overall,
            "mae_per_dim": mae_per_dim.tolist(),
            "rmse_per_dim": rmse_per_dim.tolist(),
        }

        print(
            f"[SEGMENT] first_{t:02d}_steps "
            f"mae_overall={mae_overall:.3f}, "
            f"rmse_overall={rmse_overall:.3f}, "
            f"mae_per_dim=[{format_values(mae_per_dim)}]"
        )

    return results


def direction_agreement(
    state_deg: np.ndarray,
    pred_action_chunk_deg: np.ndarray,
    gt_action_chunk_deg: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1.0,
) -> dict:
    """
    判断 pred_delta 和 gt_delta 的方向是否一致。

    eps:
        忽略 GT delta 绝对值小于 eps 的位置，因为这些位置方向不稳定。
    """
    pred_delta, gt_delta, _ = compute_delta_arrays(
        state_deg=state_deg,
        pred_action_chunk_deg=pred_action_chunk_deg,
        gt_action_chunk_deg=gt_action_chunk_deg,
    )

    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    t = min(pred_delta.shape[0], gt_delta.shape[0], valid_mask.shape[0])

    pred_delta = pred_delta[:t][valid_mask[:t]]
    gt_delta = gt_delta[:t][valid_mask[:t]]

    gt_moving_mask = np.abs(gt_delta) > eps
    agree = np.sign(pred_delta) == np.sign(gt_delta)

    per_dim = []
    print("\n[DIRECTION]")
    print(f"[DIRECTION] eps={eps}")

    if gt_moving_mask.any():
        overall = float(agree[gt_moving_mask].mean())
    else:
        overall = None

    print(f"[DIRECTION] overall={overall}")

    for dim, name in enumerate(ACTION_NAMES):
        dim_mask = gt_moving_mask[:, dim]
        if dim_mask.sum() == 0:
            value = None
            count = 0
        else:
            value = float(agree[:, dim][dim_mask].mean())
            count = int(dim_mask.sum())

        per_dim.append(value)
        print(f"[DIRECTION] {dim}:{name} agreement={value}, count={count}")

    return {
        "eps": eps,
        "overall": overall,
        "per_dim": per_dim,
    }


def diagnose_dim_swap(
    pred_action_chunk_deg: np.ndarray,
    gt_action_chunk_deg: np.ndarray,
    valid_mask: np.ndarray,
) -> list[dict]:
    """
    诊断 action 维度顺序是否可能错。

    做法：
        临时交换 pred 的任意两个维度，再算 MAE。
    如果某个 swap 让 MAE 大幅下降，说明 action dim order 可疑。

    注意：
        这只是诊断，不是正式修正。
    """
    pred = np.asarray(pred_action_chunk_deg, dtype=np.float32)
    gt = np.asarray(gt_action_chunk_deg, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    base_metrics = compute_action_chunk_metrics(pred, gt, valid_mask)
    base_mae = base_metrics["mae_overall"]

    results = []

    print("\n[DIM_SWAP_DIAG]")
    print(f"[DIM_SWAP] base_mae={base_mae:.3f}")

    d = pred.shape[1]
    for i in range(d):
        for j in range(i + 1, d):
            pred_swapped = pred.copy()
            pred_swapped[:, [i, j]] = pred_swapped[:, [j, i]]

            m = compute_action_chunk_metrics(pred_swapped, gt, valid_mask)
            mae = m["mae_overall"]
            delta = mae - base_mae

            item = {
                "swap": [int(i), int(j)],
                "names": [ACTION_NAMES[i], ACTION_NAMES[j]],
                "mae_overall": float(mae),
                "delta_vs_base": float(delta),
            }
            results.append(item)

            print(
                f"[DIM_SWAP] swap {i}-{j} "
                f"({ACTION_NAMES[i]} <-> {ACTION_NAMES[j]}): "
                f"mae={mae:.3f}, delta={delta:+.3f}"
            )

    best = min(results, key=lambda x: x["mae_overall"]) if results else None
    if best is not None:
        print(
            "[DIM_SWAP] best_swap="
            f"{best['swap']} {best['names']}, "
            f"mae={best['mae_overall']:.3f}, "
            f"delta={best['delta_vs_base']:+.3f}"
        )

    return results


def diagnose_dim_sign_flip(
    state_deg: np.ndarray,
    pred_action_chunk_deg: np.ndarray,
    gt_action_chunk_deg: np.ndarray,
    valid_mask: np.ndarray,
) -> list[dict]:
    """
    诊断某个 joint 的 delta 符号是否可能反了。

    做法：
        pred_flip[:, dim] = 2 * state[dim] - pred[:, dim]

    也就是以当前 observation.state 为中心，把该维度的 pred_delta 反向。

    如果 flip 某个维度后 MAE 大幅下降，说明该维度存在符号 convention 问题的可能性。
    """
    state = np.asarray(state_deg, dtype=np.float32).reshape(-1)
    pred = np.asarray(pred_action_chunk_deg, dtype=np.float32)
    gt = np.asarray(gt_action_chunk_deg, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    base_metrics = compute_action_chunk_metrics(pred, gt, valid_mask)
    base_mae = base_metrics["mae_overall"]

    results = []

    print("\n[DIM_SIGN_FLIP_DIAG]")
    print(f"[SIGN_FLIP] base_mae={base_mae:.3f}")

    for dim, name in enumerate(ACTION_NAMES):
        pred_flip = pred.copy()
        pred_flip[:, dim] = 2.0 * state[dim] - pred_flip[:, dim]

        m = compute_action_chunk_metrics(pred_flip, gt, valid_mask)
        mae = m["mae_overall"]
        delta = mae - base_mae

        item = {
            "dim": int(dim),
            "name": name,
            "mae_overall": float(mae),
            "delta_vs_base": float(delta),
        }
        results.append(item)

        print(
            f"[SIGN_FLIP] flip {dim}:{name}: "
            f"mae={mae:.3f}, delta={delta:+.3f}"
        )

    best = min(results, key=lambda x: x["mae_overall"]) if results else None
    if best is not None:
        print(
            "[SIGN_FLIP] best_flip="
            f"{best['dim']}:{best['name']}, "
            f"mae={best['mae_overall']:.3f}, "
            f"delta={best['delta_vs_base']:+.3f}"
        )

    return results


def inspect_action_state_semantics(
    dataset_id: str,
    sample_index: int,
    window: int = 5,
) -> None:
    """
    检查 dataset 里的 action 和 observation.state 的关系。

    用来判断：
        action_t 是否接近 state_t
        action_t 是否接近 state_{t+1}
        action 是绝对目标位置还是 delta/command

    这里只打印，不参与主评估。
    """
    print("\n[ACTION_STATE_SEMANTICS]")
    ds = LeRobotDataset(dataset_id)

    for offset in range(window):
        idx = sample_index + offset
        if idx >= len(ds):
            break

        s = ds[idx]
        state = s["observation.state"].detach().cpu().numpy().astype(np.float32)
        action = s["action"].detach().cpu().numpy().astype(np.float32)

        print(f"\n[ACTION_STATE] idx={idx}")
        print(f"state=[{format_values(state)}]")
        print(f"action=[{format_values(action)}]")
        print(f"action_minus_state=[{format_values(action - state)}]")

        if idx + 1 < len(ds):
            s_next = ds[idx + 1]
            state_next = s_next["observation.state"].detach().cpu().numpy().astype(np.float32)
            print(f"next_state=[{format_values(state_next)}]")
            print(f"action_minus_next_state=[{format_values(action - state_next)}]")


def print_processor_check(
    obs_frame: dict,
    obs_processed: dict,
    raw_pred: torch.Tensor,
    post_pred: torch.Tensor,
) -> None:
    """
    检查 preprocessor/postprocessor 是否确实改变了 state/action 数值。
    """
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
        f"[PROCESSOR_CHECK] post_pred degree?: "
        f"shape={tuple(post_tensor.shape)}, "
        f"min={post_tensor.min().item():.4f}, max={post_tensor.max().item():.4f}"
    )
    print(
        f"[PROCESSOR_CHECK] post_pred first action=[{format_values(post_tensor[0, 0, :6])}]"
    )


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

    np.save(OUT_DIR / "input_state_deg.npy", state_deg)

    np.save(OUT_DIR / "predicted_action_chunk_deg.npy", pred_action_chunk_deg)
    np.save(OUT_DIR / "predicted_action_chunk_rad.npy", maybe_deg_to_rad(pred_action_chunk_deg))

    np.save(OUT_DIR / "dataset_gt_action_chunk_deg.npy", gt_action_chunk_deg)
    np.save(OUT_DIR / "dataset_gt_action_chunk_rad.npy", maybe_deg_to_rad(gt_action_chunk_deg))

    np.save(OUT_DIR / "diff_pred_minus_gt_action_chunk_deg.npy", diff_action_chunk_deg)
    np.save(OUT_DIR / "valid_gt_action_mask.npy", valid_mask)

    np.save(OUT_DIR / "pred_delta_from_obs_state_deg.npy", pred_delta_deg)
    np.save(OUT_DIR / "gt_delta_from_obs_state_deg.npy", gt_delta_deg)
    np.save(OUT_DIR / "delta_diff_pred_minus_gt_deg.npy", delta_diff_deg)

    with (OUT_DIR / "pred_vs_dataset_gt_action_chunk_deg.csv").open("w", encoding="utf-8") as f:
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
        "input_state_deg": state_deg.tolist(),
        "pred_chunk_shape_saved_and_replayed": list(pred_action_chunk_deg.shape),
        "gt_chunk_shape_saved": list(gt_action_chunk_deg.shape),
        "joint_order_used_for_urdf": JOINT_ORDER,
        "n_view_steps": N_VIEW_STEPS,
        "metrics_pred_vs_gt": metrics,
        "diagnostics": diagnostics or {},
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
        rr.log(f"plots/obs_state_deg/{name}", rr.Scalars(float(state_deg[dim])))


def log_pred_gt_scalar_group_to_rerun(
    pred_action_deg: np.ndarray,
    gt_action_deg: np.ndarray,
    diff_deg: np.ndarray,
    valid_gt: bool,
) -> None:
    for dim, name in enumerate(ACTION_NAMES):
        rr.log(f"plots/action_deg/{name}/pred", rr.Scalars(float(pred_action_deg[dim])))
        rr.log(f"plots/action_deg/{name}/gt", rr.Scalars(float(gt_action_deg[dim])))
        rr.log(f"plots/action_deg/{name}/diff_pred_minus_gt", rr.Scalars(float(diff_deg[dim])))

    rr.log("plots/action_deg/valid_gt", rr.Scalars(float(valid_gt)))


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
                rr.log(f"plots/action_deg/{name}/pred", rr.Scalars(float(pred_action_deg[dim])))

        time.sleep(SLEEP_PER_STEP_SEC)


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

    # 0) 新增：检查 action 和 state 的语义关系
    #    这个会额外加载一次不带 delta_timestamps 的 dataset，只打印几个样本。
    inspect_action_state_semantics(
        dataset_id=DATASET_ID,
        sample_index=SAMPLE_INDEX,
        window=5,
    )

    # 1) 加载模型，并明确放到 DEVICE
    model = SmolVLAPolicy.from_pretrained(MODEL_ID).to(device)
    model.eval()

    # 2) 用 delta_timestamps 加载数据集
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

    task = sample.get("task", None)
    if task is None or (isinstance(task, str) and len(task.strip()) == 0):
        task = FALLBACK_TASK

    
    # task = FALLBACK_TASK

    print("[INFO] task used for inference:", task)
    print("[INFO] observation.state:", sample["observation.state"])

    # 3) 从 dataset 里取真实演示 action chunk
    gt_action_chunk_deg, valid_gt_mask = extract_gt_action_chunk_from_sample(
        sample=sample,
        n_steps=N_VIEW_STEPS,
    )

    # 4) 构建 pre/post processor
    preprocess, postprocess = make_pre_post_processors(
        model.config,
        MODEL_ID,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    chosen_legacy_key = LEGACY_ACTION_STATS_KEY
    print(f"[INFO] chosen legacy action stats key: {chosen_legacy_key}")

    # Explicitly remap legacy action stat keys from pretrained state to the generic `action` key.
    remap_legacy_action_stats_to_action(preprocess, preferred_key=chosen_legacy_key)
    remap_legacy_action_stats_to_action(postprocess, preferred_key=chosen_legacy_key)

    # 5) 构造离线输入
    obs_frame = build_offline_obs_frame(sample, task)

    # 6) preprocess
    obs = preprocess(obs_frame)

    # 7) 一次性预测整个 action chunk
    with torch.no_grad():
        raw_pred_action_chunk = model.predict_action_chunk(obs)
        post_pred_action_chunk = postprocess(raw_pred_action_chunk)

    if hasattr(raw_pred_action_chunk, "actions"):
        raw_pred_tensor = raw_pred_action_chunk.actions
    else:
        raw_pred_tensor = raw_pred_action_chunk

    if hasattr(post_pred_action_chunk, "actions"):
        post_pred_tensor = post_pred_action_chunk.actions
    else:
        post_pred_tensor = post_pred_action_chunk

    if not isinstance(raw_pred_tensor, torch.Tensor):
        raise TypeError(f"Unexpected raw_pred_tensor type: {type(raw_pred_tensor)}")

    if not isinstance(post_pred_tensor, torch.Tensor):
        raise TypeError(f"Unexpected post_pred_tensor type: {type(post_pred_tensor)}")

    if post_pred_tensor.ndim != 3:
        raise ValueError(f"Expected post_pred_tensor shape [B, T, D], got {tuple(post_pred_tensor.shape)}")

    # 新增：检查 pre/postprocessor 是否真的工作
    print_processor_check(
        obs_frame=obs_frame,
        obs_processed=obs,
        raw_pred=raw_pred_tensor,
        post_pred=post_pred_tensor,
    )

    pred_action_chunk = post_pred_tensor.detach().cpu()
    full_pred_action_chunk_deg = pred_action_chunk[0].numpy()

    pred_action_chunk_deg = full_pred_action_chunk_deg[:N_VIEW_STEPS]

    # 8) 预测 chunk vs 数据集真实 chunk 对齐并计算误差
    t = min(pred_action_chunk_deg.shape[0], gt_action_chunk_deg.shape[0], valid_gt_mask.shape[0])
    pred_action_chunk_deg = pred_action_chunk_deg[:t]
    gt_action_chunk_deg = gt_action_chunk_deg[:t]
    valid_gt_mask = valid_gt_mask[:t]

    metrics = compute_action_chunk_metrics(
        pred_action_chunk_deg=pred_action_chunk_deg,
        gt_action_chunk_deg=gt_action_chunk_deg,
        valid_mask=valid_gt_mask,
    )

    print("[METRICS]", json.dumps(metrics, indent=2, ensure_ascii=False))

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

    # 8.5) Delta 分析
    init_state_deg = sample["observation.state"].detach().cpu().numpy()
    print_delta_analysis(
        state_deg=init_state_deg,
        pred_action_chunk_deg=pred_action_chunk_deg,
        gt_action_chunk_deg=gt_action_chunk_deg,
        valid_mask=valid_gt_mask,
    )

    # 8.6) 新增：误差定位诊断
    segment_metrics = compute_segment_metrics(
        pred_action_chunk_deg=pred_action_chunk_deg,
        gt_action_chunk_deg=gt_action_chunk_deg,
        valid_mask=valid_gt_mask,
    )

    direction_metrics = direction_agreement(
        state_deg=init_state_deg,
        pred_action_chunk_deg=pred_action_chunk_deg,
        gt_action_chunk_deg=gt_action_chunk_deg,
        valid_mask=valid_gt_mask,
        eps=1.0,
    )

    dim_swap_results = diagnose_dim_swap(
        pred_action_chunk_deg=pred_action_chunk_deg,
        gt_action_chunk_deg=gt_action_chunk_deg,
        valid_mask=valid_gt_mask,
    )

    sign_flip_results = diagnose_dim_sign_flip(
        state_deg=init_state_deg,
        pred_action_chunk_deg=pred_action_chunk_deg,
        gt_action_chunk_deg=gt_action_chunk_deg,
        valid_mask=valid_gt_mask,
    )

    diagnostics = {
        "segment_metrics": segment_metrics,
        "direction_agreement": direction_metrics,
        "dim_swap": dim_swap_results,
        "dim_sign_flip": sign_flip_results,
    }

    # 9) 保存调试结果
    save_debug_artifacts(
        sample=sample,
        task=task,
        pred_action_chunk_deg=pred_action_chunk_deg,
        gt_action_chunk_deg=gt_action_chunk_deg,
        valid_mask=valid_gt_mask,
        metrics=metrics,
        diagnostics=diagnostics,
    )

    # 10) 初始化 Rerun 并加载 URDF
    robot_urdf = init_rerun_and_load_urdf(SO101_URDF_PATH)

    # 11) 先把读入的 observation.state 作为 URDF 初始姿态记录进去
    log_observation_state_urdf_to_rerun(
        robot_urdf=robot_urdf,
        state_deg=init_state_deg,
        step_idx=0,
    )

    time.sleep(SLEEP_PER_STEP_SEC)

    # 12) 再从 step=1 开始连续回放 predicted action chunk
    replay_action_chunk_with_urdf(
        robot_urdf=robot_urdf,
        pred_action_chunk_deg=pred_action_chunk_deg,
        gt_action_chunk_deg=gt_action_chunk_deg,
        valid_mask=valid_gt_mask,
        start_step_idx=1,
    )


if __name__ == "__main__":
    main()