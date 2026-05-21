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
MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/svla_so100_pickplace"
SAMPLE_INDEX = 160
FALLBACK_TASK = "Pick up the cube and place it in the box."

SO100_URDF_PATH = "/home/shuowen/Repos/SO-ARM100/Simulation/SO100/so100.urdf"

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
    # "gripper": "gripper",
}

ACTION_NAMES = [
    "main_shoulder_pan",
    "main_shoulder_lift",
    "main_elbow_flex",
    "main_wrist_flex",
    "main_wrist_roll",
    "main_gripper",
]

DATA_IS_DEGREES = True
SLEEP_PER_STEP_SEC = 0.08

# 模型仍然预测完整 chunk，但这里只保存 / 打印 / Rerun 前 N_VIEW_STEPS step
N_VIEW_STEPS = 50

# svla_so100_pickplace 通常是 30fps
DATASET_FPS = 30

# =========================
# 新增：GT action 时间 offset
# =========================
# 0 表示 GT = action_t, action_t+1, ...
# 1 表示 GT = action_t+1, action_t+2, ...
# 3 表示 GT = action_t+3, action_t+4, ...
#
# 你可以依次试：
# ACTION_OFFSET_STEPS = 0
# ACTION_OFFSET_STEPS = 1
# ACTION_OFFSET_STEPS = 2
# ACTION_OFFSET_STEPS = 3
# ACTION_OFFSET_STEPS = 5
# ACTION_OFFSET_STEPS = 10
ACTION_OFFSET_STEPS = 0

# 文献里的图像优先顺序：
#   OBS_IMAGE_1 = top
#   OBS_IMAGE_2 = wrist
#   OBS_IMAGE_3 = side
#
# 当前 SO100 数据集只有 top / wrist，没有 side，所以 OBS_IMAGE_3 用全 0 图占位。
#
# smolvla_base 的 config 仍然期望 observation.images.camera1/2/3，
# 所以下面只改变三路 camera 的语义映射。
CAMERA_MAPPING_MODE = "top_wrist_side"

# 固定随机种子，方便比较不同 offset / camera mapping 的结果
SEED = 0

OUT_DIR = Path("outputs/offline_smolvla_so100_rerun")
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
    """
    让 dataset[SAMPLE_INDEX]["action"] 返回：

    offset_steps = 0:
        [action_t, action_t+1, ..., action_t+n_steps-1]

    offset_steps = 3:
        [action_t+3, action_t+4, ..., action_t+3+n_steps-1]

    用来测试 observation 和 action 之间是否存在时间偏移。
    """
    return {
        "action": [(i + offset_steps) / float(fps) for i in range(n_steps)]
    }


def build_offline_obs_frame(sample: dict, task: str) -> dict:
    """
    按文献的 top / wrist / side 优先顺序构造三路图像输入。

    smolvla_base 期望的 key:
        observation.images.camera1 = OBS_IMAGE_1 = dataset top
        observation.images.camera2 = OBS_IMAGE_2 = dataset wrist
        observation.images.camera3 = OBS_IMAGE_3 = empty side

    当前 svla_so100_pickplace 没有 side 图像，所以用全 0 图占位。
    """
    img_top = sample["observation.images.top"].detach().cpu()
    img_wrist = sample["observation.images.wrist"].detach().cpu()
    empty = torch.zeros_like(img_top)

    return {
        "observation.images.camera1": img_top,
        "observation.images.camera2": img_wrist,
        "observation.images.camera3": empty,
        "observation.state": sample["observation.state"].detach().cpu(),
        "task": task,
    }


def get_action_pad_mask(sample: dict, n_steps: int) -> np.ndarray:
    """
    有些 LeRobotDataset 在 delta_timestamps 越过 episode 末尾时会返回 padding mask。

    返回:
        valid_mask: shape [T], True 表示这个 step 是有效真实数据，不是 padding。
    """
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

            # 通常 mask=True 表示 padding，所以 valid = not mask
            return ~mask

    return np.ones(n_steps, dtype=bool)


def extract_gt_action_chunk_from_sample(sample: dict, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    从带 delta_timestamps 的 sample 中提取数据集真实 action chunk。

    期望 sample["action"] 是 [T, D]。
    """
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
    """
    计算预测 chunk 和数据集真实 chunk 的误差。
    """
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


def save_debug_artifacts(
    sample: dict,
    task: str,
    pred_action_chunk_deg: np.ndarray,
    gt_action_chunk_deg: np.ndarray,
    valid_mask: np.ndarray,
    metrics: dict,
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

    np.save(OUT_DIR / "input_state_deg.npy", state_deg)

    np.save(OUT_DIR / "predicted_action_chunk_deg.npy", pred_action_chunk_deg)
    np.save(OUT_DIR / "predicted_action_chunk_rad.npy", maybe_deg_to_rad(pred_action_chunk_deg))

    np.save(OUT_DIR / "dataset_gt_action_chunk_deg.npy", gt_action_chunk_deg)
    np.save(OUT_DIR / "dataset_gt_action_chunk_rad.npy", maybe_deg_to_rad(gt_action_chunk_deg))

    np.save(OUT_DIR / "diff_pred_minus_gt_action_chunk_deg.npy", diff_action_chunk_deg)
    np.save(OUT_DIR / "valid_gt_action_mask.npy", valid_mask)

    with (OUT_DIR / "pred_vs_dataset_gt_action_chunk_deg.csv").open("w", encoding="utf-8") as f:
        header = ["step", "valid_gt"]
        for name in ACTION_NAMES:
            header.append(f"pred_{name}")
        for name in ACTION_NAMES:
            header.append(f"gt_{name}")
        for name in ACTION_NAMES:
            header.append(f"diff_pred_minus_gt_{name}")

        f.write(",".join(header) + "\n")

        for i in range(t):
            row_values = [
                str(i),
                str(bool(valid_mask[i])),
            ]

            row_values += [f"{v:.6f}" for v in pred_action_chunk_deg[i]]
            row_values += [f"{v:.6f}" for v in gt_action_chunk_deg[i]]
            row_values += [f"{v:.6f}" for v in diff_action_chunk_deg[i]]

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
        "urdf_path": SO100_URDF_PATH,
        "input_state_deg": state_deg.tolist(),
        "pred_chunk_shape_saved_and_replayed": list(pred_action_chunk_deg.shape),
        "gt_chunk_shape_saved": list(gt_action_chunk_deg.shape),
        "joint_order_used_for_urdf": JOINT_ORDER,
        "n_view_steps": N_VIEW_STEPS,
        "metrics_pred_vs_gt": metrics,
    }

    (OUT_DIR / "debug_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def init_rerun_and_load_urdf(urdf_path: str):
    rr.init("smolvla_so100_offline_chunk_rerun", spawn=True)

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
    """
    把每个关节的 transform 写入 Rerun。

    注意：
    每个 joint 必须写到不同 path。
    不能都写 rr.log("transforms", transform)，否则后面的 joint 会覆盖前面的 joint。
    """
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
    """
    把当前读入的 observation.state 作为 URDF 初始姿态记录到 Rerun。

    Rerun 时间安排：
    - step=0: observation.state
    - step=1: predicted_action_chunk[0]
    - step=2: predicted_action_chunk[1]
    ...
    """
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

    # observation.state 单独放在 obs_state 组里，不和 pred/gt 混在一起
    for dim, name in enumerate(ACTION_NAMES):
        rr.log(f"plots/obs_state_deg/{name}", rr.Scalars(float(state_deg[dim])))


def log_pred_gt_scalar_group_to_rerun(
    pred_action_deg: np.ndarray,
    gt_action_deg: np.ndarray,
    diff_deg: np.ndarray,
    valid_gt: bool,
) -> None:
    """
    Rerun 折线图分组方式：

    plots/action_deg/shoulder_pan/pred
    plots/action_deg/shoulder_pan/gt

    这样在 Rerun 里选中：
        plots/action_deg/shoulder_pan

    就能看到 shoulder_pan 的 pred 和 gt 两条线在同一组下。
    gripper 同理：
        plots/action_deg/gripper/pred
        plots/action_deg/gripper/gt
    """
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
    """
    从 start_step_idx 开始回放预测 action chunk。

    Rerun 中：
    - URDF 动画仍然用 pred action chunk 驱动
    - 折线图按关节分组，每组 pred/gt 两条线
    """
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

            # 新版折线图：按 joint 分组，每个 joint 下 pred / gt / diff
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
    """
    支持 LeRobot 常见图像格式：
    - [C, H, W]
    - [H, W, C]
    - float: 0~1 或 0~255
    - uint8: 0~255

    输出 uint8 [H, W, C]
    """
    img = img.detach().cpu()

    if img.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got shape={tuple(img.shape)}")

    # [C, H, W] -> [H, W, C]
    if img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)

    img_np = img.numpy()

    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = img_np * 255.0
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    return img_np


def save_input_frame_images(sample: dict, sample_index: int) -> None:
    """
    保存当前用于推理的 dataset 图像帧。
    """
    from PIL import Image

    img_top = tensor_image_to_numpy(sample["observation.images.top"])
    img_wrist = tensor_image_to_numpy(sample["observation.images.wrist"])

    top_path = OUT_DIR / f"sample_{sample_index}_OBS_IMAGE_1_top.png"
    wrist_path = OUT_DIR / f"sample_{sample_index}_OBS_IMAGE_2_wrist.png"

    Image.fromarray(img_top).save(top_path)
    Image.fromarray(img_wrist).save(wrist_path)


# =========================
# 主流程
# =========================
def main():
    set_seed(SEED)

    print(f"[INFO] DEVICE              = {DEVICE}")
    print(f"[INFO] MODEL_ID            = {MODEL_ID}")
    print(f"[INFO] DATASET_ID          = {DATASET_ID}")
    print(f"[INFO] SAMPLE_INDEX        = {SAMPLE_INDEX}")
    print(f"[INFO] N_VIEW_STEPS        = {N_VIEW_STEPS}")
    print(f"[INFO] DATASET_FPS         = {DATASET_FPS}")
    print(f"[INFO] ACTION_OFFSET_STEPS = {ACTION_OFFSET_STEPS}")
    print(f"[INFO] ACTION_OFFSET_SEC   = {ACTION_OFFSET_STEPS / float(DATASET_FPS):.6f}")
    print(f"[INFO] CAMERA_MAPPING_MODE = {CAMERA_MAPPING_MODE}")
    print("[INFO] DATASET_IMAGE_KEYS  = observation.images.top, observation.images.wrist")
    print("[INFO] CAMERA_MAPPING      = OBS_IMAGE_1=top, OBS_IMAGE_2=wrist, OBS_IMAGE_3=empty(side)")
    print(f"[INFO] SEED                = {SEED}")

    # 1) 加载模型
    model = SmolVLAPolicy.from_pretrained(MODEL_ID)
    model.eval()

    # 2) 用 delta_timestamps 加载数据集
    #    ACTION_OFFSET_STEPS 控制 GT action chunk 从当前帧之后偏移多少步开始取
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

    # 保存当前用于推理的这帧图像
    save_input_frame_images(sample, SAMPLE_INDEX)

    task = sample.get("task", None)
    if task is None or (isinstance(task, str) and len(task.strip()) == 0):
        task = FALLBACK_TASK

    # 如果想强制用自己的 task，取消下一行注释
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
        dataset_stats=dataset.meta.stats,
    )

    # 5) 构造离线输入
    obs_frame = build_offline_obs_frame(sample, task)

    # 6) preprocess
    obs = preprocess(obs_frame)

    # 7) 一次性预测整个 action chunk
    with torch.no_grad():
        pred_action_chunk = model.predict_action_chunk(obs)
        pred_action_chunk = postprocess(pred_action_chunk)

    if hasattr(pred_action_chunk, "actions"):
        pred_action_chunk = pred_action_chunk.actions

    if not isinstance(pred_action_chunk, torch.Tensor):
        raise TypeError(f"Unexpected pred_action_chunk type: {type(pred_action_chunk)}")

    if pred_action_chunk.ndim != 3:
        raise ValueError(f"Expected pred_action_chunk shape [B, T, D], got {tuple(pred_action_chunk.shape)}")

    pred_action_chunk = pred_action_chunk.detach().cpu()
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

    # 9) 保存调试结果
    save_debug_artifacts(
        sample=sample,
        task=task,
        pred_action_chunk_deg=pred_action_chunk_deg,
        gt_action_chunk_deg=gt_action_chunk_deg,
        valid_mask=valid_gt_mask,
        metrics=metrics,
    )

    # 10) 初始化 Rerun 并加载 URDF
    robot_urdf = init_rerun_and_load_urdf(SO100_URDF_PATH)

    # 11) 先把读入的 observation.state 作为 URDF 初始姿态记录进去
    init_state_deg = sample["observation.state"].detach().cpu().numpy()
    log_observation_state_urdf_to_rerun(
        robot_urdf=robot_urdf,
        state_deg=init_state_deg,
        step_idx=0,
    )

    time.sleep(SLEEP_PER_STEP_SEC)

    # 12) 再从 step=1 开始连续回放 predicted action chunk
    #     同时把 dataset GT chunk 和 diff 作为分组折线图写入 Rerun
    replay_action_chunk_with_urdf(
        robot_urdf=robot_urdf,
        pred_action_chunk_deg=pred_action_chunk_deg,
        gt_action_chunk_deg=gt_action_chunk_deg,
        valid_mask=valid_gt_mask,
        start_step_idx=1,
    )


if __name__ == "__main__":
    main()
