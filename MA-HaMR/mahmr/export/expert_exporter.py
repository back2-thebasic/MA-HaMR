from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from mahmr.data.dynhamr_log import (
    discover_log,
    infer_seq_len,
    load_npz,
    load_smooth_fit_params,
    load_track_info,
    pick_expert_npz,
)
from mahmr.data.schema import SequenceMeta
from mahmr.utils.io import save_json, save_torch


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.from_numpy(np.asarray(x)).cpu()


def _time_dim_index(shape: tuple, seq_len: int) -> int:
    for axis, size in enumerate(shape):
        if size == seq_len:
            return axis
    raise ValueError(f"No axis matches seq_len={seq_len} in shape={shape}")


def _align_time_dim(arr: np.ndarray, seq_len: int, name: str) -> np.ndarray:
    """Normalize to (B, T, ...) or (T, ...) with T == seq_len."""
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == seq_len:
        return arr  # (T, D)
    if arr.ndim == 3 and arr.shape[0] == seq_len and arr.shape[1:] == (3, 3):
        return arr  # (T, 3, 3) camera
    if arr.ndim >= 2:
        t_axis = _time_dim_index(arr.shape, seq_len)
        if t_axis == 0 and arr.ndim >= 3:
            return arr  # already (T, ...)
        if t_axis == 1:
            return arr  # (B, T, ...)
    raise ValueError(f"{name}: cannot align shape {arr.shape} to seq_len={seq_len}")


def _sparse_indices(seq_len: int, stride: int) -> np.ndarray:
    return np.arange(0, seq_len, stride, dtype=np.int64)


def _gather_time(arr: np.ndarray, indices: np.ndarray, seq_len: int) -> np.ndarray:
    """Gather frames along time axis. arr: (B, T, ...), (T, ...), or (T, 3, 3)."""
    if arr.ndim == 2 and arr.shape[0] == seq_len:
        return arr[indices]
    if arr.ndim == 3 and arr.shape[0] == seq_len and arr.shape[1:] == (3, 3):
        return arr[indices]
    if arr.ndim >= 3 and arr.shape[1] == seq_len:
        return arr[:, indices]
    raise ValueError(f"Cannot gather time from shape {arr.shape}, seq_len={seq_len}")


def _compute_residuals(
    expert: Dict[str, torch.Tensor],
    init: Dict[str, torch.Tensor],
    frame_indices: np.ndarray,
    seq_len: int,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    idx = frame_indices
    for key, delta_name in (
        ("trans", "delta_trans"),
        ("root_orient", "delta_root_orient"),
        ("pose_body", "delta_pose_body"),
    ):
        if key not in expert or key not in init:
            continue
        e_sparse = expert[key].numpy()
        i_dense = _align_time_dim(init[key].numpy(), seq_len, key)
        i_sparse = _gather_time(i_dense, idx, seq_len)
        if e_sparse.shape != i_sparse.shape:
            raise ValueError(
                f"{key}: expert sparse {e_sparse.shape} != init sparse {i_sparse.shape}"
            )
        out[delta_name] = _to_tensor(e_sparse - i_sparse)
    if "world_scale" in expert and "world_scale" in init:
        out["delta_world_scale"] = expert["world_scale"] - init["world_scale"]
    return out


def export_sequence(
    dynhamr_log_dir: str,
    output_dir: str,
    seq_name: Optional[str] = None,
    expert_stride: int = 10,
    expert_stage: str = "smooth_fit",
    expert_iter: Optional[int] = None,
    fps: float = 30.0,
    camera_source: str = "vipe",
) -> Dict[str, Any]:
    """
    Step 1 mini: export expert_label, init_state, valid_mask from Dyn-HaMR logs.

    Reads npz/pth only — no import from Dyn-HaMR.
    """
    paths = discover_log(dynhamr_log_dir, seq_name=seq_name)
    seq_name = paths.seq_name

    init_raw = load_npz(paths.init_npz)
    track_info = load_track_info(paths.track_info_json)
    seq_len = infer_seq_len(track_info, init_raw)

    if expert_stage != "smooth_fit":
        raise NotImplementedError(f"Only smooth_fit supported for now, got {expert_stage}")

    expert_npz_path, used_iter = pick_expert_npz(
        paths.smooth_fit_dir, seq_name, prefer_iter=expert_iter
    )
    expert_raw = load_npz(expert_npz_path)
    params = load_smooth_fit_params(paths.smooth_fit_params_pth)

    frame_indices = _sparse_indices(seq_len, expert_stride)
    valid_mask = np.zeros(seq_len, dtype=bool)
    valid_mask[frame_indices] = True

    init_state = _pack_dense_state(init_raw, params, seq_len)
    expert_label = _pack_sparse_expert(
        expert_raw, params, frame_indices, seq_len, used_iter, expert_npz_path
    )
    residuals = _compute_residuals(expert_label, init_state, frame_indices, seq_len)

    meta = SequenceMeta(
        seq_name=seq_name,
        seq_len=seq_len,
        fps=fps,
        num_hands=int(init_state.get("trans", torch.zeros(1)).shape[0]),
        expert_stride=expert_stride,
        dynhamr_log_dir=os.path.abspath(dynhamr_log_dir),
        expert_stage=expert_stage,
        expert_iter=used_iter,
        camera_source=camera_source,
        extra={
            "expert_npz": expert_npz_path,
            "init_npz": paths.init_npz,
            "hamer_npz": paths.hamer_npz,
            "num_expert_frames": int(len(frame_indices)),
        },
    )

    os.makedirs(output_dir, exist_ok=True)
    save_torch(os.path.join(output_dir, "init_state.pth"), init_state)
    save_torch(os.path.join(output_dir, "expert_label.pth"), expert_label)
    save_torch(os.path.join(output_dir, "residuals.pth"), residuals)
    save_torch(
        os.path.join(output_dir, "valid_mask.pth"),
        {"valid_mask": torch.from_numpy(valid_mask), "frame_indices": torch.from_numpy(frame_indices)},
    )
    save_json(os.path.join(output_dir, "meta.json"), meta.to_dict())

    summary = {
        "seq_name": seq_name,
        "seq_len": seq_len,
        "num_expert_frames": len(frame_indices),
        "expert_stride": expert_stride,
        "expert_iter": used_iter,
        "output_dir": os.path.abspath(output_dir),
        "init_world_scale": float(init_state["world_scale"].reshape(-1)[0]),
        "expert_world_scale": float(expert_label["world_scale"].reshape(-1)[0]),
        "delta_world_scale": float(residuals.get("delta_world_scale", torch.zeros(1)).reshape(-1)[0]),
        "shapes": {k: list(v.shape) for k, v in init_state.items() if hasattr(v, "shape")},
    }
    save_json(os.path.join(output_dir, "summary.json"), summary)
    return summary


def _read_world_scale(
    raw: Dict[str, np.ndarray],
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Read scalar world_scale from npz first, then params, else 1.0."""
    if "world_scale" in raw:
        return _to_tensor(raw["world_scale"]).reshape(1).float()
    if "world_scale" in params:
        return params["world_scale"].detach().cpu().reshape(1).float()
    return torch.ones(1)


def _pack_dense_state(
    init_raw: Dict[str, np.ndarray],
    params: Dict[str, torch.Tensor],
    seq_len: int,
) -> Dict[str, torch.Tensor]:
    time_varying = {"trans", "root_orient", "pose_body", "cam_R", "cam_t"}
    state: Dict[str, torch.Tensor] = {}
    for key in ("trans", "root_orient", "pose_body", "betas", "cam_R", "cam_t", "intrins", "is_right"):
        if key in init_raw:
            arr = init_raw[key]
            if key in time_varying:
                state[key] = _to_tensor(_align_time_dim(arr, seq_len, key))
            else:
                state[key] = _to_tensor(arr)
        elif key in params:
            val = params[key]
            if key in time_varying and hasattr(val, "shape") and val.ndim >= 2:
                t = val.shape[1] if val.shape[0] <= 4 else val.shape[0]
                if t == seq_len:
                    state[key] = _to_tensor(val)
                else:
                    state[key] = _to_tensor(val)
            else:
                state[key] = _to_tensor(val)

    # Init world scale must come from init npz (default ω=1), not smooth_fit params.
    state["world_scale"] = _read_world_scale(init_raw, params)
    return state


def _pack_sparse_expert(
    expert_raw: Dict[str, np.ndarray],
    params: Dict[str, torch.Tensor],
    frame_indices: np.ndarray,
    seq_len: int,
    used_iter: int,
    expert_npz_path: str,
) -> Dict[str, torch.Tensor]:
    label: Dict[str, torch.Tensor] = {
        "frame_indices": torch.from_numpy(frame_indices).long(),
        "expert_iter": torch.tensor([used_iter], dtype=torch.int64),
        "source_npz": expert_npz_path,
    }
    for key in ("trans", "root_orient", "pose_body", "betas", "cam_R", "cam_t", "intrins"):
        if key not in expert_raw:
            if key in params:
                val = _to_tensor(params[key])
                if key in ("trans", "root_orient", "pose_body", "cam_R", "cam_t") and val.ndim >= 2:
                    t = val.shape[1] if val.shape[0] <= 4 else val.shape[0]
                    if t == seq_len:
                        label[key] = val[:, frame_indices] if val.shape[0] <= 4 else val[frame_indices]
                    else:
                        label[key] = val
                else:
                    label[key] = val
            continue
        arr = expert_raw[key]
        if key in ("trans", "root_orient", "pose_body", "cam_R", "cam_t"):
            arr = _align_time_dim(arr, seq_len, key)
            label[key] = _to_tensor(_gather_time(arr, frame_indices, seq_len))
        else:
            label[key] = _to_tensor(arr)

    # Expert world scale from smooth_fit npz (optimized ω), not init.
    label["world_scale"] = _read_world_scale(expert_raw, params)

    return label
