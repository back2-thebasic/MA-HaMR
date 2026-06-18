"""Read Dyn-HaMR optimization logs without importing dyn-hamr."""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class DynHaMRLogPaths:
    log_dir: str
    seq_name: str
    init_npz: str
    hamer_npz: str
    track_info_json: str
    cameras_json: str
    smooth_fit_dir: str
    smooth_fit_params_pth: str


def discover_log(log_dir: str, seq_name: Optional[str] = None) -> DynHaMRLogPaths:
    """Discover standard Dyn-HaMR output files under a hydra run directory."""
    log_dir = os.path.abspath(log_dir)
    if seq_name is None:
        seq_name = _guess_seq_name(log_dir)

    init_dir = os.path.join(log_dir, "init")
    hamer_dir = os.path.join(log_dir, "hamer")
    smooth_dir = os.path.join(log_dir, "smooth_fit")

    init_npz = _single_or_glob(
        init_dir, f"{seq_name}_*_init_world_results.npz"
    )
    hamer_npz = _single_or_glob(
        hamer_dir, f"{seq_name}_*_phalp_world_results.npz"
    )
    track_info = os.path.join(log_dir, "track_info.json")
    cameras_json = os.path.join(log_dir, "cameras.json")
    params_pth = os.path.join(log_dir, "smooth_fit_params.pth")

    for p in (init_npz, hamer_npz, track_info):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required Dyn-HaMR artifact: {p}")

    return DynHaMRLogPaths(
        log_dir=log_dir,
        seq_name=seq_name,
        init_npz=init_npz,
        hamer_npz=hamer_npz,
        track_info_json=track_info,
        cameras_json=cameras_json if os.path.isfile(cameras_json) else "",
        smooth_fit_dir=smooth_dir,
        smooth_fit_params_pth=params_pth if os.path.isfile(params_pth) else "",
    )


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def load_world_results_npz(path: str) -> Dict[str, np.ndarray]:
    return load_npz(path)


def pick_expert_npz(
    smooth_fit_dir: str,
    seq_name: str,
    prefer_iter: Optional[int] = None,
) -> Tuple[str, int]:
    """Pick the latest (or requested) smooth_fit world_results npz."""
    pattern = os.path.join(smooth_fit_dir, f"{seq_name}_*_world_results.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No expert npz under {pattern}")

    if prefer_iter is not None:
        target = f"{seq_name}_{prefer_iter:06d}_world_results.npz"
        for f in files:
            if os.path.basename(f) == target:
                return f, prefer_iter
        raise FileNotFoundError(f"Requested iter {prefer_iter} not in {smooth_fit_dir}")

    best_path, best_iter = files[-1], -1
    for f in files:
        m = re.search(rf"{re.escape(seq_name)}_(\d+)_world_results\.npz", os.path.basename(f))
        if m:
            it = int(m.group(1))
            if it >= best_iter:
                best_iter = it
                best_path = f
    return best_path, best_iter


def load_smooth_fit_params(pth_path: str) -> Dict[str, torch.Tensor]:
    if not pth_path or not os.path.isfile(pth_path):
        return {}
    obj = torch.load(pth_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {pth_path}, got {type(obj)}")
    return obj


def load_track_info(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_seq_len(track_info: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> int:
    for key in ("trans", "root_orient", "pose_body", "cam_R"):
        if key not in arrays:
            continue
        arr = arrays[key]
        if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[-2] == 3:
            return int(arr.shape[0])  # (T, 3, 3)
        if arr.ndim >= 2:
            return int(arr.shape[1] if arr.shape[0] <= 4 else arr.shape[0])
    if "meta" in track_info and "data_interval" in track_info["meta"]:
        start, end = track_info["meta"]["data_interval"]
        return int(end - start + 1)
    raise ValueError("Cannot infer seq_len from track_info or arrays")


def _guess_seq_name(log_dir: str) -> str:
    for sub in ("init", "hamer", "smooth_fit"):
        d = os.path.join(log_dir, sub)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = re.match(r"(\w+)_\d+_", f)
            if m:
                return m.group(1)
    raise ValueError(f"Cannot guess seq_name from {log_dir}")


def _single_or_glob(directory: str, pattern: str) -> str:
    hits = sorted(glob.glob(os.path.join(directory, pattern)))
    if not hits:
        raise FileNotFoundError(f"No match for {pattern} in {directory}")
    return hits[0]
