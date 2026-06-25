#!/usr/bin/env python3
"""Visual sanity check for MANO joint projection against dumped kp_2d."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mahmr.geometry import MANOJointLayer
from mahmr.loss import _project_points
from mahmr.utils.io import load_torch, save_json


SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay MANO projection and kp_2d")
    parser.add_argument("--seq", default="h2o_s1_h1")
    parser.add_argument("--frames", nargs="*", type=int, default=[0, 50, 100, 150, 200, 250, 300])
    parser.add_argument("--processed-root", default="/extra/SuC/data/mahmr/processed")
    parser.add_argument("--features-root", default="/extra/SuC/data/mahmr/features")
    parser.add_argument("--hamer-out-root", default="/extra/SuC/dynhamr_io/dynhamr/hamer_out")
    parser.add_argument("--mano-model-path", default="/data/SuC/Dyn-HaMR/_DATA/data/mano")
    parser.add_argument("--output-dir", default="/extra/SuC/experiments/mahmr/geometry_check")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--source", choices=["init", "expert"], default="init")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    seq_out = os.path.join(args.output_dir, args.seq)
    os.makedirs(seq_out, exist_ok=True)

    processed_dir = os.path.join(args.processed_root, args.seq)
    features_dir = os.path.join(args.features_root, args.seq)
    init_state = load_torch(os.path.join(processed_dir, "init_state.pth"))
    kp_2d = load_torch(os.path.join(features_dir, "kp_2d.pth"))["kp_2d"].unsqueeze(0).to(device)
    if args.source == "expert":
        expert_label = load_torch(os.path.join(processed_dir, "expert_label.pth"))
        mano_local, cam_init, is_right, source_frames = _build_expert_source(expert_label, init_state, device)
        kp_2d = kp_2d[:, :, source_frames]
        frame_lookup = source_frames
    else:
        mano_local = load_torch(os.path.join(features_dir, "mano_local_init.pth"))["mano_local_init"].unsqueeze(0).to(device)
        cam_init = _to_device(load_torch(os.path.join(features_dir, "cam_init.pth")), device)
        is_right = init_state.get("is_right")
        if is_right is not None:
            is_right = is_right.unsqueeze(0).to(device)
        frame_lookup = list(range(mano_local.shape[2]))

    mano = MANOJointLayer(model_path=args.mano_model_path).to(device).eval()
    with torch.no_grad():
        joints = mano(mano_local, is_right=is_right)
        world_scale = _normalize_world_scale(cam_init.get("world_scale"), joints.shape[0], joints.shape[2], device)
        pred_xy = _project_points(joints, cam_init, world_scale)

    image_paths = _frame_paths(args.hamer_out_root, args.seq)
    if args.source == "expert" and args.frames == [0, 50, 100, 150, 200, 250, 300]:
        frames = list(range(min(7, joints.shape[2])))
    else:
        frame_to_local = {frame: idx for idx, frame in enumerate(frame_lookup)}
        frames = [frame_to_local.get(f, f) for f in args.frames]
        frames = [f for f in frames if 0 <= f < joints.shape[2]]
    records = []
    for frame_idx in frames:
        global_frame = frame_lookup[frame_idx]
        image = cv2.imread(image_paths[global_frame], cv2.IMREAD_COLOR)
        if image is None:
            continue
        pred = pred_xy[0, :, frame_idx].detach().cpu().numpy()
        gt = kp_2d[0, :, frame_idx].detach().cpu().numpy()
        rec = _frame_errors(args.seq, int(global_frame), pred, gt)
        records.append(rec)
        canvas = draw_overlay(image, pred, gt)
        out_path = os.path.join(seq_out, f"{args.source}_{args.seq}_{int(global_frame):06d}.jpg")
        cv2.imwrite(out_path, canvas)

    summary = {
        "seq": args.seq,
        "source": args.source,
        "num_frames_checked": len(records),
        "mean_error_px": float(np.mean([r["mean_error_px"] for r in records])) if records else None,
        "median_error_px": float(np.median([r["median_error_px"] for r in records])) if records else None,
        "records": records,
        "output_dir": seq_out,
        "legend": {
            "green": "kp_2d detector keypoints",
            "red": "projected MANO joints",
            "cyan_text": "frame/hand pixel errors",
        },
    }
    save_json(os.path.join(seq_out, "summary.json"), summary)
    print(json.dumps(summary, indent=2))


def draw_overlay(image: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    colors = [(0, 0, 255), (255, 0, 0)]
    for h in range(pred.shape[0]):
        _draw_hand(canvas, gt[h, :, :2], gt[h, :, 2], (0, 255, 0), radius=3)
        _draw_hand(canvas, pred[h], np.ones(21), colors[h % len(colors)], radius=2)
        err = _valid_errors(pred[h], gt[h])
        text = f"hand{h}: {float(err.mean()):.1f}px" if err.size else f"hand{h}: no kp"
        cv2.putText(canvas, text, (20, 35 + 25 * h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(canvas, "green=kp_2d red/blue=MANO projection", (20, canvas.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


def _draw_hand(image: np.ndarray, points: np.ndarray, conf: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
    valid = conf > 0.1
    for a, b in SKELETON:
        if valid[a] and valid[b] and _in_reasonable_range(points[a]) and _in_reasonable_range(points[b]):
            cv2.line(image, tuple(points[a].round().astype(int)), tuple(points[b].round().astype(int)), color, 2)
    for idx, xy in enumerate(points):
        if valid[idx] and _in_reasonable_range(xy):
            cv2.circle(image, tuple(xy.round().astype(int)), radius, color, -1)


def _frame_errors(seq: str, frame_idx: int, pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
    hand_errors = []
    all_err = []
    for h in range(pred.shape[0]):
        err = _valid_errors(pred[h], gt[h])
        hand_errors.append(float(err.mean()) if err.size else None)
        if err.size:
            all_err.extend(err.tolist())
    return {
        "seq": seq,
        "frame_idx": int(frame_idx),
        "mean_error_px": float(np.mean(all_err)) if all_err else None,
        "median_error_px": float(np.median(all_err)) if all_err else None,
        "hand_mean_errors_px": hand_errors,
    }


def _valid_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    valid = gt[:, 2] > 0.1
    if not np.any(valid):
        return np.zeros((0,), dtype=np.float32)
    return np.linalg.norm(pred[valid] - gt[valid, :2], axis=-1)


def _in_reasonable_range(xy: np.ndarray) -> bool:
    return np.isfinite(xy).all() and abs(float(xy[0])) < 10000 and abs(float(xy[1])) < 10000


def _frame_paths(hamer_out_root: str, seq: str) -> List[str]:
    pkl_path = os.path.join(hamer_out_root, seq, f"{seq}.pkl")
    if os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return sorted(data.keys(), key=lambda p: int(Path(p).stem) if Path(p).stem.isdigit() else Path(p).stem)
    image_dir = f"/extra/SuC/dynhamr_io/images/{seq}"
    return [os.path.join(image_dir, name) for name in sorted(os.listdir(image_dir)) if name.lower().endswith((".jpg", ".png", ".jpeg"))]


def _build_expert_source(
    expert_label: Dict[str, torch.Tensor],
    init_state: Dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, list[int]]:
    frame_indices = torch.as_tensor(expert_label["frame_indices"]).long().tolist()
    root = expert_label["root_orient"].to(device).float()
    pose = expert_label["pose_body"].to(device).float().reshape(root.shape[0], root.shape[1], -1)
    trans = expert_label["trans"].to(device).float()
    betas = expert_label["betas"].to(device).float()[:, None].expand(-1, root.shape[1], -1)
    mano_local = torch.cat([root, pose, betas, trans], dim=-1).unsqueeze(0)
    is_right = init_state["is_right"][:, frame_indices].unsqueeze(0).to(device)
    # Dyn-HaMR saves cam_t from get_extrinsics() == raw_cam_t * world_scale, so
    # the exported expert cam_t ALREADY includes world_scale. Projecting with an
    # extra world_scale factor would double-apply it (this was the source of the
    # spuriously large ~110-148px "expert" reprojection error). Use ones here.
    cam_t = expert_label["cam_t"].to(device).float().unsqueeze(0)
    cam_init = {
        "cam_R": expert_label["cam_R"].to(device).float().unsqueeze(0),
        "cam_t": cam_t,
        "intrins": expert_label["intrins"].to(device).float(),
        "world_scale": torch.ones(cam_t.shape[0], cam_t.shape[2], 1, device=device),
    }
    return mano_local, cam_init, is_right, frame_indices


def _normalize_world_scale(scale: torch.Tensor | None, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    if scale is None:
        return torch.ones(batch_size, seq_len, 1, device=device)
    scale = torch.as_tensor(scale, device=device).float()
    if scale.ndim == 0:
        scale = scale.view(1, 1, 1)
    elif scale.ndim == 1:
        scale = scale.view(1, 1, -1)
    elif scale.ndim == 2:
        scale = scale.unsqueeze(-1)
    if scale.shape[0] == 1 and batch_size > 1:
        scale = scale.expand(batch_size, -1, -1)
    if scale.shape[1] == 1 and seq_len > 1:
        scale = scale.expand(batch_size, seq_len, 1)
    return scale[..., :1]


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


if __name__ == "__main__":
    main()
