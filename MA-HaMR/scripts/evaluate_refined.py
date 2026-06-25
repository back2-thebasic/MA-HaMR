#!/usr/bin/env python3
"""Evaluate and visualize MA-HaMR refined predictions against init/expert."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mahmr.geometry import MANOJointLayer
from mahmr.loss import _project_points, build_window_residual_targets
from mahmr.models.refinement import MAHaMRRefiner
from mahmr.models.safeguard import reprojection_safeguard
from mahmr.utils.io import load_torch, load_yaml, save_json


SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MA-HaMR refined checkpoint")
    parser.add_argument("--config", default="configs/train/h2o_gpu3_mano_geometry_3000.yaml")
    parser.add_argument("--checkpoint", default="/extra/SuC/experiments/mahmr/h2o_gpu3_mano_geometry_3000/last.pt")
    parser.add_argument("--output-dir", default="/extra/SuC/experiments/mahmr/eval_h2o_gpu3_mano_geometry_3000")
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--features-root", default=None)
    parser.add_argument("--hamer-out-root", default="/extra/SuC/dynhamr_io/dynhamr/hamer_out")
    parser.add_argument("--device", default=None)
    parser.add_argument("--memory-mode", choices=["none", "teacher"], default="none")
    parser.add_argument("--sequence", action="append", dest="sequences", default=None)
    parser.add_argument("--num-viz-frames", type=int, default=7)
    parser.add_argument(
        "--safeguard",
        choices=["on", "off"],
        default="on",
        help="Reprojection self-verification gate that reverts refined frames to init when they disagree with kp_2d.",
    )
    parser.add_argument("--safeguard-margin", type=float, default=0.1)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(args.device or cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    processed_root = args.processed_root or cfg["dataset"]["processed_root"]
    features_root = args.features_root or cfg["dataset"]["features_root"]
    sequences = args.sequences or cfg["dataset"].get("sequence_names") or []

    os.makedirs(args.output_dir, exist_ok=True)
    model = _load_model(cfg, args.checkpoint, device)
    mano = MANOJointLayer(
        model_path=cfg.get("geometry", {}).get("mano_model_path", "/data/SuC/Dyn-HaMR/_DATA/data/mano"),
        flat_hand_mean=bool(cfg.get("geometry", {}).get("flat_hand_mean", False)),
    ).to(device).eval()

    rows = []
    summaries = []
    for seq in sequences:
        rec = _load_sequence(processed_root, features_root, seq, device)
        result = _run_model(model, rec, device, memory_mode=args.memory_mode)
        if args.safeguard == "on":
            result = _apply_safeguard(result, rec, mano, device, args.safeguard_margin)
        metrics = _evaluate_sequence(seq, rec, result, mano, device)
        summaries.append(metrics)
        rows.append(_flatten_summary(metrics))
        _write_visualizations(
            seq,
            rec,
            result,
            mano,
            args.hamer_out_root,
            os.path.join(args.output_dir, "visualizations"),
            args.num_viz_frames,
            device,
        )

    aggregate = _aggregate(summaries)
    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "config": os.path.abspath(args.config),
        "memory_mode": args.memory_mode,
        "sequences": summaries,
        "aggregate": aggregate,
        "notes": {
            "memory_mode=none": "Strict student inference: memory starts empty and no expert residuals are written.",
            "memory_mode=teacher": "Teacher-forced upper-bound: sparse expert residuals are written to memory, matching training.",
            "scale_error": "Scale comparison uses absolute world_scale because the refiner enforces positive scale.",
        },
    }
    save_json(os.path.join(args.output_dir, f"summary_{args.memory_mode}.json"), summary)
    _write_csv(os.path.join(args.output_dir, f"metrics_{args.memory_mode}.csv"), rows)
    print(json.dumps(summary, indent=2))


def _load_model(cfg: Dict[str, Any], checkpoint_path: str, device: torch.device) -> MAHaMRRefiner:
    model = MAHaMRRefiner(**cfg.get("model", {})).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _load_sequence(processed_root: str, features_root: str, seq: str, device: torch.device) -> Dict[str, Any]:
    processed_dir = os.path.join(processed_root, seq)
    features_dir = os.path.join(features_root, seq)
    init_state = load_torch(os.path.join(processed_dir, "init_state.pth"))
    expert_label = load_torch(os.path.join(processed_dir, "expert_label.pth"))
    residuals = load_torch(os.path.join(processed_dir, "residuals.pth"))
    valid_mask = load_torch(os.path.join(processed_dir, "valid_mask.pth"))
    meta = _load_json(os.path.join(processed_dir, "meta.json"))
    features = {
        "mano_local_init": load_torch(os.path.join(features_dir, "mano_local_init.pth"))["mano_local_init"],
        "kp_2d": load_torch(os.path.join(features_dir, "kp_2d.pth"))["kp_2d"],
        "img_feat": load_torch(os.path.join(features_dir, "img_feat.pth"))["img_feat"],
        "cam_init": load_torch(os.path.join(features_dir, "cam_init.pth")),
        "uncertainty": load_torch(os.path.join(features_dir, "uncertainty.pth"))["uncertainty"],
    }
    batch = {
        "seq_name": seq,
        "start": 0,
        "end": int(meta["seq_len"]),
        "meta": meta,
        "init_state": init_state,
        "expert_label": expert_label,
        "residuals": residuals,
        "valid_mask": valid_mask,
        "features": features,
    }
    return _to_device(batch, device)


def _run_model(
    model: MAHaMRRefiner,
    rec: Dict[str, Any],
    device: torch.device,
    *,
    memory_mode: str,
) -> Dict[str, Any]:
    targets = build_window_residual_targets(rec, device=device)
    kwargs: Dict[str, Any] = {"reset_memory": True, "return_attention": False}
    if memory_mode == "teacher":
        kwargs["memory_values"] = targets["packed_residual"]
        kwargs["write_mask"] = targets["valid_mask"]
    with torch.no_grad():
        output = model(rec["features"], **kwargs)
    output["targets"] = targets
    return output


def _apply_safeguard(
    output: Dict[str, Any],
    rec: Dict[str, Any],
    mano: MANOJointLayer,
    device: torch.device,
    margin: float,
) -> Dict[str, Any]:
    features = rec["features"]
    init_state = rec["init_state"]
    mano_init = features["mano_local_init"].unsqueeze(0).to(device).float()
    cam_init = _to_device(features["cam_init"], device)
    init_scale = _normalize_world_scale(cam_init.get("world_scale"), 1, mano_init.shape[2], device)
    kp = features["kp_2d"].unsqueeze(0).to(device).float()
    is_right = init_state.get("is_right")
    if is_right is not None:
        is_right = is_right.unsqueeze(0).to(device)
    gated = reprojection_safeguard(
        output,
        mano_local_init=mano_init,
        init_world_scale=init_scale,
        kp_2d=kp,
        cam_init=cam_init,
        mano_layer=mano,
        project_fn=_project_points,
        is_right=is_right,
        margin=margin,
    )
    gated["targets"] = output["targets"]
    return gated


def _evaluate_sequence(
    seq: str,
    rec: Dict[str, Any],
    output: Dict[str, Any],
    mano: MANOJointLayer,
    device: torch.device,
) -> Dict[str, Any]:
    features = rec["features"]
    init_state = rec["init_state"]
    expert = rec["expert_label"]
    targets = output["targets"]
    kp = features["kp_2d"].unsqueeze(0).to(device).float()
    is_right = init_state.get("is_right")
    if is_right is not None:
        is_right = is_right.unsqueeze(0).to(device)

    mano_init = features["mano_local_init"].unsqueeze(0).to(device).float()
    cam_init = _to_device(features["cam_init"], device)
    init_scale = _normalize_world_scale(cam_init.get("world_scale"), 1, mano_init.shape[2], device)

    with torch.no_grad():
        init_joints = mano(mano_init, is_right=is_right)
        refined_joints = mano(output["mano_local_refined"], is_right=is_right)
        init_xy = _project_points(init_joints, cam_init, init_scale)
        refined_xy = _project_points(refined_joints, cam_init, output["world_scale_refined"])

    init_reproj = _reproj_stats(init_xy[0], kp[0])
    refined_reproj = _reproj_stats(refined_xy[0], kp[0])

    mask = targets["valid_mask"].bool()
    target = targets["packed_residual"]
    pred = output["packed_residual"]
    init_packed = torch.zeros_like(target)
    init_residual_err = _packed_residual_stats(init_packed, target, mask)
    refined_residual_err = _packed_residual_stats(pred, target, mask)
    scale_stats = _scale_stats(rec, output, targets, mask)
    expert_reproj = _expert_reprojection_stats(expert, init_state, features, mano, device)

    return {
        "seq": seq,
        "num_frames": int(mano_init.shape[2]),
        "num_expert_frames": int(mask.sum().item()),
        "init": {
            "reprojection": init_reproj,
            "residual_to_expert": init_residual_err,
        },
        "refined": {
            "reprojection": refined_reproj,
            "residual_to_expert": refined_residual_err,
            "scale": scale_stats,
        },
        "expert": {
            "reprojection": expert_reproj,
        },
        "improvement": {
            "reprojection_mean_px_delta": init_reproj["mean_px"] - refined_reproj["mean_px"],
            "reprojection_mean_px_ratio": _safe_ratio(refined_reproj["mean_px"], init_reproj["mean_px"]),
            "mano_residual_mae_delta": init_residual_err["mano_mae"] - refined_residual_err["mano_mae"],
            "mano_residual_mae_ratio": _safe_ratio(refined_residual_err["mano_mae"], init_residual_err["mano_mae"]),
            "scale_abs_error_delta": scale_stats["init_abs_error_mean"] - scale_stats["refined_abs_error_mean"],
            "scale_abs_error_ratio": _safe_ratio(scale_stats["refined_abs_error_mean"], scale_stats["init_abs_error_mean"]),
        },
    }


def _reproj_stats(xy: torch.Tensor, kp: torch.Tensor) -> Dict[str, float]:
    conf = kp[..., 2] > 0.1
    err = torch.linalg.norm(xy - kp[..., :2], dim=-1)
    vals = err[conf]
    if vals.numel() == 0:
        return {"mean_px": float("nan"), "median_px": float("nan"), "p90_px": float("nan"), "num_points": 0}
    return {
        "mean_px": float(vals.mean().detach().cpu()),
        "median_px": float(vals.median().detach().cpu()),
        "p90_px": float(vals.quantile(0.9).detach().cpu()),
        "num_points": int(vals.numel()),
    }


def _packed_residual_stats(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    if not torch.any(mask):
        return {"mano_mae": float("nan"), "mano_rmse": float("nan"), "scale_delta_mae": float("nan")}
    diff = pred[mask] - target[mask]
    mano = diff[..., :-1]
    scale = diff[..., -1]
    return {
        "mano_mae": float(mano.abs().mean().detach().cpu()),
        "mano_rmse": float(mano.pow(2).mean().sqrt().detach().cpu()),
        "scale_delta_mae": float(scale.abs().mean().detach().cpu()),
    }


def _scale_stats(
    rec: Dict[str, Any],
    output: Dict[str, Any],
    targets: Dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> Dict[str, float]:
    init_scale = rec["features"]["cam_init"].get("world_scale")
    init_scale = _normalize_world_scale(init_scale, output["world_scale_refined"].shape[0], output["world_scale_refined"].shape[1], output["world_scale_refined"].device)
    target_log = targets["target_log_abs_world_scale"]
    target_abs = torch.exp(target_log)
    refined = output["world_scale_refined"]
    init_err = (init_scale[mask] - target_abs[mask]).abs()
    refined_err = (refined[mask] - target_abs[mask]).abs()
    return {
        "init_abs_error_mean": float(init_err.mean().detach().cpu()) if init_err.numel() else float("nan"),
        "refined_abs_error_mean": float(refined_err.mean().detach().cpu()) if refined_err.numel() else float("nan"),
        "refined_mean": float(refined.mean().detach().cpu()),
        "refined_min": float(refined.min().detach().cpu()),
        "refined_max": float(refined.max().detach().cpu()),
    }


def _expert_reprojection_stats(
    expert: Dict[str, torch.Tensor],
    init_state: Dict[str, torch.Tensor],
    features: Dict[str, Any],
    mano: MANOJointLayer,
    device: torch.device,
) -> Dict[str, float]:
    frame_indices = torch.as_tensor(expert["frame_indices"], device=device).long()
    root = expert["root_orient"].to(device).float()
    pose = expert["pose_body"].to(device).float().reshape(root.shape[0], root.shape[1], -1)
    trans = expert["trans"].to(device).float()
    betas = expert["betas"].to(device).float()[:, None].expand(-1, root.shape[1], -1)
    mano_local = torch.cat([root, pose, betas, trans], dim=-1).unsqueeze(0)
    is_right = init_state["is_right"][:, frame_indices].unsqueeze(0).to(device)
    cam_t = expert["cam_t"].to(device).float().unsqueeze(0)
    cam = {
        "cam_R": expert["cam_R"].to(device).float().unsqueeze(0),
        "cam_t": cam_t,
        "intrins": expert["intrins"].to(device).float(),
        # Expert cam_t is already scaled by Dyn-HaMR get_extrinsics().
        "world_scale": torch.ones(cam_t.shape[0], cam_t.shape[2], 1, device=device),
    }
    kp = features["kp_2d"][:, frame_indices].unsqueeze(0).to(device).float()
    with torch.no_grad():
        joints = mano(mano_local, is_right=is_right)
        xy = _project_points(joints, cam, cam["world_scale"])
    return _reproj_stats(xy[0], kp[0])


def _write_visualizations(
    seq: str,
    rec: Dict[str, Any],
    output: Dict[str, Any],
    mano: MANOJointLayer,
    hamer_out_root: str,
    output_root: str,
    num_frames: int,
    device: torch.device,
) -> None:
    seq_out = os.path.join(output_root, seq)
    os.makedirs(seq_out, exist_ok=True)
    frames = torch.as_tensor(rec["expert_label"]["frame_indices"]).long().tolist()
    frames = frames[: max(0, num_frames)]
    if not frames:
        return
    image_paths = _frame_paths(hamer_out_root, seq)
    projections = _all_visual_projection_sources(rec, output, mano, device)
    records = []
    for frame in frames:
        if frame >= len(image_paths):
            continue
        image = cv2.imread(image_paths[frame], cv2.IMREAD_COLOR)
        if image is None:
            continue
        gt = rec["features"]["kp_2d"][:, frame].detach().cpu().numpy()
        panels = []
        for name in ("init", "refined", "expert"):
            pred = projections[name][:, frame].detach().cpu().numpy()
            panel = _draw_overlay(image, pred, gt, title=name)
            panels.append(panel)
        canvas = np.concatenate(panels, axis=1)
        out_path = os.path.join(seq_out, f"{seq}_{frame:06d}_init_refined_expert.jpg")
        cv2.imwrite(out_path, canvas)
        records.append({"frame_idx": int(frame), "path": out_path})
    save_json(os.path.join(seq_out, "visualization_summary.json"), {"seq": seq, "frames": records})


def _all_visual_projection_sources(
    rec: Dict[str, Any],
    output: Dict[str, Any],
    mano: MANOJointLayer,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    features = rec["features"]
    init_state = rec["init_state"]
    expert = rec["expert_label"]
    is_right = init_state["is_right"].unsqueeze(0).to(device)
    mano_init = features["mano_local_init"].unsqueeze(0).to(device).float()
    cam_init = _to_device(features["cam_init"], device)
    init_scale = _normalize_world_scale(cam_init.get("world_scale"), 1, mano_init.shape[2], device)
    with torch.no_grad():
        init_xy = _project_points(mano(mano_init, is_right=is_right), cam_init, init_scale)[0]
        refined_xy = _project_points(mano(output["mano_local_refined"], is_right=is_right), cam_init, output["world_scale_refined"])[0]

    expert_xy_full = torch.full_like(init_xy, float("nan"))
    frame_indices = torch.as_tensor(expert["frame_indices"], device=device).long()
    root = expert["root_orient"].to(device).float()
    pose = expert["pose_body"].to(device).float().reshape(root.shape[0], root.shape[1], -1)
    trans = expert["trans"].to(device).float()
    betas = expert["betas"].to(device).float()[:, None].expand(-1, root.shape[1], -1)
    expert_mano = torch.cat([root, pose, betas, trans], dim=-1).unsqueeze(0)
    expert_is_right = init_state["is_right"][:, frame_indices].unsqueeze(0).to(device)
    expert_cam_t = expert["cam_t"].to(device).float().unsqueeze(0)
    expert_cam = {
        "cam_R": expert["cam_R"].to(device).float().unsqueeze(0),
        "cam_t": expert_cam_t,
        "intrins": expert["intrins"].to(device).float(),
        "world_scale": torch.ones(expert_cam_t.shape[0], expert_cam_t.shape[2], 1, device=device),
    }
    with torch.no_grad():
        expert_sparse_xy = _project_points(
            mano(expert_mano, is_right=expert_is_right),
            expert_cam,
            expert_cam["world_scale"],
        )[0]
    expert_xy_full[:, frame_indices] = expert_sparse_xy
    return {"init": init_xy, "refined": refined_xy, "expert": expert_xy_full}


def _draw_overlay(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, *, title: str) -> np.ndarray:
    canvas = image.copy()
    colors = [(0, 0, 255), (255, 0, 0)]
    for h in range(pred.shape[0]):
        _draw_hand(canvas, gt[h, :, :2], gt[h, :, 2], (0, 255, 0), radius=3)
        _draw_hand(canvas, pred[h], np.ones(21), colors[h % len(colors)], radius=2)
        err = _valid_errors(pred[h], gt[h])
        text = f"{title} h{h}: {float(err.mean()):.1f}px" if err.size else f"{title} h{h}: no kp"
        cv2.putText(canvas, text, (20, 35 + 25 * h), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    cv2.putText(canvas, "green=kp_2d red/blue=MANO", (20, canvas.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return canvas


def _draw_hand(image: np.ndarray, points: np.ndarray, conf: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
    valid = conf > 0.1
    for a, b in SKELETON:
        if valid[a] and valid[b] and _reasonable(points[a]) and _reasonable(points[b]):
            cv2.line(image, tuple(points[a].round().astype(int)), tuple(points[b].round().astype(int)), color, 2)
    for idx, xy in enumerate(points):
        if valid[idx] and _reasonable(xy):
            cv2.circle(image, tuple(xy.round().astype(int)), radius, color, -1)


def _valid_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    valid = gt[:, 2] > 0.1
    valid = valid & np.isfinite(pred).all(axis=-1)
    if not np.any(valid):
        return np.zeros((0,), dtype=np.float32)
    return np.linalg.norm(pred[valid] - gt[valid, :2], axis=-1)


def _reasonable(xy: np.ndarray) -> bool:
    return np.isfinite(xy).all() and abs(float(xy[0])) < 10000 and abs(float(xy[1])) < 10000


def _frame_paths(hamer_out_root: str, seq: str) -> List[str]:
    pkl_path = os.path.join(hamer_out_root, seq, f"{seq}.pkl")
    if os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return sorted(data.keys(), key=lambda p: int(Path(p).stem) if Path(p).stem.isdigit() else Path(p).stem)
    image_dir = f"/extra/SuC/dynhamr_io/images/{seq}"
    return [os.path.join(image_dir, name) for name in sorted(os.listdir(image_dir)) if name.lower().endswith((".jpg", ".png", ".jpeg"))]


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


def _flatten_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    row = {"seq": summary["seq"]}
    for group in ("init", "refined", "expert"):
        for name, metrics in summary[group].items():
            for key, value in metrics.items():
                row[f"{group}_{name}_{key}"] = value
    for key, value in summary["improvement"].items():
        row[f"improvement_{key}"] = value
    return row


def _aggregate(summaries: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = [
        "reprojection_mean_px_delta",
        "reprojection_mean_px_ratio",
        "mano_residual_mae_delta",
        "mano_residual_mae_ratio",
        "scale_abs_error_delta",
        "scale_abs_error_ratio",
    ]
    out = {}
    for key in keys:
        vals = [s["improvement"][key] for s in summaries if np.isfinite(s["improvement"][key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and abs(den) > 1e-12 else float("nan")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


if __name__ == "__main__":
    main()
