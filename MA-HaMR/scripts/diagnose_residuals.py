#!/usr/bin/env python3
"""Decompose init/refined vs expert errors per MANO component on expert frames."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mahmr.loss import build_window_residual_targets
from mahmr.models.refinement import MAHaMRRefiner
from mahmr.utils.io import load_torch, load_yaml

# mano_local layout (per hand, 61 dims)
SLICES = {
    "root_orient": slice(0, 3),
    "pose_body": slice(3, 48),
    "betas": slice(48, 58),
    "trans": slice(58, 61),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/h2o_gpu3_mano_geometry_3000.yaml")
    ap.add_argument("--checkpoint", default="/extra/SuC/experiments/mahmr/h2o_gpu3_mano_geometry_3000/last.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--memory-mode", choices=["none", "teacher"], default="teacher")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(args.device or cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    processed_root = cfg["dataset"]["processed_root"]
    features_root = cfg["dataset"]["features_root"]
    sequences = cfg["dataset"]["sequence_names"]

    model = MAHaMRRefiner(**cfg.get("model", {})).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    report: Dict[str, Any] = {}
    for seq in sequences:
        rec = _load_sequence(processed_root, features_root, seq, device)
        targets = build_window_residual_targets(rec, device=device)
        kwargs: Dict[str, Any] = {"reset_memory": True}
        if args.memory_mode == "teacher":
            kwargs["memory_values"] = targets["packed_residual"]
            kwargs["write_mask"] = targets["valid_mask"]
        with torch.no_grad():
            out = model(rec["features"], **kwargs)

        mask = targets["valid_mask"].bool()  # (1, T)
        num_hands = cfg["model"].get("num_hands", 2)
        # refined/init mano_local: (1, H, T, 61) ; expert residual target packed (1, T, 2*61+1)
        mano_init = rec["features"]["mano_local_init"].unsqueeze(0).to(device).float()
        mano_refined = out["mano_local_refined"]
        target_packed = targets["packed_residual"]  # init + this = expert state
        # reconstruct expert mano_local on expert frames
        tgt_hand = target_packed[..., : num_hands * 61].view(1, -1, num_hands, 61).permute(0, 2, 1, 3)
        expert_state = mano_init + tgt_hand  # only valid where mask

        seq_rep: Dict[str, Any] = {"num_expert_frames": int(mask.sum().item())}
        # per-component state error on expert frames
        m = mask[0]  # (T,)
        for comp, sl in SLICES.items():
            init_c = mano_init[0, :, m, sl]
            ref_c = mano_refined[0, :, m, sl]
            exp_c = expert_state[0, :, m, sl]
            init_err = (init_c - exp_c).abs().mean().item()
            ref_err = (ref_c - exp_c).abs().mean().item()
            seq_rep[comp] = {
                "init_err": round(init_err, 5),
                "refined_err": round(ref_err, 5),
                "ratio": round(ref_err / init_err, 4) if init_err > 1e-9 else None,
                "target_mag": round((exp_c - init_c).abs().mean().item(), 5),
            }
        # scale
        tgt_scale_abs = torch.exp(targets["target_log_abs_world_scale"])  # (1,T,1)
        init_scale = _norm_scale(rec["features"]["cam_init"].get("world_scale"), out["world_scale_refined"].shape[1], device)
        ref_scale = out["world_scale_refined"]
        init_s_err = (init_scale[0, m] - tgt_scale_abs[0, m]).abs().mean().item()
        ref_s_err = (ref_scale[0, m] - tgt_scale_abs[0, m]).abs().mean().item()
        seq_rep["world_scale"] = {
            "init_err": round(init_s_err, 5),
            "refined_err": round(ref_s_err, 5),
            "ratio": round(ref_s_err / init_s_err, 4) if init_s_err > 1e-9 else None,
            "target_abs": round(tgt_scale_abs[0, m].mean().item(), 5),
            "refined_mean": round(ref_scale[0, m].mean().item(), 5),
        }
        report[seq] = seq_rep

    print(json.dumps(report, indent=2))


def _norm_scale(scale, seq_len, device):
    if scale is None:
        return torch.ones(1, seq_len, 1, device=device)
    s = torch.as_tensor(scale, device=device).float()
    if s.ndim == 0:
        s = s.view(1, 1, 1)
    elif s.ndim == 1:
        s = s.view(1, 1, -1)
    elif s.ndim == 2:
        s = s.unsqueeze(-1)
    if s.shape[1] == 1 and seq_len > 1:
        s = s.expand(1, seq_len, 1)
    return s[..., :1]


def _load_sequence(processed_root, features_root, seq, device):
    pd = os.path.join(processed_root, seq)
    fd = os.path.join(features_root, seq)
    meta = json.load(open(os.path.join(pd, "meta.json")))
    rec = {
        "seq_name": seq,
        "start": 0,
        "end": int(meta["seq_len"]),
        "meta": meta,
        "init_state": load_torch(os.path.join(pd, "init_state.pth")),
        "expert_label": load_torch(os.path.join(pd, "expert_label.pth")),
        "residuals": load_torch(os.path.join(pd, "residuals.pth")),
        "valid_mask": load_torch(os.path.join(pd, "valid_mask.pth")),
        "features": {
            "mano_local_init": load_torch(os.path.join(fd, "mano_local_init.pth"))["mano_local_init"],
            "kp_2d": load_torch(os.path.join(fd, "kp_2d.pth"))["kp_2d"],
            "img_feat": load_torch(os.path.join(fd, "img_feat.pth"))["img_feat"],
            "cam_init": load_torch(os.path.join(fd, "cam_init.pth")),
            "uncertainty": load_torch(os.path.join(fd, "uncertainty.pth"))["uncertainty"],
        },
    }
    return _to_device(rec, device)


def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


if __name__ == "__main__":
    main()
