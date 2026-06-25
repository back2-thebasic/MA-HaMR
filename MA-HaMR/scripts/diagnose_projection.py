#!/usr/bin/env python3
"""Compare camera projection conventions against kp_2d."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mahmr.geometry import MANOJointLayer
from mahmr.utils.io import load_torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="h2o_s1_h1")
    parser.add_argument("--frames", nargs="*", type=int, default=[0, 50, 100, 150, 200, 250, 300])
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    proc = f"/extra/SuC/data/mahmr/processed/{args.seq}"
    feat = f"/extra/SuC/data/mahmr/features/{args.seq}"
    init = load_torch(os.path.join(proc, "init_state.pth"))
    mano_local = load_torch(os.path.join(feat, "mano_local_init.pth"))["mano_local_init"].unsqueeze(0).to(device)
    kp = load_torch(os.path.join(feat, "kp_2d.pth"))["kp_2d"].unsqueeze(0).to(device)
    cam = load_torch(os.path.join(feat, "cam_init.pth"))
    cam = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cam.items()}
    is_right = init["is_right"].unsqueeze(0).to(device)
    layer = MANOJointLayer().to(device).eval()
    with torch.no_grad():
        joints = layer(mano_local, is_right=is_right)

    R = cam["cam_R"].unsqueeze(0)
    t = cam["cam_t"].unsqueeze(0)
    intr = cam["intrins"].to(device).view(1, 1, 1, 1, 4).expand(1, 2, joints.shape[2], 21, 4)
    scale = cam.get("world_scale", torch.ones(1, device=device)).view(1, 1, 1, 1, 1).expand(1, 2, joints.shape[2], 21, 1)
    R5 = R[:, :, :, None].expand(-1, -1, -1, 21, -1, -1)
    t4 = t[:, :, :, None].expand(-1, -1, -1, 21, -1)

    variants = {
        "R_P_minus_s_t": torch.matmul(R5, (joints - scale * t4).unsqueeze(-1)).squeeze(-1),
        "R_P_plus_s_t": torch.matmul(R5, joints.unsqueeze(-1)).squeeze(-1) + scale * t4,
        "R_P_minus_t": torch.matmul(R5, (joints - t4).unsqueeze(-1)).squeeze(-1),
        "R_P_plus_t": torch.matmul(R5, joints.unsqueeze(-1)).squeeze(-1) + t4,
        "direct_world_xy": joints,
    }
    conf = kp[..., 2] > 0.1
    frames = [f for f in args.frames if 0 <= f < joints.shape[2]]

    for name, campts in variants.items():
        if name == "direct_world_xy":
            xy = campts[..., :2]
        else:
            z = campts[..., 2].clamp_min(1e-3)
            xy = torch.stack(
                [
                    intr[..., 0] * campts[..., 0] / z + intr[..., 2],
                    intr[..., 1] * campts[..., 1] / z + intr[..., 3],
                ],
                dim=-1,
            )
        normal = _errors(xy, kp, conf, frames, swap=False)
        swapped = _errors(xy, kp, conf, frames, swap=True)
        print(
            f"{name}: mean={np.mean(normal):.2f} median={np.median(normal):.2f} "
            f"swap_mean={np.mean(swapped):.2f} swap_median={np.median(swapped):.2f}"
        )


def _errors(xy: torch.Tensor, kp: torch.Tensor, conf: torch.Tensor, frames: list[int], *, swap: bool) -> list[float]:
    out = []
    hand_idx = torch.tensor([1, 0], device=xy.device) if swap else torch.tensor([0, 1], device=xy.device)
    for frame_idx in frames:
        err = torch.linalg.norm(xy[0, hand_idx, frame_idx] - kp[0, :, frame_idx, :, :2], dim=-1)
        out.extend(err[conf[0, :, frame_idx]].detach().cpu().tolist())
    return out


if __name__ == "__main__":
    main()
