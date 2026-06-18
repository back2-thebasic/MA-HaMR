#!/usr/bin/env python3
"""Smoke-check MA-HaMR Step 3 model forward."""

from __future__ import annotations

import argparse
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mahmr.data.dataset import MAHaMRSequenceDataset
from mahmr.models.memory import pack_residual_value
from mahmr.models.refinement import MAHaMRRefiner
from mahmr.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MA-HaMR Step 3 model")
    parser.add_argument("--config", default="configs/data/dev.yaml")
    parser.add_argument("--item", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_yaml(args.config) if args.config else {}
    dataset_cfg = cfg.get("dataset", cfg)
    ds = MAHaMRSequenceDataset(**dataset_cfg)
    item = ds[args.item]
    features = _to_device(item["features"], args.device)
    memory_values, write_mask = build_window_memory_targets(item, device=args.device)

    model = MAHaMRRefiner(
        img_feat_dim=features["img_feat"].shape[-1],
        topk=4,
        exclude_recent=5,
        hidden_dim=128,
        num_blocks=2,
    ).to(args.device)
    model.eval()

    with torch.no_grad():
        out = model(
            features,
            memory_values=memory_values,
            write_mask=write_mask,
            reset_memory=True,
            return_attention=True,
        )

    print(f"seq={item['seq_name']} frames=[{item['start']}, {item['end']})")
    print(f"writes={int(write_mask.sum())} memory_size={out['memory_size']}")
    for key in (
        "delta_mano_local",
        "delta_root_orient",
        "delta_pose_body",
        "delta_betas",
        "delta_trans",
        "delta_world_scale",
        "mano_local_refined",
        "world_scale_refined",
        "memory_context",
    ):
        print(key, tuple(out[key].shape))
    print("world_scale_min", float(out["world_scale_refined"].min()))
    print("packed_abs_mean", float(out["packed_residual"].abs().mean()))


def build_window_memory_targets(item: dict, *, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    start, end = int(item["start"]), int(item["end"])
    seq_len = end - start
    num_hands = int(item["features"]["mano_local_init"].shape[0])
    d_val = num_hands * 61 + 1
    values = torch.zeros(seq_len, d_val, dtype=torch.float32, device=device)
    write_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

    frame_indices = item["valid_mask"].get("frame_indices", item["expert_label"]["frame_indices"])
    frame_indices = torch.as_tensor(frame_indices).long()
    residuals = item["residuals"]
    for residual_idx, frame_idx in enumerate(frame_indices.tolist()):
        if frame_idx < start or frame_idx >= end:
            continue
        local_t = frame_idx - start
        packed = pack_residual_value(
            delta_root_orient=residuals["delta_root_orient"][:, residual_idx : residual_idx + 1],
            delta_pose_body=residuals["delta_pose_body"][:, residual_idx : residual_idx + 1],
            delta_trans=residuals["delta_trans"][:, residual_idx : residual_idx + 1],
            delta_world_scale=residuals.get("delta_world_scale"),
            num_hands=num_hands,
        )
        values[local_t] = packed[0, 0].to(device)
        write_mask[local_t] = True
    return values, write_mask


def _to_device(obj, device: str):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


if __name__ == "__main__":
    main()
