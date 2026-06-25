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
from mahmr.loss import build_window_residual_targets
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
    item = _to_device(item, args.device)
    targets = build_window_residual_targets(item, device=args.device)

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
            memory_values=targets["packed_residual"],
            write_mask=targets["valid_mask"],
            reset_memory=True,
            return_attention=True,
        )

    print(f"seq={item['seq_name']} frames=[{item['start']}, {item['end']})")
    print(f"writes={int(targets['valid_mask'].sum())} memory_size={out['memory_size']}")
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

def _to_device(obj, device: str):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


if __name__ == "__main__":
    main()
