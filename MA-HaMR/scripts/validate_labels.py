#!/usr/bin/env python3
"""Sanity-check exported Step 1 tensors."""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mahmr.utils.io import load_torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    args = parser.parse_args()

    d = args.processed_dir
    meta = json.load(open(os.path.join(d, "meta.json")))
    init_state = load_torch(os.path.join(d, "init_state.pth"))
    expert = load_torch(os.path.join(d, "expert_label.pth"))
    valid = load_torch(os.path.join(d, "valid_mask.pth"))
    residuals = load_torch(os.path.join(d, "residuals.pth"))

    print("=== meta ===")
    print(json.dumps(meta, indent=2))

    print("\n=== init_state shapes ===")
    for k, v in init_state.items():
        if k == "source_npz":
            continue
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)}")

    print("\n=== expert_label shapes ===")
    for k, v in expert.items():
        if isinstance(v, str):
            print(f"  {k}: {v}")
        elif hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)}")

    print("\n=== valid_mask ===")
    print(f"  valid frames: {valid['valid_mask'].sum().item()} / {len(valid['valid_mask'])}")
    print(f"  stride coverage: {100 * valid['valid_mask'].float().mean():.1f}%")

    print("\n=== world_scale (数值, 非 shape) ===")
    if "world_scale" in init_state:
        print(f"  init_state:   {init_state['world_scale'].reshape(-1).tolist()}")
    if "world_scale" in expert:
        print(f"  expert_label: {expert['world_scale'].reshape(-1).tolist()}")
    if "delta_world_scale" in residuals:
        print(f"  delta:        {residuals['delta_world_scale'].reshape(-1).tolist()}")

    print("\n=== residuals (action-conditioned correction) ===")
    for k, v in residuals.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)}, mean_abs={v.abs().mean().item():.6f}")

    seq_len = meta["seq_len"]
    assert valid["valid_mask"].shape[0] == seq_len
    assert expert["frame_indices"].max().item() < seq_len
    print("\nOK: basic consistency checks passed.")


if __name__ == "__main__":
    main()
