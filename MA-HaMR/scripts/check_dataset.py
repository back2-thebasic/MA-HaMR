#!/usr/bin/env python3
"""Smoke-check MA-HaMR dataset loading."""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mahmr.data.dataset import MAHaMRSequenceDataset
from mahmr.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MA-HaMR dataset")
    parser.add_argument("--config", default="configs/data/dev.yaml")
    parser.add_argument("--window-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config) if args.config else {}
    dataset_cfg = cfg.get("dataset", cfg)
    if args.window_size is not None:
        dataset_cfg["window_size"] = args.window_size

    ds = MAHaMRSequenceDataset(**dataset_cfg)
    print(f"num_sequences={len(ds.sequences)} num_items={len(ds)}")
    item = ds[0]
    print(f"first={item['seq_name']} frames=[{item['start']}, {item['end']})")
    feats = item["features"]
    print("mano_local_init", tuple(feats["mano_local_init"].shape))
    print("kp_2d", tuple(feats["kp_2d"].shape))
    print("img_feat", tuple(feats["img_feat"].shape))
    print("uncertainty", tuple(feats["uncertainty"].shape))
    print("valid_mask", tuple(item["valid_mask"]["valid_mask"].shape))


if __name__ == "__main__":
    main()
