#!/usr/bin/env python3
"""MA-HaMR Step 2: dump training features from Step 1 artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mahmr.data.feature_dump import discover_processed_dirs, dump_many, dump_sequence_features
from mahmr.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="MA-HaMR Step 2 feature dump")
    parser.add_argument("--config", help="Optional YAML config")
    parser.add_argument("--processed-dir", help="One processed sequence directory")
    parser.add_argument("--processed-root", default="/extra/SuC/data/mahmr/processed")
    parser.add_argument("--features-root", default="/extra/SuC/data/mahmr/features")
    parser.add_argument("--img-feat-dim", type=int, default=512)
    parser.add_argument("--all", action="store_true", help="Dump all sequences under processed-root")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = {}
    if args.config:
        cfg = load_yaml(args.config) or {}

    processed_root = cfg.get("processed_root", args.processed_root)
    features_root = cfg.get("features_root", args.features_root)
    img_feat_dim = int(cfg.get("img_feat_dim", args.img_feat_dim))
    overwrite = bool(args.overwrite or cfg.get("overwrite", False))
    sequence_names = cfg.get("sequence_names")

    if args.processed_dir or cfg.get("processed_dir"):
        processed_dir = args.processed_dir or cfg["processed_dir"]
        seq_name = _read_seq_name(processed_dir)
        summary = dump_sequence_features(
            processed_dir,
            os.path.join(features_root, seq_name),
            img_feat_dim=img_feat_dim,
            overwrite=overwrite,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.all or cfg.get("all", False):
        processed_dirs = discover_processed_dirs(processed_root)
        if sequence_names:
            keep = set(sequence_names)
            processed_dirs = [d for d in processed_dirs if os.path.basename(d) in keep]
        summaries = dump_many(
            processed_dirs,
            features_root,
            img_feat_dim=img_feat_dim,
            overwrite=overwrite,
        )
        print(f"Dumped {len(summaries)} sequences to {features_root}")
        for item in summaries:
            print(f"  {item['seq_name']}: {item['shapes']}")
        return

    parser.error("Specify --processed-dir or --all (or set one in --config).")


def _read_seq_name(processed_dir: str) -> str:
    with open(os.path.join(processed_dir, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)["seq_name"]


if __name__ == "__main__":
    main()
