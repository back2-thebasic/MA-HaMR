#!/usr/bin/env python3
"""Export sparse expert pseudo labels from a Dyn-HaMR optimization log."""

from __future__ import annotations

import argparse
import json
import os
import sys

# allow running without pip install -e .
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mahmr.export.expert_exporter import export_sequence
from mahmr.utils.io import load_yaml


def append_manifest(manifest_path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="MA-HaMR Step 1: export expert labels")
    parser.add_argument("--config", required=True, help="YAML config, e.g. configs/export/demo1.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    summary = export_sequence(
        dynhamr_log_dir=cfg["dynhamr_log_dir"],
        output_dir=cfg["output_dir"],
        seq_name=cfg.get("seq_name"),
        expert_stride=int(cfg.get("expert_stride", 10)),
        expert_stage=cfg.get("expert_stage", "smooth_fit"),
        expert_iter=cfg.get("expert_iter"),
        fps=float(cfg.get("fps", 30.0)),
        camera_source=cfg.get("camera_source", "vipe"),
    )

    manifest = cfg.get("manifest_path")
    if manifest:
        append_manifest(manifest, {"seq_name": summary["seq_name"], **summary})

    print("Export complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
