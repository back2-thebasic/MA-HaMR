#!/usr/bin/env python3
"""Audit which sequences are ready for MA-HaMR training/evaluation expansion."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


REQUIRED_PROCESSED = ["init_state.pth", "expert_label.pth", "residuals.pth", "valid_mask.pth", "meta.json"]
REQUIRED_FEATURES = ["mano_local_init.pth", "kp_2d.pth", "img_feat.pth", "cam_init.pth", "uncertainty.pth", "meta.json"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MA-HaMR sequence readiness")
    parser.add_argument("--processed-root", default="/extra/SuC/data/mahmr/processed")
    parser.add_argument("--features-root", default="/extra/SuC/data/mahmr/features")
    parser.add_argument("--hamer-out-root", default="/extra/SuC/dynhamr_io/dynhamr/hamer_out")
    parser.add_argument("--output", default="/extra/SuC/experiments/mahmr/expansion_readiness.json")
    args = parser.parse_args()

    names = set(_dirs(args.processed_root)) | set(_dirs(args.features_root)) | set(_dirs(args.hamer_out_root))
    records = []
    for name in sorted(names):
        processed_missing = _missing(os.path.join(args.processed_root, name), REQUIRED_PROCESSED)
        feature_missing = _missing(os.path.join(args.features_root, name), REQUIRED_FEATURES)
        hamer_ready = os.path.isfile(os.path.join(args.hamer_out_root, name, f"{name}.pkl"))
        is_demo = name.lower().startswith("demo")
        ready = not is_demo and not processed_missing and not feature_missing
        records.append(
            {
                "seq_name": name,
                "ready_for_train_eval": ready,
                "is_demo_or_placeholder": is_demo,
                "has_hamer_out": hamer_ready,
                "missing_processed": processed_missing,
                "missing_features": feature_missing,
                "recommendation": _recommendation(name, ready, is_demo, hamer_ready, processed_missing, feature_missing),
            }
        )

    report = {
        "ready_sequences": [r["seq_name"] for r in records if r["ready_for_train_eval"]],
        "blocked_sequences": [r for r in records if not r["ready_for_train_eval"]],
        "records": records,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def _dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]


def _missing(root: str, required: List[str]) -> List[str]:
    return [name for name in required if not os.path.isfile(os.path.join(root, name))]


def _recommendation(
    name: str,
    ready: bool,
    is_demo: bool,
    hamer_ready: bool,
    processed_missing: List[str],
    feature_missing: List[str],
) -> str:
    if ready:
        return "Ready for MA-HaMR training/evaluation."
    if is_demo:
        return "Keep excluded from real experiments because it is a demo/placeholder sequence."
    if processed_missing and feature_missing and hamer_ready:
        return "Has HaMeR output but still needs Step 1 expert export and Step 2 feature dumping."
    if processed_missing:
        return "Needs Step 1 expert export before training/evaluation."
    if feature_missing:
        return "Needs Step 2 feature dumping before training/evaluation."
    return "Blocked; inspect sequence artifacts manually."


if __name__ == "__main__":
    main()
