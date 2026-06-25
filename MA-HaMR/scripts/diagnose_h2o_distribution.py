#!/usr/bin/env python3
"""Diagnose H2O train/val/test distribution gaps for MA-HaMR.

This script is intentionally data-centric: before adding denser pseudo labels or
a stronger global alignment head, we need to know whether held-out failures are
caused by scale/translation/bbox ranges that are under-represented in training.
It reads the official MA-HaMR config files, loads processed labels/features, and
writes sequence-level + split-level summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mahmr.utils.io import load_torch, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze H2O split distribution gaps")
    parser.add_argument("--train-config", default="configs/train/h2o_official_train_v4.yaml")
    parser.add_argument("--val-config", default="configs/eval/h2o_official_val_v1.yaml")
    parser.add_argument("--test-config", default="configs/eval/h2o_official_test_v1.yaml")
    parser.add_argument("--output-dir", default="/extra/SuC/experiments/mahmr/h2o_distribution_diagnostics")
    parser.add_argument("--conf-thresh", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    split_cfgs = {
        "train": load_yaml(args.train_config),
        "val": load_yaml(args.val_config),
        "test": load_yaml(args.test_config),
    }
    rows: List[Dict[str, Any]] = []
    for split, cfg in split_cfgs.items():
        ds = cfg["dataset"]
        for seq in ds["sequence_names"]:
            rows.append(
                analyze_sequence(
                    split=split,
                    seq=seq,
                    processed_root=ds["processed_root"],
                    features_root=ds["features_root"],
                    conf_thresh=args.conf_thresh,
                )
            )

    split_summary = summarize_splits(rows)
    train_ref = split_summary["train"]
    ood_rows = [add_ood_flags(row, train_ref) for row in rows]

    write_csv(os.path.join(args.output_dir, "sequence_distribution.csv"), ood_rows)
    write_json(os.path.join(args.output_dir, "split_distribution_summary.json"), split_summary)
    write_json(os.path.join(args.output_dir, "sequence_distribution.json"), {"sequences": ood_rows})
    write_markdown(os.path.join(args.output_dir, "distribution_report.md"), ood_rows, split_summary)

    print(json.dumps({"output_dir": os.path.abspath(args.output_dir), "split_summary": split_summary}, indent=2))


def analyze_sequence(
    *,
    split: str,
    seq: str,
    processed_root: str,
    features_root: str,
    conf_thresh: float,
) -> Dict[str, Any]:
    processed_dir = os.path.join(processed_root, seq)
    features_dir = os.path.join(features_root, seq)
    meta = load_json(os.path.join(processed_dir, "meta.json"))
    expert = load_torch(os.path.join(processed_dir, "expert_label.pth"))
    residuals = load_torch(os.path.join(processed_dir, "residuals.pth"))
    valid = load_torch(os.path.join(processed_dir, "valid_mask.pth"))
    features = {
        "mano_local_init": load_torch(os.path.join(features_dir, "mano_local_init.pth"))["mano_local_init"],
        "kp_2d": load_torch(os.path.join(features_dir, "kp_2d.pth"))["kp_2d"],
        "cam_init": load_torch(os.path.join(features_dir, "cam_init.pth")),
    }

    seq_len = int(meta["seq_len"])
    frame_indices = torch.as_tensor(expert["frame_indices"]).long()
    init_scale = normalize_scale(features["cam_init"].get("world_scale"), seq_len)
    expert_scale = expert_scale_abs(expert, residuals, init_scale, frame_indices)
    kp_stats = bbox_stats(features["kp_2d"], conf_thresh=conf_thresh)
    init_trans = features["mano_local_init"][..., 58:61]
    expert_trans = torch.as_tensor(expert["trans"]).float() if "trans" in expert else torch.empty(0)
    cam_t = torch.as_tensor(features["cam_init"].get("cam_t", torch.empty(0))).float()

    row: Dict[str, Any] = {
        "split": split,
        "seq": seq,
        "num_frames": seq_len,
        "num_expert_frames": int(frame_indices.numel()),
        "expert_coverage": safe_float(frame_indices.numel() / max(seq_len, 1)),
        "init_scale_mean": stat(init_scale, "mean"),
        "init_scale_min": stat(init_scale, "min"),
        "init_scale_max": stat(init_scale, "max"),
        "expert_scale_abs_mean": stat(expert_scale, "mean"),
        "expert_scale_abs_min": stat(expert_scale, "min"),
        "expert_scale_abs_max": stat(expert_scale, "max"),
        "expert_scale_abs_p90": stat(expert_scale, "p90"),
        "expert_scale_abs_p99": stat(expert_scale, "p99"),
        "scale_abs_delta_mean": stat((expert_scale - init_scale[frame_indices].abs()).abs(), "mean"),
        "scale_abs_delta_max": stat((expert_scale - init_scale[frame_indices].abs()).abs(), "max"),
        "bbox_diag_mean": kp_stats["diag_mean"],
        "bbox_diag_p90": kp_stats["diag_p90"],
        "bbox_diag_max": kp_stats["diag_max"],
        "bbox_area_mean": kp_stats["area_mean"],
        "bbox_area_p90": kp_stats["area_p90"],
        "bbox_valid_frac": kp_stats["valid_frac"],
        "init_trans_norm_mean": stat(init_trans.norm(dim=-1), "mean"),
        "init_trans_norm_p90": stat(init_trans.norm(dim=-1), "p90"),
        "init_trans_norm_max": stat(init_trans.norm(dim=-1), "max"),
        "expert_trans_norm_mean": stat(expert_trans.norm(dim=-1), "mean"),
        "expert_trans_norm_p90": stat(expert_trans.norm(dim=-1), "p90"),
        "expert_trans_norm_max": stat(expert_trans.norm(dim=-1), "max"),
        "delta_trans_norm_mean": stat(torch.as_tensor(residuals.get("delta_trans", torch.empty(0))).float().norm(dim=-1), "mean"),
        "delta_trans_norm_p90": stat(torch.as_tensor(residuals.get("delta_trans", torch.empty(0))).float().norm(dim=-1), "p90"),
        "delta_trans_norm_max": stat(torch.as_tensor(residuals.get("delta_trans", torch.empty(0))).float().norm(dim=-1), "max"),
        "cam_t_norm_mean": stat(cam_t.norm(dim=-1), "mean"),
        "cam_t_norm_p90": stat(cam_t.norm(dim=-1), "p90"),
        "cam_t_norm_max": stat(cam_t.norm(dim=-1), "max"),
        "valid_mask_count": int(torch.as_tensor(valid.get("valid_mask", torch.empty(0))).bool().sum().item()),
    }
    return row


def normalize_scale(scale: Any, seq_len: int) -> torch.Tensor:
    if scale is None:
        return torch.ones(seq_len)
    x = torch.as_tensor(scale).float().reshape(-1)
    if x.numel() == 0:
        return torch.ones(seq_len)
    if x.numel() == 1:
        return x.expand(seq_len).clone()
    if x.numel() < seq_len:
        # Some legacy exports store a sequence-level scale. Repeat the last value
        # instead of failing so the diagnostic can still run.
        pad = x[-1:].expand(seq_len - x.numel())
        x = torch.cat([x, pad], dim=0)
    return x[:seq_len]


def expert_scale_abs(
    expert: Dict[str, torch.Tensor],
    residuals: Dict[str, torch.Tensor],
    init_scale: torch.Tensor,
    frame_indices: torch.Tensor,
) -> torch.Tensor:
    if "world_scale" in expert:
        return torch.as_tensor(expert["world_scale"]).float().reshape(-1).abs()
    delta = residuals.get("delta_world_scale")
    if delta is None:
        return init_scale[frame_indices].abs()
    delta = torch.as_tensor(delta).float().reshape(-1)
    if delta.numel() == 1:
        delta = delta.expand(frame_indices.numel())
    return (init_scale[frame_indices] + delta[: frame_indices.numel()]).abs()


def bbox_stats(kp: torch.Tensor, *, conf_thresh: float) -> Dict[str, float]:
    kp = torch.as_tensor(kp).float()
    # H, T, K, 3
    diags: List[float] = []
    areas: List[float] = []
    total = int(kp.shape[0] * kp.shape[1])
    valid_frames = 0
    for h in range(kp.shape[0]):
        for t in range(kp.shape[1]):
            hand = kp[h, t]
            mask = hand[:, 2] > conf_thresh
            if int(mask.sum().item()) < 4:
                continue
            pts = hand[mask, :2]
            lo = pts.min(dim=0).values
            hi = pts.max(dim=0).values
            wh = (hi - lo).clamp_min(0.0)
            diags.append(float(torch.linalg.norm(wh)))
            areas.append(float(wh[0] * wh[1]))
            valid_frames += 1
    return {
        "diag_mean": stat_list(diags, "mean"),
        "diag_p90": stat_list(diags, "p90"),
        "diag_max": stat_list(diags, "max"),
        "area_mean": stat_list(areas, "mean"),
        "area_p90": stat_list(areas, "p90"),
        "valid_frac": safe_float(valid_frames / max(total, 1)),
    }


def summarize_splits(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    metrics = [
        "expert_scale_abs_mean",
        "expert_scale_abs_max",
        "scale_abs_delta_mean",
        "scale_abs_delta_max",
        "bbox_diag_mean",
        "bbox_diag_max",
        "expert_trans_norm_mean",
        "expert_trans_norm_max",
        "delta_trans_norm_mean",
        "delta_trans_norm_max",
        "cam_t_norm_mean",
        "cam_t_norm_max",
    ]
    for split in sorted({r["split"] for r in rows}):
        subset = [r for r in rows if r["split"] == split]
        out[split] = {
            "num_sequences": len(subset),
            "num_frames": int(sum(r["num_frames"] for r in subset)),
            "num_expert_frames": int(sum(r["num_expert_frames"] for r in subset)),
            "expert_coverage": safe_float(sum(r["num_expert_frames"] for r in subset) / max(sum(r["num_frames"] for r in subset), 1)),
            "metrics": {},
        }
        for key in metrics:
            vals = [float(r[key]) for r in subset if np.isfinite(float(r[key]))]
            out[split]["metrics"][key] = summarize_values(vals)
    return out


def add_ood_flags(row: Dict[str, Any], train_ref: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    metrics = train_ref["metrics"]
    checks = {
        "expert_scale_abs_max": "train_p99",
        "scale_abs_delta_max": "train_p99",
        "bbox_diag_max": "train_p99",
        "expert_trans_norm_max": "train_p99",
        "delta_trans_norm_max": "train_p99",
        "cam_t_norm_max": "train_p99",
    }
    flags = []
    for key, threshold_name in checks.items():
        threshold = metrics.get(key, {}).get(threshold_name)
        value = out.get(key)
        if threshold is None or value is None or not np.isfinite(float(value)):
            continue
        ratio = safe_ratio(float(value), float(threshold))
        out[f"{key}_over_train_p99"] = ratio
        if ratio > 1.05:
            flags.append(f"{key}>{ratio:.2f}x_train_p99")
    out["ood_flags"] = ";".join(flags)
    out["ood_flag_count"] = len(flags)
    return out


def summarize_values(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "p50": float("nan"), "p90": float("nan"), "train_p99": float("nan")}
    return {
        "mean": safe_float(arr.mean()),
        "min": safe_float(arr.min()),
        "max": safe_float(arr.max()),
        "p50": safe_float(np.quantile(arr, 0.50)),
        "p90": safe_float(np.quantile(arr, 0.90)),
        "train_p99": safe_float(np.quantile(arr, 0.99)),
    }


def stat(x: torch.Tensor, mode: str) -> float:
    arr = torch.as_tensor(x).float().reshape(-1)
    arr = arr[torch.isfinite(arr)]
    if arr.numel() == 0:
        return float("nan")
    if mode == "mean":
        return safe_float(arr.mean().item())
    if mode == "min":
        return safe_float(arr.min().item())
    if mode == "max":
        return safe_float(arr.max().item())
    if mode == "p90":
        return safe_float(arr.quantile(0.90).item())
    if mode == "p99":
        return safe_float(arr.quantile(0.99).item())
    raise ValueError(mode)


def stat_list(values: List[float], mode: str) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if mode == "mean":
        return safe_float(arr.mean())
    if mode == "max":
        return safe_float(arr.max())
    if mode == "p90":
        return safe_float(np.quantile(arr, 0.90))
    raise ValueError(mode)


def safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
        return float("nan")
    return safe_float(num / den)


def safe_float(x: float) -> float:
    return float(x) if np.isfinite(float(x)) else float("nan")


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=True)


def write_markdown(path: str, rows: List[Dict[str, Any]], split_summary: Dict[str, Any]) -> None:
    lines = ["# H2O Distribution Diagnostic Report", ""]
    lines.append("## Split Summary")
    lines.append("")
    lines.append("| split | seqs | frames | expert_frames | coverage | scale_abs_max_mean | scale_delta_max_mean | bbox_diag_max_mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for split in ("train", "val", "test"):
        s = split_summary[split]
        m = s["metrics"]
        lines.append(
            f"| {split} | {s['num_sequences']} | {s['num_frames']} | {s['num_expert_frames']} | "
            f"{s['expert_coverage']:.3f} | {m['expert_scale_abs_max']['mean']:.3f} | "
            f"{m['scale_abs_delta_max']['mean']:.3f} | {m['bbox_diag_max']['mean']:.1f} |"
        )
    lines.extend(["", "## OOD Flags By Sequence", ""])
    lines.append("| split | seq | expert_scale_abs_max | scale_delta_max | bbox_diag_max | ood_flags |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['seq']} | {float(row['expert_scale_abs_max']):.3f} | "
            f"{float(row['scale_abs_delta_max']):.3f} | {float(row['bbox_diag_max']):.1f} | {row.get('ood_flags','')} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
