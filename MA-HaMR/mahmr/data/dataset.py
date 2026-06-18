"""PyTorch Dataset for MA-HaMR Step 2 artifacts."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from mahmr.data.feature_dump import discover_processed_dirs
from mahmr.utils.io import load_torch


class MAHaMRSequenceDataset(Dataset):
    """Sequence/window dataset over processed labels and dumped features."""

    def __init__(
        self,
        processed_root: str = "/extra/SuC/data/mahmr/processed",
        features_root: str = "/extra/SuC/data/mahmr/features",
        *,
        manifest_path: Optional[str] = None,
        sequence_names: Optional[List[str]] = None,
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> None:
        self.processed_root = os.path.abspath(processed_root)
        self.features_root = os.path.abspath(features_root)
        self.window_size = window_size
        self.stride = stride or window_size or 0

        self.sequences = self._discover_sequences(manifest_path, sequence_names)
        self.index: List[tuple[int, int, int]] = []
        for seq_idx, seq in enumerate(self.sequences):
            seq_len = int(seq["meta"]["seq_len"])
            if window_size is None or window_size >= seq_len:
                self.index.append((seq_idx, 0, seq_len))
            else:
                for start in range(0, max(seq_len - window_size + 1, 1), self.stride):
                    self.index.append((seq_idx, start, start + window_size))
                if self.index[-1][0] != seq_idx or self.index[-1][2] < seq_len:
                    self.index.append((seq_idx, seq_len - window_size, seq_len))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_idx, start, end = self.index[idx]
        seq = self.sequences[seq_idx]
        processed_dir = seq["processed_dir"]
        features_dir = seq["features_dir"]

        init_state = load_torch(os.path.join(processed_dir, "init_state.pth"))
        expert_label = load_torch(os.path.join(processed_dir, "expert_label.pth"))
        residuals = load_torch(os.path.join(processed_dir, "residuals.pth"))
        valid_mask = load_torch(os.path.join(processed_dir, "valid_mask.pth"))

        features = {
            "mano_local_init": load_torch(os.path.join(features_dir, "mano_local_init.pth"))["mano_local_init"],
            "kp_2d": load_torch(os.path.join(features_dir, "kp_2d.pth"))["kp_2d"],
            "img_feat": load_torch(os.path.join(features_dir, "img_feat.pth"))["img_feat"],
            "cam_init": load_torch(os.path.join(features_dir, "cam_init.pth")),
            "uncertainty": load_torch(os.path.join(features_dir, "uncertainty.pth"))["uncertainty"],
        }

        if start != 0 or end != int(seq["meta"]["seq_len"]):
            init_state = _slice_dict_time(init_state, start, end)
            features = _slice_features(features, start, end)
            valid_mask = _slice_valid_mask(valid_mask, start, end)
            # Sparse expert/residual tensors are kept sequence-level for now;
            # training code can gather them with valid_mask/frame_indices.

        return {
            "seq_name": seq["meta"]["seq_name"],
            "start": start,
            "end": end,
            "meta": seq["meta"],
            "feature_meta": seq["feature_meta"],
            "init_state": init_state,
            "expert_label": expert_label,
            "residuals": residuals,
            "valid_mask": valid_mask,
            "features": features,
        }

    def _discover_sequences(
        self,
        manifest_path: Optional[str],
        sequence_names: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        by_name: Dict[str, str] = {}
        if manifest_path and os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    name = rec["seq_name"]
                    candidate = rec.get("output_dir") or os.path.join(self.processed_root, name)
                    if not os.path.isdir(candidate):
                        candidate = os.path.join(self.processed_root, name)
                    by_name[name] = candidate
        else:
            for d in discover_processed_dirs(self.processed_root):
                by_name[os.path.basename(d)] = d

        if sequence_names:
            keep = set(sequence_names)
            by_name = {k: v for k, v in by_name.items() if k in keep}

        sequences = []
        for name in sorted(by_name):
            processed_dir = os.path.abspath(by_name[name])
            features_dir = os.path.join(self.features_root, name)
            meta_path = os.path.join(processed_dir, "meta.json")
            feature_meta_path = os.path.join(features_dir, "meta.json")
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(meta_path)
            if not os.path.isfile(feature_meta_path):
                raise FileNotFoundError(
                    f"Missing feature dump for {name}: {feature_meta_path}. "
                    "Run scripts/dump_features.py first."
                )
            sequences.append(
                {
                    "processed_dir": processed_dir,
                    "features_dir": features_dir,
                    "meta": _load_json(meta_path),
                    "feature_meta": _load_json(feature_meta_path),
                }
            )
        return sequences


def _slice_features(features: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    out = dict(features)
    for key in ("mano_local_init", "kp_2d", "img_feat", "uncertainty"):
        out[key] = out[key][:, start:end]
    out["cam_init"] = _slice_dict_time(out["cam_init"], start, end)
    return out


def _slice_dict_time(data: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    out = {}
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            out[key] = value
            continue
        if value.ndim >= 2 and value.shape[1] >= end:
            out[key] = value[:, start:end].clone()
        elif value.ndim >= 1 and value.shape[0] >= end and key.startswith("cam_"):
            out[key] = value[start:end].clone()
        else:
            out[key] = value
    return out


def _slice_valid_mask(valid: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
    out = dict(valid)
    if "valid_mask" in out:
        out["valid_mask"] = out["valid_mask"][start:end].clone()
    return out


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
