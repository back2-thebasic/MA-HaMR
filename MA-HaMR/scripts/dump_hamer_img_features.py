#!/usr/bin/env python3
"""Dump real frozen HaMeR/ViT ROI features for MA-HaMR.

The script reuses existing Dyn-HaMR HaMeR outputs:

  /extra/SuC/dynhamr_io/dynhamr/hamer_out/<seq>/<seq>.pkl

and runs only the frozen HaMeR backbone on the saved image crops. It writes
``img_feat.pth`` with shape (2, T, 512), matching the MA-HaMR data contract.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

HAMER_ROOT = "/data/SuC/Dyn-HaMR/third-party/hamer"
if HAMER_ROOT not in sys.path:
    sys.path.insert(0, HAMER_ROOT)

from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.models import load_hamer
from hamer.utils import recursive_to
from mahmr.utils.io import load_torch, save_json, save_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump frozen HaMeR/ViT image features")
    parser.add_argument("--processed-root", default="/extra/SuC/data/mahmr/processed")
    parser.add_argument("--features-root", default="/extra/SuC/data/mahmr/features")
    parser.add_argument("--hamer-out-root", default="/extra/SuC/dynhamr_io/dynhamr/hamer_out")
    parser.add_argument("--checkpoint-root", default="/data/SuC/Dyn-HaMR")
    parser.add_argument("--seq", nargs="*", default=None, help="Sequence names. Default: all with hamer pkl")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--feat-dim", type=int, default=512)
    parser.add_argument("--rescale-factor", type=float, default=2.5)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seqs = args.seq or _discover_sequences(args.processed_root)
    seqs = [s for s in seqs if os.path.isfile(_hamer_pkl_path(args.hamer_out_root, s))]
    if not seqs:
        raise FileNotFoundError("No sequences with hamer_out pkl found.")

    print(f"Loading HaMeR from {args.checkpoint_root} on {device} ...")
    model, model_cfg = load_hamer(args.checkpoint_root)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for seq in seqs:
        dump_one_sequence(
            seq,
            model,
            model_cfg,
            processed_root=args.processed_root,
            features_root=args.features_root,
            hamer_out_root=args.hamer_out_root,
            device=device,
            batch_size=args.batch_size,
            feat_dim=args.feat_dim,
            rescale_factor=args.rescale_factor,
        )


def dump_one_sequence(
    seq: str,
    model: torch.nn.Module,
    model_cfg: Any,
    *,
    processed_root: str,
    features_root: str,
    hamer_out_root: str,
    device: torch.device,
    batch_size: int,
    feat_dim: int,
    rescale_factor: float,
) -> None:
    processed_dir = os.path.join(processed_root, seq)
    features_dir = os.path.join(features_root, seq)
    meta_path = os.path.join(processed_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    seq_len = int(meta["seq_len"])
    num_hands = int(meta.get("num_hands", 2))
    kp_path = os.path.join(features_dir, "kp_2d.pth")
    if not os.path.isfile(kp_path):
        raise FileNotFoundError(f"Run scripts/dump_features.py first: {kp_path}")
    kp_2d = load_torch(kp_path)["kp_2d"].float()
    hamer_pkl = _load_pickle(_hamer_pkl_path(hamer_out_root, seq))
    frame_paths = _sorted_hamer_frames(hamer_pkl)[:seq_len]

    img_feat = torch.zeros(num_hands, seq_len, feat_dim, dtype=torch.float32)
    filled = torch.zeros(num_hands, seq_len, dtype=torch.bool)

    pending: List[Dict[str, Any]] = []
    print(f"[{seq}] frames={seq_len} writing real HaMeR/ViT img_feat ...")
    with torch.no_grad():
        for t, frame_path in enumerate(frame_paths):
            image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            for hand_idx in range(num_hands):
                kp = kp_2d[hand_idx, t].numpy()
                if kp[:, 2].sum() <= 0:
                    continue
                bbox = _bbox_from_keypoints(kp, image.shape)
                pending.append({"t": t, "hand_idx": hand_idx, "image": image, "bbox": bbox, "right": hand_idx})
                if len(pending) >= batch_size:
                    _flush_batch(pending, model, model_cfg, device, img_feat, filled, feat_dim, rescale_factor)
                    pending.clear()
        if pending:
            _flush_batch(pending, model, model_cfg, device, img_feat, filled, feat_dim, rescale_factor)

    save_torch(os.path.join(features_dir, "img_feat.pth"), {"img_feat": img_feat})
    feature_meta_path = os.path.join(features_dir, "meta.json")
    feature_meta = _load_json(feature_meta_path)
    feature_meta["sources"]["img_feat"] = "hamer_vit_backbone:pooled_to_512"
    feature_meta["img_feat_valid_ratio"] = float(filled.float().mean())
    feature_meta["img_feat_abs_mean"] = float(img_feat.abs().mean())
    save_json(feature_meta_path, feature_meta)
    print(
        f"[{seq}] done valid_ratio={feature_meta['img_feat_valid_ratio']:.3f} "
        f"abs_mean={feature_meta['img_feat_abs_mean']:.4f}"
    )


def _flush_batch(
    items: List[Dict[str, Any]],
    model: torch.nn.Module,
    model_cfg: Any,
    device: torch.device,
    img_feat: torch.Tensor,
    filled: torch.Tensor,
    feat_dim: int,
    rescale_factor: float,
) -> None:
    # Group by source image because ViTDetDataset accepts one image at a time.
    by_image: Dict[int, List[Dict[str, Any]]] = {}
    for item in items:
        by_image.setdefault(id(item["image"]), []).append(item)

    for group in by_image.values():
        image = group[0]["image"]
        boxes = np.stack([g["bbox"] for g in group], axis=0)
        right = np.asarray([g["right"] for g in group], dtype=np.float32)
        kps = np.zeros((len(group), 21, 3), dtype=np.float32)
        dataset = ViTDetDataset(model_cfg, image, boxes, right, kps, rescale_factor=rescale_factor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(group), shuffle=False, num_workers=0)
        batch = recursive_to(next(iter(loader)), device)
        x = batch["img"][:, :, :, 32:-32]
        feats = model.backbone(x)
        vec = _to_feature_vector(feats, feat_dim).detach().cpu()
        for i, item in enumerate(group):
            img_feat[item["hand_idx"], item["t"]] = vec[i]
            filled[item["hand_idx"], item["t"]] = True


def _to_feature_vector(feats: Any, feat_dim: int) -> torch.Tensor:
    if isinstance(feats, (list, tuple)):
        feats = feats[-1]
    if feats.ndim == 4:
        vec = feats.mean(dim=(-2, -1))
    elif feats.ndim == 3:
        vec = feats.mean(dim=1)
    elif feats.ndim == 2:
        vec = feats
    else:
        raise ValueError(f"Unsupported backbone feature shape: {tuple(feats.shape)}")
    if vec.shape[-1] > feat_dim:
        vec = vec[:, :feat_dim]
    elif vec.shape[-1] < feat_dim:
        vec = torch.nn.functional.pad(vec, (0, feat_dim - vec.shape[-1]))
    return vec.float()


def _bbox_from_keypoints(keypoints: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    valid = keypoints[:, 2] > 0.1
    h, w = image_shape[:2]
    if valid.sum() >= 3:
        xy = keypoints[valid, :2]
        x1, y1 = xy.min(axis=0)
        x2, y2 = xy.max(axis=0)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        size = max(x2 - x1, y2 - y1, 1.0) * 1.8
    else:
        cx, cy = w * 0.5, h * 0.5
        size = min(w, h) * 0.5
    return np.asarray(
        [
            max(0.0, cx - size * 0.5),
            max(0.0, cy - size * 0.5),
            min(float(w - 1), cx + size * 0.5),
            min(float(h - 1), cy + size * 0.5),
        ],
        dtype=np.float32,
    )


def _discover_sequences(processed_root: str) -> List[str]:
    return sorted(
        name
        for name in os.listdir(processed_root)
        if os.path.isfile(os.path.join(processed_root, name, "meta.json"))
    )


def _hamer_pkl_path(root: str, seq: str) -> str:
    return os.path.join(root, seq, f"{seq}.pkl")


def _load_pickle(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _sorted_hamer_frames(hamer_pkl: Dict[str, Any]) -> List[str]:
    def key_fn(path: str) -> tuple[str, int]:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            return os.path.dirname(path), int(stem)
        except ValueError:
            return os.path.dirname(path), 0

    return sorted(hamer_pkl.keys(), key=key_fn)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
