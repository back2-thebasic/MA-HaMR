"""Step 2 feature dumping for MA-HaMR.

This module converts the Step 1 export artifacts into training-ready tensors.
It deliberately avoids importing Dyn-HaMR code; any teacher output is read from
files recorded in ``meta.json``.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import cv2

from mahmr.data.dynhamr_log import load_npz
from mahmr.utils.io import load_torch, save_json, save_torch


KP2D_KEYS = (
    "kp_2d",
    "keypoints_2d",
    "keypoints2d",
    "joints2d",
    "joints_2d",
    "hand_kp2d",
    "pred_keypoints_2d",
)


def dump_sequence_features(
    processed_dir: str,
    output_dir: str,
    *,
    img_feat_dim: int = 512,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create feature tensors for one processed sequence.

    The current dump uses available Dyn-HaMR/HaMeR artifacts when present and
    writes deterministic placeholders for unavailable image ROI features. The
    placeholder is explicit in ``meta.json`` so it can be replaced later without
    changing the Dataset contract.
    """

    processed_dir = os.path.abspath(processed_dir)
    output_dir = os.path.abspath(output_dir)
    done_path = os.path.join(output_dir, "meta.json")
    if os.path.isfile(done_path) and not overwrite:
        with open(done_path, "r", encoding="utf-8") as f:
            return json.load(f)

    init_state = load_torch(os.path.join(processed_dir, "init_state.pth"))
    meta = _load_json(os.path.join(processed_dir, "meta.json"))
    seq_name = meta["seq_name"]
    seq_len = int(meta["seq_len"])
    num_hands = int(meta.get("num_hands", _infer_num_hands(init_state)))

    hamer_raw = _try_load_npz(meta.get("hamer_npz"))
    hamer_pkl = _try_load_hamer_pkl(seq_name, meta)
    mano_local_init = build_mano_local_init(init_state, seq_len, num_hands)
    kp_2d, kp_source = build_kp_2d(
        seq_name=seq_name,
        hamer_pkl=hamer_pkl,
        hamer_raw=hamer_raw,
        init_state=init_state,
        seq_len=seq_len,
        num_hands=num_hands,
    )
    img_feat, img_feat_source = build_img_feat(
        hamer_pkl=hamer_pkl,
        kp_2d=kp_2d,
        seq_len=seq_len,
        num_hands=num_hands,
        img_feat_dim=img_feat_dim,
    )
    cam_init = build_cam_init(init_state)
    uncertainty = build_uncertainty(kp_2d, init_state, seq_len, num_hands)

    os.makedirs(output_dir, exist_ok=True)
    save_torch(os.path.join(output_dir, "mano_local_init.pth"), {"mano_local_init": mano_local_init})
    save_torch(os.path.join(output_dir, "kp_2d.pth"), {"kp_2d": kp_2d})
    save_torch(os.path.join(output_dir, "img_feat.pth"), {"img_feat": img_feat})
    save_torch(os.path.join(output_dir, "cam_init.pth"), cam_init)
    save_torch(os.path.join(output_dir, "uncertainty.pth"), {"uncertainty": uncertainty})

    summary = {
        "seq_name": seq_name,
        "seq_len": seq_len,
        "num_hands": num_hands,
        "processed_dir": processed_dir,
        "output_dir": output_dir,
        "img_feat_dim": img_feat_dim,
        "sources": {
            "mano_local_init": "processed/init_state.pth",
            "cam_init": "processed/init_state.pth",
            "kp_2d": kp_source,
            "img_feat": img_feat_source,
            "uncertainty": "kp_confidence_and_visibility",
        },
        "shapes": {
            "mano_local_init": list(mano_local_init.shape),
            "kp_2d": list(kp_2d.shape),
            "img_feat": list(img_feat.shape),
            "uncertainty": list(uncertainty.shape),
            "cam_R": list(cam_init["cam_R"].shape),
            "cam_t": list(cam_init["cam_t"].shape),
            "intrins": list(cam_init["intrins"].shape),
        },
    }
    save_json(done_path, summary)
    return summary


def build_mano_local_init(
    init_state: Dict[str, torch.Tensor],
    seq_len: int,
    num_hands: int,
) -> torch.Tensor:
    """Pack local MANO-like init into (B, T, 61).

    Layout: root_orient(3), pose_body(45), betas(10), trans(3).
    """

    root = _as_btd(init_state.get("root_orient"), seq_len, num_hands, 3)
    pose = init_state.get("pose_body")
    if pose is None:
        pose_flat = torch.zeros(num_hands, seq_len, 45)
    else:
        pose = torch.as_tensor(pose).float()
        if pose.ndim == 4:
            pose_flat = pose.reshape(pose.shape[0], pose.shape[1], -1)
        elif pose.ndim == 3:
            pose_flat = pose
        else:
            raise ValueError(f"pose_body must be 3D/4D, got {tuple(pose.shape)}")
    trans = _as_btd(init_state.get("trans"), seq_len, num_hands, 3)
    betas = torch.as_tensor(init_state.get("betas", torch.zeros(num_hands, 10))).float()
    if betas.ndim == 1:
        betas = betas.view(1, -1).expand(num_hands, -1)
    betas_bt = betas[:, None, :].expand(-1, seq_len, -1)
    return torch.cat([root, pose_flat, betas_bt, trans], dim=-1).contiguous()


def build_kp_2d(
    *,
    seq_name: str,
    hamer_pkl: Optional[Dict[str, Any]],
    hamer_raw: Optional[Dict[str, np.ndarray]],
    init_state: Dict[str, torch.Tensor],
    seq_len: int,
    num_hands: int,
) -> tuple[torch.Tensor, str]:
    """Return keypoints as (B, T, 21, 3), or zeros if unavailable."""

    if hamer_pkl:
        kp = _kp2d_from_hamer_pkl(hamer_pkl, seq_len, num_hands)
        if kp[..., 2].sum() > 0:
            return kp, "hamer_out_pkl:extra_data"

    if hamer_raw:
        for key in KP2D_KEYS:
            if key not in hamer_raw:
                continue
            try:
                kp = _normalize_kp2d(hamer_raw[key], seq_len, num_hands)
                return kp, f"hamer_npz:{key}"
            except ValueError:
                continue

    # Fallback confidence from visibility, coordinates unknown at this stage.
    kp = torch.zeros(num_hands, seq_len, 21, 3, dtype=torch.float32)
    vis = init_state.get("vis_mask")
    if vis is not None:
        vis_t = torch.as_tensor(vis).float()
        if vis_t.shape[:2] == (num_hands, seq_len):
            kp[..., 2] = vis_t[:, :, None]
    return kp, "zeros_placeholder"


def build_img_feat(
    *,
    hamer_pkl: Optional[Dict[str, Any]],
    kp_2d: torch.Tensor,
    seq_len: int,
    num_hands: int,
    img_feat_dim: int,
) -> tuple[torch.Tensor, str]:
    """Build deterministic image ROI descriptors with shape (B, T, D).

    This is a lightweight substitute for frozen ViT ROI embeddings. It uses
    actual image crops and keypoint-derived boxes, so it is suitable for
    debugging memory/refinement plumbing without running another heavy model.
    """

    img_feat = torch.zeros(num_hands, seq_len, img_feat_dim, dtype=torch.float32)
    if not hamer_pkl:
        return img_feat, "zeros_placeholder"

    frames = _sorted_hamer_frames(hamer_pkl)
    for t, frame_path in enumerate(frames[:seq_len]):
        if not os.path.isfile(frame_path):
            continue
        image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        for hand_idx in range(num_hands):
            crop = _crop_from_keypoints(image, kp_2d[hand_idx, t].numpy())
            img_feat[hand_idx, t] = _roi_descriptor(crop, img_feat_dim)
    return img_feat, "hamer_images:roi_gray_grad_512"


def build_cam_init(init_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in ("cam_R", "cam_t", "intrins", "world_scale"):
        if key in init_state:
            out[key] = torch.as_tensor(init_state[key]).float()
    return out


def build_uncertainty(
    kp_2d: torch.Tensor,
    init_state: Dict[str, torch.Tensor],
    seq_len: int,
    num_hands: int,
) -> torch.Tensor:
    conf = kp_2d[..., 2].mean(dim=-1, keepdim=True)
    if torch.all(conf == 0):
        vis = init_state.get("vis_mask")
        if vis is not None:
            vis_t = torch.as_tensor(vis).float()
            if vis_t.shape[:2] == (num_hands, seq_len):
                conf = vis_t.unsqueeze(-1)
    # Higher value means less certain.
    return (1.0 - conf.clamp(0.0, 1.0)).contiguous()


def discover_processed_dirs(processed_root: str) -> List[str]:
    processed_root = os.path.abspath(processed_root)
    if not os.path.isdir(processed_root):
        raise FileNotFoundError(processed_root)
    out = []
    for name in sorted(os.listdir(processed_root)):
        d = os.path.join(processed_root, name)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "init_state.pth")):
            out.append(d)
    return out


def dump_many(
    processed_dirs: Iterable[str],
    features_root: str,
    *,
    img_feat_dim: int = 512,
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    summaries = []
    for processed_dir in processed_dirs:
        meta = _load_json(os.path.join(processed_dir, "meta.json"))
        output_dir = os.path.join(features_root, meta["seq_name"])
        summaries.append(
            dump_sequence_features(
                processed_dir,
                output_dir,
                img_feat_dim=img_feat_dim,
                overwrite=overwrite,
            )
        )
    return summaries


def _normalize_kp2d(arr: np.ndarray, seq_len: int, num_hands: int) -> torch.Tensor:
    x = torch.as_tensor(arr).float()
    if x.ndim == 4 and x.shape[-2:] == (21, 3):
        if x.shape[:2] == (num_hands, seq_len):
            return x.contiguous()
        if x.shape[:2] == (seq_len, num_hands):
            return x.permute(1, 0, 2, 3).contiguous()
    if x.ndim == 4 and x.shape[-1] == 2 and x.shape[-2] == 21:
        if x.shape[:2] == (num_hands, seq_len):
            conf = torch.ones(*x.shape[:-1], 1)
            return torch.cat([x, conf], dim=-1).contiguous()
        if x.shape[:2] == (seq_len, num_hands):
            x = x.permute(1, 0, 2, 3)
            conf = torch.ones(*x.shape[:-1], 1)
            return torch.cat([x, conf], dim=-1).contiguous()
    raise ValueError(f"Cannot normalize kp_2d shape {tuple(x.shape)}")


def _kp2d_from_hamer_pkl(
    hamer_pkl: Dict[str, Any],
    seq_len: int,
    num_hands: int,
) -> torch.Tensor:
    kp = torch.zeros(num_hands, seq_len, 21, 3, dtype=torch.float32)
    frames = _sorted_hamer_frames(hamer_pkl)
    for t, frame_path in enumerate(frames[:seq_len]):
        entry = hamer_pkl.get(frame_path, {})
        tids = np.asarray(entry.get("tid", []), dtype=np.int64)
        extras = entry.get("extra_data", [])
        for det_idx, tid in enumerate(tids.tolist()):
            if tid < 0 or tid >= num_hands or det_idx >= len(extras):
                continue
            arr = np.asarray(extras[det_idx], dtype=np.float32)
            if arr.shape == (21, 3):
                kp[int(tid), t] = torch.from_numpy(arr)
    return kp


def _sorted_hamer_frames(hamer_pkl: Dict[str, Any]) -> List[str]:
    def key_fn(path: str) -> tuple[str, int]:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            return os.path.dirname(path), int(stem)
        except ValueError:
            return os.path.dirname(path), 0

    return sorted(hamer_pkl.keys(), key=key_fn)


def _crop_from_keypoints(image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    valid = keypoints[:, 2] > 0.1
    if valid.sum() >= 3:
        xy = keypoints[valid, :2]
        x1, y1 = xy.min(axis=0)
        x2, y2 = xy.max(axis=0)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        size = max(x2 - x1, y2 - y1, 1.0) * 1.6
    else:
        cx, cy = w * 0.5, h * 0.5
        size = min(w, h) * 0.5
    x1 = int(max(0, round(cx - size * 0.5)))
    y1 = int(max(0, round(cy - size * 0.5)))
    x2 = int(min(w, round(cx + size * 0.5)))
    y2 = int(min(h, round(cy + size * 0.5)))
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def _roi_descriptor(crop: np.ndarray, dim: int) -> torch.Tensor:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad = grad / (float(grad.max()) + 1e-6)
    feat = np.concatenate([gray.reshape(-1), grad.reshape(-1)], axis=0)
    if feat.shape[0] < dim:
        feat = np.pad(feat, (0, dim - feat.shape[0]))
    elif feat.shape[0] > dim:
        feat = feat[:dim]
    return torch.from_numpy(feat.astype(np.float32))


def _as_btd(
    x: Optional[torch.Tensor],
    seq_len: int,
    num_hands: int,
    dim: int,
) -> torch.Tensor:
    if x is None:
        return torch.zeros(num_hands, seq_len, dim)
    t = torch.as_tensor(x).float()
    if t.shape == (num_hands, seq_len, dim):
        return t
    if t.shape == (seq_len, num_hands, dim):
        return t.permute(1, 0, 2).contiguous()
    raise ValueError(f"Expected (B,T,{dim}) or (T,B,{dim}), got {tuple(t.shape)}")


def _infer_num_hands(init_state: Dict[str, torch.Tensor]) -> int:
    for key in ("trans", "root_orient", "pose_body", "is_right"):
        if key in init_state and hasattr(init_state[key], "shape"):
            return int(init_state[key].shape[0])
    return 2


def _try_load_npz(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    if not path or not os.path.isfile(path):
        return None
    return load_npz(path)


def _try_load_hamer_pkl(seq_name: str, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = [
        f"/extra/SuC/dynhamr_io/dynhamr/hamer_out/{seq_name}/{seq_name}.pkl",
        f"/data/SuC/Dyn-HaMR/test/dynhamr/hamer_out/{seq_name}/{seq_name}.pkl",
    ]
    log_dir = meta.get("dynhamr_log_dir")
    if log_dir:
        hydra_cfg = os.path.join(log_dir, ".hydra", "config.yaml")
        if os.path.isfile(hydra_cfg):
            try:
                import yaml

                with open(hydra_cfg, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                root = cfg.get("data", {}).get("root")
                if root:
                    candidates.insert(0, os.path.join(root, "dynhamr", "hamer_out", seq_name, f"{seq_name}.pkl"))
            except Exception:
                pass
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
