"""Losses and sparse-target utilities for MA-HaMR Step 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from mahmr.geometry import MANOJointLayer
from mahmr.models.memory import MANO_LOCAL_DIM


# Per-hand mano_local residual layout (61 dims).
_COMPONENT_SPANS = {
    "root_orient": (0, 3),
    "pose_body": (3, 48),
    "betas": (48, 58),
    "trans": (58, 61),
}
# Typical residual magnitudes, used to normalize the distillation loss. Without
# this, large-magnitude corrections (trans in meters, betas) dominate the
# gradient and drown small but important articulation corrections (root/pose in
# radians), which were observed to actually DEGRADE after refinement.
_COMPONENT_NORM = {
    "root_orient": 0.15,
    "pose_body": 0.06,
    "betas": 0.25,
    "trans": 1.0,
}
# Relative distillation weights per component (after normalization). pose_body
# is the largest remaining gap to expert quality, so it gets the most emphasis.
_COMPONENT_DISTILL_W = {
    "root_orient": 1.2,
    "pose_body": 2.0,
    "betas": 0.3,
    "trans": 1.2,
}
# Residual-magnitude prior applied on ALL frames (not just expert frames):
# articulation corrections should stay small unless the data demands otherwise,
# preventing the network from corrupting already-good pose/root on unlabeled
# frames. trans/betas need large per-sequence corrections, so they are free.
_COMPONENT_PRIOR_W = {
    "root_orient": 0.8,
    "pose_body": 1.5,
    "betas": 0.2,
    "trans": 0.15,
}


@dataclass
class LossWeights:
    distill: float = 1.0
    proj: float = 1.0
    proj_huber: float = 0.0
    proj_huber_delta: float = 50.0
    bone: float = 0.5
    consistency: float = 0.2
    scale: float = 0.2
    scale_distill: float = 0.1
    scale_anchor: float = 1.0
    prior: float = 0.0


class MAHaMRLoss(nn.Module):
    """Hybrid supervision objective.

    This is the first trainable Step 4 implementation. It uses exact
    residual distillation on sparse expert frames and geometry-inspired proxy
    losses that do not require a MANO layer. Once MANO joints are integrated,
    ``loss_proj`` and ``loss_bone`` can be swapped for mesh/joint versions
    without changing the training loop contract.
    """

    def __init__(
        self,
        *,
        weights: Optional[Dict[str, float]] = None,
        mano_layer: Optional[MANOJointLayer] = None,
        robust_sigma: float = 100.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weights = LossWeights(**(weights or {}))
        self.mano_layer = mano_layer
        self.robust_sigma = robust_sigma
        self.eps = eps

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        geometry = self._forward_geometry(output, batch)
        losses: Dict[str, torch.Tensor] = {}
        losses["loss_distill"] = self.loss_distill(output, targets)
        losses["loss_proj"] = self.loss_projection(output, batch, geometry)
        losses["loss_proj_huber"] = self.loss_projection_huber(output, batch, geometry)
        losses["loss_bone"] = self.loss_bone(output, geometry)
        losses["loss_consistency"] = self.loss_memory_consistency(output)
        losses["loss_scale"] = self.loss_scale_smooth(output)
        losses["loss_prior"] = self.loss_residual_prior(output)

        total = output["packed_residual"].new_tensor(0.0)
        total = total + self.weights.distill * losses["loss_distill"]
        total = total + self.weights.proj * losses["loss_proj"]
        total = total + self.weights.proj_huber * losses["loss_proj_huber"]
        total = total + self.weights.bone * losses["loss_bone"]
        total = total + self.weights.consistency * losses["loss_consistency"]
        total = total + self.weights.scale * losses["loss_scale"]
        total = total + self.weights.prior * losses["loss_prior"]
        losses["loss"] = total
        return losses

    def _forward_geometry(self, output: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self.mano_layer is None:
            return {}
        init_state = batch.get("init_state", {})
        is_right = init_state.get("is_right") if isinstance(init_state, dict) else None
        joints = self.mano_layer(output["mano_local_refined"], is_right=is_right)
        return {"joints3d": joints}

    def loss_distill(self, output: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred = output["packed_residual"]
        target = targets["packed_residual"]
        mask = targets["valid_mask"].bool()
        if not torch.any(mask):
            return pred.sum() * 0.0

        num_hands = (pred.shape[-1] - 1) // MANO_LOCAL_DIM
        pred_hand = pred[..., : num_hands * MANO_LOCAL_DIM].view(*pred.shape[:2], num_hands, MANO_LOCAL_DIM)
        tgt_hand = target[..., : num_hands * MANO_LOCAL_DIM].view(*target.shape[:2], num_hands, MANO_LOCAL_DIM)
        pred_m = pred_hand[mask]
        tgt_m = tgt_hand[mask]

        # Component-balanced distillation: normalize each component by its typical
        # magnitude so every part of the MANO state receives comparable gradient.
        mano_loss = pred.new_tensor(0.0)
        for name, (a, b) in _COMPONENT_SPANS.items():
            norm = _COMPONENT_NORM[name]
            mano_loss = mano_loss + _COMPONENT_DISTILL_W[name] * F.smooth_l1_loss(
                pred_m[..., a:b] / norm, tgt_m[..., a:b] / norm
            )

        pred_scale = output["world_scale_refined"][mask].clamp_min(self.eps)
        target_log_abs = targets["target_log_abs_world_scale"][mask]
        scale_loss = F.smooth_l1_loss(torch.log(pred_scale), target_log_abs)
        return mano_loss + self.weights.scale_distill * scale_loss

    def loss_residual_prior(self, output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Keep articulation corrections small on all frames (stay near init)."""
        pred = output["packed_residual"]
        num_hands = (pred.shape[-1] - 1) // MANO_LOCAL_DIM
        pred_hand = pred[..., : num_hands * MANO_LOCAL_DIM].view(*pred.shape[:2], num_hands, MANO_LOCAL_DIM)
        loss = pred.new_tensor(0.0)
        for name, (a, b) in _COMPONENT_SPANS.items():
            w = _COMPONENT_PRIOR_W[name]
            if w == 0.0:
                continue
            loss = loss + w * (pred_hand[..., a:b] / _COMPONENT_NORM[name]).abs().mean()
        return loss

    def loss_projection(
        self,
        output: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        geometry: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        features = batch["features"]
        kp = features["kp_2d"].to(output["mano_local_refined"].device).float()
        conf = kp[..., 2].clamp(0.0, 1.0)
        if torch.all(conf <= 0):
            return output["packed_residual"].sum() * 0.0

        if "joints3d" in geometry:
            pred_xy = _project_points(
                geometry["joints3d"],
                features["cam_init"],
                output["world_scale_refined"],
            )
            err = (pred_xy - kp[..., :2]).norm(dim=-1)
            weight = conf
        else:
            target_xy = (kp[..., :2] * conf.unsqueeze(-1)).sum(dim=-2)
            target_xy = target_xy / conf.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            pred_xy = _project_hand_translation(
                output["mano_local_refined"][..., -3:],
                features["cam_init"],
                output["world_scale_refined"],
            )
            err = (pred_xy - target_xy).abs().sum(dim=-1)
            weight = conf.mean(dim=-1)

        # Normalize the Geman-McClure response to [0, 1) so the loss is in a
        # stable, interpretable range regardless of sigma (1.0 ~ a gross outlier,
        # ~0 ~ a near-perfect reprojection). This lets ``weights.proj`` be a sane
        # O(0.1) value instead of a tiny number fighting raw pixel^2 magnitudes.
        robust = geman_mcclure(err, sigma=self.robust_sigma) / (self.robust_sigma ** 2)
        return (robust * weight).sum() / weight.sum().clamp_min(self.eps)

    def loss_projection_huber(
        self,
        output: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        geometry: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Non-saturating visual fitting loss in pixel space.

        ``loss_projection`` is deliberately robust and saturates for very large
        errors, which is good for stability but weak when the visual goal is to
        pull a visibly bad hand back onto the 2D observations. This companion
        loss keeps a linear tail beyond ``proj_huber_delta`` pixels so large
        reprojection errors still produce useful gradients.
        """
        if self.mano_layer is None or "joints3d" not in geometry:
            return output["packed_residual"].sum() * 0.0

        features = batch["features"]
        kp = features["kp_2d"].to(output["mano_local_refined"].device).float()
        conf = kp[..., 2].clamp(0.0, 1.0)
        if torch.all(conf <= 0):
            return output["packed_residual"].sum() * 0.0

        pred_xy = _project_points(
            geometry["joints3d"],
            features["cam_init"],
            output["world_scale_refined"],
        )
        err = (pred_xy - kp[..., :2]).norm(dim=-1)
        delta = max(float(self.weights.proj_huber_delta), self.eps)
        scaled = err / delta
        huber = torch.where(scaled < 1.0, 0.5 * scaled.pow(2), scaled - 0.5)
        return (huber * conf).sum() / conf.sum().clamp_min(self.eps)

    def loss_bone(self, output: Dict[str, torch.Tensor], geometry: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "joints3d" in geometry:
            bones = _mano_bone_lengths(geometry["joints3d"])
            center = bones.median(dim=2, keepdim=True).values
            return (bones - center).pow(2).mean()

        # MANO beta is identity-level and should not fluctuate over time.
        betas = output["mano_local_refined"][..., 48:58]
        center = betas.mean(dim=2, keepdim=True)
        beta_var = (betas - center).pow(2).mean()

        # Pose norm acts as a lightweight kinematic regularizer until real MANO
        # bone lengths are available.
        pose = output["mano_local_refined"][..., 3:48]
        pose_norm = pose.view(*pose.shape[:3], 15, 3).norm(dim=-1)
        pose_center = pose_norm.mean(dim=2, keepdim=True)
        pose_var = (pose_norm - pose_center).pow(2).mean()
        return beta_var + 0.05 * pose_var

    def loss_memory_consistency(self, output: Dict[str, torch.Tensor]) -> torch.Tensor:
        context = output["memory_context"].detach()
        pred = output["packed_residual"]
        valid = context.abs().mean(dim=-1) > 0
        if not torch.any(valid):
            return pred.sum() * 0.0
        return F.smooth_l1_loss(pred[valid], context[valid])

    def loss_scale_smooth(self, output: Dict[str, torch.Tensor]) -> torch.Tensor:
        # world_scale is a per-sequence near-constant in Dyn-HaMR. We regularize
        # it on three fronts: (1) first/second-order temporal smoothness, and
        # (2) an anchor pulling every frame toward the per-window median so the
        # 90% of frames without an expert label cannot drift freely. The median
        # is detached so the distillation term (on sparse expert frames) is what
        # sets the absolute level, while this term only fights drift around it.
        scale = output["world_scale_refined"]
        loss = scale.sum() * 0.0
        if scale.shape[1] >= 2:
            first = scale[:, 1:] - scale[:, :-1]
            loss = loss + first.pow(2).mean()
        if scale.shape[1] >= 3:
            second = scale[:, 2:] - 2 * scale[:, 1:-1] + scale[:, :-2]
            loss = loss + second.pow(2).mean()
        ref = scale.median(dim=1, keepdim=True).values.detach()
        loss = loss + self.weights.scale_anchor * (scale - ref).pow(2).mean()
        return loss


def build_window_residual_targets(batch: Dict[str, Any], *, device: torch.device | str) -> Dict[str, torch.Tensor]:
    """Align sparse expert residuals to the current dense window.

    Returns tensors with the same packed residual layout as ``MAHaMRRefiner``:
    ``(N, T, 2 * 61 + 1)`` and a boolean valid mask ``(N, T)``.
    """

    device = torch.device(device)
    mano_init = batch["features"]["mano_local_init"].to(device).float()
    if mano_init.ndim == 3:
        mano_init = mano_init.unsqueeze(0)
    batch_size, num_hands, seq_len, _ = mano_init.shape
    d_val = num_hands * MANO_LOCAL_DIM + 1
    packed = torch.zeros(batch_size, seq_len, d_val, dtype=torch.float32, device=device)
    valid = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    starts = _as_batch_vector(batch["start"], batch_size).long()
    ends = _as_batch_vector(batch["end"], batch_size).long()
    frame_indices = torch.as_tensor(batch["valid_mask"]["frame_indices"]).long()
    if frame_indices.ndim == 1:
        frame_indices = frame_indices.unsqueeze(0).expand(batch_size, -1)

    residuals = batch["residuals"]
    expert_label = batch["expert_label"]
    init_state = batch["init_state"]

    root = residuals["delta_root_orient"].to(device).float()
    pose = residuals["delta_pose_body"].to(device).float()
    trans = residuals["delta_trans"].to(device).float()
    if root.ndim == 3:
        root = root.unsqueeze(0)
        pose = pose.unsqueeze(0)
        trans = trans.unsqueeze(0)

    delta_betas = _delta_betas(expert_label, init_state, batch_size, num_hands, device)
    delta_scale = residuals.get("delta_world_scale")
    delta_scale = _normalize_delta_scale(delta_scale, batch_size, device)
    init_scale = _normalize_init_scale(batch["features"]["cam_init"].get("world_scale"), batch_size, device)

    for n in range(batch_size):
        start, end = int(starts[n].item()), int(ends[n].item())
        for ridx, frame_idx in enumerate(frame_indices[n].tolist()):
            if frame_idx < start or frame_idx >= end:
                continue
            local_t = frame_idx - start
            hand_parts = []
            for h in range(num_hands):
                hand_parts.append(
                    torch.cat(
                        [
                            root[n, h, ridx].reshape(-1),
                            pose[n, h, ridx].reshape(-1),
                            delta_betas[n, h].reshape(-1),
                            trans[n, h, ridx].reshape(-1),
                        ],
                        dim=0,
                    )
                )
            packed[n, local_t, :-1] = torch.cat(hand_parts, dim=0)
            packed[n, local_t, -1] = delta_scale[n]
            valid[n, local_t] = True

    target_scale_signed = init_scale[:, None, :] + packed[..., -1:]
    target_log_abs = torch.log(target_scale_signed.abs().clamp_min(1e-6))
    return {
        "packed_residual": packed,
        "valid_mask": valid,
        "target_log_abs_world_scale": target_log_abs,
    }


def geman_mcclure(error: torch.Tensor, *, sigma: float = 100.0) -> torch.Tensor:
    sigma2 = float(sigma) ** 2
    err2 = error.pow(2)
    return sigma2 * err2 / (err2 + sigma2)


def _project_points(
    points_world: torch.Tensor,
    cam_init: Dict[str, torch.Tensor],
    world_scale: torch.Tensor,
) -> torch.Tensor:
    device = points_world.device
    cam_R = cam_init.get("cam_R")
    cam_t = cam_init.get("cam_t")
    intrins = cam_init.get("intrins")
    if cam_R is None or cam_t is None or intrins is None:
        return points_world[..., :2]

    cam_R = torch.as_tensor(cam_R, device=device).float()
    cam_t = torch.as_tensor(cam_t, device=device).float()
    intrins = torch.as_tensor(intrins, device=device).float()
    if cam_R.ndim == 4:
        cam_R = cam_R.unsqueeze(0)
    if cam_t.ndim == 3:
        cam_t = cam_t.unsqueeze(0)
    if intrins.ndim == 1:
        intrins = intrins.view(1, 1, 1, 1, 4).expand(
            points_world.shape[0], points_world.shape[1], points_world.shape[2], points_world.shape[3], 4
        )
    elif intrins.ndim == 2:
        intrins = intrins[:, None, None, None, :].expand(
            -1, points_world.shape[1], points_world.shape[2], points_world.shape[3], -1
        )

    scale = world_scale[:, None, :, None, :].expand(-1, points_world.shape[1], -1, points_world.shape[3], -1)
    cam_t = cam_t[:, :, :, None, :].expand(-1, -1, -1, points_world.shape[3], -1)
    cam_R = cam_R[:, :, :, None, :, :].expand(-1, -1, -1, points_world.shape[3], -1, -1)
    cam = torch.matmul(cam_R, points_world.unsqueeze(-1)).squeeze(-1) + scale * cam_t
    z = cam[..., 2].clamp_min(1e-3)
    fx, fy, cx, cy = [intrins[..., i] for i in range(4)]
    x = fx * cam[..., 0] / z + cx
    y = fy * cam[..., 1] / z + cy
    return torch.stack([x, y], dim=-1)


def _mano_bone_lengths(joints: torch.Tensor) -> torch.Tensor:
    # OpenPose-style MANO order:
    # wrist, thumb chain, index chain, middle chain, ring chain, pinky chain.
    edges = torch.tensor(
        [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [0, 9], [9, 10], [10, 11], [11, 12],
            [0, 13], [13, 14], [14, 15], [15, 16],
            [0, 17], [17, 18], [18, 19], [19, 20],
        ],
        device=joints.device,
        dtype=torch.long,
    )
    diffs = joints[..., edges[:, 1], :] - joints[..., edges[:, 0], :]
    return diffs.norm(dim=-1)


def _project_hand_translation(
    trans_world: torch.Tensor,
    cam_init: Dict[str, torch.Tensor],
    world_scale: torch.Tensor,
) -> torch.Tensor:
    device = trans_world.device
    cam_R = cam_init.get("cam_R")
    cam_t = cam_init.get("cam_t")
    intrins = cam_init.get("intrins")
    if cam_R is None or cam_t is None or intrins is None:
        return trans_world[..., :2]

    cam_R = torch.as_tensor(cam_R, device=device).float()
    cam_t = torch.as_tensor(cam_t, device=device).float()
    intrins = torch.as_tensor(intrins, device=device).float()
    if cam_R.ndim == 4:
        cam_R = cam_R.unsqueeze(0)
    if cam_t.ndim == 3:
        cam_t = cam_t.unsqueeze(0)
    if intrins.ndim == 1:
        intrins = intrins.view(1, 1, 1, 4).expand(trans_world.shape[0], trans_world.shape[1], trans_world.shape[2], 4)
    elif intrins.ndim == 2:
        intrins = intrins[:, None, None, :].expand(-1, trans_world.shape[1], trans_world.shape[2], -1)

    scale = world_scale[:, None].expand(-1, trans_world.shape[1], -1, -1)
    cam = torch.matmul(cam_R, trans_world.unsqueeze(-1)).squeeze(-1) + scale * cam_t
    z = cam[..., 2].clamp_min(1e-3)
    fx, fy, cx, cy = [intrins[..., i] for i in range(4)]
    x = fx * cam[..., 0] / z + cx
    y = fy * cam[..., 1] / z + cy
    return torch.stack([x, y], dim=-1)


def _delta_betas(
    expert_label: Dict[str, torch.Tensor],
    init_state: Dict[str, torch.Tensor],
    batch_size: int,
    num_hands: int,
    device: torch.device,
) -> torch.Tensor:
    expert = expert_label.get("betas")
    init = init_state.get("betas")
    if expert is None or init is None:
        return torch.zeros(batch_size, num_hands, 10, device=device)
    expert = torch.as_tensor(expert, device=device).float()
    init = torch.as_tensor(init, device=device).float()
    if expert.ndim == 2:
        expert = expert.unsqueeze(0)
    if init.ndim == 2:
        init = init.unsqueeze(0)
    return expert[:, :num_hands] - init[:, :num_hands]


def _normalize_delta_scale(delta_scale: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
    if delta_scale is None:
        return torch.zeros(batch_size, device=device)
    scale = torch.as_tensor(delta_scale, device=device).float()
    if scale.ndim == 0:
        scale = scale.view(1)
    if scale.ndim == 2 and scale.shape[-1] == 1:
        scale = scale[:, 0]
    if scale.numel() == 1 and batch_size > 1:
        scale = scale.expand(batch_size)
    return scale.reshape(batch_size)


def _normalize_init_scale(init_scale: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
    if init_scale is None:
        return torch.ones(batch_size, 1, device=device)
    scale = torch.as_tensor(init_scale, device=device).float()
    if scale.ndim == 0:
        scale = scale.view(1, 1)
    elif scale.ndim == 1:
        scale = scale.view(1, -1)
    if scale.shape[0] == 1 and batch_size > 1:
        scale = scale.expand(batch_size, -1)
    return scale[:, :1]


def _as_batch_vector(value: Any, batch_size: int) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 0:
        tensor = tensor.view(1)
    if tensor.numel() == 1 and batch_size > 1:
        tensor = tensor.expand(batch_size)
    return tensor.reshape(batch_size)
