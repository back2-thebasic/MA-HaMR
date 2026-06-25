"""Reprojection self-verification safeguard for MA-HaMR inference.

The amortized refiner can occasionally degrade on out-of-distribution sequences
(e.g. a clip whose true ``world_scale`` lies far outside the training range). In
those rare cases the predicted residual moves the hand *away* from the observed
2D keypoints. Because the 2D keypoints ``kp_2d`` are part of the streaming input,
we can cheaply verify each prediction against them and fall back to the
initialization wherever the refinement is not supported by the observation.

This is a per-frame trust region: it never lets the refined output be (much)
worse than the initialization under the observable 2D metric, while keeping the
full refinement gain on the majority of frames where it helps.

The gate is split in two coherent stages because ``world_scale`` is shared
across both hands while the articulated MANO residual is per-hand:

1. ``world_scale`` gate (per time step): scaling the camera translation moves
   *both* hands together, so a bad scale is the dominant catastrophic-failure
   mode. We decide it from the aggregate reprojection error over both hands.
2. MANO gate (per hand, per time step): with the scale already settled, we keep
   the refined articulation only where it is consistent with ``kp_2d``.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch


def _mean_reproj_error(
    xy: torch.Tensor,
    kp: torch.Tensor,
    conf_thresh: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean reprojection error per (hand, time) over confident keypoints.

    Args:
        xy: projected joints, shape (H, T, K, 2).
        kp: detected keypoints with confidence, shape (H, T, K, 3).
        conf_thresh: keypoints with confidence above this count.

    Returns:
        (mean_err, count) each shaped (H, T). ``mean_err`` is +inf where the
        projection is non-finite so that such frames always lose the gate.
    """
    conf = kp[..., 2] > conf_thresh
    err = torch.linalg.norm(xy - kp[..., :2], dim=-1)
    finite = torch.isfinite(err)
    valid = conf & finite
    count = valid.sum(dim=-1)
    safe_err = torch.where(valid, err, torch.zeros_like(err))
    summed = safe_err.sum(dim=-1)
    mean = summed / count.clamp_min(1)
    # Frames whose projection blew up (non-finite anywhere) must never win.
    blew_up = (~finite).any(dim=-1)
    mean = torch.where(blew_up, torch.full_like(mean, float("inf")), mean)
    return mean, count


def reprojection_safeguard(
    output: Dict[str, torch.Tensor],
    *,
    mano_local_init: torch.Tensor,
    init_world_scale: torch.Tensor,
    kp_2d: torch.Tensor,
    cam_init: Dict[str, torch.Tensor],
    mano_layer,
    project_fn: Callable,
    is_right: Optional[torch.Tensor] = None,
    margin: float = 0.1,
    conf_thresh: float = 0.1,
    min_kp: int = 4,
) -> Dict[str, torch.Tensor]:
    """Gate refined predictions against the 2D observation.

    All tensors are batched with a leading batch dim of 1.

    Args:
        output: refiner output containing ``mano_local_refined`` (1,H,T,61) and
            ``world_scale_refined`` (1,T,1).
        mano_local_init: (1,H,T,61) initialization fed to the refiner.
        init_world_scale: (1,T,1) initialization world scale.
        kp_2d: (1,H,T,K,3) detected keypoints with confidence.
        cam_init: camera dict consumed by ``project_fn``.
        mano_layer: MANO joint layer producing (1,H,T,K,3) joints.
        project_fn: projection callable ``(joints, cam, world_scale) -> (1,H,T,K,2)``.
        is_right: (1,H,T) handedness flags for the MANO layer.
        margin: keep refined only if its error <= init error * (1 + margin).
        conf_thresh: keypoint confidence threshold.
        min_kp: below this many confident keypoints the frame is unverifiable;
            we then keep refined only if it did not blow up.

    Returns:
        A new output dict with gated ``mano_local_refined`` / ``world_scale_refined``
        plus diagnostic ``safeguard_*`` fields. The original output is not mutated.
    """
    mano_refined = output["mano_local_refined"]
    scale_refined = output["world_scale_refined"]
    kp = kp_2d

    with torch.no_grad():
        init_joints = mano_layer(mano_local_init, is_right=is_right)
        refined_joints = mano_layer(mano_refined, is_right=is_right)
        init_xy = project_fn(init_joints, cam_init, init_world_scale)[0]
        refined_xy = project_fn(refined_joints, cam_init, scale_refined)[0]

    init_err, count = _mean_reproj_error(init_xy, kp[0], conf_thresh)
    refined_err, _ = _mean_reproj_error(refined_xy, kp[0], conf_thresh)

    # --- Stage 1: world_scale gate (per time step, aggregated over hands). ---
    valid_t = count >= min_kp  # (H,T)
    any_valid_t = valid_t.any(dim=0)  # (T,)
    init_t = _aggregate_over_hands(init_err, valid_t)
    refined_t = _aggregate_over_hands(refined_err, valid_t)
    keep_scale = refined_t <= init_t * (1.0 + margin)
    # Unverifiable time steps: keep refined scale only if it stays finite.
    keep_scale = torch.where(any_valid_t, keep_scale, torch.isfinite(refined_t))
    keep_scale_b = keep_scale.view(1, -1, 1)
    scale_gated = torch.where(keep_scale_b, scale_refined, init_world_scale)

    # Re-project the refined articulation with the (possibly reverted) scale so
    # the MANO gate is evaluated under the scale that will actually be used.
    with torch.no_grad():
        refined_xy2 = project_fn(refined_joints, cam_init, scale_gated)[0]
    refined_err2, _ = _mean_reproj_error(refined_xy2, kp[0], conf_thresh)

    # --- Stage 2: MANO gate (per hand, per time step). ---
    keep_mano = refined_err2 <= init_err * (1.0 + margin)  # (H,T)
    keep_mano = torch.where(valid_t, keep_mano, torch.isfinite(refined_err2))
    keep_mano_b = keep_mano.unsqueeze(0).unsqueeze(-1)  # (1,H,T,1)
    mano_gated = torch.where(keep_mano_b, mano_refined, mano_local_init)

    gated = dict(output)
    gated["mano_local_refined"] = mano_gated
    gated["world_scale_refined"] = scale_gated
    gated["safeguard_keep_scale_frac"] = keep_scale.float().mean()
    gated["safeguard_keep_mano_frac"] = keep_mano.float().mean()

    # Keep packed_residual consistent with the gated output so residual-to-expert
    # diagnostics reflect what is actually used: reverted entries get zero residual.
    packed = output.get("packed_residual")
    if packed is not None:
        num_hands = mano_refined.shape[1]
        mano_dim = mano_refined.shape[-1]
        hand = packed[..., : num_hands * mano_dim].view(*packed.shape[:2], num_hands, mano_dim)
        keep_hand = keep_mano.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # (1,T,H,1)
        hand = torch.where(keep_hand, hand, torch.zeros_like(hand))
        scale_delta = packed[..., -1:]
        scale_delta = torch.where(keep_scale.view(1, -1, 1), scale_delta, torch.zeros_like(scale_delta))
        gated["packed_residual"] = torch.cat(
            [hand.reshape(*packed.shape[:2], num_hands * mano_dim), scale_delta], dim=-1
        )
    return gated


def _aggregate_over_hands(err: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Reduce a per-(hand,time) error to per-time, ignoring invalid hands."""
    safe = torch.where(valid, err, torch.zeros_like(err))
    count = valid.sum(dim=0).clamp_min(1)
    mean = safe.sum(dim=0) / count
    no_valid = valid.sum(dim=0) == 0
    # If no hand is verifiable at a time step, propagate +inf so the gate reverts.
    blew = torch.isinf(err).any(dim=0)
    mean = torch.where(blew, torch.full_like(mean, float("inf")), mean)
    mean = torch.where(no_valid, torch.full_like(mean, float("inf")), mean)
    return mean
