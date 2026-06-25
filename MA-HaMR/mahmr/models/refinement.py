"""Amortized refinement network for MA-HaMR Step 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from mahmr.models.memory import (
    MANO_LOCAL_DIM,
    LongTermMemoryBank,
    MemoryConfig,
    MemoryKeyEncoder,
)


@dataclass
class RefinementConfig:
    num_hands: int = 2
    mano_dim: int = MANO_LOCAL_DIM
    kp_dim: int = 21 * 3
    img_feat_dim: int = 512
    d_mem: int = 128
    d_val: int = 123
    hidden_dim: int = 256
    num_blocks: int = 3
    dropout: float = 0.0
    scale_eps: float = 1e-4


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class AmortizedRefinementNet(nn.Module):
    """Predict Dyn-HaMR-style correction residuals from current state + memory."""

    def __init__(
        self,
        *,
        num_hands: int = 2,
        img_feat_dim: int = 512,
        d_val: int = 123,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_hands = num_hands
        self.img_feat_dim = img_feat_dim
        self.d_val = d_val
        obs_dim = (
            num_hands * MANO_LOCAL_DIM
            + num_hands * 21 * 3
            + num_hands * img_feat_dim
            + num_hands
            + num_hands * 3
            + 1
        )
        self.input = nn.Sequential(
            nn.LayerNorm(obs_dim + d_val),
            nn.Linear(obs_dim + d_val, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output = nn.Linear(hidden_dim, d_val)

    def forward(
        self,
        observation_t: torch.Tensor,
        memory_context_t: torch.Tensor,
    ) -> torch.Tensor:
        if memory_context_t.ndim == 3:
            memory_context_t = memory_context_t.squeeze(1)
        x = torch.cat([observation_t, memory_context_t], dim=-1)
        h = self.blocks(self.input(x))
        return self.output(h)


class MAHaMRRefiner(nn.Module):
    """Streaming MA-HaMR Step 3 wrapper.

    The wrapper is intentionally small: it reads Step 2 features, performs
    causal memory retrieval frame by frame, and predicts residual corrections.
    Training losses and MANO geometry are introduced in Step 4.
    """

    def __init__(
        self,
        *,
        num_hands: int = 2,
        img_feat_dim: int = 512,
        d_mem: int = 128,
        d_val: int = 123,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        topk: int = 16,
        exclude_recent: int = 30,
        max_memory_size: int = 4096,
        dropout: float = 0.0,
        scale_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.cfg = RefinementConfig(
            num_hands=num_hands,
            img_feat_dim=img_feat_dim,
            d_mem=d_mem,
            d_val=d_val,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            scale_eps=scale_eps,
        )
        self.key_encoder = MemoryKeyEncoder(num_hands=num_hands, d_mem=d_mem, hidden_dim=hidden_dim)
        self.memory = LongTermMemoryBank(
            d_mem=d_mem,
            d_val=d_val,
            topk=topk,
            exclude_recent=exclude_recent,
            max_size=max_memory_size,
            write_conf_threshold=0.0,
        )
        self.refine = AmortizedRefinementNet(
            num_hands=num_hands,
            img_feat_dim=img_feat_dim,
            d_val=d_val,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )

    def reset_memory(self) -> None:
        self.memory.reset()

    def forward(
        self,
        features: Dict[str, Any],
        *,
        memory_values: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
        reset_memory: bool = True,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        tensors = _normalize_features(features, self.cfg.num_hands, self.cfg.img_feat_dim)
        mano = tensors["mano_local_init"]
        batch_size, _, seq_len, _ = mano.shape

        if reset_memory:
            self.reset_memory()

        keys = self.key_encoder(mano)
        observations = _build_observation(tensors)
        if memory_values is not None:
            memory_values = _normalize_memory_values(memory_values, batch_size, seq_len, self.cfg.d_val, mano.device)
        if write_mask is not None:
            write_mask = _normalize_write_mask(write_mask, batch_size, seq_len, mano.device)

        residuals: List[torch.Tensor] = []
        contexts: List[torch.Tensor] = []
        attn_list: List[Optional[torch.Tensor]] = []
        top_idx_list: List[Optional[torch.Tensor]] = []

        for t in range(seq_len):
            key_t = keys[:, t : t + 1]
            context_t, attn_t, top_idx_t = self.memory.retrieve(key_t)
            delta_t = self.refine(observations[:, t], context_t)
            residuals.append(delta_t)
            contexts.append(context_t.squeeze(1))
            if return_attention:
                attn_list.append(None if attn_t is None else attn_t.detach().cpu())
                top_idx_list.append(None if top_idx_t is None else top_idx_t.detach().cpu())

            if memory_values is not None:
                conf_t = _confidence_from_uncertainty(tensors["uncertainty"][:, :, t])
                mask_t = write_mask[:, t] if write_mask is not None else None
                self.memory.write(
                    key_t,
                    memory_values[:, t : t + 1],
                    conf_t,
                    write_mask=mask_t,
                    time_t=torch.full((batch_size, 1, 1), float(t), device=mano.device),
                )

        packed = torch.stack(residuals, dim=1)
        memory_context = torch.stack(contexts, dim=1)
        out = _unpack_refinement(
            packed,
            mano,
            tensors["world_scale_init"],
            num_hands=self.cfg.num_hands,
            scale_eps=self.cfg.scale_eps,
        )
        out["memory_context"] = memory_context
        out["memory_size"] = self.memory.size
        if return_attention:
            out["attn"] = attn_list
            out["top_idx"] = top_idx_list
        return out


def _normalize_features(features: Dict[str, Any], num_hands: int, img_feat_dim: int) -> Dict[str, torch.Tensor]:
    mano = _ensure_nhtd(features["mano_local_init"], num_hands, MANO_LOCAL_DIM)
    device = mano.device
    kp = _ensure_nhtkd(features["kp_2d"], num_hands, 21, 3).to(device)
    img = _ensure_nhtd(features["img_feat"], num_hands, img_feat_dim).to(device)
    uncertainty = _ensure_nhtd(features["uncertainty"], num_hands, 1).to(device)

    cam_init = features.get("cam_init", {})
    cam_t = cam_init.get("cam_t") if isinstance(cam_init, dict) else None
    cam_t = _normalize_cam_t(cam_t, mano.shape[0], num_hands, mano.shape[2], device)
    world_scale = cam_init.get("world_scale") if isinstance(cam_init, dict) else None
    world_scale = _normalize_world_scale(world_scale, mano.shape[0], mano.shape[2], device)
    return {
        "mano_local_init": mano,
        "kp_2d": kp,
        "img_feat": img,
        "uncertainty": uncertainty,
        "cam_t": cam_t,
        "world_scale_init": world_scale,
    }


def _build_observation(tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    mano = tensors["mano_local_init"].permute(0, 2, 1, 3).flatten(-2)
    kp = tensors["kp_2d"].permute(0, 2, 1, 3, 4).flatten(-3)
    img = tensors["img_feat"].permute(0, 2, 1, 3).flatten(-2)
    uncertainty = tensors["uncertainty"].permute(0, 2, 1, 3).flatten(-2)
    cam_t = tensors["cam_t"].permute(0, 2, 1, 3).flatten(-2)
    scale = tensors["world_scale_init"]
    return torch.cat([mano, kp, img, uncertainty, cam_t, scale], dim=-1)


# Per-component soft bounds (in MANO/world units) on the predicted residual.
# A tanh saturation keeps in-distribution corrections near-linear while
# preventing catastrophic out-of-distribution frames from producing huge
# pose / translation / scale jumps that blow up the reprojection.
_RESIDUAL_BOUNDS = {
    "root": (0, 3, 1.5),
    "pose": (3, 48, 1.5),
    "betas": (48, 58, 3.0),
    "trans": (58, 61, 2.5),
}
_SCALE_DELTA_BOUND = 6.0


def _soft_bound(x: torch.Tensor, bound: float) -> torch.Tensor:
    return bound * torch.tanh(x / bound)


def _unpack_refinement(
    packed: torch.Tensor,
    mano_init: torch.Tensor,
    world_scale_init: torch.Tensor,
    *,
    num_hands: int,
    scale_eps: float,
    bound_residuals: bool = True,
) -> Dict[str, torch.Tensor]:
    batch_size, seq_len, dim = packed.shape
    expected = num_hands * MANO_LOCAL_DIM + 1
    if dim != expected:
        raise ValueError(f"Expected residual dim {expected}, got {dim}")

    hand = packed[..., : num_hands * MANO_LOCAL_DIM].view(batch_size, seq_len, num_hands, MANO_LOCAL_DIM)
    hand = hand.permute(0, 2, 1, 3).contiguous()
    delta_world_scale = packed[..., -1:].contiguous()
    if bound_residuals:
        bounded = torch.empty_like(hand)
        for _name, (a, b, bnd) in _RESIDUAL_BOUNDS.items():
            bounded[..., a:b] = _soft_bound(hand[..., a:b], bnd)
        hand = bounded
        delta_world_scale = _soft_bound(delta_world_scale, _SCALE_DELTA_BOUND)
    mano_refined = mano_init + hand
    # Anchor scale so a zero residual reproduces |init scale| exactly. The naive
    # softplus(init + delta) introduces a fixed offset (softplus(1.0)=1.31 for the
    # default init scale of 1.0), which itself manifests as scale drift away from
    # the initialization. We instead place the softplus base at softplus^-1(|init|)
    # so that delta=0 -> scale=|init|, while keeping the output strictly positive.
    scale_ref = world_scale_init.abs().clamp_min(scale_eps)
    scale_base = torch.log(torch.expm1(scale_ref.clamp_min(1e-4)))
    world_scale_refined = F.softplus(scale_base + delta_world_scale) + scale_eps
    packed_bounded = torch.cat(
        [
            hand.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_hands * MANO_LOCAL_DIM),
            delta_world_scale,
        ],
        dim=-1,
    )
    return {
        "packed_residual": packed_bounded,
        "delta_mano_local": hand,
        "delta_root_orient": hand[..., :3],
        "delta_pose_body": hand[..., 3:48].reshape(batch_size, num_hands, seq_len, 15, 3),
        "delta_betas": hand[..., 48:58],
        "delta_trans": hand[..., 58:61],
        "delta_world_scale": delta_world_scale,
        "mano_local_refined": mano_refined,
        "world_scale_refined": world_scale_refined,
    }


def _confidence_from_uncertainty(uncertainty_t: torch.Tensor) -> torch.Tensor:
    conf = 1.0 - uncertainty_t.mean(dim=(1, 2), keepdim=False).clamp(0.0, 1.0)
    return conf.view(-1, 1, 1)


def _ensure_nhtd(x: torch.Tensor, num_hands: int, dim: int) -> torch.Tensor:
    x = torch.as_tensor(x).float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected (H,T,D) or (N,H,T,D), got {tuple(x.shape)}")
    if x.shape[1] != num_hands:
        x = _fit_hands(x, num_hands)
    if x.shape[-1] != dim:
        raise ValueError(f"Expected last dim {dim}, got {tuple(x.shape)}")
    return x


def _ensure_nhtkd(x: torch.Tensor, num_hands: int, num_keypoints: int, dim: int) -> torch.Tensor:
    x = torch.as_tensor(x).float()
    if x.ndim == 4:
        x = x.unsqueeze(0)
    if x.ndim != 5:
        raise ValueError(f"Expected (H,T,K,D) or (N,H,T,K,D), got {tuple(x.shape)}")
    if x.shape[1] != num_hands:
        x = _fit_hands(x, num_hands)
    if x.shape[-2:] != (num_keypoints, dim):
        raise ValueError(f"Expected keypoints (...,{num_keypoints},{dim}), got {tuple(x.shape)}")
    return x


def _fit_hands(x: torch.Tensor, num_hands: int) -> torch.Tensor:
    if x.shape[1] == num_hands:
        return x
    if x.shape[1] > num_hands:
        return x[:, :num_hands]
    pad = x.new_zeros(x.shape[0], num_hands - x.shape[1], *x.shape[2:])
    return torch.cat([x, pad], dim=1)


def _normalize_cam_t(
    cam_t: Optional[torch.Tensor],
    batch_size: int,
    num_hands: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if cam_t is None:
        return torch.zeros(batch_size, num_hands, seq_len, 3, device=device)
    cam_t = torch.as_tensor(cam_t, device=device).float()
    if cam_t.ndim == 3:
        cam_t = cam_t.unsqueeze(0)
    if cam_t.ndim != 4:
        raise ValueError(f"Expected cam_t as (H,T,3) or (N,H,T,3), got {tuple(cam_t.shape)}")
    if cam_t.shape[1] != num_hands:
        cam_t = _fit_hands(cam_t, num_hands)
    return cam_t


def _normalize_world_scale(
    world_scale: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if world_scale is None:
        return torch.ones(batch_size, seq_len, 1, device=device)
    scale = torch.as_tensor(world_scale, device=device).float()
    if scale.ndim == 0:
        scale = scale.view(1, 1, 1)
    elif scale.ndim == 1:
        scale = scale.view(1, 1, -1)
        if scale.shape[-1] != 1:
            scale = scale[..., :1]
    elif scale.ndim == 2:
        scale = scale.unsqueeze(-1)
    if scale.shape[0] == 1 and batch_size > 1:
        scale = scale.expand(batch_size, -1, -1)
    if scale.shape[1] == 1 and seq_len > 1:
        scale = scale.expand(batch_size, seq_len, 1)
    return scale


def _normalize_memory_values(
    values: torch.Tensor,
    batch_size: int,
    seq_len: int,
    d_val: int,
    device: torch.device,
) -> torch.Tensor:
    values = torch.as_tensor(values, device=device).float()
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if values.shape[0] == 1 and batch_size > 1:
        values = values.expand(batch_size, -1, -1)
    if values.shape != (batch_size, seq_len, d_val):
        raise ValueError(f"Expected memory values {(batch_size, seq_len, d_val)}, got {tuple(values.shape)}")
    return values


def _normalize_write_mask(
    mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.as_tensor(mask, device=device).bool()
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1)
    if mask.shape != (batch_size, seq_len):
        raise ValueError(f"Expected write mask {(batch_size, seq_len)}, got {tuple(mask.shape)}")
    return mask
