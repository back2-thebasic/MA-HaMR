"""Online self-calibration memory for MA-HaMR.

The memory follows the constraints in ``Instruction.md``:

* keys are built from coordinate-invariant quantities only;
* values store correction residuals, not absolute world coordinates;
* writes are detached, so long videos do not backpropagate through history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


MANO_LOCAL_DIM = 61
HAND_POSE_DIM = 48  # root_orient(3) + pose_body(15 * 3)


@dataclass
class MemoryConfig:
    d_mem: int = 128
    d_val: int = 123  # two hands * mano_local(61) + delta_world_scale(1)
    hidden_dim: int = 256
    topk: int = 16
    temperature: Optional[float] = None
    exclude_recent: int = 30
    max_size: int = 4096
    write_conf_threshold: float = 0.5


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3) -> None:
        super().__init__()
        layers = []
        cur = in_dim
        for _ in range(max(depth - 1, 1)):
            layers.extend([nn.Linear(cur, hidden_dim), nn.GELU()])
            cur = hidden_dim
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MemoryKeyEncoder(nn.Module):
    """Encode coordinate-invariant retrieval keys.

    Input ``mano_local`` has layout root_orient(3), pose_body(45), betas(10),
    trans(3). The key uses:

    * local MANO pose for both hands;
    * left-right relative translation;
    * short-term pose velocity.

    A common world translation added to both hands leaves the key unchanged.
    """

    def __init__(
        self,
        *,
        num_hands: int = 2,
        d_mem: int = 128,
        hidden_dim: int = 256,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.num_hands = num_hands
        self.normalize = normalize
        in_dim = num_hands * HAND_POSE_DIM * 2 + 3
        self.encoder = MLP(in_dim, hidden_dim, d_mem, depth=3)

    def forward(self, mano_local: torch.Tensor) -> torch.Tensor:
        mano_local, squeezed = _ensure_bhtd(mano_local)
        if mano_local.shape[1] != self.num_hands:
            mano_local = _fit_num_hands(mano_local, self.num_hands)

        pose = mano_local[..., :HAND_POSE_DIM]
        trans = mano_local[..., -3:]
        pose_vel = torch.zeros_like(pose)
        pose_vel[:, :, 1:] = pose[:, :, 1:] - pose[:, :, :-1]

        if self.num_hands >= 2:
            d_rel = trans[:, 0] - trans[:, 1]
        else:
            d_rel = torch.zeros(mano_local.shape[0], mano_local.shape[2], 3, device=mano_local.device)

        feats = torch.cat(
            [
                pose.permute(0, 2, 1, 3).flatten(-2),
                pose_vel.permute(0, 2, 1, 3).flatten(-2),
                d_rel,
            ],
            dim=-1,
        )
        key = self.encoder(feats)
        if self.normalize:
            key = F.normalize(key, dim=-1)
        return key.squeeze(0) if squeezed else key


class LongTermMemoryBank(nn.Module):
    """Causal per-sequence KV memory with top-k retrieval and detached writes."""

    def __init__(
        self,
        d_mem: int = 128,
        d_val: int = 123,
        topk: int = 16,
        temp: Optional[float] = None,
        exclude_recent: int = 30,
        max_size: int = 4096,
        write_conf_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.d_mem = d_mem
        self.d_val = d_val
        self.topk = topk
        self.temp = temp if temp is not None else d_mem**0.5
        self.exclude_recent = exclude_recent
        self.max_size = max_size
        self.write_conf_threshold = write_conf_threshold
        self.reset()

    def reset(self) -> None:
        self.keys: Optional[torch.Tensor] = None
        self.vals: Optional[torch.Tensor] = None
        self.conf: Optional[torch.Tensor] = None
        self.times: Optional[torch.Tensor] = None

    @property
    def size(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[1])

    def retrieve(self, query_t: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        query_t = _ensure_b1d(query_t, self.d_mem)
        batch_size = query_t.shape[0]
        if self.keys is None or self.size <= self.exclude_recent:
            zeros = query_t.new_zeros(batch_size, 1, self.d_val)
            return zeros, None, None

        usable = self.size - self.exclude_recent
        keys = self.keys[:, :usable].to(query_t.device)
        vals = self.vals[:, :usable].to(query_t.device)
        conf = self.conf[:, :usable].to(query_t.device)
        scores = torch.bmm(query_t, keys.transpose(1, 2)) / self.temp
        scores = scores + torch.log(conf.clamp_min(1e-6)).transpose(1, 2)

        k = min(self.topk, scores.shape[-1])
        top_scores, top_idx = scores.topk(k, dim=-1)
        attn = torch.softmax(top_scores, dim=-1)
        gather_idx = top_idx.squeeze(1).unsqueeze(-1).expand(-1, -1, self.d_val)
        selected = torch.gather(vals, dim=1, index=gather_idx)
        context = torch.bmm(attn, selected)
        return context, attn, top_idx

    @torch.no_grad()
    def write(
        self,
        key_t: torch.Tensor,
        val_t: torch.Tensor,
        conf_t: torch.Tensor,
        *,
        time_t: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
    ) -> None:
        key_t = _ensure_b1d(key_t, self.d_mem).detach()
        val_t = _ensure_b1d(val_t, self.d_val).detach()
        conf_t = _ensure_conf(conf_t, key_t.shape[0], key_t.device).detach()
        if write_mask is not None:
            mask = write_mask.detach().to(device=key_t.device, dtype=torch.bool).view(-1, 1, 1)
            conf_t = torch.where(mask, conf_t, torch.zeros_like(conf_t))
        conf_t = torch.where(
            conf_t >= self.write_conf_threshold,
            conf_t,
            torch.zeros_like(conf_t),
        )
        if torch.all(conf_t <= 0):
            return

        if time_t is None:
            time_t = torch.full((key_t.shape[0], 1, 1), float(self.size), device=key_t.device)
        else:
            time_t = _ensure_conf(time_t, key_t.shape[0], key_t.device)

        if self.keys is None:
            self.keys = key_t
            self.vals = val_t
            self.conf = conf_t
            self.times = time_t.detach()
        else:
            self.keys = torch.cat([self.keys.to(key_t.device), key_t], dim=1)
            self.vals = torch.cat([self.vals.to(val_t.device), val_t], dim=1)
            self.conf = torch.cat([self.conf.to(conf_t.device), conf_t], dim=1)
            self.times = torch.cat([self.times.to(time_t.device), time_t.detach()], dim=1)

        if self.size > self.max_size:
            start = self.size - self.max_size
            self.keys = self.keys[:, start:]
            self.vals = self.vals[:, start:]
            self.conf = self.conf[:, start:]
            self.times = self.times[:, start:]

    def forward(
        self,
        query_t: torch.Tensor,
        key_t: Optional[torch.Tensor] = None,
        val_t: Optional[torch.Tensor] = None,
        conf_t: Optional[torch.Tensor] = None,
        *,
        write_mask: Optional[torch.Tensor] = None,
        time_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        context, attn, top_idx = self.retrieve(query_t)
        if key_t is not None and val_t is not None and conf_t is not None:
            self.write(key_t, val_t, conf_t, write_mask=write_mask, time_t=time_t)
        return context, attn, top_idx


def pack_residual_value(
    *,
    delta_root_orient: torch.Tensor,
    delta_pose_body: torch.Tensor,
    delta_trans: torch.Tensor,
    delta_world_scale: Optional[torch.Tensor] = None,
    delta_betas: Optional[torch.Tensor] = None,
    num_hands: int = 2,
) -> torch.Tensor:
    """Pack teacher residuals into the memory/refinement vector layout.

    Layout per time step:
    ``[hand0 mano_local_res(61), hand1 mano_local_res(61), delta_world_scale]``.
    The 61 hand dimensions follow Step 2's ``mano_local_init`` layout:
    root_orient(3), pose_body(45), betas(10), trans(3).
    """

    root = _as_bhtd(delta_root_orient, num_hands, 3)
    pose = torch.as_tensor(delta_pose_body).float()
    if pose.ndim == 5:
        pose = pose.reshape(pose.shape[0], pose.shape[1], pose.shape[2], -1)
    elif pose.ndim == 4 and pose.shape[-2:] == (15, 3):
        pose = pose.reshape(pose.shape[0], pose.shape[1], -1).unsqueeze(0)
    pose = _as_bhtd(pose, num_hands, 45)
    trans = _as_bhtd(delta_trans, num_hands, 3)

    if delta_betas is None:
        betas = torch.zeros(*root.shape[:3], 10, device=root.device, dtype=root.dtype)
    else:
        betas = _as_bhtd(delta_betas, num_hands, 10).to(root.device)

    hand_value = torch.cat([root, pose.to(root.device), betas, trans.to(root.device)], dim=-1)
    value = hand_value.permute(0, 2, 1, 3).flatten(-2)

    if delta_world_scale is None:
        scale = torch.zeros(value.shape[0], value.shape[1], 1, device=value.device, dtype=value.dtype)
    else:
        scale = torch.as_tensor(delta_world_scale, device=value.device, dtype=value.dtype)
        if scale.ndim == 0:
            scale = scale.view(1, 1, 1).expand(value.shape[0], value.shape[1], 1)
        elif scale.ndim == 1:
            if scale.numel() == 1:
                scale = scale.view(1, 1, 1).expand(value.shape[0], value.shape[1], 1)
            else:
                scale = scale.view(1, -1, 1)
        elif scale.ndim == 2:
            scale = scale.unsqueeze(-1)
        if scale.shape[1] == 1 and value.shape[1] > 1:
            scale = scale.expand(value.shape[0], value.shape[1], 1)
    return torch.cat([value, scale.to(value.device)], dim=-1).contiguous()


def _ensure_bhtd(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    x = torch.as_tensor(x).float()
    if x.ndim == 3:
        return x.unsqueeze(0), True
    if x.ndim == 4:
        return x, False
    raise ValueError(f"Expected (H,T,D) or (N,H,T,D), got {tuple(x.shape)}")


def _as_bhtd(x: torch.Tensor, num_hands: int, dim: int) -> torch.Tensor:
    x = torch.as_tensor(x).float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(x.shape)}")
    if x.shape[1] != num_hands:
        x = _fit_num_hands(x, num_hands)
    if x.shape[-1] != dim:
        x = x.reshape(*x.shape[:3], -1)
    if x.shape[-1] != dim:
        raise ValueError(f"Expected last dim {dim}, got {tuple(x.shape)}")
    return x


def _fit_num_hands(x: torch.Tensor, num_hands: int) -> torch.Tensor:
    if x.shape[1] == num_hands:
        return x
    if x.shape[1] > num_hands:
        return x[:, :num_hands]
    pad = x.new_zeros(x.shape[0], num_hands - x.shape[1], *x.shape[2:])
    return torch.cat([x, pad], dim=1)


def _ensure_b1d(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = torch.as_tensor(x).float()
    if x.ndim == 2:
        x = x.unsqueeze(1)
    if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] != dim:
        raise ValueError(f"Expected (N,1,{dim}), got {tuple(x.shape)}")
    return x


def _ensure_conf(x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(x, device=device).float()
    if x.ndim == 0:
        x = x.view(1, 1, 1).expand(batch_size, 1, 1)
    elif x.ndim == 1:
        x = x.view(-1, 1, 1)
    elif x.ndim == 2:
        x = x.unsqueeze(-1)
    if x.shape[0] == 1 and batch_size > 1:
        x = x.expand(batch_size, -1, -1)
    if x.ndim != 3 or x.shape[1:] != (1, 1):
        raise ValueError(f"Expected confidence as (N,1,1), got {tuple(x.shape)}")
    return x
