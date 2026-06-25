"""Minimal MANO joint layer used by MA-HaMR losses."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


MANO_OPENPOSE_JOINT_MAP = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


class MANOJointLayer(nn.Module):
    """Generate 21 MANO joints from Step-2/3 MANO local vectors.

    The project stores one 61D vector per hand/frame:
    root_orient(3), pose_body(45), betas(10), trans(3).

    We use the right-hand MANO model and mirror x coordinates for left hands,
    matching Dyn-HaMR's ``run_mano`` convention.
    """

    def __init__(
        self,
        model_path: str = "/data/SuC/Dyn-HaMR/_DATA/data/mano",
        *,
        flat_hand_mean: bool = False,
    ) -> None:
        super().__init__()
        try:
            import smplx
            from smplx.utils import to_tensor
            from smplx.vertex_ids import vertex_ids
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise ImportError("MANOJointLayer requires the smplx package in the active environment.") from exc

        self.mano = smplx.MANO(
            model_path=model_path,
            use_pca=False,
            flat_hand_mean=flat_hand_mean,
            is_rhand=True,
        )
        self.register_buffer("extra_joints_idxs", to_tensor(list(vertex_ids["mano"].values()), dtype=torch.long))
        self.register_buffer("joint_map", torch.tensor(MANO_OPENPOSE_JOINT_MAP, dtype=torch.long))

    def forward(self, mano_local: torch.Tensor, is_right: Optional[torch.Tensor] = None) -> torch.Tensor:
        mano_local = torch.as_tensor(mano_local).float()
        if mano_local.ndim == 3:
            mano_local = mano_local.unsqueeze(0)
        if mano_local.ndim != 4 or mano_local.shape[-1] != 61:
            raise ValueError(f"Expected mano_local as (N,H,T,61), got {tuple(mano_local.shape)}")

        n, h, t, _ = mano_local.shape
        flat = mano_local.reshape(n * h * t, 61)
        out = self.mano(
            global_orient=flat[:, :3],
            hand_pose=flat[:, 3:48],
            betas=flat[:, 48:58],
            transl=flat[:, 58:61],
            pose2rot=True,
        )
        extra = torch.index_select(out.vertices, dim=1, index=self.extra_joints_idxs.to(out.vertices.device))
        joints = torch.cat([out.joints, extra], dim=1)
        joints = joints[:, self.joint_map.to(joints.device)]
        joints = joints.reshape(n, h, t, 21, 3)

        if is_right is not None:
            is_right_t = torch.as_tensor(is_right, device=joints.device).float()
            if is_right_t.ndim == 2:
                is_right_t = is_right_t.unsqueeze(0)
            if is_right_t.ndim != 3:
                raise ValueError(f"Expected is_right as (N,H,T), got {tuple(is_right_t.shape)}")
            if is_right_t.shape[:3] != (n, h, t):
                if is_right_t.shape[0] == 1 and n > 1:
                    is_right_t = is_right_t.expand(n, -1, -1)
                if is_right_t.shape[2] == 1 and t > 1:
                    is_right_t = is_right_t.expand(-1, -1, t)
            mirror = (2.0 * is_right_t - 1.0).view(n, h, t, 1)
            joints = joints.clone()
            joints[..., 0] = mirror * joints[..., 0]

        return joints
