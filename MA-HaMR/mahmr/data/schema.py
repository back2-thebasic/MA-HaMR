"""Tensor schema for processed sequences (aligned with Instruction.md §3.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SequenceMeta:
    seq_name: str
    seq_len: int
    fps: float
    num_hands: int
    expert_stride: int
    dynhamr_log_dir: str
    expert_stage: str  # e.g. smooth_fit
    expert_iter: int
    camera_source: str  # vipe | droid
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq_name": self.seq_name,
            "seq_len": self.seq_len,
            "fps": self.fps,
            "num_hands": self.num_hands,
            "expert_stride": self.expert_stride,
            "dynhamr_log_dir": self.dynhamr_log_dir,
            "expert_stage": self.expert_stage,
            "expert_iter": self.expert_iter,
            "camera_source": self.camera_source,
            **self.extra,
        }


# Keys stored in expert_label.pth (sparse, every `expert_stride` frames)
EXPERT_KEYS = (
    "frame_indices",  # (N,) int64
    "trans",          # (N, B, 3) world translation
    "root_orient",    # (N, B, 3) world root axis-angle
    "pose_body",      # (N, B, 45) local hand pose (15 joints x 3)
    "betas",          # (B, 10) shared shape per track
    "world_scale",    # (1,) or scalar
    "cam_R",          # (N, T_cam, 3, 3) or (N, 3, 3) w2c
    "cam_t",          # (N, T_cam, 3) or (N, 3)
    "intrins",        # (4,) fx fy cx cy
)

# Keys stored in init_state.pth (dense, all frames)
INIT_KEYS = (
    "trans",
    "root_orient",
    "pose_body",
    "betas",
    "cam_R",
    "cam_t",
    "intrins",
    "is_right",       # (B, T) optional
    "vis_mask",         # (B, T) optional
)

# Action-conditioned correction residual (core memory Value)
RESIDUAL_KEYS = (
    "delta_trans",      # expert - init
    "delta_root_orient",
    "delta_pose_body",
    "delta_world_scale",  # scalar diff if both available
)
