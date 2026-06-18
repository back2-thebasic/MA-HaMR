#!/usr/bin/env python3
"""Export HOT3D Aria sequence (VRS) to mp4 for Dyn-HaMR."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _resolve_ffmpeg() -> str:
    for candidate in (
        shutil.which("ffmpeg"),
        "/home/phanteks/miniconda3/envs/dynhamr/bin/ffmpeg",
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("ffmpeg not found. Install ffmpeg or activate dynhamr env.")


def _add_hot3d_to_path(hot3d_repo: Path) -> None:
    repo = hot3d_repo / "hot3d"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def _estimate_fps(timestamps_ns: list[int]) -> float:
    if len(timestamps_ns) < 2:
        return 30.0
    deltas = np.diff(np.asarray(timestamps_ns, dtype=np.float64)) / 1e9
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return 30.0
    return float(1.0 / np.median(deltas))


def export_hot3d_sequence(
    sequence_dir: Path,
    output_mp4: Path,
    *,
    hot3d_repo: Path,
    stream_id: str = "214-1",
    undistort: bool = True,
    fps: float | None = None,
    max_frames: int | None = None,
    keep_frames: bool = False,
    frames_dir: Path | None = None,
) -> dict:
    sequence_dir = sequence_dir.resolve()
    vrs_path = sequence_dir / "recording.vrs"
    if not vrs_path.is_file():
        raise FileNotFoundError(f"Missing recording.vrs: {vrs_path}")

    _add_hot3d_to_path(hot3d_repo)
    from data_loaders.AriaDataProvider import AriaDataProvider
    from projectaria_tools.core.sensor_data import TimeDomain
    from projectaria_tools.core.stream_id import StreamId

    mps_dir = sequence_dir / "mps"
    provider = AriaDataProvider(str(vrs_path), str(mps_dir) if mps_dir.is_dir() else None)
    sid = StreamId(stream_id)
    labels = {str(s): provider.get_image_stream_label(s) for s in provider.get_image_stream_ids()}
    if stream_id not in labels:
        raise ValueError(f"Stream {stream_id} not found. Available: {labels}")

    timestamps = provider.get_sequence_timestamps(sid, TimeDomain.TIME_CODE)
    if max_frames is not None:
        timestamps = timestamps[:max_frames]
    if not timestamps:
        raise RuntimeError(f"No frames for stream {stream_id}")

    out_fps = fps if fps is not None else _estimate_fps(timestamps)
    seq_name = sequence_dir.name
    if frames_dir is None:
        frames_root = sequence_dir.parent.parent / "_frames_cache" / seq_name / stream_id
    else:
        frames_root = frames_dir
    frames_root.mkdir(parents=True, exist_ok=True)

    print(f"[HOT3D] sequence={seq_name} stream={stream_id} ({labels[stream_id]})")
    print(f"[HOT3D] frames={len(timestamps)} fps={out_fps:.3f} undistort={undistort}")

    for i, ts in enumerate(timestamps):
        if undistort:
            img = provider.get_undistorted_image(ts, sid)
        else:
            img = provider.get_image(ts, sid)
        if img is None:
            raise RuntimeError(f"Failed to decode frame {i} at ts={ts}")
        Image.fromarray(img).save(frames_root / f"{i:06d}.jpg", quality=95)
        if (i + 1) % 200 == 0 or i + 1 == len(timestamps):
            print(f"  exported {i + 1}/{len(timestamps)}")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _resolve_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{out_fps:.6f}",
        "-i",
        str(frames_root / "%06d.jpg"),
        "-frames:v",
        str(len(timestamps)),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True)

    if not keep_frames:
        shutil.rmtree(frames_root, ignore_errors=True)

    return {
        "sequence": seq_name,
        "stream_id": stream_id,
        "num_frames": len(timestamps),
        "fps": out_fps,
        "output_mp4": str(output_mp4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HOT3D Aria VRS to mp4")
    parser.add_argument(
        "--sequence-dir",
        required=True,
        help="Path to HOT3D sequence folder, e.g. .../dataset/P0003_c701bd11",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output mp4 path, e.g. /extra/SuC/dynhamr_io/videos/hot3d_P0003.mp4",
    )
    parser.add_argument(
        "--hot3d-repo",
        default="/extra/SuC/data/raw/hot3d",
        help="Path to cloned facebookresearch/hot3d repo",
    )
    parser.add_argument("--stream-id", default="214-1", help="Aria RGB stream (default: 214-1)")
    parser.add_argument("--fps", type=float, default=None, help="Override fps (default: auto)")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug: export first N frames")
    parser.add_argument("--raw-fisheye", action="store_true", help="Keep fisheye (default: undistort)")
    parser.add_argument("--keep-frames", action="store_true", help="Keep intermediate jpg frames")
    parser.add_argument("--frames-dir", default=None, help="Custom intermediate frames directory")
    args = parser.parse_args()

    summary = export_hot3d_sequence(
        Path(args.sequence_dir),
        Path(args.output),
        hot3d_repo=Path(args.hot3d_repo),
        stream_id=args.stream_id,
        undistort=not args.raw_fisheye,
        fps=args.fps,
        max_frames=args.max_frames,
        keep_frames=args.keep_frames,
        frames_dir=Path(args.frames_dir) if args.frames_dir else None,
    )
    print("Done:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
