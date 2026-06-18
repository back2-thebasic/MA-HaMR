#!/usr/bin/env python3
"""Export H2O egocentric RGB to mp4 for Dyn-HaMR."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path


def _resolve_ffmpeg() -> str:
    for candidate in (
        shutil.which("ffmpeg"),
        "/home/phanteks/miniconda3/envs/dynhamr/bin/ffmpeg",
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("ffmpeg not found. Install ffmpeg or activate dynhamr env.")


FRAME_RE = re.compile(r"(\d+)\.(jpg|png|jpeg)$", re.IGNORECASE)


def _h2o_tar_name(subject: int) -> str:
    return f"subject{subject}_ego_v1_1.tar.gz"


def _h2o_rgb_member_prefix(subject: int, action: str, take: int, rgb_kind: str) -> str:
    return f"subject{subject}_ego/{action}/{take}/cam4/{rgb_kind}/"


def _extract_rgb_from_tar(
    tar_path: Path,
    member_prefix: str,
    dest_rgb_dir: Path,
) -> int:
    dest_rgb_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.startswith(member_prefix) and m.isfile()]
        if not members:
            raise FileNotFoundError(
                f"No rgb files under '{member_prefix}' in {tar_path}. "
                "Check --subject/--action/--take/--rgb-kind."
            )
        for member in members:
            src = tarfile.TarInfo(name=member.name)
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            out_name = Path(member.name).name
            with open(dest_rgb_dir / out_name, "wb") as f:
                shutil.copyfileobj(extracted, f)
            count += 1
            if count % 500 == 0:
                print(f"  extracted {count} frames...")
    return count


def _sorted_frame_paths(rgb_dir: Path) -> list[Path]:
    frames = [p for p in rgb_dir.iterdir() if p.is_file() and FRAME_RE.match(p.name)]
    frames.sort(key=lambda p: int(FRAME_RE.match(p.name).group(1)))
    if not frames:
        raise FileNotFoundError(f"No frames found in {rgb_dir}")
    return frames


def _write_concat_file(frames: list[Path], concat_path: Path, fps: float) -> None:
    # Use concat demuxer; set duration per frame for stable fps.
    dt = 1.0 / fps
    with open(concat_path, "w", encoding="utf-8") as f:
        for frame in frames:
            f.write(f"file '{frame.as_posix()}'\n")
            f.write(f"duration {dt:.9f}\n")
        f.write(f"file '{frames[-1].as_posix()}'\n")


def export_h2o_sequence(
    *,
    output_mp4: Path,
    subject: int,
    action: str,
    take: int = 0,
    rgb_kind: str = "rgb",
    fps: float = 30.0,
    h2o_root: Path,
    rgb_dir: Path | None = None,
    keep_extracted: bool = False,
) -> dict:
    h2o_root = h2o_root.resolve()
    if rgb_dir is None:
        rgb_dir = h2o_root / "extracted" / f"subject{subject}_ego" / action / str(take) / "cam4" / rgb_kind
    else:
        rgb_dir = rgb_dir.resolve()

    if not any(rgb_dir.glob("*")):
        tar_path = h2o_root / _h2o_tar_name(subject)
        if not tar_path.is_file():
            raise FileNotFoundError(f"Missing tar and rgb dir. Expected tar: {tar_path}")
        member_prefix = _h2o_rgb_member_prefix(subject, action, take, rgb_kind)
        print(f"[H2O] extracting {member_prefix} from {tar_path.name} ...")
        n_extracted = _extract_rgb_from_tar(tar_path, member_prefix, rgb_dir)
        print(f"[H2O] extracted {n_extracted} frames -> {rgb_dir}")
    else:
        print(f"[H2O] using existing rgb dir: {rgb_dir}")

    frames = _sorted_frame_paths(rgb_dir)
    print(f"[H2O] subject{subject}/{action}/{take} frames={len(frames)} fps={fps}")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    concat_path = output_mp4.with_suffix(".concat.txt")
    _write_concat_file(frames, concat_path, fps)
    ffmpeg = _resolve_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True)
    concat_path.unlink(missing_ok=True)

    if not keep_extracted and rgb_dir.is_relative_to(h2o_root / "extracted"):
        shutil.rmtree(rgb_dir.parent.parent.parent.parent, ignore_errors=True)

    return {
        "subject": subject,
        "action": action,
        "take": take,
        "rgb_kind": rgb_kind,
        "num_frames": len(frames),
        "fps": fps,
        "output_mp4": str(output_mp4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert H2O ego RGB to mp4")
    parser.add_argument("--subject", type=int, required=True, help="Subject id, e.g. 1")
    parser.add_argument("--action", required=True, help="Action folder, e.g. h1")
    parser.add_argument("--take", type=int, default=0, help="Take index under action (default: 0)")
    parser.add_argument(
        "--rgb-kind",
        default="rgb",
        choices=("rgb", "rgb256"),
        help="Use full-res rgb (1280x720) or rgb256 (455x256)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Output fps (default: 30)")
    parser.add_argument(
        "--h2o-root",
        default="/extra/SuC/data/raw/h2o",
        help="Root containing subject*_ego_v1_1.tar.gz",
    )
    parser.add_argument(
        "--rgb-dir",
        default=None,
        help="Optional pre-extracted rgb directory (skip tar extract)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output mp4 (default: /extra/SuC/dynhamr_io/videos/h2o_s{subject}_{action}.mp4)",
    )
    parser.add_argument("--keep-extracted", action="store_true", help="Keep extracted frames")
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = f"/extra/SuC/dynhamr_io/videos/h2o_s{args.subject}_{args.action}.mp4"

    summary = export_h2o_sequence(
        output_mp4=Path(output),
        subject=args.subject,
        action=args.action,
        take=args.take,
        rgb_kind=args.rgb_kind,
        fps=args.fps,
        h2o_root=Path(args.h2o_root),
        rgb_dir=Path(args.rgb_dir) if args.rgb_dir else None,
        keep_extracted=args.keep_extracted,
    )
    print("Done:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
