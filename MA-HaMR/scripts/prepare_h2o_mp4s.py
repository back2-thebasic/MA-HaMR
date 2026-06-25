#!/usr/bin/env python3
"""Batch-convert H2O ego RGB sequences to mp4 for Dyn-HaMR.

This wraps ``convert_h2o_to_mp4.py`` and prepares official H2O ego-view
sequence groups from subject*_ego_v1_1.tar.gz.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ACTIONS = ("h1", "h2", "k1", "k2", "o1", "o2")
TRAIN = (
    [(1, a) for a in ACTIONS]
    + [(2, a) for a in ACTIONS]
    + [(3, a) for a in ("h1", "h2", "k1")]
)
VAL = [(3, a) for a in ("k2", "o1", "o2")]
TEST = [(4, a) for a in ACTIONS]
DEV = [(1, "h1"), (1, "h2"), (2, "h1")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare H2O ego mp4s in batch")
    parser.add_argument(
        "--split",
        choices=("dev", "train", "val", "test", "official", "all"),
        default="official",
        help=(
            "dev: current 3 dev seqs; train/val/test: H2O official subsets; "
            "official/all: train+val+test"
        ),
    )
    parser.add_argument("--h2o-root", default="/extra/SuC/data/raw/h2o")
    parser.add_argument("--output-root", default="/extra/SuC/dynhamr_io/videos")
    parser.add_argument("--rgb-kind", choices=("rgb", "rgb256"), default="rgb")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--take", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing mp4s")
    parser.add_argument("--keep-extracted", action="store_true", help="Keep extracted RGB frames")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    jobs = _jobs(args.split)
    script = Path(__file__).with_name("convert_h2o_to_mp4.py")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"split={args.split} num_jobs={len(jobs)} output_root={output_root}")
    failures: list[str] = []
    for idx, (subject, action) in enumerate(jobs, start=1):
        seq_name = f"h2o_s{subject}_{action}"
        output = output_root / f"{seq_name}.mp4"
        if output.is_file() and not args.overwrite:
            print(f"[{idx:02d}/{len(jobs):02d}] skip existing {output}")
            continue

        cmd = [
            sys.executable,
            str(script),
            "--subject",
            str(subject),
            "--action",
            action,
            "--take",
            str(args.take),
            "--rgb-kind",
            args.rgb_kind,
            "--fps",
            str(args.fps),
            "--h2o-root",
            args.h2o_root,
            "--output",
            str(output),
        ]
        if args.keep_extracted:
            cmd.append("--keep-extracted")

        print(f"[{idx:02d}/{len(jobs):02d}] prepare {seq_name}: {' '.join(cmd)}")
        if args.dry_run:
            continue
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            failures.append(seq_name)
            print(f"[ERROR] {seq_name} failed with exit_code={exc.returncode}")

    if failures:
        print("Failed sequences:")
        for name in failures:
            print(f"  {name}")
        raise SystemExit(1)
    print("All requested H2O mp4s are ready.")


def _jobs(split: str) -> list[tuple[int, str]]:
    if split == "dev":
        return list(DEV)
    if split == "train":
        return list(TRAIN)
    if split == "val":
        return list(VAL)
    if split == "test":
        return list(TEST)
    if split in {"official", "all"}:
        return list(TRAIN + VAL + TEST)
    raise ValueError(split)


if __name__ == "__main__":
    main()
