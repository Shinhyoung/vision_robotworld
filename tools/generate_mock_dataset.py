#!/usr/bin/env python3
"""Generate the mock RGB-D dataset used for training, testing and CI.

Team Lead deliverable (claude.md section 3.1). Frames are rendered from the real
part CAD at randomised-but-bounded poses, with ground-truth pose and OK/NG label
stored alongside, so every agent can develop with no D455 and no GPU.

    python3 tools/generate_mock_dataset.py --part guide_block --train 40 --test 20

Layout::

    data/mock/<part_id>/train/frame_000.npz   defect-free, for fitting
    data/mock/<part_id>/test/frame_000.npz    mixed OK/NG, for evaluation
    data/mock/<part_id>/metadata.json

Each ``.npz`` holds ``color`` (uint8 HxWx3), ``depth`` (float32 HxW, meters),
``gt_position``, ``gt_orientation``, ``is_good`` and ``defect``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from _bootstrap import bootstrap

bootstrap()

from roboworld_core import paths  # noqa: E402
from roboworld_core.config import load_config  # noqa: E402
from roboworld_core.mock_data import (  # noqa: E402
    DEFECT_KINDS,
    MockStation,
    parts_from_config,
)
from roboworld_core.types import CameraIntrinsics, Frame  # noqa: E402


def save_frame(path: Path, frame: Frame, defect: str | None) -> None:
    np.savez_compressed(
        path,
        color=frame.color,
        depth=frame.depth,
        gt_position=frame.gt_pose.position,
        gt_orientation=frame.gt_pose.orientation,
        is_good=bool(frame.gt_is_good),
        defect=defect or "",
        stamp=frame.stamp,
        sequence=frame.sequence,
    )


def load_frame(path: Path, intrinsics: CameraIntrinsics, part_id: str) -> Frame:
    """Reload a saved mock frame (used by training and evaluation tools)."""
    from roboworld_core.geometry import Pose

    with np.load(path, allow_pickle=False) as data:
        return Frame(
            color=data["color"],
            depth=data["depth"],
            intrinsics=intrinsics,
            stamp=float(data["stamp"]),
            sequence=int(data["sequence"]),
            part_id=part_id,
            gt_pose=Pose(data["gt_position"], data["gt_orientation"], intrinsics.frame_id),
            gt_is_good=bool(data["is_good"]),
        )


def generate(args: argparse.Namespace) -> int:
    cfg = load_config()
    intrinsics = CameraIntrinsics.from_config(cfg.section("camera"))
    specs = parts_from_config(cfg)
    station = MockStation(specs, intrinsics)

    known = {spec.part_id for spec in specs}
    parts = list(known) if args.part == "all" else [args.part]
    for part_id in parts:
        if part_id not in known:
            print(f"error: unknown part '{part_id}'; known: {sorted(known)}", file=sys.stderr)
            return 2

    output_root = Path(args.output) if args.output else paths.data_dir() / "mock"

    for part_id in parts:
        part_dir = output_root / part_id
        (part_dir / "train").mkdir(parents=True, exist_ok=True)
        (part_dir / "test").mkdir(parents=True, exist_ok=True)

        # --- train: defect-free only (the detectors are unsupervised) ----
        for index in range(args.train):
            frame = station.sample_frame(
                part_id, defect=None, seed=args.seed + index, sequence=index
            )
            save_frame(part_dir / "train" / f"frame_{index:04d}.npz", frame, None)

        # --- test: alternating OK / NG so both branches are covered ------
        defect_counts: dict[str, int] = {}
        for index in range(args.test):
            defect = None
            if index % 2 == 1:
                defect = DEFECT_KINDS[(index // 2) % len(DEFECT_KINDS)]
                defect_counts[defect] = defect_counts.get(defect, 0) + 1
            frame = station.sample_frame(
                part_id,
                defect=defect,
                seed=args.seed + 100_000 + index,
                sequence=index,
            )
            save_frame(part_dir / "test" / f"frame_{index:04d}.npz", frame, defect)

        metadata = {
            "part_id": part_id,
            "train_frames": args.train,
            "test_frames": args.test,
            "test_defects": defect_counts,
            "seed": args.seed,
            "intrinsics": {
                "width": intrinsics.width, "height": intrinsics.height,
                "fx": intrinsics.fx, "fy": intrinsics.fy,
                "cx": intrinsics.cx, "cy": intrinsics.cy,
                "frame_id": intrinsics.frame_id,
            },
            "units": {"depth": "meters", "position": "meters", "orientation": "quaternion xyzw"},
            "generator": "tools/generate_mock_dataset.py",
        }
        (part_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(
            f"{part_id}: {args.train} train + {args.test} test frames -> {part_dir}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", default="all", help="part id, or 'all'")
    parser.add_argument("--train", type=int, default=40, help="defect-free training frames")
    parser.add_argument("--test", type=int, default=20, help="mixed OK/NG test frames")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", default=None, help="output root (default: data/mock)")
    return generate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
