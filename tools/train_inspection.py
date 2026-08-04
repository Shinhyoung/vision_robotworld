#!/usr/bin/env python3
"""Fit an inspection model on defect-free frames and calibrate its threshold.

    python3 tools/train_inspection.py --part guide_block
    python3 tools/train_inspection.py --part all --backend statistical

Both supported detectors are unsupervised: they are shown only good parts and
learn what "normal" looks like. Defective samples are never used for fitting,
only for the report printed at the end.

EfficientAD training runs through anomalib's Engine and needs a GPU; the
statistical backend fits on CPU in seconds and is what CI uses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from _bootstrap import bootstrap

bootstrap()

from generate_mock_dataset import load_frame  # noqa: E402

from roboworld_core import paths  # noqa: E402
from roboworld_core.config import load_config  # noqa: E402
from roboworld_core.inspection import build_backend, model_path_for  # noqa: E402
from roboworld_core.mock_data import MockStation, parts_from_config  # noqa: E402
from roboworld_core.types import CameraIntrinsics  # noqa: E402


def load_or_generate(cfg, part_id: str, count: int, seed: int, split: str):
    """Load frames from data/mock, or render them if the dataset is absent."""
    intrinsics = CameraIntrinsics.from_config(cfg.section("camera"))
    directory = paths.data_dir() / "mock" / part_id / split
    files = sorted(directory.glob("frame_*.npz")) if directory.is_dir() else []

    if files:
        frames = [load_frame(path, intrinsics, part_id) for path in files[:count]]
        print(f"  loaded {len(frames)} {split} frames from {directory}")
        return frames

    print(f"  no dataset at {directory}; rendering {count} frames on the fly")
    station = MockStation(parts_from_config(cfg), intrinsics)
    return [
        station.sample_frame(part_id, defect=None, seed=seed + i, sequence=i)
        for i in range(count)
    ]


def train_statistical(cfg, part_id: str, args: argparse.Namespace) -> int:
    backend = build_backend(cfg, part_id, backend="statistical")
    frames = load_or_generate(cfg, part_id, args.frames, args.seed, "train")

    backend.fit(frames)
    model_path = Path(args.output) if args.output else model_path_for(
        cfg, "statistical", part_id
    )
    backend.save(model_path)

    scores = np.array([backend.infer(frame).anomaly_score for frame in frames])
    print(
        f"  fitted on {len(frames)} frames -> {model_path}\n"
        f"  threshold={backend.settings.threshold:.4f}  "
        f"train scores: median={np.median(scores):.4f} p99={np.percentile(scores, 99):.4f} "
        f"max={scores.max():.4f}"
    )
    if scores.max() >= backend.settings.threshold:
        print(
            "  WARNING: a training (defect-free) frame already reaches the threshold. "
            "Raise inspection.statistical.safety_factor or capture a wider normal set.",
            file=sys.stderr,
        )
    return 0


def train_efficientad(cfg, part_id: str, args: argparse.Namespace) -> int:
    """Train EfficientAD via anomalib's Engine."""
    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import EfficientAd
    except ImportError as exc:
        print(
            f"error: anomalib is not installed ({exc}).\n"
            "Install the GPU extras with `pip install -r requirements-gpu.txt`, "
            "or train the CPU backend with --backend statistical.",
            file=sys.stderr,
        )
        return 3

    dataset_dir = paths.resolve_path(
        cfg.get("inspection.training.dataset_dir")
    ) / part_id
    normal_dir = dataset_dir / "normal"
    if not normal_dir.is_dir() or not any(normal_dir.glob("*.png")):
        print(
            f"error: EfficientAD needs PNG images at {normal_dir}.\n"
            "Export them with `python3 tools/export_mock_images.py --part "
            f"{part_id}` or point inspection.training.dataset_dir at real captures.",
            file=sys.stderr,
        )
        return 4

    section = cfg.section("inspection.efficientad")
    size = section.get("image_size", [256, 256])
    datamodule = Folder(
        name=part_id,
        root=str(dataset_dir),
        normal_dir="normal",
        train_batch_size=int(cfg.get("inspection.training.batch_size", 1)),
    )
    model = EfficientAd(model_size=str(section.get("model_size", "small")))
    engine = Engine(max_epochs=int(cfg.get("inspection.training.max_epochs", 20)))
    engine.fit(model=model, datamodule=datamodule)

    model_path = Path(args.output) if args.output else model_path_for(
        cfg, "efficientad", part_id
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    engine.trainer.save_checkpoint(str(model_path))
    print(f"  EfficientAD checkpoint ({size[0]}x{size[1]}) -> {model_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", default="all", help="part id, or 'all'")
    parser.add_argument("--backend", default="statistical",
                        choices=("statistical", "efficientad"))
    parser.add_argument("--frames", type=int, default=40, help="training frames to use")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", default=None, help="explicit model output path")
    args = parser.parse_args()

    cfg = load_config()
    known = sorted(cfg.get("parts").keys())
    parts = known if args.part == "all" else [args.part]
    for part_id in parts:
        if part_id not in known:
            print(f"error: unknown part '{part_id}'; known: {known}", file=sys.stderr)
            return 2

    if args.output and len(parts) > 1:
        print("error: --output can only be used with a single --part", file=sys.stderr)
        return 2

    for part_id in parts:
        print(f"[{args.backend}] {part_id}")
        trainer = train_statistical if args.backend == "statistical" else train_efficientad
        code = trainer(cfg, part_id, args)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
