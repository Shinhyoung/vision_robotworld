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
import json
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

        from roboworld_core.inspection.efficientad import EfficientAdBackend
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

    anchor = _calibrate_anchor(cfg, model, normal_dir, tuple(size))
    anchor_path = EfficientAdBackend.anchor_path(model_path)
    anchor_path.write_text(json.dumps({
        "anchor": anchor,
        "images": len(sorted(normal_dir.glob("*.png"))),
        "percentile": float(cfg.get("inspection.efficientad.norm_high_percentile", 95.0)),
        "safety_factor": float(cfg.get("inspection.efficientad.safety_factor", 0.7)),
    }, indent=2), encoding="utf-8")
    print(f"  정규화 기준값 {anchor:.4f} -> {anchor_path}")

    _report_separation(cfg, model, dataset_dir, tuple(size), anchor)
    return 0


def _report_separation(cfg, model, dataset_dir: Path, image_size, anchor: float) -> None:
    """Score the held-out folders and report where a threshold can sit.

    Reported, never applied. ``abnormal/`` is validation data: choosing the
    threshold that happens to score best on it is fitting to the test set, and
    the number it produces would no longer mean anything. What a human needs to
    see is whether the two distributions separate at all -- if they overlap, no
    threshold fixes it and the answer is more or better training data.
    """
    import numpy as np

    scores = {}
    for folder in ("normal_test", "abnormal"):
        paths = sorted((dataset_dir / folder).glob("*.png"))
        if paths:
            scores[folder] = np.array([
                min(0.5 * _score_png(model, p, image_size) / anchor, 1.0) for p in paths
            ])

    good, bad = scores.get("normal_test"), scores.get("abnormal")
    if good is None or not len(good):
        print("  검증용 양품(normal_test)이 없어 분리도를 잴 수 없습니다")
        return

    threshold = float(cfg.get("inspection.threshold", 0.5))
    print(f"\n  검증 (임계값 {threshold})")
    print(f"    양품 {len(good):3d}장  중앙값 {np.median(good):.3f}  최대 {good.max():.3f}"
          f"   과검출 {int((good > threshold).sum())}/{len(good)}")
    if bad is None or not len(bad):
        print("    불량 샘플이 없어 미검출은 잴 수 없습니다 "
              "(tools/capture_part.py --defect 로 촬영)")
        return

    print(f"    불량 {len(bad):3d}장  중앙값 {np.median(bad):.3f}  최소 {bad.min():.3f}"
          f"   미검출 {int((bad <= threshold).sum())}/{len(bad)}")
    if bad.min() > good.max():
        print(f"    분리됨 — {good.max():.3f} ~ {bad.min():.3f} 사이 어디든 임계값 가능")
    else:
        best = min(
            ((t, int((good > t).sum()), int((bad <= t).sum()))
             for t in np.arange(0.05, 1.0, 0.01)),
            key=lambda x: x[1] + x[2],
        )
        print(f"    ⚠ 겹침 — 양품 최대 {good.max():.3f} > 불량 최소 {bad.min():.3f}. "
              "어떤 임계값에도 오류가 남습니다")
        print(f"    참고: 임계값 {best[0]:.2f} 이면 과검출 {best[1]}/{len(good)}, "
              f"미검출 {best[2]}/{len(bad)} — inspection.threshold 를 직접 정하세요")


def _score_png(model, path: Path, image_size) -> float:
    import numpy as np
    import torch
    from PIL import Image

    from roboworld_core.imageops import resize_nearest
    from roboworld_core.inspection.efficientad import _extract_map, _inference_module

    image = np.asarray(Image.open(path).convert("RGB"))
    resized = resize_nearest(image, image_size).astype(np.float32) / 255.0
    device = next(model.parameters()).device
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        raw = float(np.max(_extract_map(_inference_module(model)(tensor))))
    model.train(was_training)
    return raw


def _calibrate_anchor(cfg, model, normal_dir: Path, image_size: tuple[int, int]) -> float:
    """Anchor the [0, 1] score scale on the training set's own score level.

    anomalib emits an unbounded anomaly map and, since 2.x, publishes no
    threshold of its own -- ``model.pixel_threshold`` is ``None``. Without an
    anchor the raw map becomes the score: measured on mock guide_block a good
    part scored 0.665 against a 0.5 threshold, so every good part was rejected.

    Same rule as the statistical backend: score 0.5 lands at ``safety_factor``
    times the configured upper percentile of the *training* scores. Anchored on
    the level, not the spread -- a wider normal set must not drag defects under
    the threshold (see docs/handoff.md section 8).
    """
    import numpy as np
    import torch
    from PIL import Image

    from roboworld_core.imageops import resize_nearest
    from roboworld_core.inspection.efficientad import _extract_map, _inference_module

    # EfficientAD's own numbers. Borrowing the statistical backend's put score
    # 0.5 above most real defects: the two detectors place defects at different
    # multiples of the normal level, so one constant cannot serve both.
    percentile = float(cfg.get("inspection.efficientad.norm_high_percentile", 95.0))
    safety = float(cfg.get("inspection.efficientad.safety_factor", 0.7))

    # EfficientAd's forward branches on training mode -- in train mode it takes
    # a batch and returns losses, and feeding it a bare tensor fails inside
    # imagenet_norm_batch. Score the way the runtime backend will.
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    paths = sorted(normal_dir.glob("*.png"))
    scores = []
    for path in paths:
        image = np.asarray(Image.open(path).convert("RGB"))
        resized = resize_nearest(image, image_size).astype(np.float32) / 255.0
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            scores.append(float(np.max(_extract_map(_inference_module(model)(tensor)))))

    model.train(was_training)
    if not scores:
        raise ValueError(f"no PNGs to calibrate on in {normal_dir}")

    scores = np.asarray(scores, dtype=np.float64)
    keep, rejected = _reject_high_outliers(
        scores, float(cfg.get("inspection.training.outlier_z_max", 3.5))
    )
    print(f"  학습 점수 분포  중앙값 {np.median(scores):.4f}  "
          f"p{percentile:.0f} {np.percentile(scores, percentile):.4f}  "
          f"최대 {np.max(scores):.4f}  ({len(scores)}장)")

    if rejected:
        print(f"  이상치 {len(rejected)}장 제외 (캘리브레이션에서만):")
        for index, z in rejected:
            print(f"    {paths[index].name}  점수 {scores[index]:.4f}  "
                  f"(중앙값의 {scores[index] / max(np.median(scores), 1e-9):.1f}배, z={z:.1f})")
        print("    ↑ 손이 들어갔거나 분할이 어긋난 프레임일 수 있습니다. "
              "확인 후 원본에서 지우는 것을 권합니다.")
        kept = scores[keep]
        print(f"  제외 후  중앙값 {np.median(kept):.4f}  "
              f"p{percentile:.0f} {np.percentile(kept, percentile):.4f}  ({len(kept)}장)")
        scores = kept

    return float(safety * np.percentile(scores, percentile))


def _reject_high_outliers(scores: np.ndarray, z_max: float = 3.5):
    """Flag training frames scoring wildly above the rest.

    The anchor is a high percentile of the training scores, so a handful of
    contaminated frames drag it up and push real defects under the threshold.
    Measured on 64 real captures: one frame with the operator's hand still in
    shot scored 1.70 against a median of 0.075 -- 23x -- which lifted p95 from
    0.219 to 0.347 and made the detector miss 12 of 19 defects.

    Modified z-score on the median absolute deviation, so the test itself is not
    moved by the outliers it is looking for. **High side only**: a frame that
    scores low is a particularly clean part, not contamination.
    """
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    if mad <= 0.0:  # scores nearly identical -- nothing to reject against
        return np.ones(len(scores), dtype=bool), []

    z = 0.6745 * (scores - median) / mad
    keep = z <= z_max
    if keep.sum() < 3:  # refuse to calibrate on almost nothing
        return np.ones(len(scores), dtype=bool), []
    return keep, [(int(i), float(z[i])) for i in np.flatnonzero(~keep)]


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
