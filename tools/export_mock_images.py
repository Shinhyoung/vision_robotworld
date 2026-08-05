#!/usr/bin/env python3
"""Export a dataset as PNG folders for anomalib (ticket INS-4).

    python3 tools/export_mock_images.py --part guide_block
    python3 tools/export_mock_images.py --part new_part \
        --source data/captures --output data/capture_images   # real frames

anomalib's ``Folder`` datamodule wants an image directory tree, not the ``.npz``
archives ``generate_mock_dataset.py`` writes. Layout produced::

    data/mock_images/<part_id>/normal/000.png      (from train/, defect-free)
    data/mock_images/<part_id>/abnormal/000.png    (from test/, defective)
    data/mock_images/<part_id>/normal_test/000.png (from test/, defect-free)

Only ``normal/`` is used for fitting -- EfficientAD is unsupervised. The other
two folders are for validation metrics.

PNG writing goes through Pillow when available and falls back to a minimal
built-in encoder, so this runs on a bare numpy install.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from _bootstrap import bootstrap

bootstrap()

from roboworld_core import paths  # noqa: E402
from roboworld_core.config import load_config  # noqa: E402


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Write an RGB uint8 array as a PNG."""
    try:
        from PIL import Image

        Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(path)
        return
    except ImportError:
        pass

    # Minimal encoder: filter type 0 per scanline, single IDAT.
    array = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width = array.shape[:2]
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def crop_to_part(cfg, part_id: str, data) -> np.ndarray:
    """The same window the EfficientAD backend feeds its model at runtime.

    Exporting whole frames trains the detector on a picture that is 98 % belt.
    Measured: the part covers 1.8 % of a 640x480 frame, so at the model's
    256x256 input it survives as ~1170 px and a 15 px defect becomes 6 px --
    EfficientAD missed 24/24 defects that way.

    Goes through ``part_crop_box`` so training and inference cannot drift: a
    crop that differs between them shifts the score distribution and quietly
    invalidates the calibrated normalisation anchor.
    """
    from roboworld_core.segmentation import part_crop_box, segment_from_config
    from roboworld_core.types import CameraIntrinsics, Frame

    # The mock archives carry no intrinsics or mask (see generate_mock_dataset);
    # a real capture does, but this tool only ever reads the mock set.
    intrinsics = CameraIntrinsics.from_config(cfg.section("camera"))
    frame = Frame(data["color"], data["depth"], intrinsics, 0.0, part_id=part_id)

    mask = np.asarray(data["part_mask"], dtype=bool) if "part_mask" in data.files else None
    if mask is None or not mask.any():
        mask = segment_from_config(frame, cfg, part_id=part_id).mask
    r0, r1, c0, c1 = part_crop_box(np.asarray(mask, dtype=bool))
    return np.ascontiguousarray(data["color"][r0:r1, c0:c1])


def plan_mock(source: Path) -> dict[str, list[tuple[Path, str]]]:
    """Mock layout: ``<part>/train`` is all normal, ``<part>/test`` is labelled."""
    plan: dict[str, list] = {"normal": [], "abnormal": [], "normal_test": []}
    for path in sorted((source / "train").glob("frame_*.npz")):
        plan["normal"].append(path)
    for path in sorted((source / "test").glob("frame_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            good = bool(data["is_good"])
        plan["normal_test" if good else "abnormal"].append(path)
    return plan


def plan_captures(source: Path, holdout: float) -> dict[str, list[Path]]:
    """Real-capture layout: ``<part>/normal`` and ``<part>/defect``.

    ``tools/capture_part.py`` writes every good frame into one folder, so the
    held-out normals have to be carved out here. Without them there is nothing
    to measure false rejects on -- and a detector fitted on every good frame it
    will ever be scored against reports a false-reject rate of zero that means
    nothing.

    The holdout is taken by stride rather than from the tail: captures arrive in
    the order the part was turned, so the last N frames are all one orientation.
    """
    normal = sorted((source / "normal").glob("*.npz"))
    defect = sorted((source / "defect").glob("*.npz")) if (source / "defect").is_dir() else []

    step = max(int(round(1.0 / holdout)), 2) if holdout > 0 else 0
    held = set(normal[::step]) if step else set()
    return {
        "normal": [p for p in normal if p not in held],
        "normal_test": sorted(held),
        "abnormal": defect,
    }


def export(part_id: str, source_root: Path, output_root: Path, cfg,
           holdout: float = 0.2) -> dict[str, int]:
    part_out = output_root / part_id
    source = source_root / part_id
    for folder in ("normal", "abnormal", "normal_test"):
        (part_out / folder).mkdir(parents=True, exist_ok=True)

    if (source / "train").is_dir():
        plan = plan_mock(source)
    elif (source / "normal").is_dir():
        plan = plan_captures(source, holdout)
    else:
        raise FileNotFoundError(
            f"{source} has neither train/ (mock dataset) nor normal/ (captures)"
        )

    counts = {}
    for folder, sources in plan.items():
        for index, source_path in enumerate(sources):
            with np.load(source_path, allow_pickle=False) as data:
                write_png(part_out / folder / f"{index:05d}.png",
                          crop_to_part(cfg, part_id, data))
        counts[folder] = len(sources)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", default="all")
    parser.add_argument("--source", default=None, help="dataset root (default: data/mock)")
    parser.add_argument("--output", default=None, help="output root (default: data/mock_images)")
    parser.add_argument("--holdout", type=float, default=0.2,
                        help="fraction of real captures kept aside as normal_test "
                             "(captures layout only; the mock set is already split)")
    args = parser.parse_args()

    cfg = load_config()
    source_root = Path(args.source) if args.source else paths.data_dir() / "mock"
    output_root = Path(args.output) if args.output else paths.data_dir() / "mock_images"

    known = sorted(cfg.get("parts").keys())
    parts = known if args.part == "all" else [args.part]

    for part_id in parts:
        if part_id not in known:
            print(f"error: unknown part '{part_id}'; known: {known}", file=sys.stderr)
            return 2
        if not (source_root / part_id).is_dir():
            print(
                f"error: no dataset at {source_root / part_id}. Run "
                f"`python3 tools/generate_mock_dataset.py --part {part_id}` first, "
                f"or point --source at data/captures for real frames.",
                file=sys.stderr,
            )
            return 3
        counts = export(part_id, source_root, output_root, cfg, args.holdout)
        print(
            f"{part_id}: normal={counts['normal']} abnormal={counts['abnormal']} "
            f"normal_test={counts['normal_test']} -> {output_root / part_id}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
