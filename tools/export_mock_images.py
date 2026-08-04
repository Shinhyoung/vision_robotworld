#!/usr/bin/env python3
"""Export the mock dataset as PNG folders for anomalib (ticket INS-4).

    python3 tools/export_mock_images.py --part guide_block

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


def export(part_id: str, source_root: Path, output_root: Path) -> dict[str, int]:
    counts = {"normal": 0, "abnormal": 0, "normal_test": 0}
    part_out = output_root / part_id
    for folder in counts:
        (part_out / folder).mkdir(parents=True, exist_ok=True)

    train_dir = source_root / part_id / "train"
    for path in sorted(train_dir.glob("frame_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            write_png(part_out / "normal" / f"{counts['normal']:05d}.png", data["color"])
        counts["normal"] += 1

    test_dir = source_root / part_id / "test"
    for path in sorted(test_dir.glob("frame_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            folder = "normal_test" if bool(data["is_good"]) else "abnormal"
            write_png(part_out / folder / f"{counts[folder]:05d}.png", data["color"])
        counts[folder] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", default="all")
    parser.add_argument("--source", default=None, help="dataset root (default: data/mock)")
    parser.add_argument("--output", default=None, help="output root (default: data/mock_images)")
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
                f"`python3 tools/generate_mock_dataset.py --part {part_id}` first.",
                file=sys.stderr,
            )
            return 3
        counts = export(part_id, source_root, output_root)
        print(
            f"{part_id}: normal={counts['normal']} abnormal={counts['abnormal']} "
            f"normal_test={counts['normal_test']} -> {output_root / part_id}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
