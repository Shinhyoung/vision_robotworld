#!/usr/bin/env python3
"""Capture a new part with the D455 so it can be registered without CAD.

    python3 tools/capture_part.py --part my_part            # 정상품 촬영
    python3 tools/capture_part.py --part my_part --defect   # 불량 샘플 촬영

Opens a live window with the segmentation overlay so you can confirm the part is
actually being separated from the table **before** spending captures on it.

    SPACE / c  capture the current frame
    a          auto-capture on/off (one frame every --interval seconds)
    q / ESC    finish

Captures are written to ``data/captures/<part_id>/{normal,defect}/*.npz`` in the
same format as the mock dataset, so every existing tool can read them.

What to shoot
-------------
* **normal/**: only defect-free parts. Both detectors are unsupervised -- they
  learn what good looks like, so a single defective frame in here teaches the
  model that the defect is normal.
* Vary what the line will vary: position, rotation, and which face is up. The
  model can only tolerate variation it has seen.
* Keep the camera at the working distance and the part on a flat surface --
  segmentation finds the dominant plane and takes what sits above it.
* 40+ frames is a sensible minimum; more is better.

``defect/`` frames are never used for fitting, only for checking the result.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from _bootstrap import bootstrap

bootstrap()

from live_view import RealSenseSource, draw_hud  # noqa: E402

from roboworld_core import paths  # noqa: E402
from roboworld_core.config import load_config  # noqa: E402
from roboworld_core.segmentation import segment_from_config  # noqa: E402
from roboworld_core.viz import colorize_depth, hstack_panels, tint_mask  # noqa: E402

_WHITE = (245, 245, 245)
_GREEN = (90, 230, 130)
_RED = (255, 90, 90)
_AMBER = (255, 200, 90)

#: Segmented pixel counts outside this band usually mean a bad setup rather than
#: a bad part: too few = nothing found, too many = the plane fit grabbed the
#: background instead of the table.
_MIN_PIXELS = 800
_MAX_PIXELS = 120_000


def save_capture(path, frame, segmentation) -> None:
    """Store one capture in the same ``.npz`` layout as the mock dataset."""
    np.savez_compressed(
        path,
        color=frame.color,
        depth=frame.depth,
        # No ground-truth pose exists for a real capture; the fields are kept so
        # the file stays loadable by tools written against the mock dataset.
        gt_position=np.zeros(3),
        gt_orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        is_good=True,
        defect="",
        stamp=frame.stamp,
        sequence=frame.sequence,
        part_mask=segmentation.mask,
        intrinsics=np.array([
            frame.intrinsics.fx, frame.intrinsics.fy,
            frame.intrinsics.cx, frame.intrinsics.cy,
            frame.intrinsics.width, frame.intrinsics.height,
        ]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", required=True, help="new part id, e.g. my_part")
    parser.add_argument("--defect", action="store_true",
                        help="write to defect/ instead of normal/")
    parser.add_argument("--interval", type=float, default=0.7,
                        help="auto-capture period in seconds")
    parser.add_argument("--target", type=int, default=40,
                        help="capture count to aim for (shown in the HUD)")
    parser.add_argument("--depth-range", type=float, nargs=2, default=(0.3, 1.5),
                        metavar=("NEAR", "FAR"))
    args = parser.parse_args()

    if not args.part.replace("_", "").isalnum():
        print("error: --part must be alphanumeric with underscores", file=sys.stderr)
        return 2

    try:
        import cv2
    except ImportError:
        print("error: opencv-python required.\n  python3 -m pip install --user opencv-python",
              file=sys.stderr)
        return 3

    # Identification off: a part being registered for the first time has no
    # geometry to match against, and one being re-captured should not have its
    # training frames silently filtered by the previous model's dimensions.
    # The station ROI is off for the same reason: capturing happens wherever the
    # camera and part can be set up, not at the conveyor stop position.
    cfg = load_config().merged_with(
        {"pose": {"segmentation": {
            "identify_by_size": False,
            "station_roi": {"enabled": False},
        }}}
    )
    camera = cfg.section("camera")
    try:
        source = RealSenseSource(
            cfg, args.part,
            int(camera.get("width", 640)), int(camera.get("height", 480)),
            int(camera.get("fps", 30)),
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 5

    split = "defect" if args.defect else "normal"
    output_dir = paths.data_dir() / "captures" / args.part / split
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(output_dir.glob("*.npz")))

    title = f"capture '{args.part}' [{split}] - SPACE capture | a auto | q finish"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1400, 400)

    saved = 0
    auto = False
    last_auto = 0.0
    print(f"capturing to {output_dir}  (already there: {existing})")
    print("  SPACE/c capture | a auto | q finish")

    try:
        while True:
            frame = source.read()
            segmentation = segment_from_config(frame, cfg)
            pixels = segmentation.pixel_count

            if pixels < _MIN_PIXELS:
                status, color = "NO PART FOUND - 평평한 면 위에 놓고 카메라를 맞추세요", _RED
            elif pixels > _MAX_PIXELS:
                status, color = "SEGMENT TOO LARGE - 배경이 잡힘. 카메라를 더 가까이", _RED
            else:
                status, color = "READY", _GREEN

            canvas = hstack_panels([
                frame.color,
                colorize_depth(frame.depth, tuple(args.depth_range)),
                tint_mask(frame.color, segmentation.mask),
            ])
            draw_hud(cv2, canvas, [
                (f"{args.part} [{split}]   saved {existing + saved}/{args.target}"
                 f"{'   AUTO' if auto else ''}", _WHITE),
                (status, color),
                (f"segmented {pixels} px", _AMBER),
            ])
            cv2.imshow(title, canvas[..., ::-1])
            key = cv2.waitKey(1) & 0xFF

            capture = key in (ord(" "), ord("c"))
            if auto and status == "READY" and (time.time() - last_auto) >= args.interval:
                capture, last_auto = True, time.time()

            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                auto = not auto
                last_auto = 0.0
            if capture:
                if status != "READY":
                    print(f"  skipped: {status}")
                else:
                    index = existing + saved
                    save_capture(output_dir / f"frame_{index:04d}.npz", frame, segmentation)
                    saved += 1
                    print(f"  captured {index + 1}  ({pixels} px)")

            try:
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        source.close()
        cv2.destroyAllWindows()

    total = existing + saved
    print(f"\n{saved} new capture(s); {total} total in {output_dir}")
    if not args.defect:
        if total < 20:
            print(f"WARNING: {total} frames is thin for fitting a detector. "
                  "Aim for 40+ covering the placement variation you expect.")
        print("\nNext:")
        print(f"  python3 tools/register_part.py --part {args.part}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
