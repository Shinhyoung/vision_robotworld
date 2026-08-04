#!/usr/bin/env python3
"""Render what the pipeline saw and decided, as a PNG panel.

    python3 tools/visualize.py --part guide_block --defect chip

A vision pipeline that only prints numbers is hard to trust. This dumps the
four images that matter for one cycle, side by side:

    [ color ] [ depth ] [ segmentation ROI ] [ anomaly map + defect mask ]

Use it to sanity-check a threshold change, to see *where* a defect was found,
or to confirm the part was segmented off the belt at all. Written by the Team
Lead as the visual counterpart to tools/e2e_dryrun.py; ticket INS-8 turns the
same information into an RViz overlay for live operation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from _bootstrap import bootstrap

bootstrap()

from export_mock_images import write_png  # noqa: E402

from roboworld_core import paths  # noqa: E402
from roboworld_core.config import load_config  # noqa: E402
from roboworld_core.inspection import build_backend as build_inspection  # noqa: E402
from roboworld_core.mock_data import MockStation, parts_from_config  # noqa: E402
from roboworld_core.pose import build_backend as build_pose  # noqa: E402
from roboworld_core.segmentation import segment_from_config  # noqa: E402
from roboworld_core.symmetry import group_for_part, pose_error  # noqa: E402
from roboworld_core.types import CameraIntrinsics  # noqa: E402

#: Gap between panels, in pixels.
_GAP = 8
_LABEL_BAR = 3


def colorize(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Blue -> red heat map for a [0, 1] array; invalid pixels stay dark."""
    v = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    rgb = np.zeros((*v.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(2.0 * v - 0.5, 0, 1)          # red rises late
    rgb[..., 1] = np.clip(1.5 - np.abs(3.0 * v - 1.5), 0, 1)  # green peaks mid
    rgb[..., 2] = np.clip(1.5 - 3.0 * v, 0, 1)          # blue falls early
    rgb[~valid] = 0.08
    return (rgb * 255).astype(np.uint8)


def panel(images: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """Lay images out horizontally with a coloured bar marking each panel."""
    height = max(img.shape[0] for _, img in images)
    width = sum(img.shape[1] for _, img in images) + _GAP * (len(images) - 1)
    canvas = np.full((height + _LABEL_BAR, width, 3), 24, dtype=np.uint8)

    bar_colors = [(90, 160, 255), (120, 200, 140), (255, 200, 90), (255, 110, 110)]
    x = 0
    for index, (_, image) in enumerate(images):
        h, w = image.shape[:2]
        canvas[_LABEL_BAR:_LABEL_BAR + h, x:x + w] = image
        canvas[0:_LABEL_BAR, x:x + w] = bar_colors[index % len(bar_colors)]
        x += w + _GAP
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", default="guide_block")
    parser.add_argument("--defect", default="chip",
                        choices=("none", "scratch", "dent", "stain", "chip"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fit-frames", type=int, default=25)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config()
    if args.part not in cfg.get("parts"):
        print(f"error: unknown part '{args.part}'", file=sys.stderr)
        return 2

    intrinsics = CameraIntrinsics.from_config(cfg.section("camera"))
    station = MockStation(parts_from_config(cfg), intrinsics)
    defect = None if args.defect == "none" else args.defect

    inspection = build_inspection(cfg, args.part, backend="statistical")
    if not inspection.is_fitted:
        print(f"fitting on {args.fit_frames} defect-free frames...", flush=True)
        inspection.fit([
            station.sample_frame(args.part, seed=90_000 + i, sequence=i)
            for i in range(args.fit_frames)
        ])

    frame = station.sample_frame(args.part, defect=defect, seed=args.seed, sequence=1)
    segmentation = segment_from_config(frame, cfg)
    result = inspection.infer(frame)
    estimate = build_pose(cfg, args.part, backend="icp").run(frame)

    # --- panels ---------------------------------------------------------
    color = frame.color

    depth_valid = frame.depth > 0
    span = frame.depth[depth_valid]
    normalized = np.zeros_like(frame.depth)
    if span.size:
        normalized[depth_valid] = 1.0 - (frame.depth[depth_valid] - span.min()) / max(
            span.max() - span.min(), 1e-6
        )
    depth_view = colorize(normalized, depth_valid)

    # Segmentation: part tinted green over a dimmed original.
    roi_view = (color * 0.35).astype(np.uint8)
    roi_view[segmentation.mask] = (
        color[segmentation.mask] * 0.5 + np.array([0, 255, 90]) * 0.5
    ).astype(np.uint8)

    # Anomaly: heat map inside the ROI, defect mask outlined in red.
    anomaly = result.anomaly_map if result.anomaly_map is not None else np.zeros(color.shape[:2])
    anomaly_view = colorize(anomaly, segmentation.mask)
    if result.defect_mask is not None:
        defect_pixels = result.defect_mask > 0
        anomaly_view[defect_pixels] = (255, 40, 40)

    canvas = panel([
        ("color", color),
        ("depth", depth_view),
        ("segmentation", roi_view),
        ("anomaly", anomaly_view),
    ])

    output = Path(args.output) if args.output else (
        paths.data_dir() / f"view_{args.part}_{args.defect}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_png(output, canvas)

    verdict = "OK (양품)" if result.is_good else "NG (불량)"
    defect_px = int((result.defect_mask > 0).sum()) if result.defect_mask is not None else 0
    print(f"part={args.part} injected_defect={args.defect}")
    print(f"  inspection : {verdict}  score={result.anomaly_score:.3f} "
          f"threshold={result.threshold:.3f}  defect_px={defect_px}")
    print(f"  segmentation: {segmentation.pixel_count} px on the part")
    if estimate.valid:
        error = pose_error(estimate.pose, frame.gt_pose, group_for_part(cfg, args.part))
        position = estimate.pose.position
        print(f"  pose       : ({position[0]:+.4f}, {position[1]:+.4f}, {position[2]:+.4f}) m "
              f"in '{estimate.pose.frame_id}'  fitness={estimate.fitness:.2f}")
        print(f"               error {error['translation_mm']:.2f} mm / "
              f"{error['rotation_deg_symmetry_reduced']:.2f} deg (대칭 보정)")
    else:
        print(f"  pose       : rejected -- {estimate.message}")
    print(f"  panel      : {output}   [color | depth | segmentation | anomaly]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
