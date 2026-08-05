#!/usr/bin/env python3
"""Build a part mesh from D455 depth, so 6D pose works without CAD.

    python3 tools/reconstruct_part.py --part my_part

Live window; place the part on a flat surface, top face up, and capture a few
views. Then the mesh is built, measured and saved.

**Do not move or rotate the part between captures.** Views are stacked in one
shared frame, so extra views average the depth noise down -- they do not add
geometry. Anything that moves lands in the wrong place and smears the mesh; the
tool refuses a view that has drifted more than a few millimetres.

    SPACE / c   capture a view
    r           rebuild and preview the mesh from what is captured so far
    R           discard all captures and start over
    q / ESC     finish and save

Why depth rather than photos: these parts are matte and untextured, which is the
worst case for photogrammetry (no features to match) and irrelevant to depth.
Depth also gives metric scale directly.

What a single resting orientation can and cannot give you
---------------------------------------------------------
It reconstructs the top surface exactly and extrudes it down to the support
plane. That is the true shape for a prismatic part sitting on its flat face --
which most machined blocks are. It cannot see undercuts or features on the
underside; those come out as solid material.

Check the printed dimensions against calipers before trusting the pose.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from _bootstrap import bootstrap

bootstrap()

from capture_part import save_capture  # noqa: E402
from live_view import RealSenseSource, draw_hud  # noqa: E402

from roboworld_core import paths  # noqa: E402
from roboworld_core.config import load_config  # noqa: E402
from roboworld_core.mesh_io import save_ply  # noqa: E402
from roboworld_core.reconstruct import reconstruct  # noqa: E402
from roboworld_core.segmentation import segment_from_config  # noqa: E402
from roboworld_core.viz import colorize_depth, hstack_panels, tint_mask  # noqa: E402

_WHITE = (245, 245, 245)
_GREEN = (90, 230, 130)
_RED = (255, 90, 90)
_AMBER = (255, 200, 90)

_MIN_PIXELS = 800
_MAX_PIXELS = 120_000
#: How far the part may drift between captures before a view is refused, in mm.
#: Views are stacked in a shared frame, so drift smears the mesh directly; a few
#: millimetres is within depth noise, more is real movement.
_DRIFT_LIMIT_MM = 5.0


def report_lines(report) -> list[str]:
    e = report.extents_mm
    lines = [
        f"치수  {e[0]:.1f} x {e[1]:.1f} x {e[2]:.1f} mm",
        f"높이  {report.height_min_mm:.1f} ~ {report.height_max_mm:.1f} mm"
        f"   윗면 기울기 {report.top_tilt_deg:.2f}°"
        + (f"  (제거 {report.tilt_removed_deg:.2f}°)" if report.tilt_removed_deg else ""),
        f"뷰 {report.views_merged}   점 {report.points_used}   "
        f"셀 {report.cells_filled} (보간 {report.interpolated_fraction:.0%})",
        f"뷰 일치도  중심 산포 {report.view_centroid_spread_mm:.0f} mm   "
        f"외곽 편차 {report.view_extent_spread_mm:.0f} mm",
    ]
    return lines


def load_saved_views(cfg, part_id: str, views_dir) -> list:
    """Re-segment views saved by an earlier session, for an offline rebuild."""
    from register_part import load_capture

    files = sorted(views_dir.glob("frame_*.npz")) if views_dir.is_dir() else []
    return [
        segment_from_config(load_capture(path, part_id), cfg) for path in files
    ]


def finish(args, captured: list, views_dir) -> int:
    """Reconstruct, save and report. Shared by the live and offline paths."""
    selected = captured
    if args.single_view is not None:
        if not 0 <= args.single_view < len(captured):
            print(f"error: --single-view must be 0..{len(captured) - 1}", file=sys.stderr)
            return 2
        selected = [captured[args.single_view]]
        print(f"뷰 {args.single_view} 만 사용")

    try:
        mesh, report = reconstruct(
            selected, cell_size_m=args.cell_size_mm / 1000.0, level_top=args.level
        )
    except ValueError as exc:
        print(f"error: 복원 실패 — {exc}", file=sys.stderr)
        return 6

    output = args.output or str(paths.data_dir() / "meshes" / f"{args.part}.ply")
    path = save_ply(mesh, output)

    print(f"\n복원 결과 ({len(selected)} 뷰)")
    for line in report_lines(report):
        print(f"  {line}")
    print(f"  삼각형 {len(mesh.faces)}")
    for warning in report.warnings:
        print(f"  ⚠ {warning}")
    print(f"\n  메시 -> {path}")

    if report.warnings:
        print("\n경고가 있습니다. 촬영본은 저장되어 있으니 재촬영 없이 다시 시도할 수 있습니다:")
        print(f"  python3 tools/reconstruct_part.py --part {args.part} "
              f"--from-saved --single-view 0")
        print(f"  python3 tools/reconstruct_part.py --part {args.part} "
              f"--from-saved --cell-size-mm 1")
        print(f"  python3 tools/reconstruct_part.py --part {args.part} "
              f"--from-saved --level     # 각기둥형 부품의 기울기 제거")

    print("\n다음 단계:")
    print("  1. 치수를 실측(캘리퍼)과 대조하세요. 크게 다르면 재촬영")
    print(f"  2. python3 tools/capture_part.py  --part {args.part}      # 검사용 정상품 촬영")
    print(f"  3. python3 tools/register_part.py --part {args.part} --mesh {path}")
    print(f"  4. python3 tools/live_view.py --source realsense "
          f"--part {args.part} --inspect --pose")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", required=True)
    parser.add_argument("--cell-size-mm", type=float, default=2.0,
                        help="reconstruction grid resolution")
    parser.add_argument("--views", type=int, default=5,
                        help="target number of views (shown in the HUD)")
    parser.add_argument("--depth-range", type=float, nargs=2, default=(0.3, 1.5),
                        metavar=("NEAR", "FAR"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--from-saved", action="store_true",
                        help="rebuild from previously saved views; no camera needed")
    parser.add_argument("--single-view", type=int, default=None, metavar="N",
                        help="use only view N (0-based). Use when views disagree.")
    parser.add_argument("--use-station-roi", action="store_true",
                        help="isolate the part with pose.segmentation.station_roi "
                             "instead of taking the largest object. Use when the "
                             "rig is cluttered and the background wins")
    parser.add_argument("--level", action="store_true",
                        help="remove the measured top-face tilt. Correct for a "
                             "prismatic part (top parallel to bottom) that is "
                             "sitting slightly rocked on the surface.")
    args = parser.parse_args()

    if not args.part.replace("_", "").isalnum():
        print("error: --part must be alphanumeric with underscores", file=sys.stderr)
        return 2

    try:
        import cv2
    except ImportError:
        print("error: opencv-python required.\n"
              "  python3 -m pip install --user opencv-python", file=sys.stderr)
        return 3

    # Size-based identification is deliberately OFF here: this tool *defines*
    # the part's dimensions, so filtering candidates by dimensions we do not
    # have yet is circular -- and for a part being re-measured it would reject
    # exactly the tilted or unexpected placements worth capturing.
    #
    # The station ROI is off by default for the same class of reason: views are
    # normally shot on a turntable or by hand, not at the conveyor stop
    # position. --use-station-roi turns it back on for the case the default
    # cannot handle: a cluttered rig where the largest object above the plane is
    # not the part. Unlike size, position says nothing about the dimensions
    # being measured, so isolating by it stays non-circular.
    cfg = load_config().merged_with(
        {"pose": {"segmentation": {
            "identify_by_size": False,
            "station_roi": {"enabled": bool(args.use_station_roi)},
        }}}
    )
    if args.use_station_roi:
        from roboworld_core.segmentation import station_roi_from_config

        roi = station_roi_from_config(cfg)
        if roi is None:
            print("error: --use-station-roi 이지만 pose.segmentation.station_roi 가 "
                  "설정에 없습니다", file=sys.stderr)
            return 2
        print(f"스테이션 ROI 로 부품을 격리합니다: "
              f"{np.round(roi.center_m * 1000).astype(int)} "
              f"+-{np.round(roi.half_extents_m * 1000).astype(int)} mm")
    camera = cfg.section("camera")
    views_dir = paths.data_dir() / "captures" / args.part / "views"

    # Offline rebuild: retry different settings without re-shooting.
    if args.from_saved:
        captured = load_saved_views(cfg, args.part, views_dir)
        if not captured:
            print(f"error: no saved views at {views_dir}", file=sys.stderr)
            return 4
        print(f"{len(captured)} saved view(s) loaded from {views_dir}")
        return finish(args, captured, views_dir)

    try:
        source = RealSenseSource(
            cfg, args.part,
            int(camera.get("width", 640)), int(camera.get("height", 480)),
            int(camera.get("fps", 30)),
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 5

    views_dir.mkdir(parents=True, exist_ok=True)
    # A new session starts from frame_0000, so leftovers from a previous run
    # would survive wherever the new session captured fewer views -- and then
    # --from-saved would silently reconstruct from a mix of two sessions.
    stale = sorted(views_dir.glob("frame_*.npz"))
    if stale:
        for path in stale:
            path.unlink()
        print(f"이전 세션의 뷰 {len(stale)}장을 삭제하고 새로 시작합니다.")
    title = (f"reconstruct '{args.part}' - 부품 고정! | SPACE capture | "
             "r preview | R reset | q finish")
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1400, 400)

    captured: list = []
    preview: list[str] = []
    #: Reference centroid of the first capture, in camera frame. Views are merged
    #: by stacking them in a shared frame, so anything that moves between
    #: captures smears the mesh. Catching that at capture time beats discovering
    #: it after twenty shots.
    reference_centroid = None
    moved_warning = ""

    print("촬영 방법: 카메라와 부품을 모두 고정한 채 SPACE 를 여러 번 누르세요.")
    print("           부품을 움직이거나 회전시키면 안 됩니다 (뷰가 겹쳐 쌓입니다).")
    print("           q 로 종료하면 메시가 저장됩니다.")

    try:
        while True:
            frame = source.read()
            segmentation = segment_from_config(frame, cfg)
            pixels = segmentation.pixel_count

            if pixels < _MIN_PIXELS:
                status, color = "NO PART FOUND - 평평한 면 위에 놓으세요", _RED
            elif pixels > _MAX_PIXELS:
                status, color = "SEGMENT TOO LARGE - 배경이 잡힘", _RED
            else:
                status, color = "READY", _GREEN

            canvas = hstack_panels([
                frame.color,
                colorize_depth(frame.depth, tuple(args.depth_range)),
                tint_mask(frame.color, segmentation.mask),
            ])
            # Live drift check against the first capture.
            drift_mm = 0.0
            if reference_centroid is not None and segmentation.ok:
                drift_mm = float(
                    np.linalg.norm(segmentation.points.mean(axis=0) - reference_centroid)
                    * 1000.0
                )

            hud = [
                (f"{args.part}   views {len(captured)}/{args.views}", _WHITE),
                (status, color),
                (f"segmented {pixels} px", _AMBER),
            ]
            if reference_centroid is not None:
                hud.append((
                    f"drift {drift_mm:5.1f} mm" + ("   << 부품이 움직였습니다"
                                                   if drift_mm > _DRIFT_LIMIT_MM else ""),
                    _RED if drift_mm > _DRIFT_LIMIT_MM else _GREEN,
                ))
            if moved_warning:
                hud.append((moved_warning, _RED))
            hud += [(line, _WHITE) for line in preview]
            draw_hud(cv2, canvas, hud)
            cv2.imshow(title, canvas[..., ::-1])
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key in (ord(" "), ord("c")):
                if status != "READY":
                    print(f"  skipped: {status}")
                elif reference_centroid is not None and drift_mm > _DRIFT_LIMIT_MM:
                    moved_warning = (
                        f"거부: 부품이 {drift_mm:.0f} mm 움직였습니다. "
                        "원위치시키거나 r 로 초기화하세요"
                    )
                    print(f"  rejected: part moved {drift_mm:.0f} mm from the first view")
                else:
                    # Persist every view: a reconstruction that comes out wrong
                    # must never cost another trip to the camera.
                    save_capture(
                        views_dir / f"frame_{len(captured):04d}.npz", frame, segmentation
                    )
                    captured.append(segmentation)
                    if reference_centroid is None:
                        reference_centroid = segmentation.points.mean(axis=0)
                    moved_warning = ""
                    print(f"  captured view {len(captured)}  ({pixels} px)")
            if key == ord("R"):
                captured.clear()
                reference_centroid = None
                moved_warning = ""
                preview = []
                print("  reset: 촬영본을 비우고 다시 시작합니다")
            if key == ord("r") and captured:
                try:
                    _, report = reconstruct(
                        captured, cell_size_m=args.cell_size_mm / 1000.0,
                        level_top=args.level,
                    )
                    preview = report_lines(report)
                    print("  preview: " + " | ".join(preview))
                except ValueError as exc:
                    preview = [f"복원 실패: {exc}"]

            try:
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        source.close()
        cv2.destroyAllWindows()

    if not captured:
        print("촬영된 뷰가 없습니다.", file=sys.stderr)
        return 4
    print(f"\n{len(captured)} 뷰 저장됨 -> {views_dir}")
    return finish(args, captured, views_dir)


if __name__ == "__main__":
    raise SystemExit(main())
