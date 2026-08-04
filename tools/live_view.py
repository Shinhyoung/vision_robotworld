#!/usr/bin/env python3
"""Live camera / depth viewer for the index station.

    python3 tools/live_view.py                      # 컬러 + 깊이 실시간 창
    python3 tools/live_view.py --inspect            # + 분할 + 이상맵 (검사 동작)
    python3 tools/live_view.py --inspect --pose     # + 6D 포즈 (느림)

Opens a window streaming the RGB-D frames the pipeline sees. Under WSL2 this
needs WSLg (Windows 11) -- it is already active if ``$DISPLAY`` is set. With no
display available, use ``--backend mjpeg`` and open the printed URL in a
browser on the Windows side; that path needs no GUI libraries at all.

Sources (``--source``)
----------------------
``mock``       frames rendered from the part CAD (default; no hardware needed)
``realsense``  **live D455 RGB-D** through pyrealsense2, no ROS required
``dataset``    replay a dataset written by tools/generate_mock_dataset.py

    python3 tools/live_view.py --source realsense   # 실제 카메라 실시간 영상

The camera must be attached to WSL first (docs/setup_wsl.md section 4)::

    usbipd list                                  # find the BUSID
    usbipd bind   --busid <BUSID>                # once, as Administrator
    usbipd attach --wsl --busid <BUSID>          # each session
    usbipd detach --busid <BUSID>                # give it back to Windows

A **rosbag** stream arrives as ROS topics instead; view that with RViz2 against
the running graph::

    ros2 launch roboworld_bringup rosbag_replay.launch.py bag:=<path>
    rviz2

Keys
----
q / ESC  quit          space  pause        s  save a PNG snapshot
d        cycle the injected defect (none -> scratch -> dent -> stain -> chip)
p        cycle the part type
i        toggle inspection panels        o  toggle the pose overlay
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from _bootstrap import bootstrap

bootstrap()

from export_mock_images import write_png  # noqa: E402

from roboworld_core import paths  # noqa: E402
from roboworld_core.config import load_config  # noqa: E402
from roboworld_core.geometry import quaternion_to_euler_deg  # noqa: E402
from roboworld_core.inspection import build_backend as build_inspection  # noqa: E402
from roboworld_core.mock_data import DEFECT_KINDS, MockStation, parts_from_config  # noqa: E402
from roboworld_core.pose import build_backend as build_pose  # noqa: E402
from roboworld_core.pose import load_part_mesh  # noqa: E402
from roboworld_core.segmentation import (  # noqa: E402
    segment_from_config,
    station_roi_from_config,
)
from roboworld_core.types import CameraIntrinsics  # noqa: E402
from roboworld_core.viz import (  # noqa: E402
    anomaly_view,
    colorize_depth,
    draw_station_roi,
    hstack_panels,
    pose_overlay,
    tint_mask,
)

DEFECT_CYCLE = (None, *DEFECT_KINDS)

#: HUD colours (RGB).
_WHITE = (245, 245, 245)
_GREEN = (90, 230, 130)
_RED = (255, 90, 90)
_DIM = (150, 150, 150)


# --------------------------------------------------------------------------
# frame sources
# --------------------------------------------------------------------------
class MockSource:
    """Renders a fresh frame per tick from the part CAD."""

    def __init__(self, cfg, part_id: str, seed: int = 0) -> None:
        self.cfg = cfg
        self.station = MockStation(
            parts_from_config(cfg), CameraIntrinsics.from_config(cfg.section("camera"))
        )
        self.parts = sorted(cfg.get("parts").keys())
        self.part_id = part_id
        self.defect = None
        self.seed = seed
        self._counter = 0

    def next_part(self) -> str:
        self.part_id = self.parts[(self.parts.index(self.part_id) + 1) % len(self.parts)]
        return self.part_id

    def next_defect(self) -> str | None:
        self.defect = DEFECT_CYCLE[(DEFECT_CYCLE.index(self.defect) + 1) % len(DEFECT_CYCLE)]
        return self.defect

    def read(self):
        self._counter += 1
        return self.station.sample_frame(
            self.part_id,
            defect=self.defect,
            seed=self.seed + self._counter * 17,
            sequence=self._counter,
            stamp=time.time(),
        )

    def close(self) -> None:
        """No resources to release; kept so every source has one interface."""


class RealSenseSource:
    """Live RGB-D straight from a D455 via ``pyrealsense2`` -- no ROS involved.

    This is the fastest way to look at the real sensor: attach the camera to WSL
    with ``usbipd``, then stream. The ROS path (realsense-ros + RViz) is still
    what production uses; this exists so the camera can be checked, aimed and
    focused without building the workspace first.

    Depth is aligned to colour and converted to meters here, exactly as
    ``roboworld_ros_utils.conversions`` does for the ROS path, so the frames are
    interchangeable with mock frames downstream.

    Intrinsics come from the **device**, not from camera.yaml -- the YAML values
    are documented placeholders and must never be used for real measurements.
    """

    def __init__(self, cfg, part_id: str, width: int, height: int, fps: int) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is required for --source realsense:\n"
                "  python3 -m pip install --user pyrealsense2\n"
                "and the camera must be attached to WSL:\n"
                "  usbipd attach --wsl --busid <BUSID>   (see docs/setup_wsl.md section 4)"
            ) from exc

        self._rs = rs
        self.cfg = cfg
        self.parts = sorted(cfg.get("parts").keys())
        self.part_id = part_id
        self.defect = None  # a real camera sees whatever is actually there
        self._counter = 0

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        try:
            profile = self.pipeline.start(config)
        except RuntimeError as exc:
            raise RuntimeError(
                f"could not start the RealSense stream ({exc}).\n"
                "Checks: is the camera attached to WSL (`lsusb | grep 8086`)? "
                "Is it on a USB 3 port? Is another process already using it?"
            ) from exc

        # ICD section 3: depth must be registered to colour.
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        video = profile.get_stream(rs.stream.color).as_video_stream_profile()
        native = video.get_intrinsics()
        self.intrinsics = CameraIntrinsics(
            width=native.width, height=native.height,
            fx=native.fx, fy=native.fy, cx=native.ppx, cy=native.ppy,
            frame_id=str(cfg.get("camera.optical_frame_id", "camera_color_optical_frame")),
        )
        device = profile.get_device()
        print(f"  RealSense {device.get_info(rs.camera_info.name)} "
              f"sn={device.get_info(rs.camera_info.serial_number)} "
              f"fw={device.get_info(rs.camera_info.firmware_version)}")
        print(f"  intrinsics {native.width}x{native.height} "
              f"fx={native.fx:.1f} fy={native.fy:.1f} "
              f"cx={native.ppx:.1f} cy={native.ppy:.1f}  "
              f"depth_scale={self.depth_scale} m/unit")

    def next_part(self) -> str:
        self.part_id = self.parts[(self.parts.index(self.part_id) + 1) % len(self.parts)]
        return self.part_id

    def next_defect(self) -> str | None:
        return None  # nothing to inject into a real scene

    def read(self):
        from roboworld_core.types import Frame

        self._counter += 1
        frames = self.align.process(self.pipeline.wait_for_frames(5000))
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = (
            np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32)
            * self.depth_scale
        )
        return Frame(
            color=color.copy(),
            depth=depth,
            intrinsics=self.intrinsics,
            stamp=time.time(),
            sequence=self._counter,
            part_id=self.part_id,
        )

    def close(self) -> None:
        self.pipeline.stop()


class DatasetSource:
    """Replays ``.npz`` frames written by tools/generate_mock_dataset.py."""

    def __init__(self, cfg, part_id: str, split: str = "test") -> None:
        from generate_mock_dataset import load_frame

        self._load_frame = load_frame
        self.cfg = cfg
        self.intrinsics = CameraIntrinsics.from_config(cfg.section("camera"))
        self.parts = sorted(cfg.get("parts").keys())
        self.part_id = part_id
        self.split = split
        self.defect = None  # dataset frames carry their own labels
        self._index = 0
        self._files = self._scan()

    def _scan(self) -> list[Path]:
        directory = paths.data_dir() / "mock" / self.part_id / self.split
        files = sorted(directory.glob("frame_*.npz")) if directory.is_dir() else []
        if not files:
            raise FileNotFoundError(
                f"no dataset frames at {directory}. Generate them with:\n"
                f"  python3 tools/generate_mock_dataset.py --part {self.part_id}"
            )
        return files

    def next_part(self) -> str:
        self.part_id = self.parts[(self.parts.index(self.part_id) + 1) % len(self.parts)]
        self._files = self._scan()
        self._index = 0
        return self.part_id

    def next_defect(self) -> str | None:
        return None  # replay has fixed content

    def read(self):
        path = self._files[self._index % len(self._files)]
        self._index += 1
        return self._load_frame(path, self.intrinsics, self.part_id)

    def close(self) -> None:
        """No resources to release; kept so every source has one interface."""


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
class Analyzer:
    """Lazily builds and caches per-part inspection / pose backends."""

    def __init__(self, cfg, fit_frames: int) -> None:
        self.cfg = cfg
        self.fit_frames = fit_frames
        self._inspection: dict[str, object] = {}
        self._pose: dict[str, object] = {}
        self._mesh: dict[str, object] = {}

    def inspection(self, part_id: str):
        if part_id not in self._inspection:
            backend = build_inspection(self.cfg, part_id, backend="statistical")
            if not backend.is_fitted:
                print(f"  fitting inspection for '{part_id}' "
                      f"({self.fit_frames} frames)...", flush=True)
                station = MockStation(
                    parts_from_config(self.cfg),
                    CameraIntrinsics.from_config(self.cfg.section("camera")),
                )
                backend.fit([
                    station.sample_frame(part_id, seed=90_000 + i, sequence=i)
                    for i in range(self.fit_frames)
                ])
            self._inspection[part_id] = backend
        return self._inspection[part_id]

    def pose(self, part_id: str):
        if part_id not in self._pose:
            self._pose[part_id] = build_pose(self.cfg, part_id, backend="icp")
        return self._pose[part_id]

    def mesh(self, part_id: str):
        """CAD/reconstructed mesh, for re-projecting the estimated pose."""
        if part_id not in self._mesh:
            self._mesh[part_id] = load_part_mesh(self.cfg, part_id)
        return self._mesh[part_id]


def draw_hud(cv2, canvas: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    """Draw HUD text with a dark plate behind it so it stays readable."""
    if not lines:
        return
    pad, line_height = 8, 22
    height = pad * 2 + line_height * len(lines)
    width = max(len(text) for text, _ in lines) * 11 + pad * 2

    plate = canvas[0:height, 0:width].astype(np.float32) * 0.25
    canvas[0:height, 0:width] = plate.astype(np.uint8)
    for index, (text, color) in enumerate(lines):
        cv2.putText(
            canvas, text, (pad, pad + line_height * (index + 1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )


def apply_gain(color: np.ndarray, gain: float) -> np.ndarray:
    """Brighten the colour panel for display only.

    The mock renderer's lighting is dim; this makes the stream easier to read
    without touching the pixels the detector actually sees. Display-only, so it
    can never influence a score.
    """
    if gain == 1.0:
        return color
    return np.clip(np.asarray(color, dtype=np.float32) * gain, 0, 255).astype(np.uint8)


def build_canvas(cfg, frame, analyzer: Analyzer, show_inspect: bool, show_pose: bool,
                 depth_range: tuple[float, float], cv2,
                 gain: float = 1.0) -> tuple[np.ndarray, list]:
    """Compose the panel image and the HUD lines for one frame."""
    panels = [apply_gain(frame.color, gain), colorize_depth(frame.depth, depth_range)]
    hud: list[tuple[str, tuple[int, int, int]]] = []

    # Segmentation feeds both inspection and pose, so either one needs it.
    if show_inspect or show_pose:
        segmentation = segment_from_config(frame, cfg, part_id=frame.part_id)
        roi = station_roi_from_config(cfg)
        if segmentation.ok:
            panels.append(tint_mask(apply_gain(frame.color, gain), segmentation.mask))
            hud.append((f"segmented {segmentation.pixel_count} px", _DIM))
        else:
            # Identification refused everything. Showing a blank panel leaves the
            # operator with no idea where to aim, so paint the closest candidate
            # in red and report what it measured -- that is what tells you
            # whether to move the camera or the part.
            best = segmentation.selected
            if best is not None:
                panels.append(
                    tint_mask(apply_gain(frame.color, gain), best.mask, tint=(255, 70, 70))
                )
                if best.roi_offset_m > 0.0:
                    # Right shape, wrong place. Distinguished from a size
                    # failure because the fix is different: move the part, the
                    # camera or the ROI, not the tolerance.
                    hud.append((
                        f"OUTSIDE STATION  {best.roi_offset_m * 1000:.0f} mm out, "
                        f"at {np.round(best.center_m * 1000).astype(int)} mm", _RED,
                    ))
                else:
                    hud.append((
                        f"NO MATCH  closest {np.round(best.extents_m * 1000).astype(int)} mm "
                        f"({best.size_error:.0%} off)", _RED,
                    ))
            else:
                panels.append(apply_gain(frame.color, gain))
                hud.append((f"NO OBJECT  {segmentation.reason[:52]}", _RED))
            hud.append((f"candidates {len(segmentation.candidates)}", _DIM))

        # Aiming the camera against an invisible acceptance volume is guesswork,
        # so put it on the panel that shows what was selected.
        if roi is not None:
            panels[-1] = draw_station_roi(panels[-1], roi, frame.intrinsics)
            hud.append((
                f"station ROI +-{np.round(roi.half_extents_m * 1000).astype(int)} mm "
                f"@ {np.round(roi.center_m * 1000).astype(int)} mm", _DIM,
            ))

    if show_inspect:
        result = analyzer.inspection(frame.part_id).infer(frame)
        panels.append(
            anomaly_view(
                result.anomaly_map
                if result.anomaly_map is not None
                else np.zeros(frame.depth.shape, dtype=np.float32),
                segmentation.mask,
                result.defect_mask,
            )
        )
        verdict = "OK" if result.is_good else "NG"
        hud.append((
            f"inspect {verdict}  score {result.anomaly_score:.3f} / thr {result.threshold:.3f}",
            _GREEN if result.is_good else _RED,
        ))

    # Pose is independent of inspection: a part registered from captures has
    # geometry but no trained detector, and that combination must still work.
    if show_pose:
        estimate = analyzer.pose(frame.part_id).run(frame)
        if estimate.valid:
            # Re-project the model at the estimated pose. This is the panel that
            # actually shows whether the pose is right -- fitness cannot.
            overlay, _ = pose_overlay(
                apply_gain(frame.color, gain),
                analyzer.mesh(frame.part_id),
                estimate.pose,
                frame.intrinsics,
            )
            panels.append(overlay)
            # A 6D pose is 3 position + 3 orientation. Showing only the position
            # made the display look like a 3D pose; the orientation is what the
            # robot needs to know how to approach the part.
            x, y, z = estimate.pose.position
            roll, pitch, yaw = quaternion_to_euler_deg(estimate.pose.orientation)
            hud.append((
                f"pos  x{x:+.3f} y{y:+.3f} z{z:+.3f} m", _GREEN,
            ))
            hud.append((
                f"rot  R{roll:+6.1f} P{pitch:+6.1f} Y{yaw:+6.1f} deg", _GREEN,
            ))
            hud.append((
                f"quality  fit {estimate.fitness:.2f}  "
                f"rmse {estimate.rmse_m * 1000:.1f}mm", _DIM,
            ))
        else:
            panels.append(apply_gain(frame.color, gain))
            hud.append((f"pose rejected: {estimate.message[:44]}", _RED))

    canvas = hstack_panels(panels)
    return canvas, hud


# --------------------------------------------------------------------------
# display backends
# --------------------------------------------------------------------------
def run_window(args, cfg, source, analyzer, depth_range) -> int:
    """Stream into an OpenCV window."""
    try:
        import cv2
    except ImportError:
        print(
            "error: opencv-python is required for the window backend.\n"
            "  python3 -m pip install --user opencv-python\n"
            "Or stream to a browser instead:  --backend mjpeg",
            file=sys.stderr,
        )
        return 3

    title = "RoboWorld - color | depth | segmentation | anomaly"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1800, 420)

    show_inspect, show_pose = args.inspect, args.pose
    paused = False
    frame = None
    fps = 0.0
    frames = 0
    started = time.time()

    print("streaming... keys: q quit | space pause | d defect | p part | i inspect | "
          "o pose | s snapshot", flush=True)
    while True:
        tick = time.perf_counter()
        if not paused or frame is None:
            frame = source.read()

        canvas, hud = build_canvas(
            cfg, frame, analyzer, show_inspect, show_pose, depth_range, cv2, args.gain
        )
        header = [
            (f"{frame.part_id}   defect={source.defect or 'none'}   "
             f"{fps:4.1f} fps{'   [PAUSED]' if paused else ''}", _WHITE)
        ]
        draw_hud(cv2, canvas, header + hud)

        # OpenCV windows expect BGR; everything upstream is RGB.
        cv2.imshow(title, canvas[..., ::-1])
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("d"):
            print(f"  defect -> {source.next_defect() or 'none'}", flush=True)
        elif key == ord("p"):
            print(f"  part -> {source.next_part()}", flush=True)
        elif key == ord("i"):
            show_inspect = not show_inspect
        elif key == ord("o"):
            show_pose = not show_pose
        elif key == ord("s"):
            output = paths.data_dir() / f"snapshot_{frames:04d}.png"
            write_png(output, canvas)
            print(f"  saved {output}", flush=True)

        # Closing the window with the title-bar X must end the loop too. With
        # the Qt backend the property query does not return 0 once the window is
        # gone -- it raises "NULL guiReceiver" -- so both outcomes mean closed.
        try:
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        frames += 1
        elapsed = time.perf_counter() - tick
        fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 1e-6)) if frames > 1 else 1.0 / max(
            elapsed, 1e-6
        )
        if args.max_frames and frames >= args.max_frames:
            break
        if args.duration and (time.time() - started) >= args.duration:
            break

    cv2.destroyAllWindows()
    print(f"closed after {frames} frames ({fps:.1f} fps)")
    return 0


def run_mjpeg(args, cfg, source, analyzer, depth_range) -> int:
    """Serve the stream as MJPEG over HTTP -- no GUI libraries needed."""
    import http.server
    import socketserver
    import threading

    try:
        import cv2  # only for JPEG encoding
        encode = lambda rgb: cv2.imencode(".jpg", rgb[..., ::-1])[1].tobytes()  # noqa: E731
    except ImportError:
        print(
            "error: the mjpeg backend needs opencv-python for JPEG encoding.\n"
            "  python3 -m pip install --user opencv-python-headless",
            file=sys.stderr,
        )
        return 3

    state = {"jpeg": None, "stop": False}

    def produce():
        while not state["stop"]:
            frame = source.read()
            canvas, hud = build_canvas(
                cfg, frame, analyzer, args.inspect, args.pose, depth_range, cv2, args.gain
            )
            draw_hud(cv2, canvas, [(f"{frame.part_id}  defect={source.defect or 'none'}",
                                    _WHITE)] + hud)
            state["jpeg"] = encode(canvas)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the console clean
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            try:
                while not state["stop"]:
                    jpeg = state["jpeg"]
                    if jpeg is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(1.0 / max(args.fps, 1.0))
            except (BrokenPipeError, ConnectionResetError):
                pass

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()

    with socketserver.ThreadingTCPServer(("0.0.0.0", args.port), Handler) as server:
        server.daemon_threads = True
        print(f"MJPEG stream on http://localhost:{args.port}   (Ctrl+C to stop)")
        print("  Windows 쪽 브라우저에서 위 주소를 열면 됩니다.")
        try:
            if args.duration:
                threading.Timer(args.duration, server.shutdown).start()
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            state["stop"] = True
    return 0


def run_save(args, cfg, source, analyzer, depth_range) -> int:
    """Headless: write N frames to PNG. Used to smoke-test without a display."""
    try:
        import cv2
    except ImportError:
        cv2 = None

    count = args.max_frames or 3
    output_dir = paths.data_dir() / "live"
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        frame = source.read()
        canvas, hud = build_canvas(
            cfg, frame, analyzer, args.inspect, args.pose, depth_range, cv2, args.gain
        )
        if cv2 is not None:
            draw_hud(cv2, canvas, [(f"{frame.part_id} frame {index}", _WHITE)] + hud)
        path = output_dir / f"live_{index:03d}.png"
        write_png(path, canvas)
        print(f"  {path}  " + "  ".join(text for text, _ in hud))
    return 0


# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--part", default="guide_block")
    parser.add_argument("--source", default="mock",
                        choices=("mock", "realsense", "dataset"))
    parser.add_argument("--depth-range", type=float, nargs=2, metavar=("NEAR", "FAR"),
                        default=None,
                        help="depth colour window in meters (default: per source)")
    parser.add_argument("--split", default="test", help="dataset source: train | test")
    parser.add_argument("--backend", default="window",
                        choices=("window", "mjpeg", "save"))
    parser.add_argument("--inspect", action="store_true",
                        help="add segmentation + anomaly panels")
    parser.add_argument("--pose", action="store_true",
                        help="also estimate 6D pose (slower)")
    parser.add_argument("--defect", default="none",
                        choices=("none", *DEFECT_KINDS),
                        help="mock source: defect injected into every frame")
    parser.add_argument("--gain", type=float, default=None,
                        help="display-only brightness for the colour panel "
                             "(default: 2.2 for mock, 1.0 for a real camera)")
    parser.add_argument("--fps", type=float, default=10.0, help="mjpeg target rate")
    parser.add_argument("--port", type=int, default=8080, help="mjpeg port")
    parser.add_argument("--fit-frames", type=int, default=25)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds, 0 = unlimited")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config()
    if args.part not in cfg.get("parts"):
        print(f"error: unknown part '{args.part}'; known: {sorted(cfg.get('parts'))}",
              file=sys.stderr)
        return 2

    camera = cfg.section("camera")
    station_height = float(cfg.get("station.camera_height_m", 0.60))

    if args.source == "mock":
        source = MockSource(cfg, args.part, seed=args.seed)
        source.defect = None if args.defect == "none" else args.defect
        # Tight window around the belt (0.09 m spans a 55 mm block with margin)
        # so the part renders warm against a cool belt instead of both landing
        # mid-ramp. Fixed, not auto-scaled: auto-scaling makes the belt shimmer
        # whenever the part moves, which reads as sensor drift.
        default_range = (station_height - 0.08, station_height + 0.01)
    elif args.source == "realsense":
        try:
            source = RealSenseSource(
                cfg, args.part,
                int(camera.get("width", 640)), int(camera.get("height", 480)),
                int(camera.get("fps", 30)),
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 5
        # A real scene spans far more than the station window.
        default_range = (0.3, 3.0)
        if args.inspect:
            print(
                "note: --inspect uses a model fitted on mock station frames "
                "(camera 0.6 m above a belt). Pointed at anything else the "
                "segmentation and score are meaningless.",
                file=sys.stderr,
            )
    else:
        try:
            source = DatasetSource(cfg, args.part, args.split)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 4
        default_range = (station_height - 0.08, station_height + 0.01)

    depth_range = tuple(args.depth_range) if args.depth_range else default_range
    if args.gain is None:
        # The mock renderer is deliberately dim; a real camera runs auto-exposure
        # and brightening it again just blows out the highlights.
        args.gain = 1.0 if args.source == "realsense" else 2.2

    analyzer = Analyzer(cfg, args.fit_frames)
    if args.inspect:
        analyzer.inspection(args.part)  # fit up front, not mid-stream

    runner = {"window": run_window, "mjpeg": run_mjpeg, "save": run_save}[args.backend]
    try:
        return runner(args, cfg, source, analyzer, depth_range)
    finally:
        # A RealSense pipeline left running keeps the USB device claimed, and
        # the next launch then fails with a confusing "device busy".
        source.close()


if __name__ == "__main__":
    raise SystemExit(main())
