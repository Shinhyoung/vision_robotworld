"""Where the pipeline gets an RGB-D frame from.

Three interchangeable sources, selected by ``camera.source`` (ROS2 agent,
claude.md section 3.4):

``realsense``
    Subscribe to ``realsense-ros``. Requires the D455 to be attached to WSL via
    ``usbipd-win`` -- see docs/setup_wsl.md.
``rosbag``
    Identical subscription path; ``ros2 bag play`` supplies the topics. This is
    the recommended way to develop without hardware.
``mock``
    Frames rendered in-process from the part CAD. No ROS traffic at all, which
    is what makes the E2E dry-run possible on a machine with neither camera nor
    GPU.

All three yield the same :class:`roboworld_core.types.Frame`, so nothing
downstream knows or cares which one is active.
"""

from __future__ import annotations

import abc
import threading
import time

import numpy as np

from roboworld_core.mock_data import MockStation, parts_from_config
from roboworld_core.types import CameraIntrinsics, Frame


class FrameUnavailable(RuntimeError):
    """No frame is available within the configured timeout."""


class FrameSource(abc.ABC):
    """Supplies one RGB-D frame per trigger."""

    @abc.abstractmethod
    def get(self, part_id: str, sequence: int) -> Frame:
        """Return the frame to inspect for this cycle."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""


class MockFrameSource(FrameSource):
    """Renders frames from the CAD meshes, with scripted defect injection."""

    def __init__(
        self,
        cfg,
        defect_every: int = 0,
        defect_kinds: tuple[str, ...] = ("scratch", "dent", "stain", "chip"),
        seed: int = 0,
    ) -> None:
        """Args:
        defect_every: inject a defect on every Nth cycle (0 disables). Used by
            the dry-run to exercise the NG branch deterministically.
        """
        intrinsics = CameraIntrinsics.from_config(cfg.section("camera"))
        self.station = MockStation(parts_from_config(cfg), intrinsics)
        self.defect_every = int(defect_every)
        self.defect_kinds = tuple(defect_kinds)
        self.seed = int(seed)

    def get(self, part_id: str, sequence: int) -> Frame:
        defect = None
        if self.defect_every > 0 and sequence % self.defect_every == 0:
            defect = self.defect_kinds[(sequence // self.defect_every - 1) % len(self.defect_kinds)]
        return self.station.sample_frame(
            part_id,
            defect=defect,
            seed=self.seed + sequence * 17,
            sequence=sequence,
            stamp=time.time(),
        )


class TopicFrameSource(FrameSource):
    """Latest synchronised color+depth+info triple from ROS topics.

    The station is stop-and-go, so the pipeline does not need a stream: it needs
    *the* frame captured while the part was standing still. This source keeps
    only the most recent synchronised triple and rejects it if it is older than
    ``frame_timeout_s``, which is what stops a stale frame from being inspected
    as if it were the current part.
    """

    def __init__(self, node, cfg) -> None:
        import message_filters
        from sensor_msgs.msg import CameraInfo, Image

        from roboworld_ros_utils import sensor_qos

        self.node = node
        camera = cfg.section("camera")
        self.depth_units_m = float(camera.get("depth_units_m", 0.001))
        self.timeout_s = float(camera.get("frame_timeout_s", 1.0))
        self.sync_tolerance_s = float(camera.get("sync_tolerance_s", 0.030))

        self._lock = threading.Lock()
        self._latest: tuple[float, object, object, object] | None = None

        qos = sensor_qos()
        color_sub = message_filters.Subscriber(
            node, Image, str(camera.get("color_topic")), qos_profile=qos
        )
        depth_sub = message_filters.Subscriber(
            node, Image, str(camera.get("depth_topic")), qos_profile=qos
        )
        info_sub = message_filters.Subscriber(
            node, CameraInfo, str(camera.get("camera_info_topic")), qos_profile=qos
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub], queue_size=5, slop=self.sync_tolerance_s
        )
        self._sync.registerCallback(self._on_frame)

    def _on_frame(self, color, depth, info) -> None:
        with self._lock:
            self._latest = (time.time(), color, depth, info)

    def get(self, part_id: str, sequence: int) -> Frame:
        from roboworld_ros_utils import messages_to_frame

        with self._lock:
            latest = self._latest

        if latest is None:
            raise FrameUnavailable(
                "no synchronised color/depth/camera_info received yet. "
                "Check that realsense-ros (or `ros2 bag play`) is running and that "
                "camera.align_depth is true."
            )
        received_at, color, depth, info = latest
        age = time.time() - received_at
        if age > self.timeout_s:
            raise FrameUnavailable(
                f"newest frame is {age:.2f}s old, older than "
                f"camera.frame_timeout_s ({self.timeout_s:.2f}s)"
            )

        frame = messages_to_frame(color, depth, info, part_id, sequence, self.depth_units_m)
        if not np.any(frame.depth > 0.0):
            raise FrameUnavailable(
                "depth image is entirely zero -- is aligned_depth_to_color being published?"
            )
        return frame


def build_frame_source(cfg, node=None, **kwargs) -> FrameSource:
    """Instantiate the source named by ``camera.source``."""
    source = str(cfg.get("camera.source", "mock")).lower()
    if source == "mock":
        return MockFrameSource(cfg, **kwargs)
    if source in ("realsense", "rosbag"):
        if node is None:
            raise ValueError(f"camera.source '{source}' requires a ROS node")
        return TopicFrameSource(node, cfg)
    raise ValueError(
        f"unknown camera.source '{source}' (expected 'realsense', 'rosbag' or 'mock')"
    )
