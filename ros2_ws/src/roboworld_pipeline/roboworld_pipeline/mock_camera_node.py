"""Mock RealSense D455 publisher.

Publishes CAD-rendered RGB-D on exactly the topics ``realsense-ros`` would use,
so the pipeline can run with ``camera.source: rosbag`` and exercise the real
subscription, synchronisation and conversion path without a camera attached
(claude.md section 2: no hardware dependency during parallel development).

Use this when you want to test the ROS plumbing. Use ``camera.source: mock``
when you want the fastest possible dry-run with no ROS traffic at all.
"""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from roboworld_core.mock_data import DEFECT_KINDS, MockStation, parts_from_config
from roboworld_core.types import CameraIntrinsics
from roboworld_ros_utils import (
    config_from_node,
    declare_override,
    frame_to_messages,
    node_kwargs,
    sensor_qos,
)


class MockCameraNode(Node):
    """Renders and publishes synthetic D455 frames at a fixed rate."""

    def __init__(self) -> None:
        super().__init__("mock_camera_node", **node_kwargs())
        self.cfg = config_from_node(self)

        self.part_id = str(
            declare_override(self, "part_id", str(self.cfg.get("default_part_id", "guide_block")))
        )
        # Inject a defect every Nth published frame (0 = never). Lets the NG
        # branch be demonstrated end to end with no defective hardware.
        self.defect_every = int(declare_override(self, "defect_every", 0))
        self.seed = int(declare_override(self, "seed", 0))
        fps = max(0.1, float(declare_override(self, "fps", 5.0)))

        intrinsics = CameraIntrinsics.from_config(self.cfg.section("camera"))
        self.station = MockStation(parts_from_config(self.cfg), intrinsics)

        camera = self.cfg.section("camera")
        qos = sensor_qos()
        self.color_publisher = self.create_publisher(
            Image, str(camera.get("color_topic")), qos
        )
        self.depth_publisher = self.create_publisher(
            Image, str(camera.get("depth_topic")), qos
        )
        self.info_publisher = self.create_publisher(
            CameraInfo, str(camera.get("camera_info_topic")), qos
        )

        self._counter = 0
        self.timer = self.create_timer(1.0 / fps, self._publish)
        self.get_logger().info(
            f"mock_camera_node publishing '{self.part_id}' at {fps:.1f} fps on "
            f"{camera.get('color_topic')} (defect_every={self.defect_every})"
        )

    def _publish(self) -> None:
        self._counter += 1
        defect = None
        if self.defect_every > 0 and self._counter % self.defect_every == 0:
            defect = DEFECT_KINDS[(self._counter // self.defect_every - 1) % len(DEFECT_KINDS)]

        frame = self.station.sample_frame(
            self.part_id,
            defect=defect,
            seed=self.seed + self._counter * 17,
            sequence=self._counter,
            stamp=time.time(),
        )
        color, depth, info = frame_to_messages(frame)
        self.color_publisher.publish(color)
        self.depth_publisher.publish(depth)
        self.info_publisher.publish(info)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MockCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
