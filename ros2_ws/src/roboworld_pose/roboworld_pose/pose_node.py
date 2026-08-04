"""6D pose estimation node (Pose agent, claude.md section 3.3).

Serves ``roboworld_interfaces/EstimatePose`` and mirrors results on
``pipeline.pose_topic``.

The pipeline only calls this service for parts that passed inspection, so this
node never has to know about OK/NG. It is responsible for the *frame* the pose
is published in: the backend always works in the camera optical frame, and this
node optionally transforms into the agreed output frame via TF2 (ICD section 4).
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from roboworld_core.pose import LICENSE_NOTICE, build_backend
from roboworld_core.types import Frame, PoseEstimate
from roboworld_interfaces.msg import PoseResult as PoseResultMsg
from roboworld_interfaces.srv import EstimatePose
from roboworld_ros_utils import (
    config_from_node,
    declare_override,
    make_header,
    messages_to_frame,
    node_kwargs,
    pose_to_stamped,
    qos_from_config,
    transform_to_matrix,
)


class PoseNode(Node):
    """Wraps :mod:`roboworld_core.pose` behind the EstimatePose service."""

    def __init__(self) -> None:
        super().__init__("pose_node", **node_kwargs())
        self.cfg = config_from_node(self)

        self.part_id = str(
            declare_override(self, "part_id", str(self.cfg.get("default_part_id", "guide_block")))
        )
        self._backend_name = str(
            declare_override(self, "backend", str(self.cfg.get("pose.backend", "icp")))
        )

        self.depth_units_m = float(self.cfg.get("camera.depth_units_m", 0.001))
        self.output_frame_id = str(
            self.cfg.get("pose.output_frame_id", "camera_color_optical_frame")
        )
        self.transform_to_output = bool(self.cfg.get("pose.transform_to_output_frame", False))

        self._backends: dict[str, object] = {}
        self._backend(self.part_id)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        callback_group = ReentrantCallbackGroup()
        self.service = self.create_service(
            EstimatePose,
            str(self.cfg.get("pipeline.pose_service", "/roboworld/estimate_pose")),
            self.on_estimate,
            callback_group=callback_group,
        )
        self.publisher = self.create_publisher(
            PoseResultMsg,
            str(self.cfg.get("pipeline.pose_topic", "/roboworld/pose_result")),
            qos_from_config(self.cfg.section("pipeline.output_qos")),
        )

        if self._backend_name == "foundationpose":
            self.get_logger().warning(LICENSE_NOTICE)
        self.get_logger().info(
            f"pose_node ready: backend='{self._backend_name}' part='{self.part_id}' "
            f"output_frame='{self.output_frame_id}' "
            f"transform_to_output={self.transform_to_output}"
        )

    # -- backend management ---------------------------------------------
    def _backend(self, part_id: str):
        if part_id not in self._backends:
            kwargs = {}
            if self._backend_name == "foundationpose":
                kwargs["ros_bridge"] = self._foundationpose_bridge
            # A part registered from camera captures has no mesh. Degrade to
            # STATUS_NO_POSE for that part instead of failing the node, which
            # would take inspection down with it.
            self._backends[part_id] = build_backend(
                self.cfg, part_id, backend=self._backend_name,
                allow_missing_cad=True, **kwargs
            )
            if self._backends[part_id].name == "no_cad":
                self.get_logger().warning(
                    f"part '{part_id}' has no CAD mesh: inspection will run, but "
                    "every cycle for this part reports STATUS_NO_POSE"
                )
        return self._backends[part_id]

    def _foundationpose_bridge(self, frame: Frame):
        """Hand a frame to the Isaac ROS FoundationPose graph.

        Wiring this up is ticket POSE-4 (docs/agent_tickets.md): it forwards the
        RGB-D pair plus the object mask to ``isaac_ros_foundationpose`` and waits
        for ``/pose_estimation/output``. Until that graph is deployed the node
        must fail loudly rather than silently return a wrong pose.
        """
        raise NotImplementedError(
            "Isaac ROS FoundationPose bridge not deployed yet (ticket POSE-4). "
            "Run with pose.backend:=icp until the Isaac ROS graph is available."
        )

    # -- service ---------------------------------------------------------
    def on_estimate(self, request: EstimatePose.Request, response: EstimatePose.Response):
        part_id = request.part_id or self.part_id
        try:
            frame = messages_to_frame(
                request.color,
                request.depth,
                request.camera_info,
                part_id,
                int(request.sequence),
                self.depth_units_m,
            )
            estimate = self._backend(part_id).run(frame)
        except Exception as exc:
            self.get_logger().error(f"pose estimation failed (seq {request.sequence}): {exc}")
            response.success = False
            response.message = f"{type(exc).__name__}: {exc}"
            return response

        estimate = self._maybe_transform(estimate)
        response.result = self.to_msg(estimate)
        response.success = True
        response.message = ""
        self.publisher.publish(response.result)

        position = estimate.pose.position
        self.get_logger().info(
            f"seq={estimate.sequence} part={part_id} valid={estimate.valid} "
            f"pos=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}) m "
            f"frame={estimate.pose.frame_id} fitness={estimate.fitness:.3f} "
            f"rmse={estimate.rmse_m * 1000:.2f}mm {estimate.inference_time_ms:.1f}ms"
        )
        return response

    # -- frames ----------------------------------------------------------
    def _maybe_transform(self, estimate: PoseEstimate) -> PoseEstimate:
        """Transform into the configured output frame when TF allows it.

        A missing transform is NOT an error: the pose is still valid, just
        expressed in the camera frame. That fact is recorded in ``message`` and
        in ``pose.frame_id``, which the ICD declares authoritative -- so a
        subscriber can always tell which frame it received.
        """
        if not self.transform_to_output or not estimate.valid:
            return estimate
        source = estimate.pose.frame_id
        if source == self.output_frame_id:
            return estimate

        try:
            transform = self.tf_buffer.lookup_transform(
                self.output_frame_id, source, rclpy.time.Time()
            )
        except Exception as exc:
            note = (
                f"TF {self.output_frame_id} <- {source} unavailable ({exc}); "
                f"pose published in {source}"
            )
            self.get_logger().warning(note, throttle_duration_sec=10.0)
            estimate.message = "; ".join(filter(None, (estimate.message, note)))
            return estimate

        matrix = transform_to_matrix(transform)
        estimate.pose = estimate.pose.transformed_by(matrix, self.output_frame_id)
        return estimate

    # -- conversion ------------------------------------------------------
    def to_msg(self, estimate: PoseEstimate) -> PoseResultMsg:
        msg = PoseResultMsg()
        msg.header = make_header(estimate.stamp, estimate.pose.frame_id)
        msg.part_id = estimate.part_id
        msg.sequence = int(estimate.sequence)
        msg.valid = bool(estimate.valid)
        msg.pose = pose_to_stamped(estimate.pose, estimate.stamp)
        msg.fitness = float(estimate.fitness)
        msg.rmse_m = float(estimate.rmse_m)
        msg.covariance = [float(v) for v in np.asarray(estimate.covariance).reshape(36)]
        msg.inference_time_ms = float(estimate.inference_time_ms)
        msg.backend = estimate.backend
        msg.message = estimate.message
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseNode()
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
