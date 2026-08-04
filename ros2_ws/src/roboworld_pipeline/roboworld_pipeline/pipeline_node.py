"""Pipeline orchestration node (ROS2 agent, claude.md section 3.4).

Owns the index cycle end to end:

    trigger -> capture -> inspect -> branch -> (good only) pose -> publish

and is the **only** node that publishes the robot department's contract topic
``/roboworld/part_result``. The branch lives here rather than inside the
inspection node so Inspection and Pose stay decoupled (claude.md section 2).

The state machine itself is :class:`roboworld_core.pipeline.Pipeline`; this node
supplies it with a frame source and two service-backed adapters.
"""

from __future__ import annotations

import threading

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool

from roboworld_core.inspection.base import InspectionSettings
from roboworld_core.pipeline import CycleReport, build_pipeline
from roboworld_core.pose.base import PoseSettings
from roboworld_core.types import PartStatus
from roboworld_interfaces.msg import PartResult as PartResultMsg
from roboworld_interfaces.srv import TriggerCapture
from roboworld_ros_utils import (
    config_from_node,
    declare_override,
    make_header,
    node_kwargs,
    pose_to_stamped,
    qos_from_config,
)

from .frame_source import build_frame_source
from .service_backends import ServiceInspectionBackend, ServicePoseBackend


class PipelineNode(Node):
    """Trigger-driven orchestrator and ICD publisher."""

    def __init__(self) -> None:
        super().__init__("pipeline_node", **node_kwargs())
        self.cfg = config_from_node(self)

        self.part_id = str(
            declare_override(self, "part_id", str(self.cfg.get("default_part_id", "guide_block")))
        )
        mock_defect_every = int(declare_override(self, "mock_defect_every", 0))

        timeout_s = float(self.cfg.get("pipeline.service_timeout_s", 5.0))
        self.debounce_s = float(self.cfg.get("pipeline.trigger_debounce_s", 0.30))

        # One cycle at a time: the station is an index station, and a second
        # concurrent cycle would publish results for a part that has moved on.
        self._cycle_lock = threading.Lock()
        self._last_trigger_time = 0.0
        self._last_trigger_value = False

        service_group = ReentrantCallbackGroup()
        # `defect_every` only exists on MockFrameSource; passing it to the topic
        # source would be a TypeError.
        source_kwargs = (
            {"defect_every": mock_defect_every}
            if str(self.cfg.get("camera.source", "mock")).lower() == "mock"
            else {}
        )
        self.frame_source = build_frame_source(self.cfg, node=self, **source_kwargs)

        inspection_backend = ServiceInspectionBackend(
            self,
            str(self.cfg.get("pipeline.inspect_service", "/roboworld/inspect_part")),
            timeout_s,
            InspectionSettings.from_config(self.cfg),
        )
        pose_backend = ServicePoseBackend(
            self,
            str(self.cfg.get("pipeline.pose_service", "/roboworld/estimate_pose")),
            timeout_s,
            PoseSettings.from_config(self.cfg),
        )
        self.pipeline = build_pipeline(
            self.cfg, inspection_backend, pose_backend, self._capture
        )

        # --- ICD output ------------------------------------------------
        self.result_publisher = self.create_publisher(
            PartResultMsg,
            str(self.cfg.get("pipeline.result_topic", "/roboworld/part_result")),
            qos_from_config(self.cfg.section("pipeline.output_qos")),
        )
        self.status_publisher = self.create_publisher(
            DiagnosticArray,
            str(self.cfg.get("pipeline.status_topic", "/roboworld/pipeline_status")),
            10,
        )

        # --- inputs ----------------------------------------------------
        self.trigger_subscription = self.create_subscription(
            Bool,
            str(self.cfg.get("pipeline.trigger_topic", "/roboworld/trigger")),
            self._on_trigger,
            10,
            callback_group=service_group,
        )
        self.trigger_service = self.create_service(
            TriggerCapture,
            str(self.cfg.get("pipeline.trigger_service", "/roboworld/trigger_capture")),
            self._on_trigger_service,
            callback_group=service_group,
        )

        self.get_logger().info(
            f"pipeline_node ready: source='{self.cfg.get('camera.source')}' "
            f"part='{self.part_id}' -> publishing "
            f"'{self.cfg.get('pipeline.result_topic')}' "
            f"(pipeline_version {self.cfg.get('pipeline.version')})"
        )

    # -- capture ---------------------------------------------------------
    def _capture(self, part_id: str, sequence: int):
        return self.frame_source.get(part_id, sequence)

    # -- triggers --------------------------------------------------------
    def _on_trigger(self, msg: Bool) -> None:
        """Rising-edge trigger from the station photo sensor."""
        rising = bool(msg.data) and not self._last_trigger_value
        self._last_trigger_value = bool(msg.data)
        if not rising:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_trigger_time < self.debounce_s:
            self.get_logger().debug("trigger ignored (debounce)")
            return
        self._last_trigger_time = now
        self.run_cycle(self.part_id)

    def _on_trigger_service(
        self, request: TriggerCapture.Request, response: TriggerCapture.Response
    ):
        if self._cycle_lock.locked():
            response.accepted = False
            response.sequence = self.pipeline.sequence
            response.message = "pipeline busy with the previous cycle"
            return response

        sequence = self.pipeline.sequence + 1
        response.accepted = True
        response.sequence = sequence
        response.message = ""
        self.run_cycle(request.part_id or self.part_id, sequence=sequence)
        return response

    # -- cycle -----------------------------------------------------------
    def run_cycle(self, part_id: str, sequence: int | None = None) -> CycleReport | None:
        """Run one index cycle and publish exactly one PartResult."""
        if not self._cycle_lock.acquire(blocking=False):
            self.get_logger().warning("trigger dropped: a cycle is already running")
            return None
        try:
            report = self.pipeline.run_cycle(part_id, sequence=sequence)
        finally:
            self._cycle_lock.release()

        self.result_publisher.publish(self.to_msg(report))
        self._publish_status(report)
        self._log(report)
        return report

    def _log(self, report: CycleReport) -> None:
        result = report.result
        stages = " ".join(f"{k}={v:.0f}ms" for k, v in report.stage_times_ms.items())
        line = (
            f"seq={result.sequence} part={result.part_id} "
            f"status={PartStatus(result.status).name} "
            f"score={result.anomaly_score:.3f} pose_valid={result.pose_valid} "
            f"tact={result.tact_time_ms:.0f}ms [{stages}]"
        )
        if result.status in (PartStatus.ERROR, PartStatus.NO_POSE):
            self.get_logger().error(f"{line} :: {result.message}")
        elif report.budget_exceeded:
            self.get_logger().warning(f"{line} :: over budget {report.budget_exceeded}")
        else:
            self.get_logger().info(line)

    def _publish_status(self, report: CycleReport) -> None:
        result = report.result
        status = DiagnosticStatus(
            name="roboworld/pipeline",
            hardware_id=str(self.cfg.get("camera.source", "mock")),
            message=result.message or PartStatus(result.status).name,
        )
        if result.status == PartStatus.OK:
            status.level = DiagnosticStatus.OK
        elif result.status == PartStatus.NG:
            status.level = DiagnosticStatus.OK  # a detected defect is nominal behaviour
        elif result.status == PartStatus.NO_POSE:
            status.level = DiagnosticStatus.WARN
        else:
            status.level = DiagnosticStatus.ERROR

        status.values = [
            KeyValue(key="sequence", value=str(result.sequence)),
            KeyValue(key="part_id", value=result.part_id),
            KeyValue(key="anomaly_score", value=f"{result.anomaly_score:.4f}"),
            KeyValue(key="pose_fitness", value=f"{result.pose_fitness:.4f}"),
            KeyValue(key="tact_time_ms", value=f"{result.tact_time_ms:.1f}"),
            KeyValue(key="budget_exceeded", value=",".join(report.budget_exceeded)),
        ] + [
            KeyValue(key=f"stage_{name}_ms", value=f"{value:.1f}")
            for name, value in report.stage_times_ms.items()
        ]

        array = DiagnosticArray()
        array.header = make_header(result.stamp, result.frame_id)
        array.status = [status]
        self.status_publisher.publish(array)

    # -- conversion ------------------------------------------------------
    def to_msg(self, report: CycleReport) -> PartResultMsg:
        result = report.result
        msg = PartResultMsg()
        msg.header = make_header(result.stamp, result.frame_id)
        msg.sequence = int(result.sequence)
        msg.part_id = result.part_id
        msg.status = int(result.status)
        msg.is_good = bool(result.is_good)
        msg.anomaly_score = float(result.anomaly_score)
        msg.anomaly_threshold = float(result.anomaly_threshold)
        msg.pose_valid = bool(result.pose_valid)
        msg.pose = pose_to_stamped(result.pose, result.stamp)
        msg.pose_fitness = float(result.pose_fitness)
        msg.tact_time_ms = float(result.tact_time_ms)
        msg.pipeline_version = result.pipeline_version
        msg.message = result.message
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PipelineNode()
    # Blocking service calls happen inside callbacks, so a single-threaded
    # executor would deadlock waiting for its own responses.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.frame_source.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
