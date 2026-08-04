"""Surface defect inspection node (Inspection agent, claude.md section 3.2).

Serves ``roboworld_interfaces/InspectPart`` and mirrors every result on
``pipeline.inspection_topic`` for diagnostics.

It never calls the pose node. Section 2 requires the two to stay decoupled; the
OK/NG branch belongs to the pipeline node.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from roboworld_core.inspection import build_backend, model_path_for
from roboworld_core.types import Frame, InspectionResult
from roboworld_interfaces.msg import InspectionResult as InspectionResultMsg
from roboworld_interfaces.srv import InspectPart
from roboworld_ros_utils import (
    config_from_node,
    declare_override,
    empty_image,
    make_header,
    messages_to_frame,
    node_kwargs,
    numpy_to_image,
    qos_from_config,
)


class InspectionNode(Node):
    """Wraps :mod:`roboworld_core.inspection` behind the InspectPart service."""

    def __init__(self) -> None:
        super().__init__("inspection_node", **node_kwargs())
        self.cfg = config_from_node(self)

        self.part_id = str(
            declare_override(self, "part_id", str(self.cfg.get("default_part_id", "guide_block")))
        )
        default_backend = str(self.cfg.get("inspection.backend", "statistical"))
        backend_name = str(declare_override(self, "backend", default_backend))

        self.depth_units_m = float(self.cfg.get("camera.depth_units_m", 0.001))
        self._backends: dict[str, object] = {}
        self._backend_name = backend_name
        self._load_backend(self.part_id)

        callback_group = ReentrantCallbackGroup()
        self.service = self.create_service(
            InspectPart,
            str(self.cfg.get("pipeline.inspect_service", "/roboworld/inspect_part")),
            self.on_inspect,
            callback_group=callback_group,
        )
        self.publisher = self.create_publisher(
            InspectionResultMsg,
            str(self.cfg.get("pipeline.inspection_topic", "/roboworld/inspection_result")),
            qos_from_config(self.cfg.section("pipeline.output_qos")),
        )

        self.get_logger().info(
            f"inspection_node ready: backend='{backend_name}' part='{self.part_id}' "
            f"threshold={self._backend(self.part_id).settings.threshold:.3f}"
        )

    # -- backend management ---------------------------------------------
    def _load_backend(self, part_id: str):
        """Build (and cache) the backend for ``part_id``.

        One backend per part: EfficientAD checkpoints and the statistical model
        are both fitted per part type.
        """
        backend = build_backend(self.cfg, part_id, backend=self._backend_name)
        if not backend.is_fitted:
            model_path = model_path_for(self.cfg, self._backend_name, part_id)
            self.get_logger().warning(
                f"inspection backend '{self._backend_name}' for part '{part_id}' is not "
                f"fitted (expected model at {model_path}). Train it with "
                f"`python3 tools/train_inspection.py --part {part_id}`; until then every "
                "inspection request will fail."
            )
        self._backends[part_id] = backend
        return backend

    def _backend(self, part_id: str):
        return self._backends.get(part_id) or self._load_backend(part_id)

    # -- service ---------------------------------------------------------
    def on_inspect(self, request: InspectPart.Request, response: InspectPart.Response):
        part_id = request.part_id or self.part_id
        try:
            frame: Frame = messages_to_frame(
                request.color,
                request.depth,
                request.camera_info,
                part_id,
                int(request.sequence),
                self.depth_units_m,
            )
            result = self._backend(part_id).infer(frame)
        except Exception as exc:
            self.get_logger().error(f"inspection failed (seq {request.sequence}): {exc}")
            response.success = False
            response.message = f"{type(exc).__name__}: {exc}"
            return response

        response.result = self.to_msg(result)
        response.success = True
        response.message = ""
        self.publisher.publish(response.result)

        self.get_logger().info(
            f"seq={result.sequence} part={part_id} "
            f"{'OK' if result.is_good else 'NG'} score={result.anomaly_score:.3f} "
            f"thr={result.threshold:.3f} {result.inference_time_ms:.1f}ms"
        )
        return response

    # -- conversion ------------------------------------------------------
    def to_msg(self, result: InspectionResult) -> InspectionResultMsg:
        header = make_header(result.stamp, result.frame_id)
        msg = InspectionResultMsg()
        msg.header = header
        msg.part_id = result.part_id
        msg.sequence = int(result.sequence)
        msg.is_good = bool(result.is_good)
        msg.anomaly_score = float(result.anomaly_score)
        msg.threshold = float(result.threshold)
        msg.inference_time_ms = float(result.inference_time_ms)
        msg.backend = result.backend

        if result.anomaly_map is not None:
            msg.anomaly_map = numpy_to_image(
                np.asarray(result.anomaly_map, dtype=np.float32), "32FC1", header
            )
        else:
            msg.anomaly_map = empty_image(header)

        if result.defect_mask is not None:
            msg.defect_mask = numpy_to_image(
                np.asarray(result.defect_mask, dtype=np.uint8), "mono8", header
            )
        else:
            msg.defect_mask = empty_image(header)
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InspectionNode()
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
