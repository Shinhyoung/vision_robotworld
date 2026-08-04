"""Core backend interfaces implemented over ROS services.

The pipeline state machine in :mod:`roboworld_core.pipeline` is deliberately
middleware-agnostic: it calls ``inspection_backend.infer(frame)`` and
``pose_backend.run(frame)``. These adapters satisfy those interfaces by calling
the ``InspectPart`` / ``EstimatePose`` services instead of computing anything
locally.

That is what lets the exact same branch logic, tact-time budget and NG-skips-pose
rule be unit-tested with in-process backends and run in production across three
processes, without a second implementation to keep in sync.
"""

from __future__ import annotations

import numpy as np

from roboworld_core.geometry import Pose
from roboworld_core.inspection.base import InspectionBackend, InspectionSettings
from roboworld_core.pose.base import PoseBackend, PoseSettings
from roboworld_core.types import Frame, InspectionResult, PoseEstimate
from roboworld_interfaces.srv import EstimatePose, InspectPart
from roboworld_ros_utils import (
    empty_image,
    frame_to_messages,
    image_to_numpy,
    stamped_to_pose,
)


class ServiceUnavailable(RuntimeError):
    """The remote node did not answer within the configured timeout."""


def _call_sync(client, request, timeout_s: float, name: str):
    """Blocking service call, safe inside a ReentrantCallbackGroup callback.

    The pipeline runs under a MultiThreadedExecutor with reentrant callback
    groups, so blocking here does not deadlock the executor -- other callbacks
    (including the service responses being waited on) continue on other threads.
    """
    if not client.wait_for_service(timeout_sec=timeout_s):
        raise ServiceUnavailable(f"service '{name}' not available after {timeout_s:.1f}s")
    future = client.call_async(request)
    if not _spin_until_done(future, timeout_s):
        future.cancel()
        raise ServiceUnavailable(f"service '{name}' timed out after {timeout_s:.1f}s")
    return future.result()


def _spin_until_done(future, timeout_s: float) -> bool:
    import time

    deadline = time.monotonic() + timeout_s
    while not future.done():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.002)
    return True


class ServiceInspectionBackend(InspectionBackend):
    """Forwards inspection to the ``InspectPart`` service."""

    name = "service"

    def __init__(self, node, service_name: str, timeout_s: float,
                 settings: InspectionSettings) -> None:
        super().__init__(settings)
        self.node = node
        self.service_name = service_name
        self.timeout_s = float(timeout_s)
        self.client = node.create_client(InspectPart, service_name)

    def score_map(self, frame: Frame):  # pragma: no cover - never used
        raise NotImplementedError("ServiceInspectionBackend overrides infer() directly")

    def infer(self, frame: Frame) -> InspectionResult:
        color, depth, camera_info = frame_to_messages(frame)
        request = InspectPart.Request(
            color=color,
            depth=depth,
            camera_info=camera_info,
            part_id=frame.part_id,
            sequence=int(frame.sequence),
        )
        response = _call_sync(self.client, request, self.timeout_s, self.service_name)
        if response is None or not response.success:
            message = getattr(response, "message", "no response")
            raise RuntimeError(f"inspection service failed: {message}")

        result = response.result
        return InspectionResult(
            part_id=result.part_id,
            sequence=int(result.sequence),
            stamp=frame.stamp,
            frame_id=result.header.frame_id,
            is_good=bool(result.is_good),
            anomaly_score=float(result.anomaly_score),
            threshold=float(result.threshold),
            anomaly_map=(
                image_to_numpy(result.anomaly_map) if result.anomaly_map.height else None
            ),
            defect_mask=(
                image_to_numpy(result.defect_mask) if result.defect_mask.height else None
            ),
            inference_time_ms=float(result.inference_time_ms),
            backend=result.backend,
        )


class ServicePoseBackend(PoseBackend):
    """Forwards pose estimation to the ``EstimatePose`` service."""

    name = "service"

    def __init__(self, node, service_name: str, timeout_s: float, settings: PoseSettings) -> None:
        super().__init__(settings)
        self.node = node
        self.service_name = service_name
        self.timeout_s = float(timeout_s)
        self.client = node.create_client(EstimatePose, service_name)

    def estimate(self, frame: Frame):  # pragma: no cover - never used
        raise NotImplementedError("ServicePoseBackend overrides run() directly")

    def run(self, frame: Frame) -> PoseEstimate:
        color, depth, camera_info = frame_to_messages(frame)
        header = color.header
        request = EstimatePose.Request(
            color=color,
            depth=depth,
            camera_info=camera_info,
            roi_mask=empty_image(header),
            part_id=frame.part_id,
            sequence=int(frame.sequence),
        )
        try:
            response = _call_sync(self.client, request, self.timeout_s, self.service_name)
        except ServiceUnavailable as exc:
            # Mirrors PoseBackend.run's contract: never raise, report invalid.
            return PoseEstimate(
                part_id=frame.part_id,
                sequence=frame.sequence,
                stamp=frame.stamp,
                valid=False,
                pose=Pose.identity(frame.intrinsics.frame_id),
                backend=self.name,
                message=str(exc),
            )

        if response is None or not response.success:
            return PoseEstimate(
                part_id=frame.part_id,
                sequence=frame.sequence,
                stamp=frame.stamp,
                valid=False,
                pose=Pose.identity(frame.intrinsics.frame_id),
                backend=self.name,
                message=f"pose service failed: {getattr(response, 'message', 'no response')}",
            )

        result = response.result
        return PoseEstimate(
            part_id=result.part_id,
            sequence=int(result.sequence),
            stamp=frame.stamp,
            valid=bool(result.valid),
            pose=stamped_to_pose(result.pose),
            fitness=float(result.fitness),
            rmse_m=float(result.rmse_m),
            covariance=np.asarray(result.covariance, dtype=np.float64),
            inference_time_ms=float(result.inference_time_ms),
            backend=result.backend,
            message=result.message,
        )
