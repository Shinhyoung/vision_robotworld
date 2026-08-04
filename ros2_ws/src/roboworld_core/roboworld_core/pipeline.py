"""The index-cycle state machine, free of ROS.

claude.md section 0 fixes the flow: **trigger -> capture -> inspect -> branch ->
(good only) pose -> publish**. Section 2 additionally forbids Inspection from
calling Pose directly; the branch belongs here, in the orchestrator, and the ROS
layer turns each step into a service call.

Keeping the state machine ROS-free means the branch logic, the tact-time budget
and the NG-skips-pose rule are all unit-testable without a middleware, and the
ROS node in ``roboworld_pipeline`` stays a thin adapter.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from .geometry import Pose
from .inspection.base import InspectionBackend
from .pose.base import PoseBackend
from .types import (
    PIPELINE_VERSION,
    Frame,
    InspectionResult,
    PartResult,
    PartStatus,
    PoseEstimate,
)

LOGGER = logging.getLogger("roboworld.pipeline")


class Stage(str, Enum):
    """Cycle stages, mirroring ``InspectAndLocate.action`` feedback."""

    IDLE = "idle"
    CAPTURE = "capture"
    INSPECT = "inspect"
    POSE = "pose"
    PUBLISH = "publish"


@dataclass
class TactBudget:
    """Per-stage time budget in milliseconds (see pipeline.yaml)."""

    capture: float = 120.0
    inspect: float = 400.0
    pose: float = 900.0
    total_warn: float = 1200.0
    total_limit: float = 5000.0

    @classmethod
    def from_config(cls, cfg) -> TactBudget:
        section = cfg.section("pipeline.tact_budget_ms")
        return cls(
            capture=float(section.get("capture", 120.0)),
            inspect=float(section.get("inspect", 400.0)),
            pose=float(section.get("pose", 900.0)),
            total_warn=float(section.get("total_warn", 1200.0)),
            total_limit=float(section.get("total_limit", 5000.0)),
        )


@dataclass
class CycleReport:
    """Everything one index cycle produced. The ROS node publishes from this."""

    result: PartResult
    inspection: InspectionResult | None = None
    pose: PoseEstimate | None = None
    stage_times_ms: dict[str, float] = field(default_factory=dict)
    budget_exceeded: list[str] = field(default_factory=list)

    @property
    def pose_skipped(self) -> bool:
        return self.pose is None


class Pipeline:
    """Runs one index cycle: capture -> inspect -> (branch) -> pose -> publish."""

    def __init__(
        self,
        inspection_backend: InspectionBackend,
        pose_backend: PoseBackend,
        capture: Callable[[str, int], Frame],
        skip_pose_when_ng: bool = True,
        budget: TactBudget | None = None,
        pipeline_version: str = PIPELINE_VERSION,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.inspection_backend = inspection_backend
        self.pose_backend = pose_backend
        self.capture = capture
        self.skip_pose_when_ng = bool(skip_pose_when_ng)
        self.budget = budget or TactBudget()
        self.pipeline_version = pipeline_version
        self._clock = clock
        self._sequence = 0
        self.stage = Stage.IDLE

    @property
    def sequence(self) -> int:
        """Sequence number of the last started cycle."""
        return self._sequence

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    # -- main entry point ------------------------------------------------
    def run_cycle(self, part_id: str, sequence: int | None = None) -> CycleReport:
        """Execute one full cycle and return everything it produced.

        Never raises: a backend blowing up becomes ``STATUS_ERROR`` so the robot
        department always receives exactly one message per trigger.
        """
        sequence = self.next_sequence() if sequence is None else sequence
        started = self._clock()
        stage_times: dict[str, float] = {}
        exceeded: list[str] = []

        def elapsed_ms() -> float:
            return (self._clock() - started) * 1000.0

        # --- capture ----------------------------------------------------
        self.stage = Stage.CAPTURE
        mark = self._clock()
        try:
            frame = self.capture(part_id, sequence)
        except Exception as exc:
            LOGGER.exception("capture failed for sequence %d", sequence)
            return self._error_report(
                sequence, part_id, f"capture failed: {type(exc).__name__}: {exc}",
                elapsed_ms(), stage_times,
            )
        stage_times["capture"] = (self._clock() - mark) * 1000.0
        self._check_budget("capture", stage_times["capture"], self.budget.capture, exceeded)

        # --- inspect ----------------------------------------------------
        self.stage = Stage.INSPECT
        mark = self._clock()
        try:
            inspection = self.inspection_backend.infer(frame)
        except Exception as exc:
            LOGGER.exception("inspection failed for sequence %d", sequence)
            return self._error_report(
                sequence, part_id, f"inspection failed: {type(exc).__name__}: {exc}",
                elapsed_ms(), stage_times, frame=frame,
            )
        stage_times["inspect"] = (self._clock() - mark) * 1000.0
        self._check_budget("inspect", stage_times["inspect"], self.budget.inspect, exceeded)

        # --- branch: NG reports the result and skips pose entirely -------
        if not inspection.is_good and self.skip_pose_when_ng:
            self.stage = Stage.PUBLISH
            total = elapsed_ms()
            self._check_total(total, exceeded)
            result = PartResult(
                sequence=sequence,
                part_id=part_id,
                stamp=frame.stamp,
                frame_id=frame.intrinsics.frame_id,
                status=PartStatus.NG,
                is_good=False,
                anomaly_score=inspection.anomaly_score,
                anomaly_threshold=inspection.threshold,
                pose_valid=False,
                pose=Pose.identity(self.pose_backend.settings.output_frame_id),
                pose_fitness=0.0,
                tact_time_ms=total,
                pipeline_version=self.pipeline_version,
                message=(
                    f"defect detected (score {inspection.anomaly_score:.3f} > "
                    f"threshold {inspection.threshold:.3f}); pose skipped"
                ),
            )
            self.stage = Stage.IDLE
            return CycleReport(result, inspection, None, stage_times, exceeded)

        # --- pose (good parts only) -------------------------------------
        self.stage = Stage.POSE
        mark = self._clock()
        pose_estimate = self.pose_backend.run(frame)  # PoseBackend.run never raises
        stage_times["pose"] = (self._clock() - mark) * 1000.0
        self._check_budget("pose", stage_times["pose"], self.budget.pose, exceeded)

        # --- publish -----------------------------------------------------
        self.stage = Stage.PUBLISH
        total = elapsed_ms()
        self._check_total(total, exceeded)

        if total > self.budget.total_limit:
            status = PartStatus.ERROR
            message = (
                f"tact time {total:.0f} ms exceeded hard limit "
                f"{self.budget.total_limit:.0f} ms"
            )
        elif pose_estimate.valid:
            status = PartStatus.OK
            message = pose_estimate.message
        else:
            status = PartStatus.NO_POSE
            message = f"pose estimation rejected: {pose_estimate.message}"

        result = PartResult(
            sequence=sequence,
            part_id=part_id,
            stamp=frame.stamp,
            frame_id=frame.intrinsics.frame_id,
            status=status,
            is_good=bool(inspection.is_good),
            anomaly_score=inspection.anomaly_score,
            anomaly_threshold=inspection.threshold,
            pose_valid=bool(pose_estimate.valid and status == PartStatus.OK),
            pose=pose_estimate.pose,
            pose_fitness=pose_estimate.fitness,
            tact_time_ms=total,
            pipeline_version=self.pipeline_version,
            message=message,
        )
        self.stage = Stage.IDLE
        return CycleReport(result, inspection, pose_estimate, stage_times, exceeded)

    # -- helpers ---------------------------------------------------------
    def _check_budget(
        self, stage: str, actual_ms: float, budget_ms: float, exceeded: list[str]
    ) -> None:
        if actual_ms > budget_ms:
            exceeded.append(stage)
            LOGGER.warning(
                "stage '%s' took %.1f ms, budget %.1f ms", stage, actual_ms, budget_ms
            )

    def _check_total(self, total_ms: float, exceeded: list[str]) -> None:
        if total_ms > self.budget.total_warn:
            exceeded.append("total")
            LOGGER.warning(
                "tact time %.1f ms exceeded warn budget %.1f ms",
                total_ms, self.budget.total_warn,
            )

    def _error_report(
        self,
        sequence: int,
        part_id: str,
        message: str,
        total_ms: float,
        stage_times: dict[str, float],
        frame: Frame | None = None,
    ) -> CycleReport:
        """Build the STATUS_ERROR message emitted when a stage throws."""
        self.stage = Stage.IDLE
        frame_id = (
            frame.intrinsics.frame_id if frame is not None
            else self.pose_backend.settings.output_frame_id
        )
        result = PartResult(
            sequence=sequence,
            part_id=part_id,
            stamp=frame.stamp if frame is not None else 0.0,
            frame_id=frame_id,
            status=PartStatus.ERROR,
            is_good=False,
            anomaly_score=0.0,
            anomaly_threshold=self.inspection_backend.settings.threshold,
            pose_valid=False,
            pose=Pose.identity(frame_id),
            pose_fitness=0.0,
            tact_time_ms=total_ms,
            pipeline_version=self.pipeline_version,
            message=message,
        )
        return CycleReport(result, None, None, stage_times, ["error"])


def build_pipeline(cfg, inspection_backend, pose_backend, capture) -> Pipeline:
    """Assemble a :class:`Pipeline` from config."""
    return Pipeline(
        inspection_backend=inspection_backend,
        pose_backend=pose_backend,
        capture=capture,
        skip_pose_when_ng=bool(cfg.get("pipeline.skip_pose_when_ng", True)),
        budget=TactBudget.from_config(cfg),
        pipeline_version=str(cfg.get("pipeline.version", PIPELINE_VERSION)),
    )
