"""Index-cycle state machine tests.

The branch rules from claude.md section 0 are verified here with fake backends,
so they hold regardless of which detector or estimator is plugged in.
"""

from __future__ import annotations

import numpy as np
import pytest

from roboworld_core.contract import check_result
from roboworld_core.geometry import Pose
from roboworld_core.inspection.base import InspectionBackend, InspectionSettings
from roboworld_core.pipeline import Pipeline, Stage, TactBudget
from roboworld_core.pose.base import PoseBackend, PoseSettings
from roboworld_core.types import CameraIntrinsics, Frame, InspectionResult, PartStatus


class FakeInspection(InspectionBackend):
    name = "fake"

    def __init__(self, is_good: bool = True, score: float = 0.1, raises: bool = False):
        super().__init__(InspectionSettings(threshold=0.5))
        self.is_good = is_good
        self.score = score
        self.raises = raises
        self.calls = 0

    def score_map(self, frame):  # pragma: no cover - infer is overridden
        raise NotImplementedError

    def infer(self, frame: Frame) -> InspectionResult:
        self.calls += 1
        if self.raises:
            raise RuntimeError("inspection exploded")
        return InspectionResult(
            part_id=frame.part_id,
            sequence=frame.sequence,
            stamp=frame.stamp,
            frame_id=frame.intrinsics.frame_id,
            is_good=self.is_good,
            anomaly_score=self.score,
            threshold=0.5,
            backend=self.name,
        )


class FakePose(PoseBackend):
    name = "fake"

    def __init__(self, valid: bool = True, raises: bool = False):
        super().__init__(PoseSettings(output_frame_id="camera_color_optical_frame"))
        self.valid = valid
        self.raises = raises
        self.calls = 0

    def estimate(self, frame: Frame):
        self.calls += 1
        if self.raises:
            raise RuntimeError("pose exploded")
        fitness = 0.9 if self.valid else 0.0
        return (
            Pose(np.array([0.0, 0.0, 0.573]), np.array([0.0, 0.0, 0.0, 1.0]),
                 frame.intrinsics.frame_id),
            fitness,
            0.001,
            "",
        )


@pytest.fixture
def frame():
    intrinsics = CameraIntrinsics(64, 48, 40.0, 40.0, 32.0, 24.0)
    return Frame(
        color=np.zeros((48, 64, 3), dtype=np.uint8),
        depth=np.full((48, 64), 0.6, dtype=np.float32),
        intrinsics=intrinsics,
        stamp=100.0,
        part_id="guide_block",
    )


def make_pipeline(inspection, pose, frame, **kwargs) -> Pipeline:
    def capture(part_id, sequence):
        frame.part_id = part_id
        frame.sequence = sequence
        return frame

    return Pipeline(inspection, pose, capture, **kwargs)


def test_good_part_runs_pose_and_reports_ok(frame):
    inspection, pose = FakeInspection(is_good=True), FakePose(valid=True)
    report = make_pipeline(inspection, pose, frame).run_cycle("guide_block")

    assert report.result.status == PartStatus.OK
    assert report.result.pose_valid is True
    assert pose.calls == 1
    assert check_result(report.result, strict=False) == []


def test_defective_part_skips_pose(frame):
    """claude.md section 0: NG reports the result only, pose is skipped."""
    inspection, pose = FakeInspection(is_good=False, score=0.9), FakePose()
    report = make_pipeline(inspection, pose, frame).run_cycle("guide_block")

    assert report.result.status == PartStatus.NG
    assert pose.calls == 0, "pose must not run for a defective part"
    assert report.pose_skipped
    assert report.result.pose_valid is False
    assert check_result(report.result, strict=False) == []


def test_skip_pose_when_ng_can_be_disabled(frame):
    inspection, pose = FakeInspection(is_good=False, score=0.9), FakePose()
    report = make_pipeline(
        inspection, pose, frame, skip_pose_when_ng=False
    ).run_cycle("guide_block")

    assert pose.calls == 1
    # The part is still bad, so it must never be reported as pickable.
    assert report.result.is_good is False


def test_rejected_pose_yields_no_pose_status(frame):
    inspection, pose = FakeInspection(is_good=True), FakePose(valid=False)
    report = make_pipeline(inspection, pose, frame).run_cycle("guide_block")

    assert report.result.status == PartStatus.NO_POSE
    assert report.result.pose_valid is False
    assert report.result.is_good is True
    assert check_result(report.result, strict=False) == []


def test_inspection_failure_becomes_error_not_an_exception(frame):
    """One message per trigger, even when a backend throws."""
    report = make_pipeline(FakeInspection(raises=True), FakePose(), frame).run_cycle("x")

    assert report.result.status == PartStatus.ERROR
    assert "inspection failed" in report.result.message
    assert report.result.pose_valid is False


def test_capture_failure_becomes_error(frame):
    def capture(part_id, sequence):
        raise OSError("camera disconnected")

    pipeline = Pipeline(FakeInspection(), FakePose(), capture)
    report = pipeline.run_cycle("guide_block")

    assert report.result.status == PartStatus.ERROR
    assert "camera disconnected" in report.result.message


def test_pose_backend_exception_is_contained(frame):
    """PoseBackend.run swallows backend errors into an invalid estimate."""
    report = make_pipeline(FakeInspection(), FakePose(raises=True), frame).run_cycle("x")

    assert report.result.status == PartStatus.NO_POSE
    assert "pose exploded" in report.result.message


def test_sequence_increments_monotonically(frame):
    pipeline = make_pipeline(FakeInspection(), FakePose(), frame)
    sequences = [pipeline.run_cycle("guide_block").result.sequence for _ in range(4)]
    assert sequences == [1, 2, 3, 4]


def test_explicit_sequence_is_honoured(frame):
    pipeline = make_pipeline(FakeInspection(), FakePose(), frame)
    assert pipeline.run_cycle("guide_block", sequence=42).result.sequence == 42


def test_tact_budget_overrun_is_reported(frame):
    budget = TactBudget(capture=0.0, inspect=0.0, pose=0.0, total_warn=0.0, total_limit=1e9)
    report = make_pipeline(FakeInspection(), FakePose(), frame, budget=budget).run_cycle("x")

    assert set(report.budget_exceeded) >= {"capture", "inspect", "pose", "total"}
    assert report.result.status == PartStatus.OK, "a warning must not fail the cycle"


def test_hard_tact_limit_produces_error(frame):
    budget = TactBudget(total_limit=0.0)
    report = make_pipeline(FakeInspection(), FakePose(), frame, budget=budget).run_cycle("x")

    assert report.result.status == PartStatus.ERROR
    assert "hard limit" in report.result.message


def test_pipeline_returns_to_idle(frame):
    pipeline = make_pipeline(FakeInspection(), FakePose(), frame)
    pipeline.run_cycle("guide_block")
    assert pipeline.stage is Stage.IDLE


def test_stage_times_are_recorded(frame):
    report = make_pipeline(FakeInspection(), FakePose(), frame).run_cycle("guide_block")
    assert set(report.stage_times_ms) == {"capture", "inspect", "pose"}
    assert all(value >= 0.0 for value in report.stage_times_ms.values())
