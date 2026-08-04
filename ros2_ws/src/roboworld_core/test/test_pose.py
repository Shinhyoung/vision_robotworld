"""Pose backend, segmentation and symmetry tests (claude.md section 3.3 DoD)."""

from __future__ import annotations

import numpy as np
import pytest

from roboworld_core.geometry import Pose, euler_to_matrix, quaternion_from_matrix
from roboworld_core.pose import build_backend, voxel_downsample
from roboworld_core.pose.base import PoseSettings
from roboworld_core.segmentation import fit_plane_ransac, segment_from_config
from roboworld_core.symmetry import build_group, group_for_part, pose_error, rotation_about

PART_ID = "guide_block"

#: Tolerances for the CPU ICP backend on mock data, measured with
#: tools/evaluate.py (median 1.4 mm / 0.7 deg, max 3.7 mm / 2.3 deg).
MAX_TRANSLATION_MM = 6.0
MAX_ROTATION_DEG = 6.0


# --- segmentation -------------------------------------------------------
def test_plane_ransac_recovers_a_known_plane():
    rng = np.random.default_rng(0)
    points = np.column_stack(
        [rng.uniform(-0.5, 0.5, 800), rng.uniform(-0.3, 0.3, 800), np.full(800, 0.6)]
    )
    points += rng.normal(0.0, 0.0005, points.shape)

    plane = fit_plane_ransac(points, iterations=80, distance_threshold=0.003, seed=1)
    assert plane is not None
    assert abs(abs(plane.normal[2]) - 1.0) < 1e-2
    assert plane.inlier_ratio > 0.95
    # Normal points towards the camera: a point above the plane is positive.
    assert plane.signed_distance(np.array([[0.0, 0.0, 0.55]]))[0] > 0.0


def test_plane_ransac_needs_three_points():
    assert fit_plane_ransac(np.zeros((2, 3))) is None


def test_segmentation_isolates_the_part(cfg, good_frame):
    segmentation = segment_from_config(good_frame, cfg)
    assert segmentation.ok
    assert segmentation.plane is not None
    # 200 x 55 mm at 0.6 m with fx 384 is roughly 128 x 35 px.
    assert 2000 < segmentation.pixel_count < 12000
    # Every segmented point must sit above the belt, not on it.
    heights = segmentation.plane.signed_distance(segmentation.points)
    assert heights.min() > 0.0


def test_segmentation_of_an_empty_scene_reports_nothing(cfg, good_frame):
    from roboworld_core.types import Frame

    empty = Frame(
        color=np.zeros_like(good_frame.color),
        depth=np.zeros_like(good_frame.depth),
        intrinsics=good_frame.intrinsics,
        stamp=0.0,
        part_id=PART_ID,
    )
    assert not segment_from_config(empty, cfg).ok


# --- symmetry -----------------------------------------------------------
def test_symmetry_group_has_the_expected_order(cfg):
    """4 rotations about the long axis x 2 end-for-end flips = 8."""
    assert len(group_for_part(cfg, PART_ID)) == 8


def test_build_group_deduplicates():
    generators = [{"axis": [0, 0, 1], "angles_deg": [0.0, 0.0, 360.0]}]
    assert len(build_group(generators)) == 1


def test_rotation_about_rejects_zero_axis():
    with pytest.raises(ValueError):
        rotation_about(np.zeros(3), 1.0)


def test_symmetry_reduces_an_equivalent_orientation(cfg):
    """A 180-degree flip about the long axis is a 180-degree raw error but ~0 reduced."""
    group = group_for_part(cfg, PART_ID)
    truth = Pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    flipped = Pose(np.zeros(3), quaternion_from_matrix(euler_to_matrix(np.pi, 0.0, 0.0)))

    error = pose_error(flipped, truth, group)
    assert error["rotation_deg"] == pytest.approx(180.0, abs=1e-3)
    assert error["rotation_deg_symmetry_reduced"] < 1e-6


def test_symmetry_does_not_hide_a_real_error(cfg):
    group = group_for_part(cfg, PART_ID)
    truth = Pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    tilted = Pose(np.zeros(3), quaternion_from_matrix(euler_to_matrix(0.0, np.radians(30), 0.0)))
    assert pose_error(tilted, truth, group)["rotation_deg_symmetry_reduced"] > 25.0


# --- helpers ------------------------------------------------------------
def test_voxel_downsample_reduces_and_stays_inside_the_cloud():
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.05, 0.05, (5000, 3))
    reduced = voxel_downsample(points, 0.005)
    assert 0 < len(reduced) < len(points)
    assert np.all(reduced.min(axis=0) >= points.min(axis=0) - 1e-9)
    assert np.all(reduced.max(axis=0) <= points.max(axis=0) + 1e-9)


def test_voxel_downsample_passthrough_on_zero_size():
    points = np.zeros((10, 3))
    assert len(voxel_downsample(points, 0.0)) == 10


# --- stub backend -------------------------------------------------------
def test_stub_returns_ground_truth(cfg, good_frame):
    estimate = build_backend(cfg, PART_ID, backend="stub").run(good_frame)
    assert estimate.valid
    assert np.allclose(estimate.pose.position, good_frame.gt_pose.position)


def test_stub_pose_is_in_the_camera_optical_frame(cfg, good_frame):
    estimate = build_backend(cfg, PART_ID, backend="stub").run(good_frame)
    assert estimate.pose.frame_id == good_frame.intrinsics.frame_id


# --- acceptance gate ----------------------------------------------------
@pytest.mark.parametrize(
    "fitness,rmse,position,expected_ok",
    [
        (0.9, 0.001, [0.0, 0.0, 0.57], True),
        (0.1, 0.001, [0.0, 0.0, 0.57], False),   # fitness too low
        (0.9, 0.050, [0.0, 0.0, 0.57], False),   # rmse too high
        (0.9, 0.001, [0.0, 0.0, 2.50], False),   # outside the z window
        (0.9, 0.001, [0.9, 0.0, 0.57], False),   # lateral offset too large
    ],
)
def test_acceptance_gate(fitness, rmse, position, expected_ok):
    from roboworld_core.pose.base import PoseBackend

    class Dummy(PoseBackend):
        name = "dummy"

        def estimate(self, frame):  # pragma: no cover - unused
            raise NotImplementedError

    backend = Dummy(PoseSettings())
    pose = Pose(np.array(position), np.array([0.0, 0.0, 0.0, 1.0]), "camera")
    valid, reason = backend.validate(pose, fitness, rmse)
    assert valid is expected_ok
    assert (reason == "") is expected_ok


def test_acceptance_gate_rejects_non_finite_positions():
    from roboworld_core.pose.base import PoseBackend

    class Dummy(PoseBackend):
        name = "dummy"

        def estimate(self, frame):  # pragma: no cover - unused
            raise NotImplementedError

    pose = Pose(np.array([np.nan, 0.0, 0.5]), np.array([0.0, 0.0, 0.0, 1.0]))
    valid, reason = Dummy(PoseSettings()).validate(pose, 0.9, 0.001)
    assert not valid and "non-finite" in reason


# --- ICP ----------------------------------------------------------------
@pytest.mark.parametrize("part_id", ["guide_block", "spacer_block", "end_stopper"])
def test_icp_recovers_the_ground_truth_pose(cfg, station, part_id):
    """The DoD check: pose matches ground truth in the documented units."""
    backend = build_backend(cfg, part_id, backend="icp")
    group = group_for_part(cfg, part_id)

    frame = station.sample_frame(part_id, seed=2024, sequence=1)
    estimate = backend.run(frame)

    assert estimate.valid, estimate.message
    assert estimate.pose.frame_id == frame.intrinsics.frame_id

    error = pose_error(estimate.pose, frame.gt_pose, group)
    assert error["translation_mm"] < MAX_TRANSLATION_MM, error
    assert error["rotation_deg_symmetry_reduced"] < MAX_ROTATION_DEG, error


def test_icp_reports_fitness_and_rmse(cfg, station):
    estimate = build_backend(cfg, PART_ID, backend="icp").run(
        station.sample_frame(PART_ID, seed=99, sequence=1)
    )
    assert 0.0 <= estimate.fitness <= 1.0
    assert 0.0 <= estimate.rmse_m < 0.02
    assert estimate.inference_time_ms > 0.0


def test_icp_on_an_empty_scene_is_invalid_not_an_exception(cfg, good_frame):
    from roboworld_core.types import Frame

    empty = Frame(
        color=np.zeros_like(good_frame.color),
        depth=np.zeros_like(good_frame.depth),
        intrinsics=good_frame.intrinsics,
        stamp=0.0,
        part_id=PART_ID,
    )
    estimate = build_backend(cfg, PART_ID, backend="icp").run(empty)
    assert estimate.valid is False
    assert estimate.message


def test_unknown_pose_backend_raises(cfg):
    with pytest.raises(ValueError, match="unknown pose backend"):
        build_backend(cfg, PART_ID, backend="does_not_exist")


def test_unknown_part_raises(cfg):
    with pytest.raises(ValueError, match="unknown part_id"):
        build_backend(cfg, "not_a_part", backend="icp")


def test_foundationpose_without_a_bridge_fails_loudly(cfg, good_frame):
    """A missing Isaac ROS graph must never silently produce a wrong pose."""
    from roboworld_core.pose import FoundationPoseUnavailable

    backend = build_backend(cfg, PART_ID, backend="foundationpose")
    with pytest.raises(FoundationPoseUnavailable):
        backend.estimate(good_frame)
    # PoseBackend.run contains it instead of propagating.
    assert backend.run(good_frame).valid is False


# --- parts registered without CAD ---------------------------------------
def test_has_cad_distinguishes_registered_parts(cfg):
    from roboworld_core.pose import has_cad

    assert has_cad(cfg, PART_ID) is True
    cadless = cfg.merged_with({"parts": {"scanned": {"mesh": ""}}})
    assert has_cad(cadless, "scanned") is False


def test_missing_cad_raises_by_default(cfg):
    """Tools must see the problem immediately, not a silent wrong answer."""
    cadless = cfg.merged_with({"parts": {"scanned": {"mesh": ""}}})
    with pytest.raises(ValueError, match="no CAD mesh"):
        build_backend(cadless, "scanned", backend="icp")


def test_missing_cad_degrades_when_allowed(cfg, good_frame):
    """A long-running node keeps inspecting; only the pose is unavailable."""
    cadless = cfg.merged_with({"parts": {"scanned": {"mesh": ""}}})
    backend = build_backend(cadless, "scanned", backend="icp", allow_missing_cad=True)

    assert backend.name == "no_cad"
    estimate = backend.run(good_frame)
    assert estimate.valid is False
    assert "no CAD mesh" in estimate.message


def test_cadless_part_yields_no_pose_status(cfg, good_frame):
    """End to end: the contract's STATUS_NO_POSE covers this case already."""
    from roboworld_core.inspection.base import InspectionBackend, InspectionSettings
    from roboworld_core.pipeline import Pipeline
    from roboworld_core.types import InspectionResult, PartStatus

    class AlwaysGood(InspectionBackend):
        name = "always_good"

        def score_map(self, frame):  # pragma: no cover - infer overridden
            raise NotImplementedError

        def infer(self, frame):
            return InspectionResult(
                part_id=frame.part_id, sequence=frame.sequence, stamp=frame.stamp,
                frame_id=frame.intrinsics.frame_id, is_good=True,
                anomaly_score=0.1, threshold=0.5, backend=self.name,
            )

    cadless = cfg.merged_with({"parts": {"scanned": {"mesh": ""}}})
    pose_backend = build_backend(cadless, "scanned", backend="icp", allow_missing_cad=True)
    pipeline = Pipeline(
        AlwaysGood(InspectionSettings()), pose_backend,
        lambda part_id, sequence: good_frame,
    )
    result = pipeline.run_cycle("scanned").result

    assert result.status == PartStatus.NO_POSE
    assert result.is_good is True          # inspection still decided
    assert result.pose_valid is False
    from roboworld_core.contract import check_result
    assert check_result(result, strict=False) == []
