"""Mock data generation: determinism, ground truth and defect injection.

The whole parallel-development plan rests on mock frames being trustworthy, so
they get tested like production code.
"""

from __future__ import annotations

import numpy as np
import pytest

from roboworld_core.mock_data import DEFECT_KINDS, apply_defect
from roboworld_core.render import Renderer, depth_to_points, plane_item
from roboworld_core.types import CameraIntrinsics

PART_ID = "guide_block"


def test_same_seed_gives_identical_frames(station):
    """CI cannot assert on numbers unless generation is deterministic."""
    first = station.sample_frame(PART_ID, seed=42, sequence=1)
    second = station.sample_frame(PART_ID, seed=42, sequence=1)
    assert np.array_equal(first.color, second.color)
    assert np.array_equal(first.depth, second.depth)
    assert np.allclose(first.gt_pose.position, second.gt_pose.position)


def test_different_seeds_give_different_poses(station):
    first = station.sample_frame(PART_ID, seed=1, sequence=1)
    second = station.sample_frame(PART_ID, seed=2, sequence=1)
    assert not np.allclose(first.gt_pose.position, second.gt_pose.position)


def test_frame_carries_ground_truth(station):
    good = station.sample_frame(PART_ID, defect=None, seed=5, sequence=1)
    defective = station.sample_frame(PART_ID, defect="scratch", seed=5, sequence=1)
    assert good.gt_is_good is True
    assert defective.gt_is_good is False
    assert good.gt_pose is not None


def test_depth_is_in_meters_and_plausible(station):
    frame = station.sample_frame(PART_ID, seed=7, sequence=1)
    valid = frame.depth > 0
    assert valid.mean() > 0.5
    # Belt at 0.60 m, part top face ~0.545 m.
    assert 0.4 < frame.depth[valid].min() < 0.6
    assert 0.55 < frame.depth[valid].max() < 0.7


def test_color_and_depth_are_aligned(station):
    """The ICD requires depth registered to color; shapes must match exactly."""
    frame = station.sample_frame(PART_ID, seed=8, sequence=1)
    assert frame.depth.shape == frame.color.shape[:2]
    assert frame.color.dtype == np.uint8
    assert frame.depth.dtype == np.float32


def test_frame_rejects_mismatched_depth(intrinsics):
    from roboworld_core.types import Frame

    with pytest.raises(ValueError, match="must match color"):
        Frame(
            color=np.zeros((48, 64, 3), dtype=np.uint8),
            depth=np.zeros((10, 10), dtype=np.float32),
            intrinsics=intrinsics,
            stamp=0.0,
        )


@pytest.mark.parametrize("kind", DEFECT_KINDS)
def test_every_defect_kind_changes_the_image(clean_station, kind):
    pose = clean_station.sample_pose(np.random.default_rng(3))
    clean = clean_station.render_frame(PART_ID, pose, defect=None, seed=1)
    damaged = clean_station.render_frame(PART_ID, pose, defect=kind, seed=1)

    changed = np.abs(clean.color.astype(int) - damaged.color.astype(int)).sum(axis=2) > 40
    assert changed.sum() > 20, f"{kind} barely changed the image"


def test_unknown_defect_kind_raises():
    with pytest.raises(ValueError, match="unknown defect kind"):
        apply_defect(
            np.zeros((10, 10, 3)), np.zeros((10, 10)), np.ones((10, 10), bool),
            "explosion", np.random.default_rng(0),
        )


def test_defects_land_on_the_part_not_the_belt(clean_station):
    """A defect straddling the silhouette is not a fair inspection sample."""
    pose = clean_station.sample_pose(np.random.default_rng(9))
    clean = clean_station.render_frame(PART_ID, pose, defect=None, seed=2)
    damaged = clean_station.render_frame(PART_ID, pose, defect="scratch", seed=2)

    changed = np.abs(clean.color.astype(int) - damaged.color.astype(int)).sum(axis=2) > 40
    # Use the renderer's exact silhouette: a depth threshold would misclassify
    # the sloped side faces of a tilted part as belt.
    assert (changed & clean.gt_part_mask).sum() / max(1, changed.sum()) > 0.95


def test_unknown_part_raises(station):
    with pytest.raises(KeyError, match="unknown part_id"):
        station.mesh("not_a_part")


# --- renderer -----------------------------------------------------------
def test_renderer_z_buffer_keeps_the_nearer_surface():
    intrinsics = CameraIntrinsics(64, 64, 60.0, 60.0, 32.0, 32.0)
    renderer = Renderer(intrinsics)
    from roboworld_core.geometry import make_transform

    far = plane_item(make_transform(np.eye(3), np.array([0.0, 0.0, 1.0])),
                     (2.0, 2.0), (200, 0, 0), label=1)
    near = plane_item(make_transform(np.eye(3), np.array([0.0, 0.0, 0.5])),
                      (0.2, 0.2), (0, 200, 0), label=2)

    result = renderer.render([far, near])
    assert result.mask[32, 32] == 2
    assert result.depth[32, 32] == pytest.approx(0.5, abs=1e-3)
    assert result.mask[2, 2] == 1


def test_depth_to_points_backprojects_correctly():
    intrinsics = CameraIntrinsics(64, 64, 100.0, 100.0, 32.0, 32.0)
    depth = np.zeros((64, 64), dtype=np.float32)
    depth[32, 32] = 0.5

    points = depth_to_points(depth, intrinsics)
    assert len(points) == 1
    assert points[0, 2] == pytest.approx(0.5)
    # Pixel (32, 32) center is at 32.5, half a pixel off the principal point.
    assert abs(points[0, 0]) < 0.01 and abs(points[0, 1]) < 0.01


def test_depth_to_points_honours_the_range_filter():
    intrinsics = CameraIntrinsics(8, 8, 10.0, 10.0, 4.0, 4.0)
    depth = np.full((8, 8), 9.0, dtype=np.float32)
    assert len(depth_to_points(depth, intrinsics, depth_range=(0.05, 5.0))) == 0
