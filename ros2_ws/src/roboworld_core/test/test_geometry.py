"""Geometry and unit conventions (claude.md section 4)."""

from __future__ import annotations

import numpy as np
import pytest

from roboworld_core.geometry import (
    Pose,
    euler_to_matrix,
    invert_transform,
    make_transform,
    matrix_from_quaternion,
    orthonormalize,
    quaternion_angle_deg,
    quaternion_from_matrix,
    quaternion_multiply,
    quaternion_normalize,
    scale_factor,
    transform_points,
)


def test_quaternion_ordering_is_xyzw():
    """Identity must be (0, 0, 0, 1) -- the ROS ordering, not (w, x, y, z)."""
    identity = Pose.identity()
    assert np.allclose(identity.orientation, [0.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "angles",
    [(0.0, 0.0, 0.0), (0.3, -0.2, 1.1), (np.pi / 2, 0.0, 0.0),
     (0.0, 0.0, np.pi), (-1.2, 0.9, -2.7)],
)
def test_quaternion_matrix_roundtrip(angles):
    rotation = euler_to_matrix(*angles)
    recovered = matrix_from_quaternion(quaternion_from_matrix(rotation))
    assert np.allclose(rotation, recovered, atol=1e-9)


def test_quaternion_from_matrix_near_180_degrees():
    """Shepperd's method must stay stable where the naive trace formula fails."""
    for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])):
        rotation = euler_to_matrix(*(axis * np.pi))
        quaternion = quaternion_from_matrix(rotation)
        assert np.isclose(np.linalg.norm(quaternion), 1.0)
        assert np.allclose(matrix_from_quaternion(quaternion), rotation, atol=1e-9)


def test_quaternion_normalize_rejects_zero():
    with pytest.raises(ValueError):
        quaternion_normalize(np.zeros(4))


def test_quaternion_canonical_sign():
    """q and -q are the same rotation; normalisation must pick one."""
    q = quaternion_normalize(np.array([0.1, 0.2, 0.3, -0.9]))
    assert q[3] >= 0.0


def test_quaternion_multiply_matches_matrix_product():
    a = quaternion_from_matrix(euler_to_matrix(0.4, -0.1, 0.9))
    b = quaternion_from_matrix(euler_to_matrix(-0.7, 0.3, 0.2))
    combined = matrix_from_quaternion(quaternion_multiply(a, b))
    expected = matrix_from_quaternion(a) @ matrix_from_quaternion(b)
    assert np.allclose(combined, expected, atol=1e-9)


def test_quaternion_angle_is_symmetric_and_bounded():
    a = quaternion_from_matrix(euler_to_matrix(0.0, 0.0, 0.0))
    b = quaternion_from_matrix(euler_to_matrix(0.0, 0.0, np.pi / 2))
    assert np.isclose(quaternion_angle_deg(a, b), 90.0, atol=1e-6)
    assert np.isclose(quaternion_angle_deg(b, a), 90.0, atol=1e-6)


def test_invert_transform_is_exact_inverse():
    transform = make_transform(euler_to_matrix(0.2, 0.4, -0.6), np.array([0.1, -0.2, 0.57]))
    assert np.allclose(transform @ invert_transform(transform), np.eye(4), atol=1e-12)


def test_transform_points_matches_manual_application():
    transform = make_transform(euler_to_matrix(0.1, 0.2, 0.3), np.array([1.0, 2.0, 3.0]))
    points = np.random.default_rng(0).normal(size=(20, 3))
    expected = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    assert np.allclose(transform_points(transform, points), expected)


def test_orthonormalize_rejects_reflections():
    """A reflection must come back as a proper rotation (det = +1)."""
    reflection = np.diag([1.0, 1.0, -1.0])
    assert np.isclose(np.linalg.det(orthonormalize(reflection)), 1.0)


def test_pose_matrix_roundtrip():
    pose = Pose(np.array([0.01, -0.02, 0.573]),
                quaternion_from_matrix(euler_to_matrix(0.05, -0.1, 2.0)), "camera")
    recovered = Pose.from_matrix(pose.as_matrix(), "camera")
    assert np.allclose(recovered.position, pose.position, atol=1e-12)
    assert quaternion_angle_deg(recovered.orientation, pose.orientation) < 1e-9


def test_pose_transformed_by_composes_frames():
    pose = Pose(np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 0.0, 1.0]), "camera")
    transform = make_transform(np.eye(3), np.array([1.0, 0.0, 0.0]))
    moved = pose.transformed_by(transform, "world")
    assert moved.frame_id == "world"
    assert np.allclose(moved.position, [1.0, 0.0, 0.5])


def test_pose_rejects_non_unit_input_by_normalizing():
    pose = Pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 5.0]))
    assert np.isclose(np.linalg.norm(pose.orientation), 1.0)


def test_scale_factor_millimeter_conversion():
    """CAD in mm must convert to m -- the documented mm->m rule."""
    assert scale_factor("mm") == pytest.approx(1e-3)
    assert scale_factor("m") == pytest.approx(1.0)
    assert scale_factor("CM") == pytest.approx(1e-2)


def test_scale_factor_rejects_unknown_unit():
    with pytest.raises(ValueError, match="unsupported length unit"):
        scale_factor("inch")


# --- Euler angles (display only) -----------------------------------------
@pytest.mark.parametrize(
    "angles",
    [(0.0, 0.0, 0.0), (0.3, -0.2, 1.1), (-1.2, 0.9, -2.7), (0.0, 0.0, np.pi / 2)],
)
def test_euler_roundtrip(angles):
    from roboworld_core.geometry import matrix_to_euler

    rotation = euler_to_matrix(*angles)
    recovered = euler_to_matrix(*matrix_to_euler(rotation))
    assert np.allclose(rotation, recovered, atol=1e-9)


def test_euler_handles_gimbal_lock():
    """Pitch at +/-90 degrees is degenerate; it must not produce NaNs."""
    from roboworld_core.geometry import matrix_to_euler

    for pitch in (np.pi / 2, -np.pi / 2):
        rotation = euler_to_matrix(0.4, pitch, 0.0)
        roll, recovered_pitch, yaw = matrix_to_euler(rotation)
        assert all(np.isfinite([roll, recovered_pitch, yaw]))
        assert np.allclose(euler_to_matrix(roll, recovered_pitch, yaw), rotation, atol=1e-6)


def test_quaternion_to_euler_deg_matches_the_matrix_path():
    from roboworld_core.geometry import quaternion_to_euler_deg

    rotation = euler_to_matrix(0.2, -0.4, 1.0)
    roll, pitch, yaw = quaternion_to_euler_deg(quaternion_from_matrix(rotation))
    assert np.allclose(
        euler_to_matrix(np.radians(roll), np.radians(pitch), np.radians(yaw)),
        rotation, atol=1e-9,
    )
