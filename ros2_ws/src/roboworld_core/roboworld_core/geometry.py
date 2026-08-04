"""Rigid transform and quaternion helpers.

Conventions fixed by claude.md section 4 and enforced here:

* lengths in **meters**,
* rotations as **unit quaternions ordered (x, y, z, w)** -- the ROS ordering,
* transforms as 4x4 homogeneous matrices, column vectors, ``T @ p``,
* ``T_a_b`` names the transform that maps a point expressed in frame ``b``
  into frame ``a`` (``p_a = T_a_b @ p_b``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Below this quaternion norm the rotation is treated as degenerate.
_EPS = 1e-12


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Return ``q`` (x, y, z, w) scaled to unit length, sign-fixed to w >= 0."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < _EPS:
        raise ValueError("cannot normalize a zero quaternion")
    q = q / norm
    # w < 0 encodes the same rotation; canonicalise so equality tests are sane.
    if q[3] < 0.0:
        q = -q
    return q


def quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion (x, y, z, w).

    Uses Shepperd's branch selection, which stays numerically stable for all
    rotations (the naive trace formula degrades near 180 degrees).
    """
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return quaternion_normalize(np.array([x, y, z, w]))


def matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    x, y, z, w = quaternion_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product ``q1 * q2`` in (x, y, z, w) ordering."""
    x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64).reshape(4)
    x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64).reshape(4)
    return quaternion_normalize(
        np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ]
        )
    )


def quaternion_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Smallest rotation angle in degrees between two orientations."""
    a = quaternion_normalize(q1)
    b = quaternion_normalize(q2)
    dot = abs(float(np.dot(a, b)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic Z-Y-X (yaw-pitch-roll) Euler angles in radians to 3x3 matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def matrix_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`euler_to_matrix`: 3x3 -> (roll, pitch, yaw) in radians.

    Euler angles are for **human consumption only** -- logs, HUDs, docs. The
    contract publishes quaternions (ICD section 4), because Euler angles have
    gimbal lock and three competing conventions.
    """
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll), so -r[2, 0] = sin(pitch).
    sin_pitch = -r[2, 0]
    sin_pitch = min(1.0, max(-1.0, sin_pitch))
    pitch = float(np.arcsin(sin_pitch))

    if abs(sin_pitch) > 1.0 - 1e-9:  # gimbal lock: roll and yaw are degenerate
        roll = float(np.arctan2(-r[1, 2], r[1, 1]))
        yaw = 0.0
    else:
        roll = float(np.arctan2(r[2, 1], r[2, 2]))
        yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    return roll, pitch, yaw


def quaternion_to_euler_deg(q: np.ndarray) -> tuple[float, float, float]:
    """Quaternion (x, y, z, w) -> (roll, pitch, yaw) in degrees, for display."""
    roll, pitch, yaw = matrix_to_euler(matrix_from_quaternion(q))
    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from a 3x3 R and a 3-vector t."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform (transpose trick, no general solve)."""
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation = t[:3, :3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ t[:3, 3]
    return inverse


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an ``(N, 3)`` array of points."""
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return pts @ t[:3, :3].T + t[:3, 3]


def orthonormalize(rotation: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3) via SVD (right-handed)."""
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64).reshape(3, 3))
    r = u @ vt
    if np.linalg.det(r) < 0.0:  # reflection -> flip the least significant axis
        u[:, -1] *= -1.0
        r = u @ vt
    return r


@dataclass(frozen=True)
class Pose:
    """A 6D pose: position in meters, orientation as (x, y, z, w) quaternion."""

    position: np.ndarray
    orientation: np.ndarray
    frame_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position", np.asarray(self.position, dtype=np.float64).reshape(3)
        )
        object.__setattr__(self, "orientation", quaternion_normalize(self.orientation))

    @classmethod
    def identity(cls, frame_id: str = "") -> Pose:
        return cls(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), frame_id)

    @classmethod
    def from_matrix(cls, transform: np.ndarray, frame_id: str = "") -> Pose:
        t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        return cls(t[:3, 3], quaternion_from_matrix(orthonormalize(t[:3, :3])), frame_id)

    def as_matrix(self) -> np.ndarray:
        return make_transform(matrix_from_quaternion(self.orientation), self.position)

    def transformed_by(self, transform: np.ndarray, frame_id: str) -> Pose:
        """Express this pose in another frame: ``T_new_old @ self``."""
        return Pose.from_matrix(np.asarray(transform).reshape(4, 4) @ self.as_matrix(), frame_id)

    def translation_error_m(self, other: Pose) -> float:
        return float(np.linalg.norm(self.position - other.position))

    def rotation_error_deg(self, other: Pose) -> float:
        return quaternion_angle_deg(self.orientation, other.orientation)


def scale_factor(units: str) -> float:
    """Return the multiplier converting ``units`` to meters.

    CAD arrives in millimeters often enough that section 4 of claude.md makes
    the mm->m rule an explicit documented conversion rather than a magic 0.001.
    """
    table = {"m": 1.0, "meter": 1.0, "meters": 1.0, "mm": 1e-3, "millimeter": 1e-3,
             "millimeters": 1e-3, "cm": 1e-2, "centimeter": 1e-2, "centimeters": 1e-2}
    key = str(units).strip().lower()
    if key not in table:
        raise ValueError(f"unsupported length unit '{units}' (expected one of {sorted(table)})")
    return table[key]
