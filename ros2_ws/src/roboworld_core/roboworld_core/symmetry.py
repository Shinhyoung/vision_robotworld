"""Part symmetry bookkeeping.

The three MC nylon blocks are 200 x 55 x 55 mm: a square cross-section prism
with machined features. Measured chamfer distance between a mesh and its own
rotations (``tools/check_symmetry.py``) is 3.5-12.5 mm, i.e. the features break
the symmetry, but by *less than the ICP inlier gate* when only one face is
visible. Two consequences the robot department must know about (ICD section 6):

1. A single-view pose may settle on a near-symmetric alternative orientation.
   The reported position is unaffected; the orientation can be off by an element
   of the near-symmetry group.
2. Evaluating orientation error against ground truth therefore has to be done
   modulo that group, otherwise a geometrically indistinguishable result counts
   as a 180-degree failure.

Both the raw and the symmetry-reduced error are reported everywhere, so the
ambiguity is visible rather than hidden by the metric.
"""

from __future__ import annotations

import numpy as np

from .geometry import Pose, matrix_from_quaternion, quaternion_angle_deg, quaternion_from_matrix


def rotation_about(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix about a (not necessarily unit) axis."""
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(a)
    if norm < 1e-12:
        raise ValueError("rotation axis must be non-zero")
    a = a / norm
    cross = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return (
        np.eye(3)
        + np.sin(angle_rad) * cross
        + (1.0 - np.cos(angle_rad)) * (cross @ cross)
    )


def build_group(generators: list[dict]) -> list[np.ndarray]:
    """Expand ``[{axis, angles_deg}, ...]`` into the full rotation set.

    The product of every listed rotation about every listed axis. Duplicates
    (within 1e-6) are removed, so listing 0 degrees on each axis is harmless.
    """
    group: list[np.ndarray] = [np.eye(3)]
    for generator in generators or []:
        axis = np.asarray(generator["axis"], dtype=np.float64)
        rotations = [
            rotation_about(axis, np.radians(float(angle)))
            for angle in generator.get("angles_deg", [0.0])
        ]
        group = [existing @ rotation for existing in group for rotation in rotations]

    unique: list[np.ndarray] = []
    for candidate in group:
        if not any(np.allclose(candidate, seen, atol=1e-6) for seen in unique):
            unique.append(candidate)
    return unique


def group_for_part(cfg, part_id: str) -> list[np.ndarray]:
    """Symmetry group of ``part_id`` from the ``parts`` config section."""
    entry = cfg.get(f"parts.{part_id}", {}) or {}
    return build_group(entry.get("symmetry", []))


def rotation_error_deg(
    estimated: Pose, truth: Pose, group: list[np.ndarray] | None = None
) -> tuple[float, float]:
    """Return ``(raw_deg, symmetry_reduced_deg)`` orientation error.

    ``symmetry_reduced_deg`` is the minimum over the part's symmetry group; with
    no group supplied the two values are identical.
    """
    raw = quaternion_angle_deg(estimated.orientation, truth.orientation)
    if not group:
        return raw, raw

    truth_matrix = matrix_from_quaternion(truth.orientation)
    best = raw
    for rotation in group:
        candidate = quaternion_from_matrix(truth_matrix @ rotation)
        best = min(best, quaternion_angle_deg(estimated.orientation, candidate))
    return raw, best


def pose_error(
    estimated: Pose, truth: Pose, group: list[np.ndarray] | None = None
) -> dict[str, float]:
    """Translation and (raw / symmetry-reduced) rotation error of one estimate."""
    raw_deg, reduced_deg = rotation_error_deg(estimated, truth, group)
    return {
        "translation_mm": float(np.linalg.norm(estimated.position - truth.position) * 1000.0),
        "rotation_deg": raw_deg,
        "rotation_deg_symmetry_reduced": reduced_deg,
    }


def canonical_rotation(
    rotation: np.ndarray,
    group: list[np.ndarray],
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Pick one representative from a rotation's symmetry-equivalent set.

    A part whose shape repeats under a rotation has several equally good fits,
    and registration picks whichever noise favours that frame. Measured on a
    static part, 15 consecutive frames alternated between yaw +20.6 and -159.4
    -- exactly 180 apart -- with fitness 0.97-1.00 on every one. Nothing was
    wrong with any of them; they are the same physical placement.

    A consumer cannot act on a number that jumps like that, so this collapses
    the set to one member: the candidate whose model +x lies closest to
    ``reference``, ties broken on +y. Deterministic, and independent of frame
    order -- there is no filtering or history here.

    **It does not resolve the ambiguity, it hides it.** The part really may be
    end-for-end reversed and no single view can tell. That is safe only when
    the grasp is invariant under the group -- gripping the long axis at its
    centre, for this part. If a gripper ever keys on a specific end, this
    function will hand it a confident wrong answer; see ICD section 6.1.
    """
    if len(group) <= 1:
        return rotation

    reference = np.eye(3) if reference is None else np.asarray(reference, dtype=np.float64)
    best, best_key = None, None
    for element in group:
        candidate = np.asarray(rotation, dtype=np.float64) @ element
        # Lexicographic on the first two axes: a total order, so the choice
        # cannot depend on which element happens to be enumerated first.
        key = (
            round(float(candidate[:, 0] @ reference[:, 0]), 9),
            round(float(candidate[:, 1] @ reference[:, 1]), 9),
        )
        if best_key is None or key > best_key:
            best, best_key = candidate, key
    return best


def canonical_pose(pose, group: list[np.ndarray], reference: np.ndarray | None = None):
    """:func:`canonical_rotation` applied to a :class:`~roboworld_core.geometry.Pose`.

    Position is untouched: the symmetry acts about the part centre, which is
    where the model frame origin sits, so every member of the set reports the
    same point.
    """
    from .geometry import Pose, matrix_from_quaternion, quaternion_from_matrix

    if len(group) <= 1:
        return pose
    rotation = canonical_rotation(matrix_from_quaternion(pose.orientation), group, reference)
    return Pose(pose.position, quaternion_from_matrix(rotation), pose.frame_id)
