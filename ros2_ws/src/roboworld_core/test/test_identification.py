"""Only the registered part may be inspected or pose-estimated.

Segmentation used to take the largest object above the plane. With a bigger
decoy in the scene that produced a *valid* pose 150 mm from the truth -- the
worst possible failure, since the robot department is told to pick.
"""

from __future__ import annotations

import numpy as np
import pytest

from roboworld_core.geometry import (
    Pose,
    euler_to_matrix,
    make_transform,
    quaternion_from_matrix,
)
from roboworld_core.mesh_io import load_ply
from roboworld_core.paths import resolve_path
from roboworld_core.pose import build_backend
from roboworld_core.render import Renderer, RenderItem, plane_item
from roboworld_core.segmentation import (
    measure_extents,
    segment_from_config,
    size_mismatch,
)
from roboworld_core.symmetry import group_for_part, pose_error
from roboworld_core.types import Frame

PART_ID = "guide_block"
_PART_POSITION = np.array([-0.085, -0.02, 0.5725])


def build_scene(cfg, intrinsics, decoy_half_extents=None, decoy_at=(0.11, 0.03, 0.57)):
    """Render the part on a plane, optionally with a box decoy beside it."""
    mesh = load_ply(resolve_path(cfg.get(f"parts.{PART_ID}.mesh")))
    rotation = euler_to_matrix(0.0, 0.0, 0.25)
    items = [
        plane_item(make_transform(np.eye(3), np.array([0.0, 0.0, 0.60])),
                   (1.4, 0.7), (58, 62, 70), 0),
        RenderItem(mesh.vertices, mesh.faces, np.array([196, 188, 170], np.uint8),
                   make_transform(rotation, _PART_POSITION), 1),
    ]
    if decoy_half_extents is not None:
        hx, hy, hz = decoy_half_extents
        vertices = np.array([[x, y, z] for x in (-hx, hx)
                             for y in (-hy, hy) for z in (-hz, hz)])
        faces = np.array([
            [0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7], [0, 5, 1], [0, 4, 5],
            [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ])
        items.append(RenderItem(vertices, faces, np.array([120, 140, 200], np.uint8),
                                make_transform(np.eye(3), np.array(decoy_at)), 2))

    rendered = Renderer(intrinsics).render(items)
    frame = Frame(rendered.color, rendered.depth, intrinsics, 0.0, part_id=PART_ID)
    truth = Pose(_PART_POSITION, quaternion_from_matrix(rotation), intrinsics.frame_id)
    return frame, rendered.mask, truth


def with_identification(cfg, enabled: bool):
    return cfg.merged_with({"pose": {"segmentation": {"identify_by_size": enabled}}})


# --- measurement helpers -------------------------------------------------
def test_measure_extents_is_rotation_invariant(cfg, station):
    """Extents describe the object, not how it happens to lie on the surface."""
    from roboworld_core.segmentation import segment_part

    measured = []
    for seed in (3, 17, 42):
        frame = station.sample_frame(PART_ID, seed=seed, sequence=1)
        segmentation = segment_part(frame)
        measured.append(measure_extents(segmentation.points, segmentation.plane))

    reference = measured[0]
    for extents in measured[1:]:
        assert np.allclose(extents, reference, atol=0.006), (reference, extents)


def test_size_mismatch_scores_relative_error():
    assert size_mismatch(np.array([0.2, 0.055, 0.055]),
                         np.array([0.2, 0.055, 0.055])) == pytest.approx(0.0)
    # 100 mm against 200 mm on the longest side -> 50 % off.
    assert size_mismatch(np.array([0.1, 0.055, 0.055]),
                         np.array([0.2, 0.055, 0.055])) == pytest.approx(0.5, abs=1e-6)


# --- selection -----------------------------------------------------------
def test_picks_the_part_when_a_bigger_decoy_is_present(cfg, intrinsics):
    frame, labels, _ = build_scene(cfg, intrinsics, decoy_half_extents=(0.05, 0.05, 0.025))
    segmentation = segment_from_config(frame, with_identification(cfg, True),
                                       part_id=PART_ID)

    assert segmentation.ok, segmentation.reason
    on_part = (labels == 1)[segmentation.mask].mean()
    assert on_part > 0.9, "selected something other than the part"


def test_rejects_a_scene_containing_only_a_decoy(cfg, intrinsics):
    """Nothing matching the part means no result -- not the nearest blob."""
    mesh_free = cfg.get(f"parts.{PART_ID}")
    assert mesh_free  # sanity

    intrinsics_local = intrinsics
    items_frame, _, _ = build_scene(cfg, intrinsics_local)
    # Replace the scene with a decoy only.
    vertices = np.array([[x, y, z] for x in (-0.09, 0.09)
                         for y in (-0.07, 0.07) for z in (-0.03, 0.03)])
    faces = np.array([
        [0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7], [0, 5, 1], [0, 4, 5],
        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ])
    rendered = Renderer(intrinsics_local).render([
        plane_item(make_transform(np.eye(3), np.array([0.0, 0.0, 0.60])),
                   (1.4, 0.7), (58, 62, 70), 0),
        RenderItem(vertices, faces, np.array([120, 140, 200], np.uint8),
                   make_transform(np.eye(3), np.array([0.0, 0.0, 0.57])), 2),
    ])
    frame = Frame(rendered.color, rendered.depth, intrinsics_local, 0.0, part_id=PART_ID)

    segmentation = segment_from_config(frame, with_identification(cfg, True),
                                       part_id=PART_ID)
    assert not segmentation.ok
    assert "does not" in segmentation.reason or "no object matches" in segmentation.reason
    assert items_frame is not None


def test_pose_backend_refuses_a_decoy(cfg, intrinsics):
    """The regression that matters: a decoy must never yield a *valid* pose."""
    frame, _, truth = build_scene(cfg, intrinsics, decoy_half_extents=(0.09, 0.07, 0.03))

    without = build_backend(with_identification(cfg, False), PART_ID,
                            backend="icp").run(frame)
    with_id = build_backend(with_identification(cfg, True), PART_ID,
                            backend="icp").run(frame)

    group = group_for_part(cfg, PART_ID)
    if without.valid:
        # Historical behaviour: a confidently wrong pose.
        assert pose_error(without.pose, truth, group)["translation_mm"] > 50.0

    assert not with_id.valid, "a decoy produced a valid pose"
    assert with_id.message


def test_identification_does_not_disturb_a_clean_scene(cfg, intrinsics):
    """With only the part present, identification must change nothing."""
    frame, _, truth = build_scene(cfg, intrinsics)
    estimate = build_backend(with_identification(cfg, True), PART_ID,
                             backend="icp").run(frame)

    assert estimate.valid, estimate.message
    error = pose_error(estimate.pose, truth, group_for_part(cfg, PART_ID))
    assert error["translation_mm"] < 6.0, error


def test_part_without_geometry_falls_back_to_largest(cfg, intrinsics):
    """A part registered without a mesh cannot be identified by size."""
    from roboworld_core.segmentation import expected_extents_for

    cadless = cfg.merged_with({"parts": {"scanned": {"mesh": ""}}})
    assert expected_extents_for(cadless, "scanned") is None

    frame, _, _ = build_scene(cfg, intrinsics)
    segmentation = segment_from_config(frame, with_identification(cfg, True),
                                       part_id="scanned")
    assert segmentation.ok, "fallback must still segment something"
