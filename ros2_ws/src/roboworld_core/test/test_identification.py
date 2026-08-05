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


# --- support surface -----------------------------------------------------
def _ribbed_scene(crown_z=0.600, roller_radius_m=0.030, part_height_m=0.051, seed=0):
    """A roller conveyor with a part on it, as a point cloud.

    The rollers are **cylinders**, so the visible surface is a continuum of
    depths running from each crown down its flanks -- not two discrete levels.
    That continuum is what defeats RANSAC: there is no dominant plane to lock
    onto, so the fit settles somewhere down the flanks. A two-level model does
    not reproduce it (RANSAC just picks the more numerous level).

    Built directly rather than rendered: the mock renderer only makes flat
    belts, which is exactly the case this does not cover.
    """
    rng = np.random.default_rng(seed)
    n = 12000
    x = rng.uniform(-0.30, 0.30, n)
    y = rng.uniform(-0.20, 0.20, n)
    # Across-roller coordinate, wrapped: each roller spans +/- the visible arc.
    visible = roller_radius_m * 0.93
    across = rng.uniform(-visible, visible, n)
    relief = roller_radius_m - np.sqrt(
        np.maximum(roller_radius_m**2 - across**2, 0.0)
    )
    z = crown_z + relief + rng.normal(0, 0.0008, n)
    support = np.stack([x, y, z], axis=1)
    # Part: a flat top face sitting ON the crowns.
    m = 4000
    part = np.stack([
        rng.uniform(-0.115, 0.115, m),
        rng.uniform(-0.0245, 0.0245, m),
        np.full(m, crown_z - part_height_m) + rng.normal(0, 0.0008, m),
    ], axis=1)
    return np.concatenate([support, part]), part


def test_ransac_plane_lands_between_the_crowns_and_the_gaps():
    """The failure this exists to document, measured on a real conveyor.

    A plane fitted to crowns-plus-gaps sits below the surface the part rests
    on, so every height is measured from too low a datum: a part known to be
    51 mm tall read 71 mm (+39 %) on the D455.
    """
    from roboworld_core.segmentation import fit_plane_ransac

    cloud, part = _ribbed_scene()
    plane = fit_plane_ransac(cloud, iterations=200, distance_threshold=0.006, seed=0)

    assert plane is not None
    measured_mm = plane.signed_distance(part).max() * 1000
    assert measured_mm > 55.0, (
        f"expected the naive plane to overstate 51 mm, got {measured_mm:.1f} mm"
    )


def test_crown_refit_recovers_the_true_height_on_a_ribbed_surface():
    from roboworld_core.segmentation import fit_plane_ransac, refine_support_to_crowns

    cloud, part = _ribbed_scene()
    plane = fit_plane_ransac(cloud, iterations=200, distance_threshold=0.006, seed=0)
    refined = refine_support_to_crowns(cloud, plane, seed=0)

    assert refined is not None
    measured_mm = refined.signed_distance(part).max() * 1000
    assert abs(measured_mm - 51.0) < 4.0, f"got {measured_mm:.1f} mm, expected ~51"


def test_crown_refit_leaves_a_flat_surface_alone():
    """A flat belt is already the support surface; refitting must not move it."""
    from roboworld_core.segmentation import fit_plane_ransac, refine_support_to_crowns

    cloud, part = _ribbed_scene(roller_radius_m=1e-6)  # zero relief = flat
    plane = fit_plane_ransac(cloud, iterations=200, distance_threshold=0.006, seed=0)
    refined = refine_support_to_crowns(cloud, plane, seed=0)

    before = plane.signed_distance(part).max() * 1000
    after = (refined or plane).signed_distance(part).max() * 1000
    assert abs(after - before) < 3.0, f"moved {after - before:+.1f} mm on a flat belt"


def test_crown_refit_declines_when_there_is_no_support_to_fit():
    """Too few points must return None so the caller keeps the plane it has."""
    from roboworld_core.segmentation import PlaneModel, refine_support_to_crowns

    plane = PlaneModel(np.array([0.0, 0.0, -1.0]), 0.60, 0.5)
    assert refine_support_to_crowns(np.zeros((10, 3)), plane) is None


# --- colour edge snap ----------------------------------------------------
def _fattened(mask, pixels=3):
    """A depth mask as a stereo camera delivers it: a few pixels too big."""
    from roboworld_core.segmentation import binary_dilate

    return binary_dilate(np.asarray(mask, bool), pixels)


def test_colour_snap_pulls_a_fattened_mask_back_onto_the_part(cfg, intrinsics):
    """The whole point: recover the silhouette depth reports too generously."""
    from roboworld_core.segmentation import snap_mask_to_color_edge

    frame, labels, _ = build_scene(cfg, intrinsics)
    truth = labels == 1
    fat = _fattened(truth, 3)

    snapped = snap_mask_to_color_edge(fat, frame.color, max_shift_px=3)

    def iou(a, b):
        return (a & b).sum() / (a | b).sum()

    assert iou(snapped, truth) > iou(fat, truth), "snap did not improve on the fat mask"
    assert snapped.sum() < fat.sum(), "snap must shrink a fattened mask"


def test_colour_snap_never_grows_the_mask(cfg, intrinsics):
    """Inward-only. Allowed outward it locked onto the conveyor's own edges."""
    from roboworld_core.segmentation import snap_mask_to_color_edge

    frame, labels, _ = build_scene(cfg, intrinsics)
    mask = labels == 1
    snapped = snap_mask_to_color_edge(mask, frame.color, max_shift_px=4)

    assert not (snapped & ~mask).any(), "the boundary moved outward"


def test_colour_snap_respects_its_shift_bound(cfg, intrinsics):
    """A defect on the rim must not be able to carve out the part."""
    from roboworld_core.segmentation import binary_erode, snap_mask_to_color_edge

    frame, labels, _ = build_scene(cfg, intrinsics)
    mask = labels == 1
    snapped = snap_mask_to_color_edge(mask, frame.color, max_shift_px=2)

    # Nothing further in than the bound may be removed.
    assert (binary_erode(mask, 2) & ~snapped).sum() == 0


def test_colour_snap_leaves_a_featureless_boundary_alone(cfg, intrinsics):
    """No edge to snap to -> keep the depth boundary rather than invent one."""
    from roboworld_core.segmentation import snap_mask_to_color_edge

    frame, labels, _ = build_scene(cfg, intrinsics)
    mask = labels == 1
    flat = np.full_like(frame.color, 128)

    assert np.array_equal(snap_mask_to_color_edge(mask, flat, max_shift_px=3), mask)


# --- inspection must not be starved by the size gate ----------------------
def test_size_gate_refuses_for_pose_but_not_for_inspection(cfg, intrinsics):
    """A defect changes the silhouette, so the size gate refuses the very part
    inspection exists to look at.

    Measured on the rig: a part with a foreign object on it read 83 mm wide
    against 62 mm, the size gate refused it, and inspection was handed an empty
    mask -- which scores 1.0 by convention. The NG was right by accident; the
    detector never saw the part.
    """
    from roboworld_core.segmentation import segment_part

    frame, _, _ = build_scene(cfg, intrinsics)
    # Expect a part half the size of the real one: a guaranteed mismatch.
    wrong = np.array([0.100, 0.028, 0.028])

    refused = segment_part(frame, expected_extents_m=wrong, size_tolerance=0.25)
    kept = segment_part(frame, expected_extents_m=wrong, size_tolerance=0.25,
                        refuse_on_size_mismatch=False)

    assert not refused.ok, "pose must still refuse an object of the wrong size"
    assert kept.ok, "inspection must still get pixels to score"
    assert kept.pixel_count == refused.candidates[0].pixel_count
    assert "size mismatch" in kept.reason, "the mismatch must stay visible"


def test_inspection_factory_does_not_refuse_on_size(cfg):
    """The flag has to reach the backends -- they call segment_part directly."""
    from roboworld_core.inspection import _segmentation_kwargs as inspection_kwargs
    from roboworld_core.pose import _segmentation_kwargs as pose_kwargs

    assert inspection_kwargs(cfg, PART_ID)["refuse_on_size_mismatch"] is False
    assert pose_kwargs(cfg, PART_ID).get("refuse_on_size_mismatch", True) is True


# --- symmetry canonicalisation -------------------------------------------
def test_canonical_pose_collapses_an_end_for_end_flip():
    """A static part must not publish a yaw that alternates by 180 degrees.

    Measured on the rig: 15 consecutive frames of a part nobody touched
    alternated between yaw +20.6 and -159.4, fitness 0.97-1.00 on every one.
    Both fits are correct -- the shape repeats -- but a consumer cannot act on
    a number that jumps.
    """
    from roboworld_core.geometry import Pose, euler_to_matrix, quaternion_from_matrix
    from roboworld_core.symmetry import build_group, canonical_pose

    group = build_group([{"axis": [0.0, 0.0, 1.0], "angles_deg": [0.0, 180.0]}])
    assert len(group) == 2

    position = np.array([0.01, -0.02, 0.65])
    upright = euler_to_matrix(0.0, 0.0, np.radians(20.6))
    flipped = upright @ group[1]

    a = canonical_pose(Pose(position, quaternion_from_matrix(upright), "c"), group)
    b = canonical_pose(Pose(position, quaternion_from_matrix(flipped), "c"), group)

    assert np.allclose(a.orientation, b.orientation, atol=1e-9), (
        "the two members of the same equivalence class must publish as one"
    )
    assert np.allclose(a.position, position), "position is untouched by symmetry"


def test_canonical_pose_is_a_noop_without_symmetry():
    """No declared symmetry means every orientation is distinguishable."""
    from roboworld_core.geometry import Pose, euler_to_matrix, quaternion_from_matrix
    from roboworld_core.symmetry import canonical_pose

    pose = Pose(np.zeros(3),
                quaternion_from_matrix(euler_to_matrix(0.1, 0.2, 0.3)), "c")
    assert canonical_pose(pose, []) is pose


def test_pose_backend_gets_the_parts_symmetry(cfg):
    """The group has to reach the backend, or run() has nothing to collapse."""
    from roboworld_core.symmetry import group_for_part

    backend = build_backend(cfg, PART_ID, backend="icp")
    assert len(backend.symmetry_group) == len(group_for_part(cfg, PART_ID))
