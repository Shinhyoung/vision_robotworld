"""Depth-based mesh reconstruction (CAD-free part registration).

The measured numbers here are the justification for telling a user that a
reconstructed mesh is good enough for 6D pose; if they regress, that claim in
docs/new_part_registration.md is no longer true.
"""

from __future__ import annotations

import numpy as np
import pytest

from roboworld_core.mesh_io import Mesh, load_ply, save_ply
from roboworld_core.mock_data import MockStation, StationLayout, parts_from_config
from roboworld_core.reconstruct import plane_frame, reconstruct
from roboworld_core.segmentation import segment_from_config

PART_ID = "guide_block"
#: The three blocks are 200 x 55 x 55 mm.
TRUE_EXTENTS_MM = (200.0, 55.0, 55.0)
#: 2 mm grid quantisation plus depth noise; measured errors are ~+2/+1/+1 mm.
EXTENT_TOLERANCE_MM = 5.0


@pytest.fixture(scope="module")
def recon_cfg(cfg):
    """Reconstruction never identifies by size -- it is defining the size.

    Nor by station ROI: views are shot wherever the rig is, not on the belt.
    """
    return cfg.merged_with({"pose": {"segmentation": {
        "identify_by_size": False,
        "station_roi": {"enabled": False},
    }}})


@pytest.fixture(scope="module")
def flat_station(cfg, intrinsics):
    """A part resting flat, as it would on a table -- no belt tilt."""
    return MockStation(
        parts_from_config(cfg), intrinsics,
        layout=StationLayout(tilt_jitter_deg=0.0),
    )


@pytest.fixture(scope="module")
def reconstruction(recon_cfg, flat_station):
    pose = flat_station.sample_pose(np.random.default_rng(5))
    views = [
        segment_from_config(flat_station.render_frame(PART_ID, pose, seed=200 + i), recon_cfg)
        for i in range(4)
    ]
    return reconstruct(views)


def test_reconstructed_dimensions_match_the_real_part(reconstruction):
    _, report = reconstruction
    for measured, expected in zip(report.extents_mm, TRUE_EXTENTS_MM, strict=True):
        assert abs(measured - expected) < EXTENT_TOLERANCE_MM, report.extents_mm


def test_reconstruction_is_a_usable_mesh(reconstruction):
    mesh, _ = reconstruction
    assert len(mesh.faces) > 100
    assert np.allclose(mesh.centroid, 0.0, atol=1e-9), "must use the model-frame origin"
    # Long axis on +x, matching the parts.yaml convention.
    assert mesh.extents[0] == max(mesh.extents)
    assert len(mesh.sample_surface(500, seed=0)) == 500


def test_flat_part_reports_no_tilt(reconstruction):
    _, report = reconstruction
    assert report.top_tilt_deg < 1.0
    assert not any("tilted" in w for w in report.warnings)


def test_tilted_part_is_flagged(recon_cfg, intrinsics):
    """A tilt baked into the mesh would silently produce a wedge."""
    station = MockStation(
        parts_from_config(recon_cfg), intrinsics,
        layout=StationLayout(tilt_jitter_deg=0.0),
    )
    pose = station.sample_pose(np.random.default_rng(1))
    # Tip the part about its short axis by ~6 degrees.
    from roboworld_core.geometry import (
        Pose,
        euler_to_matrix,
        matrix_from_quaternion,
        quaternion_from_matrix,
    )

    rotation = euler_to_matrix(0.0, np.radians(6.0), 0.0) @ matrix_from_quaternion(
        pose.orientation
    )
    tilted = Pose(pose.position, quaternion_from_matrix(rotation), pose.frame_id)

    views = [
        segment_from_config(station.render_frame(PART_ID, tilted, seed=300 + i), recon_cfg)
        for i in range(3)
    ]
    _, report = reconstruct(views)
    assert report.top_tilt_deg > 2.0
    assert any("tilted" in w for w in report.warnings)


def test_reconstruction_needs_a_visible_part(good_frame):
    from roboworld_core.segmentation import Segmentation

    empty = Segmentation(
        np.zeros(good_frame.depth.shape, bool), np.zeros((0, 3)), None, 0
    )
    with pytest.raises(ValueError, match="no usable segmentation"):
        reconstruct([empty])


def test_plane_frame_is_right_handed_and_on_the_plane(recon_cfg, flat_station):
    pose = flat_station.sample_pose(np.random.default_rng(7))
    segmentation = segment_from_config(
        flat_station.render_frame(PART_ID, pose, seed=11), recon_cfg
    )
    transform = plane_frame(segmentation)
    rotation = transform[:3, :3]

    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(rotation), 1.0)
    # +z of the frame points back towards the camera at the origin.
    assert rotation[2, 2] < 0.0


def test_ply_roundtrip(tmp_path, reconstruction):
    mesh, _ = reconstruction
    path = save_ply(mesh, tmp_path / "recon.ply")
    restored = load_ply(path, center=False)

    assert len(restored.faces) == len(mesh.faces)
    assert np.allclose(restored.extents, mesh.extents, atol=1e-5)


def test_ply_roundtrip_in_millimeters(tmp_path):
    mesh = Mesh(
        vertices=np.array([[0.0, 0, 0], [0.1, 0, 0], [0.0, 0.05, 0]]),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
    )
    path = save_ply(mesh, tmp_path / "mm.ply", units="mm")
    restored = load_ply(path, units="mm", center=False)
    assert np.allclose(restored.vertices, mesh.vertices, atol=1e-6)


# --- levelling -----------------------------------------------------------
def tilted_views(recon_cfg, station, degrees: float, seed: int = 1):
    """Views of the part rocked about its long axis by ``degrees``."""
    from roboworld_core.geometry import (
        Pose,
        euler_to_matrix,
        matrix_from_quaternion,
        quaternion_from_matrix,
    )

    pose = station.sample_pose(np.random.default_rng(seed))
    rotation = euler_to_matrix(np.radians(degrees), 0.0, 0.0) @ matrix_from_quaternion(
        pose.orientation
    )
    tilted = Pose(pose.position, quaternion_from_matrix(rotation), pose.frame_id)
    return [
        segment_from_config(station.render_frame(PART_ID, tilted, seed=700 + i), recon_cfg)
        for i in range(3)
    ]


def test_levelling_removes_the_tilt(recon_cfg, flat_station):
    views = tilted_views(recon_cfg, flat_station, 5.0)
    _, before = reconstruct(views)
    _, after = reconstruct(views, level_top=True)

    assert before.top_tilt_deg > 2.0
    assert after.top_tilt_deg < 1.0
    assert after.tilt_removed_deg > 2.0
    assert not any("tilted" in w for w in after.warnings)


def test_levelling_preserves_the_part_height(recon_cfg, flat_station):
    """Levelling must un-tilt the part, not slide it into the table.

    An earlier version subtracted the fitted plane including its offset, which
    dropped a 53 mm part to 37 mm while reporting a perfect tilt of 0.
    """
    views = tilted_views(recon_cfg, flat_station, 5.0)
    _, before = reconstruct(views)
    _, after = reconstruct(views, level_top=True)

    # The height may shrink by roughly the tilt amplitude, no more.
    assert after.extents_mm[2] > before.extents_mm[2] - 6.0
    assert after.extents_mm[2] == pytest.approx(TRUE_EXTENTS_MM[2], abs=6.0)
    # Footprint is untouched by levelling.
    assert after.extents_mm[0] == pytest.approx(before.extents_mm[0], abs=1.0)


def test_levelling_is_a_noop_on_a_flat_part(reconstruction, recon_cfg, flat_station):
    """A part already sitting flat must not be distorted by asking to level it."""
    mesh, report = reconstruction
    pose = flat_station.sample_pose(np.random.default_rng(5))
    views = [
        segment_from_config(flat_station.render_frame(PART_ID, pose, seed=200 + i), recon_cfg)
        for i in range(4)
    ]
    _, levelled = reconstruct(views, level_top=True)

    for measured, original in zip(levelled.extents_mm, report.extents_mm, strict=True):
        assert measured == pytest.approx(original, abs=1.5)


def test_pose_overlay_lands_on_the_part(cfg, flat_station):
    """Re-projecting the model at the estimated pose must cover the real part.

    Fitness and RMSE only say two clouds agree; they cannot catch a pose locked
    onto the wrong face or a mesh built at the wrong scale. Overlap between the
    rendered silhouette and the segmented part can.
    """
    from roboworld_core.pose import build_backend
    from roboworld_core.render import Renderer, RenderItem
    from roboworld_core.viz import pose_overlay

    frame = flat_station.sample_frame(PART_ID, seed=808, sequence=1)
    segmentation = segment_from_config(frame, cfg)
    estimate = build_backend(cfg, PART_ID, backend="icp").run(frame)
    assert estimate.valid

    from roboworld_core.pose import load_part_mesh

    mesh = load_part_mesh(cfg, PART_ID)
    rendered = Renderer(frame.intrinsics).render([
        RenderItem(mesh.vertices, mesh.faces,
                   np.array([255, 220, 40], np.uint8), estimate.pose.as_matrix(), 1)
    ])
    silhouette = rendered.mask == 1
    intersection = float((silhouette & segmentation.mask).sum())
    union = float((silhouette | segmentation.mask).sum())
    assert intersection / union > 0.7, f"IoU {intersection / union:.3f}"

    image, coverage = pose_overlay(frame.color, mesh, estimate.pose, frame.intrinsics)
    assert image.shape == frame.color.shape
    assert coverage > 0.0
