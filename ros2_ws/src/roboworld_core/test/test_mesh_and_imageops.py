"""Mesh loading, unit conversion and the numpy image primitives."""

from __future__ import annotations

import numpy as np
import pytest

from roboworld_core.imageops import (
    binary_dilate,
    binary_erode,
    connected_components,
    gaussian_blur,
    largest_component,
    resize_nearest,
    sobel_magnitude,
    to_gray,
)
from roboworld_core.mesh_io import load_ply
from roboworld_core.paths import resolve_path


#: Parts registered from camera captures have no mesh (``mesh: ""``); only the
#: CAD parts can be asserted on here.
def cad_parts(cfg) -> dict:
    return {
        part_id: entry
        for part_id, entry in cfg.get("parts").items()
        if entry.get("mesh")
    }


# --- mesh ---------------------------------------------------------------
def test_all_part_meshes_load_and_are_centered(cfg):
    parts = cad_parts(cfg)
    assert parts, "no CAD parts configured"
    for part_id, entry in parts.items():
        mesh = load_ply(resolve_path(entry["mesh"]), units=entry.get("mesh_units", "m"))
        assert len(mesh.vertices) > 0, part_id
        assert len(mesh.faces) > 0, part_id
        assert np.allclose(mesh.centroid, 0.0, atol=1e-9), f"{part_id} is not centered"


def test_part_dimensions_match_the_datasheet(cfg):
    """The three shipped blocks are 200 x 55 x 55 mm; a unit slip shows up here."""
    for part_id in ("guide_block", "spacer_block", "end_stopper"):
        entry = cfg.get(f"parts.{part_id}")
        mesh = load_ply(resolve_path(entry["mesh"]), units=entry.get("mesh_units", "m"))
        extents = np.sort(mesh.extents)[::-1]
        assert extents[0] == pytest.approx(0.200, abs=1e-3), part_id
        assert extents[1] == pytest.approx(0.055, abs=1e-3), part_id
        assert extents[2] == pytest.approx(0.055, abs=1e-3), part_id


def test_millimeter_units_are_scaled(cfg, tmp_path):
    entry = cfg.get("parts.guide_block")
    in_meters = load_ply(resolve_path(entry["mesh"]), units="m")
    as_millimeters = load_ply(resolve_path(entry["mesh"]), units="mm")
    assert np.allclose(as_millimeters.extents, in_meters.extents * 1e-3)


def test_surface_sampling_is_deterministic_and_on_the_mesh(cfg):
    entry = cfg.get("parts.guide_block")
    mesh = load_ply(resolve_path(entry["mesh"]))
    first = mesh.sample_surface(500, seed=3)
    second = mesh.sample_surface(500, seed=3)
    assert np.array_equal(first, second), "same seed must give the same samples"

    lower, upper = mesh.bounds
    assert np.all(first >= lower - 1e-9) and np.all(first <= upper + 1e-9)


# --- image primitives ---------------------------------------------------
def test_to_gray_matches_bt601():
    color = np.zeros((1, 1, 3), dtype=np.uint8)
    color[0, 0] = (255, 0, 0)
    assert to_gray(color)[0, 0] == pytest.approx(255 * 0.299, abs=1e-3)


def test_gaussian_blur_preserves_mean_and_shape():
    rng = np.random.default_rng(0)
    image = rng.normal(50.0, 5.0, (40, 60)).astype(np.float32)
    blurred = gaussian_blur(image, 2.0)
    assert blurred.shape == image.shape
    assert blurred.mean() == pytest.approx(image.mean(), abs=0.5)
    assert blurred.std() < image.std(), "blurring must reduce variance"


def test_gaussian_blur_zero_sigma_is_identity():
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.allclose(gaussian_blur(image, 0.0), image)


def test_sobel_finds_a_vertical_edge():
    image = np.zeros((20, 20), dtype=np.float32)
    image[:, 10:] = 100.0
    magnitude = sobel_magnitude(image)
    assert magnitude[:, 9:11].max() > magnitude[:, 0:5].max()


def test_erode_dilate_are_inverse_on_a_thick_blob():
    mask = np.zeros((30, 30), dtype=bool)
    mask[10:20, 10:20] = True
    assert binary_erode(mask, 1).sum() < mask.sum()
    assert binary_dilate(mask, 1).sum() > mask.sum()
    assert np.array_equal(binary_erode(binary_dilate(mask, 1), 1), mask)


def test_connected_components_separates_and_counts():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:5, 2:5] = True     # 9 px
    mask[10:16, 10:18] = True  # 48 px
    labels, sizes = connected_components(mask)
    assert labels.max() == 2
    assert sorted(sizes[1:].tolist()) == [9, 48]


def test_connected_components_joins_diagonals():
    """8-connectivity: a diagonal touch is one component, not two."""
    mask = np.zeros((6, 6), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    labels, sizes = connected_components(mask)
    assert labels.max() == 1
    assert sizes[1] == 2


def test_connected_components_on_empty_mask():
    labels, sizes = connected_components(np.zeros((5, 5), dtype=bool))
    assert labels.max() == 0
    assert sizes.sum() == 0


def test_largest_component_respects_min_size():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:4, 2:4] = True
    assert largest_component(mask, min_size=100).sum() == 0
    assert largest_component(mask, min_size=2).sum() == 4


def test_resize_nearest_changes_shape_and_keeps_values():
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    resized = resize_nearest(image, (8, 8))
    assert resized.shape == (8, 8)
    assert set(np.unique(resized)).issubset(set(np.unique(image)))
