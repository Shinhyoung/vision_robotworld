"""Depth-based part segmentation.

Both agents need the same answer to "which pixels are the part?": Inspection
restricts its anomaly statistics to the part surface, Pose needs the object
cloud without the conveyor belt. Keeping one implementation here stops the two
from drifting apart.

Method: RANSAC-fit the dominant plane (the belt), keep points in a height band
above it, take the largest connected component.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .imageops import binary_dilate, binary_erode, connected_components
from .types import CameraIntrinsics, Frame


@dataclass
class PlaneModel:
    """Plane ``n . p + d = 0`` with a unit normal, in camera frame."""

    normal: np.ndarray
    offset: float
    inlier_ratio: float

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) @ self.normal + self.offset


@dataclass
class Candidate:
    """One object found above the plane, measured for identification."""

    mask: np.ndarray
    points: np.ndarray
    #: Bounding extents in the plane frame, sorted descending: length, width,
    #: height. Measured with PCA in-plane, so it does not depend on how the
    #: object happens to be rotated on the surface.
    extents_m: np.ndarray
    #: Worst relative dimension error against the expected part, or ``inf`` when
    #: no expectation was given. Lower is a better match.
    size_error: float = float("inf")

    @property
    def pixel_count(self) -> int:
        return int(self.mask.sum())


@dataclass
class Segmentation:
    """Result of segmenting one frame."""

    mask: np.ndarray  # (H, W) bool, part pixels
    points: np.ndarray  # (N, 3) float64, part points in camera frame
    plane: PlaneModel | None
    pixel_count: int
    #: Every object found above the plane, best match first. Populated whenever
    #: identification ran; useful for diagnosing "why did it pick that one".
    candidates: list[Candidate] = field(default_factory=list)
    #: Why nothing was selected, when ``ok`` is False and a plane was found.
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.pixel_count > 0

    @property
    def selected(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None


def measure_extents(points: np.ndarray, plane: PlaneModel) -> np.ndarray:
    """Length, width and height of a cloud relative to its support plane.

    In-plane extents come from a PCA frame rather than the camera axes, so the
    numbers describe the object itself and not how it is rotated on the surface.
    Returned sorted descending, which makes them directly comparable to a mesh's
    sorted extents without having to solve the axis correspondence first.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return np.zeros(3)

    normal = plane.normal / np.linalg.norm(plane.normal)
    height = float(plane.signed_distance(pts).max())

    centered = pts - pts.mean(axis=0)
    in_plane = centered - np.outer(centered @ normal, normal)
    _, _, vt = np.linalg.svd(in_plane, full_matrices=False)
    projected = in_plane @ vt[:2].T
    span = projected.max(axis=0) - projected.min(axis=0)

    return np.sort(np.array([span[0], span[1], height]))[::-1]


def size_mismatch(measured: np.ndarray, expected: np.ndarray) -> float:
    """Worst relative error between two sorted extent triples."""
    expected = np.sort(np.asarray(expected, dtype=np.float64))[::-1]
    measured = np.asarray(measured, dtype=np.float64)
    scale = np.maximum(expected, 1e-4)
    return float(np.max(np.abs(measured - expected) / scale))


def fit_plane_ransac(
    points: np.ndarray,
    iterations: int = 120,
    distance_threshold: float = 0.006,
    seed: int = 0,
) -> PlaneModel | None:
    """RANSAC plane fit, refined by least squares on the inlier set."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return None

    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    best_count = 0

    for _ in range(max(1, iterations)):
        idx = rng.choice(len(pts), size=3, replace=False)
        a, b, c = pts[idx]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:  # collinear sample
            continue
        normal = normal / norm
        offset = -float(normal @ a)
        inliers = np.abs(pts @ normal + offset) < distance_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    if best_inliers is None or best_count < 3:
        return None

    # Least-squares refit: the plane normal is the smallest singular vector of
    # the centered inlier set.
    inlier_points = pts[best_inliers]
    centroid = inlier_points.mean(axis=0)
    _, _, vt = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vt[-1] / np.linalg.norm(vt[-1])
    offset = -float(normal @ centroid)
    # Orient the normal towards the camera (origin) so "above the plane" is
    # unambiguous: a point closer to the camera gets a positive distance.
    if offset < 0.0:
        normal, offset = -normal, -offset

    inliers = np.abs(pts @ normal + offset) < distance_threshold
    return PlaneModel(normal, offset, float(inliers.mean()))


def segment_part(
    frame: Frame,
    plane_iterations: int = 120,
    plane_distance_threshold_m: float = 0.006,
    min_height_above_plane_m: float = 0.008,
    max_height_above_plane_m: float = 0.120,
    min_cluster_points: int = 300,
    depth_range_m: tuple[float, float] = (0.05, 5.0),
    max_plane_samples: int = 20000,
    seed: int = 0,
    expected_extents_m: np.ndarray | None = None,
    size_tolerance: float = 0.25,
) -> Segmentation:
    """Segment the part from a frame using depth only.

    Colour is deliberately unused: an MC nylon block on a belt is separable by
    geometry, and a colour cue would be confounded by the very surface defects
    Inspection is looking for.

    Args:
        expected_extents_m: the registered part's dimensions. When given, every
            object above the plane is measured and the **best size match** is
            selected -- not merely the largest blob. Anything off by more than
            ``size_tolerance`` is refused, so a hand, a tool or a neighbouring
            part cannot be inspected or pose-estimated as if it were the part.
            ``None`` keeps the largest-blob behaviour.
        size_tolerance: allowed relative error on the worst dimension.
    """
    depth = np.asarray(frame.depth, dtype=np.float64)
    intr: CameraIntrinsics = frame.intrinsics
    valid = np.isfinite(depth) & (depth > depth_range_m[0]) & (depth < depth_range_m[1])
    if not valid.any():
        return Segmentation(np.zeros(depth.shape, bool), np.zeros((0, 3)), None, 0,
                            reason="no valid depth in range")

    rows, cols = np.nonzero(valid)
    z = depth[rows, cols]
    x = (cols + 0.5 - intr.cx) * z / intr.fx
    y = (rows + 0.5 - intr.cy) * z / intr.fy
    points = np.stack([x, y, z], axis=1)

    # Subsample for the RANSAC fit; the plane does not need every pixel.
    rng = np.random.default_rng(seed)
    if len(points) > max_plane_samples:
        sample_idx = rng.choice(len(points), size=max_plane_samples, replace=False)
    else:
        sample_idx = np.arange(len(points))
    plane = fit_plane_ransac(
        points[sample_idx], plane_iterations, plane_distance_threshold_m, seed=seed
    )
    if plane is None:
        return Segmentation(np.zeros(depth.shape, bool), np.zeros((0, 3)), None, 0,
                            reason="no support plane found")

    height_above = plane.signed_distance(points)
    selected = (
        (height_above > min_height_above_plane_m)
        & (height_above < max_height_above_plane_m)
    )

    above = np.zeros(depth.shape, dtype=bool)
    above[rows[selected], cols[selected]] = True
    # Close depth dropouts, then drop the speckle the closing created.
    above = binary_erode(binary_dilate(above, 2), 2)

    labels, sizes = connected_components(above)
    empty = np.zeros(depth.shape, dtype=bool)
    if len(sizes) <= 1:
        return Segmentation(empty, np.zeros((0, 3)), plane, 0,
                            reason="nothing found above the plane")

    candidates: list[Candidate] = []
    for label in range(1, len(sizes)):
        if sizes[label] < min_cluster_points:
            continue
        component = labels == label
        component_points = points[component[rows, cols]]
        candidates.append(
            Candidate(component, component_points, measure_extents(component_points, plane))
        )

    if not candidates:
        return Segmentation(empty, np.zeros((0, 3)), plane, 0,
                            reason=f"no cluster reached {min_cluster_points} points")

    if expected_extents_m is None:
        # No part identity to check against: keep the historical behaviour of
        # taking the biggest thing on the surface.
        candidates.sort(key=lambda c: -c.pixel_count)
    else:
        expected = np.asarray(expected_extents_m, dtype=np.float64)
        for candidate in candidates:
            candidate.size_error = size_mismatch(candidate.extents_m, expected)
        candidates.sort(key=lambda c: c.size_error)
        if candidates[0].size_error > size_tolerance:
            best = candidates[0]
            return Segmentation(
                empty, np.zeros((0, 3)), plane, 0, candidates=candidates,
                reason=(
                    f"no object matches the registered part: closest is "
                    f"{np.round(best.extents_m * 1000, 1)} mm vs expected "
                    f"{np.round(np.sort(expected)[::-1] * 1000, 1)} mm "
                    f"({best.size_error:.0%} off, tolerance {size_tolerance:.0%})"
                ),
            )

    chosen = candidates[0]
    return Segmentation(
        chosen.mask, chosen.points, plane, chosen.pixel_count, candidates=candidates
    )


def segment_from_config(
    frame: Frame, cfg, seed: int = 0, part_id: str | None = None
) -> Segmentation:
    """Segment using the ``pose.segmentation`` config section.

    When ``identify_by_size`` is enabled and the part has geometry, the
    registered part's dimensions are used to pick the right object among
    everything on the surface.
    """
    section = cfg.section("pose.segmentation")
    camera = cfg.section("camera")

    expected = None
    tolerance = float(section.get("size_tolerance", 0.25))
    if section.get("identify_by_size", True):
        expected = expected_extents_for(cfg, part_id or frame.part_id)

    return segment_part(
        frame,
        plane_iterations=int(section.get("plane_ransac_iterations", 120)),
        plane_distance_threshold_m=float(section.get("plane_distance_threshold_m", 0.006)),
        min_height_above_plane_m=float(section.get("min_height_above_plane_m", 0.008)),
        max_height_above_plane_m=float(section.get("max_height_above_plane_m", 0.120)),
        min_cluster_points=int(section.get("min_cluster_points", 300)),
        depth_range_m=(
            float(camera.get("depth_min_m", 0.05)),
            float(camera.get("depth_max_m", 5.0)),
        ),
        seed=seed,
        expected_extents_m=expected,
        size_tolerance=tolerance,
    )


def expected_extents_for(cfg, part_id: str) -> np.ndarray | None:
    """The registered part's extents, or ``None`` if it has no geometry.

    A part registered without a mesh cannot be identified by size, so
    segmentation falls back to the largest object -- the same behaviour as
    before, with the same caveat.
    """
    if not part_id:
        return None
    try:
        from .pose import has_cad, load_part_mesh

        if not has_cad(cfg, part_id):
            return None
        return load_part_mesh(cfg, part_id).extents
    except Exception:  # unknown part, unreadable mesh -- do not block segmentation
        return None
