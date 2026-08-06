"""Depth-based part segmentation.

Both agents need the same answer to "which pixels are the part?": Inspection
restricts its anomaly statistics to the part surface, Pose needs the object
cloud without the conveyor belt. Keeping one implementation here stops the two
from drifting apart.

Method: RANSAC-fit the dominant plane (the belt), keep points in a height band
above it, take the largest connected component.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .config import ConfigError
from .imageops import (
    binary_dilate,
    binary_erode,
    connected_components,
    gaussian_blur,
    sobel_magnitude,
    to_gray,
)
from .types import CameraIntrinsics, Frame


@dataclass
class StationRoi:
    """The volume a part must occupy to count as "at the station".

    An axis-aligned box in the **camera optical frame**, in meters. Objects
    whose centre falls outside it are not selectable: with several parts on the
    belt, dimension matching alone cannot tell the one at the stop position from
    an identical one further down, and measurement showed it picking the wrong
    one at 200 mm spacing.

    A box rather than an image rectangle so the numbers come straight off the
    conveyor drawing and survive a resolution or lens change. Note this is *not*
    robust to moving the camera -- both forms are expressed in the camera frame,
    so both break together. A truly invariant ROI needs a station frame and TF,
    which is still open with the robot department (ICD section 4).
    """

    center_m: np.ndarray
    half_extents_m: np.ndarray

    def __post_init__(self) -> None:
        self.center_m = self._as_vector(self.center_m, "center_m")
        self.half_extents_m = self._as_vector(self.half_extents_m, "half_extents_m")
        if np.any(self.half_extents_m <= 0.0):
            raise ValueError(
                f"station ROI half extents must be positive, got {self.half_extents_m}"
            )

    @staticmethod
    def _as_vector(value, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64).ravel()
        if vector.size != 3:
            raise ValueError(f"station ROI {name} must have 3 elements, got {vector.tolist()}")
        return vector

    def offset_m(self, point: np.ndarray) -> float:
        """Distance from ``point`` to the box surface; ``0.0`` when inside.

        Reported rather than a bare bool so an operator can tell "just outside"
        from "the part is nowhere near the station".
        """
        outside = np.abs(np.asarray(point, dtype=np.float64) - self.center_m)
        outside = outside - self.half_extents_m
        return float(np.linalg.norm(np.maximum(outside, 0.0)))

    def contains(self, point: np.ndarray) -> bool:
        return self.offset_m(point) <= 0.0

    def corners(self) -> np.ndarray:
        """The 8 box corners, for drawing it in a viewer."""
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        return self.center_m + signs * self.half_extents_m


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
    #: Distance from :attr:`center_m` to the station ROI box, ``0.0`` when inside
    #: it or when no ROI is configured.
    roi_offset_m: float = 0.0

    @property
    def pixel_count(self) -> int:
        return int(self.mask.sum())

    @property
    def center_m(self) -> np.ndarray:
        """Centroid of the observed points, in camera frame.

        Depth only sees the faces turned towards the camera, so this sits in
        front of the object's true centre by roughly half its thickness. That is
        fine for deciding *which* object is at the station -- the ROI is sized in
        hundreds of millimeters -- but it is not a position estimate; the pose
        backends compute that properly.
        """
        return self.points.mean(axis=0) if len(self.points) else np.zeros(3)

    @property
    def in_roi(self) -> bool:
        return self.roi_offset_m <= 0.0


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


def part_crop_box(
    mask: np.ndarray, margin: float = 0.15, square: bool = True
) -> tuple[int, int, int, int]:
    """Window around the part, as ``(r0, r1, c0, c1)``.

    A detector that resizes the whole frame to its input spends almost all of
    that input on belt. Measured on the mock station: the part covers 1.8 % of a
    640x480 frame, so at 256x256 it survives as ~1170 px and a 15 px defect
    becomes 6 px. EfficientAD missed 24/24 defects that way while the CPU
    reference, which works at native resolution inside the mask, missed none.

    ``square`` keeps the aspect ratio so a 200x50 mm block is not stretched into
    the square input; the margin leaves context around the edge, which an
    anomaly detector needs to see the boundary as normal rather than as an edge
    of the image.

    Training and inference MUST use this same function -- a crop that differs
    between them shifts the whole score distribution and silently invalidates
    the calibrated anchor.
    """
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    if not mask.any():
        return 0, height, 0, width

    rows, cols = np.nonzero(mask)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1

    pad_r = int(round((r1 - r0) * margin))
    pad_c = int(round((c1 - c0) * margin))
    r0, r1 = r0 - pad_r, r1 + pad_r
    c0, c1 = c0 - pad_c, c1 + pad_c

    if square:
        side = max(r1 - r0, c1 - c0)
        centre_r, centre_c = (r0 + r1) // 2, (c0 + c1) // 2
        r0, r1 = centre_r - side // 2, centre_r - side // 2 + side
        c0, c1 = centre_c - side // 2, centre_c - side // 2 + side

    # Slide the window back inside the image rather than clipping it, so the
    # aspect ratio survives a part near the border.
    r0, r1 = max(r0, 0), min(r1, height)
    c0, c1 = max(c0, 0), min(c1, width)
    return r0, r1, c0, c1


def color_edge_gradient(color: np.ndarray, blur_sigma_px: float = 1.0) -> np.ndarray:
    """Edge strength of a colour frame, for :func:`snap_mask_to_color_edge`.

    Split out because it depends on the frame alone: one frame holding several
    candidates needs it once, not once per candidate.
    """
    return sobel_magnitude(gaussian_blur(to_gray(color), blur_sigma_px))


def snap_mask_to_color_edge(
    mask: np.ndarray,
    color: np.ndarray,
    max_shift_px: int = 3,
    min_gradient: float = 8.0,
    blur_sigma_px: float = 1.0,
    gradient: np.ndarray | None = None,
) -> np.ndarray:
    """Move a depth mask's boundary onto the nearest strong colour edge.

    A stereo camera fattens: the matcher paints object depth onto a band of
    background pixels around the silhouette. Measured on a D455 capture of a
    part known to be 49 mm wide, the mask read ~60 mm, and neither averaging
    (the skirt is stable across 10 frames) nor resolution (848x480 was worse)
    removes it -- the depth boundary is simply in the wrong place. The colour
    image has it in the right place.

    **Colour never decides membership.** Depth already chose which pixels are
    part; this only relocates an existing boundary, by at most
    ``max_shift_px``, and only towards a gradient stronger than
    ``min_gradient``. Where no such edge exists the depth boundary stands. That
    bound is what stops a stain or a chip touching the rim from carving a piece
    out of the part -- which is why segmentation otherwise refuses colour: the
    defects Inspection hunts for are colour features too.

    The search runs **inward only**. Fattening can only ever add pixels, so
    there is no reason to let the boundary travel outward -- and every reason
    not to: allowed both ways on a roller conveyor it locked onto the rollers'
    own specular edges and made the part 23 % wider instead of 9 % (measured,
    the same 10 frames). Inward-only took the worst axis from 5.1 % to 2.7 %.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or max_shift_px < 1:
        return mask

    if gradient is None:
        gradient = color_edge_gradient(color, blur_sigma_px)

    # Work in the candidate's own bounding box. Everything below is either
    # morphology or a blur over the array it is given, and a part covers a few
    # percent of a 640x480 frame -- doing it full-size for every candidate was
    # most of what remained of the cost.
    rows_any, cols_any = np.nonzero(mask)
    pad = max_shift_px + 3
    r0 = max(int(rows_any.min()) - pad, 0)
    r1 = min(int(rows_any.max()) + pad + 1, mask.shape[0])
    c0 = max(int(cols_any.min()) - pad, 0)
    c1 = min(int(cols_any.max()) + pad + 1, mask.shape[1])
    window = (slice(r0, r1), slice(c0, c1))

    local = mask[window]
    gradient = gradient[window]
    height, width = local.shape
    full, mask = mask, local

    # Outward normal from a smoothed indicator: the field falls off outward, so
    # the negated gradient points out of the object.
    field = gaussian_blur(mask.astype(np.float32), max(blur_sigma_px, 1.5))
    dy, dx = np.gradient(field.astype(np.float64))
    magnitude = np.hypot(dx, dy)

    edge = mask & ~binary_erode(mask, 1)
    rows, cols = np.nonzero(edge & (magnitude > 1e-6))
    if len(rows) == 0:
        return full

    normal_x = -dx[rows, cols] / magnitude[rows, cols]
    normal_y = -dy[rows, cols] / magnitude[rows, cols]

    offsets = np.arange(-max_shift_px, 1)
    sample_r = np.clip(
        np.rint(rows[:, None] + normal_y[:, None] * offsets).astype(np.int64), 0, height - 1
    )
    sample_c = np.clip(
        np.rint(cols[:, None] + normal_x[:, None] * offsets).astype(np.int64), 0, width - 1
    )
    profile = gradient[sample_r, sample_c]

    chosen = profile.argmax(axis=1)
    strong = profile[np.arange(len(chosen)), chosen] >= min_gradient
    # No edge worth moving to -> stay put (offset 0 is the last index).
    chosen = np.where(strong, chosen, len(offsets) - 1)

    core = binary_erode(mask, max_shift_px)
    if not core.any():
        return full  # too thin to reshape safely

    refined = core.copy()
    inside = offsets[None, :] <= offsets[chosen][:, None]
    refined[sample_r[inside], sample_c[inside]] = True
    # The rays are discrete, so close the pinholes they leave between them.
    refined = binary_erode(binary_dilate(refined, 1), 1)
    # Never let the boundary travel further than it was allowed to.
    refined &= binary_dilate(mask, max_shift_px)

    out = np.zeros_like(full)
    out[window] = refined
    return out


def refine_support_to_crowns(
    points: np.ndarray,
    plane: PlaneModel,
    band_m: float = 0.035,
    percentile: float = 95.0,
    slab_m: float = 0.003,
    seed: int = 0,
) -> PlaneModel | None:
    """Lift the support plane onto the crowns of a ribbed surface.

    A roller conveyor is not a plane. The cloud holds both the roller crowns and
    the gaps between them, and RANSAC fits that mixture, so the plane lands
    *below* the surface the part actually rests on. Every height is then
    measured from too low a datum.

    Measured on a D455 capture of a part known to be 51 mm tall: the plane sat
    ~20 mm low and the part read 71 mm (+39 %). Refitting to the top slice of
    the support band reads 50 mm (-2.6 %). Widening
    ``plane_distance_threshold_m`` does not help -- swept 3 to 30 mm, the height
    stayed 71-73 mm, because the problem is which surface is being fitted, not
    how tolerantly.

    Returns ``None`` when there is too little support to refit, so the caller
    keeps the plane it already has rather than trusting a fit to a handful of
    points.
    """
    pts = np.asarray(points, dtype=np.float64)
    distance = plane.signed_distance(pts)
    # The support band: near the fitted plane, either side. The part itself
    # stands well clear of this, so it does not vote on where the surface is.
    band = pts[np.abs(distance) < band_m]
    if len(band) < 200:
        return None

    band_distance = plane.signed_distance(band)
    cut = np.percentile(band_distance, percentile) - slab_m
    crowns = band[band_distance >= cut]
    if len(crowns) < 50:
        return None

    return fit_plane_ransac(crowns, 120, max(slab_m, 0.002), seed=seed)


#: Prefix marking "the best candidate is the wrong size, taken anyway".
#: :func:`segment_part` turns it into a refusal for callers that want one.
_SIZE_MISMATCH_NOTE = "selected despite size mismatch: "

#: One frame's geometry, keyed by the frame object and the settings that shape
#: it. Inspection and Pose segment the SAME frame with settings that differ only
#: in whether a size mismatch is fatal, so without this the plane fit, the
#: clustering and the colour snap all run twice per cycle -- measured at 101 ms
#: a time, a third of the whole tact.
#:
#: Two entries, and the frame is held by identity: a cycle uses one frame, and
#: comparing with ``is`` cannot collide the way ``id()`` can after a free. Sized
#: for the sequential pipeline; it is a memo, not a store.
_GEOMETRY_CACHE: list[tuple] = []
_GEOMETRY_CACHE_SIZE = 2


def _segment_permissive(
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
    station_roi: StationRoi | None = None,
    support_crowns: bool = False,
    crown_percentile: float = 95.0,
    crown_slab_m: float = 0.003,
    crown_band_m: float = 0.035,
    color_snap_px: int = 0,
    color_snap_min_gradient: float = 8.0,
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
        station_roi: volume a candidate's centre must lie in to be selectable.
            Dimension matching answers "is this the right part"; it cannot
            answer "is this the one that stopped in front of the camera",
            because the next identical part on the belt matches just as well.
            ``None`` disables the check.
        support_crowns: refit the support plane onto the crowns of a ribbed
            surface -- see :func:`refine_support_to_crowns`. Leave off for a
            flat belt, where the plane already is the support surface.
        color_snap_px: pull each candidate's boundary in onto the colour edge by
            at most this many pixels -- see :func:`snap_mask_to_color_edge`.
            ``0`` disables it and the depth boundary is used as-is.
        refuse_on_size_mismatch: return nothing when the best candidate is
            outside ``size_tolerance``. True for pose -- registering the model
            against a hand gives a confidently wrong pick. False for inspection,
            which still ranks candidates by size but must not be handed an empty
            mask just because the defect changed the silhouette; the reason
            records that it was selected anyway.
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

    if support_crowns:
        # Falls back to the plane already fitted when there is too little
        # support to refit -- a bad datum is worse than a low one.
        refined = refine_support_to_crowns(
            points[sample_idx], plane, crown_band_m, crown_percentile, crown_slab_m,
            seed=seed,
        )
        if refined is not None:
            plane = refined

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

    # Depends on the frame, not on any candidate -- build it once.
    snap_gradient = color_edge_gradient(frame.color) if color_snap_px > 0 else None

    candidates: list[Candidate] = []
    for label in range(1, len(sizes)):
        if sizes[label] < min_cluster_points:
            continue
        component = labels == label
        if color_snap_px > 0:
            snapped = snap_mask_to_color_edge(
                component, frame.color, color_snap_px, color_snap_min_gradient,
                gradient=snap_gradient,
            )
            # Only take it if something is left; a boundary this thin is not
            # worth reshaping.
            if snapped.sum() >= min_cluster_points:
                component = snapped
        component_points = points[component[rows, cols]]
        candidate = Candidate(
            component, component_points, measure_extents(component_points, plane)
        )
        if station_roi is not None:
            candidate.roi_offset_m = station_roi.offset_m(candidate.center_m)
        candidates.append(candidate)

    if not candidates:
        return Segmentation(empty, np.zeros((0, 3)), plane, 0,
                            reason=f"no cluster reached {min_cluster_points} points")

    # Ranking is two-tier: anything outside the station volume sorts last
    # regardless of how well it matches, so it can never be selected, but it
    # stays in ``candidates`` because "the part is 200 mm too far down the belt"
    # is exactly what an operator needs to see.
    if expected_extents_m is None:
        # No part identity to check against: keep the historical behaviour of
        # taking the biggest thing on the surface.
        candidates.sort(key=lambda c: (not c.in_roi, -c.pixel_count))
    else:
        expected = np.asarray(expected_extents_m, dtype=np.float64)
        for candidate in candidates:
            candidate.size_error = size_mismatch(candidate.extents_m, expected)
        candidates.sort(key=lambda c: (not c.in_roi, c.size_error))

    if station_roi is not None and not candidates[0].in_roi:
        # Distinct from a size failure, and the line acts on it differently: no
        # part has reached the station yet, so waiting is the correct response.
        best = candidates[0]
        return Segmentation(
            empty, np.zeros((0, 3)), plane, 0, candidates=candidates,
            reason=(
                f"no object inside the station volume: best of "
                f"{len(candidates)} is {best.roi_offset_m * 1000:.0f} mm "
                f"outside, centred at {np.round(best.center_m * 1000, 1)} mm"
            ),
        )

    note = ""
    if expected_extents_m is not None and candidates[0].size_error > size_tolerance:
        best = candidates[0]
        mismatch = (
            f"closest is {np.round(best.extents_m * 1000, 1)} mm vs expected "
            f"{np.round(np.sort(expected)[::-1] * 1000, 1)} mm "
            f"({best.size_error:.0%} off, tolerance {size_tolerance:.0%})"
        )
        # Selected regardless, and said so; :func:`segment_part` turns this
        # note into a refusal for the callers that want one.
        note = f"{_SIZE_MISMATCH_NOTE}{mismatch}"

    chosen = candidates[0]
    return Segmentation(
        chosen.mask, chosen.points, plane, chosen.pixel_count,
        candidates=candidates, reason=note,
    )


def segment_part(
    frame: Frame, *, refuse_on_size_mismatch: bool = True, **kwargs
) -> Segmentation:
    """Segment the part, reusing this frame's geometry across callers.

    ``refuse_on_size_mismatch`` is the only thing Inspection and Pose disagree
    on, and it is decided *after* all the expensive work: Pose refuses an object
    of the wrong size because registering the model against a hand yields a
    confidently wrong pick, while Inspection must still be handed pixels --
    a defect changes the silhouette, so refusing there means scoring an empty
    mask, which reads 1.0 by convention and is an NG that never saw the part.

    Everything before that decision is identical, so it is computed once per
    frame and shared. See :func:`_segment_permissive` for the parameters.
    """
    key = tuple(sorted(
        (name, _hashable(value)) for name, value in kwargs.items()
    ))
    result = None
    for cached_frame, cached_key, cached_result in _GEOMETRY_CACHE:
        if cached_frame is frame and cached_key == key:
            result = cached_result
            break
    if result is None:
        result = _segment_permissive(frame, **kwargs)
        if _GEOMETRY_CACHE_SIZE > 0:
            _GEOMETRY_CACHE.append((frame, key, result))
            # Guarded: `del lst[:-0]` is `del lst[:0]`, which trims nothing and
            # turns the memo into an unbounded store holding every frame alive.
            del _GEOMETRY_CACHE[:-_GEOMETRY_CACHE_SIZE]

    if refuse_on_size_mismatch and result.reason.startswith(_SIZE_MISMATCH_NOTE):
        detail = result.reason[len(_SIZE_MISMATCH_NOTE):]
        return Segmentation(
            np.zeros(frame.depth.shape, dtype=bool), np.zeros((0, 3)),
            result.plane, 0, candidates=result.candidates,
            reason=f"no object matches the registered part: {detail}",
        )
    return result


def _hashable(value):
    """Cache-key form of a segmentation setting.

    Arrays and the ROI object are not hashable, and two equal arrays are not the
    same object; reduce them to their contents so an unchanged setting hits the
    cache instead of silently recomputing.
    """
    if isinstance(value, np.ndarray):
        return value.tobytes()
    if isinstance(value, StationRoi):
        return (value.center_m.tobytes(), value.half_extents_m.tobytes())
    return value


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
        station_roi=station_roi_from_config(cfg),
        **_crown_kwargs(section),
    )


def _crown_kwargs(section) -> dict:
    """Support-surface and boundary-refinement settings."""
    snap = section.get("color_edge_snap", None)
    extra = {}
    if isinstance(snap, Mapping) and snap.get("enabled", False):
        extra = {
            "color_snap_px": int(snap.get("max_shift_px", 3)),
            "color_snap_min_gradient": float(snap.get("min_gradient", 8.0)),
        }
    surface = section.get("support_surface", None)
    if not isinstance(surface, Mapping):
        return extra
    return {
        **extra,
        "support_crowns": str(surface.get("mode", "plane")).lower() == "crowns",
        "crown_percentile": float(surface.get("crown_percentile", 95.0)),
        "crown_slab_m": float(surface.get("crown_slab_m", 0.003)),
        "crown_band_m": float(surface.get("band_m", 0.035)),
    }


def station_roi_from_config(cfg) -> StationRoi | None:
    """The station volume from ``pose.segmentation.station_roi``, if enabled.

    Absent or ``enabled: false`` returns ``None``, which restores the
    pre-ROI behaviour of considering every object on the belt.
    """
    section = cfg.section("pose.segmentation")
    roi = section.get("station_roi", None)
    if not isinstance(roi, Mapping) or not roi.get("enabled", False):
        return None
    try:
        return StationRoi(roi["center_m"], roi["half_extents_m"])
    except KeyError as missing:
        raise ConfigError(
            f"pose.segmentation.station_roi is enabled but {missing} is missing "
            f"(source: {cfg.source})"
        ) from None


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
