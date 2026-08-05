"""Reconstruct a part mesh from D455 depth, with no CAD and no texture.

Why depth and not photos
------------------------
Photogrammetry matches image features between views. The parts here are
machined, single-colour and matte, so there are almost no features to match and
photo-based reconstruction degrades badly. Depth does not care about texture at
all: the stereo/IR module measures geometry directly, and gives **metric scale**
for free -- which monocular photo reconstruction cannot.

Method: height-map extrusion
----------------------------
The part rests on a flat surface, which segmentation already finds. Working in
that plane's frame:

1. project the segmented points into the plane frame -> ``(u, v, height)``
2. rasterise into a regular grid, taking a high percentile of the heights in
   each cell (robust to the depth speckle a plain max would latch onto)
3. emit a watertight mesh: top surface at the measured height, bottom at the
   plane, vertical walls around the silhouette

Assumptions, and when they break
--------------------------------
* **The bottom face is flat.** True whenever the part is resting stably on the
  plane -- it is the face in contact.
* **No undercuts.** A single top-down view cannot see beneath an overhang, so an
  overhanging part is reconstructed as a solid column below the overhang.
* **No hidden internal geometry.** Through-holes appear only if they are visible
  from above; blind features on the underside are lost.

For a prismatic machined block -- which most of them are -- this reconstructs
the true shape. For a complex organic part it will not. :func:`reconstruct`
returns the statistics needed to judge which case you are in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import orthonormalize
from .mesh_io import Mesh
from .segmentation import Segmentation


@dataclass
class ReconstructionReport:
    """Quality signals for a reconstruction; print these, do not hide them."""

    cells_filled: int = 0
    cells_interpolated: int = 0
    height_min_mm: float = 0.0
    height_max_mm: float = 0.0
    extents_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    views_merged: int = 1
    points_used: int = 0
    #: Tilt of the part's top surface relative to the support plane. A part
    #: resting flat reads ~0; anything larger means the tilt is being baked into
    #: the mesh as a wedge.
    top_tilt_deg: float = 0.0
    #: How far the per-view part centroids scatter in the common frame, in mm.
    #: Merging assumes camera and part are both static; if either moved, the
    #: views land in different places and the mesh smears out. This is the
    #: number that catches it.
    view_centroid_spread_mm: float = 0.0
    #: Spread of each view's in-plane footprint diagonal, in mm.
    view_extent_spread_mm: float = 0.0
    #: Tilt removed by ``level_top``; 0 when levelling was not requested.
    tilt_removed_deg: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def interpolated_fraction(self) -> float:
        total = self.cells_filled + self.cells_interpolated
        return self.cells_interpolated / total if total else 0.0


def plane_frame(segmentation: Segmentation) -> np.ndarray:
    """Build ``T_camera_plane``: a frame on the support plane under the part.

    +z is the plane normal towards the camera, +x is the part's dominant in-plane
    direction (its long axis), and the origin sits under the part centroid.
    """
    points = segmentation.points
    centroid = points.mean(axis=0)

    normal = np.asarray(segmentation.plane.normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    if normal[2] > 0.0:  # point it back towards the camera at the origin
        normal = -normal

    centered = points - centroid
    in_plane = centered - np.outer(centered @ normal, normal)
    _, _, vt = np.linalg.svd(in_plane, full_matrices=False)
    long_axis = vt[0] - (vt[0] @ normal) * normal
    norm = np.linalg.norm(long_axis)
    long_axis = long_axis / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])

    third = np.cross(normal, long_axis)
    rotation = orthonormalize(np.stack([long_axis, third, normal], axis=1))

    # Origin: the part centroid dropped onto the plane.
    height = float(segmentation.plane.signed_distance(centroid[None, :])[0])
    origin = centroid - normal * height

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = origin
    return transform


def _height_grid(
    local: np.ndarray,
    cell_size: float,
    percentile: float,
    min_points_per_cell: int,
    view_ids: np.ndarray | None = None,
    min_views: int = 1,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Rasterise ``(u, v, h)`` points into a height grid.

    ``min_views`` requires a cell to be seen by that many distinct views before
    it counts as part. Pooling views is a union, so without it every extra view
    can only push the footprint outward: measured on 10 real views of a part
    known to be 230 x 49 x 51 mm, one view gave 226 x 54 x 51 and ten gave
    230 x 58 x 59. The per-cell height percentile does average noise down, as
    the module docstring says -- but the *extent* does the opposite.
    """
    u_min, v_min = local[:, 0].min(), local[:, 1].min()
    cols = np.floor((local[:, 0] - u_min) / cell_size).astype(np.int64)
    rows = np.floor((local[:, 1] - v_min) / cell_size).astype(np.int64)
    height_cells = int(rows.max()) + 1
    width_cells = int(cols.max()) + 1

    flat_index = rows * width_cells + cols
    order = np.argsort(flat_index)
    flat_sorted = flat_index[order]
    heights_sorted = local[order, 2]

    views_sorted = view_ids[order] if view_ids is not None else None

    heights = np.full(height_cells * width_cells, np.nan)
    counts = np.zeros(height_cells * width_cells, dtype=np.int64)

    # One pass over contiguous runs of the same cell.
    boundaries = np.flatnonzero(np.diff(flat_sorted)) + 1
    for start, stop in zip(
        np.concatenate(([0], boundaries)),
        np.concatenate((boundaries, [len(flat_sorted)])),
        strict=True,
    ):
        count = stop - start
        if count < min_points_per_cell:
            continue
        if (
            views_sorted is not None
            and min_views > 1
            and len(np.unique(views_sorted[start:stop])) < min_views
        ):
            continue
        cell = flat_sorted[start]
        heights[cell] = np.percentile(heights_sorted[start:stop], percentile)
        counts[cell] = count

    return (
        heights.reshape(height_cells, width_cells),
        counts.reshape(height_cells, width_cells),
        (float(u_min), float(v_min)),
    )


def _fill_holes(heights: np.ndarray, max_passes: int = 6) -> tuple[np.ndarray, int]:
    """Fill NaN cells from their filled neighbours (depth dropouts, IR speckle)."""
    filled = heights.copy()
    interpolated = 0
    for _ in range(max_passes):
        holes = np.isnan(filled)
        if not holes.any():
            break
        padded = np.pad(filled, 1, constant_values=np.nan)
        neighbours = np.stack([
            padded[:-2, 1:-1], padded[2:, 1:-1],
            padded[1:-1, :-2], padded[1:-1, 2:],
        ])
        # A hole with no filled neighbour yields an all-NaN slice; that is the
        # normal "not fillable yet" case, not an error worth warning about.
        finite = np.isfinite(neighbours)
        with np.errstate(invalid="ignore"):
            candidate = np.where(
                finite.any(axis=0),
                np.nansum(np.where(finite, neighbours, 0.0), axis=0)
                / np.maximum(finite.sum(axis=0), 1),
                np.nan,
            )
        fillable = holes & np.isfinite(candidate)
        if not fillable.any():
            break
        filled[fillable] = candidate[fillable]
        interpolated += int(fillable.sum())
    return filled, interpolated


def _mesh_from_grid(
    heights: np.ndarray, cell_size: float, offset: tuple[float, float]
) -> Mesh:
    """Turn an occupied height grid into a watertight extruded mesh."""
    occupied = np.isfinite(heights)
    rows, cols = heights.shape
    u_min, v_min = offset

    # Vertex per cell corner, for the top surface and the bottom surface.
    corner_rows, corner_cols = rows + 1, cols + 1
    corner_height = np.zeros((corner_rows, corner_cols))
    corner_count = np.zeros((corner_rows, corner_cols))
    for dr in (0, 1):
        for dc in (0, 1):
            values = np.where(occupied, heights, 0.0)
            corner_height[dr:dr + rows, dc:dc + cols] += values
            corner_count[dr:dr + rows, dc:dc + cols] += occupied
    with np.errstate(invalid="ignore", divide="ignore"):
        corner_height = np.where(corner_count > 0, corner_height / corner_count, 0.0)

    grid_u = u_min + np.arange(corner_cols) * cell_size
    grid_v = v_min + np.arange(corner_rows) * cell_size
    mesh_u, mesh_v = np.meshgrid(grid_u, grid_v)

    n_corners = corner_rows * corner_cols
    top = np.stack([mesh_u.ravel(), mesh_v.ravel(), corner_height.ravel()], axis=1)
    bottom = np.stack([mesh_u.ravel(), mesh_v.ravel(), np.zeros(n_corners)], axis=1)
    vertices = np.concatenate([top, bottom], axis=0)

    def top_index(r, c):
        return r * corner_cols + c

    def bottom_index(r, c):
        return n_corners + r * corner_cols + c

    faces: list[tuple[int, int, int]] = []
    for r, c in zip(*np.nonzero(occupied), strict=True):
        tl, tr = top_index(r, c), top_index(r, c + 1)
        bl, br = top_index(r + 1, c), top_index(r + 1, c + 1)
        faces.append((tl, br, tr))
        faces.append((tl, bl, br))

        ul, ur = bottom_index(r, c), bottom_index(r, c + 1)
        ll, lr = bottom_index(r + 1, c), bottom_index(r + 1, c + 1)
        faces.append((ul, ur, lr))  # reversed winding: faces downward
        faces.append((ul, lr, ll))

        # Vertical wall wherever this cell borders empty space.
        for dr, dc, (a, b) in (
            (-1, 0, (tl, tr)), (1, 0, (bl, br)),
            (0, -1, (tl, bl)), (0, 1, (tr, br)),
        ):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and occupied[nr, nc]:
                continue
            a_bottom = a + n_corners
            b_bottom = b + n_corners
            faces.append((a, b, b_bottom))
            faces.append((a, b_bottom, a_bottom))

    return Mesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int32),
        name="reconstructed",
    )


def _top_surface_tilt_deg(
    heights: np.ndarray, cell_size: float, band_m: float = 0.004
) -> float:
    """Angle between the dominant top surface and the support plane.

    The top face is selected as the cells near the **most common height**, not
    as a fraction of the height range. Selecting by range pulls in the sloped
    side walls the camera sees at the part's edges -- on a real 208 mm block
    that produced a spurious 3.8 deg tilt on a part whose median height was
    constant to 0.3 mm along its whole length.
    """
    occupied = np.isfinite(heights)
    if occupied.sum() < 20:
        return 0.0
    values = heights[occupied]

    # Dominant height = the tallest histogram bin; that is the top face, since
    # it is the largest flat area the camera sees.
    counts, edges = np.histogram(values, bins=min(40, max(8, values.size // 20)))
    peak = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])

    selected = occupied & (np.abs(heights - peak) <= band_m)
    if selected.sum() < 10:
        return 0.0

    rows, cols = np.nonzero(selected)
    u = cols * cell_size
    v = rows * cell_size
    h = heights[selected]
    # Plane h = a*u + b*v + c
    design = np.stack([u, v, np.ones_like(u)], axis=1)
    try:
        coefficients, *_ = np.linalg.lstsq(design, h, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
        return 0.0
    slope = float(np.hypot(coefficients[0], coefficients[1]))
    return float(np.degrees(np.arctan(slope)))


def _level_top(
    heights: np.ndarray, cell_size: float, band_m: float = 0.004
) -> tuple[np.ndarray, float]:
    """Shear the height field so the top face becomes parallel to the base.

    For a machined part the top face *is* parallel to the bottom, so a tilt
    measured between them means the part is sitting rocked on the surface, not
    that it is wedge-shaped. Removing the tilt therefore recovers the true
    geometry -- but only under that assumption, which is why it is opt-in.

    The bottom stays on the plane and the top is flattened, so the result is a
    proper prism rather than the wedge a rocked part would otherwise produce.
    """
    occupied = np.isfinite(heights)
    if occupied.sum() < 20:
        return heights, 0.0
    values = heights[occupied]

    counts, edges = np.histogram(values, bins=min(40, max(8, values.size // 20)))
    peak = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])
    selected = occupied & (np.abs(heights - peak) <= band_m)
    if selected.sum() < 10:
        return heights, 0.0

    rows, cols = np.nonzero(selected)
    design = np.stack([cols * cell_size, rows * cell_size, np.ones(len(rows))], axis=1)
    try:
        coefficients, *_ = np.linalg.lstsq(design, heights[selected], rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
        return heights, 0.0

    all_rows, all_cols = np.mgrid[0:heights.shape[0], 0:heights.shape[1]]
    correction = coefficients[0] * all_cols * cell_size + coefficients[1] * all_rows * cell_size
    # Centre the correction on the part. The fitted plane is expressed from the
    # grid origin, so `a*u + b*v` carries a large constant offset there; removing
    # it un-tilts the part without also sliding it up or down. The mesh bottom is
    # already on the support plane, so no re-seating is wanted either.
    correction = correction - correction[occupied].mean()
    levelled = heights - correction

    removed = float(np.degrees(np.arctan(np.hypot(coefficients[0], coefficients[1]))))
    return levelled, removed


def reconstruct(
    segmentations: list[Segmentation],
    cell_size_m: float = 0.002,
    # Median, not a high percentile: depth noise is roughly symmetric, so the
    # median is unbiased while an 80th percentile rides the noise ceiling.
    # Measured on 200x55x55 mm blocks at 0.6 m (5 merged views, 2 mm cells):
    #   p50 -> +2.0 / +1.0 / +1.1 mm     p80 -> +2.0 / +1.0 / +2.2 mm
    height_percentile: float = 50.0,
    min_points_per_cell: int = 2,
    min_height_m: float = 0.002,
    level_top: bool = False,
    # Fraction of the merged views a cell must appear in. Views are pooled as a
    # union, so a cell held up by one view's noise would otherwise widen the
    # mesh -- and the more views are captured, the wider it gets.
    min_view_fraction: float = 0.5,
) -> tuple[Mesh, ReconstructionReport]:
    """Reconstruct a mesh from one or more segmented top-down views.

    Multiple views of the *same resting orientation* are merged by pooling their
    points in the plane frame: it averages down depth noise. It does **not** add
    faces the camera never saw -- for that the part has to be turned over, which
    needs a registration step this function deliberately does not attempt.
    """
    usable = [s for s in segmentations if s.ok and s.plane is not None]
    if not usable:
        raise ValueError("no usable segmentation: the part was never found")

    # Every view shares the resting plane, so the first view's frame is the
    # common frame; pooling in it is what averages the noise down.
    transform = plane_frame(usable[0])
    inverse = np.linalg.inv(transform)

    local_points = []
    view_ids = []
    centroids = []
    diagonals = []
    for index, segmentation in enumerate(usable):
        points = segmentation.points
        local = points @ inverse[:3, :3].T + inverse[:3, 3]
        local_points.append(local)
        view_ids.append(np.full(len(local), index, dtype=np.int64))
        centroids.append(local[:, :2].mean(axis=0))
        span = local[:, :2].max(axis=0) - local[:, :2].min(axis=0)
        diagonals.append(float(np.hypot(*span)))
    local = np.concatenate(local_points, axis=0)
    views = np.concatenate(view_ids, axis=0)

    report = ReconstructionReport(views_merged=len(usable), points_used=len(local))

    # Merging is only valid if every view saw the same thing in the same place.
    if len(usable) > 1:
        centroids = np.asarray(centroids)
        report.view_centroid_spread_mm = float(
            np.linalg.norm(centroids - centroids.mean(axis=0), axis=1).max() * 1000.0
        )
        report.view_extent_spread_mm = float(
            (max(diagonals) - min(diagonals)) * 1000.0
        )

    # Drop points below the plane (noise) and anything implausibly low.
    above = local[:, 2] > 0.0
    local, views = local[above], views[above]
    if len(local) < 100:
        raise ValueError(f"only {len(local)} points above the plane; check the setup")

    min_views = max(1, int(np.ceil(min_view_fraction * len(usable))))
    heights, counts, offset = _height_grid(
        local, cell_size_m, height_percentile, min_points_per_cell,
        view_ids=views, min_views=min_views,
    )
    report.cells_filled = int(np.isfinite(heights).sum())
    heights, interpolated = _fill_holes(heights)
    report.cells_interpolated = interpolated

    # Cells shorter than min_height are table, not part.
    heights[heights < min_height_m] = np.nan
    if not np.isfinite(heights).any():
        raise ValueError(
            f"nothing taller than {min_height_m * 1000:.1f} mm was found above the plane"
        )

    report.top_tilt_deg = _top_surface_tilt_deg(heights, cell_size_m)
    if level_top:
        heights, removed = _level_top(heights, cell_size_m)
        report.tilt_removed_deg = removed
        report.top_tilt_deg = _top_surface_tilt_deg(heights, cell_size_m)

    mesh = _mesh_from_grid(heights, cell_size_m, offset)

    valid = heights[np.isfinite(heights)]
    report.height_min_mm = float(valid.min() * 1000.0)
    report.height_max_mm = float(valid.max() * 1000.0)

    # Match the model-frame convention: origin at the AABB center, +x long axis.
    mesh = mesh.centered()
    extents = mesh.extents
    if extents[1] > extents[0]:  # swap so the long axis is +x
        mesh = Mesh(mesh.vertices[:, [1, 0, 2]] * np.array([1.0, -1.0, 1.0]),
                    mesh.faces, mesh.face_colors, mesh.name)
        extents = mesh.extents
    report.extents_mm = tuple(float(v * 1000.0) for v in extents)

    # Checked before the geometry warnings: if the views did not agree, every
    # dimension below is meaningless and the user should be told that first.
    if report.view_centroid_spread_mm > 5.0 or report.view_extent_spread_mm > 15.0:
        report.warnings.append(
            f"the {len(usable)} views disagree "
            f"(centroids scatter {report.view_centroid_spread_mm:.0f} mm, "
            f"footprint varies {report.view_extent_spread_mm:.0f} mm). Merging "
            "assumes the camera AND the part stay put between captures; if either "
            "moved, the views land in different places and the mesh smears out. "
            "Mount the camera, do not touch the part, and re-run -- or "
            "reconstruct from a single view."
        )
    if report.interpolated_fraction > 0.25:
        report.warnings.append(
            f"{report.interpolated_fraction:.0%} of the surface was interpolated -- "
            "poor depth coverage, or views that do not overlap."
        )
    if report.top_tilt_deg > 2.0:
        report.warnings.append(
            f"the top surface is tilted {report.top_tilt_deg:.1f} deg relative to the "
            "support plane. If the part should be resting flat, that tilt is being "
            "baked into the mesh as a wedge. Either re-seat the part on a flatter "
            "surface, or pass level_top=True (--level) to remove the tilt -- which "
            "is correct when the part is prismatic, i.e. its top face really is "
            "parallel to its bottom. Ignore this if the top is genuinely sloped."
        )
    if report.height_max_mm < 5.0:
        report.warnings.append(
            f"the part is only {report.height_max_mm:.1f} mm tall; a very flat part "
            "gives ICP little to lock onto and the pose will be weakly constrained."
        )
    return mesh, report
