"""Shared visualisation helpers.

Used by ``tools/visualize.py`` (single PNG panel) and ``tools/live_view.py``
(live window). Kept here rather than in the tools so both render the *same*
depth colour scale and the same overlay semantics -- a viewer that colours
depth differently from the still export is a debugging trap.

Everything returns plain uint8 RGB arrays; no GUI toolkit is imported, so this
module stays importable in the ROS-free core and in headless CI.
"""

from __future__ import annotations

import numpy as np

#: Colour of the background/invalid region in colourised maps.
_INVALID = 0.08


def colorize(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Blue -> green -> red ramp for a [0, 1] array, as uint8 RGB.

    Invalid pixels are rendered near-black so "no data" never reads as "cold".
    """
    v = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    rgb = np.zeros((*v.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(2.0 * v - 0.5, 0, 1)                 # red rises late
    rgb[..., 1] = np.clip(1.5 - np.abs(3.0 * v - 1.5), 0, 1)   # green peaks mid
    rgb[..., 2] = np.clip(1.5 - 3.0 * v, 0, 1)                 # blue falls early
    if valid is not None:
        rgb[~np.asarray(valid, dtype=bool)] = _INVALID
    return (rgb * 255).astype(np.uint8)


def colorize_depth(
    depth: np.ndarray,
    depth_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Colourise a metric depth image. Nearer surfaces render warmer.

    Args:
        depth: float32 meters; ``0`` (or non-finite) means "no return".
        depth_range: fixed ``(near, far)`` window in meters. Pass a fixed range
            for a live stream -- auto-scaling per frame makes the belt appear to
            change colour whenever the part moves, which looks like sensor drift.
            ``None`` auto-scales to the frame's own valid range.
    """
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.0)
    if not valid.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)

    if depth_range is None:
        near, far = float(d[valid].min()), float(d[valid].max())
    else:
        near, far = float(depth_range[0]), float(depth_range[1])
    span = max(far - near, 1e-6)

    normalized = np.zeros_like(d)
    # Invert so near = 1.0 = red: closer things should pop, not recede.
    normalized[valid] = np.clip(1.0 - (d[valid] - near) / span, 0.0, 1.0)
    return colorize(normalized, valid)


def tint_mask(
    color: np.ndarray,
    mask: np.ndarray,
    tint: tuple[int, int, int] = (0, 255, 90),
    dim: float = 0.35,
    strength: float = 0.5,
) -> np.ndarray:
    """Dim the image and tint the masked region, e.g. a segmentation ROI."""
    out = (np.asarray(color, dtype=np.float32) * dim).astype(np.uint8)
    selected = np.asarray(mask, dtype=bool)
    if selected.any():
        blended = (
            np.asarray(color, dtype=np.float32)[selected] * (1.0 - strength)
            + np.asarray(tint, dtype=np.float32) * strength
        )
        out[selected] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def anomaly_view(
    anomaly_map: np.ndarray,
    roi: np.ndarray,
    defect_mask: np.ndarray | None = None,
    defect_color: tuple[int, int, int] = (255, 40, 40),
) -> np.ndarray:
    """Heat map inside the ROI with the accepted defect regions painted solid."""
    view = colorize(anomaly_map, roi)
    if defect_mask is not None:
        selected = np.asarray(defect_mask) > 0
        if selected.any():
            view[selected] = defect_color
    return view


def pose_overlay(
    color: np.ndarray,
    mesh,
    pose,
    intrinsics,
    outline_color: tuple[int, int, int] = (255, 220, 40),
    fill_strength: float = 0.25,
) -> tuple[np.ndarray, float]:
    """Draw the model, at the estimated pose, back onto the image.

    This is the check that actually answers "is the pose right?". Fitness and
    RMSE only say the point clouds agree; they cannot reveal a pose that locked
    onto the wrong face or a mesh built at the wrong scale. Re-projecting the
    model and looking at whether it lands on the part does.

    Returns ``(image, coverage)`` where ``coverage`` is the fraction of the
    rendered silhouette that falls on pixels with valid, comparable depth --
    a quick numeric companion to the visual check.
    """
    from .render import Renderer, RenderItem

    renderer = Renderer(intrinsics, background_color=(0, 0, 0))
    result = renderer.render([
        RenderItem(
            vertices=mesh.vertices,
            faces=mesh.faces,
            base_color=np.asarray(outline_color, dtype=np.uint8),
            transform=pose.as_matrix(),
            label=1,
        )
    ])
    silhouette = result.mask == 1

    out = np.asarray(color, dtype=np.float32).copy()
    if silhouette.any():
        out[silhouette] = (
            out[silhouette] * (1.0 - fill_strength)
            + np.asarray(outline_color, dtype=np.float32) * fill_strength
        )
        from .imageops import binary_erode

        edge = silhouette & ~binary_erode(silhouette, 2)
        out[edge] = outline_color

    coverage = float(silhouette.mean()) if silhouette.size else 0.0
    return np.clip(out, 0, 255).astype(np.uint8), coverage


def hstack_panels(
    images: list[np.ndarray],
    gap: int = 8,
    bar: int = 3,
    background: int = 24,
    bar_colors: tuple[tuple[int, int, int], ...] = (
        (90, 160, 255), (120, 200, 140), (255, 200, 90), (255, 110, 110),
    ),
) -> np.ndarray:
    """Lay panels out horizontally, each under a coloured identification bar."""
    if not images:
        raise ValueError("hstack_panels needs at least one image")
    height = max(img.shape[0] for img in images)
    width = sum(img.shape[1] for img in images) + gap * (len(images) - 1)
    canvas = np.full((height + bar, width, 3), background, dtype=np.uint8)

    x = 0
    for index, image in enumerate(images):
        h, w = image.shape[:2]
        canvas[bar:bar + h, x:x + w] = image
        canvas[0:bar, x:x + w] = bar_colors[index % len(bar_colors)]
        x += w + gap
    return canvas
