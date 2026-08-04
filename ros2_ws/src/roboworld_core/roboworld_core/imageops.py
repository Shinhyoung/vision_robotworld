"""Small numpy image primitives.

OpenCV and SciPy are available in the deployed container but pulling either into
the core library would make ``pytest`` on a bare CI runner impossible. These are
the four operations the pipeline actually needs, implemented on numpy alone.
"""

from __future__ import annotations

import numpy as np


def to_gray(color: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma from an RGB uint8 image, returned as float32 [0, 255]."""
    rgb = np.asarray(color, dtype=np.float32)
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with edge clamping."""
    if sigma <= 0.0:
        return np.asarray(image, dtype=np.float32)
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()

    out = np.asarray(image, dtype=np.float32)
    padded = np.pad(out, ((0, 0), (radius, radius)), mode="edge")
    out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    padded = np.pad(out, ((radius, radius), (0, 0)), mode="edge")
    out = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, padded)
    return out.astype(np.float32)


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of a 2D float image."""
    img = np.asarray(gray, dtype=np.float32)
    padded = np.pad(img, 1, mode="edge")
    # Explicit 3x3 taps: faster and clearer than building a convolution here.
    gx = (
        -padded[:-2, :-2] - 2.0 * padded[1:-1, :-2] - padded[2:, :-2]
        + padded[:-2, 2:] + 2.0 * padded[1:-1, 2:] + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2] - 2.0 * padded[:-2, 1:-1] - padded[:-2, 2:]
        + padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """4-connected binary erosion."""
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, iterations)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
            & padded[1:-1, :-2] & padded[1:-1, 2:]
        )
    return out


def binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """4-connected binary dilation."""
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, iterations)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[1:-1, 1:-1] | padded[:-2, 1:-1] | padded[2:, 1:-1]
            | padded[1:-1, :-2] | padded[1:-1, 2:]
        )
    return out


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label 8-connected components of a boolean mask.

    Returns ``(labels, sizes)`` where ``labels`` is int32 with 0 = background
    and ``sizes[i]`` is the pixel count of label ``i`` (``sizes[0]`` = 0).

    Two-pass union-find. Row-wise vectorisation keeps it fast enough for
    640x480 without SciPy.
    """
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    labels = np.zeros((height, width), dtype=np.int32)

    parent: list[int] = [0]  # parent[0] is the background sentinel

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Pass 1: provisional labels from the already-scanned neighbourhood
    # (west, north-west, north, north-east).
    for y in range(height):
        row = binary[y]
        if not row.any():
            continue
        prev = labels[y - 1] if y > 0 else None
        for x in np.nonzero(row)[0]:
            neighbours = []
            if x > 0 and labels[y, x - 1]:
                neighbours.append(labels[y, x - 1])
            if prev is not None:
                for nx in (x - 1, x, x + 1):
                    if 0 <= nx < width and prev[nx]:
                        neighbours.append(prev[nx])
            if neighbours:
                current = min(neighbours)
                labels[y, x] = current
                for other in neighbours:
                    union(current, other)
            else:
                parent.append(len(parent))
                labels[y, x] = len(parent) - 1

    if len(parent) == 1:
        return labels, np.zeros(1, dtype=np.int64)

    # Pass 2: resolve equivalences and compact the label numbering.
    resolved = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
    unique = np.unique(resolved[1:])
    remap = np.zeros(len(parent), dtype=np.int32)
    for new_id, old_id in enumerate(unique, start=1):
        remap[resolved == old_id] = new_id
    labels = remap[labels]

    sizes = np.bincount(labels.ravel(), minlength=len(unique) + 1).astype(np.int64)
    sizes[0] = 0
    return labels, sizes


def largest_component(mask: np.ndarray, min_size: int = 1) -> np.ndarray:
    """Keep only the largest connected component of ``mask``."""
    labels, sizes = connected_components(mask)
    if len(sizes) <= 1 or sizes[1:].max() < min_size:
        return np.zeros_like(mask, dtype=bool)
    return labels == int(np.argmax(sizes))


def resize_nearest(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize to ``(height, width)``."""
    src = np.asarray(image)
    height, width = shape
    rows = (np.arange(height) * src.shape[0] / height).astype(np.int64).clip(0, src.shape[0] - 1)
    cols = (np.arange(width) * src.shape[1] / width).astype(np.int64).clip(0, src.shape[1] - 1)
    return src[rows[:, None], cols[None, :]]
