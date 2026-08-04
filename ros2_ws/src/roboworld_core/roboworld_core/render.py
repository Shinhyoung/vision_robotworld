"""A small numpy software rasterizer used to synthesise mock RGB-D frames.

claude.md section 2 requires every agent to be able to develop and test with no
D455 attached and no GPU. Rather than shipping opaque canned images, mock
frames are *rendered from the real part CAD* (``01_input/*.ply``) at a known
pose. That gives the Pose agent ground truth to score against and the
Inspection agent realistic geometry to inject defects into.

Camera convention: OpenCV / ROS optical frame -- ``+x`` right, ``+y`` down,
``+z`` forward into the scene. Depth is the ``z`` coordinate in meters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import CameraIntrinsics

#: Depth value marking "no return", matching the D455 convention.
INVALID_DEPTH = 0.0


@dataclass
class RenderItem:
    """One mesh instance to draw."""

    vertices: np.ndarray  # (V, 3) in model frame, meters
    faces: np.ndarray  # (F, 3) int
    base_color: np.ndarray  # (3,) uint8 or (F, 3) uint8
    transform: np.ndarray  # 4x4 T_camera_model
    label: int = 1  # written into the instance mask


@dataclass
class RenderResult:
    color: np.ndarray  # (H, W, 3) uint8 RGB
    depth: np.ndarray  # (H, W) float32 meters, 0.0 where invalid
    mask: np.ndarray  # (H, W) int32 instance labels, 0 = background


class Renderer:
    """Z-buffered triangle rasterizer with Lambertian shading."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        light_direction: tuple[float, float, float] = (-0.3, -0.5, 1.0),
        ambient: float = 0.35,
        background_color: tuple[int, int, int] = (18, 18, 20),
    ) -> None:
        self.intrinsics = intrinsics
        light = np.asarray(light_direction, dtype=np.float64)
        self.light_direction = light / np.linalg.norm(light)
        self.ambient = float(ambient)
        self.background_color = np.asarray(background_color, dtype=np.uint8)

    def render(self, items: list[RenderItem]) -> RenderResult:
        height, width = self.intrinsics.height, self.intrinsics.width
        color = np.tile(self.background_color, (height, width, 1)).astype(np.float64)
        depth = np.full((height, width), np.inf, dtype=np.float64)
        mask = np.zeros((height, width), dtype=np.int32)

        for item in items:
            self._draw(item, color, depth, mask)

        depth_out = np.where(np.isfinite(depth), depth, INVALID_DEPTH).astype(np.float32)
        return RenderResult(
            color=np.clip(color, 0, 255).astype(np.uint8),
            depth=depth_out,
            mask=mask,
        )

    # -- internals -------------------------------------------------------
    def _draw(
        self,
        item: RenderItem,
        color: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        intr = self.intrinsics
        height, width = intr.height, intr.width

        transform = np.asarray(item.transform, dtype=np.float64).reshape(4, 4)
        verts_cam = (
            np.asarray(item.vertices, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]
        )
        faces = np.asarray(item.faces, dtype=np.int64)

        # Perspective projection; guard against points at/behind the pinhole.
        z = verts_cam[:, 2]
        safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = intr.fx * verts_cam[:, 0] / safe_z + intr.cx
        v = intr.fy * verts_cam[:, 1] / safe_z + intr.cy
        uv = np.stack([u, v], axis=1)

        tri_uv = uv[faces]  # (F, 3, 2)
        tri_z = z[faces]  # (F, 3)
        tri_xyz = verts_cam[faces]  # (F, 3, 3)

        # Backface / behind-camera / off-screen rejection before rasterizing.
        normals = np.cross(
            tri_xyz[:, 1] - tri_xyz[:, 0], tri_xyz[:, 2] - tri_xyz[:, 0]
        )
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(lengths, 1e-12)
        centroids = tri_xyz.mean(axis=1)
        view = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
        facing = np.einsum("ij,ij->i", normals, view)
        # Flip normals that point away so shading works regardless of winding.
        normals = np.where(facing[:, None] > 0.0, -normals, normals)

        visible = (tri_z > 1e-4).all(axis=1)
        visible &= (tri_uv[..., 0].max(axis=1) >= 0) & (tri_uv[..., 0].min(axis=1) < width)
        visible &= (tri_uv[..., 1].max(axis=1) >= 0) & (tri_uv[..., 1].min(axis=1) < height)

        shade = self.ambient + (1.0 - self.ambient) * np.clip(
            normals @ self.light_direction, 0.0, 1.0
        )

        base = np.asarray(item.base_color, dtype=np.float64)
        if base.ndim == 1:
            base = np.tile(base, (len(faces), 1))

        for index in np.nonzero(visible)[0]:
            self._raster_triangle(
                tri_uv[index], tri_z[index], base[index] * shade[index],
                item.label, color, depth, mask,
            )

    @staticmethod
    def _raster_triangle(
        uv: np.ndarray,
        z: np.ndarray,
        rgb: np.ndarray,
        label: int,
        color: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        height, width = depth.shape
        x_min = max(int(np.floor(uv[:, 0].min())), 0)
        x_max = min(int(np.ceil(uv[:, 0].max())), width - 1)
        y_min = max(int(np.floor(uv[:, 1].min())), 0)
        y_max = min(int(np.ceil(uv[:, 1].max())), height - 1)
        if x_min > x_max or y_min > y_max:
            return

        (x0, y0), (x1, y1), (x2, y2) = uv
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:  # degenerate sliver
            return

        xs = np.arange(x_min, x_max + 1) + 0.5
        ys = np.arange(y_min, y_max + 1) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)

        # Barycentric coordinates of every pixel center in the bounding box.
        w0 = ((x1 - grid_x) * (y2 - grid_y) - (x2 - grid_x) * (y1 - grid_y)) / area
        w1 = ((x2 - grid_x) * (y0 - grid_y) - (x0 - grid_x) * (y2 - grid_y)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            return

        # Perspective-correct depth: interpolate 1/z, not z.
        inv_z = w0 / z[0] + w1 / z[1] + w2 / z[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            pixel_z = np.where(inv_z > 1e-9, 1.0 / inv_z, np.inf)

        window = depth[y_min:y_max + 1, x_min:x_max + 1]
        closer = inside & (pixel_z < window)
        if not closer.any():
            return

        window[closer] = pixel_z[closer]
        color[y_min:y_max + 1, x_min:x_max + 1][closer] = rgb
        mask[y_min:y_max + 1, x_min:x_max + 1][closer] = label


def plane_item(
    transform: np.ndarray,
    size: tuple[float, float],
    color: tuple[int, int, int],
    label: int = 0,
) -> RenderItem:
    """Build a rectangular plane (e.g. the conveyor belt) as two triangles.

    The plane spans ``size`` in the local XY plane at ``z = 0``.
    """
    half_x, half_y = size[0] / 2.0, size[1] / 2.0
    vertices = np.array(
        [[-half_x, -half_y, 0.0], [half_x, -half_y, 0.0],
         [half_x, half_y, 0.0], [-half_x, half_y, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return RenderItem(
        vertices=vertices,
        faces=faces,
        base_color=np.asarray(color, dtype=np.uint8),
        transform=transform,
        label=label,
    )


def depth_to_points(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: np.ndarray | None = None,
    depth_range: tuple[float, float] = (0.05, 5.0),
) -> np.ndarray:
    """Back-project a depth image to an ``(N, 3)`` point cloud in camera frame."""
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > depth_range[0]) & (depth < depth_range[1])
    if mask is not None:
        valid &= np.asarray(mask).astype(bool)
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float64)

    rows, cols = np.nonzero(valid)
    z = depth[rows, cols]
    x = (cols + 0.5 - intrinsics.cx) * z / intrinsics.fx
    y = (rows + 0.5 - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x, y, z], axis=1)
