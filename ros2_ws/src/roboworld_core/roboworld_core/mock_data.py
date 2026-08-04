"""Synthetic RGB-D scene generation for hardware-free development.

This is the "mock data" deliverable of the Team Lead role (claude.md section
3.1). A :class:`MockStation` reproduces the index station: a conveyor plane, one
MC nylon block at a randomised but bounded pose, and optional surface defects.
Every frame carries ground truth (``gt_pose``, ``gt_is_good``) so the Inspection
and Pose agents can score themselves without a camera or a labelled dataset.

Determinism matters: identical ``seed`` values must produce identical frames,
otherwise CI cannot assert on numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .geometry import Pose, euler_to_matrix, make_transform
from .imageops import binary_erode
from .mesh_io import Mesh, load_ply
from .render import Renderer, RenderItem, plane_item
from .types import CameraIntrinsics, Frame

#: Defect kinds the generator can paint onto a part surface.
DEFECT_KINDS = ("scratch", "dent", "stain", "chip")


@dataclass
class PartSpec:
    """One inspectable part type."""

    part_id: str
    mesh_path: str
    mesh_units: str = "m"
    color: tuple[int, int, int] = (196, 188, 170)  # MC nylon: off-white / ivory

    def load(self) -> Mesh:
        return load_ply(self.mesh_path, units=self.mesh_units, center=True)


@dataclass
class StationLayout:
    """Geometry of the stop station, all lengths in meters."""

    #: Nominal camera height above the belt, looking straight down.
    camera_height_m: float = 0.60
    #: Belt rectangle rendered under the part.
    belt_size_m: tuple[float, float] = (1.2, 0.6)
    belt_color: tuple[int, int, int] = (58, 62, 70)
    #: Part placement jitter about the station nominal, meters and degrees.
    position_jitter_m: tuple[float, float, float] = (0.035, 0.025, 0.004)
    yaw_jitter_deg: float = 180.0
    tilt_jitter_deg: float = 4.0


class MockStation:
    """Renders mock frames of a part at the stop station."""

    def __init__(
        self,
        parts: Sequence[PartSpec],
        intrinsics: CameraIntrinsics,
        layout: StationLayout | None = None,
        depth_noise_m: float = 0.0015,
        color_noise: float = 4.0,
        dropout_ratio: float = 0.01,
    ) -> None:
        if not parts:
            raise ValueError("MockStation needs at least one PartSpec")
        self.parts = {spec.part_id: spec for spec in parts}
        self._meshes: dict[str, Mesh] = {}
        self.intrinsics = intrinsics
        self.layout = layout or StationLayout()
        self.renderer = Renderer(intrinsics)
        self.depth_noise_m = float(depth_noise_m)
        self.color_noise = float(color_noise)
        self.dropout_ratio = float(dropout_ratio)

    def mesh(self, part_id: str) -> Mesh:
        if part_id not in self._meshes:
            if part_id not in self.parts:
                raise KeyError(
                    f"unknown part_id '{part_id}' (known: {sorted(self.parts)})"
                )
            self._meshes[part_id] = self.parts[part_id].load()
        return self._meshes[part_id]

    # -- pose sampling ---------------------------------------------------
    def sample_pose(self, rng: np.random.Generator) -> Pose:
        """Sample a plausible part pose in the camera optical frame.

        The part lies flat on the belt directly under the camera; the belt sits
        at ``camera_height_m`` and the part rests half its thickness above it.
        """
        layout = self.layout
        jitter = np.asarray(layout.position_jitter_m)
        offset = rng.uniform(-jitter, jitter)

        # Part center: on the belt, minus half the block thickness (0.055 m).
        position = np.array(
            [offset[0], offset[1], layout.camera_height_m - 0.0275 + offset[2]]
        )
        yaw = np.radians(rng.uniform(-layout.yaw_jitter_deg, layout.yaw_jitter_deg))
        roll = np.radians(rng.uniform(-layout.tilt_jitter_deg, layout.tilt_jitter_deg))
        pitch = np.radians(rng.uniform(-layout.tilt_jitter_deg, layout.tilt_jitter_deg))
        rotation = euler_to_matrix(roll, pitch, yaw)
        return Pose(position, _quat(rotation), frame_id=self.intrinsics.frame_id)

    # -- frame generation ------------------------------------------------
    def render_frame(
        self,
        part_id: str,
        pose: Pose,
        defect: str | None = None,
        seed: int = 0,
        sequence: int = 0,
        stamp: float = 0.0,
    ) -> Frame:
        """Render one RGB-D frame of ``part_id`` at ``pose``."""
        rng = np.random.default_rng(seed)
        spec = self.parts[part_id]
        mesh = self.mesh(part_id)

        belt_transform = make_transform(
            np.eye(3), np.array([0.0, 0.0, self.layout.camera_height_m])
        )
        items = [
            plane_item(belt_transform, self.layout.belt_size_m, self.layout.belt_color, label=0),
            RenderItem(
                vertices=mesh.vertices,
                faces=mesh.faces,
                base_color=np.asarray(spec.color, dtype=np.uint8),
                transform=pose.as_matrix(),
                label=1,
            ),
        ]
        result = self.renderer.render(items)
        color = result.color.astype(np.float64)
        depth = result.depth.astype(np.float32)
        part_mask = result.mask == 1

        if defect is not None:
            color, depth = apply_defect(color, depth, part_mask, defect, rng)

        color = self._add_sensor_noise(color, rng)
        depth = self._add_depth_noise(depth, rng)

        return Frame(
            color=color,
            depth=depth,
            intrinsics=self.intrinsics,
            stamp=stamp,
            sequence=sequence,
            part_id=part_id,
            gt_pose=pose,
            gt_is_good=defect is None,
            gt_part_mask=part_mask,
        )

    def sample_frame(
        self,
        part_id: str,
        defect: str | None = None,
        seed: int = 0,
        sequence: int = 0,
        stamp: float = 0.0,
    ) -> Frame:
        """Sample a random pose and render it."""
        rng = np.random.default_rng(seed)
        pose = self.sample_pose(rng)
        return self.render_frame(
            part_id, pose, defect=defect, seed=seed + 1, sequence=sequence, stamp=stamp
        )

    # -- sensor imperfections -------------------------------------------
    def _add_sensor_noise(self, color: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.color_noise > 0.0:
            color = color + rng.normal(0.0, self.color_noise, color.shape)
        return np.clip(color, 0, 255).astype(np.uint8)

    def _add_depth_noise(self, depth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = depth.astype(np.float32).copy()
        valid = out > 0.0
        if self.depth_noise_m > 0.0:
            # D455 stereo error grows roughly with the square of range.
            sigma = self.depth_noise_m * (out / 0.6) ** 2
            out[valid] += rng.normal(0.0, 1.0, out.shape).astype(np.float32)[valid] * sigma[valid]
        if self.dropout_ratio > 0.0:
            holes = rng.random(out.shape) < self.dropout_ratio
            out[holes & valid] = 0.0
        return out


def apply_defect(
    color: np.ndarray,
    depth: np.ndarray,
    part_mask: np.ndarray,
    kind: str,
    rng: np.random.Generator,
    edge_margin_px: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Paint a surface defect onto the visible part region.

    ``edge_margin_px`` keeps the defect center away from the silhouette. A
    defect straddling the object boundary is half painted onto the conveyor and
    is not a fair inspection sample -- it would be scored as a missed detection
    when in truth the mock never put it on the part.

    Returns modified ``(color, depth)``. ``color`` stays float64 so downstream
    noise addition does not clip twice.
    """
    if kind not in DEFECT_KINDS:
        raise ValueError(f"unknown defect kind '{kind}' (known: {DEFECT_KINDS})")

    interior = binary_erode(part_mask, edge_margin_px)
    if not interior.any():  # thin/occluded view: fall back to the raw silhouette
        interior = part_mask
    rows, cols = np.nonzero(interior)
    if rows.size == 0:
        return color, depth  # part not visible; nothing to damage

    color = color.copy()
    depth = depth.copy()
    height, width = part_mask.shape
    pick = rng.integers(0, rows.size)
    center_y, center_x = int(rows[pick]), int(cols[pick])

    yy, xx = np.mgrid[0:height, 0:width]

    if kind == "scratch":
        angle = rng.uniform(0.0, np.pi)
        length = rng.uniform(25.0, 70.0)
        thickness = rng.uniform(0.8, 2.0)
        dx, dy = np.cos(angle), np.sin(angle)
        rel_x = xx - center_x
        rel_y = yy - center_y
        along = rel_x * dx + rel_y * dy
        across = np.abs(-rel_x * dy + rel_y * dx)
        region = part_mask & (np.abs(along) < length / 2.0) & (across < thickness)
        color[region] *= 0.45
        depth[region] += 0.0004  # a groove reads slightly farther away

    elif kind == "dent":
        radius = rng.uniform(6.0, 14.0)
        dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        region = part_mask & (dist < radius)
        falloff = np.clip(1.0 - dist / max(radius, 1e-6), 0.0, 1.0)
        color[region] *= (0.55 + 0.25 * (1.0 - falloff[region]))[:, None]
        depth[region] += (0.0018 * falloff[region]).astype(depth.dtype)

    elif kind == "stain":
        radius = rng.uniform(10.0, 22.0)
        dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        # Irregular blob: modulate the radius with a low-frequency angular term.
        theta = np.arctan2(yy - center_y, xx - center_x)
        wobble = 1.0 + 0.35 * np.sin(3.0 * theta + rng.uniform(0, 6.28))
        region = part_mask & (dist < radius * wobble)
        tint = np.array([0.62, 0.55, 0.40])
        color[region] *= tint

    elif kind == "chip":
        size_x = rng.uniform(8.0, 18.0)
        size_y = rng.uniform(8.0, 18.0)
        region = (
            part_mask
            & (np.abs(xx - center_x) < size_x)
            & (np.abs(yy - center_y) < size_y)
        )
        color[region] *= 0.30
        depth[region] += rng.uniform(0.002, 0.005)  # material missing

    return color, depth


def _quat(rotation: np.ndarray) -> np.ndarray:
    from .geometry import quaternion_from_matrix

    return quaternion_from_matrix(rotation)


def parts_from_config(cfg) -> list[PartSpec]:
    """Build :class:`PartSpec` list from the ``parts`` config section.

    Parts with no ``mesh`` are skipped: the mock station renders from CAD, so a
    camera-registered part with no geometry simply cannot be simulated. It is
    still inspectable from real frames.
    """
    from . import paths

    specs: list[PartSpec] = []
    for part_id, entry in cfg.get("parts").items():
        if not entry.get("mesh"):
            continue
        specs.append(
            PartSpec(
                part_id=part_id,
                mesh_path=str(paths.resolve_path(entry["mesh"])),
                mesh_units=str(entry.get("mesh_units", "m")),
                color=tuple(int(c) for c in entry.get("color", (196, 188, 170))),
            )
        )
    return specs
