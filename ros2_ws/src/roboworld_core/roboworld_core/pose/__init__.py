"""6D pose estimation backends and their factory."""

from __future__ import annotations

from .base import PoseBackend, PoseSettings
from .foundationpose import (
    LICENSE_NOTICE,
    FoundationPoseBackend,
    FoundationPoseUnavailable,
)
from .icp import IcpParams, IcpPoseBackend, voxel_downsample
from .nocad import NoCadPoseBackend
from .stub import StubPoseBackend

__all__ = [
    "PoseBackend",
    "PoseSettings",
    "IcpPoseBackend",
    "IcpParams",
    "voxel_downsample",
    "FoundationPoseBackend",
    "FoundationPoseUnavailable",
    "LICENSE_NOTICE",
    "NoCadPoseBackend",
    "StubPoseBackend",
    "build_backend",
    "has_cad",
    "load_part_mesh",
]


def has_cad(cfg, part_id: str) -> bool:
    """Whether ``part_id`` declares a CAD mesh.

    Parts registered from camera captures alone (tools/register_part.py) have no
    mesh. They can still be **inspected** -- inspection and segmentation never
    touch the geometry -- but pose estimation is unavailable until a mesh exists.
    """
    entry = cfg.get(f"parts.{part_id}", None) or {}
    return bool(entry.get("mesh"))


def load_part_mesh(cfg, part_id: str):
    """Load the CAD mesh of ``part_id``, honouring the configured units."""
    from .. import paths
    from ..mesh_io import load_ply

    entry = cfg.get(f"parts.{part_id}", None)
    if entry is None:
        raise ValueError(
            f"unknown part_id '{part_id}'; known parts: {sorted(cfg.get('parts', {}))}"
        )
    if not entry.get("mesh"):
        raise ValueError(
            f"part '{part_id}' has no CAD mesh, so 6D pose cannot be estimated. "
            "Inspection still works. To enable pose, reconstruct or obtain a mesh, "
            f"then set parts.{part_id}.mesh in the config."
        )
    mesh_path = paths.resolve_path(entry["mesh"])
    if not mesh_path.exists():
        raise FileNotFoundError(
            f"CAD mesh for '{part_id}' not found at {mesh_path} "
            "(check parts.yaml and ROBOWORLD_ASSETS_DIR)"
        )
    return load_ply(mesh_path, units=str(entry.get("mesh_units", "m")), center=True)


def build_backend(
    cfg,
    part_id: str,
    backend: str | None = None,
    allow_missing_cad: bool = False,
    **kwargs,
) -> PoseBackend:
    """Instantiate the pose backend named in the config.

    Args:
        cfg: loaded :class:`roboworld_core.config.Config`.
        part_id: selects the CAD model.
        backend: overrides ``pose.backend``.
        allow_missing_cad: when the part has no mesh, return
            :class:`NoCadPoseBackend` instead of raising. Long-running services
            set this so one CAD-less part cannot take the whole node down;
            tools leave it off so the problem surfaces immediately.
        **kwargs: forwarded to the backend (e.g. ``ros_bridge`` for FoundationPose).
    """
    settings = PoseSettings.from_config(cfg)
    name = (backend or cfg.get("pose.backend", "icp")).lower()

    if name == "stub":
        station_height = float(cfg.get("station.camera_height_m", 0.60))
        return StubPoseBackend(settings, nominal_z_m=station_height - 0.0275, **kwargs)

    if allow_missing_cad and not has_cad(cfg, part_id):
        return NoCadPoseBackend(settings, part_id)

    mesh = load_part_mesh(cfg, part_id)

    if name == "icp":
        section = cfg.section("pose.icp")
        params = IcpParams(
            model_points=int(section.get("model_points", 2000)),
            scene_max_points=int(section.get("scene_max_points", 600)),
            voxel_size_m=float(section.get("voxel_size_m", 0.003)),
            max_iterations=int(section.get("max_iterations", 40)),
            tolerance_m=float(section.get("tolerance_m", 1e-5)),
            max_correspondence_start_m=float(section.get("max_correspondence_start_m", 0.030)),
            max_correspondence_end_m=float(section.get("max_correspondence_end_m", 0.006)),
            resting_faces=tuple(section.get("resting_faces", ("+y", "-y", "+z", "-z"))),
            try_long_axis_flip=bool(section.get("try_long_axis_flip", True)),
            surface_percentile=float(section.get("surface_percentile", 98.0)),
            coarse_iterations=int(section.get("coarse_iterations", 12)),
            coarse_model_points=int(section.get("coarse_model_points", 900)),
            coarse_scene_points=int(section.get("coarse_scene_points", 300)),
        )
        return IcpPoseBackend(
            settings,
            mesh,
            params=params,
            segmentation_kwargs=_segmentation_kwargs(cfg, part_id),
            use_open3d=bool(section.get("use_open3d", True)),
            **kwargs,
        )

    if name == "foundationpose":
        from .. import paths

        section = cfg.section("pose.foundationpose")
        entry = cfg.get(f"parts.{part_id}")
        return FoundationPoseBackend(
            settings,
            mesh,
            mesh_path=paths.resolve_path(entry["mesh"]),
            mode=str(section.get("mode", "ros_action")),
            refine_iterations=int(section.get("refine_iterations", 5)),
            track=bool(section.get("track", False)),
            device=str(section.get("device", "cuda")),
            python_module_path=section.get("python_module_path", None),
            **kwargs,
        )

    raise ValueError(
        f"unknown pose backend '{name}' (expected 'foundationpose', 'icp' or 'stub')"
    )


def _segmentation_kwargs(cfg, part_id: str = "") -> dict:
    """Segmentation settings for a backend.

    ``part_id`` enables size-based identification: without it the backends fall
    back to "largest object on the surface", which will happily register the
    model against a hand or a neighbouring box. Measured on a scene with a
    larger decoy present, that produced a 150 mm position error reported as a
    *valid* pose -- the backends must therefore get the expectation too, not
    only the config-level helper.
    """
    from ..segmentation import expected_extents_for, station_roi_from_config

    section = cfg.section("pose.segmentation")
    camera = cfg.section("camera")
    kwargs = {
        "plane_iterations": int(section.get("plane_ransac_iterations", 120)),
        "plane_distance_threshold_m": float(section.get("plane_distance_threshold_m", 0.006)),
        "min_height_above_plane_m": float(section.get("min_height_above_plane_m", 0.008)),
        "max_height_above_plane_m": float(section.get("max_height_above_plane_m", 0.120)),
        "min_cluster_points": int(section.get("min_cluster_points", 300)),
        "depth_range_m": (
            float(camera.get("depth_min_m", 0.05)),
            float(camera.get("depth_max_m", 5.0)),
        ),
        "size_tolerance": float(section.get("size_tolerance", 0.25)),
        # Size matching cannot separate the part at the station from an
        # identical one further down the belt -- at 200 mm spacing it published
        # the *next* part's pose. The station volume is what separates them.
        "station_roi": station_roi_from_config(cfg),
    }
    if part_id and section.get("identify_by_size", True):
        kwargs["expected_extents_m"] = expected_extents_for(cfg, part_id)
    return kwargs
