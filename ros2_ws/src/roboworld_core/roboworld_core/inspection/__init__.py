"""Surface defect inspection backends and their factory."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .base import InspectionBackend, InspectionSettings
from .efficientad import EfficientAdBackend, EfficientAdUnavailable
from .statistical import StatisticalBackend
from .stub import StubBackend

__all__ = [
    "InspectionBackend",
    "InspectionSettings",
    "StatisticalBackend",
    "EfficientAdBackend",
    "EfficientAdUnavailable",
    "StubBackend",
    "build_backend",
    "backend_for",
    "model_path_for",
]


def backend_for(cfg, part_id: str = "") -> str:
    """Which detector this part uses.

    ``parts.<id>.inspection_backend`` overrides the global ``inspection.backend``.
    Parts do not arrive at the same readiness: EfficientAD needs a GPU and a
    trained checkpoint *per part*, so a line can have one part running the
    production detector while the rest still fall back to the CPU reference.
    Forcing one global choice would mean either holding the ready part back or
    breaking the others with a missing checkpoint.
    """
    entry = cfg.get(f"parts.{part_id}", None) if part_id else None
    if isinstance(entry, Mapping) and entry.get("inspection_backend"):
        return str(entry["inspection_backend"])
    return str(cfg.get("inspection.backend", "statistical"))


def model_path_for(cfg, backend: str, part_id: str) -> Path | None:
    """Resolve the per-part model path, expanding the ``${part_id}`` token."""
    from .. import paths

    key = f"inspection.{backend}.model_path"
    template = cfg.get(key, None)
    if not template:
        return None
    return paths.resolve_path(str(template).replace("${part_id}", part_id))


def build_backend(cfg, part_id: str = "", backend: str | None = None) -> InspectionBackend:
    """Instantiate the inspection backend named in the config.

    Args:
        cfg: loaded :class:`roboworld_core.config.Config`.
        part_id: used to resolve per-part model files.
        backend: overrides ``inspection.backend`` (used by tools and tests).

    Raises:
        ValueError: unknown backend name.
        EfficientAdUnavailable: EfficientAD requested but unusable.
    """
    settings = InspectionSettings.from_config(cfg)
    name = (backend or backend_for(cfg, part_id)).lower()
    segmentation_kwargs = _segmentation_kwargs(cfg, part_id)

    if name == "stub":
        return StubBackend(settings, segmentation_kwargs=segmentation_kwargs)

    if name == "statistical":
        section = cfg.section("inspection.statistical")
        instance = StatisticalBackend(
            settings,
            patch_size=int(section.get("patch_size", 16)),
            stride=int(section.get("stride", 8)),
            blur_sigma_px=float(section.get("blur_sigma_px", 2.0)),
            regularization=float(section.get("regularization", 1e-3)),
            norm_low_percentile=float(section.get("norm_low_percentile", 50.0)),
            norm_high_percentile=float(section.get("norm_high_percentile", 95.0)),
            safety_factor=float(section.get("safety_factor", 1.6)),
            segmentation_kwargs=segmentation_kwargs,
        )
        model_path = model_path_for(cfg, "statistical", part_id)
        if model_path is not None and model_path.exists():
            instance.load(model_path)
        return instance

    if name == "efficientad":
        section = cfg.section("inspection.efficientad")
        size = section.get("image_size", [256, 256])
        return EfficientAdBackend(
            settings,
            model_path=model_path_for(cfg, "efficientad", part_id),
            image_size=(int(size[0]), int(size[1])),
            device=str(section.get("device", "cuda")),
            model_size=str(section.get("model_size", "small")),
            segmentation_kwargs=segmentation_kwargs,
            crop_margin=float(section.get("crop_margin", 0.15)),
        )

    raise ValueError(
        f"unknown inspection backend '{name}' (expected 'efficientad', 'statistical' or 'stub')"
    )


def _segmentation_kwargs(cfg, part_id: str = "") -> dict:
    from ..segmentation import (
        _crown_kwargs,
        expected_extents_for,
        station_roi_from_config,
    )

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
        # Inspection and pose must agree on *which* object is at the station, or
        # the line would grade one part and pick another.
        "station_roi": station_roi_from_config(cfg),
    }
    # Size still RANKS candidates -- inspecting a hand or a neighbouring part
    # would compute the anomaly statistics over the wrong thing -- but it must
    # not REJECT. A defect changes the silhouette (a chip removes material,
    # something stuck on adds it), so the size gate refuses exactly the parts
    # worth inspecting, and an empty mask scores 1.0 by convention: an NG that
    # never looked at the part. Measured on the rig: a part with a foreign
    # object read 83 mm wide against 62 mm, was refused, and the "defect" score
    # of 1.000 came from the empty-ROI rule rather than the detector.
    kwargs["refuse_on_size_mismatch"] = False
    kwargs.update(_crown_kwargs(section))
    if part_id and section.get("identify_by_size", True):
        kwargs["expected_extents_m"] = expected_extents_for(cfg, part_id)
    return kwargs
