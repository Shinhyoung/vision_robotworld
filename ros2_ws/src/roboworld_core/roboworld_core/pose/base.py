"""Pose backend interface and the shared acceptance gate.

All backends return a :class:`~roboworld_core.types.PoseEstimate` expressed in
the **camera optical frame**, in **meters**, with a unit quaternion. Converting
into ``world`` (or whatever frame the robot department agreed to) is the job of
the ROS layer, which owns TF -- see ICD section 4.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass

import numpy as np

from ..geometry import Pose
from ..types import Frame, PoseEstimate


@dataclass
class PoseSettings:
    """Acceptance criteria shared by all backends (see pose.yaml)."""

    min_fitness: float = 0.35
    max_rmse_m: float = 0.006
    valid_z_range_m: tuple[float, float] = (0.30, 1.00)
    max_lateral_offset_m: float = 0.25
    output_frame_id: str = "camera_color_optical_frame"

    @classmethod
    def from_config(cls, cfg) -> PoseSettings:
        section = cfg.section("pose")
        z_range = section.get("valid_z_range_m", [0.30, 1.00])
        return cls(
            min_fitness=float(section.get("min_fitness", 0.35)),
            max_rmse_m=float(section.get("max_rmse_m", 0.006)),
            valid_z_range_m=(float(z_range[0]), float(z_range[1])),
            max_lateral_offset_m=float(section.get("max_lateral_offset_m", 0.25)),
            output_frame_id=str(
                section.get("output_frame_id", "camera_color_optical_frame")
            ),
        )


class PoseBackend(abc.ABC):
    """Base class for 6D pose estimators."""

    name = "base"

    def __init__(self, settings: PoseSettings) -> None:
        self.settings = settings

    @abc.abstractmethod
    def estimate(self, frame: Frame) -> tuple[Pose, float, float, str]:
        """Return ``(pose, fitness, rmse_m, message)`` in the camera optical frame.

        Backends report their raw result; whether it is good enough is decided
        once, in :meth:`validate`, so every backend is held to the same bar.
        """

    def run(self, frame: Frame) -> PoseEstimate:
        """Estimate a pose and apply the shared acceptance gate."""
        started = time.perf_counter()
        try:
            pose, fitness, rmse, message = self.estimate(frame)
        except Exception as exc:  # backend failure must not kill the pipeline
            return PoseEstimate(
                part_id=frame.part_id,
                sequence=frame.sequence,
                stamp=frame.stamp,
                valid=False,
                pose=Pose.identity(frame.intrinsics.frame_id),
                inference_time_ms=(time.perf_counter() - started) * 1000.0,
                backend=self.name,
                message=f"{type(exc).__name__}: {exc}",
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        valid, reason = self.validate(pose, fitness, rmse)
        return PoseEstimate(
            part_id=frame.part_id,
            sequence=frame.sequence,
            stamp=frame.stamp,
            valid=valid,
            pose=pose,
            fitness=float(fitness),
            rmse_m=float(rmse),
            inference_time_ms=elapsed_ms,
            backend=self.name,
            message=message if valid else "; ".join(filter(None, (message, reason))),
        )

    def validate(self, pose: Pose, fitness: float, rmse: float) -> tuple[bool, str]:
        """Check a raw estimate against the configured acceptance criteria."""
        settings = self.settings
        reasons: list[str] = []

        if not np.all(np.isfinite(pose.position)):
            return False, "pose contains non-finite values"
        if fitness < settings.min_fitness:
            reasons.append(f"fitness {fitness:.3f} < min_fitness {settings.min_fitness:.3f}")
        if rmse > settings.max_rmse_m:
            reasons.append(f"rmse {rmse * 1000:.2f} mm > max {settings.max_rmse_m * 1000:.2f} mm")

        z = float(pose.position[2])
        low, high = settings.valid_z_range_m
        if not (low <= z <= high):
            reasons.append(f"z {z:.3f} m outside valid range [{low:.3f}, {high:.3f}]")

        lateral = float(np.linalg.norm(pose.position[:2]))
        if lateral > settings.max_lateral_offset_m:
            reasons.append(
                f"lateral offset {lateral:.3f} m > max {settings.max_lateral_offset_m:.3f} m"
            )

        return (not reasons), "; ".join(reasons)
