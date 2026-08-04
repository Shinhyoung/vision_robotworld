"""ROS-free mirrors of the :mod:`roboworld_interfaces` messages.

The core library must run without a ROS installation (claude.md section 2:
every agent develops against mocks, hardware and middleware come later), so the
contract is expressed twice: once as ``.msg`` files for the wire, and once here
as dataclasses. :mod:`roboworld_core.contract` verifies the two stay aligned,
which is the contract test demanded by section 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

from .geometry import Pose

#: Semantic version of the pipeline contract. Bump on any ICD change.
PIPELINE_VERSION = "0.1.0"


class PartStatus(IntEnum):
    """Mirrors the ``STATUS_*`` constants of ``PartResult.msg``."""

    OK = 0
    NG = 1
    NO_POSE = 2
    ERROR = 3


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics for the color frame, in pixels."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    frame_id: str = "camera_color_optical_frame"
    distortion: tuple[float, ...] = ()

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def from_config(cls, cfg: Any) -> CameraIntrinsics:
        """Build from a :class:`roboworld_core.config.Config` ``camera`` section."""
        return cls(
            width=int(cfg.get("width")),
            height=int(cfg.get("height")),
            fx=float(cfg.get("fx")),
            fy=float(cfg.get("fy")),
            cx=float(cfg.get("cx")),
            cy=float(cfg.get("cy")),
            frame_id=str(cfg.get("optical_frame_id", "camera_color_optical_frame")),
            distortion=tuple(float(v) for v in cfg.get("distortion", ()) or ()),
        )


@dataclass
class Frame:
    """One captured RGB-D frame plus the metadata needed downstream."""

    color: np.ndarray  # (H, W, 3) uint8, RGB
    depth: np.ndarray  # (H, W) float32, meters; 0.0 or NaN = invalid
    intrinsics: CameraIntrinsics
    stamp: float  # capture time, seconds (ROS time when running under ROS)
    sequence: int = 0
    part_id: str = ""
    #: Ground truth, present only for mock/synthetic frames. Never published.
    gt_pose: Pose | None = None
    gt_is_good: bool | None = None
    #: Exact part silhouette from the renderer. Used to score segmentation and
    #: as the ``roi_mask`` hint in tests; a real camera never provides this.
    gt_part_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.color = np.asarray(self.color)
        self.depth = np.asarray(self.depth, dtype=np.float32)
        if self.color.ndim != 3 or self.color.shape[2] != 3:
            raise ValueError(f"color must be (H, W, 3), got {self.color.shape}")
        if self.depth.shape != self.color.shape[:2]:
            raise ValueError(
                f"depth {self.depth.shape} must match color {self.color.shape[:2]} "
                "(depth must be aligned to color, see ICD section 3)"
            )


@dataclass
class InspectionResult:
    """Mirror of ``InspectionResult.msg``."""

    part_id: str
    sequence: int
    stamp: float
    is_good: bool
    anomaly_score: float
    threshold: float
    frame_id: str = "camera_color_optical_frame"
    anomaly_map: np.ndarray | None = None  # (H, W) float32
    defect_mask: np.ndarray | None = None  # (H, W) uint8, 0/255
    inference_time_ms: float = 0.0
    backend: str = "stub"


@dataclass
class PoseEstimate:
    """Mirror of ``PoseResult.msg``."""

    part_id: str
    sequence: int
    stamp: float
    valid: bool
    pose: Pose
    fitness: float = 0.0
    rmse_m: float = 0.0
    covariance: np.ndarray = field(default_factory=lambda: np.zeros(36, dtype=np.float64))
    inference_time_ms: float = 0.0
    backend: str = "stub"
    message: str = ""


@dataclass
class PartResult:
    """Mirror of ``PartResult.msg`` -- the final robot-department output."""

    sequence: int
    part_id: str
    stamp: float
    frame_id: str
    status: PartStatus
    is_good: bool
    anomaly_score: float
    anomaly_threshold: float
    pose_valid: bool
    pose: Pose
    pose_fitness: float = 0.0
    tact_time_ms: float = 0.0
    pipeline_version: str = PIPELINE_VERSION
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, used by the dry-run harness and contract tests."""
        return {
            "sequence": int(self.sequence),
            "part_id": self.part_id,
            "stamp": float(self.stamp),
            "frame_id": self.frame_id,
            "status": int(self.status),
            "status_name": self.status.name,
            "is_good": bool(self.is_good),
            "anomaly_score": round(float(self.anomaly_score), 6),
            "anomaly_threshold": round(float(self.anomaly_threshold), 6),
            "pose_valid": bool(self.pose_valid),
            "pose": {
                "frame_id": self.pose.frame_id,
                "position": [round(float(v), 6) for v in self.pose.position],
                "orientation": [round(float(v), 6) for v in self.pose.orientation],
            },
            "pose_fitness": round(float(self.pose_fitness), 6),
            "tact_time_ms": round(float(self.tact_time_ms), 3),
            "pipeline_version": self.pipeline_version,
            "message": self.message,
        }
