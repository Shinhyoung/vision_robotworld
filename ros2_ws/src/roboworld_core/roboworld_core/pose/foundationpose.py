"""FoundationPose backend (NVIDIA Isaac ROS Pose Estimation).

claude.md section 1 fixes FoundationPose as the production 6D estimator, run
through the ``isaac_ros_foundationpose`` package on Ubuntu 22.04 / Humble.

LICENSING (claude.md section 1): the public FoundationPose release is
**non-commercial**. It is acceptable for this demo only. Commercialisation
requires switching to the NGC commercial build. :data:`LICENSE_NOTICE` is logged
once at startup so the constraint stays visible in the field logs.

Isaac ROS exposes FoundationPose as ROS nodes rather than an importable Python
API, so this backend has two modes:

* ``mode="ros_action"`` -- the ROS pose node forwards the frame to the Isaac ROS
  graph and feeds the answer back here. Selected automatically when running
  under ``roboworld_pose``.
* ``mode="python"``     -- direct import of a local FoundationPose checkout for
  offline experiments (``pose.foundationpose.python_module_path``).

Neither is importable on a CPU-only CI runner, so both fail loudly with an
actionable message rather than at import time.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry import Pose
from ..mesh_io import Mesh
from ..types import Frame
from .base import PoseBackend, PoseSettings

LICENSE_NOTICE = (
    "FoundationPose public release is NON-COMMERCIAL (CC BY-NC-SA 4.0). "
    "Demo/evaluation use only -- switch to the NGC commercial build before "
    "productisation. See claude.md section 1."
)


class FoundationPoseUnavailable(RuntimeError):
    """Raised when the Isaac ROS graph or a local FoundationPose is missing."""


class FoundationPoseBackend(PoseBackend):
    """Adapter around Isaac ROS FoundationPose."""

    name = "foundationpose"

    def __init__(
        self,
        settings: PoseSettings,
        mesh: Mesh,
        mesh_path: str | Path,
        mode: str = "ros_action",
        refine_iterations: int = 5,
        track: bool = False,
        device: str = "cuda",
        python_module_path: str | Path | None = None,
        symmetry_group: list | None = None,
        #: Injected by ``roboworld_pose`` -- called with the frame, returns
        #: ``(4x4 T_camera_model, fitness)``. Keeps ROS out of the core package.
        ros_bridge: Callable[[Frame], tuple[np.ndarray, float]] | None = None,
    ) -> None:
        super().__init__(settings, symmetry_group)
        self.mesh = mesh
        self.mesh_path = Path(mesh_path)
        self.mode = str(mode)
        self.refine_iterations = int(refine_iterations)
        self.track = bool(track)
        self.device = str(device)
        self.python_module_path = Path(python_module_path) if python_module_path else None
        self.ros_bridge = ros_bridge
        self._estimator: Any = None

    # -- lifecycle -------------------------------------------------------
    def _load_python_estimator(self) -> Any:
        if self._estimator is not None:
            return self._estimator
        if self.python_module_path is None or not self.python_module_path.exists():
            raise FoundationPoseUnavailable(
                "pose.foundationpose.python_module_path is not set or does not exist. "
                "Either clone FoundationPose and point that key at it, run under "
                "Isaac ROS (mode='ros_action'), or set pose.backend to 'icp'."
            )
        import sys

        sys.path.insert(0, str(self.python_module_path))
        try:
            import trimesh  # type: ignore[import-not-found]
            from estimater import FoundationPose  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on deployment env
            raise FoundationPoseUnavailable(
                f"FoundationPose python API not importable from "
                f"{self.python_module_path}: {exc}"
            ) from exc

        mesh = trimesh.load(str(self.mesh_path))
        self._estimator = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh
        )
        return self._estimator

    # -- backend API -----------------------------------------------------
    def estimate(self, frame: Frame) -> tuple[Pose, float, float, str]:
        if self.mode == "ros_action":
            if self.ros_bridge is None:
                raise FoundationPoseUnavailable(
                    "mode='ros_action' requires the Isaac ROS bridge supplied by "
                    "roboworld_pose/pose_node. Launch via "
                    "`ros2 launch roboworld_bringup pipeline.launch.py "
                    "pose_backend:=foundationpose`, or set pose.backend to 'icp'."
                )
            transform, fitness = self.ros_bridge(frame)
            pose = Pose.from_matrix(np.asarray(transform), frame.intrinsics.frame_id)
            # Isaac ROS reports no RMSE; keep it at 0 so the acceptance gate
            # falls back to the fitness and sanity-window checks.
            return pose, float(fitness), 0.0, LICENSE_NOTICE

        if self.mode == "python":
            estimator = self._load_python_estimator()
            intr = frame.intrinsics
            depth = np.asarray(frame.depth, dtype=np.float32)
            transform = estimator.register(
                K=intr.matrix,
                rgb=np.asarray(frame.color),
                depth=depth,
                ob_mask=None,
                iteration=self.refine_iterations,
            )
            pose = Pose.from_matrix(np.asarray(transform), intr.frame_id)
            return pose, 1.0, 0.0, LICENSE_NOTICE

        raise ValueError(
            f"unknown foundationpose mode '{self.mode}' (expected 'ros_action' or 'python')"
        )
