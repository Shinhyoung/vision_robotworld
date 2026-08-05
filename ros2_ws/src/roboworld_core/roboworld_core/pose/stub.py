"""Stub pose backend.

Part of the Team Lead's stub-node deliverable (claude.md section 3.1): it lets
the pipeline, the ICD publisher and the robot department's subscriber be built
and tested before FoundationPose exists.

``use_ground_truth`` returns the mock frame's true pose (so an E2E dry-run
produces meaningful numbers); otherwise it returns a fixed nominal pose at the
configured station height.
"""

from __future__ import annotations

import numpy as np

from ..geometry import Pose
from ..types import Frame
from .base import PoseBackend, PoseSettings


class StubPoseBackend(PoseBackend):
    """Fixed / ground-truth pose source for dry-runs and interface tests."""

    name = "stub"

    def __init__(
        self,
        settings: PoseSettings,
        use_ground_truth: bool = True,
        nominal_z_m: float = 0.5725,
        fitness: float = 0.95,
        rmse_m: float = 0.001,
        symmetry_group: list | None = None,
    ) -> None:
        super().__init__(settings, symmetry_group)
        self.use_ground_truth = bool(use_ground_truth)
        self.nominal_z_m = float(nominal_z_m)
        self.fitness = float(fitness)
        self.rmse_m = float(rmse_m)

    def estimate(self, frame: Frame) -> tuple[Pose, float, float, str]:
        if self.use_ground_truth and frame.gt_pose is not None:
            pose = Pose(
                frame.gt_pose.position,
                frame.gt_pose.orientation,
                frame.intrinsics.frame_id,
            )
            return pose, self.fitness, self.rmse_m, "stub: ground truth"

        pose = Pose(
            np.array([0.0, 0.0, self.nominal_z_m]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            frame.intrinsics.frame_id,
        )
        return pose, self.fitness, self.rmse_m, "stub: fixed nominal pose"
