"""Pose backend for parts registered without CAD.

A part registered from camera captures (tools/register_part.py) has no geometry,
so no 6D pose can be produced for it. That is a *known, expected* state, not a
failure: inspection still works and the line can still sort OK from NG.

Rather than letting the node fail at startup -- which would stop inspection too
-- this backend reports every estimate as invalid with an explanatory message.
The pipeline then emits ``STATUS_NO_POSE``, which the ICD already defines as
"good part, no usable pose -- do not pick" (section 3.1). The robot department
needs no new case to handle.
"""

from __future__ import annotations

from ..geometry import Pose
from ..types import Frame
from .base import PoseBackend, PoseSettings


class NoCadPoseBackend(PoseBackend):
    """Always-invalid pose source for parts that have no mesh."""

    name = "no_cad"

    def __init__(self, settings: PoseSettings, part_id: str = "",
                 symmetry_group: list | None = None) -> None:
        super().__init__(settings, symmetry_group)
        self.part_id = part_id

    def estimate(self, frame: Frame) -> tuple[Pose, float, float, str]:
        return (
            Pose.identity(frame.intrinsics.frame_id),
            0.0,
            float("inf"),
            f"part '{self.part_id or frame.part_id}' has no CAD mesh; "
            "6D pose unavailable (inspection unaffected). Set "
            f"parts.{self.part_id or frame.part_id}.mesh to enable it.",
        )
