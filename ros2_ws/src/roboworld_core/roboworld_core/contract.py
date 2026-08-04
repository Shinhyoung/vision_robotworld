"""Contract verification for the robot-department interface.

claude.md section 4 requires the interface to be covered by a *contract test*,
and section 2 makes the interface the thing that has to be right before anything
else can proceed in parallel. Two failure modes matter:

1. **Drift** -- ``PartResult.msg`` gains a field that the core dataclass never
   learns about (or vice versa), so the ROS layer silently publishes a default.
   :func:`check_schema_alignment` parses the ``.msg`` file and compares.
2. **Violation** -- a published result is structurally fine but semantically
   illegal: a non-unit quaternion, a position in millimeters, ``pose_valid``
   true on an NG part. :func:`check_result` enforces the documented rules.

Both run in CI without a ROS installation, because both work off the ``.msg``
text and the core dataclasses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import paths
from .types import PartResult, PartStatus

#: Fields of ``PartResult.msg`` that are supplied by the ROS ``std_msgs/Header``
#: rather than by a dataclass attribute of the same name.
_HEADER_FIELDS = {"header"}

#: Mapping from ``.msg`` field name to the dataclass attribute carrying it.
_FIELD_ALIASES = {
    "pose": "pose",
    "header": "stamp",  # header.stamp + header.frame_id
}

_FIELD_PATTERN = re.compile(
    r"^\s*(?P<type>[A-Za-z_][\w/\[\]<=]*)\s+(?P<name>[a-z_][a-z0-9_]*)\s*(?P<default>.*)$"
)
_CONSTANT_PATTERN = re.compile(
    r"^\s*(?P<type>[A-Za-z_]\w*)\s+(?P<name>[A-Z][A-Z0-9_]*)\s*=\s*(?P<value>\S+)"
)


@dataclass
class MessageSchema:
    """Parsed ``.msg`` definition."""

    name: str
    fields: list[tuple[str, str]] = field(default_factory=list)  # (type, name)
    constants: dict[str, str] = field(default_factory=dict)

    @property
    def field_names(self) -> list[str]:
        return [name for _, name in self.fields]


class ContractViolation(AssertionError):
    """A published result breaks the interface contract."""


def interfaces_dir() -> Path:
    """Locate the ``roboworld_interfaces`` package in the source tree."""
    candidate = paths.repo_root() / "ros2_ws" / "src" / "roboworld_interfaces"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"roboworld_interfaces not found at {candidate}")


def parse_msg(path: str | Path) -> MessageSchema:
    """Parse a ROS ``.msg`` file into a :class:`MessageSchema`."""
    file_path = Path(path)
    schema = MessageSchema(name=file_path.stem)
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        constant = _CONSTANT_PATTERN.match(line)
        if constant:
            schema.constants[constant.group("name")] = constant.group("value")
            continue
        match = _FIELD_PATTERN.match(line)
        if match:
            schema.fields.append((match.group("type"), match.group("name")))
    return schema


def check_schema_alignment() -> list[str]:
    """Compare ``PartResult.msg`` with :class:`roboworld_core.types.PartResult`.

    Returns a list of problems; empty means the two are aligned.
    """
    schema = parse_msg(interfaces_dir() / "msg" / "PartResult.msg")
    problems: list[str] = []

    attributes = set(PartResult.__dataclass_fields__)
    for _, name in schema.fields:
        if name in _HEADER_FIELDS:
            if "stamp" not in attributes or "frame_id" not in attributes:
                problems.append(
                    "PartResult.msg has a header but the dataclass lacks stamp/frame_id"
                )
            continue
        attribute = _FIELD_ALIASES.get(name, name)
        if attribute not in attributes:
            problems.append(
                f"PartResult.msg field '{name}' has no counterpart on the dataclass "
                f"(expected attribute '{attribute}')"
            )

    message_names = set(schema.field_names) | {"stamp", "frame_id"}
    for attribute in attributes:
        if attribute not in message_names and attribute not in _FIELD_ALIASES.values():
            problems.append(
                f"dataclass attribute '{attribute}' is not published by PartResult.msg"
            )

    # The status enum and the message constants must agree, or a subscriber
    # switching on STATUS_NG will act on a different case than we meant.
    for status in PartStatus:
        key = f"STATUS_{status.name}"
        if key not in schema.constants:
            problems.append(f"PartResult.msg is missing constant {key}")
        elif int(schema.constants[key]) != int(status):
            problems.append(
                f"PartResult.msg {key}={schema.constants[key]} but "
                f"PartStatus.{status.name}={int(status)}"
            )
    return problems


def check_result(result: PartResult, strict: bool = True) -> list[str]:
    """Validate one :class:`PartResult` against the documented rules.

    Rules (docs/robot_interface_ICD.md sections 3 and 6):

    * ``pose.frame_id`` is non-empty whenever ``pose_valid`` is true,
    * orientation is a unit quaternion (1e-6),
    * position is finite and plausibly in meters, not millimeters,
    * ``anomaly_score`` and ``anomaly_threshold`` lie in [0, 1],
    * ``status`` is consistent with ``is_good`` / ``pose_valid``,
    * ``pipeline_version`` is populated.
    """
    problems: list[str] = []

    if not result.pipeline_version:
        problems.append("pipeline_version is empty")
    if not (0.0 <= result.anomaly_score <= 1.0):
        problems.append(f"anomaly_score {result.anomaly_score} outside [0, 1]")
    if not (0.0 <= result.anomaly_threshold <= 1.0):
        problems.append(f"anomaly_threshold {result.anomaly_threshold} outside [0, 1]")
    if not result.part_id:
        problems.append("part_id is empty")

    norm = float(np.linalg.norm(result.pose.orientation))
    if abs(norm - 1.0) > 1e-6:
        problems.append(f"orientation is not a unit quaternion (norm {norm:.9f})")
    if not np.all(np.isfinite(result.pose.position)):
        problems.append(f"position contains non-finite values: {result.pose.position}")

    # --- status consistency ---------------------------------------------
    status = PartStatus(result.status)
    if status == PartStatus.OK:
        if not result.is_good:
            problems.append("status OK but is_good is false")
        if not result.pose_valid:
            problems.append("status OK but pose_valid is false")
    elif status == PartStatus.NG:
        if result.is_good:
            problems.append("status NG but is_good is true")
        if result.pose_valid:
            problems.append("status NG must not carry a valid pose (pose is skipped)")
    elif status in (PartStatus.NO_POSE, PartStatus.ERROR):
        if result.pose_valid:
            problems.append(f"status {status.name} must not carry a valid pose")

    if result.pose_valid:
        if not result.pose.frame_id:
            problems.append("pose_valid is true but pose.frame_id is empty")
        # A part 570 mm from the camera reads as 0.57 in meters and 570 in
        # millimeters -- the classic unit slip this pipeline must never ship.
        magnitude = float(np.linalg.norm(result.pose.position))
        if magnitude > 10.0:
            problems.append(
                f"|position| = {magnitude:.3f}; values this large suggest millimeters, "
                "the contract requires meters"
            )
    if result.tact_time_ms < 0.0:
        problems.append(f"tact_time_ms is negative ({result.tact_time_ms})")

    if strict and problems:
        raise ContractViolation(
            f"PartResult seq={result.sequence} violates the ICD: " + "; ".join(problems)
        )
    return problems
