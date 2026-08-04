"""QoS profile construction from YAML.

The output topic's QoS is part of the robot-department contract (ICD section 5):
a subscriber whose profile is incompatible silently receives nothing, so the
profile is configuration, not a hardcoded constant.
"""

from __future__ import annotations

from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

_RELIABILITY = {
    "reliable": ReliabilityPolicy.RELIABLE,
    "best_effort": ReliabilityPolicy.BEST_EFFORT,
}
_DURABILITY = {
    "volatile": DurabilityPolicy.VOLATILE,
    "transient_local": DurabilityPolicy.TRANSIENT_LOCAL,
}
_HISTORY = {
    "keep_last": HistoryPolicy.KEEP_LAST,
    "keep_all": HistoryPolicy.KEEP_ALL,
}


def qos_from_config(section, default_depth: int = 10) -> QoSProfile:
    """Build a :class:`QoSProfile` from a config mapping.

    Expected keys: ``reliability``, ``durability``, ``history``, ``depth``.
    Unknown values raise instead of silently falling back -- a QoS typo that
    degrades to the default is exactly the kind of bug that shows up as
    "the robot sometimes misses a part".
    """
    def pick(table: dict, key: str, default: str):
        value = str(section.get(key, default)).lower()
        if value not in table:
            raise ValueError(
                f"invalid QoS {key} '{value}' (expected one of {sorted(table)})"
            )
        return table[value]

    return QoSProfile(
        reliability=pick(_RELIABILITY, "reliability", "reliable"),
        durability=pick(_DURABILITY, "durability", "volatile"),
        history=pick(_HISTORY, "history", "keep_last"),
        depth=int(section.get("depth", default_depth)),
    )


def sensor_qos(depth: int = 5) -> QoSProfile:
    """Best-effort profile matching realsense-ros image publishers."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )
