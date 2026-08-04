"""Bridging ROS node parameters and the YAML :class:`Config`.

A node is launched with the same YAML files the core library reads, then any
parameter given on the command line or in a launch file overrides the file
value. Dotted names (``camera.source``, ``inspection.threshold``) map onto the
nested config structure.

Nodes using this **must** pass
``automatically_declare_parameters_from_overrides=True`` to ``Node.__init__``
(:func:`node_kwargs` supplies it). Without it rclpy keeps undeclared overrides
out of ``get_parameters_by_prefix``, so a launch argument like
``{"camera.source": "realsense"}`` would be accepted at the command line and
then silently ignored -- the pipeline would keep running on mock frames while
the log claimed otherwise.
"""

from __future__ import annotations

import contextlib
from typing import Any

from rclpy.exceptions import ParameterAlreadyDeclaredException

from roboworld_core.config import Config, load_config

#: Parameters that configure the loader itself and must not be treated as
#: config overrides.
_RESERVED = {"config_dir", "use_sim_time"}


def node_kwargs() -> dict[str, Any]:
    """Keyword arguments every RoboWorld node passes to ``Node.__init__``."""
    return {"automatically_declare_parameters_from_overrides": True}


def declare_override(node, name: str, default: Any) -> Any:
    """Declare ``name`` if needed and return its effective value.

    Tolerates the parameter already existing, which it does whenever the launch
    file supplied it and auto-declaration picked it up first.
    """
    with contextlib.suppress(ParameterAlreadyDeclaredException):
        node.declare_parameter(name, default)
    return node.get_parameter(name).value


def config_from_node(node, config_dir: str | None = None) -> Config:
    """Load the configuration for ``node``, applying its parameter overrides.

    ``config_dir`` may be passed as the node parameter ``config_dir``; otherwise
    :func:`roboworld_core.paths.config_dir` decides.
    """
    declared = declare_override(node, "config_dir", config_dir or "")
    cfg = load_config(config_dir=declared or None)

    overrides = dotted_overrides(node)
    if overrides:
        node.get_logger().info(f"config overrides from parameters: {overrides}")
        return cfg.merged_with(overrides)
    return cfg


def dotted_overrides(node) -> dict[str, Any]:
    """Turn ``a.b.c`` parameters into a nested override mapping."""
    nested: dict[str, Any] = {}
    for name, parameter in node.get_parameters_by_prefix("").items():
        if "." not in name or name in _RESERVED:
            continue
        value = parameter.value
        if value is None or value == "":
            # An empty string is how launch expresses "argument not supplied";
            # letting it through would blank out a real configured value.
            continue
        cursor = nested
        parts = name.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested
