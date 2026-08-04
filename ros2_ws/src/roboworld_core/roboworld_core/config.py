"""YAML configuration loading.

Per claude.md section 4 nothing operational may be hardcoded: thresholds,
camera parameters and paths all come from YAML. The same files are readable as
plain YAML *and* as ROS 2 parameter files, so a node and a unit test always see
identical values. ROS parameter files wrap their payload like::

    /**:
      ros__parameters:
        threshold: 0.5

:func:`load_config` transparently unwraps that envelope.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from . import paths

#: Config files loaded (in order) when no explicit list is given. Later files
#: win on key collisions, but the files are namespaced so collisions are rare.
#:
#: ``parts_local.yaml`` is optional and holds parts registered from camera
#: captures (tools/register_part.py). Keeping them out of ``parts.yaml`` means a
#: generated entry never has to be merged into a hand-curated, commented file.
DEFAULT_CONFIG_FILES = (
    "parts.yaml",
    "parts_local.yaml",
    "camera.yaml",
    "inspection.yaml",
    "pose.yaml",
    "pipeline.yaml",
)

_ROS_WILDCARD_KEYS = ("/**", "**")
_ROS_PARAMS_KEY = "ros__parameters"

#: Sentinel distinguishing "no default supplied" from an explicit ``None``.
_MISSING = object()


class ConfigError(RuntimeError):
    """Raised when configuration is missing or malformed."""


def _unwrap_ros_params(data: Mapping[str, Any]) -> dict[str, Any]:
    """Strip the ``/**: ros__parameters:`` envelope used by ROS 2 param files."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key in _ROS_WILDCARD_KEYS or key.startswith("/"):
            if isinstance(value, Mapping) and _ROS_PARAMS_KEY in value:
                result.update(_unwrap_ros_params(value[_ROS_PARAMS_KEY]))
                continue
            if isinstance(value, Mapping):
                result.update(_unwrap_ros_params(value))
                continue
        if key == _ROS_PARAMS_KEY and isinstance(value, Mapping):
            result.update(dict(value))
            continue
        result[key] = value
    return result


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` without mutating either."""
    merged = dict(copy.deepcopy(dict(base)))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class Config:
    """Read-only nested configuration with dotted-path access."""

    def __init__(self, data: Mapping[str, Any], source: str = "<memory>") -> None:
        self._data = dict(copy.deepcopy(dict(data)))
        self.source = source

    # -- access ----------------------------------------------------------
    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        """Return the value at ``dotted`` (e.g. ``"inspection.threshold"``).

        Raises :class:`ConfigError` when the key is absent and no default was
        supplied -- a silent ``None`` for a missing threshold is far worse than
        a loud failure at startup.
        """
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is _MISSING:
                    raise ConfigError(f"missing config key '{dotted}' (source: {self.source})")
                return default
            node = node[part]
        return copy.deepcopy(node)

    def section(self, dotted: str) -> Config:
        value = self.get(dotted)
        if not isinstance(value, Mapping):
            raise ConfigError(f"config key '{dotted}' is not a section (source: {self.source})")
        return Config(value, source=f"{self.source}:{dotted}")

    def path(self, dotted: str, default: Any = None) -> Path:
        """Return a config value resolved as a filesystem path."""
        value = self.get(dotted) if default is None else self.get(dotted, default)
        return paths.resolve_path(value)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def merged_with(self, override: Mapping[str, Any]) -> Config:
        return Config(deep_merge(self._data, override), source=f"{self.source}+override")

    # -- dunder ----------------------------------------------------------
    def __contains__(self, dotted: str) -> bool:
        return self.get(dotted, None) is not None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config(source={self.source!r}, keys={sorted(self._data)})"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file, unwrapping the ROS parameter envelope."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"config file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"config file must contain a mapping: {file_path}")
    return _unwrap_ros_params(data)


def load_config(
    files: Iterable[str] | None = None,
    config_dir: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Load and merge the configuration set.

    Args:
        files: file names relative to ``config_dir``; defaults to
            :data:`DEFAULT_CONFIG_FILES`.
        config_dir: directory to read from; defaults to :func:`paths.config_dir`.
        overrides: mapping deep-merged last (used by ROS nodes to inject
            parameters declared on the command line).
    """
    directory = Path(config_dir) if config_dir is not None else paths.config_dir()
    names = tuple(files) if files is not None else DEFAULT_CONFIG_FILES

    merged: dict[str, Any] = {}
    loaded: list[str] = []
    for name in names:
        file_path = directory / name
        if not file_path.is_file():
            if files is not None:  # explicitly requested -> must exist
                raise ConfigError(f"config file not found: {file_path}")
            continue
        merged = deep_merge(merged, load_yaml(file_path))
        loaded.append(name)

    if not loaded:
        raise ConfigError(f"no configuration files found in {directory}")
    if overrides:
        merged = deep_merge(merged, overrides)
    return Config(merged, source=f"{directory}[{','.join(loaded)}]")
