"""Filesystem discovery helpers.

The same YAML config files are consumed twice:

* by plain Python (unit tests, ``tools/e2e_dryrun.py``) straight out of the
  source tree, and
* by ROS 2 nodes, where the files live in the installed
  ``roboworld_bringup`` share directory.

Everything here exists so no module ever has to hardcode an absolute path.
Resolution order is always: explicit argument -> environment variable ->
ament share directory -> source tree discovery.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Marker files that identify the repository root when walking upwards.
_ROOT_MARKERS = ("claude.md", ".roboworld_root")

_BRINGUP_PACKAGE = "roboworld_bringup"


def repo_root() -> Path:
    """Return the repository root directory.

    Honours ``ROBOWORLD_ROOT`` when set, otherwise walks up from this file
    looking for a marker (see :data:`_ROOT_MARKERS`).
    """
    env = os.environ.get("ROBOWORLD_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    # Installed layout (share/<pkg>/...) has no marker; fall back to the
    # source-tree grandparent so relative lookups still produce a sane path.
    return here.parents[2]


def _ament_share(package: str) -> Path | None:
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:
        return None
    try:
        return Path(get_package_share_directory(package))
    except Exception:  # package not built/installed
        return None


def config_dir() -> Path:
    """Directory holding the canonical YAML configuration files."""
    env = os.environ.get("ROBOWORLD_CONFIG_DIR")
    if env:
        return Path(env).expanduser().resolve()

    share = _ament_share(_BRINGUP_PACKAGE)
    if share is not None and (share / "config").is_dir():
        return share / "config"

    return repo_root() / "ros2_ws" / "src" / _BRINGUP_PACKAGE / "config"


def assets_dir() -> Path:
    """Directory holding the part CAD meshes (``01_input`` in the source tree)."""
    env = os.environ.get("ROBOWORLD_ASSETS_DIR")
    if env:
        return Path(env).expanduser().resolve()

    share = _ament_share(_BRINGUP_PACKAGE)
    if share is not None and (share / "meshes").is_dir():
        return share / "meshes"

    return repo_root() / "01_input"


def data_dir() -> Path:
    """Writable directory for generated mock datasets and trained models."""
    env = os.environ.get("ROBOWORLD_DATA_DIR")
    if env:
        path = Path(env).expanduser().resolve()
    else:
        path = repo_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(value: str | os.PathLike[str]) -> Path:
    """Expand ``${VAR}``/``~`` and anchor relative paths at the repo root.

    Recognised substitutions beyond the process environment:
    ``${ROBOWORLD_ROOT}``, ``${ROBOWORLD_ASSETS_DIR}``, ``${ROBOWORLD_DATA_DIR}``.
    """
    text = str(value)
    defaults = {
        "ROBOWORLD_ROOT": str(repo_root()),
        "ROBOWORLD_ASSETS_DIR": str(assets_dir()),
        "ROBOWORLD_DATA_DIR": str(data_dir()),
    }
    for key, fallback in defaults.items():
        token = "${" + key + "}"
        if token in text:
            text = text.replace(token, os.environ.get(key, fallback))
    text = os.path.expandvars(os.path.expanduser(text))

    path = Path(text)
    return path if path.is_absolute() else (repo_root() / path).resolve()
