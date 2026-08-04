"""Make the workspace packages importable when running tools from a clone.

Under ROS the packages are on ``PYTHONPATH`` after ``source install/setup.bash``.
These tools are meant to run *before* anything is built -- that is the point of
the mock-first workflow -- so they add the source directories themselves.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "ros2_ws" / "src"

#: Packages importable without a ROS installation.
ROS_FREE_PACKAGES = ("roboworld_core",)


def bootstrap(extra_packages: tuple[str, ...] = ()) -> Path:
    """Prepend the requested package source directories to ``sys.path``."""
    for package in ROS_FREE_PACKAGES + extra_packages:
        path = str(SRC_ROOT / package)
        if path not in sys.path:
            sys.path.insert(0, path)
    return REPO_ROOT
