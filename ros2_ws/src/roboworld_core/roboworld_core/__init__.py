"""RoboWorld vision core: ROS-free inspection, pose and pipeline logic.

Import layout:

* :mod:`roboworld_core.config` / :mod:`roboworld_core.paths` -- configuration
* :mod:`roboworld_core.geometry` / :mod:`roboworld_core.types` -- contracts
* :mod:`roboworld_core.mock_data` / :mod:`roboworld_core.render` -- mock frames
* :mod:`roboworld_core.inspection` / :mod:`roboworld_core.pose` -- backends
* :mod:`roboworld_core.pipeline` -- the trigger -> inspect -> pose -> publish cycle

Nothing here imports ``rclpy``; the ROS nodes in ``roboworld_inspection``,
``roboworld_pose`` and ``roboworld_pipeline`` are thin adapters over this
package (claude.md section 2).
"""

from .types import PIPELINE_VERSION, PartStatus

__version__ = PIPELINE_VERSION

__all__ = ["PIPELINE_VERSION", "PartStatus", "__version__"]
