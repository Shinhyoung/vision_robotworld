"""Station trigger simulator.

In production the trigger is a photo sensor wired to a PLC that publishes
``std_msgs/Bool`` on ``pipeline.trigger_topic``. This node emits the same rising
edges on a timer so the pipeline can be demonstrated without the station.

It publishes a proper edge (True then False) rather than a level, because the
pipeline debounces on the rising edge -- holding True forever would produce one
cycle and then silence, which is a confusing way to find that out.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from roboworld_ros_utils import config_from_node, declare_override, node_kwargs


class TriggerNode(Node):
    """Emits periodic rising edges on the trigger topic."""

    def __init__(self) -> None:
        super().__init__("trigger_node", **node_kwargs())
        self.cfg = config_from_node(self)

        period = max(0.2, float(declare_override(self, "period_s", 3.0)))
        self.pulse_s = max(0.01, float(declare_override(self, "pulse_s", 0.1)))

        topic = str(self.cfg.get("pipeline.trigger_topic", "/roboworld/trigger"))
        self.publisher = self.create_publisher(Bool, topic, 10)
        self.timer = self.create_timer(period, self._pulse)
        self._reset_timer = None
        self.get_logger().info(f"trigger_node pulsing '{topic}' every {period:.1f}s")

    def _pulse(self) -> None:
        self.publisher.publish(Bool(data=True))
        if self._reset_timer is not None:
            self._reset_timer.cancel()
        self._reset_timer = self.create_timer(self.pulse_s, self._release)

    def _release(self) -> None:
        self.publisher.publish(Bool(data=False))
        if self._reset_timer is not None:
            self._reset_timer.cancel()
            self._reset_timer = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TriggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
